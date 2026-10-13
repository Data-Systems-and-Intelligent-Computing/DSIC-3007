"""Small, explicit scientific bundles and an append-only complete-event journal."""

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def relative(path):
    p = Path(path)
    if p.is_absolute() or not p.parts or any(v in ("..", ".") for v in p.parts):
        raise ValueError(f"unsafe artifact path: {path}")
    return p


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def require(settings, *paths):
    for path in paths:
        value = settings
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            raise ValueError(f"Scientific decision pending in configs/research.json: {path}")


def unresolved(settings, prefix=""):
    result = []
    for key, value in settings.items():
        name = f"{prefix}.{key}" if prefix else key
        if value is None:
            result.append(name)
        elif isinstance(value, dict):
            result.extend(unresolved(value, name))
    return result


def priority(seed, purpose, identity):
    if type(seed) is not int:
        raise ValueError("master seed must be an explicitly selected integer")
    return digest(["DSIC-3007", seed, purpose, identity])


def seal(directory, kind, metadata=None):
    """Seal explicitly selected research directories only; caller owns their contents."""
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        raise ValueError("bundle already sealed")
    files = []
    for p in sorted(directory.rglob("*")):
        if p.is_symlink():
            raise ValueError("symlinks are not portable research inputs")
        if p.is_file():
            if p.name.endswith(".tmp") or p.name.endswith(".lock"):
                raise ValueError("unfinished file in bundle")
            files.append({"path": p.relative_to(directory).as_posix(),
                          "size_bytes": p.stat().st_size, "sha256": file_digest(p)})
    manifest = {"schema_version": 1, "kind": kind, "status": "complete",
                "metadata": metadata or {}, "files": files}
    atomic(directory / "manifest.json", canonical(manifest) + b"\n")
    return manifest


def verify_bundle(directory, kind=None):
    directory = Path(directory).resolve(strict=True)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") != "complete" or (kind and manifest.get("kind") != kind):
        raise ValueError("wrong kind or incomplete bundle")
    seen = set()
    for row in manifest["files"]:
        p = directory / relative(row["path"])
        if row["path"] in seen or p.is_symlink() or directory not in p.resolve().parents:
            raise ValueError("duplicate or escaping artifact")
        seen.add(row["path"])
        if p.stat().st_size != row["size_bytes"] or file_digest(p) != row["sha256"]:
            raise ValueError(f"artifact mismatch: {row['path']}")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != seen | {"manifest.json"}:
        raise ValueError("unlisted files in sealed bundle")
    return manifest


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        for row in rows:
            f.write(canonical(row) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def validate_measurement(row):
    ids = row["routing"]["expert_ids"]
    weights = row["routing"]["weights"]
    if len(ids) != 8 or len(set(ids)) != 8 or any(type(e) is not int or not 0 <= e < 64 for e in ids):
        raise ValueError("expected eight distinct native expert IDs")
    if len(weights) != 8 or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("invalid routing coefficients")
    import struct
    if struct.pack("<8f", *weights).hex() != row["routing"]["weights_fp32_le_hex"]:
        raise ValueError("routing coefficient bytes do not match values")
    if not math.isfinite(row["baseline_nll"]):
        raise ValueError("nonfinite baseline NLL")
    if row["kind"] == "calibration_window":
        window = row["window_routing"]
        if window["shape"] != [row["context_length"], 8] or len(window["expert_ids"]) != row["context_length"]:
            raise ValueError("incomplete window routing")
        for active in window["expert_ids"]:
            if len(active) != 8 or len(set(active)) != 8 or any(type(e) is not int or not 0 <= e < 64 for e in active):
                raise ValueError("invalid whole-window expert IDs")
        raw = bytes.fromhex(window["weights_fp32_le_hex"])
        if len(raw) != row["context_length"] * 8 * 4:
            raise ValueError("whole-window coefficient bytes have wrong size")
        if any(not math.isfinite(w) or w < 0 for (w,) in struct.iter_unpack("<f", raw)):
            raise ValueError("invalid whole-window coefficients")
    if row["kind"] == "effects":
        effects = row["effects"]
        if len(effects) != 8 or {e["expert_id"] for e in effects} != set(ids):
            raise ValueError("all eight individual expert effects are required")
        for effect in effects:
            if (not math.isfinite(effect["ablated_nll"]) or not math.isfinite(effect["delta_nll"])
                    or effect["delta_nll"] != effect["ablated_nll"] - row["baseline_nll"]):
                raise ValueError("effect does not reconstruct from NLLs")


class _JournalRecordOffsets(Mapping):
    """Keep an ID/offset index; read at most one complete raw event per lookup."""

    def __init__(self, path):
        self.path, self.offsets = path, {}

    def __getitem__(self, key):
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[key])
            return json.loads(stream.readline())

    def __iter__(self):
        return iter(self.offsets)

    def __len__(self):
        return len(self.offsets)

    def __contains__(self, key):
        return key in self.offsets

    def remember(self, row, offset):
        self.offsets[row["event_id"]] = offset


class Journal:
    """Only complete events count. A failed event stays in the frozen plan for repair."""

    def __init__(self, directory, identity, *, compact=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = self.directory / "writer.lock"
        self.lock_fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        p = self.directory / "events.jsonl"
        self.records = _JournalRecordOffsets(p) if compact else {}
        try:
            config = self.directory / "identity.json"
            if config.exists():
                if read_json(config) != identity:
                    raise ValueError("resume input/model/runtime/code identity differs")
            else:
                atomic(config, canonical(identity) + b"\n")
            if p.exists():
                with p.open("rb") as f:
                    while True:
                        offset = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        if not line.endswith(b"\n"):
                            raise ValueError("torn journal tail; explicit repair required")
                        row = json.loads(line)
                        validate_measurement(row)
                        if row["event_id"] in self.records:
                            raise ValueError("duplicate completed event")
                        if compact:
                            self.records.remember(row, offset)
                        else:
                            self.records[row["event_id"]] = row
            if (self.directory / "manifest.json").exists():
                raise ValueError("completed run is immutable; use it directly")
            self.stream = p.open("ab")
        except BaseException:
            self.close()
            raise

    def append(self, row):
        validate_measurement(row)
        if row["event_id"] in self.records:
            raise ValueError("event already complete")
        offset = self.stream.tell()
        self.stream.write(canonical(row) + b"\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        if isinstance(self.records, _JournalRecordOffsets):
            self.records.remember(row, offset)
        else:
            self.records[row["event_id"]] = row

    def close(self):
        if hasattr(self, "stream"):
            self.stream.close()
        if getattr(self, "lock_fd", None) is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
            self.lock.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

"""Small shared I/O helpers. Scientific policy lives in config/research.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import zipfile
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_verified_prepared = set()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seed_for(seed, *parts):
    return int(digest([seed, *parts])[:16], 16)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--research", default=str(ROOT / "config/research.json"))
    p.add_argument("--execution", default=str(ROOT / "config/execution.json"))
    return p


def settings(args):
    research, execution = read_json(args.research), read_json(args.execution)
    for key in ("prepared_path", "results_path", "model_path", "tokenizer_path"):
        value = os.environ.get("PATH_B_" + key.upper(), execution[key])
        execution[key] = str(Path(value) if Path(value).is_absolute() else ROOT / value)
    return research, execution


def preparation_settings(research):
    """Only parameters that change token eligibility or source partitions."""
    return {"model": research["model"], "context": research["context"], "seed": research["seed"],
            "domains": research["domains"], "outcome_domains": research["outcome_domains"],
            "sources": research["data_sources"],
            "technical_groups": research["sampling"]["technical_groups"],
            "probe_documents": research["sampling"]["probe_documents"],
            "calibration_max": max(research["calibration"]["ladder"]),
            "pilot_max": max(research["sampling"]["pilot_ladder"])}


def load_prepared(execution, research):
    base = Path(execution["prepared_path"])
    info = read_json(base / "prepared.json")
    if preparation_settings(info["research"]) != preparation_settings(research):
        raise ValueError("Prepared data uses different research settings; prepare a new output directory")
    documents = [json.loads(line) for line in (base / "documents.jsonl").open()]
    identity = dict(info)
    identity.pop("id")
    if digest(identity) != info["id"]:
        raise ValueError("Prepared metadata changed")
    fingerprint = (str(base.resolve()), info["id"], *((p.stat().st_size, p.stat().st_mtime_ns) for p in (base / "documents.jsonl", base / "tokens.bin")))
    if fingerprint not in _verified_prepared:
        with (base / "tokens.bin").open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest(documents) != info["documents_id"] or checksum != info["tokens_sha256"]:
            raise ValueError("Prepared document/token bytes changed; restore or prepare a successor")
        _verified_prepared.add(fingerprint)
    tokens = np.memmap(base / "tokens.bin", dtype="<i4", mode="r")
    return info, documents, tokens


def sample_positions(document, count, seed, context, technical=False):
    eligible = document["tokens"] - context
    if eligible <= 0:
        return []
    if technical:
        offsets = sorted({i * (eligible - 1) // max(count - 1, 1) for i in range(count)})
    else:
        # One seeded order per document makes each larger position sample nested.
        offsets = np.random.default_rng(seed_for(seed, document["id"], "positions")).permutation(eligible)[:count]
    return [context - 1 + int(i) for i in offsets]


def window(document, predictor, tokens, context):
    start = document["offset"] + predictor - context + 1
    ids = np.asarray(tokens[start:start + context], dtype="<i4").copy()
    target = int(tokens[document["offset"] + predictor + 1])
    if len(ids) != context or predictor < context - 1 or predictor + 1 >= document["tokens"]:
        raise ValueError("Invalid original-document predictor/target alignment")
    return ids, target


def event_key(document, predictor):
    return f"{document['domain']}:{document['id']}:{predictor}"


def records(path):
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        for key, payload in db.execute("SELECT id,payload FROM events ORDER BY rowid"):
            yield key, json.loads(payload)


def metadata(path):
    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True) as db:
        return {k: json.loads(v) for k, v in db.execute("SELECT key,value FROM metadata")}


class Journal:
    """One SQLite transaction per complete input; no partial expert vectors."""

    def __init__(self, path, research, prepared, stage, extra=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,payload TEXT NOT NULL,input BLOB NOT NULL,routes BLOB,weights BLOB)")
        self.db.execute("CREATE TABLE IF NOT EXISTS failures(time REAL,id TEXT,error TEXT)")
        source = {p.name: p.read_text() for p in (ROOT / "scripts").glob("*.py")}
        binding = {"research": research, "prepared_id": prepared["id"], "stage": stage,
                   "measurement_code": digest((ROOT / "scripts/measurement.py").read_text()), **(extra or {})}
        previous = self.db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
        if previous and json.loads(previous[0]) != binding:
            self.db.close()
            raise ValueError("Resume settings, data, calibration, or measurement code differ; use a new results directory")
        if not previous:
            for key, value in {"binding": binding, "created": time.time(), "source": source}.items():
                self.db.execute("INSERT INTO metadata VALUES (?,?)", (key, canonical(value)))
        self.db.commit()

    def get(self, key, ids=None, target=None):
        row = self.db.execute("SELECT payload,input FROM events WHERE id=?", (key,)).fetchone()
        if row and ids is not None:
            if zlib.decompress(row[1]) != np.asarray(ids, dtype="<i4").tobytes() or json.loads(row[0]).get("target") != target:
                raise ValueError("Cached observation has different input/target")
        return json.loads(row[0]) if row else None

    def append(self, key, payload, ids, routes=None, weights=None):
        if "ablated_nll" in payload:
            expected = json.loads(self.db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()[0])["research"]["model"]["active_experts"]
            if (len(payload["ablated_nll"]) != expected or len(payload["active"]) != expected
                    or len(payload.get("effects", [])) != expected or not payload.get("cleanup_exact")):
                raise ValueError("Refusing an incomplete expert measurement")
        blobs = [zlib.compress(np.asarray(ids, dtype="<i4").tobytes()),
                 None if routes is None else zlib.compress(np.asarray(routes, dtype="u1").tobytes()),
                 None if weights is None else zlib.compress(np.asarray(weights, dtype="<f4").tobytes())]
        with self.db:
            self.db.execute("INSERT INTO events VALUES (?,?,?,?,?)", (key, canonical(payload), *blobs))

    def fail(self, key, error):
        with self.db:
            self.db.execute("INSERT INTO failures VALUES (?,?,?)", (time.time(), key, repr(error)))

    def close(self):
        self.db.close()


def snapshot(directory, research, execution):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "settings.json", {"research": research, "execution": execution})
    # Actual source bytes, including uncommitted edits; Git alone is insufficient.
    temporary = directory / "source.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for folder in ("scripts", "config", "terraform"):
            for p in (ROOT / folder).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts and ".terraform" not in p.parts and p.suffix not in (".tfstate", ".tfplan"):
                    if p.name == "execution.json" or "tfvars" in p.name or ".tfstate" in p.name or ".tfplan" in p.name:
                        continue  # resolved execution settings above; never archive credentials/state
                    out.write(p, p.relative_to(ROOT))
        for name in ("README.md", "pyproject.toml", "uv.lock", "data/manifests/model-fp32.json"):
            path = ROOT / name
            if path.exists():
                out.write(path, name)
    os.replace(temporary, directory / "source.zip")


def progress(stage, completed, total, start, memory=None):
    elapsed = time.monotonic() - start
    eta = elapsed / max(completed, 1) * (total - completed)
    print(canonical({"stage": stage, "completed": completed, "total": total,
                     "elapsed_s": round(elapsed, 2), "eta_s": round(eta, 2), **(memory or {})}), flush=True)

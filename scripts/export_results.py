"""Online SQLite backup and verified transfer inventory; no remote service dependencies."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sqlite3

from scripts.common import ROOT, parser, read_json, settings, write_json


def export(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    included = []
    for path in source.rglob("*"):
        if not path.is_file() or path.name.endswith(("-journal", "-wal", "-shm", ".tmp")) or path == source / "transfer.json":
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".sqlite":
            with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as original, sqlite3.connect(target) as copy:
                original.backup(copy)
        else:
            shutil.copy2(path, target)
        with target.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        included.append({"path": str(relative), "size": target.stat().st_size, "sha256": checksum})
    write_json(destination / "transfer.json", included)


def check(directory):
    directory = Path(directory)
    files = read_json(directory / "transfer.json")
    for item in files:
        path = directory / item["path"]
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != item["sha256"] or path.stat().st_size != item["size"]:
            raise ValueError(f"Transferred artifact differs: {item['path']}")
        if path.suffix == ".sqlite":
            with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
                if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError(f"SQLite integrity check failed: {path}")
    print(f"Verified {len(files)} transferred artifacts")


def main():
    p = parser(__doc__)
    p.add_argument("--check", type=Path)
    p.add_argument("--destination", type=Path, default=ROOT / "export")
    args = p.parse_args()
    if args.check:
        check(args.check)
    else:
        _, execution = settings(args)
        source = Path(execution["results_path"])
        if not source.exists():
            raise ValueError("No results to retrieve; preserve the pod and inspect its paths")
        export(source, args.destination)


if __name__ == "__main__":
    main()

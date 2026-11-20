"""Acquire pinned original FP32 model assets once, off the GPU execution path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def acquire(research, execution):
    catalog = json.loads((ROOT / "data/manifests/model-fp32.json").read_text())
    if catalog["revision"] != research["model"]["revision"] or catalog["repo_id"] != research["model"]["repo"]:
        raise ValueError("Asset catalog differs from the selected original FP32 revision")
    destination = Path(execution["model_path"])
    marker = destination / "path-b-acquisition.json"
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if recorded["revision"] == catalog["revision"] and all((destination / f["path"]).stat().st_size == f["size_bytes"] for f in catalog["files"]):
            print("Pinned model already acquired; no repeated full-weight hashing")
            return
        raise ValueError("Previously acquired model has changed or is incomplete")
    files = []
    for asset in catalog["files"]:
        path = Path(hf_hub_download(repo_id=catalog["repo_id"], revision=catalog["revision"], filename=asset["path"], local_dir=destination))
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
            checksum = digest.hexdigest()
        if path.stat().st_size != asset["size_bytes"] or checksum != asset["sha256"]:
            raise ValueError(f"Pinned asset verification failed: {asset['path']}")
        if path.suffix == ".safetensors":
            # Safetensors: 8-byte little-endian header length, then JSON tensor
            # metadata. No tensor materialization or PyTorch dependency is needed.
            with path.open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                if not 0 < size <= min(100 * 1024**2, path.stat().st_size - 8):
                    raise ValueError("Invalid safetensors header length")
                header = json.loads(stream.read(size))
            tensors = [value for key, value in header.items() if key != "__metadata__"]
            if not tensors or any(tensor["dtype"] != "F32" for tensor in tensors):
                raise ValueError("Model source is not original FP32 storage")
        files.append({"path": asset["path"], "size_bytes": asset["size_bytes"], "sha256": checksum})
        print(f"Acquired {asset['path']}", flush=True)
    temporary = marker.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump({"repo": catalog["repo_id"], "revision": catalog["revision"], "files": files}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, default=ROOT / "config/research.json")
    parser.add_argument("--execution", type=Path, default=ROOT / "config/execution.json")
    args = parser.parse_args()
    research, execution = json.loads(args.research.read_text()), json.loads(args.execution.read_text())
    destination = Path(os.environ.get("PATH_B_MODEL_PATH", execution["model_path"]))
    execution["model_path"] = str(destination if destination.is_absolute() else ROOT / destination)
    acquire(research, execution)


if __name__ == "__main__":
    main()

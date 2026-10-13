"""Acquire pinned original FP32 model assets once, off the GPU execution path."""
from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open

from scripts.common import ROOT, parser, read_json, settings, write_json


def acquire(research, execution):
    catalog = read_json(ROOT / "data/manifests/model-fp32.json")
    if catalog["revision"] != research["model"]["revision"] or catalog["repo_id"] != research["model"]["repo"]:
        raise ValueError("Asset catalog differs from the selected original FP32 revision")
    destination = Path(execution["model_path"])
    marker = destination / "path-b-acquisition.json"
    if marker.exists():
        recorded = read_json(marker)
        if recorded["revision"] == catalog["revision"] and all((destination / f["path"]).stat().st_size == f["size_bytes"] for f in catalog["files"]):
            print("Pinned model already acquired; no repeated full-weight hashing")
            return
        raise ValueError("Previously acquired model has changed or is incomplete")
    files = []
    for asset in catalog["files"]:
        path = Path(hf_hub_download(repo_id=catalog["repo_id"], revision=catalog["revision"], filename=asset["path"], local_dir=destination))
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != asset["size_bytes"] or checksum != asset["sha256"]:
            raise ValueError(f"Pinned asset verification failed: {asset['path']}")
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as tensors:
                if any(tensors.get_slice(key).get_dtype() != "F32" for key in tensors.keys()):
                    raise ValueError("Model source is not original FP32 storage")
        files.append({"path": asset["path"], "size_bytes": asset["size_bytes"], "sha256": checksum})
        print(f"Acquired {asset['path']}", flush=True)
    write_json(marker, {"repo": catalog["repo_id"], "revision": catalog["revision"], "files": files})


def main():
    args = parser(__doc__).parse_args()
    research, execution = settings(args)
    acquire(research, execution)


if __name__ == "__main__":
    main()

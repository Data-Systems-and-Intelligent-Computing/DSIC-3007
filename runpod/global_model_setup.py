"""User-invoked, deadline-bounded pinned download into an already mounted Global Volume."""

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from src.artifacts import atomic, canonical, digest, read_json


def prepare(catalog_path, mount, output, deadline_epoch):
    mount, output = Path(mount), Path(output)
    if not os.path.ismount(mount):
        raise ValueError("Global Volume path is not mounted; no local-storage fallback")
    catalog = read_json(catalog_path)
    if catalog["revision"] != "c6f6954f182ffdd9f6ab4f4d772714b7dcd6836b":
        raise ValueError("original pinned FP32 revision required")
    root = mount / "models/OLMoE-fp32"
    root.mkdir(parents=True, exist_ok=True)
    owner = {"catalog_sha256": digest(catalog), "revision": catalog["revision"]}
    marker = root / ".dsic-download-owner.json"
    if marker.exists():
        if read_json(marker) != owner:
            raise ValueError("model directory belongs to a different catalog; preserve it")
    else:
        if any(root.iterdir()):
            raise ValueError("unowned nonempty model directory; no automatic overwrite")
        # Global storage has no atomic rename. Only weights/provenance live here;
        # status, journals and result receipts use local POSIX disk.
        with marker.open("xb") as stream:
            stream.write(canonical(owner)); stream.flush()
    receipt = {"status": "incomplete", "catalog_sha256": digest(catalog),
               "revision": catalog["revision"], "model_path": str(root),
               "verified_files": [], "deadline_epoch": deadline_epoch}
    started, last_progress = time.time(), 0

    def check():
        if time.time() >= deadline_epoch-20:
            raise TimeoutError("model preparation deadline reached; partial cache retained")

    def persist(phase, **extra):
        receipt.update(phase=phase, elapsed_seconds=time.time()-started, **extra)
        atomic(output.with_suffix(".status.json"), canonical(receipt)+b"\n")
        print(f"[model-setup] {phase} elapsed={receipt['elapsed_seconds']:.1f}s "
              f"remaining={max(0,deadline_epoch-time.time()):.1f}s {extra}", flush=True)

    try:
        for asset in catalog["files"]:
            check()
            name = asset["path"]
            if Path(name).name != name or not name:
                raise ValueError("model catalog requires plain file names")
            target = root / name
            offset = target.stat().st_size if target.exists() else 0
            if offset > asset["size_bytes"]:
                raise ValueError("existing model asset larger than catalog; preserve it")
            h = hashlib.sha256()
            if offset:
                persist("verify_existing_prefix", file=name, bytes=offset)
                with target.open("rb") as stream:
                    for block in iter(lambda: stream.read(16*1024*1024), b""):
                        check(); h.update(block)
            if offset < asset["size_bytes"]:
                persist("download", file=name, bytes=offset, expected_bytes=asset["size_bytes"])
                url = f"https://huggingface.co/{catalog['repo_id']}/resolve/{catalog['revision']}/{name}"
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(request, timeout=min(20, max(1, deadline_epoch-time.time()-20))) as response:
                    if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                        raise ValueError("resume response does not match existing prefix; no overwrite")
                    with target.open("ab" if target.exists() else "xb") as stream:
                        while True:
                            check()
                            block = response.read(16*1024*1024)
                            if not block:
                                break
                            if offset+len(block) > asset["size_bytes"]:
                                raise ValueError("download exceeds catalog size")
                            stream.write(block); h.update(block); offset += len(block)
                            if time.time()-last_progress >= 5:
                                stream.flush()
                                persist("download", file=name, bytes=offset, expected_bytes=asset["size_bytes"])
                                last_progress = time.time()
                        stream.flush()
            if offset != asset["size_bytes"] or h.hexdigest() != asset["sha256"]:
                raise ValueError("downloaded model bytes do not match pinned catalog")
            receipt["verified_files"].append({"path": name, "bytes": offset, "sha256": h.hexdigest()})
            persist("asset_verified", file=name)
        # Check FP32 metadata without a second full-shard hash pass. Every byte
        # was already hashed above; normal inference still uses verify_model.
        from safetensors import safe_open
        for asset in catalog["files"]:
            check()
            if asset["path"].endswith(".safetensors"):
                with safe_open(root/asset["path"], framework="pt", device="cpu") as tensors:
                    if any(tensors.get_slice(k).get_dtype() != "F32" for k in tensors.keys()):
                        raise ValueError("original FP32 tensors required")
        config = read_json(root/"config.json")
        if (config["num_hidden_layers"],config["num_experts"],config["num_experts_per_tok"],config["norm_topk_prob"]) != (16,64,8,False):
            raise ValueError("unexpected pinned model architecture")
        receipt["status"] = "pass"
    except BaseException as error:
        receipt.update(status="incomplete" if isinstance(error,(TimeoutError,KeyboardInterrupt)) else "fail",
                       error_type=type(error).__name__)
        # Do not log HTTP redirect/presigned URLs or authentication headers.
    finally:
        persist("finished")
        atomic(output, canonical(receipt)+b"\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", required=True)
    p.add_argument("--mount", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    if not time.time() < a.deadline_epoch <= time.time()+900:
        p.error("deadline must be within the next 900 seconds")
    return 0 if prepare(a.catalog,a.mount,a.output,a.deadline_epoch)["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Copy sealed research bundles to and from a private Cloudflare R2 bucket.

The transfer boundary is deliberately narrow: only individual, complete bundles
of an approved kind under the repository's staging directories can be pushed.
The project root and arbitrary directories are never valid sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.artifacts import read_json, verify_bundle


ALLOWED_KINDS = {
    "inventory", "researchinventory", "prepared-inputs", "experiment-plan", "measurements",
    "calibration", "local-analysis",
}
INVENTORY_FILES = {"documents.jsonl", "exact-duplicates.jsonl", "cross-split-groups.jsonl",
                   "collection.json", "calibration.jsonl", "evaluation.jsonl", "metadata.json"}
PLAN_FILES = {"documents.jsonl", "inputs.jsonl", "research.json", "source-review.json",
              "model-catalog.json", "main-freeze.json", "manifest.json"}
CALIBRATION_FILES = {"frozen-tables.json", "stability-report.json", "routing-document-statistics.jsonl",
                     "routing-domain-statistics.jsonl", "probe-selections.jsonl", "manifest.json"}
KNOWN_FILES = {
    "inventory": INVENTORY_FILES,
    "researchinventory": INVENTORY_FILES,
    "prepared-inputs": {"documents.jsonl", "collection.json", "model-catalog.json", "research.json", "tokenizer-identity.json"},
    "experiment-plan": PLAN_FILES - {"manifest.json"},
    "calibration": CALIBRATION_FILES - {"manifest.json"},
    "measurements": {"identity.json", "events.jsonl", "failures.jsonl", "mechanics.json"},
    "local-analysis": {"event-analysis.jsonl", "document-summary.jsonl",
                       "source-group-summary.jsonl", "bootstrap-draws.jsonl", "analysis.json",
                       "sensitivity-event-comparison.jsonl", "sensitivity-analysis.json",
                       "main-planning.json", "main-freeze.json", "research-results.md"},
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_NAMES = {
    ".aws", ".env", ".netrc", "credentials", "credentials.json",
    "id_rsa", "id_ed25519", "secrets.json", "terraform.tfstate",
}


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_bundle_path(path: str | Path, kind: str, *, must_exist: bool) -> Path:
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"unsupported bundle kind: {kind}")
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _workspace_root() / candidate
    # Reject symlinked path components, even if resolution would land in an
    # approved directory. This prevents a staging entry from exposing secrets.
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlinked bundle paths are not allowed")
    candidate = candidate.resolve(strict=must_exist)
    workspace = _workspace_root()
    if candidate == workspace:
        raise ValueError("the project root cannot be transferred as a bundle")
    if must_exist and not candidate.is_dir():
        raise ValueError("bundle path must be a directory")
    if must_exist and any((candidate / marker).exists() for marker in (
            ".git", ".aws", ".codex", ".agents", "pyproject.toml", "README.md")):
        raise ValueError("project or credential directories cannot be transferred")
    return candidate


def _validate_manifest(manifest: dict[str, Any], kind: str) -> None:
    if manifest.get("kind") != kind or manifest.get("status") != "complete":
        raise ValueError("bundle kind/status does not match transfer request")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("manifest files must be a list")
    for row in rows:
        name = row.get("path")
        if not isinstance(name, str):
            raise ValueError("manifest file path must be text")
        if "\\" in name or ":" in name:
            raise ValueError(f"unsafe or sensitive file in bundle: {name}")
        parts = Path(name).parts
        leaf = Path(name).name
        if kind == "measurements":
            allowed_path = ((len(parts) == 1 and leaf in KNOWN_FILES[kind]) or
                            (len(parts) == 2 and parts[0] == "plan" and leaf in PLAN_FILES) or
                            (len(parts) == 2 and parts[0] == "calibration" and leaf in CALIBRATION_FILES))
        else:
            allowed_path = len(parts) == 1 and leaf in KNOWN_FILES[kind]
        if (Path(name).is_absolute() or not parts or any(p in ("", ".", "..") for p in parts)
                or any(p.lower() in _SENSITIVE_NAMES | {".git", ".codex", ".agents"} for p in parts)
                or any(p.lower().endswith((".pem", ".key", ".p12", ".pfx")) for p in parts)
                or not allowed_path):
            raise ValueError(f"unsafe or sensitive file in bundle: {name}")
        if not isinstance(row.get("size_bytes"), int) or row["size_bytes"] < 0:
            raise ValueError(f"invalid file size in manifest: {name}")
        if not isinstance(row.get("sha256"), str) or not _SHA256.fullmatch(row["sha256"]):
            raise ValueError(f"invalid SHA-256 in manifest: {name}")


def _client():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("R2 transfer requires boto3; install the project transfer extra") from exc
    endpoint = os.environ.get("R2_ENDPOINT_URL")
    account_id = os.environ.get("R2_ACCOUNT_ID")
    if not endpoint and account_id:
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    bucket = os.environ.get("R2_BUCKET")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    missing = [name for name, value in (
        ("R2_ENDPOINT_URL or R2_ACCOUNT_ID", endpoint), ("R2_BUCKET", bucket),
        ("R2_ACCESS_KEY_ID", access_key), ("R2_SECRET_ACCESS_KEY", secret_key),
    ) if not value]
    if missing:
        raise ValueError("missing R2 environment settings: " + ", ".join(missing))
    # Explicit credentials avoid the AWS credential-chain lookup and make the
    # private R2 endpoint visible in the configuration.
    session = boto3.session.Session()
    return session.client(
        "s3", endpoint_url=endpoint, region_name="auto",
        aws_access_key_id=access_key, aws_secret_access_key=secret_key,
    ), bucket


def _bundle_id(directory: Path) -> tuple[str, bytes]:
    manifest_bytes = (directory / "manifest.json").read_bytes()
    return hashlib.sha256(manifest_bytes).hexdigest(), manifest_bytes


def push(bundle: str | Path, kind: str) -> str:
    """Upload one verified bundle; write its manifest object last."""
    directory = _safe_bundle_path(bundle, kind, must_exist=True)
    manifest = verify_bundle(directory, kind)
    _validate_manifest(manifest, kind)
    client, bucket = _client()
    bundle_id, manifest_bytes = _bundle_id(directory)
    prefix = f"{kind}/{bundle_id}"
    for row in manifest["files"]:
        name = row["path"]
        client.upload_file(str(directory / name), bucket, f"{prefix}/{name}")
    verify_bundle(directory, kind)
    # A manifest is the commit marker: a reader cannot discover the bundle as
    # complete until every listed object has been uploaded.
    client.put_object(Bucket=bucket, Key=f"{prefix}/manifest.json", Body=manifest_bytes,
                      ContentType="application/json")
    return bundle_id


def pull(bundle_id: str, kind: str, destination: str | Path) -> Path:
    """Download a complete content-addressed bundle and verify every byte."""
    if not _SHA256.fullmatch(bundle_id):
        raise ValueError("bundle id must be a lowercase SHA-256 hex digest")
    target = _safe_bundle_path(destination, kind, must_exist=False)
    if target.exists():
        raise FileExistsError(f"destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    client, bucket = _client()
    prefix = f"{kind}/{bundle_id}"
    # Download to a sibling temporary directory. A failed transfer leaves no
    # partial destination and never alters an existing local bundle.
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.download-", dir=target.parent))
    try:
        response = client.get_object(Bucket=bucket, Key=f"{prefix}/manifest.json")
        manifest_bytes = response["Body"].read()
        if hashlib.sha256(manifest_bytes).hexdigest() != bundle_id:
            raise ValueError("remote manifest does not match requested bundle id")
        manifest = json.loads(manifest_bytes)
        _validate_manifest(manifest, kind)
        for row in manifest["files"]:
            name = row["path"]
            local = temp / name
            local.parent.mkdir(parents=True, exist_ok=True)
            response = client.get_object(Bucket=bucket, Key=f"{prefix}/{name}")
            size, checksum = 0, hashlib.sha256()
            with local.open("xb") as stream:
                for block in iter(lambda: response["Body"].read(1024 * 1024), b""):
                    stream.write(block)
                    size += len(block)
                    checksum.update(block)
                    if size > row["size_bytes"]:
                        raise ValueError(f"download exceeds manifest size: {name}")
            if size != row["size_bytes"] or checksum.hexdigest() != row["sha256"]:
                raise ValueError(f"download integrity mismatch: {name}")
        (temp / "manifest.json").write_bytes(manifest_bytes)
        verify_bundle(temp, kind)
        _validate_manifest(read_json(temp / "manifest.json"), kind)
        temp.rename(target)
        return target
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    push_parser = sub.add_parser("push", help="upload one sealed bundle")
    push_parser.add_argument("--kind", required=True, choices=sorted(ALLOWED_KINDS))
    push_parser.add_argument("--bundle", required=True, help="individual sealed scientific bundle; project root is forbidden")
    pull_parser = sub.add_parser("pull", help="retrieve and verify one bundle")
    pull_parser.add_argument("--kind", required=True, choices=sorted(ALLOWED_KINDS))
    pull_parser.add_argument("--bundle-id", required=True, help="SHA-256 of the remote manifest")
    pull_parser.add_argument("--destination", required=True, help="new local directory; project root is forbidden")
    args = parser.parse_args(argv)
    try:
        if args.action == "push":
            print(push(args.bundle, args.kind))
        else:
            print(pull(args.bundle_id, args.kind, args.destination))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"transfer failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

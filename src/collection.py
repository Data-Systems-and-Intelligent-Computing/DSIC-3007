"""Collect one pinned dataset into a bounded, audit-ready supplement bundle."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import islice
import re
from pathlib import Path

from .artifacts import atomic, canonical, file_digest, read_json, seal, write_rows
from .preparation import eligible_positions, verify_catalog


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset
    return load_dataset(*args, **kwargs)


def _load_tokenizer(directory):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                         trust_remote_code=False, use_fast=True)


def _versions():
    result = {"python": __import__("platform").python_version()}
    for package in ("datasets", "huggingface_hub", "transformers", "tokenizers"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _verify_tokenizer(directory, model_catalog):
    if model_catalog.get("revision_type") != "full_commit_sha":
        raise ValueError("model catalog must pin the tokenizer to a full model commit")
    assets = {row["path"]: row for row in model_catalog.get("files", [])}
    directory = Path(directory).resolve(strict=True)
    verified = []
    for name in TOKENIZER_FILES:
        asset = assets.get(name)
        path = directory / name
        if (asset is None or not path.is_file() or path.is_symlink() or
                path.stat().st_size != asset["size_bytes"] or
                file_digest(path) != asset["sha256"]):
            raise ValueError(f"local tokenizer does not match pinned model asset: {name}")
        verified.append({key: asset[key] for key in ("path", "size_bytes", "sha256")})
    return verified


def collect_dataset(*, domain, config, revision, max_source_rows, tokenizer_dir,
                    model_catalog_path, settings, output):
    """Scan all of a bounded pinned stream and retain every unique eligible row.

    `config` must provide `repo_id`, `split`, and `text_column`; `name`,
    `data_dir`, and `source_group_column` are optional dataset-specific fields.
    There is no document target, row shuffle, repository cap, or partition split.
    Every output document is placed in the reader's `evaluation` ingestion role;
    that role is not a scientific allocation.
    """
    if domain not in ("c4", "scientific", "code"):
        raise ValueError("domain must be c4, scientific, or code")
    if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
        raise ValueError("dataset revision must be a caller-supplied 40-character commit SHA")
    if type(max_source_rows) is not int or max_source_rows < 1:
        raise ValueError("max_source_rows must be a positive finite integer")
    if not config.get("repo_id") or not config.get("split") or not config.get("text_column"):
        raise ValueError("config requires repo_id, split, and text_column")
    if settings.get("sampling", {}).get("master_seed") != 3007:
        raise ValueError("supplement must use the approved master seed 3007")
    context = settings.get("context", {})
    window = context.get("primary_window")
    if window != 4096 or context.get("minimum_preceding_tokens") != 4095:
        raise ValueError("supplement eligibility requires the approved 4096 input-token window")
    excluded = context.get("excluded_predictor_target_ids", [])
    if not isinstance(excluded, list):
        raise ValueError("excluded predictor/target IDs must be a list")

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"choose a new output path: {output}")

    model_catalog = read_json(model_catalog_path)
    tokenizer_assets = _verify_tokenizer(tokenizer_dir, model_catalog)
    tokenizer = _load_tokenizer(Path(tokenizer_dir).resolve())
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("the pinned fast tokenizer is required")

    output.mkdir(parents=True, exist_ok=False)

    kwargs = {"path": config["repo_id"], "split": config["split"],
              "revision": revision, "streaming": True}
    if config.get("name") is not None:
        kwargs["name"] = config["name"]
    if config.get("data_dir") is not None:
        kwargs["data_dir"] = config["data_dir"]
    stream = _load_dataset(**kwargs)

    unique = {}
    rows_seen = 0
    text_rows = 0
    missing_text = 0
    duplicate_rows = 0
    tokenized_rows = 0
    ineligible_rows = 0
    eligible_positions_total = 0
    eligible_documents = 0
    probe_eligible_documents = 0
    source_group_column = config.get("source_group_column")
    for example in islice(stream, max_source_rows):
        rows_seen += 1
        if rows_seen % 1000 == 0:
            print(f"Scanned {rows_seen} source rows", flush=True)
        text = example.get(config["text_column"])
        if not isinstance(text, str) or not text:
            missing_text += 1
            continue
        text_rows += 1
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        tokenized_rows += 1
        positions = eligible_positions(token_ids, window, excluded)
        raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        metadata = {key: value for key, value in example.items()
                    if key != config["text_column"]}
        if source_group_column and source_group_column in example:
            metadata.setdefault(source_group_column, example[source_group_column])
        if raw_hash in unique:
            unique[raw_hash]["duplicate_source_metadata"].append(metadata)
            duplicate_rows += 1
            continue
        if not positions:
            ineligible_rows += 1
            continue
        eligible_positions_total += len(positions)
        eligible_documents += 1
        if len(positions) >= settings["sampling"]["probe_positions_per_document"]:
            probe_eligible_documents += 1
        unique[raw_hash] = {"text": text, "num_tokens": len(token_ids),
                            "content_sha256": raw_hash, "source_metadata": metadata,
                            "duplicate_source_metadata": []}

    # Keep only documents with an eligible 4096-token input and target. The
    # counts above still account for ineligible rows in the bounded scan.
    docs = list(unique.values())
    output_domain = output / domain
    output_domain.mkdir()
    asset_path = f"{domain}/evaluation.jsonl"
    write_rows(output / asset_path, docs)

    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "repo_id": config["repo_id"],
        "revision": revision,
        "split": config["split"],
        "config_name": config.get("name"),
        "data_dir": config.get("data_dir"),
        "text_column": config["text_column"],
        "source_group_column": source_group_column,
        "master_seed": 3007,
        "selection": "all unique eligible documents observed in the pinned bounded stream; no document subsampling or partition split",
        "ingestion_role": "evaluation (reader compatibility only; no scientific allocation implied)",
        "coverage": {
            "bound": max_source_rows,
            "bound_unit": "yielded source rows; streaming decoder/network read-ahead may exceed",
            "rows_seen": rows_seen,
            "source_exhausted_before_bound": rows_seen < max_source_rows,
            "rows_with_text": text_rows,
            "unique_eligible_documents": len(docs),
            "duplicate_rows_collapsed": duplicate_rows,
            "duplicate_count_scope": "eligible source rows only; repeated ineligible rows remain in the below-threshold row count",
            "rows_missing_text": missing_text,
            "rows_below_4096_input_plus_target": ineligible_rows,
            "tokenized_source_rows": tokenized_rows,
            "eligible_4096_input_plus_target_documents": eligible_documents,
            "eligible_predictor_positions": eligible_positions_total,
            "documents_with_at_least_four_positions": probe_eligible_documents,
            "eligibility_rule": "4096 input tokens and one subsequent target, with configured exclusions",
        },
        "tokenizer": {
            "repo_id": model_catalog["repo_id"],
            "revision": model_catalog["revision"],
            "files": tokenizer_assets,
            "add_special_tokens": False,
            "external_normalization": False,
        },
        "software_versions": _versions(),
    }
    metadata_path = output / domain / "metadata.json"
    atomic(metadata_path, canonical(metadata) + b"\n")

    files = []
    for path, role in ((asset_path, "evaluation"),
                       (f"{domain}/metadata.json", "collection_metadata")):
        file = output / path
        files.append({"path": path, "domain": domain, "role": role,
                      "size_bytes": file.stat().st_size, "sha256": file_digest(file),
                      "public_url": None})
    catalog = {
        "schema_version": 1,
        "kind": "bounded_supplemental_collection",
        "collection_id": f"supplement-{domain}-{revision[:12]}",
        "snapshot_identity_sha256": "",
        "sources": [{"domain": domain, "repo_id": config["repo_id"],
                     "revision": revision,
                     "tokenizer_repo_id": model_catalog["repo_id"],
                     "tokenizer_revision_at_collection": model_catalog["revision"],
                     "coverage": metadata["coverage"]}],
        "files": files,
        "distribution": {"provider": "cloudflare_r2", "status": "not_uploaded"},
    }
    catalog["snapshot_identity_sha256"] = __import__("hashlib").sha256(
        canonical(sorted(({"path": row["path"], "size_bytes": row["size_bytes"],
                          "sha256": row["sha256"]} for row in files), key=lambda row: row["path"]))
    ).hexdigest()
    catalog_path = output / "catalog.json"
    atomic(catalog_path, canonical(catalog) + b"\n")
    verify_catalog(output, catalog)
    manifest = seal(output, "supplemental-collection", {
        "domain": domain, "repo_id": config["repo_id"], "revision": revision,
        "max_source_rows": max_source_rows, "rows_seen": rows_seen,
        "unique_documents": len(docs), "eligible_4096_documents": eligible_documents,
        "probe_eligible_documents": probe_eligible_documents,
        "catalog_sha256": file_digest(catalog_path),
    })
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("c4", "scientific", "code"), required=True)
    parser.add_argument("--source-config", type=Path, required=True,
                        help="JSON with repo_id, split, text_column, and optional name/data_dir/source_group_column")
    parser.add_argument("--revision", required=True, help="Pinned 40-character dataset commit SHA")
    parser.add_argument("--max-source-rows", type=int, required=True,
                        help="Maximum yielded rows in the bounded source scan")
    parser.add_argument("--tokenizer-dir", type=Path, required=True,
                        help="Local tokenizer files matching the model catalog")
    parser.add_argument("--model-catalog", type=Path,
                        default=Path("data/manifests/model-fp32.json"))
    parser.add_argument("--settings", type=Path, default=Path("configs/research.json"))
    parser.add_argument("--output", type=Path, required=True,
                        help="New local output directory")
    args = parser.parse_args(argv)
    manifest = collect_dataset(
        domain=args.domain,
        config=read_json(args.source_config),
        revision=args.revision,
        max_source_rows=args.max_source_rows,
        tokenizer_dir=args.tokenizer_dir,
        model_catalog_path=args.model_catalog,
        settings=read_json(args.settings),
        output=args.output,
    )
    print(canonical({"status": manifest["status"], "kind": manifest["kind"],
                     "metadata": manifest["metadata"]}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

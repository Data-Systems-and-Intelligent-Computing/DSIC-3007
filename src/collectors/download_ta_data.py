from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from itertools import islice
import argparse
import hashlib
import json
import random

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

# =========================
# CONFIG
# =========================

MODEL = "allenai/OLMoE-1B-7B-0924"
# Agreed collection budget: inventory for later pilot/main selection, not power.
TARGET_DOCUMENTS = 10_000
MAX_FILES_PER_REPO = 10
CALIBRATION_RATIO = 0.15
SEED = 42
MAX_SOURCE_ROWS = 100_000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data"



DATASETS = {
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "split": "validation",
        "text_column": "text",
    },

    # Replace these with the exact dataset IDs/configs
    # once we lock the scientific/code datasets.
    "scientific": {
        "path": "common-pile/peS2o",
        "name": None,
        "split": "train",
        "text_column": "text",
    },

    "code": {
        "path": "bigcode/starcoderdata",
        "name": None,
        "split": "train",
        "text_column": "content",
        "data_dir": "python",
        "repo_group_column": "max_stars_repo_name",
    },
}


def split_documents(documents, *, calibration_ratio, seed, group_column=None):
    """Split docs, keeping each value of a source-metadata field together."""
    if not 0 < calibration_ratio < 1:
        raise ValueError("calibration_ratio must be between zero and one")
    # The collection and allocation budget is documents, not token volume.
    target_documents = max(1, int(len(documents) * calibration_ratio))

    if group_column is None:
        shuffled = list(documents)
        random.Random(seed).shuffle(shuffled)
        return shuffled[:target_documents], shuffled[target_documents:]

    def metadata_occurrences(document):
        primary = document.get("source_metadata", {})
        if isinstance(primary, dict):
            yield primary
        for extra in document.get("duplicate_source_metadata", []):
            if isinstance(extra, dict):
                yield extra

    def group_labels(document):
        labels = set()
        for metadata in metadata_occurrences(document):
            value = metadata.get(group_column)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Every selected document must have a non-empty source field "
                    f"{group_column!r}"
                )
            labels.add(value.strip())
        if not labels:
            raise ValueError(f"No source values found for {group_column!r}")
        return labels

    # If an exact duplicate carries multiple repo labels, join those labels so
    # the deduplicated document and all related repo groups remain on one side.
    parent = {}

    def find(label):
        parent.setdefault(label, label)
        if parent[label] != label:
            parent[label] = find(parent[label])
        return parent[label]

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    labels_by_document = []
    for document in documents:
        labels = sorted(group_labels(document))
        for label in labels:
            find(label)
        for label in labels[1:]:
            union(labels[0], label)
        labels_by_document.append(labels)

    component_labels = defaultdict(set)
    for label in parent:
        component_labels[find(label)].add(label)

    groups = defaultdict(list)
    for document, labels in zip(documents, labels_by_document):
        component_root = find(labels[0])
        component_key = min(component_labels[component_root])
        groups[component_key].append(document)

    if len(groups) < 2:
        raise ValueError(
            f"At least two independent {group_column!r} groups are required "
            "for a group-disjoint split"
        )

    ordered_groups = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{seed}:{group}".encode("utf-8")
        ).hexdigest(),
    )
    calibration = []
    evaluation = []

    for index, group in enumerate(ordered_groups):
        group_documents = groups[group]
        # Reserve at least one whole group for evaluation.
        if index < len(ordered_groups) - 1 and len(calibration) < target_documents:
            calibration.extend(group_documents)
        else:
            evaluation.extend(group_documents)

    return calibration, evaluation


def select_documents(documents, *, target_documents, max_files_per_repo,
                     group_column=None, seed=SEED):
    """Random order within the bounded pool; cap every observed repo label.

    Duplicate aliases count against each associated repository's cap. This
    conservative rule avoids silently exceeding the cap through copied files.
    """
    if target_documents < 2:
        raise ValueError("target_documents must be at least two")
    if group_column and (max_files_per_repo is None or max_files_per_repo < 1):
        raise ValueError("A positive max_files_per_repo is required for code")
    candidates = list(documents)
    random.Random(seed).shuffle(candidates)
    selected = []
    counts = defaultdict(int)
    for document in candidates:
        labels = set()
        if group_column:
            for metadata in [document["source_metadata"],
                             *document["duplicate_source_metadata"]]:
                labels.add(metadata[group_column].strip())
            if any(counts[label] >= max_files_per_repo for label in labels):
                continue
        selected.append(document)
        for label in labels:
            counts[label] += 1
        if len(selected) == target_documents:
            return selected
    raise RuntimeError(
        f"Bounded pool supplied only {len(selected)}/{target_documents} usable "
        "documents after deduplication and repository caps; no outputs written."
    )


def _write_jsonl(path, documents):
    with path.open("w", encoding="utf-8") as stream:
        for document in documents:
            stream.write(json.dumps(document, ensure_ascii=False) + "\n")


def _make_document(
    *, source_metadata, text, num_tokens, content_sha256
):
    """Keep the source payload once and preserve every other source field."""
    return {
        "text": text,
        "num_tokens": num_tokens,
        "content_sha256": content_sha256,
        "source_metadata": source_metadata,
        "duplicate_source_metadata": [],
    }


def _package_versions():
    names = ("datasets", "huggingface_hub", "transformers", "tokenizers")
    result = {"python": __import__("platform").python_version()}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _ensure_outputs_are_new(output):
    existing = [
        output / name
        for name in ("calibration.jsonl", "evaluation.jsonl", "metadata.json")
        if (output / name).exists()
    ]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing dataset artifacts: {paths}. "
            "Choose a new OUTPUT_DIR for a provenance-preserving collection."
        )


def process_dataset(dataset_name, config, tokenizer, tokenizer_revision, *,
                    target_documents, max_files_per_repo,
                    max_source_rows=MAX_SOURCE_ROWS,
                    calibration_ratio=CALIBRATION_RATIO, output_dir=OUTPUT_DIR):
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    if target_documents < 2 or max_source_rows < target_documents:
        raise ValueError("Require 2 <= target_documents <= max_source_rows")
    if not 0 < calibration_ratio < 1:
        raise ValueError("calibration_ratio must be between zero and one")
    if config.get("repo_group_column") and (
        max_files_per_repo is None or max_files_per_repo < 1
    ):
        raise ValueError("A positive repository cap is required for code")
    output = output_dir / dataset_name
    _ensure_outputs_are_new(output)

    requested_revision = config.get("revision")
    dataset_info = HfApi().dataset_info(
        config["path"], revision=requested_revision
    )
    resolved_revision = dataset_info.sha

    print("Loading dataset in streaming mode...")
    kwargs = {
        "path": config["path"],
        "split": config["split"],
        "streaming": True,
        "revision": resolved_revision,
    }
    if config["name"] is not None:
        kwargs["name"] = config["name"]
    if "data_dir" in config:
        kwargs["data_dir"] = config["data_dir"]

    ds = load_dataset(**kwargs)
    # HF shuffles source shards where supported. A one-row buffer avoids a
    # large implicit lookahead; selection below scans a bounded yielded pool.
    # This is not uniform sampling across the complete remote corpus.
    ds = ds.shuffle(seed=SEED, buffer_size=1)
    documents = []
    total_tokens = 0
    documents_seen = 0
    documents_missing_repo_label = 0
    exact_duplicates_removed = 0
    documents_by_hash = {}
    source_columns = set()
    text_column = config["text_column"]
    repo_group_column = config.get("repo_group_column")

    # islice enforces the bound even if every row takes a continue branch.
    # It bounds yielded records, not network bytes/Parquet decoder read-ahead.
    for example in islice(ds, max_source_rows):
        documents_seen += 1
        source_columns.update(example.keys())

        text = example.get(text_column)
        if not text or not isinstance(text, str):
            continue

        if repo_group_column is not None:
            repo_label = example.get(repo_group_column)
            if not isinstance(repo_label, str) or not repo_label.strip():
                documents_missing_repo_label += 1
                continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 10:
            continue

        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_metadata = {
            key: value for key, value in example.items() if key != text_column
        }
        if content_sha256 in documents_by_hash:
            documents_by_hash[content_sha256]["duplicate_source_metadata"].append(
                source_metadata
            )
            exact_duplicates_removed += 1
            continue

        document = _make_document(
            source_metadata=source_metadata,
            text=text,
            num_tokens=len(tokens),
            content_sha256=content_sha256,
        )
        documents.append(document)
        documents_by_hash[content_sha256] = document
        total_tokens += len(tokens)

        if documents_seen % 1000 == 0:
            print(
                f"seen={documents_seen:,} "
                f"selected={len(documents):,} "
                f"tokens={total_tokens:,}"
            )

    candidate_count = len(documents)
    documents = select_documents(
        documents, target_documents=target_documents,
        max_files_per_repo=max_files_per_repo,
        group_column=repo_group_column,
    )
    total_tokens = sum(document["num_tokens"] for document in documents)

    calibration, evaluation = split_documents(
        documents,
        calibration_ratio=calibration_ratio,
        seed=SEED,
        group_column=repo_group_column,
    )
    if not calibration or not evaluation:
        raise ValueError(
            f"{dataset_name} did not produce non-empty calibration and evaluation "
            "partitions; refusing to write incomplete outputs"
        )

    calibration_tokens = sum(item["num_tokens"] for item in calibration)
    evaluation_tokens = sum(item["num_tokens"] for item in evaluation)
    split_tokens = calibration_tokens + evaluation_tokens
    calibration_groups = set()
    evaluation_groups = set()
    if repo_group_column is not None:
        for item in calibration:
            metadata_rows = [
                item["source_metadata"],
                *item["duplicate_source_metadata"],
            ]
            calibration_groups.update(
                row[repo_group_column].strip() for row in metadata_rows
            )
        for item in evaluation:
            metadata_rows = [
                item["source_metadata"],
                *item["duplicate_source_metadata"],
            ]
            evaluation_groups.update(
                row[repo_group_column].strip() for row in metadata_rows
            )
    shared_groups = calibration_groups & evaluation_groups
    if shared_groups:
        raise AssertionError("Repository groups crossed the split boundary")

    # All validation completes before writing any collection artifacts.
    output.mkdir(parents=True, exist_ok=True)

    calibration_path = output / "calibration.jsonl"
    evaluation_path = output / "evaluation.jsonl"
    _write_jsonl(calibration_path, calibration)
    _write_jsonl(evaluation_path, evaluation)

    metadata = {
        "artifact_schema_version": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "hf_path": config["path"],
        "hf_config": config["name"],
        "hf_data_dir": config.get("data_dir"),
        "split": config["split"],
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "source_text_column": text_column,
        "source_metadata_fields_saved": sorted(source_columns - {text_column}),
        "source_metadata_policy": (
            "All source-row fields except the original text field are preserved "
            "under each record's source_metadata; exact-duplicate metadata rows "
            "are retained in duplicate_source_metadata; text is stored once."
        ),
        "model_tokenizer": MODEL,
        "model_tokenizer_revision": tokenizer_revision,
        "software_versions": _package_versions(),
        "seed": SEED,
        "target_documents": target_documents,
        "max_files_per_repo": max_files_per_repo if repo_group_column else None,
        "max_source_rows": max_source_rows,
        "unique_candidates": candidate_count,
        "sampling_policy": (
            "Seeded HF source-shard shuffle (buffer_size=1), bounded yielded "
            "record pool, exact deduplication, seeded candidate shuffle, "
            "then repository caps. Not uniform full-corpus sampling."
        ),
        "source_bound_unit": "yielded rows; decoder/network read-ahead may exceed",
        "actual_tokens": total_tokens,
        "calibration_ratio_target": calibration_ratio,
        "calibration_ratio_unit": "documents; whole groups may overshoot",
        "calibration_documents": len(calibration),
        "evaluation_documents": len(evaluation),
        "calibration_ratio_actual_documents": len(calibration) / len(documents),
        "calibration_ratio_actual_tokens": (
            calibration_tokens / split_tokens if split_tokens else None
        ),
        "calibration_tokens": calibration_tokens,
        "evaluation_tokens": evaluation_tokens,
        "documents_seen": documents_seen,
        "documents_selected": len(documents),
        "documents_missing_repo_label": documents_missing_repo_label,
        "exact_duplicates_removed": exact_duplicates_removed,
        "exact_duplicates_count_scope": "bounded candidate pool before caps",
        "split_unit": "repository_label" if repo_group_column else "document",
        "repository_grouping_field": repo_group_column,
        "calibration_repository_groups": len(calibration_groups),
        "evaluation_repository_groups": len(evaluation_groups),
        "repository_groups_shared_across_splits": len(shared_groups),
        "repository_grouping_warning": (
            "max_stars_repo_name is preserved and used as a repository-label "
            "proxy; it is not complete file provenance."
            if repo_group_column
            else None
        ),
    }

    with (output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)

    print("\nDONE")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Collect a bounded document pool")
    parser.add_argument("--target-documents", type=int, default=TARGET_DOCUMENTS,
                        help="Unique documents per domain (default: 10000)")
    parser.add_argument("--max-files-per-repo", type=int, default=MAX_FILES_PER_REPO,
                        help="Maximum selected files per code repo label (default: 10)")
    parser.add_argument("--max-source-rows", type=int, default=MAX_SOURCE_ROWS)
    parser.add_argument("--calibration-ratio", type=float, default=CALIBRATION_RATIO,
                        help="Target document fraction (default: 0.15)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New output directory; existing artifacts are refused")
    args = parser.parse_args()
    if not 2 <= args.target_documents <= args.max_source_rows:
        parser.error("Require 2 <= target-documents <= max-source-rows")
    if args.max_files_per_repo < 1 or not 0 < args.calibration_ratio < 1:
        parser.error("Require positive repo cap and calibration ratio in (0, 1)")
    for name in DATASETS:
        _ensure_outputs_are_new(args.output_dir / name)

    print(f"Loading tokenizer: {MODEL}")
    tokenizer_revision = HfApi().model_info(MODEL).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=tokenizer_revision)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for name, config in DATASETS.items():
        process_dataset(
            name, config, tokenizer, tokenizer_revision,
            target_documents=args.target_documents,
            max_files_per_repo=args.max_files_per_repo,
            max_source_rows=args.max_source_rows,
            calibration_ratio=args.calibration_ratio,
            output_dir=args.output_dir,
        )

    print("\nAll datasets processed.")


if __name__ == "__main__":
    main()

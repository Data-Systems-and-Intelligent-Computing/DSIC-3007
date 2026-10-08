"""Bounded, read-only profile of StarCoderData Python repository metadata.

This samples a small shuffled buffer from the source, prints aggregate counts,
and never prints or saves code contents. It is an exploratory probe, not a
representative dataset audit or a data collection/splitting pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi


DATASET_ID = "bigcode/starcoderdata"
DATA_DIR = "python"
SOURCE_SPLIT = "train"
MAX_PROBE_ROWS = 500
MAX_SHUFFLE_BUFFER = 128


def _repo_goes_to_calibration(repo_name: str, *, seed: int, share: float) -> bool:
    """Assign a whole repo label to one side reproducibly."""
    payload = f"{seed}:{repo_name}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    return value < share


def profile(
    *,
    max_rows: int,
    shuffle_buffer: int,
    max_files_per_repo: int,
    calibration_share: float,
    seed: int,
) -> dict[str, Any]:
    revision = HfApi().dataset_info(DATASET_ID).sha
    source = load_dataset(
        DATASET_ID,
        data_dir=DATA_DIR,
        split=SOURCE_SPLIT,
        streaming=True,
        revision=revision,
    )
    bounded_sample = source.shuffle(seed=seed, buffer_size=shuffle_buffer).take(max_rows)

    counts = Counter()
    seen_content_hashes: set[str] = set()
    files_per_repo: Counter[str] = Counter()
    split_by_repo: dict[str, str] = {}
    selected_per_split: Counter[str] = Counter()
    selected_repo_labels: set[str] = set()

    for row in bounded_sample:
        counts["records_examined"] += 1
        content = row.get("content")
        repo_name = row.get("max_stars_repo_name")

        if not isinstance(content, str) or not content:
            counts["records_without_content"] += 1
            continue
        if not isinstance(repo_name, str) or not repo_name.strip():
            counts["records_without_repo_label"] += 1
            continue

        counts["records_with_content_and_repo_label"] += 1
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest in seen_content_hashes:
            counts["exact_content_duplicates_in_sample"] += 1
            continue
        seen_content_hashes.add(digest)

        counts["unique_contents_in_sample"] += 1
        split = split_by_repo.setdefault(
            repo_name,
            "calibration"
            if _repo_goes_to_calibration(
                repo_name, seed=seed, share=calibration_share
            )
            else "evaluation",
        )
        if files_per_repo[repo_name] >= max_files_per_repo:
            counts["unique_contents_over_repo_cap"] += 1
            continue

        files_per_repo[repo_name] += 1
        selected_repo_labels.add(repo_name)
        selected_per_split[split] += 1

    calibration_repos = {
        repo for repo in selected_repo_labels
        if split_by_repo[repo] == "calibration"
    }
    evaluation_repos = selected_repo_labels - calibration_repos

    return {
        "source": {
            "dataset": DATASET_ID,
            "data_dir": DATA_DIR,
            "split": SOURCE_SPLIT,
            "revision": revision,
        },
        "probe": {
            "max_rows": max_rows,
            "shuffle_buffer": shuffle_buffer,
            "seed": seed,
            "sampling_note": (
                "Bounded shuffle-buffer sample from a streaming source; "
                "not a uniform sample of the full corpus."
            ),
        },
        "candidate_rules_not_final": {
            "repo_label": "max_stars_repo_name",
            "max_unique_files_per_repo": max_files_per_repo,
            "calibration_repo_share": calibration_share,
            "evaluation_repo_share": 1.0 - calibration_share,
            "deduplication": "exact UTF-8 content SHA-256 within this probe",
        },
        "counts": {
            **counts,
            "unique_repo_labels_before_cap": len(split_by_repo),
            "repo_labels_with_selected_files": len(selected_repo_labels),
            "calibration_repo_labels": len(calibration_repos),
            "evaluation_repo_labels": len(evaluation_repos),
            "calibration_selected_files": selected_per_split["calibration"],
            "evaluation_selected_files": selected_per_split["evaluation"],
            "total_selected_files": sum(selected_per_split.values()),
        },
        "limitations": [
            "max_stars_repo_name is a representative repo label, not complete provenance.",
            "The bounded shuffle-buffer probe cannot establish full-corpus coverage.",
            "Different files from related or forked repos may still cross the split.",
            "Counts describe only this probe and do not finalize study sample sizes.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--shuffle-buffer", type=int, default=128)
    parser.add_argument("--max-files-per-repo", type=int, default=2)
    parser.add_argument("--calibration-share", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.max_rows <= MAX_PROBE_ROWS:
        parser.error(f"--max-rows must be between 1 and {MAX_PROBE_ROWS}")
    if not 1 <= args.shuffle_buffer <= MAX_SHUFFLE_BUFFER:
        parser.error(f"--shuffle-buffer must be between 1 and {MAX_SHUFFLE_BUFFER}")
    if args.max_files_per_repo < 1:
        parser.error("--max-files-per-repo must be positive")
    if not 0.0 < args.calibration_share < 1.0:
        parser.error("--calibration-share must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    result = profile(
        max_rows=args.max_rows,
        shuffle_buffer=args.shuffle_buffer,
        max_files_per_repo=args.max_files_per_repo,
        calibration_share=args.calibration_share,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

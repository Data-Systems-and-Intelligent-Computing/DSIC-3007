"""Screen preparation must preserve group completeness and paired context bytes."""

from copy import deepcopy

import pytest

from src.artifacts import atomic, canonical, digest, file_digest, seal, write_rows
from src.benchmark import prepare_benchmark


def parents(tmp_path):
    primary, short = tmp_path / "primary", tmp_path / "short"
    settings = {"domains": ["scientific", "code", "c4"], "sampling": {"master_seed": 3007}}
    documents, inputs = [], []
    for domain in settings["domains"]:
        for group in range(4):
            for member in range(2 if domain == "code" else 1):
                doc_id = f"{domain}-{group}-{member}"
                documents.append(dict(domain=domain, document_id=doc_id, source_group=str(group)))
                inputs.append(dict(domain=domain, document_id=doc_id, source_group=str(group),
                    input_id=digest(doc_id), layers=[0, 7, 15], predictor_position=4095,
                    target_position=4096, target_id=1, input_ids=[2] * 4096))
    for path, window in [(primary, 4096), (short, 2048)]:
        path.mkdir()
        rows = deepcopy(inputs)
        if window == 2048:
            for row in rows:
                row["parent_input_id"] = row["input_id"]
                row["input_id"] = digest([row["input_id"], window])
                row["input_ids"] = row["input_ids"][-window:]
        write_rows(path / "documents.jsonl", documents)
        write_rows(path / "inputs.jsonl", rows)
        for name, value in [("research.json", settings), ("source-review.json", {}), ("model-catalog.json", {})]:
            atomic(path / name, canonical(value) + b"\n")
        metadata = dict(stage="technical", window=window, units={domain: 4 for domain in settings["domains"]})
        if window == 2048:
            metadata["parent_plan_sha256"] = file_digest(primary / "manifest.json")
        seal(path, "experiment-plan", metadata)
    return primary, short


def test_screen_keeps_all_files_in_selected_groups_and_counts_reference(tmp_path):
    primary, short = parents(tmp_path)
    result = prepare_benchmark(primary, short, tmp_path / "screen")
    assert result["domain_counts"]["4096"]["code"] == {"documents": 6, "groups": 3, "windows": 6}
    assert result["windows"] == 24
    assert result["native_reference_forwards"] == 624
    assert result["independent_validator_full_forwards"] == 648
    assert result["optimized_full_baseline_cleanup_forwards"] == 72
    assert result["individual_ablations"] == 576
    assert result["status"] == "prepared_local_not_authorized"
    from src.artifacts import verify_bundle
    assert verify_bundle(tmp_path / "screen" / "4096")["metadata"]["units"]["code"] == 3


def test_screen_rejects_context_byte_mismatch_even_when_bundle_is_sealed(tmp_path):
    primary, short = parents(tmp_path)
    from src.artifacts import read_rows
    rows = list(read_rows(short / "inputs.jsonl"))
    for row in rows:
        row["input_ids"][0] = 3
    (short / "inputs.jsonl").unlink()
    write_rows(short / "inputs.jsonl", rows)
    (short / "manifest.json").unlink()
    seal(short, "experiment-plan", dict(stage="technical", window=2048, units={d: 4 for d in ("code", "c4", "scientific")},
        parent_plan_sha256=file_digest(primary / "manifest.json")))
    with pytest.raises(ValueError, match="different input bytes"):
        prepare_benchmark(primary, short, tmp_path / "screen")

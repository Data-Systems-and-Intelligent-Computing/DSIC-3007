"""Source metadata evidence cannot silently approve a scientific allocation."""

import pytest

from src.artifacts import canonical, read_rows, seal, verify_bundle, write_rows
from src.source_audit import audit_sources


def document(identifier, domain="scientific", group=None, split="calibration", **extra):
    return dict(document_id=identifier, domain=domain, source_group=group or identifier,
        retained_split=split, content_sha256=identifier, actual_tokens=4100,
        eligible_positions=4, source_metadata={"metadata": {}}, **extra)


def test_audit_flags_variants_duplicates_and_short_record_source_overlap(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    a, b = document("a"), document("b", split="evaluation")
    for d in (a, b):
        d["source_metadata"]["metadata"]["oa_url"] = "https://publisher/paper"
    c = document("c", domain="c4", group="host")
    d = document("d", domain="c4", group="host", split="evaluation")
    d.update(eligible_positions=0, actual_tokens=30, content_sha256="c")
    e = document("e", domain="code", group="Repo/Name")
    f = document("f", domain="code", group="repo/name")
    for row in (e, f):
        row["source_metadata"] = {"max_stars_repo_name": row["source_group"]}
    write_rows(prepared / "documents.jsonl", [a, b, c, d, e, f])
    seal(prepared, "prepared-inputs", {"tokenizer_sha256": "fixture"})
    result = audit_sources(prepared, tmp_path / "audit")["metadata"]
    assert result["exact_duplicate_content_sets"] == 1
    assert result["identity_links_requiring_review"] == {
        "scientific/oa_url": 1, "code/repository_name_casefold": 1}
    assert result["counts"]["c4"]["retained_split_overlapping_groups"] == 1
    assert result["counts"]["c4"]["eligible_split_overlapping_groups"] == 0
    assert result["counts"]["c4"]["eligible_native_groups"] == 1
    assert result["status"] == "evidence_complete_review_pending"
    assert all(v != "pass" for v in result["checks"].values())
    links = list(read_rows(tmp_path / "audit" / "identity-links.jsonl"))
    assert all(r["decision"] == "pending" for r in links)
    verify_bundle(tmp_path / "audit", "source-audit")
    canonical(result)


def test_audit_rejects_duplicate_document_identity(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    write_rows(prepared / "documents.jsonl", [document("a"), document("a")])
    seal(prepared, "prepared-inputs")
    with pytest.raises(ValueError, match="duplicate document identity"):
        audit_sources(prepared, tmp_path / "audit")

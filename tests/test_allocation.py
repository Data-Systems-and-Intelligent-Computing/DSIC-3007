import hashlib
import json

from src.allocation import allocate
from src.artifacts import canonical, read_json, seal, write_rows
from src.preparation import reviewed_documents


def _document(domain, index, *, group=None, content=None, metadata=None,
              aliases=(), duplicate_metadata=(), eligible=4):
    if group is None:
        group = {"c4": f"host-{index}.example", "scientific": f"paper-{index}",
                 "code": f"owner/repo-{index}"}[domain]
    if metadata is None:
        if domain == "c4":
            metadata = {"url": f"https://{group}/doc/{index}"}
        elif domain == "scientific":
            metadata = {"id": group, "metadata": {}}
        else:
            metadata = {"max_stars_repo_name": group}
    content = content or f"{domain}-document-{index}"
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return {
        "document_id": f"{domain}-document-{index}", "domain": domain,
        "source_group": group, "source_group_aliases": list(aliases),
        "content_sha256": content_hash, "actual_tokens": 4200,
        "eligible_positions": eligible, "source_metadata": metadata,
        "duplicate_source_metadata": list(duplicate_metadata),
        "retained_split": "evaluation",
    }


def _prepared(tmp_path, documents):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    settings = read_json("configs/research.json")
    write_rows(prepared / "documents.jsonl", documents)
    (prepared / "research.json").write_bytes(canonical(settings) + b"\n")
    seal(prepared, "prepared-inputs", {"tokenizer_sha256": "fixture"})
    return prepared, settings


def _sufficient_documents():
    result = []
    for domain in ("c4", "scientific", "code"):
        # 12 technical groups + 32 probe documents + 100 calibration
        # documents + 80 pilot units + 200 main units.
        for index in range(424):
            result.append(_document(domain, index))
    return result


def test_allocate_freezes_disjoint_partitions_and_keeps_whole_code_groups(tmp_path):
    documents = _sufficient_documents()
    documents[0]["source_group_aliases"] = ["host-1.example"]
    documents[0]["duplicate_source_metadata"] = [{"url": "https://host-1.example/alias"}]
    documents.extend([
        _document("code", 424, group="owner/shared", content="shared-file-a"),
        _document("code", 425, group="owner/shared", content="shared-file-b"),
    ])
    prepared, settings = _prepared(tmp_path, documents)
    output = tmp_path / "source-review.json"
    review = allocate(prepared, settings, output, calibration_ceiling=100)

    assert review["status"] == "approved"
    assert review["checks"]["source_group_disjointness"] == "pass"
    _, groups = reviewed_documents(prepared, output)
    for domain in settings["domains"]:
        stages_by_group = {}
        for (group_domain, stage, group), members in groups.items():
            if group_domain != domain:
                continue
            assert group not in stages_by_group
            stages_by_group[group] = stage
            assert members
        assert review["capacity"][domain]["technical_groups"] == 12
        assert review["capacity"][domain]["probe_documents_available"] >= 32
        assert review["capacity"][domain]["calibration_documents_available"] >= 100
        assert review["capacity"][domain]["pilot_units_available"] >= 80
        assert review["capacity"][domain]["main_units"] >= 200

    shared = [row for row in documents if row["source_group"] == "owner/shared"]
    decisions = review["documents"]
    assert len({decisions[row["document_id"]]["stage"] for row in shared}) == 1
    assert all(decisions[row["document_id"]]["stage"] != "excluded" for row in shared)
    linked_c4 = [documents[0], documents[1]]
    assert {decisions[row["document_id"]]["source_group"] for row in linked_c4} == {"host-0.example"}
    assert len({decisions[row["document_id"]]["stage"] for row in linked_c4}) == 1
    assert json.loads(output.read_text())["status"] == "approved"


def test_allocate_uses_stable_exact_duplicate_representative_and_known_alias_groups(tmp_path):
    documents = [
        _document("code", 0, group="Owner/Repo", content="same exact file"),
        _document("code", 1, group="owner/repo", content="same exact file"),
        _document("scientific", 0, group="paper-a", content="paper copy",
                  metadata={"id": "paper-a", "metadata": {"pdf_hash": "hash-1"}}),
        _document("scientific", 1, group="paper-b", content="different copy",
                  metadata={"id": "paper-b", "metadata": {"pdf_hash": "hash-1"}}),
    ]
    prepared, settings = _prepared(tmp_path, documents)
    first = allocate(prepared, settings, tmp_path / "review-a.json", calibration_ceiling=100)
    second = allocate(prepared, settings, tmp_path / "review-b.json", calibration_ceiling=100)

    assert first["documents"] == second["documents"]
    code_rows = [row for row in documents if row["domain"] == "code"]
    assert sum(first["documents"][row["document_id"]]["stage"] != "excluded"
               for row in code_rows) == 1
    loser = next(row for row in code_rows if first["documents"][row["document_id"]]["stage"] == "excluded")
    assert first["documents"][loser["document_id"]]["reason"].startswith("exact duplicate content")
    science = [first["documents"][row["document_id"]] for row in documents
               if row["domain"] == "scientific"]
    assert {row["source_group"] for row in science} == {"paper-a"}


def test_allocate_excludes_ambiguous_publication_url_groups_and_reports_shortfall(tmp_path):
    documents = [
        _document("scientific", 0, group="paper-a", metadata={"id": "paper-a",
                  "metadata": {"oa_url": "https://publisher/paper"}}),
        _document("scientific", 1, group="paper-b", metadata={"id": "paper-b",
                  "metadata": {"oa_url": "https://publisher/paper"}}),
    ]
    prepared, settings = _prepared(tmp_path, documents)
    output = tmp_path / "candidate.json"
    review = allocate(prepared, settings, output, calibration_ceiling=100)

    assert review["status"] == "pending_insufficient_capacity"
    assert review["checks"]["article_variants"] == "pass"
    assert all(review["documents"][row["document_id"]]["stage"] == "excluded"
               for row in documents)
    assert all("ambiguous shared scientific publication URL" in
               review["documents"][row["document_id"]]["reason"] for row in documents)
    assert review["capacity"]["scientific"]["calibration_documents_available"] == 0
    assert review["shortfalls"]


def test_explicit_document_exclusion_does_not_make_content_dedup_nondeterministic(tmp_path):
    documents = [
        _document("c4", 0, content="same c4 text"),
        _document("c4", 1, content="same c4 text"),
        _document("c4", 2),
    ]
    prepared, settings = _prepared(tmp_path, documents)
    excluded = documents[0]["document_id"]
    review = allocate(prepared, settings, tmp_path / "review.json",
                      excluded_document_ids=[excluded], calibration_ceiling=100)
    assert review["documents"][excluded]["reason"].startswith("explicit")
    assert sum(review["documents"][row["document_id"]]["stage"] != "excluded"
               for row in documents[:2]) == 1
    assert review["audit_scope"]["repository_aliases"].endswith("not claimed as checked")

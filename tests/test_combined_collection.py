"""Overlapping corpus acquisitions must not duplicate observation identities."""

import hashlib

import pytest

from src.artifacts import atomic, canonical, file_digest, read_json, seal, write_rows
from src.preparation import combine_collections, retained_documents, source_group, verify_catalog


def test_c4_source_unit_preserves_the_complete_hostname():
    assert source_group("c4", {"url": "https://WWW.Example.org/a"}) == "www.example.org"
    assert source_group("c4", {"url": "https://example.org/a"}) == "example.org"
    assert source_group("c4", {"url": "https://sub.example.org/a"}) == "sub.example.org"


def collection(root, records, revision="a" * 40):
    root.mkdir()
    path = root / "documents.jsonl"
    write_rows(path, [{"text": text, "num_tokens": 4100,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_metadata": {"id": native}, "duplicate_source_metadata": aliases}
        for text, native, aliases in records])
    catalog = {"sources": [{"domain": "scientific", "repo_id": "org/source", "revision": revision}],
               "files": [{"domain": "scientific", "role": "evaluation", "path": path.name,
                          "size_bytes": path.stat().st_size, "sha256": file_digest(path)}]}
    atomic(root / "catalog.json", canonical(catalog))
    return catalog


def test_combined_identity_collapse_preserves_aliases_and_both_raw_occurrences(tmp_path):
    original = tmp_path / "original"
    original_catalog = collection(original, [("original text", "paper", [])])
    # The retained catalog convention makes paths relative to the workspace.
    original_catalog["files"][0]["path"] = "original/documents.jsonl"
    atomic(tmp_path / "retained-catalog.json", canonical(original_catalog))
    supplement = tmp_path / "supplement"
    collection(supplement, [("original text", "paper", [{"id": "known-alias"}]),
                           ("different text", "another", [])], revision="a" * 40)
    seal(supplement, "supplemental-collection")
    combined = combine_collections(tmp_path, tmp_path / "retained-catalog.json",
                                   [supplement], tmp_path / "combined.json")
    verify_catalog(tmp_path, combined)
    assert [a["source_index"] for a in combined["files"]] == [0, 1]
    docs = list(retained_documents(tmp_path, combined))
    assert len(docs) == 2
    assert len(docs[0]["collection_occurrences"]) == 2
    assert docs[0]["source_group_aliases"] == ["known-alias"]
    assert docs[0]["duplicate_source_metadata"] == [{"id": "known-alias"}]
    with pytest.raises(FileExistsError):
        combine_collections(tmp_path, tmp_path / "retained-catalog.json",
                            [supplement], tmp_path / "combined.json")


def test_catalog_cannot_silently_choose_last_source_for_a_domain(tmp_path):
    catalog = collection(tmp_path / "source", [("text", "paper", [])])
    catalog["sources"].append(dict(catalog["sources"][0], revision="b" * 40))
    with pytest.raises(ValueError, match="ambiguous dataset source"):
        list(retained_documents(tmp_path / "source", catalog))
    catalog["files"][0]["source_index"] = 0
    assert next(retained_documents(tmp_path / "source", catalog))["revision"] == "a" * 40
    catalog["files"][0]["source_index"] = 1
    assert next(retained_documents(tmp_path / "source", catalog))["revision"] == "b" * 40

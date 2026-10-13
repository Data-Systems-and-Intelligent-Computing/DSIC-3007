"""Local source identity audit; evidence, not automatic scientific approval."""

from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import (atomic, canonical, file_digest, read_rows, seal,
                        verify_bundle, write_rows)


def audit_sources(prepared, output):
    """Check every retained record, including short records that can link sources.

    Known metadata links are reported for review. A common publication URL is
    not proof that two paper IDs should be merged; no similarity threshold or
    source grouping rule is invented here.
    """
    manifest = verify_bundle(prepared, "prepared-inputs")
    groups, content, identities = defaultdict(list), defaultdict(list), defaultdict(list)
    counts, fields = defaultdict(Counter), defaultdict(Counter)
    rows, seen = [], set()
    for d in read_rows(Path(prepared) / "documents.jsonl"):
        if d["document_id"] in seen:
            raise ValueError("duplicate document identity in prepared inventory")
        seen.add(d["document_id"])
        domain = d["domain"]
        compact = {k: d[k] for k in ("document_id", "domain", "source_group",
            "retained_split", "content_sha256", "actual_tokens", "eligible_positions")}
        aliases = sorted(set(a for a in d.get("source_group_aliases", []) if a))
        compact["source_group_aliases"] = aliases
        rows.append(compact)
        counts[domain]["retained_documents"] += 1
        counts[domain]["eligible_documents"] += int(d["eligible_positions"] > 0)
        counts[domain]["probe_eligible_documents"] += int(d["eligible_positions"] >= 4)
        counts[domain]["missing_native_group_documents"] += int(not d["source_group"])
        counts[domain]["documents_with_duplicate_source_aliases"] += bool(aliases)
        content[d["content_sha256"]].append(compact)
        if d["source_group"]:
            groups[domain, d["source_group"]].append(compact)
        metadata = [d["source_metadata"], *d.get("duplicate_source_metadata", [])]
        for meta in metadata:
            fields[domain].update(meta.keys())
            if domain == "scientific":
                nested = meta.get("metadata") or {}
                for key in ("pdf_hash", "sha1", "oa_url", "doi", "arxiv_id"):
                    value = nested.get(key) or meta.get(key)
                    if value:
                        identities[domain, key, str(value)].append(compact)
            elif domain == "code" and meta.get("max_stars_repo_name"):
                identities[domain, "repository_name_casefold",
                           str(meta["max_stars_repo_name"]).casefold()].append(compact)
    duplicates = [{"content_sha256": h, "documents": docs}
                  for h, docs in sorted(content.items()) if len(docs) > 1]
    links = []
    for (domain, key, value), docs in sorted(identities.items()):
        unique = {d["document_id"]: d for d in docs}
        if len({d["source_group"] for d in unique.values()}) > 1:
            links.append({"domain": domain, "field": key, "value": value,
                          "documents": list(unique.values()), "decision": "pending"})
    overlap = []
    for (domain, group), docs in sorted(groups.items()):
        splits = sorted({d["retained_split"] for d in docs})
        if len(splits) > 1:
            overlap.append({"domain": domain, "source_group": group, "splits": splits,
                "document_ids": [d["document_id"] for d in docs],
                "eligible_document_ids": [d["document_id"] for d in docs if d["eligible_positions"]]})
    for domain in counts:
        domain_groups = [ds for (dm, _), ds in groups.items() if dm == domain]
        eligible_groups = [ds for ds in domain_groups if any(d["eligible_positions"] for d in ds)]
        counts[domain]["native_groups"] = len(domain_groups)
        counts[domain]["eligible_native_groups"] = len(eligible_groups)
        counts[domain]["retained_split_overlapping_groups"] = sum(o["domain"] == domain for o in overlap)
        counts[domain]["eligible_split_overlapping_groups"] = sum(o["domain"] == domain and
            len({d["retained_split"] for d in groups[domain, o["source_group"]]
                 if d["eligible_positions"]}) > 1 for o in overlap)
    summary = {
        "status": "evidence_complete_review_pending",
        "prepared_manifest_sha256": file_digest(Path(prepared) / "manifest.json"),
        "tokenizer_sha256": manifest["metadata"].get("tokenizer_sha256"),
        "counts": {d: dict(c) for d, c in sorted(counts.items())},
        "exact_duplicate_content_sets": len(duplicates),
        "identity_links_requiring_review": dict(Counter(l["domain"] + "/" + l["field"] for l in links)),
        "available_source_fields": {d: dict(c) for d, c in sorted(fields.items())},
        "checks": {"duplicate_content": "checked_exact_bytes",
            "article_variants": "metadata_links_require_review" if any(l["domain"] == "scientific" for l in links)
                                else "no_metadata_links_found_not_all_variants_excluded",
            "repository_aliases_and_known_clones": "metadata_checked_external_clone_relationships_not_supplied",
            "source_group_disjointness": "retained_split_overlap_checked_new_stage_allocation_pending"},
        "limits": ["No article-text similarity threshold or repository-clone detector was specified.",
            "A shared publication URL is a review flag, not an automatic paper-ID merge.",
            "Source identity checks cover retained metadata, not an external repository genealogy.",
            "Original calibration/evaluation labels are not the five new scientific partitions.",
            "Eligibility and native group counts precede variant decisions and final allocation."]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, records in (("documents.jsonl", rows), ("exact-duplicates.jsonl", duplicates),
                          ("identity-links.jsonl", links), ("retained-split-overlap.jsonl", overlap)):
        write_rows(output / name, records)
    atomic(output / "summary.json", canonical(summary) + b"\n")
    return seal(output, "source-audit", summary)

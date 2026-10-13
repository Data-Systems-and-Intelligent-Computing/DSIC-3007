"""Deterministic, outcome-blind source-group allocation for the research stages."""

from collections import defaultdict
import os
from pathlib import Path

from .artifacts import (atomic, canonical, file_digest, priority, read_json,
                        read_rows, verify_bundle)
from .preparation import reviewed_documents, source_group
from .scope import is_path_b, scope_identity, stage_domains


_STAGES = ("technical", "probe", "calibration", "pilot", "main")
_PILOT_MIN_SOURCE_GROUPS = 10


def _metadata_rows(document):
    yield document.get("source_metadata") or {}
    yield from (document.get("duplicate_source_metadata") or [])


def _identities(document):
    """Return recorded group and conservative scientific identity evidence."""
    domain = document["domain"]
    groups = {g for g in [document.get("source_group"),
                           *(document.get("source_group_aliases") or [])] if g}
    paper_hashes, publication_urls = set(), set()
    for metadata in _metadata_rows(document):
        group = source_group(domain, metadata)
        if group:
            groups.add(group)
        if domain == "scientific":
            nested = metadata.get("metadata") or {}
            for field in ("pdf_hash", "sha1"):
                value = nested.get(field) or metadata.get(field)
                if value and group:
                    paper_hashes.add((field, str(value), group))
            url = nested.get("oa_url") or metadata.get("oa_url")
            if url and group:
                publication_urls.add((str(url), group))
    return groups, paper_hashes, publication_urls


class _Components:
    def __init__(self):
        self.parent = {}
        self.labels = defaultdict(set)

    def key(self, domain, group):
        # Repository names are identifiers with case-insensitive aliases in
        # the retained metadata. Keep their original spelling as the label.
        normalized = group.casefold() if domain == "code" else group
        key = (domain, normalized)
        self.parent.setdefault(key, key)
        self.labels[self.find(key)].add(group)
        return key

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, keys):
        keys = sorted({self.find(key) for key in keys})
        if len(keys) < 2:
            return
        root = keys[0]
        for other in keys[1:]:
            left, right = self.find(root), self.find(other)
            if left == right:
                continue
            low, high = sorted((left, right))
            self.parent[high] = low
            self.labels[low].update(self.labels.pop(high, set()))
            root = low

    def canonical_labels(self):
        result = {}
        roots = {self.find(key) for key in self.parent}
        for root in roots:
            values = self.labels[root]
            result[root] = min(values, key=lambda x: (x.casefold(), x))
        return result


def _ordered(groups, domain, stage, seed):
    return sorted(groups, key=lambda g: (priority(seed, "source-allocation",
                                                 [domain, stage, g]), g))


def allocate(prepared, settings, output, excluded_document_ids=(), calibration_ceiling=500):
    """Write a complete source-review file, approved only when all minima fit.

    Group allocation uses only prepared metadata, eligibility counts, IDs and
    the fixed seed. No experiment outcomes or document text enter the choice.
    `calibration_ceiling` must be one of the approved nested calibration sizes;
    it is never silently reduced to fit the available source pool.
    """
    verify_bundle(prepared, "prepared-inputs")
    prepared = Path(prepared)
    frozen_settings = read_json(prepared / "research.json")
    if canonical(settings) != canonical(frozen_settings):
        raise ValueError("allocation settings must match the sealed prepared research settings")
    master_seed = settings["sampling"]["master_seed"]
    if type(master_seed) is not int or master_seed != 3007:
        raise ValueError("source allocation requires the approved master seed 3007")
    if calibration_ceiling not in settings["sampling"]["calibration_document_ladder"]:
        raise ValueError("calibration ceiling must be one of the approved nested document budgets")
    excluded_ids = set(excluded_document_ids)
    if any(not isinstance(value, str) or not value for value in excluded_ids):
        raise ValueError("explicit exclusions must be nonempty document identity strings")

    documents = []
    for row in read_rows(prepared / "documents.jsonl"):
        # Allocation needs identities and eligibility, not large token arrays.
        row.pop("token_ids", None)
        documents.append(row)
    by_id = {row["document_id"]: row for row in documents}
    if len(by_id) != len(documents):
        raise ValueError("prepared inventory contains repeated document identities")
    unknown = excluded_ids - set(by_id)
    if unknown:
        raise ValueError(f"explicit exclusion IDs are absent from prepared inputs: {len(unknown)}")

    components = _Components()
    group_keys = {}
    hash_docs, paper_hash_docs, url_groups = defaultdict(list), defaultdict(list), defaultdict(set)
    url_native_groups = defaultdict(set)
    for row in documents:
        domain = row["domain"]
        raw_groups, paper_hashes, publication_urls = _identities(row)
        keys = [components.key(domain, value) for value in sorted(raw_groups)]
        group_keys[row["document_id"]] = keys
        components.union(keys)
        for field, value, group in paper_hashes:
            paper_hash_docs[domain, field, value].append(components.key(domain, group))
        for value, group in publication_urls:
            key = components.key(domain, group)
            url_groups[domain, value].add(key)
            url_native_groups[domain, value].add(group)
        hash_docs[row["content_sha256"]].append(row)

    # Known identical paper files and exact repeated text connect their source
    # groups, including aliases found on short or explicitly excluded records.
    for keys in paper_hash_docs.values():
        components.union(keys)
    for duplicate_rows in hash_docs.values():
        by_domain = defaultdict(list)
        for row in duplicate_rows:
            by_domain[row["domain"]].extend(group_keys[row["document_id"]])
        for keys in by_domain.values():
            components.union(keys)

    # A common publication URL is ambiguous evidence, so exclude those entire
    # known source components instead of merging paper IDs on URL alone.
    ambiguous_roots = set()
    for (domain, _url), keys in url_groups.items():
        roots = {components.find(key) for key in keys}
        if len(url_native_groups[domain, _url]) > 1:
            ambiguous_roots.update(roots)
    # Resolve after all identity unions: an ambiguity excludes every document
    # in any alias/hash-linked component it touched.
    ambiguous_roots = {components.find(root) for root in ambiguous_roots}
    canonical_by_root = components.canonical_labels()
    canonical_group = {}
    for document_id, keys in group_keys.items():
        roots = {components.find(key) for key in keys}
        if roots:
            if len(roots) != 1:
                raise ValueError("source identity evidence did not form one component")
            canonical_group[document_id] = canonical_by_root[next(iter(roots))]
        else:
            canonical_group[document_id] = None

    reasons = {}
    for row in documents:
        document_id = row["document_id"]
        root_set = {components.find(key) for key in group_keys[document_id]}
        if document_id in excluded_ids:
            reasons[document_id] = "explicit outcome-independent document exclusion"
        elif not row["eligible_positions"]:
            reasons[document_id] = "no eligible full-context predictor and target"
        elif not row.get("source_group"):
            reasons[document_id] = "missing native source group"
        elif root_set & ambiguous_roots:
            reasons[document_id] = "ambiguous shared scientific publication URL"

    # Select one deterministic representative for every exact text hash among
    # otherwise eligible records. All losing rows remain accounted as excluded.
    active_by_hash = defaultdict(list)
    for row in documents:
        if row["document_id"] not in reasons:
            active_by_hash[row["content_sha256"]].append(row)
    for content_hash, rows in active_by_hash.items():
        winner = min(rows, key=lambda row: (
            priority(master_seed, "exact-content-representative",
                     [content_hash, row["document_id"]]), row["document_id"]))
        for row in rows:
            if row["document_id"] != winner["document_id"]:
                reasons[row["document_id"]] = (
                    "exact duplicate content; representative " + winner["document_id"])

    groups = defaultdict(list)
    for row in documents:
        document_id = row["document_id"]
        if document_id in reasons:
            continue
        group = canonical_group[document_id]
        if not group:
            reasons[document_id] = "missing native source group"
            continue
        groups[row["domain"], group].append(row)

    technical_units = settings["sampling"]["technical_units_per_domain"]
    probe_units = settings["sampling"]["probe_units_per_domain"]
    pilot_units = max(settings["sampling"]["pilot_document_ladder"])
    main_minima = {
        "c4": settings["sampling"]["main_c4_documents"],
        "scientific": settings["sampling"]["main_min_scientific_documents"],
        "code": settings["sampling"]["main_min_code_groups"],
    }
    assignments = {}
    sampling_by_group = {}
    capacity = {}

    for domain in settings["domains"]:
        available = {group: members for (d, group), members in groups.items() if d == domain}
        remaining = set(available)
        selected = {stage: set() for stage in _STAGES}
        counts = {group: len(members) for group, members in available.items()}

        # Technical source units are whole groups.
        tech_order = _ordered(remaining, domain, "technical", master_seed)
        selected["technical"].update(tech_order[:technical_units])
        remaining.difference_update(selected["technical"])

        # A probe group is eligible only when every retained document in it can
        # supply the fixed four positions, because plans sample among its docs.
        probe_order = _ordered((g for g in remaining if all(
            row["eligible_positions"] >= settings["sampling"]["probe_positions_per_document"]
            for row in available[g])), domain, "probe", master_seed)
        probe_docs = 0
        for group in probe_order:
            if probe_docs >= probe_units:
                break
            selected["probe"].add(group)
            probe_docs += counts[group]
        remaining.difference_update(selected["probe"])

        # Calibration is a document pool formed from whole source groups.
        calibration_order = _ordered(remaining, domain, "calibration", master_seed)
        calibration_docs = 0
        for group in calibration_order:
            if calibration_docs >= calibration_ceiling:
                break
            selected["calibration"].add(group)
            calibration_docs += counts[group]
        remaining.difference_update(selected["calibration"])

        # Pilot uses all documents from its chosen groups; code budgets are
        # repository groups while C4/science budgets are documents.
        pilot_order = _ordered(remaining, domain, "pilot", master_seed)
        pilot_count = 0
        for group in pilot_order if domain in stage_domains(settings, "pilot") else []:
            if (pilot_count >= pilot_units and
                    (domain == "code" or len(selected["pilot"]) >= _PILOT_MIN_SOURCE_GROUPS)):
                break
            selected["pilot"].add(group)
            pilot_count += 1 if domain == "code" else counts[group]
        remaining.difference_update(selected["pilot"])

        main_documents = sum(counts[group] for group in remaining)
        main_units = len(remaining) if domain == "code" else main_documents
        capacity[domain] = {
            "eligible_documents": sum(counts.values()),
            "eligible_source_groups": len(available),
            "technical_groups": len(selected["technical"]),
            "probe_documents_available": probe_docs,
            "probe_groups": len(selected["probe"]),
            "calibration_documents_available": calibration_docs,
            "calibration_groups": len(selected["calibration"]),
            "pilot_units_available": pilot_count,
            "pilot_groups": len(selected["pilot"]),
            "pilot_minimum_source_groups": ((pilot_units if domain == "code"
                                            else _PILOT_MIN_SOURCE_GROUPS)
                                           if domain in stage_domains(settings, "pilot") else 0),
            "main_documents": main_documents,
            "main_units": main_units,
            "main_minimum": main_minima[domain],
        }
        for stage, stage_groups in selected.items():
            ordered_selected = _ordered(stage_groups, domain, stage, master_seed)
            for rank, group in enumerate(ordered_selected):
                sampling_by_group[domain, group] = {
                    "stage": stage, "unit": "source_group", "rank": rank,
                    "seed": master_seed,
                    "priority_sha256": priority(master_seed, "source-allocation",
                                                 [domain, stage, group]),
                }
                for row in available[group]:
                    assignments[row["document_id"]] = stage
        for rank, group in enumerate(_ordered(remaining, domain, "main", master_seed)):
            if domain not in stage_domains(settings, "main"):
                for row in available[group]:
                    reasons[row["document_id"]] = "Path B defers C4 pilot/main outcomes; retained calibration/probe only"
                continue
            sampling_by_group[domain, group] = {
                "stage": "main", "unit": "source_group", "rank": rank,
                "seed": master_seed,
                "priority_sha256": priority(master_seed, "source-allocation",
                                             [domain, "main", group]),
            }
            for row in available[group]:
                assignments[row["document_id"]] = "main"

    shortfalls = []
    for domain, row in capacity.items():
        for name, observed, required in (
            ("technical_groups", row["technical_groups"], technical_units),
            ("probe_documents_available", row["probe_documents_available"], probe_units),
            ("calibration_documents_available", row["calibration_documents_available"], calibration_ceiling),
            ("pilot_units_available", row["pilot_units_available"],
             pilot_units if domain in stage_domains(settings, "pilot") else 0),
            ("pilot_groups", row["pilot_groups"], row["pilot_minimum_source_groups"]),
            ("main_units", row["main_units"], row["main_minimum"]),
        ):
            if observed < required:
                shortfalls.append({"domain": domain, "capacity": name,
                                   "available": observed, "required": required})

    decisions = {}
    for row in documents:
        document_id = row["document_id"]
        native = row.get("source_group")
        recorded_groups, _, _ = _identities(row)
        stage = assignments.get(document_id)
        reason = reasons.get(document_id)
        if reason:
            stage = "excluded"
        elif stage is None:
            raise ValueError("eligible document was not assigned to a research stage")
        decisions[document_id] = {
            "domain": row["domain"],
            "stage": stage,
            "reason": reason,
            "source_group": canonical_group[document_id],
            "native_source_group": native,
            "source_group_aliases": sorted(recorded_groups - ({native} if native else set())),
            "content_sha256": row["content_sha256"],
            "sampling": (sampling_by_group.get((row["domain"], canonical_group[document_id]))
                        if stage != "excluded" else None),
        }

    stage_counts = {domain: {stage: {"documents": 0, "source_groups": set()}
                             for stage in (*_STAGES, "excluded")}
                    for domain in settings["domains"]}
    for row in documents:
        decision = decisions[row["document_id"]]
        stats = stage_counts[row["domain"]][decision["stage"]]
        stats["documents"] += 1
        if decision["source_group"]:
            stats["source_groups"].add(decision["source_group"])
    for domain in stage_counts:
        for stage in stage_counts[domain]:
            stage_counts[domain][stage]["source_groups"] = len(stage_counts[domain][stage]["source_groups"])

    status = "approved" if not shortfalls else "pending_insufficient_capacity"
    review = {
        "schema_version": 1,
        "status": status,
        "research_scope": scope_identity(settings),
        "prepared_manifest_sha256": file_digest(prepared / "manifest.json"),
        "master_seed": master_seed,
        "calibration_ceiling": calibration_ceiling,
        "capacity": capacity,
        "shortfalls": shortfalls,
        "stage_counts": stage_counts,
        "checks": {
            "duplicate_content": "pass",
            "article_variants": "pass",
            "repository_aliases_and_known_clones": "pass",
            "source_group_disjointness": "pass" if not shortfalls else "pending",
        },
        "audit_scope": {
            "duplicate_content": "exact UTF-8 text hashes use one seeded representative; aliases connect source groups",
            "scientific_article_variants": "recorded pdf_hash and sha1 connect known copies; distinct paper IDs sharing an exact recorded oa_url are excluded; no text-similarity threshold is claimed",
            "repository_aliases": "recorded source aliases and case-folded repository names connect groups; external clone relationships absent from metadata are not claimed as checked",
            "source_group_disjointness": "all eligible groups are assigned to one technical, probe, calibration, pilot, or main stage; excluded records do not enter a stage",
        },
        "excluded_document_ids": sorted(excluded_ids),
        "documents": decisions,
    }

    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new source-review output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".candidate")
    if temporary.exists():
        raise FileExistsError("remove stale allocation candidate before retrying")
    try:
        atomic(temporary, canonical(review) + b"\n")
        if status == "approved":
            # This is the same strict validator used by downstream planning.
            reviewed_documents(prepared, temporary)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return review

"""Retained-input identity, source groups, tokenization and frozen research plans."""

from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from .artifacts import (atomic, canonical, digest, file_digest, priority, read_json,
                        read_rows, require, seal, verify_bundle, write_rows)


def verify_catalog(root, catalog):
    root = Path(root).resolve()
    for row in catalog["files"]:
        from .artifacts import relative
        p = root / relative(row["path"])
        if (p.is_symlink() or root not in p.resolve().parents or
                p.stat().st_size != row["size_bytes"] or file_digest(p) != row["sha256"]):
            raise ValueError(f"input identity mismatch: {row['path']}")


def source_group(domain, metadata):
    if domain == "c4":
        host = urlsplit(metadata.get("url", "")).hostname
        return host.lower() if host else None
    value = metadata.get("id") if domain == "scientific" else metadata.get("max_stars_repo_name")
    return str(value) if value is not None and str(value).strip() else None


def _catalog_documents(root, catalog):
    sources = defaultdict(list)
    for source in catalog["sources"]:
        sources[source["domain"]].append(source)
    for asset in catalog["files"]:
        if asset["role"] not in ("calibration", "evaluation"):
            continue
        domain = asset["domain"]
        if "source_index" in asset:
            index = asset["source_index"]
            if type(index) is not int or not 0 <= index < len(catalog["sources"]):
                raise ValueError("invalid collection source reference")
            source = catalog["sources"][index]
            if source["domain"] != domain:
                raise ValueError("collection asset/source domain mismatch")
        else:
            if len(sources[domain]) != 1:
                raise ValueError("ambiguous dataset source: asset requires source_index")
            source = sources[domain][0]
        for index, row in enumerate(read_rows(Path(root) / asset["path"])):
            text_hash = __import__("hashlib").sha256(row["text"].encode("utf-8")).hexdigest()
            if text_hash != row["content_sha256"]:
                raise ValueError("retained content hash mismatch")
            meta = row["source_metadata"]
            identity = {"dataset": source["repo_id"], "domain": domain,
                        "revision": source["revision"],
                        "native_id": meta.get("id", meta.get("url")), "content_sha256": text_hash}
            aliases = [source_group(domain, m) for m in row.get("duplicate_source_metadata", [])]
            yield dict(identity, document_id=digest(identity),
                       retained_split=asset["role"], asset_path=asset["path"], asset_row=index,
                       source_group=source_group(domain, meta), source_group_aliases=aliases,
                       reported_tokens=row["num_tokens"], source_metadata=meta,
                       duplicate_source_metadata=row.get("duplicate_source_metadata", []), text=row["text"])


def retained_documents(root, catalog):
    if not catalog.get("collapse_repeated_document_identities", False):
        yield from _catalog_documents(root, catalog)
        return
    # First pass retains metadata only. Repeated appearances of the exact same
    # dataset/revision/native-ID/text document are one observation, while all
    # raw assets, occurrence identities and duplicate-source metadata survive.
    occurrences, extra_metadata = defaultdict(list), defaultdict(list)
    for row in _catalog_documents(root, catalog):
        identifier = row["document_id"]
        occurrences[identifier].append({"asset_path": row["asset_path"],
            "asset_row": row["asset_row"], "retained_split": row["retained_split"]})
        if len(occurrences[identifier]) > 1:
            extra_metadata[identifier].append(row["source_metadata"])
        extra_metadata[identifier].extend(row["duplicate_source_metadata"])
    emitted = set()
    for row in _catalog_documents(root, catalog):
        identifier = row["document_id"]
        if identifier in emitted:
            continue
        emitted.add(identifier)
        unique = {digest(meta): meta for meta in extra_metadata[identifier]
                  if meta != row["source_metadata"]}
        row["duplicate_source_metadata"] = list(unique.values())
        row["source_group_aliases"] = [source_group(row["domain"], m) for m in unique.values()]
        row["collection_occurrences"] = occurrences[identifier]
        yield row


def combine_collections(root, retained_catalog_path, supplements, output):
    """Reference verified original and supplemental bytes without copying corpora."""
    root = Path(root).resolve(strict=True)
    catalogs = [(root, read_json(retained_catalog_path))]
    for supplement in supplements:
        supplement = Path(supplement).resolve(strict=True)
        verify_bundle(supplement, "supplemental-collection")
        catalogs.append((supplement, read_json(supplement / "catalog.json")))
    combined = {"schema_version": 1, "kind": "combined_research_collection",
                "collapse_repeated_document_identities": True, "sources": [], "files": []}
    seen_paths = set()
    for asset_root, catalog in catalogs:
        verify_catalog(asset_root, catalog)
        offset = len(combined["sources"])
        combined["sources"].extend(catalog["sources"])
        for asset in catalog["files"]:
            absolute = (asset_root / asset["path"]).resolve(strict=True)
            if root not in absolute.parents:
                raise ValueError("combined inputs must be inside the selected collection root")
            relative_path = absolute.relative_to(root).as_posix()
            if relative_path in seen_paths:
                raise ValueError("repeated raw collection asset")
            seen_paths.add(relative_path)
            row = dict(asset, path=relative_path)
            if row["role"] in ("calibration", "evaluation"):
                matches = [i for i, s in enumerate(catalog["sources"]) if s["domain"] == row["domain"]]
                local_index = asset.get("source_index")
                if local_index is None:
                    if len(matches) != 1:
                        raise ValueError("ambiguous input catalog source")
                    local_index = matches[0]
                if type(local_index) is not int or local_index not in matches:
                    raise ValueError("invalid input catalog source reference")
                row["source_index"] = offset + local_index
            combined["files"].append(row)
    combined["snapshot_identity_sha256"] = digest(combined)
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new combined catalog output")
    atomic(output, canonical(combined) + b"\n")
    return combined


def inventory(root, catalog_path, output):
    catalog = read_json(catalog_path)
    verify_catalog(root, catalog)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = defaultdict(lambda: {"documents": 0, "reported_full_context_candidates": 0})
    seen, groups = defaultdict(list), defaultdict(set)
    rows = []
    for row in retained_documents(root, catalog):
        row.pop("text")
        key = row["domain"] + "/" + row["retained_split"]
        counts[key]["documents"] += 1
        counts[key]["reported_full_context_candidates"] += int(row["reported_tokens"] >= 4097)
        seen[row["content_sha256"]].append(row["document_id"])
        if row["source_group"]:
            groups[row["domain"], row["source_group"]].add(row["retained_split"])
        rows.append(row)
    write_rows(output / "documents.jsonl", rows)
    write_rows(output / "exact-duplicates.jsonl",
               ({"content_sha256": h, "document_ids": ids} for h, ids in sorted(seen.items()) if len(ids) > 1))
    write_rows(output / "cross-split-groups.jsonl",
               ({"domain": d, "source_group": g, "splits": sorted(s)}
                for (d, g), s in sorted(groups.items()) if len(s) > 1))
    atomic(output / "collection.json", canonical(catalog) + b"\n")
    return seal(output, "inventory", {"counts": dict(counts),
        "eligibility": "reported lengths only; pinned tokenization and source review still required"})


def acquire_tokenizer(catalog_path, output):
    """Download only three small pinned tokenizer files, never model weights."""
    from huggingface_hub import hf_hub_download
    catalog = read_json(catalog_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for asset in catalog["files"]:
        if asset["path"] not in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            continue
        destination = output / asset["path"]
        if not destination.exists():
            cached = hf_hub_download(catalog["repo_id"], asset["path"], revision=catalog["revision"])
            atomic(destination, Path(cached).read_bytes())
        if (destination.stat().st_size != asset["size_bytes"] or
                file_digest(destination) != asset["sha256"]):
            raise ValueError("tokenizer differs from the pinned model catalog")


def eligible_positions(tokens, window, excluded=()):
    excluded = set(excluded)
    return [q for q in range(window - 1, len(tokens) - 1)
            if tokens[q] not in excluded and tokens[q + 1] not in excluded]


def sample_positions(tokens, doc_id, window, budget, seed, stage, excluded=()):
    candidates = eligible_positions(tokens, window, excluded)
    return sorted(candidates, key=lambda q: (priority(seed, stage, [doc_id, q]), q))[:budget]


def window_for(tokens, q, window):
    if type(q) is not int or q < window - 1 or q + 1 >= len(tokens):
        raise ValueError("full input window and subsequent target required")
    return {"window_start": q - window + 1, "window_end": q + 1,
            "predictor_position": q, "local_predictor_position": window - 1,
            "context_length": window, "target_position": q + 1, "target_id": tokens[q + 1],
            "input_ids": tokens[q - window + 1:q + 1]}


def prepare(root, catalog_path, model_catalog_path, tokenizer_dir, output, settings):
    """Preserve all original records, derived tokens, candidate groups and audit status."""
    from transformers import AutoTokenizer
    import importlib.metadata
    catalog, model_catalog = read_json(catalog_path), read_json(model_catalog_path)
    verify_catalog(root, catalog)
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_assets = [a for a in model_catalog["files"] if a["path"] in
                        ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")]
    for a in tokenizer_assets:
        if ((tokenizer_dir / a["path"]).stat().st_size != a["size_bytes"] or
                file_digest(tokenizer_dir / a["path"]) != a["sha256"]):
            raise ValueError("pinned tokenizer identity mismatch")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("fast pinned tokenizer required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counts = defaultdict(lambda: {"documents": 0, "full_context_documents": 0, "eligible_positions": 0})

    def rows():
        for index, row in enumerate(retained_documents(root, catalog), 1):
            tokens = tokenizer.encode(row.pop("text"), add_special_tokens=False)
            qs = eligible_positions(tokens, settings["context"]["primary_window"],
                                    settings["context"]["excluded_predictor_target_ids"])
            key = row["domain"] + "/" + row["retained_split"]
            counts[key]["documents"] += 1
            counts[key]["full_context_documents"] += int(bool(qs))
            counts[key]["eligible_positions"] += len(qs)
            row.update(token_ids=tokens, token_ids_sha256=digest(tokens),
                       actual_tokens=len(tokens), eligible_positions=len(qs))
            if index % 1000 == 0:
                print(f"Tokenized {index} documents", flush=True)
            yield row
    write_rows(output / "documents.jsonl", rows())
    atomic(output / "collection.json", canonical(catalog) + b"\n")
    atomic(output / "model-catalog.json", canonical(model_catalog) + b"\n")
    atomic(output / "research.json", canonical(settings) + b"\n")
    identity = {"files": tokenizer_assets, "add_special_tokens": False,
                "external_normalization": False, "software": {n: importlib.metadata.version(n)
                for n in ("transformers", "tokenizers")}}
    atomic(output / "tokenizer-identity.json", canonical(identity) + b"\n")
    return seal(output, "prepared-inputs", {"counts": dict(counts), "source_audit": "pending",
                                          "tokenizer_sha256": digest(identity)})


def reviewed_documents(prepared, review_path):
    verify_bundle(prepared, "prepared-inputs")
    review = read_json(review_path)
    if review.get("status") != "approved" or review.get("prepared_manifest_sha256") != file_digest(Path(prepared) / "manifest.json"):
        raise ValueError("source review must approve this exact prepared inventory")
    checks = ("duplicate_content", "article_variants", "repository_aliases_and_known_clones", "source_group_disjointness")
    if any(review.get("checks", {}).get(name) != "pass" for name in checks):
        raise ValueError("all source and duplicate audit checks must pass before allocation")
    docs = list(read_rows(Path(prepared) / "documents.jsonl"))
    by_id = {d["document_id"]: d for d in docs}
    if len(by_id) != len(docs) or set(review["documents"]) != set(by_id):
        raise ValueError("review must account for every distinct retained document")
    # Alias evidence can bridge included units through excluded or short files.
    # Build these links from the entire prepared inventory before allocating.
    parents, native_keys = {}, {}

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for d in docs:
        aliases = [d.get("source_group"), *d.get("source_group_aliases", [])]
        aliases.extend(source_group(d["domain"], m)
                       for m in d.get("duplicate_source_metadata", []))
        keys = sorted({(d["domain"], a) for a in aliases if a})
        native_keys[d["document_id"]] = keys
        for key in keys:
            find(key)
        for key in keys[1:]:
            a, b = find(keys[0]), find(key)
            parents[max(a, b)] = min(a, b)
    group_stages, component_labels, hashes = {}, {}, set()
    groups = defaultdict(list)
    for d in docs:
        decision = review["documents"][d["document_id"]]
        if decision["stage"] not in ("excluded", "technical", "calibration", "probe", "pilot", "main"):
            raise ValueError("every document needs a final stage or exclusion")
        if decision["stage"] == "excluded":
            if not decision.get("reason"):
                raise ValueError("exclusions require an outcome-independent reason")
            continue
        group = decision.get("source_group")
        if not group or not d["eligible_positions"]:
            raise ValueError("included document requires a reviewed group and full context")
        keys = native_keys[d["document_id"]]
        if keys:
            component = find(keys[0])
            previous = component_labels.setdefault(component, group)
            if previous != group:
                raise ValueError("known source aliases must share one reviewed canonical source group")
        key = d["domain"], group
        if key in group_stages and group_stages[key] != decision["stage"]:
            raise ValueError("source-group leakage between partitions")
        group_stages[key] = decision["stage"]
        if d["content_sha256"] in hashes:
            raise ValueError("duplicate content in included source partitions")
        hashes.add(d["content_sha256"])
        groups[d["domain"], decision["stage"], group].append(dict(d, source_group=group))
    return review, groups


def rebind_prepared(prepared, output, settings):
    """Create a versioned scope successor, preserving verified token bytes."""
    import shutil
    from .scope import is_path_b, scope_identity
    manifest = verify_bundle(prepared, "prepared-inputs")
    old = read_json(Path(prepared) / "research.json")
    if not is_path_b(settings):
        raise ValueError("prepared scope migration currently supports Path B only")
    for key in ("model", "domains"):
        if old[key] != settings[key]:
            raise ValueError("scope migration cannot change model or tokenization domains")
    for key in ("primary_window", "minimum_preceding_tokens", "window_policy", "excluded_predictor_target_ids"):
        if old["context"][key] != settings["context"][key]:
            raise ValueError("scope migration cannot change context eligibility or tokenization")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for asset in manifest["files"]:
        name = asset["path"]
        if name == "research.json":
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(prepared) / name, target)
    atomic(out / "research.json", canonical(settings) + b"\n")
    return seal(out, "prepared-inputs", dict(manifest["metadata"],
        parent_prepared_manifest_sha256=file_digest(Path(prepared) / "manifest.json"),
        research_scope=scope_identity(settings), source_audit="pending_new_scope_allocation",
        tokenization_reused_unchanged=True))


def make_plan(prepared, review_path, output, settings, stage, budgets, seed=None,
              window=4096, calibration_sample=None, positions=None):
    """Review explicitly assigns disjoint groups; no hidden document reallocations."""
    metadata = verify_bundle(prepared, "prepared-inputs")
    review, groups = reviewed_documents(prepared, review_path)
    require(settings, "sampling.allocation_rule")
    if stage not in ("technical", "calibration", "probe", "pilot", "main"):
        raise ValueError("unknown research stage")
    seed = (int(priority(settings["sampling"]["master_seed"], "stage-seed", stage)[:16], 16)
            if seed is None else seed)
    from .scope import is_path_b, require_scope, scope_identity, stage_domains, stage_layers
    domains = stage_domains(settings, stage)
    if is_path_b(settings) and read_json(Path(prepared) / "research.json") != settings:
        raise ValueError("Path B needs a prepared scope successor before planning")
    require_scope(settings, review)
    if set(budgets) != set(domains):
        raise ValueError("plan budgets must match the selected stage domains")
    if is_path_b(settings) and window != 4096:
        raise ValueError("Path B uses exact 4096 inputs only; matched 2048 is deferred")
    chosen = []
    for domain in domains:
        pool = [(g, rows) for (d, role, g), rows in groups.items() if d == domain and role == stage]
        pool.sort(key=lambda pair: (priority(settings["sampling"]["master_seed"], "source-allocation", [domain, stage, pair[0]]), pair[0]))
        count = budgets[domain]
        document_budget = stage in ("calibration", "probe") or (stage in ("pilot", "main") and domain != "code")
        if document_budget:
            flat = [doc for _, members in pool for doc in members]
            flat.sort(key=lambda d: (priority(settings["sampling"]["master_seed"],
                                              "source-documents", [domain, stage, d["document_id"]]), d["document_id"]))
            if count > len(flat) or count < 1:
                raise ValueError(f"insufficient {domain} {stage} documents: {len(flat)}, requested {count}")
            selected = {d["document_id"] for d in flat[:count]}
            selected_pool = [(g, [d for d in members if d["document_id"] in selected]) for g, members in pool]
            selected_pool = [(g, members) for g, members in selected_pool if members]
        else:
            if count > len(pool) or count < 1:
                raise ValueError(f"insufficient {domain} {stage} groups: {len(pool)}, requested {count}")
            selected_pool = pool[:count]
        for rank, (group, members) in enumerate(selected_pool):
            for d in members:
                d["sampling"] = {"group_rank": rank, "partition_groups": len(pool), "sampled_groups": len(selected_pool),
                                  "requested_units": count,
                                  "inclusion_probability": count / (len(flat) if document_budget else len(pool)),
                                  "inclusion_unit": "document" if document_budget else "source_group", "seed": seed,
                                  "stage": stage, "whole_group": not document_budget}
                chosen.append(d)
    if stage == "calibration":
        budget = 1
    elif stage == "probe":
        budget = settings["sampling"]["probe_positions_per_document"]
    elif stage == "pilot":
        budget = settings["sampling"]["pilot_positions_per_document"]
    elif stage == "technical":
        require(settings, "mechanics.technical_sample_rule")
        budget = settings["mechanics"]["technical_sample_rule"]["positions"]
    else:
        if positions != settings["sampling"]["main_positions_per_document"]:
            raise ValueError("main position budget must be frozen from planning")
        budget = positions
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    events = []
    for doc in chosen:
        qs = sample_positions(doc["token_ids"], doc["document_id"], window, budget, seed,
                              "calibration-windows" if stage == "calibration" else f"{stage}-positions",
                              settings["context"]["excluded_predictor_target_ids"])
        if stage == "technical":
            eligible = eligible_positions(doc["token_ids"], window, settings["context"]["excluded_predictor_target_ids"])
            qs = list(dict.fromkeys(eligible[i] for i in (0, len(eligible)-1, (len(eligible)-1)//3, 2*(len(eligible)-1)//3)))
        doc["sampled_predictor_positions"] = qs
        if not qs:
            raise ValueError("included document has no eligible positions")
        if stage == "probe" and len(qs) != budget:
            raise ValueError("fixed probe documents must each support four eligible positions")
        for order, q in enumerate(qs):
            w = window_for(doc["token_ids"], q, window)
            identity = {"document_id": doc["document_id"], "token_ids_sha256": doc["token_ids_sha256"],
                        "window_start": w["window_start"], "window_end": w["window_end"],
                        "target_id": w["target_id"], "context_length": window}
            events.append(dict(w, input_id=digest(identity), document_id=doc["document_id"],
                dataset=doc["dataset"], domain=doc["domain"], revision=doc["revision"],
                source_group=doc["source_group"], sampling=doc["sampling"], sample_order=order,
                eligible_positions=doc["eligible_positions"], token_ids_sha256=doc["token_ids_sha256"],
                layers=stage_layers(settings, stage, order)))
    write_rows(output / "documents.jsonl", chosen)
    write_rows(output / "inputs.jsonl", events)
    atomic(output / "research.json", canonical(settings) + b"\n")
    atomic(output / "source-review.json", Path(review_path).read_bytes())
    atomic(output / "model-catalog.json", (Path(prepared) / "model-catalog.json").read_bytes())
    return seal(output, "experiment-plan", {"stage": stage, "window": window, "seed": seed,
                "source_review_sha256": file_digest(review_path), "units": budgets,
                "calibration_sample": calibration_sample, "prepared_manifest_sha256": file_digest(Path(prepared) / "manifest.json"),
                "tokenizer_sha256": metadata["metadata"]["tokenizer_sha256"], "input_count": len(events),
                "research_scope": scope_identity(settings),
                "layers": sorted({l for i in events for l in i["layers"]})})

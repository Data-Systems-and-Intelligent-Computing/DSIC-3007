"""Prepare a bounded technical screen locally; preparation launches no compute."""

from pathlib import Path

from .artifacts import (atomic, canonical, digest, file_digest, priority, read_json,
                        read_rows, seal, verify_bundle, write_rows)


def prepare_benchmark(primary, shortened, output, *, groups_per_domain=3):
    """Select whole technical groups using their original source ordering."""
    parents = [Path(primary), Path(shortened)]
    manifests = [verify_bundle(p, "experiment-plan") for p in parents]
    settings = [read_json(p / "research.json") for p in parents]
    if settings[0] != settings[1]:
        raise ValueError("benchmark parents have different accepted settings")
    for m, window in zip(manifests, (4096, 2048)):
        if m["metadata"]["stage"] != "technical" or m["metadata"]["window"] != window:
            raise ValueError("benchmark requires the accepted technical context plans")
    if manifests[1]["metadata"].get("parent_plan_sha256") != file_digest(parents[0] / "manifest.json"):
        raise ValueError("short technical plan must link to the exact primary plan")
    if type(groups_per_domain) is not int or groups_per_domain < 1:
        raise ValueError("technical screen group count must be positive")
    docs = [list(read_rows(p / "documents.jsonl")) for p in parents]
    if {d["document_id"] for d in docs[0]} != {d["document_id"] for d in docs[1]}:
        raise ValueError("technical contexts have different source documents")
    selected = {}
    seed = settings[0]["sampling"]["master_seed"]
    for domain in settings[0]["domains"]:
        groups = {d["source_group"] for d in docs[0] if d["domain"] == domain}
        if len(groups) < groups_per_domain:
            raise ValueError(f"insufficient technical groups: {domain}")
        selected[domain] = sorted(groups, key=lambda g: (
            priority(seed, "source-allocation", [domain, "technical", g]), g))[:groups_per_domain]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    plans, input_counts, domain_counts = [], {}, {}
    primary_inputs = None
    for parent, manifest, documents in zip(parents, manifests, docs):
        window = manifest["metadata"]["window"]
        chosen = [d for d in documents if d["source_group"] in selected[d["domain"]]]
        ids = {d["document_id"] for d in chosen}
        inputs = [i for i in read_rows(parent / "inputs.jsonl") if i["document_id"] in ids]
        if not inputs or any(set(i["layers"]) != {0, 7, 15} for i in inputs):
            raise ValueError("screen must cover all three accepted technical layers")
        if window == 4096:
            primary_inputs = {i["input_id"]: i for i in inputs}
        else:
            for i in inputs:
                original = primary_inputs.get(i.get("parent_input_id"))
                if original is None or any(i[k] != original[k] for k in (
                        "document_id", "source_group", "predictor_position", "target_position", "target_id")):
                    raise ValueError("screen context alignment changed")
                if i["input_ids"] != original["input_ids"][-2048:]:
                    raise ValueError("short screen contains different input bytes")
        target = out / str(window)
        target.mkdir()
        write_rows(target / "documents.jsonl", chosen)
        write_rows(target / "inputs.jsonl", inputs)
        for name in ("research.json", "source-review.json", "model-catalog.json"):
            atomic(target / name, (parent / name).read_bytes())
        metadata = dict(manifest["metadata"], input_count=len(inputs),
                        benchmark_screen=True, selected_source_groups=selected,
                        units={domain: groups_per_domain for domain in selected},
                        parent_technical_units=manifest["metadata"]["units"],
                        screen_parent_sha256=file_digest(parent / "manifest.json"))
        seal(target, "experiment-plan", metadata)
        plans.append({"context": window, "path": str(target),
                      "manifest_sha256": file_digest(target / "manifest.json")})
        input_counts[str(window)] = len(inputs)
        domain_counts[str(window)] = {domain: {
            "documents": sum(d["domain"] == domain for d in chosen),
            "groups": len(selected[domain]),
            "windows": sum(i["domain"] == domain for i in inputs)} for domain in selected}
    windows = sum(input_counts.values())
    proposal = {
        "status": "prepared_local_not_authorized", "plans": plans,
        "settings_sha256": digest(settings[0]), "domain_counts": domain_counts,
        "windows": windows, "events": 3 * windows, "individual_ablations": 24 * windows,
        "native_reference_forwards": 26 * windows,
        "independent_validator_full_forwards": 27 * windows,
        "optimized_full_baseline_cleanup_forwards": 3 * windows,
        "optimized_suffix_replays": 24 * windows,
        "maximum_aggregate_gpu_seconds": 1800,
        "cash_cap_rule": "min(remaining_benchmark_allowance, reconciled_remaining_cash)",
        "paid_deadline_rule": "includes staging/loading/failure; smaller of 1800 seconds and cash cap divided by total hourly burn",
        "stop_conditions": ["numerical discrepancy", "nonfinite result", "identity mismatch",
                            "fit failure", "cash/time cap", "another verification still running"],
        "acceptance_scope": "screen only; full technical validation and complete workflow forecast still required",
        "requires": ["explicit paid authorization", "reconciled current balance and time boundary",
                     "live placement and exact quote", "candidate implementation/runtime identity",
                     "accounted input/model staging", "full unchanged tolerance checks"],
    }
    atomic(out / "proposal.json", canonical(proposal) + b"\n")
    return proposal

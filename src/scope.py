"""Versioned research scope; legacy artifacts never implicitly authorize Path B."""

from .artifacts import digest

PATH_B = "path-b-v1"
DOMAINS = ["c4", "scientific", "code"]
STAGE_DOMAINS = {
    "technical": DOMAINS, "calibration": DOMAINS, "probe": DOMAINS,
    "pilot": ["scientific", "code"], "main": ["scientific", "code"],
}


def is_path_b(settings):
    protocol = settings.get("protocol")
    if protocol is None:
        return False
    if protocol != {"id": PATH_B, "stage_domains": STAGE_DOMAINS, "layers": [15]}:
        raise ValueError("unknown or modified research scope policy")
    sample = settings["sampling"]
    checks = (
        settings["domains"] == DOMAINS,
        settings["model"] == {"repo_id": "allenai/OLMoE-1B-7B-0924",
            "revision": "c6f6954f182ffdd9f6ab4f4d772714b7dcd6836b", "dtype": "float32",
            "layers": 16, "experts": 64, "active_experts": 8},
        settings["context"]["primary_window"] == 4096,
        settings["context"]["sensitivity_window"] is None,
        settings["context"]["minimum_preceding_tokens"] == 4095,
        settings["context"]["window_policy"] == "fixed_input_tokens_target_outside",
        sample["pilot_positions_per_document"] == sample["main_positions_per_document"] == 8,
        sample["main_secondary_layers"] == [] and sample["main_secondary_positions"] == 0,
        sample["main_c4_documents"] == 0 and sample["primary_layer"] == 15,
        sample["master_seed"] == 3007 and sample["calibration_position_streams"] == [3008,3009,3010],
        sample["calibration_document_ladder"] == [100,250,500],
        sample["pilot_document_ladder"] == [20,40,80],
        sample["main_min_scientific_documents"] == sample["main_min_code_groups"] == 200,
        sample["technical_units_per_domain"] == 12,
        sample["probe_units_per_domain"] == 32 and sample["probe_positions_per_document"] == 4,
        settings["mechanics"]["technical_sample_rule"] == {"positions": 4,
            "rule": "first_last_two_interior_thirds; all_available_if_fewer",
            "layers": [15], "experts": "all_eight"},
        settings["mechanics"]["acceptance_tolerances"] == {"atol": 1e-6, "rtol": 1e-5},
        settings["mechanics"]["nll_atol"] == 1e-6,
        settings["mechanics"]["exact_checks"] == ["routing", "survivors", "cleanup"],
        settings["planning"]["target_ci_half_width_nats"] == .005,
        settings["planning"]["planning_replicates"] == 2000,
        settings["analysis"]["initial_replicates"] == 10000,
        settings["analysis"]["minimum_valid_replicate_fraction"] == .95,
        settings["analysis"]["method"] == "paired_bootstrap_t",
        settings["analysis"]["primary"] == "0.5*scientific_document_mean_delta+0.5*code_document_mean_delta",
        settings["analysis"]["holm_domains"] == ["scientific", "code"],
        settings["calibration"]["normalized_l1_max"] == .05,
        settings["calibration"]["extreme_set_jaccard_min"] == .75,
        settings["calibration"]["comparison_pass_fraction_min"] == .90,
    )
    if not all(checks):
        raise ValueError("Path B settings differ from the accepted scope or numerical contract")
    return True


def stage_domains(settings, stage):
    if is_path_b(settings):
        if stage not in STAGE_DOMAINS:
            raise ValueError("stage deferred in Path B")
        return list(STAGE_DOMAINS[stage])
    return list(settings["domains"])


def stage_layers(settings, stage, order=0):
    if is_path_b(settings):
        stage_domains(settings, stage)
        return [15]
    if stage in ("technical", "calibration", "probe"):
        return [0, 7, 15]
    return [15, 0, 7] if stage == "main" and order < 16 else [15]


def scope_identity(settings):
    return {"policy": PATH_B if is_path_b(settings) else "legacy-v1",
            "settings_sha256": digest(settings)}


def require_scope(settings, metadata):
    if is_path_b(settings) and metadata.get("research_scope") != scope_identity(settings):
        raise ValueError("Path B requires a matching versioned research scope receipt/plan")


def validate_plan(settings, metadata, inputs):
    """Check the sealed Path B event universe before any model load or analysis."""
    if not is_path_b(settings):
        return
    require_scope(settings, metadata)
    stage = metadata["stage"]
    domains = stage_domains(settings, stage)
    if metadata.get("window") != 4096 or set(metadata.get("units", {})) != set(domains):
        raise ValueError("Path B plan context or domain budgets differ")
    rows = list(inputs)
    if not rows or {r["domain"] for r in rows} != set(domains):
        raise ValueError("Path B plan has missing or extra domains")
    if len({r["input_id"] for r in rows}) != len(rows):
        raise ValueError("Path B plan repeats physical inputs")
    by_doc = {}
    for row in rows:
        if (row.get("layers") != [15] or row.get("context_length") != 4096 or
                len(row.get("input_ids", [])) != 4096 or
                row.get("local_predictor_position") != 4095 or
                row.get("target_position") != row.get("predictor_position", -2) + 1 or
                row.get("window_end") != row.get("predictor_position", -2) + 1 or
                row.get("window_end", 0) - row.get("window_start", 0) != 4096):
            raise ValueError("Path B input layer, full-context alignment or external target differs")
        by_doc.setdefault(row["document_id"], []).append(row)
    if not metadata.get("preflight_diagnostic"):
        for doc_rows in by_doc.values():
            first = doc_rows[0]
            expected = (1 if stage == "calibration" else 4 if stage == "probe"
                        else min(8, first["eligible_positions"]) if stage in ("pilot", "main") else None)
            if expected is not None and len({r["predictor_position"] for r in doc_rows}) != expected:
                raise ValueError("Path B plan has incomplete frozen position coverage")


def main_budgets(settings, selected):
    result = {"scientific": selected["scientific_documents"],
              "code": selected["code_source_groups"]}
    if not is_path_b(settings):
        result["c4"] = settings["sampling"]["main_c4_documents"]
    return result

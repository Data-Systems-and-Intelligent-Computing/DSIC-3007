"""Explicit whole-remaining-workflow admission for the selected Path B allocation."""

import math

from .artifacts import digest
from .scope import scope_identity

REMAINING_STAGES = {
    "native_acceptance", "independent_validation", "technical_collection",
    "calibration", "probe", "pilot", "setup_load", "transfer_write",
    "storage", "final_cpu_analysis",
}


def admit_remaining(settings, main_seconds, identities, ledger):
    """A conditional forecast, never evidence of achieved precision or paid authority.

    Completed stages need explicit zero remaining cost and receipt evidence.
    Historical invoices/time are reconciled in the supplied remaining limits.
    Missing allowances fail closed rather than being silently set to zero.
    """
    if ledger is None:
        return {"status": "unknown", "feasibility_pass": False,
                "blockers": ["whole_remaining_workflow_ledger_and_reconciled_limits_required"]}
    def number(x):
        return type(x) in (int, float) and math.isfinite(x) and x >= 0
    blockers = []
    if ledger.get("research_scope") != scope_identity(settings) or ledger.get("identities") != identities:
        blockers.append("remaining_ledger_scope_or_pilot_technical_calibration_identity_mismatch")
    stages = ledger.get("remaining_stages", {})
    if set(stages) != REMAINING_STAGES:
        blockers.append("remaining_ledger_requires_every_stage_and_no_unaccounted_stage")
    gpu, wall, fixed = main_seconds, main_seconds, 0.0
    for name in REMAINING_STAGES:
        row = stages.get(name, {})
        if (row.get("evidence_status") not in ("measured", "bounded") or not row.get("evidence") or
                any(not number(row.get(k)) for k in ("gpu_seconds", "wall_seconds", "usd"))):
            blockers.append("unknown_remaining_stage:" + name)
            continue
        gpu += row["gpu_seconds"]
        wall += max(row["gpu_seconds"], row["wall_seconds"])
        fixed += row["usd"]
    account = ledger.get("accounting", {})
    if (not account.get("cash_boundary_confirmed") or not account.get("time_boundary_confirmed") or
            account.get("price_evidence") != "live_quote" or not account.get("quote_reference") or
            any(not number(account.get(k)) for k in
                ("hourly_rate_usd", "remaining_budget_usd", "remaining_wall_seconds")) or
            not account.get("hourly_rate_usd")):
        blockers.append("unknown_or_unreconciled_remaining_limits_and_live_quote")
    reserve = ledger.get("reserve_fraction")
    if not number(reserve) or not .20 <= reserve < 1:
        blockers.append("remaining_ledger_requires_at_least_20_percent_reserve")
    if blockers:
        return {"status": "unknown", "feasibility_pass": False, "blockers": blockers,
                "ledger_sha256": digest(ledger)}
    cost = gpu / 3600 * account["hourly_rate_usd"] + fixed
    if cost * (1 + reserve) > account["remaining_budget_usd"]:
        blockers.append("remaining_cash_exceeded_with_reserve")
    if wall * (1 + reserve) > account["remaining_wall_seconds"]:
        blockers.append("remaining_wall_time_exceeded_with_reserve")
    return {"status": "projected_feasible" if not blockers else "infeasible",
            "feasibility_pass": not blockers, "blockers": blockers,
            "ledger_sha256": digest(ledger), "total_remaining_gpu_seconds": gpu,
            "total_remaining_wall_seconds": wall, "total_remaining_usd": cost,
            "reserve_fraction": reserve, "accounting": account,
            "evidence_scope": "conditional_remaining_workflow_projection_not_paid_authorization"}

# Prompt for a fresh optimization findings session

Work in `/home/arya/School/TA/DSIC-3007` as an independent performance investigator.
Your task is to produce actionable findings for another coordinator to implement
later. Investigate now; do not implement optimizations or launch GPU work.

The user explicitly retains **4096-token primary context**. Keep the accepted
science, FP32/eager execution, TF32 off, batch one, no KV cache, all eight expert
effects and frozen numerical tolerances. 2048 is the existing matched sensitivity;
1024 or changing the primary context is outside this task.

Read README.md, configs/research.json and WORKFLOW.md first. Respect AGENTS.md.
Establish your own conclusions from current source and raw logs before consulting
prior optimization hypotheses. Then reconcile your findings with
docs/validation/EXECUTION-BUDGET-PLAN.md, OPTIMIZATION-TODO.md and
optimization-next-direction-2026-10-11.json in that directory. Treat current source
as authoritative over stale graph snippets or historical documentation.

Inspect this exact completed attempt; do not rely on a moving `latest` symlink:

`runpod/runs/dsic-screen-20261011T010215-fb295d6f/`

It contains console.log, result.json, cleanup.json and results/{runner-receipt.json,
validation.json, validation.json.cases.jsonl, validation.log, gpu-samples.jsonl,
weights.log}. The paired sealed plans are under
`data/plans/optimization-screen-2026-10-11-v2/{4096,2048}`. Supporting audits:
`docs/validation/optimization-l40s-{partial-results,partial-review,billing-readback}-2026-10-11.json`.

Known facts to verify: the L40S diagnostic hit its cap after 17/72 inputs and 408/1728
comparisons; recorded errors were zero on completed cases. Only 4096 C4/scientific
finished; code and 2048 are unmeasured. Weights are verified and retained on network
volume; both attempt pods were confirmed absent. Runtime image_digest is null in
the old receipt. A future-run helper fix exists, but cannot repair or relabel old
evidence. Local tests last passed 142 cases; this is not GPU acceptance.

Find the actual cost of scientific collecting, distinguishing it from the extra
native-versus-optimized work performed by the diagnostic validator. Trace callers
and consumers of src/run_experiment.py, interventions.py, model.py,
optimized_validation.py, observation_contract.py, reuse.py and execution_budget.py.
Account for every full forward, suffix replay, check, synchronization, copy and
durable write. Verify per-input versus per-layer timing and avoid double-counting
shared work. Separate download, hashing/loading, measurement and teardown.

Rank a small number of optimization candidates by expected impact and risk.
For each give: exact file/function locations; source/log evidence; operation-count
or runtime rationale; proposed algorithm/change; affected consumers and identity
contracts; memory consequences; numerical risks; and concrete local differential,
failure/cleanup and native GPU acceptance checks. Explicitly state what is
measured, inferred, assumed or unknown. Do not promise speedups from complexity
alone. Causal prefix invariance alone does not prove FP32 bitwise equivalence;
native expert batch shapes and accumulation order matter.

Do not treat previous ~50/~60-hour illustrations as measured production forecasts.
Recompute the source accounting and state missing profiles. Assess the remaining
$12/ten-hour envelope without inventing a current wallet balance or accounting
start. The weight cache saves future downloads; past billed/elapsed work remains
spent. If feasibility cannot be established, say UNKNOWN and explain why.

Write only:

- `docs/validation/FRESH-OPTIMIZATION-FINDINGS-2026-10-11.md`
- `docs/validation/fresh-optimization-findings-2026-10-11.json`

Preserve research settings, implementation source, frozen packages, raw receipts,
logs and any running verification. Do not access secrets, change cloud resources,
download weights again, or run `screen.py --run`. Keep local commands under two
minutes; provide commands for the user if longer work is needed. Use existing
diagnostics/checks where possible; do not build new infrastructure.

Finish with the top 1–3 candidates, the first concrete implementation slice for the
coordinator, and one proposed baseline-versus-candidate GPU benchmark capped at
20 minutes/$0.40 including all phases, further limited by reconciled remaining
cash/time, with durable progress/results and no automatic retry. This is a proposal,
not authorization. Report remaining scientific/numerical acceptance gates.

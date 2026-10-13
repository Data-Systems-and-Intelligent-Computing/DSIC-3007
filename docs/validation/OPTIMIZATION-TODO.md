# Optimization TODO and agent ownership

Updated 2026-10-11. Read [the execution plan](EXECUTION-BUDGET-PLAN.md),
[README](../../README.md), [research.json](../../configs/research.json), then
[WORKFLOW](../../WORKFLOW.md). README/config define accepted science. This board
tracks proposed implementation work; no paid run or research cut is authorized.

## Coordination rules

- **Coordinator:** owns scope, assignments, review/integration, evidence quality,
  budget accounting and the user decision. Keep one production executor identity.
- **Measurement owner (`/root/measurement`):** intervention capture, layer-15 and suffix
  execution, model lifetime, serialization and collector tests.
- **Planning owner (`/root/planning`):** exact workload counts, total cost/time model,
  nested-observation provenance and analysis coverage.
- **Validation owner (`/root/validation`):** independent native reference comparison,
  numerical/recovery checks and bounded benchmark evidence.
- Assign exact files and dependencies before dispatch. One writer per shared
  file; measurement/planning agree on the timing/cache schema before parallel
  edits. No agents independently edit accepted research settings or operate pods.
- Every completed task reports changed files, commands/results, artifact/hash
  links, remaining limits and a concise handoff. Distinguish implementation done,
  CPU checks passed, native fidelity passed and budget feasibility demonstrated.
- Coordinator integrates and reviews each dependency before the next phase.
  Receipt-affecting code must be frozen before scientific collection. Avoid
  unrelated refactors, new infrastructure frameworks and duplicate documents.

## Phase 0: establish the inputs and constraints

- [x] **O00 — Documentation/handoff (coordinator).** Record findings, assumptions,
  budget arithmetic, proposed amendments and acceptance gates; link from WORKFLOW.
  Keep README/config unchanged. Evidence: EXECUTION-BUDGET-PLAN.md and this board.
- [x] **O01 — Reconcile accounting and existing verification (coordinator).**
  Depends O00. Read available status/receipts and actual remaining balance/rate
  without changing the running process. Establish the dollar/time accounting
  boundary and deduct ongoing verification, storage and other accruing costs.
  Done: explicit remaining dollars/GPU-hours or named unknowns; no fabricated
  verification PASS and no assumption that the full $12 is unspent.
- [x] **O02 — Reproduce workload census (planning).** Depends O00. Verify current
  config/prepared/review identities; count actual eligible positions and whole
  code groups using existing seeded ordering. Include pilot ladders, both probes,
  calibration escalation, technical inputs and actual whole-group sensitivity.
  Done: reproduce the 605-document minimum candidate and 151-document sensitivity
  example when hashes match; report changed counts otherwise. No main freeze.

## Phase 1: define complete measurement and budget contracts

- [x] **O03 — Complete-workflow planner (planning).** Depends O01/O02. Preserve
  pilot precision/allocation rules; add C4 and every other stage, cache misses,
  setup/load, full validation, write/transfer time, host waits, storage costs and
  an explicit reserve. Measure unique input work once, avoiding repeated shared
  timings across layer events. Report main-only and total cost/time separately.
  Done: regression cases for short documents, multi-file groups, escalation,
  duplicated timing fields, insufficient pilot and infeasible budget; no forced
  allocation when gates fail. Unknown timings/prices cannot yield feasibility PASS.
- [x] **O04 — Raw-result and reuse contract (planning + measurement review).**
  Depends O02. Define exact executor/input/operator cache keys, immutable source
  measurement links and per-plan sampling metadata; retain all raw routes,
  coefficients, eight NLLs/effects and M0/M4 inputs. Review collector/analysis,
  calibration-freeze, transfer validators, recovery and identity consumers.
  Done: regression cases reject mismatched bytes, runtime, context, target or
  source identity; cached observations reconstruct each budget correctly.

## Phase 2: implement native-compatible reuse locally

- [x] **O05 — Capture native expert contributions (measurement).** Depends O04.
  Preserve GEMM shapes/row order and unweighted outputs; implement zero-coefficient
  multiplication plus original scatter accumulation. Never subtract from sums
  or replace GEMM with GEMV. Done: native/no-op and eight individual zero masks
  agree in CPU fixtures; untargeted rows and surviving coefficients stay exact.
- [x] **O06 — Layer-15 fast path (measurement).** Depends O05. Reuse the decoder
  prefix and native expert results; replay residual/norm/head at original shapes.
  Keep full cleanup initially. Done: same complete event schema, target alignment,
  all eight effects, exact cleanup, exception isolation and no baseline mutation.
  CPU success is not pretrained numerical approval.
- [x] **O07 — Layers 0/7 suffix replay (measurement).** Depends O06. Resume from
  the intervened layer output and execute remaining layers at native full shapes,
  with matching masks/positions and natural downstream routing. Done: independent
  baseline branches, all-eight coverage and differential CPU cases covering
  downstream route changes; comparison against full forwards prepared for GPU.
- [x] **O08 — Reuse nested observations (planning; collector integration by its owner).**
  Depends O04/O06/O07. Execute only added windows at calibration 100/250/500 and
  pilot 20/40/80; preserve separate budget tables, seed identities and source
  measurement links. Done: cached and uncached fixtures give the same freeze/
  analysis inputs, budget-specific metadata and resume behavior; timings retain
  actual cache-hit/miss costs. Never relabel an old journal as a new plan.
- [x] **O09 — Model lifetime, memory and I/O (measurement).** Depends O04/O07.
  Reuse one verified loaded model across stages; retain identity checks; bound
  cache and raw-row memory; optimize lossless routing-byte conversion and unused
  tensor retention. Profile synchronization/serialization/fsync before changes.
  Done: byte-equivalent payloads, complete-event durability, bounded resume memory
  and no lost/duplicated events after injected write/capture failures. Retain all
  scientific checks; do not obtain speed by omitting cleanup or raw measurements.
- [x] **O10 — Optional downstream expert reuse (measurement).** Depends O07 and
  profiling evidence of material benefit. Cache only fully unchanged expert
  dependencies; recompute changed native batch shapes and preserve accumulation.
  Done: memory/cost benefit and native equivalence established, or document rejection
  and use suffix replay. This task must not become open-ended kernel development.

## Phase 3: validate and decide before buying collection time

- [x] **O11 — Extend independent numerical validation (validation).** Depends
  O05/O06/O07; include O08/O09/O10 if integrated. Compare native full-forward and
  accelerated baseline/every ablation, both contexts, all required layers/experts.
  Check exact IDs/weights/survivors/cleanup and frozen mixture/logit/NLL tolerances.
  Add meaningful regressions for batch-shape rounding, coefficient-zero vs sum
  subtraction, context alignment, exceptions and executor mismatch. Done: local
  checks pass; per-case GPU discrepancy report and receipt path are implemented.
- [ ] **O12 — Prepare one bounded GPU benchmark (coordinator + validation).**
  Depends O01/O03/O11. Refresh live GPU stock, actual quote, volume placement and
  image/driver compatibility. Prefer L40S if fit and complete-cost evidence support
  it; do not assume another GPU's speed. Freeze technical subset, exact reference
  counts, implementation identity, 30-GPU-minute-or-smaller-cash cap and outputs.
  Done: concrete runnable proposal with charge estimate and stop conditions;
  obtain explicit paid authorization. Do not interfere with current verification.
- [ ] **O13 — Run authorized screen and full acceptance (validation).** Depends
  O12 + paid authorization. Measure native and optimized end-to-end timings,
  domain/context variation, memory/RSS, raw persistence and numerical discrepancies.
  Charge loading/failures to the cap; no automatic retry. Full technical validation
  on all accepted inputs needs its own accounted time within the authorized scope.
  Done: passing candidate-specific receipt AND measured full-workflow forecast,
  or clear numerical failure / incomplete screen / budget failure.
- [ ] **O14 — Final executor and scientific decision (coordinator).** Depends
  O13. Select one validated production GPU/code identity; review projected cost
  and time including escalation/reserve. If full design fails, present exact
  alternatives from the plan with lost measurements/precision and cost evidence.
  Done: explicit user-approved amendment if any, or unchanged accepted design;
  no README/config amendment inferred from the user's documentation request.

## Phase 4: collect and analyze only within accepted gates

- [ ] **O15 — Calibration, probes and pilot (coordinator/measurement).** Depends
  O14 + authorization for scientific execution. Collect both context calibrations
  and fixed probes, apply unchanged stability rules, collect/extend pilot as
  required. Analyze on local CPU to avoid paid GPU idle time where practical.
  Done: raw measurements, frozen valid rankings and sufficient pilot; instability
  or final-pilot insufficiency is reported rather than silently overridden.
- [ ] **O16 — Freeze exact main and sensitivity budget (planning/coordinator).**
  Depends O15. Apply 2,000-draw planning and accepted minima/precision rule;
  derive actual documents, groups, positions and whole-group sensitivity subset.
  Reconcile spending/time so far and remaining forecast with reserve. Done: exact
  qualifying freeze within both limits, or no-feasible-allocation; no outcome-based
  extension or underfilled group disguised as a complete sample.
- [ ] **O17 — Complete collection, CPU analysis and handoff (coordinator).**
  Depends O16 + authorization. Collect all planned individual effects including
  C4 and matched sensitivity; preserve/recover incomplete work without zero-filling.
  Produce M0/M4, document/domain summaries, paired bootstrap-t and Holm outputs.
  Done: exact event coverage, >=95% valid bootstrap draws where required, identities,
  raw bytes, timing/cost reconciliation and uncertainty limitations all reported.
  Label incomplete research honestly if scientific or budget gates prevent completion.

## Next agent entry point

Start at O01/O02, then agree the O03/O04 contracts. The coordinator assigns one
owner before edits; O03 planning can proceed alongside measurement implementation
after the shared schema is fixed. Validation review can proceed independently,
but collector/verification shared-file edits must be coordinated. Workers are now
finished for this local implementation slice. Start next with the O12 accounting,
placement and authorization blockers, then O13. Record status and evidence on this
board; the coordinator remains responsible for the integration decision.


## Integrated status and evidence, 2026-10-11

Checkboxes O03–O11 mean **local implementation/CPU criteria completed**, including
O10's explicit rejection path. They do not mean native GPU acceptance, scientific
sufficiency or demonstrated budget feasibility. O12 remains open because staging,
exact all-in quote and paid authorization are missing. O13–O17 have not run.

| Task | Current evidence and limitation |
|---|---|
| O01 | Read-only billing/pod/volume/catalog snapshots saved. Historical day spend is $0.7406598956; actual remaining credit and ten-hour clock start are named unknowns. Known pod EXITED does not establish the separate running verification's status. |
| O02 | Seeded census reproduced: 605 main-candidate documents, 590 groups, P=38,296/Q=9,677; matched example 151 documents, 147 groups, P=9,553/Q=2,415. No scientific main freeze. |
| O03 | C4 cost and complete-workflow ledger/forecast integrated; domain/context maxima, exact ladders/groups, persistence/host waits/storage and reserve; regressions fail closed. Current saved budget verdict UNKNOWN. |
| O04 | Exact raw-observation contract and sealed provenance links implemented; altered bytes/source/context/runtime/executor are rejected. |
| O05/O06/O07 | Native-shape batched contribution capture, final-layer replay and natural decoder suffix at 0/7; all eight effects and exact cleanup pass CPU fixtures. Partial L40S agreement is now observed for 408 comparisons; full native agreement remains pending. |
| O08 | End-to-end nested collection reuses only exact source rows; fresh sampling metadata, immutable raw links and actual hit/miss timing persist. Existing downstream calibration/analysis consume the preserved raw schemas. |
| O09 | Verified single model session, lossless routing bytes, streaming token windows, lazy compact journal/reuse byte-offset indexes and durable timing sidecar integrated. Write failure/resume, duplicate and torn-tail regressions pass. ID/offset metadata scales with event count; native memory, serialization and fsync profiles remain unmeasured. |
| O10 | Rejected for this candidate: no measured material benefit for further downstream expert caching; natural suffix replay retained. |
| O11 | Independent pristine-native baseline and full native all-eight ablation comparisons, discrepancy/timing reports and exact full-technical receipt gate implemented; CPU tests and full-plan preflight passed. An incomplete diagnostic GPU receipt exists; no full technical acceptance receipt. |
| O12 | User-run registry-v2 L40S diagnostic ended INCOMPLETE at the time cap: 17/72 inputs, 408/1,728 all-zero-error comparisons; only 4096 C4/scientific covered. Weights verified and retained; both attempt pods confirmed absent by fresh GET. Partial memory/timing observations are not full acceptance. Future-run image propagation fixed locally; old receipt retains its null image digest. Billing/remaining ten-hour boundary unresolved; complete forecast UNKNOWN with an adverse ~50h illustrative main-4096 signal. See [partial results](optimization-l40s-partial-results-2026-10-11.json), [independent review](optimization-l40s-partial-review-2026-10-11.json) and [billing read-back](optimization-l40s-billing-readback-2026-10-11.json). No paid retry or scientific collection launched. |
| O13–O17 | Pending native acceptance and measured fit/cost/throughput, calibration stability, sufficient pilot, accepted allocation and paid authorization. |

Assigned file boundaries: measurement wrote `interventions.py`, `run_experiment.py`,
`reuse.py` and their tests; planning wrote `planning.py`, `execution_budget.py`,
`observation_contract.py`, census and contract tests; independent validation wrote
`optimized_validation.py`, its tests and review JSON. The coordinator subsequently
integrated owned files, model session, streaming compact journal, CLI, forecast,
regressions, benchmark preparation and evidence. The validation worker hit its
session usage limit after implementation and focused checks; the coordinator
completed the final source review and integration checks. No workers are left running.

Final commands/results and exact file/artifact hashes:
[optimization-integration-2026-10-11.json](optimization-integration-2026-10-11.json).
Independent worker review:
[optimization-independent-review-2026-10-11.json](optimization-independent-review-2026-10-11.json).
The concrete screen proposal and unresolved accounting boundary are in
[EXECUTION-BUDGET-PLAN.md](EXECUTION-BUDGET-PLAN.md#concrete-bounded-gpu-screen-proposal--not-authorized).


## Next optimization investigation — source audit after partial L40S screen

- [x] **O18 — Actual collector forward accounting and reference consolidation (local).**
  Mandatory uninstrumented exact cleanup is now the native reference: two full
  forwards per optimized input instead of three. All8 effects and native shapes
  remain. Exact NLL/logit cleanup runs before any completed event can persist.
  Native calibration/probe paths remain unchanged. CPU differential, failure and
  consumer-gate checks pass; no pretrained GPU speedup or full acceptance claimed.
- [ ] **O19 — Profile and prototype guarded downstream expert-batch reuse.**
  Start with layer8 after the layer7 intervention; reuse only when the complete
  expert input, row map, batch shape and parameter dependencies are unchanged.
  Preserve native eager shapes and recompute changed batches. Prefix causality
  alone does not establish FP32 agreement. No implementation or measured benefit.
- [x] **O20 — Execute paired collector benchmark (bounded diagnostic complete).**
  User-run v6 attempt043533 passed six inputs/144 native comparisons/12pairs,
  with48 measured events and12 warmup events; teardown confirmed. Raw identities
  and timing counts audited locally. Full-technical/scientific acceptance pending.
  Six fixed domain/context windows, two layer shapes, all8 effects, alternating
  AB/BA, separate warmups and 144 independent native comparisons. One attempt,
  <=20min/$0.40 including every phase, further reduced by reconciled cash/time.
  Cached weights only, durable visible progress and partial evidence, no retry.
  Root does not launch paid work; the user runs the long command after reconciliation.
  The diagnostic cannot replace full technical acceptance or a complete forecast.

169 local tests passed, including the extracted-file hashing cutoff regression;
all 55 runner checks are covered by the final suite. Actual tiny CPU OLMoE comparisons against the frozen predecessor and
independent native cases passed for both layer shapes. The 4k primary, research
settings, numerical gates, predecessor packages and GPU receipts are preserved.
At local freeze candidate GPU speedup and native agreement were unmeasured;
O20 now supplies the bounded diagnostic below. Full native acceptance and
complete-workflow feasibility remain unestablished. Current accounting does not establish the actual remaining wallet
or original ten-hour clock boundary. Source edits were confined by owner:
coordinator collector/controller/docs, progress worker adapter/tests, independent
reviewer measurement/collection regressions and review JSON.

Integration and exact identities:
[cleanup-reference integration](optimization-cleanup-reference-integration-2026-10-11.json).
Independent source review:
[review](optimization-cleanup-reference-review-2026-10-11.json).
Fresh findings remain historical and unchanged:
[finding report](FRESH-OPTIMIZATION-FINDINGS-2026-10-11.md).

Live read-only preflight at 2026-10-11 04:07:24 UTC: frozen package verified,
existing registry/volume identities accepted and no account pods observed, then
stopped because L40S stock was unavailable in retained-volume region US-MO-1.
No resource was created. Current quote, wallet and original ten-hour remaining
boundary remain unresolved; repeat the read-only preflight when stock returns.


### Paired attempt upload timeout and local successor

The user-run v5 attempt at04:19UTC stopped before benchmark/model verification:
81s boot left39s upload under the shared120s setup limit. Teardown was confirmed
at04:21:34UTC (131.17s total); old logs/receipts remain unchanged. No collector
measurement or native result was produced. Nominal compute+storage allowance
was about$0.04045; actual invoice/wallet remains unknown.

The v6 controller prints stage vs overall remaining time separately and allows
240s for shared startup/upload/preflight, still clipped by overall lease and90s
teardown reserve. Model/warmup/collector caps are90/60/240s; native240s, weights180s,
finalization30s,retrieval/teardown120s. Sum1200s, cash.40 unchanged. This is a
planning reallocation; upload throughput and complete benchmark fit are unmeasured.
No paid retry launched locally; prior attempt cost/time counts toward remaining
limits. Collector/science/settings/tolerances remain unchanged. See
[upload postmortem](optimization-upload-timeout-2026-10-11.json).

Final v6 checks:172 tests passed21.16s; py_compile, diff check and frozen package
source/archive verification passed. Fresh API preflight did not run because
automatic permission approval review timed out before execution; local teardown
receipt confirms deletion at run end, current provider state is not inferred.


### Successful v6 paired collector result — latest status

User attempt043533 finished in659.65s including cache verification, model loading,
warmup/native checks, measured old/new collection, retrieval and confirmed teardown.
All six windows/144 native comparisons/12pairs passed; native mixture/logit/NLL
errors0 and paired raw routes/NLL/all192 matched effect values exact.48 measured
events plus12 warmup events retained.4k primary/settings/tolerances unchanged.

Measured4k durable elapsed: layer15-only2.842→1.901s/input (33.10% less time);
all three layers13.518→12.605s/input (6.75% less).Layer0/7 effects consume about85%
of candidate three-layer measurement; O19 remains the next local profile/prototype.
Only one selected input/domain/context/shape; six windows are three matched source
units, no repeated-run confidence interval. AB/BA is confounded with context.

Nominal compute+storage allowance$0.20339 for this run, not invoice proof. Latest
two attempts total790.82 active seconds/$0.24384 nominal, but1629.82s continuous
wall span; idle/local work also counts for the ten-hour clock. Current wallet and
original clock boundary unknown. Conditional main4096 census projection becomes
48.999h from58.932h, still adverse under10h and not a complete forecast.

O20 checkbox means this small paired diagnostic completed, not O11 full-technical
approval or O13–O17 science. No further paid work authorized/launched by coordinator.
Attempt.json has stale creating status; final result/runner/cleanup corroborate
completion. Preserve raw evidence; address lifecycle labeling in a later successor.
[Measured results](OPTIMIZATION-PAIRED-RESULTS-2026-10-11.md),
[raw-derived JSON](optimization-paired-results-2026-10-11.json),
[independent review](optimization-paired-results-review-2026-10-11.json).

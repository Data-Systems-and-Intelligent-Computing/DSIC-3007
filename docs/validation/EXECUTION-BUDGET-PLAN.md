# Execution redesign: findings, reasoning and handoff

Updated 2026-10-11. Status: **local implementation integrated and CPU tested;
pretrained GPU fidelity and budget feasibility unknown**. Work authorized in the
current session: coordinated local implementation, checks and benchmark preparation.
Later user steering authorized one capped L40S diagnostic and assigned commands
expected to take more than two minutes to the user. See the latest handoff below;
the earlier authorization snapshots remain historical.
Research authority: [README](../../README.md) and
[accepted settings](../../configs/research.json). Execution task board:
[OPTIMIZATION-TODO.md](OPTIMIZATION-TODO.md).

## Decision and authorization record

Latest steering: user-reported balance $13.99; remaining-work ceiling $12 with
$1.99 held outside it. The user replaced the old network volume. REST v2 reads
confirm volume `k6u810gze3`, 50 GB STANDARD, US-MO-1, no existing pods and low
L40S stock at $1.09/h Secure. The reported 125 GB RAM/16-vCPU host offer is
user-supplied; allocated hardware remains unmeasured. A 64 GB minimum RAM filter
does not constrain the actual allocation to 64 GB. One user-invoked screen is
authorized; no other paid stage or research amendment is authorized. The
ten-hour accounting start and full-workflow feasibility remain unresolved.

- The user fixed a **$12 budget** after requesting a maximum **10-hour remaining
  GPU workflow**. Carry both constraints forward. Working interpretation: $12
  available for remaining work; reconcile actual balance, already accruing
  verification costs and the accounting start before any spending decision.
- Count remaining verification, benchmarking, loading, calibration escalation,
  pilot escalation, main INCLUDING C4, matched sensitivity, GPU idle time and
  finalization. Sum GPU-hours across devices; parallelism does not erase cost.
- Leave the currently running A40 technical verification running. Its progress
  and remaining duration were not read remotely in this analysis. Do not stop,
  restart, patch, deploy into or compete with that process.
- No new paid resource, benchmark or scientific run is authorized by this plan.
  No research amendment has been accepted. In particular, dropping layers 0/7
  remains a proposal. Keep README and research.json free of operational budgets;
  change research decisions only after explicit approval of a concrete amendment.
- Use this fresh clone only. Missing historical documents are intentional.
  Preserve existing dirty changes, sealed inputs, original results and receipts.

## Recommended order and why

Develop native-compatible layer-15 reuse locally, then full-shape suffix replay
for layers 0/7. Add exact observation reuse across nested budgets and remove
avoidable loading/serialization overhead. Preserve every accepted measurement.
Use a bounded native comparison to determine total feasibility before collecting
scientific outcomes. L40S is the first budget candidate; choose hardware from
measured complete-workflow cost, numerical acceptance and live placement evidence.

Neither $12 nor ten hours is currently proven sufficient. If implementation
improvements cannot fit the complete design, present the smallest explicit
amendment to the user. Do not select settings by the desired sign/significance.

## Findings before implementation

| Source | Finding and consequence |
|---|---|
| [run_experiment.py](../../src/run_experiment.py), `measure_input` | Baseline shared across layers, but every expert intervention reruns the entire model. Layer-15 windows cost 10 full forwards; three-layer windows cost 26, including cleanup. Calibration/probe also run baseline plus full cleanup. |
| [interventions.py](../../src/interventions.py), `RoutingCapture` | Ordinary capture saves the predictor's route/mixture, not reusable expert outputs. Reconstruction reruns native expert batches. Single-token GEMV previously failed tolerance. |
| Same file, `reconstruct(exclude)` | Returns only the predictor row; its internal exclusion skips an entire expert. Do not reuse its entire internal mixture as a token-local intervention: other rows must remain untouched. |
| [model.py](../../src/model.py) and pinned Transformers | Original FP32, deterministic eager attention/experts, TF32 off, batch one, no KV cache. The exact 5.12.1 expert loop uses batched linear operations, weighting and ordered `index_add_`. |
| [planning.py](../../src/planning.py), `plan_main` | Uses actual eligible positions and whole code groups, but runtime projection omits C4 collection and the other workflow stages. Mean component times are not a conservative end-to-end bound. |
| [verify.py](../../src/verify.py), `verify_native` | Checks native reconstruction, repeats, locality and cleanup. Does not yet compare an accelerated executor's individual ablated logits/NLLs against native full forwards. |
| [artifacts.py](../../src/artifacts.py), [run_experiment.py](../../src/run_experiment.py) | Complete-event journal with fsync; model/runtime/implementation identity gates. Inference timings omit some loading, serialization and persistence work. Optimize without weakening durability or provenance. |
| [preparation.py](../../src/preparation.py), [workflow.py](../../src/workflow.py) | Deterministic nested inputs, whole code groups, matched whole-group sensitivity. Different predictor windows cannot be treated as one document forward. |

The combined reviewed inventory supports 500 calibration documents/domain. Main
pools contain 6,540 scientific documents, 3,171 code files in 3,092 groups, and
895 C4 documents. This supersedes the earlier retained-only shortage for planning;
it does not demonstrate calibration stability or scientific sufficiency.

### Actual minimum-allocation workload

Applying the current seeded ordering at the main minima (not freezing main):

| Domain | Documents | Groups | P: layer-15 positions | Q: positions at each of layers 0/7 |
|---|---:|---:|---:|---:|
| Scientific | 200 | 200 | 12,755 | 3,200 |
| Code | 205 | 200 | 12,957 | 3,278 |
| C4 | 200 | 190 | 12,584 | 3,199 |
| Total | 605 | 590 | 38,296 | 9,677 |

For each included document, P adds `min(64, eligible_positions)` and Q adds
`min(16, eligible_positions)`. Current forwards = `10P + 16Q = 537,792`;
individual ablations = `8(P + 2Q) = 461,200`. Do not substitute 200 files for
200 repositories or select cheaper repositories outside the prescribed ordering.

The current sensitivity algorithm, applied to that candidate, selects 50
scientific, 51 code and 50 C4 documents (50/48/49 groups), with P=9,553 and
Q=2,415. Recalculate after the pilot chooses the actual main allocation.
Pilot 20/40/80 ladders contain 120/240/482 windows; the final code pilot has
81 files in 80 groups. Probe has 384 windows/context. Technical plans have
152 windows/context: 52 C4, 48 scientific, 52 code.

Calibration needs 2,250 unique windows/context through 250 and 4,500 through
500, counting three domains and three seeds. Without reuse, separately running
100+250+500 executes 7,650 windows/context. Existing 100-window document-budget
plans are exact input-ID subsets of the 250 plans for all three seeds. Retain
each budget/seed table and provenance even when the underlying measurement is reused.

## Measurement-preserving implementation

1. **Native contribution capture.** Retain unweighted expert outputs from native
   batched GEMMs and the original row/slot mappings. At the intervention layer,
   each ablation multiplies by native coefficients with exactly one coefficient
   zeroed, then accumulates from zero in native expert/scatter order. Preserve
   shapes and all untargeted rows. Never subtract from the finished mixture,
   renormalize survivors, skip zero multiplication as an assumed equivalent, or
   replace native GEMM with single-token GEMV.
2. **Layer 15 first.** Reuse the unchanged decoder prefix, intervention-layer
   attention/router and expert computation. Replay mixture, original residual
   addition, final norm and output head. Preserve native tensor shapes/strides
   initially; the current output head already consumes only the last token.
   Produce eight independent outcomes starting from immutable baseline buffers.
3. **Layers 0/7.** Reuse the unchanged prefix through the intervention point,
   then run the remaining 15/eight decoder layers at native full shapes. Restore
   masks, rotary embeddings and position identities. Downstream routing must
   respond naturally; never freeze it to baseline routes.
4. **Optional further expert reuse.** Only after profiling: reuse a downstream
   expert output if its complete input, row order, batch shape and weighting
   dependencies are unchanged. Recompute changed batches at native shapes;
   rebuild sums in original order. Do not assume mathematically unaffected rows
   are bitwise unchanged when routing changes GEMM batch shapes. Attention and
   cache memory costs remain. This is an experiment, not a promised speedup.
5. **Nested observation reuse.** Key raw measurements by exact model, executor,
   input bytes/identity, context, predictor/target, layer and operator. Verify
   compatibility before reuse. Render each budget's correct sampling metadata
   with an explicit link to original measurement provenance; never copy stale
   plan identities or count one measured runtime repeatedly as new GPU work.
6. **Session and I/O efficiency.** Keep one verified model loaded across stages;
   use bounded streaming buffers and lossless coefficient encoding. Remove
   unused captured mixtures and avoid Python byte-list conversions where safe.
   Profile finite checks, CPU/GPU transfers and fsync before changing them.
   Preserve complete-event durability, recovery and exact cleanup. Keep the
   existing full-forward cleanup in the initial implementation and budget.

No cross-window KV reuse, mixed precision, TF32, fused attention substitution,
intervention batching or lowered measurement coverage is part of this first plan.
An optimization is retained only when validated saved observations remain complete.

### Data and statistical contract

Keep all native selected expert IDs and lossless weights; calibration routes at
all input positions for every planned layer; document/domain frequencies;
seed/budget stability tables, rankings and fixed-probe selections; baseline NLL,
all eight ablated NLLs and derived delta NLLs for each evaluated event. Preserve
dataset/revision/domain/group/document/token/window/layer/expert/sampling IDs,
M0 seeds and all 64 ordered-pair draws, M4 inputs, timings and resource evidence.
M0/M4 and alternative ranking selections reuse measured effects on CPU.

Keep equal-document domain means, 0.5 scientific + 0.5 code primary weighting,
whole-source-group bootstrap dependence with repeated draw slots, cluster variance,
2,000 planning and 10,000 final bootstrap-t draws, >=95% valid draws, prescribed
quantiles/tails and Holm correction. C4 remains descriptive. Minimum counts are
not proof that the 0.005-nat half-width target can be met. Two-position pilot
results cannot establish unchanged precision after reducing main positions.

## Numerical and hardware acceptance

Use disjoint technical inputs at 4096 and 2048; layers 0/7/15; all eight active
experts. Compare accelerated baseline AND each accelerated ablation with the
corresponding native full forward on the candidate production GPU:

- Exact baseline/local routing IDs and coefficient bytes, survivor coefficients,
  untargeted local rows, parameter integrity and cleanup/repeated baseline.
- Mixtures and predictor logits: `atol=1e-6`, `rtol=1e-5`.
- Baseline and each ablated NLL: absolute error <=1e-6 nats; retain per-case
  discrepancies and delta-NLL/metric agreement diagnostics, not only averages.
- Downstream routes may change under ablation; compare against the same native
  ablation, not against baseline. Cover exceptions, nonfinite outputs and resume.

Reject failed optimizations without widening tolerances. CPU fixtures establish
implementation behavior only. Reuse no A40 receipt to approve a new GPU/executor.
Select and freeze production implementation/runtime before calibration/pilot;
hash-changing implementation edits after collection require compatibility review
and the existing strict identity gates, never a bypass to pool mismatched runs.

## Budget model, evidence and alternatives

The A40 diagnostic reports baseline 1.824 s and cleanup 1.246 s on one 4096-token
C4 window at layer 0. Corrected no-op logits/mixture/NLL match exactly there;
`full_mechanics_pass` is false. This is not a representative throughput benchmark.
Applying 1.25-1.82 s to the current minimum main gives roughly 187-272 hours.

Published Pod prices checked during the preceding analysis on 2026-10-11:
A40 $0.59/h, L40S $1.09/h, RTX PRO 6000 96 GB $2.49/h, H100 PCIe $2.89/h,
H100 SXM $3.99/h ([Runpod pricing](https://www.runpod.io/pricing)). These are
planning quotes, not verified live offers or measured speed. Catalog requests
returned HTTP 403; stock, exact tier, CUDA host compatibility and placement with
the existing network volume remain unverified. Refresh before a paid decision.

L40S is the $12 candidate. RTX PRO 6000 offers more activation headroom; H100
may help bandwidth-heavy work, but its tensor-core ratings are not a speed
prediction for TF32-disabled FP32. Choose by measured total cost and fit.
Specifications: [L40S](https://www.nvidia.com/en-in/data-center/l40s/),
[RTX PRO 6000](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/),
[H100](https://www.nvidia.com/en-us/data-center/h100/).

Let A be measured complete baseline + eight layer-15 effects + cleanup +
persistence seconds/window; let B0/B7 be incremental secondary-layer costs.
Then `T_main = sum_domain(P*A + Q*(B0+B7))`. Compute sensitivity independently
from its actual selected groups and 2048 timing. Deduplicate shared times by
input ID, since current per-layer events repeat input timing fields.

Illustrative counterexample, NOT a forecast: assume full forwards of 0.25 s
(4096), 0.15 s (2048), equal decoder-layer costs, final-layer reuse and ordinary
suffix replay, retaining full cleanup. Main work approximates
`[2P + 8Q*(15/16 + 8/16)]*0.25 = 13.05 h` before extra capture/tail overhead.
Thus even this optimistic scenario cannot meet ten hours for main alone.

| Stage | Scenario GPU hours | Scenario cost at $2.49/h |
|---|---:|---:|
| Remaining verification + benchmark + candidate validation (allowance only) | 1.50 | $3.74 |
| Loading, transfer, finalization (allowance only) | 0.25 | $0.62 |
| 4096 calibration through 250, with observation reuse | 0.3125 | $0.78 |
| Additional 4096 calibration through 500 | 0.3125 | $0.78 |
| Matched 2048 calibration through 250 | 0.1875 | $0.47 |
| Additional 2048 calibration through 500 | 0.1875 | $0.47 |
| Both fixed probes | 0.0853 | $0.21 |
| Pilot through 80, reusing earlier observations (plus unmeasured tail overhead) | 0.0669 | $0.17 |
| Main including C4 | 13.0470 | $32.49 |
| Matched 2048 effects on actual minimum-candidate subset | 1.9533 | $4.86 |
| Subtotal | 17.90 | $44.58 |
| 20% reserve | 3.58 | $8.92 |
| Total | 21.48 | $53.49 |

Replace every assumption/allowance with measured end-to-end evidence. Do not
transfer those speeds to L40S. Unknown verification remainder can exceed the
allowance. Final analysis, acquisition/preparation and bootstrap computation can
run on local CPU; GPU billing while waiting still counts. The current C4
supplement already exists; new acquisition, if necessary, belongs in CPU/transfer
accounting rather than disappearing from the workflow.

Provisional cash envelope: $2 remaining verification/benchmark, $9 scientific
collection, $1 storage/overhead. This is not a forecast or authorization.
Reconcile actual running charges, volume/disk retention, R2 requests/storage and
tax/fees if applicable. Report main-only hours/cost AND total remaining hours/cost.
Require the full forecast plus reserve to satisfy both remaining limits; revise
after pilot allocation. Never suppress escalation, instability or incompleteness
to force a budget PASS.

## Bounded benchmark and decision points

Before any paid benchmark, finish CPU implementation checks, calculate reference
forward counts, refresh real placement/rate evidence and request authorization
for the concrete capped run. Do not run alongside the existing verification.

First screen: three existing technical groups/domain, four deterministic
positions/document (all available if fewer), both contexts, three layers, all
eight experts. Include all eligible files in selected technical groups, so derive
the exact window count instead of assuming 72 if a group has multiple files.
Compare native, final-layer reuse and suffix replay; include optional expert reuse
only if ready. Use a hard cap of 30 aggregate GPU minutes or the smaller remaining
benchmark dollar allowance. Loading and incomplete attempts count. No automatic
retries, extra GPUs or open-ended kernel experimentation.

Record synchronized total elapsed and stage/component times, domain/context
variation, persistence, GPU allocated/reserved and whole-device observations,
host RSS, cache peak, loading and storage use. Use a conservative timing estimate
with an explicit reserve, not the fastest window. Peaks sampled by one tool do
not alone prove fit. A cap hit is incomplete evidence, not a passing benchmark.

If the screen passes feasibility, complete validation on the full accepted
technical plans and obtain a new receipt within an authorized cost envelope.
After calibration stability and sufficient pilot, recompute exact allocation,
sensitivity counts and both budgets before main freeze. If no candidate meets
scientific gates and both budgets, return a no-feasible-allocation result.

## Research amendments: proposed, none approved

1. First tradeoff: reduce descriptive depth positions from 16 to the largest
   nested count fitting the measured budget; the $12 fallback discussed was
   zero layer-0/7 main and matched-sensitivity interventions. That changes
   `sampling.main_secondary_positions` from 16 to an explicitly approved value
   (zero for omission); preserve 64 primary positions, calibration, technical
   checks and controls. This loses depth precision/coverage; zero abandons that
   descriptive analysis. Current hardcoded layer/position logic must be amended
   consistently only after approval; do not merely edit one config field.
2. If still infeasible, propose a concrete smaller primary position count with
   pilot-based uncertainty assessment and a matching sensitivity plan. Preserve
   all eight experts at retained events. Do not claim unchanged precision.
3. If pilot requires more groups than affordable, an explicitly looser
   `planning.target_ci_half_width_nats` changes the intended precision. Do not
   silently reduce source-unit minima or truncate whole repositories.

Omitting calibration escalation, C4, controls, individual effects, matched
sensitivity or bootstrap safeguards is not an implementation optimization.
If complete unchanged research cannot fit, report that clearly.

## Evidence anchors and continuation

These are fresh-clone local paths; ignored files may be absent in another clone.
Absence requires locating the referenced sealed artifact, not inventing evidence.
The git working tree was already dirty; HEAD alone does not identify this code.

| Artifact | SHA-256 at inspection |
|---|---|
| `configs/research.json` | `2daadce595cfe6f38747c70c31e07052ed49a926080513873a82abf0ccf24fb4` |
| `data/prepared-inputs/combined-2026-10-10/manifest.json` | `d3321553da95fb1853b1b3e83ad486b1414ea92ba6ed2fcd53c4b30f7fb59ccd` |
| `data/plans/2026-10-10/technical/source-review.json` | `8431ba4f125c85a37491b0cfed4b1aa9a750772f4962c378f63506b0a7a2d877` |
| `src/run_experiment.py` | `78cf812a6ef4a32198920fc735eb7221097f9436623655ab89ad54fec2759fa1` |
| `src/interventions.py` | `ef77eadff25c85bacbd55a4ddfcd29cfe7835d2fd31421d917e25d736e77af34` |
| `src/verify.py` | `6d750f101780e36d999dec9d3bc61c931b2d8b60a1079bd96a4f8cf2c259a319` |
| `data/noop-fix-a40.local.json` | `7950ddc9a35713f7ee4efaa3cd7d6a54dd1ee36f0a3a0e986ec519c0706bd53b` |
| Pinned Transformers 5.12.1 `modeling_olmoe.py` | `498ab5720463327ab7235f3a3103aefa49d2f93da10c3cfb7ef7002774b05531` |

Prior analysis checked the prepared document file hash against its manifest and
the review-to-manifest link, counted candidate positions from prepared metadata,
and checked 100/250 window-ID nesting. Counts are reproducible with current
`priority(3007, ...)`, `make_plan` ordering and `shorten_plan` group selection;
no main outcomes or allocation freeze were created. Recheck hashes on handoff.

The coordinating agent owns integration and the final feasibility decision.
Worker roles and evidence gates are on the TODO board. Keep findings here and task
status there; do not create another authority document or rewrite README to
encode the resource constraints.

## Integrated local result, 2026-10-11

The measurement, planning and independent-validation workers completed separate
file assignments; the coordinator integrated shared timing, cache and receipt
contracts. The accepted research file, README and existing native verification
implementation retain their pre-session hashes. No process, pod, research setting,
remote artifact or paid workload was changed.

- The optional `native_shape_suffix_v1` path captures native batched expert
  contributions, zeros one coefficient, scatters in original order, and replays
  layer 15 or the natural downstream decoder suffix at original shapes. All
  eight outcomes and full cleanup remain. Native collection remains the default.
  Optimized collection needs fresh native mechanics plus a matching complete
  optimized full-technical receipt; a screen cannot open this gate.
- Immutable raw-result reuse checks exact input/window/target, source document,
  tokenizer, model, execution mode and runtime/implementation. New plans keep
  their own sampling identity and link the original sealed measurements. Compact
  byte-offset indexes avoid retaining cached payloads; missing nested windows
  still execute. End-to-end CPU collection/reuse and injected timing-write failure
  recovery preserve raw payloads and avoid duplicate events.
- One `VerifiedModelSession` can serve an explicit collection sequence. It checks
  catalog, directory, seed, runtime, code identity and parameter versions on each
  acquisition. Routing serialization is lossless. Durable timing sidecars record
  shared/cleanup/effect/cache/persistence time per physical input after journal
  writes. Absent sidecars and partial/cache-hit measurements stay unknown for
  throughput forecasting. Token windows now stream one at a time, and compact journal/reuse indexes
  retain byte offsets instead of raw rows. ID/offset metadata still scales with
  planned events; this is not a constant-memory collector. No native memory profile exists.
- Main planning now includes fixed C4 collection cost. The complete budget tool
  separates main from total, deduplicates shared input work, includes both contexts,
  whole code groups, calibration/pilot ladders, sensitivity, verification, screen,
  load, transfers/writes, host waits, storage, final CPU analysis and a 20% additive
  reserve. Nested reuse also requires its own measured/bounded allowance: through
  calibration 500 there are 6,300 cached consumer windows across both contexts,
  plus 360 cached pilot windows through 80. Unknown measurements, insufficient pilot or unaccepted executor block
  feasibility. Domain/context cohort maxima remain extrapolations, not guarantees.
- Independent validation compares a pristine native baseline and each full native
  ablation against contribution capture/suffix replay. It checks exact routes,
  survivor coefficients, untouched rows, downstream routing, cleanup and frozen
  mixture/logit/NLL tolerances. Per-expert discrepancies, synchronized native/suffix
  timings, progress fsync, allocator peaks and RSS are recorded. Screen receipts
  are diagnostic only. Full acceptance rejects partial technical groups, changed
  windows, mismatched context lineage and stale native receipts.

Further downstream expert caching is rejected for this candidate: no measured
benefit justifies dependency complexity or extra memory. Local tests exercise
actual pinned Transformers tiny OLMoE models, not the pretrained 4096/2048 GPU
workload. Final check results and file identities are in
[integration evidence](optimization-integration-2026-10-11.json).

### Dollars and time: evidence and unknowns

[Read-only accounting](optimization-accounting-2026-10-11.json) observed the known
pod `s8x6s5hg33mtme` as `EXITED` at 2026-10-10 19:32 UTC, with its last $0.59/hour
rate; this is not evidence that the user's separately mentioned verification
finished. No remote process was inspected or disturbed. The API's Oct-10 UTC-day
billing window totals $0.7406598956 across the account (GPU, disks and network
storage), including $0.6391525374 for that pod and $0.0291666677 for its volume.
This is historical accrued cost, not a remaining credit balance. It cannot be
deducted from $12 without knowing whether $12 is already net of those charges.
The ten-hour clock's start and remaining verification duration are also unknown.

[Live catalog snapshot](optimization-gpu-catalog-2026-10-11.json) listed secure A40
at $0.59/hour in CA-MTL-1, L40S at $1.09/hour in other centers (LOW stock there),
and RTX PRO 6000 Server 96 GB at $2.49/hour. The existing 50 GB volume is CA-MTL-1;
no L40S placement there was listed. Stock is not a reservation; staging into another
center, exact host RAM, disk costs and driver compatibility still need confirmation.
Rates are compute-only catalog prices, corroborated by
[Runpod pricing](https://www.runpod.io/pricing), not an all-in deployment quote.

For arithmetic only, ten fully billed hours would cost $5.90/$10.90/$24.90 on
those three GPUs. With the tool's 20% additive reserve, a fresh ten-hour remaining
workflow allows at most 8.333 base hours. A fresh $12 compute-only envelope permits
16.949/9.174/4.016 base GPU-hours respectively. Those are ceilings before other
costs and elapsed work; they make no throughput or fit claim. The complete unchanged
workflow remains **UNKNOWN**, as recorded by the reproducible
[budget inputs](optimization-budget-inputs-2026-10-11.json) and
[result](optimization-budget-result-2026-10-11.json). No feasibility PASS or main
allocation freeze was manufactured.

### Concrete bounded GPU screen proposal — not authorized

Use one secure L40S 48 GB, >=64 GB host RAM, FP32/eager/TF32-off, batch one/no KV
cache, in a freshly quoted center with compatible CUDA 13.0 driver and verified
pinned weights. L40S is conditional on staging fitting the cap. A40 in CA-MTL-1
is a separately approved fallback that can reuse the existing volume; it has no
assumed equivalent speed. Do not reuse or contend with a running verification.
No fallback, retry, second GPU or full acceptance is implicit in this proposal.

The sealed `data/plans/optimization-screen-2026-10-11-v2` pair selects three seeded
whole technical groups/domain and all four positions/document, at both contexts:
36 windows/context, **72 windows, 216 layer events and 1,728 individual native vs
suffix comparisons**. Its full independent validator executes **1,944 full
forwards plus 1,728 suffix replays** (27 full forwards/window: native baseline,
captured baseline, 24 native ablations, cleanup). A standalone native collector
would execute 1,872 forwards; optimized production would execute 216 full forwards
plus the suffixes. These counts differ intentionally; diagnostic overhead is not
production speed. The earlier unexecuted screen package is preserved and superseded
by v2. The proposal JSON and integration evidence pin exact plan/code hashes.

Proposed ceiling: **the smaller of 30 aggregate GPU minutes, the remaining ten-hour
allowance and $0.75 all-in (also capped by reconciled remaining cash)**. At the catalog
L40S compute price, 30 minutes costs $0.545; storage/staging/idle charges are additional
and unknown. Provisioning, model/input staging, loading, failures and shutdown must
fit the same envelope. The billing deadline is enforced by the authorized operator
outside Python; `--max-seconds` bounds the validation application and is not a cloud
termination mechanism. Set it to the remaining billed time after staging, with a
shutdown allowance. Stop on cap, discrepancy, nonfinite output, identity/fit failure,
or discovery of another running verification. Retain every partial report; no retry.

The existing image digest is
`docker.io/aryamudas/dsic-3007@sha256:ae58005a9187a2e69f3d63bd3802eea063ae206801239abdc20e9286f77612d3`.
It predates this candidate. Stage the hash-checked candidate source archive in a
separate directory; do not patch the existing verification checkout. Its expected
Python/Torch/Transformers/CUDA runtime is 3.14.6/2.12.1/5.12.1/13.0, subject to
actual driver/runtime identity and native acceptance. No build/push was performed.
After authorization and staging, run from the candidate directory:

    python -m src.workflow validate-optimized --plans /workspace/optimization-screen-2026-10-11-v2/4096 /workspace/optimization-screen-2026-10-11-v2/2048 --model-dir /workspace/models/OLMoE-fp32 --output /workspace/results/optimized-screen-attempt-1.json --scope screen --max-seconds <remaining-seconds-with-shutdown-allowance>

Outputs are the receipt and fsynced `.cases.jsonl` discrepancy/timing report,
plus provider billing and device/RSS/storage observations. A passing screen is
still diagnostic. It cannot supply complete production calibration/probe profiles
or prove fit. Full native mechanics for both exact 12-group/domain technical plans,
full optimized acceptance, collection profiles, stable calibration and a sufficient
pilot remain separately accounted gates. Only then replace the minimum census by
the actual selected allocation and reassess both limits with reserve.

## Latest user-run handoff

The reviewable implementation is `runpod/screen.py` plus
`runpod/screen_remote.py`; source and paired screen archives are frozen in a
versioned local package separate from the earlier candidate. The user runs
`python runpod/screen.py --run`; default invocation performs read-only preflight.
The pinned existing image supplies Python/Torch/Transformers, while the uploaded
archive supplies the current implementation. No paid attempt was started by the
coordinator. No research settings/tolerances or native verification were changed.

One attempt has a 30-minute aggregate lease, a ten-minute provisioning/SSH bound,
and a $0.75 planning allowance including staging/loading/failure and teardown.
At the verified catalog ceiling of $1.09/h plus an assumed conservative $0.02/h
storage allowance, 30 minutes budgets $0.555, leaving $0.195 of the attempt
allowance as margin. Storage sources: [container/standard-volume rates](https://docs.runpod.io/pods/storage/types).
The guard is local, not a provider spending limit: the operator keeps the local
computer awake/connected and inspects the console if deletion is unconfirmed.
Pod deletion retains the user network volume, whose independent storage billing
continues outside the attempt. No automatic attempt retry is permitted.

Quoted arithmetic only, not a workload forecast: at $1.11/h the $12 envelope
allows about 10.81 aggregate GPU hours, while ten hours is the tighter cap if
that full remaining time is actually available. A 20% reserve would leave
8.33 hours of unreserved workflow time under that ten-hour constraint. These
figures do not demonstrate throughput, sufficient pilot/allocation or full
scientific feasibility. The refreshed forecast remains UNKNOWN; native fidelity,
whole-device fit, production profiles, staging/finalization and the ten-hour
accounting start remain blockers.

All stdout/stderr and lifecycle heartbeats appear in `runpod/runs/latest/console.log`;
a detached guard has its own `guard.log`. Remote results retain bootstrap and
phase logs, GPU observation JSONL, live runner status and durable numerical cases.
Screen pass cannot approve production collection. Latest local check and hash
evidence: [user-run preparation](optimization-user-run-preparation-2026-10-11.json);
[refreshed budget](optimization-user-run-budget-result-2026-10-11.json).


### Native partial evidence — 2026-10-11

The user-run L40S registry-v2 attempt ended at its time cap after **17/72 inputs
and 408/1,728 comparisons**. Completed native/optimized NLL, logits and mixtures
matched exactly; cleanup and unchanged-parameter checks passed for those inputs.
Coverage is 12 C4 + 5 scientific windows at 4096 only. Code and 2048, full technical
acceptance, whole-workflow fit and production throughput remain unverified.
The independent review reconciles individual durable progress with case rows.
Both paid attempt pods are absent on fresh GET; verified weights and network
volume survive. Raw receipts remain immutable, including the null image metadata;
the future-run helper fixes declared-image propagation in a versioned successor.

The completed attempt elapsed 1,094 seconds, including a 258-second weights stage
and 669-second validator. Its nominal compute at the observed $1.09/h rate is
$0.3313, **not an invoice**. Current pod billing reads return no rows; that does
not establish zero charges. The volume reports $0.004861 for 00:00–01:00 UTC and
continues billing. A deliberately conservative 454.1-second first-attempt upper
bound gives 25.80 minutes / $0.4688 nominal compute for both attempts. Subtracting
only that estimate from the user $12 ceiling leaves $11.53 before storage/other
charges; current wallet is unverified. If the ten-hour allowance began just before
the first attempt, these attempts alone leave 9.57 hours; earlier work/idle wall
time and the actual accounting start remain unresolved. Neither provisional figure
is a verified remaining budget.

For the 17 completed 4096 inputs, synchronized means are 0.934 s captured baseline,
0.918 s cleanup and 0.0196 / 3.900 / 7.292 s for all eight optimized ablations at
layers 15 / 7 / 0. Applying those components to the candidate P=38,296 and Q=9,677
yields **about 50 hours for main 4096 alone**. This calculation excludes model
download and loading. It assumes transfer across domains/input sizes and to the
production collector; it omits 2048 and all other stages/overheads/reserve. It is
an adverse planning signal, not a measured complete forecast or universal lower
bound. The formal complete-workflow verdict stays UNKNOWN; hold full collection
under the $12/ten-hour envelope pending a defensible revised execution decision.
Cached weights save download work but do not remove this per-input cost.

A concrete **proposal only** is one further balanced diagnostic of 18 predeclared
windows (three technical groups/domain at both contexts, one deterministic existing
position/group), alternating domain/context, all eight experts at layers 0/7/15.
This is 432 comparisons, 486 full forwards and 432 suffix replays, with original
FP32/eager settings and frozen tolerances. Bound the entire single attempt to
20 minutes and $0.40, further limited by reconciled remaining cash/time. Nominal
compute is at most $0.3633 plus up to $0.0067 planned storage allowance at the
previous conservative rate. Verify cached weights; count hashing/loading/setup/
retrieval/teardown; preserve partial logs and no retry. Its sealed selection/order
and source successor still need preparation/review before a new paid authorization.
It cannot replace the original full screen or technical gate. Do not spend on
this diagnostic merely to chase a PASS: first decide whether its missing coverage
would change the budget decision.

Reproducible numbers and explicit assumptions: [partial results](optimization-l40s-partial-results-2026-10-11.json).
Independent audit: [review](optimization-l40s-partial-review-2026-10-11.json).
Actual API billing/cleanup reads: [read-back](optimization-l40s-billing-readback-2026-10-11.json).


Source-accounting correction: the optimized production collector also executes a
pristine reference forward per input before captured baseline and cleanup. The
~50h component illustration above omitted this current implementation cost.
Adding its observed diagnostic component produces ~60h main4096, still dependent
on timing-transfer assumptions and not a measured production forecast. Download
is excluded from both illustrations. The next local investigation targets
reference/cleanup consolidation and full-window suffix work; no source executor
was changed by this audit.

Context discussion:2048 is accepted matched sensitivity, not an approved primary
replacement;1024 is outside the current core. All4096 input tokens, including the
final predictor, are used; the next-token target lies outside the input. Shorter
windows reduce arithmetic work but alter conditioning, routing and effects.
Measure matched inputs/targets and context calibration before judging scientific
robustness; no wall-time speedup or unchanged conclusion is assumed.
See [source audit and optimization priorities](optimization-next-direction-2026-10-11.json).


### Cleanup-reference implementation and paired successor, 2026-10-11

O18 is implemented locally. Optimized production measurement now performs two
full forwards per input (captured baseline and exact native cleanup), saving one
from the preserved three-forward predecessor. All eight effects per requested
layer still execute; suffix replay, native arithmetic, frozen tolerances and
4096 primary are unchanged. Calibration/probe costs are unchanged. Independent
technical validation retains its separate native references and full ablations;
its overhead must not be substituted for actual collector cost.

The source-derived saving is one full forward per optimized input, not a measured
GPU throughput improvement. Applying the earlier diagnostic components yields
about 60.035h for the old main4096 schedule and 50.002h for this candidate,
conditionally saving about 10.033h at P=38,296. These illustrations assume timing
transfer and omit other stages/contexts; both exclude download. Two full-forward
components alone imply about 19.709h under that assumption. Consequently this
first change does not establish feasibility under ten hours or $12. The complete
forecast remains UNKNOWN, and actual remaining cash/clock boundaries are unresolved.

The earlier unprepared 18-window coverage-only proposal is superseded by the
prepared six-window paired collector diagnostic in
[paired specification](../../runpod/paired-benchmark.json). It measures old/new
collection for C4/scientific/code at 4096 and matched 2048, with `[15,0,7]` and
`[15]` shapes, alternating AB/BA and separate paired warmups. Independent native
checks cover 144 effects; total planning ledger is 232 full forwards and 624 suffix
replays, including measured pairs, warmups and diagnostic checks.

One attempt is proposed: <=1,200 seconds and $0.40 all phases, further reduced by
reconciled remaining work limits and live compute quote plus the explicit
$0.02/hour storage allowance and default $0.03 additional-charge bound. At the
historical $1.09/hour compute quote, 20 minutes is $0.37 compute+storage. Bounds
are planning assumptions, not invoice measurements or a provider hard cap.
Cache verification only; missing/corrupt weights stop without redownload. Phase
caps include startup/upload/preflight120s, cache verification180s, model load/hash120s,
warmups90s, independent checks240s, actual collection+persistence300s,
finalization30s, retrieval/teardown120s. They do not prove completion within cap.
The local guard depends on an awake connected computer and responsive provider;
confirmed teardown is required, and retained volume lifetime charges remain.

No paid work was launched for this implementation. User command/progress details:
[workflow](../../WORKFLOW.md#paired-collector-benchmark--prepared-locally-not-launched).
Local validation and package identities:
[integration](optimization-cleanup-reference-integration-2026-10-11.json).

Live read-only preflight at 2026-10-11 04:07:24 UTC: frozen package verified,
existing registry/volume identities accepted and no account pods observed, then
stopped because L40S stock was unavailable in retained-volume region US-MO-1.
No resource was created. Current quote, wallet and original ten-hour remaining
boundary remain unresolved; repeat the read-only preflight when stock returns.


### v5 setup timeout, v6 reallocation

User-run paired attempt041923 stopped at the shared120s startup/upload cap before
measurement. Boot81s left39s for upload; total elapsed131.17s including confirmed
teardown. The20-minute aggregate lease was not exhausted. Approximate nominal
compute at1.09/h is$0.03972, plus storage allowance totals$0.04045; this is not
actual invoice reconciliation. Deduct the attempt from any remaining work budget
and wall time; no reset or zero-cost assumption.

v6 keeps the global1200s/.40 envelope, with phases240 startup,180 cache
verification,90 model,60 warmup,240 nativechecks,240 pairedcollector,30 finalization,
120 retrieval/teardown (sum1200). These are planning caps, not measured completion
evidence. Remaining work may still end INCOMPLETE; no automatic paid retry.
[Evidence and hashes](optimization-upload-timeout-2026-10-11.json).


### Measured v6 collector schedule, successful bounded diagnostic

The v6 user attempt043533 completed and confirmed teardown in659.65s, within its
computed1037.84s lease and declared$0.35 planning allowance. All six windows and
144 independent native comparisons passed, with zero recorded numerical errors.
Measured4k durable per-input means: layer15 old2.842s/new1.901s, three-layer
old13.518s/new12.605s. These are real paired collector observations, separate
from cached hash verification82.84s, model hash/load30.29s, warmup42.07s, native
checks181.94s and paired collector phase149.19s (both arms). Download was absent;
setup/hash/load/retrieval/teardown costs are counted rather than erased.

Apply the observed warmed three-domain means to P=38,296/Q=9,677 using28,619
primary-only and9,677 three-layer inputs: conditional main4096 old58.932h becomes
48.999h. Three selected documents/groups do not establish representative timing
or a universal bound. Othercontexts/stages/reserve are missing. Complete-workflow
verdict remains UNKNOWN with an adverse main4096 signal; fullcollection staysheld.

Nominal compute+storage allowance is$0.20339 for this attempt; adding its$0.03
fixed-cost assumption gives$0.23339, below$0.35. No actual invoice/wallet read was
performed. Together with the upload failure:790.82 active attempt seconds and
$0.24384 nominal hourly charges ($0.30384 with both fixed assumptions). Continuous
wall span was1629.82s, including about839s outside active attempts. Do not claim
that the two attempts consumed only13m11s of wall time, or that the original
ten-hour clock /$12 remaining boundary is reconciled. Retained volume lifetime
charges and all earlier work still count.

Next paid work is held while O19 profiles/prototypes guarded replay-batch reuse
and reassesses baseline/cleanup costs. Fulltechnical gates and native calibration/
probe remain; no scientific sample/context/tolerance amendment is inferred from
this diagnostic.
[Results](OPTIMIZATION-PAIRED-RESULTS-2026-10-11.md),
[ledger and raw hashes](optimization-paired-results-2026-10-11.json).

Conditional candidate main4096 alone at49h ×(1.09+0.02)/h is about$54.39;
rate transfer, cohort representativeness and future allocation remain assumptions.
This is adverse to both$12/10h, not a complete feasibility proof.

# Research execution

## Path B migration status — 2026-10-12

Arya selected Path B; the [new README amendment](README.md#amendment--path-b-selected-2026-10-12)
and [versioned scope decision](configs/path-b-decision-v1.json) define that choice.
The local workflow now defaults to `configs/research-path-b-v1.json`. Original
`configs/research.json`, old commands and sealed plans below remain historical;
pass `--settings configs/research.json` explicitly for historical reproductions.
The [implementation record](docs/validation/PATH-B-IMPLEMENTATION-2026-10-12.md)
records the changed scope, local checks and pending native acceptance.

Current v11 retains removal of repeated weights verification from the Path B benchmark,
following Arya's explicit instruction. It requires the completed preparation
receipt, then loads directly from the Global Volume without another SHA256/F32/
architecture verification scan. The receipt records preparation, not fresh byte
verification. Cheap presence/size checks remain. Native comparison gates and
research settings/tolerances remain unchanged. Loader identity changed and fresh
native acceptance is pending.

Unrequested copy/cache/proof/disk changes were removed. Disk remains35GB.
The original $.20 diagnostic is consumed and the proposed $.60 retry is withdrawn.
Arya now authorized one $.30/900s user-run benchmark attempt, no automatic retry.
Setup/resume remain disabled; model preparation is complete. Planning caps are
not provider invoice guarantees. Effective lease at configured guard/allowances
is about875.68s including cleanup. User-run command:

```bash
python -u runpod/path_b_preflight.py --run --benchmark-authorization-id global-path-b-benchmark-2026-10-13-v11
```

Durable log: `runpod/runs/latest/console.log`.

Offline package verification: `python -u runpod/path_b_preflight.py`.
[Weight verification removal](docs/validation/PATH-B-WEIGHTS-REVERIFICATION-REMOVAL-2026-10-13.md).
Old packages and raw receipts are preserved. No paid action or model preparation
was performed for this local change.

## Execution redesign and agent handoff

Start with [the execution budget plan](docs/validation/EXECUTION-BUDGET-PLAN.md)
for the source findings, budget assumptions, proposed optimizations, numerical
acceptance gates and unapproved research alternatives. Then use the
[optimization TODO](docs/validation/OPTIMIZATION-TODO.md) for dependencies,
ownership and completion evidence. These are execution documents; README and
configs/research-path-b-v1.json remain the research authority. A documented task or
proposed amendment does not authorize paid execution or change accepted settings.

## Historical operational reference

The commands and inventory snapshots below predate Path B. Do not run these
old plans with the current default settings. Historical reproduction requires
explicit `--settings configs/research.json` where supported. Current inputs,
plans and remaining gates are in the implementation record linked above.

### Local snapshot — 2026-10-11

As inspected on 2026-10-11, the successor prepared bundle is
`data/prepared-inputs/combined-2026-10-10/`. Its approved source review is copied
into `data/plans/2026-10-10/technical/source-review.json`; it reports no allocation
shortfalls and reserves 500 calibration documents per domain. Sealed C4 and code
supplements exist under `data/supplements/`. The earlier inventory below is
historical and must not trigger repeated acquisition or replace this successor.

Plans for technical checks, fixed probe, calibration 100/250 with three seeds,
and the initial pilot exist under `data/plans/2026-10-10/`; matched technical,
probe and calibration plans exist under `data/plans/2026-10-10-2048/`.
Plan presence does not establish completed measurements or a main freeze.

`data/noop-fix-a40.local.json` records one passing numerical diagnostic and
explicitly has `full_mechanics_pass: false`. The user reports that full technical
verification is running and instructs agents to leave it running. This documentation
update did not inspect or change the remote process. Obtain a completed receipt
before claiming full mechanics acceptance.

### Earlier retained-only inventory

All 30,000 retained documents have been retokenized with the pinned tokenizer
into the sealed, ignored data/prepared-inputs/retained-hostnames-2026-10-10 bundle.
The pending review is data/manifests/source-review-hostnames-2026-10-10.local.json.
This successor preserves full C4 hostnames, including `www.`; the earlier
preparation remains intact.

| Domain | Full-context calibration documents | Full-context evaluation documents |
|---|---:|---:|
| C4 | 7 | 64 |
| Scientific | 1,071 | 6,095 |
| Code | 99 | 597 |

These counts precede source/duplicate exclusions. Code document counts do not
establish repository availability. See [the acquisition plan](docs/validation/FRESH-START-2026-10-10.md).
The [local source audit](docs/validation/SOURCE-AUDIT-2026-10-10.md) counts
70 eligible C4 hostnames and 692 eligible code repositories. Two scientific
records share a publication URL and remain pending variant review.
At that earlier local snapshot, original FP32 weights and native GPU measurements
were absent. See the successor state above for the later diagnostic evidence.

## 1. Prepare and freeze inputs

    uv sync --locked --extra dev
    uv run python -m src.workflow status
    uv run python -m src.workflow tokenizer --output data/tokenizer-fp32
    uv run python -m src.workflow inventory --output data/inventory/new-snapshot
    uv run python -m src.workflow prepare --tokenizer-dir data/tokenizer-fp32 --output data/prepared-inputs/new-snapshot
    uv run python -m src.workflow audit-sources --prepared <prepared> --output data/source-audits/new-snapshot
    uv run python -m src.workflow review-draft --prepared <prepared> --output data/manifests/source-review.local.json

Use new output directories. The existing sealed preparation can be used directly.
Review must account for every document, record exclusion reasons, assign canonical
source groups and disjoint technical/calibration/probe/pilot/main partitions.
All four audit checks must say pass, and review status must say approved.
No command auto-approves variants, aliases, known clones or source independence.
Recorded source aliases must share one reviewed canonical group. The allocator
checks transitive alias links across the entire inventory, including excluded
records, so linked sources cannot be separated into stages or bootstrap clusters.

### Supplemental collection

[configs/collection-proposal.json](configs/collection-proposal.json) records the
pinned-source acquisition proposal and finite scan bounds. C4 and code successors
have since been collected and included in the current combined inventory above.
The commands below describe how to prepare a new acquisition if one is needed;
they are not a request to repeat completed collection. New source choices or
scan expansions need their own authorization. Write each approved source's
`config` object to a small JSON file and invoke the collector explicitly:

    uv run --extra dev python -m src.collection --domain <c4-or-code> --source-config <source-config.json> --revision <approved-40-character-commit> --max-source-rows <approved-bound> --tokenizer-dir data/tokenizer-fp32 --output data/supplements/<new-domain-snapshot>

The collector scans the complete bounded stream, keeps every unique eligible
document, preserves duplicate-source metadata, and never caps files per code
repository. Its pool covers observed rows; it does not claim complete remote
repository coverage. Short records are counted and filtered. Progress reports
row counts without text. Streaming read-ahead may exceed the yielded-row bound.
No automatic expansion is performed if the resulting pool is insufficient.

Combine verified retained bytes with the sealed supplemental directories, then
prepare and audit this combined catalog:

    uv run python -m src.workflow combine-inputs --supplements data/supplements/<c4-snapshot> data/supplements/<code-snapshot> --output data/manifests/combined.local.json
    uv run python -m src.workflow prepare --catalog data/manifests/combined.local.json --tokenizer-dir data/tokenizer-fp32 --output data/prepared-inputs/combined
    uv run python -m src.workflow audit-sources --prepared data/prepared-inputs/combined --output data/source-audits/combined
    uv run python -m src.workflow review-draft --prepared data/prepared-inputs/combined --output data/manifests/combined-review.local.json

Combined assets carry explicit dataset source references. Repeated appearances
of exactly the same dataset/revision/native-ID/text identity become one
observation, retaining every raw asset occurrence and duplicate metadata.
Different native identities or content remain separate for the source audit.

    uv run python -m src.workflow plan --prepared <prepared> --review <review> --stage technical --units 12 --output data/plans/technical
    uv run python -m src.workflow plan --prepared <prepared> --review <review> --stage probe --units 32 --output data/plans/probe
    uv run python -m src.workflow plan --prepared <prepared> --review <review> --stage calibration --units 100 --seed 3008 --output data/plans/cal-100-3008
    uv run python -m src.workflow plan --prepared <prepared> --review <review> --stage pilot --units 20 --output data/plans/pilot-20

Repeat calibration for seeds 3008/3009/3010 at budgets 100/250; add 500 if needed.
Membership documents stay nested and identical across seeds; window positions vary.
Pilot escalates 20/40/80 according to its analysis. Code pilot units are complete
repository groups; scientific/C4 units are documents. Plans preserve token IDs,
targets, positions, layers and sampling identities. Resolve input shortages first.

## 2. Private R2

Set the environment variables from .env.example through local/runtime secret
handling. Use an existing private bucket and reviewed source-use rights.
Keep credentials out of Terraform inputs and state.

    uv run python -m src.transfer push --kind experiment-plan --bundle <sealed-plan>
    uv run python -m src.transfer pull --kind experiment-plan --bundle-id <manifest-sha256> --destination <new-directory>

Other kinds: prepared-inputs, inventory, measurements, calibration, local-analysis.
Transfer one sealed bundle. Prefixes contain the manifest SHA-256. Upload publishes
the manifest last; streamed downloads verify every checksum and size. No public
ACL or public URL is created.

## 3. Authorized Runpod execution

[runpod/README.md](runpod/README.md) describes the image and Terraform inputs.
GPU, image digest, disks and region require explicit choices. Terraform defaults
to no pod. After paid authorization, provision one idle pod using a saved plan.
Terraform starts no experiment.

Code lives at /opt/dsic3007; persistent data lives at /workspace. Enter the code
directory and use python directly; its virtual environment is on PATH. Use
absolute artifact paths. Supply catalogs from the checkout or input plans.

    python -m src.workflow weights --catalog <model-catalog.json> --output /workspace/models/OLMoE-fp32
    python -m src.workflow verify --plan <technical-plan> --model-dir /workspace/models/OLMoE-fp32 --output /workspace/mechanics-4096.json
    python -m src.workflow collect --plan <technical-plan> --model-dir /workspace/models/OLMoE-fp32 --mechanics /workspace/mechanics-4096.json --output /workspace/results/technical-timings
    python -m src.workflow collect --plan <calibration-or-probe-plan> --model-dir /workspace/models/OLMoE-fp32 --mechanics /workspace/mechanics-4096.json --output /workspace/results/<run>

Explicit weight acquisition verifies original FP32 files against the pinned
commit and hashes. There is no dtype/backend fallback. Execution uses one GPU,
eager attention/experts, batch size one, no cache, deterministic operations, TF32 off.
Native whole-model fit and image CUDA compatibility still need GPU evidence.
Mechanics checks repeat determinism, target alignment, native/no-op reconstruction,
all eight local ablations, unchanged survivors and exception cleanup.
Receipts bind context, source review, model and inference implementation/runtime.
Technical collection supplies forward/layer timings and allocator/RSS peaks;
peaks alone do not establish whole-device or host fit.

## 4. Calibration, pilot and main freeze

Push each complete run to R2 and retrieve it locally. Preserve incomplete journals
on persistent storage. Matching identities resume complete events; missing/failed
events never become zero effects. Torn tails need explicit repair. Complete runs
are immutable.

    uv run python -m src.workflow freeze-calibration --runs <100-3008> <100-3009> <100-3010> <250-3008> <250-3009> <250-3010> --probe <probe-run> --output results/calibration-4096

If unstable, add all three 500-document runs. Freeze 250 after passing 100→250,
otherwise 500 after passing 250→500. Each seed-pair/budget comparison meets the
90% rule at the candidate budget; predecessor seed stability is not required.
Unstable tables cannot authorize pilot/main effects.
Transfer frozen calibration back to the GPU. Collect pilot effects by adding
--calibration <frozen-calibration> to the collect command, then retrieve:

    uv run python -m src.workflow analyze --run <pilot-run> --calibration <frozen-calibration> --output results/pilot-analysis
    uv run python -m src.workflow plan-main --pilot <sufficient-pilot> --technical <technical-timing-run> --calibration <frozen-calibration> --prepared <prepared> --review <review> --output results/main-planning
    uv run python -m src.workflow freeze-main --planning results/main-planning --prepared <prepared> --review <review> --output data/plans/main

Planning scales pilot domain variance by pilot/main group counts. The 2000
bootstrap-t draws supply the CI quantile span for projected precision.
For each available code-group count, find the smallest qualifying scientific
document prefix, then minimize projected time from measured technical timings.
Use actual position availability. This is projected precision, not achieved
precision or power. A failed pilot or unavailable qualifying allocation blocks
main freezing. The main freeze locks exact inputs, analysis, calibration and runtime.

Collect the frozen main plan with frozen calibration, retrieve it, and analyze
locally. Do not add main samples based on observed significance.

## 5. Matched 2048 sensitivity

    uv run python -m src.workflow shorten-plan --plan data/plans/main --subset --output data/plans/sensitivity-2048
    uv run python -m src.workflow shorten-plan --plan <calibration-plan-4096> --output <calibration-plan-2048>
    uv run python -m src.workflow shorten-plan --plan <probe-plan-4096> --output <probe-plan-2048>
    uv run python -m src.workflow shorten-plan --plan <technical-plan-4096> --output <technical-plan-2048>

Shorten every required calibration budget/seed. Verify the shortened technical
plan separately, collect matched 2048 calibration/probe, and freeze 2048 rankings.
Collect sensitivity effects with that receipt and calibration, then retrieve:

    uv run python -m src.workflow analyze-sensitivity --main <main-run> --sensitivity <2048-run> --main-calibration <4096-calibration> --sensitivity-calibration <2048-calibration> --output results/context-analysis

## Saved science

Calibration keeps every input-position expert ID and lossless FP32 coefficient.
Freeze saves document counts/utilization, seed/domain/layer tables, drift/Jaccard
comparisons, frozen ranks/tie blocks and individual matched/mismatched probe
selections. Equal-length windows make the selected token-weighted calibration
sensitivity identical to equal-document utilization.

Effects keep baseline NLL, eight individual ablated NLLs/ΔNLLs, native routes and
full observation/sampling identities. Offline output retains metric selections,
M0 pairs/draws, individual CD, document/group summaries, position variance,
routing frequencies, bootstrap group slots/validity, absolute CD intervals,
paired primary inference and scientific/code Holm tests. C4 and layers 0/7 are
descriptive. Context analysis checks whole groups, exact predictors/targets and
shortened inputs, then reports document-weighted changes with uncertainty.

Analysis uses settings saved in the frozen plan; later config edits cannot
silently change it. Retain raw bundles and the exact Git commit.
CPU fixtures, hashes and Terraform validation are implementation evidence.
They do not establish pretrained FP32 mechanics or scientific findings.

## Local optimization and budget checks

The optional suffix executor, immutable nested-observation reuse and bounded
screen are integrated locally. Current status and authorization boundary:
[execution budget plan](docs/validation/EXECUTION-BUDGET-PLAN.md),
[task board](docs/validation/OPTIMIZATION-TODO.md) and
[final local evidence](docs/validation/optimization-integration-2026-10-11.json).
README/research settings and the running verification remain unchanged.

These commands prepare/read local artifacts and launch no GPU:

    python -m src.workflow prepare-benchmark --primary data/plans/2026-10-10/technical --shortened data/plans/2026-10-10-2048/technical --output <new-screen-directory>
    python -m src.workflow timing-profiles --runs <sealed-measurement-runs> --output <measured-profile-json>
    python -m src.workflow execution-budget --inputs docs/validation/optimization-budget-inputs-2026-10-11.json --output <budget-result-json>

The saved budget inputs have no GPU profiles. The user reported $13.99 and
confirmed a $12 remaining-work ceiling; the ten-hour accounting start is still
unconfirmed. Complete-workflow feasibility remains UNKNOWN. Fill the inputs with measured candidate-identity
profiles and measured/bounded overhead, then replace the minimum main census with
the sufficient-pilot allocation. Never treat diagnostic screen timings as complete
production profiles or use absent evidence as zero cost.

GPU commands require separate paid authorization. `validate-optimized --scope screen`
compares both contexts and all layers/experts, records durable partial evidence and
never approves collection. `--scope full_technical --native-mechanics <receipt-4096>
<receipt-2048>` requires the complete accepted technical plans and fresh matching
native receipts. Its passing receipt authorizes only the same implementation/runtime.
The application elapsed cap includes loading but cannot stop cloud billing; the
authorized operator must enforce the lease/cash deadline independently.

Use `collect --optimized --optimized-validation <full-technical-receipt>` only after
full acceptance. Keep native calibration/probe collection. Add `--reuse-runs <sealed
predecessor-runs>` for exact nested observations; the new run keeps its own plan
and raw source links. `collect-sequence --requests <json-list>` accepts explicit
`collect` keyword arguments (`plan_dir`, `model_dir`, `output`, `mechanics_path`,
optional `calibration_dir`, `optimized`, `optimized_validation_path`, `reuse_runs`)
and reuses one verified model session. Neither command prepares plans, bypasses
calibration/receipt gates, starts a pod or changes settings.

## Historical user-run L40S screen (superseded command)

This section records the previous 72-window attempts. Its bare `--run` command
is superseded by the paired benchmark below. The user authorized one diagnostic
attempt and requested that commands expected
to exceed two minutes be run by them. Start this command from the local checkout:

```sh
python runpod/screen.py --run
```

Without `--run`, it only checks the frozen package, replacement network volume,
existing Docker Hub registry credential, account pod list and live catalog.
The paid command attaches the existing registry ID and makes one create request,
installs SSH access in the pinned runtime image, uploads a separately frozen
source candidate and paired sealed screens, stages verified original FP32 weights,
and runs the 72-window comparison. It never retries creation or launches scientific
collection. Existing pods cause preflight to stop so a running verification is
preserved. Terraform state is not used or modified.

The 30-minute lease includes setup/download/model loading/work/failure/retrieval
and teardown. The $0.75 attempt allowance uses at most $1.09/h compute plus a
conservative $0.02/h storage allowance; this is a planning bound, not an invoice
measurement or provider-enforced spending limit. Preflight refreshes stock/rate,
and the actual pod rate and placement must match before measurement starts.
The local detached guard attempts deletion before the deadline. Keep the local
computer awake and connected; this guard cannot enforce termination while the
computer/network or provider API is unavailable. Inspect Runpod immediately if
the terminal reports that teardown could not be confirmed.

Follow progress in another local terminal:

```sh
tail -F runpod/runs/latest/console.log
```

`runpod/runs/latest/` holds lifecycle/phase output, a separate `guard.log`, the
public-key-only request, attempt identity, teardown confirmation and retrieved
results. Results contain phase logs, periodic GPU observations, runner status,
durable per-expert comparisons and the numerical receipt when reached. Setup
timeouts are recorded as incomplete. Pod deletion retains network volume
`k6u810gze3`; partial evidence and downloaded weights remain there. The retained
network volume continues to accrue storage charges outside this attempt.
A passing screen is diagnostic evidence; full technical acceptance and measured
complete-workflow cost/time/fit are still required before collection.

The first user attempt failed to pull the private image because its prepared
request omitted the existing registry ID. The user interrupted it; coordinator
read-back confirmed the pod absent. No model or numerical screen started.
[Failure evidence](docs/validation/optimization-first-screen-failure-2026-10-11.json)
records the allocated 125 GB RAM/16 vCPU host and leaves billed cost unknown.
The corrected `registry-v2` source package preserves the previous package,
checks the existing credential identity and stops on decisive startup-log errors.
Registry credential validity still requires a successful private image pull;
the metadata read alone does not prove authentication.

`--max-seconds <seconds>` can reduce a newly authorized attempt's overall lease,
including setup and teardown; it cannot exceed 1800 seconds. A new invocation is
a separate user-authorized attempt. A failed attempt is not retried automatically,
and its elapsed time and any charges still count toward remaining limits.


The second user-run attempt (registry-v2, 1,200-second lease) successfully pulled
the image, verified the retained weights, and retrieved an **incomplete** diagnostic
receipt: 17/72 inputs and 408/1,728 expert comparisons, all recorded errors zero.
Only 4096-token C4/scientific inputs finished; code and 2048 remain untested. Both
attempt pods are absent on fresh API reads; the 50 GB network volume is retained.
The weights stage took 258 seconds and the numerical validator 669 seconds;
future cached runs still include file verification, model loading, setup and teardown.
These timings are diagnostic observations, not complete production profiles.

The old numerical receipt records `image_digest: null`. A versioned `image-identity-v3`
source package now explicitly propagates the frozen image declaration through SSH
to child processes, rejects conflicting identities, and checks the package image
against the controller's pinned request. This is a declaration tied to controller
read-back, not in-container attestation. Prior archives and raw receipts remain intact.
The v3 package preserves the original 72-window diagnostic; it does not prepare
the smaller balanced successor proposed in the postmortem. No new paid invocation
is authorized by this local fix. See [partial results and bounded next proposal](docs/validation/optimization-l40s-partial-results-2026-10-11.json).


## Paired collector benchmark — prepared locally, not launched

The first fresh finding is implemented: optimized measurement now uses its
mandatory uninstrumented exact cleanup as the native reference, saving one full
forward per input. Capture and cleanup still execute the entire native input;
all eight expert effects, natural downstream routing, frozen FP32 tolerances,
and 4096 primary / matched 2048 sensitivity remain. A failed exact cleanup returns
no complete events. The v2 executor needs its own full technical receipt before
scientific collection; old receipts and this diagnostic cannot authorize it.

The controller now defaults to immutable package
`data/optimization/paired-collector-2026-10-11-v6`. It contains the new collector,
its paired adapter, unchanged sealed parent plans, and a separate exact archive
of the preceding collector. Six fixed windows cover C4, scientific and code at
both contexts. Each window compares old/new collection at `[15,0,7]` and `[15]`,
with all eight effects and alternating AB/BA order. Two paired all-layer warmups
are accounted separately; six independent native cases check 144 expert effects.
There are 12 measured pairs / 48 measured raw events plus 12 warmup raw events.
Incomplete pairs are logged and excluded from elapsed-ratio summaries.

Read-only package/account/stock/rate preflight (no pod creation):

```sh
python runpod/screen.py
```

Only the user starts the paid command expected to exceed two minutes. Reconcile
actual remaining money within the $12 work allowance and seconds within the
original ten-hour allowance first; prior downloads, failed attempts, elapsed
wall time and storage still count. This command prompts for those two numbers:

```sh
read -r 'dsic_cash?Remaining reconciled work allowance (USD): '
read -r 'dsic_seconds?Remaining reconciled ten-hour allowance (seconds): '
python runpod/screen.py --run --remaining-cash-usd "$dsic_cash" --remaining-wall-seconds "$dsic_seconds"
```

The `read` syntax above is for the user's zsh. The command makes exactly one
attempt, with no automatic retry. Its total lease is at most 1,200 seconds,
further reduced by remaining wall time and
`3600 * (min(0.40, remaining_cash) - fixed_cost_bound) / (live_compute_rate + 0.02)`.
The default fixed-cost bound is $0.03 (override upward with
`--fixed-cost-bound-usd` if needed); this and $0.02/hour storage are explicit
planning assumptions. At the historical $1.09/hour compute quote, 20 minutes
of compute plus storage is $0.37. Preflight refreshes compute price and stock.
It checks that no account pod is present and preserves existing resources.
The pod uses the retained L40S-region volume and cache; missing/corrupt weights
stop the attempt, without any model download. Hashing, loading, warmups, native
checks, actual collection, durable persistence, retrieval and teardown all count.
Phase caps are planning limits, not proof the benchmark will finish.

Keep the local computer awake and connected. The local detached guard attempts
owned-pod deletion before the deadline; there is no provider hard TTL, and an
API/network failure can overrun the intended budget. Confirm teardown in the
console and inspect Runpod immediately if it cannot be confirmed. The retained
network volume continues billing outside the attempt lifetime.

```sh
tail -F runpod/runs/latest/console.log
```

Progress includes setup/weights phases, arm starts/completions, each native expert
comparison and paired comparisons. The attempt results retain append-only,
fsynced raw events, timing and progress JSONL, phase logs, GPU observations,
runner receipts and the diagnostic summary. Timing scopes distinguish numerical
collection, event persistence and full arm duration; warmup and validation overhead
are separate. Source/runtime/plan identities accompany the result. A bounded stop
is INCOMPLETE, a discrepancy is FAIL; neither is a scientific result.

Local integration: [evidence](docs/validation/optimization-cleanup-reference-integration-2026-10-11.json).
Benchmark specification: [six-window proposal](runpod/paired-benchmark.json).
Full-workflow $12/ten-hour feasibility remains UNKNOWN. Guarded downstream expert
reuse stays a later profiling/prototyping task; GPU speedup is not yet measured.


The first paired attempt (v5, 2026-10-11 04:19 UTC) stopped before measurement:
boot/SSH took about81s of the shared120s setup limit, leaving about39s for
upload. Local teardown confirmation records the pod deleted and volume retained.
It produced no collector timing or native comparison. The v6 successor reallocates
phase limits to startup/upload/preflight240s, cached verification180s, model load90s,
warmup60s, native checks240s, collector+persistence240s, finalization30s and
retrieval/teardown120s (total1200s). Phase and overall countdowns are distinguished
in console output. This fixes the overly tight setup allocation; upload throughput
and successful completion remain unmeasured. No paid retry is started by this fix.
The elapsed attempt still counts toward remaining money/time; nominal hourly
compute+storage for its131.17s was about$0.04045, not a measured invoice amount.

# Fresh optimization findings — 2026-10-11

**Findings only. No implementation, paid work, resource action or research amendment. Complete-workflow feasibility: UNKNOWN.**

Implement the redundant-reference consolidation first. Investigate guarded reuse of complete downstream expert batches second, after profiling. The current final-layer suffix is already cheap; its shared full forwards are the larger remaining cost. Early-layer suffixes still repeat substantial full-window work. Neither candidate has a measured production speedup.

The machine-readable companion is [fresh-optimization-findings-2026-10-11.json](fresh-optimization-findings-2026-10-11.json). It includes exact source/raw hashes, individual benchmark window identities, operation counts, candidate changes and acceptance tests.

## Investigation boundary and evidence order

I read the handoff and current README, configs/research.json and WORKFLOW, then investigated source and the exact completed attempt **before** reading EXECUTION-BUDGET-PLAN, OPTIMIZATION-TODO, optimization-next-direction and supporting reviews. I used only this checkout's scientific authority. No on-disk AGENTS.md exists; the supplied AGENTS instructions apply. No existing Makefile or agent snapshot scripts were found. Existing logs, receipts, source and tests were sufficient; no diagnostic infrastructure was created.

The initial `dsic3007-fresh` graph had obsolete collector signatures. I refreshed its fast index without filesystem persistence and used caller traces, checking relationships against source. Some generic `get`/`add`/`append` graph edges still resolve to unrelated symbols; those edges were discarded. Current source is authoritative.

Current implementation SHA-256 is `f7a7e92c7ac5c92368467da986ea413bffb33b1cda8665b903bea32b8a1b5d62`. All seven requested implementation files match the frozen registry-v2 source archive byte for byte. The installed Transformers OLMoE source matches the L40S receipt hash `498ab5720463327ab7235f3a3103aefa49d2f93da10c3cfb7ef7002774b05531`. The checkout is already extensively dirty: the recent tracked history does not describe the integrated implementation. No source edits are proposed as completed changes here.

Preserve exactly 4096 original input tokens, final predictor with the real next-token target outside the window, accepted matched 2048 sensitivity, all eight independent expert effects, natural downstream routing, original FP32 weights, eager execution, deterministic operations, TF32 off, batch one and no KV cache. Keep all source groups, accepted sampling/statistics and numerical tolerances. Mixture/logits use atol=1e-6, rtol=1e-5; NLL absolute error is at most 1e-6 nats. Routing IDs/coefficient bytes, survivors, local untargeted rows and cleanup remain exact. No frozen package, plan, raw receipt or running verification was changed.

Evidence labels used below:

- **Measured diagnostic:** executed native L40S comparison measurements, restricted to completed cases and their actual instrumentation.
- **Source verified/inferred:** operations verified in current code, or counts calculated from those operations and fixed inputs.
- **Conditional/assumed:** transfers of diagnostic timings, hypothetical cache hits or sizing examples; these are not production measurements.
- **UNKNOWN:** missing measurement, acceptance or accounting evidence.

## What the L40S attempt actually measured

Use [the named attempt](../../runpod/runs/dsic-screen-20261011T010215-fb295d6f/), never a moving `latest` link. Principal evidence is its [validation receipt](../../runpod/runs/dsic-screen-20261011T010215-fb295d6f/results/validation.json), [case journal](../../runpod/runs/dsic-screen-20261011T010215-fb295d6f/results/validation.json.cases.jsonl), [phase receipt](../../runpod/runs/dsic-screen-20261011T010215-fb295d6f/results/runner-receipt.json) and [console](../../runpod/runs/dsic-screen-20261011T010215-fb295d6f/console.log).

The attempt completed 17/72 inputs and 408/1728 comparisons: 12 C4 and five scientific inputs, all at 4096 tokens. Each completed input has 24 comparisons, eight at each layer 0/7/15. All recorded mixture/logit/NLL errors are zero; completed cleanup and parameter checks pass. Input 18 stopped after approximately 1.860 seconds, before any recorded expert comparison. Code and 2048 are unmeasured. The receipt is INCOMPLETE, diagnostic-only, with empty acceptance checks and `full_coverage=false`. Its contexts list describes intended coverage, not completed coverage.

I reconciled 408 unique progress entries with the 408 completed case-array entries without counting both copies as separate observations. The raw case SHA-256 agrees with the receipt. The old runtime `image_digest` remains null. The future helper's declared-image propagation cannot upgrade or repair that old evidence.

Measured timings, recomputed from the 17 completed inputs:

| Operation | Unit and n | Mean seconds | Observed range |
|---|---|---:|---:|
| Native captured reference baseline | per input, 17 | 0.943161 | 0.910785–1.321514 |
| Native-shape contribution baseline | per input, 17 | 0.934311 | 0.918292–0.940241 |
| Plain full cleanup | per input, 17 | 0.918390 | 0.912564–0.926075 |
| Native ablations, layer 0 | eight experts per input, 17 | 7.735350 | 7.602931–7.952451 |
| Native ablations, layer 7 | eight experts per input, 17 | 7.536960 | 7.376288–7.670222 |
| Native ablations, layer 15 | eight experts per input, 17 | 7.348392 | 7.246762–7.401804 |
| Optimized suffix, layer 0 | eight experts per input, 17 | 7.292339 | 7.135554–7.453612 |
| Optimized suffix, layer 7 | eight experts per input, 17 | 3.900031 | 3.796181–4.004480 |
| Optimized suffix, layer 15 | eight experts per input, 17 | 0.019551 | 0.019436–0.019659 |
| Complete comparison engine | per input, 17 | 37.205453 | 36.947615–37.845228 |

Per expert there are 136 observations at each layer. Layer-15 suffix mean is 0.002444 seconds, versus 0.918549 seconds for its diagnostic native ablation. These are component observations, not an end-to-end collector speedup ratio.

Summed comparison-engine time is 632.493 seconds. Its explicitly timed forwards/suffixes sum to 622.684 seconds, leaving 9.808 seconds inside the input engine for other checks, setup, duplicate recomposition and progress writes. **This residual does not isolate total diagnostic overhead:** downstream route copies/packing and capture work are inside timed forward blocks. Completed-case first persistence writes sum to 0.230 seconds; this omits per-expert progress fsyncs and each timing-record's own write. It is not scientific journal persistence timing.

Observed diagnostic maxima are 30,875,910,144 GPU allocated bytes, 33,067,892,736 reserved bytes and 28,602,888,192 host peak RSS bytes. There are 78 nvidia-smi samples; the largest observed used memory is 32,075 MiB. These incomplete observations do not bound whole-device/host peaks for code, 2048, calibration, a new cache or the complete collector.

### Separate lifecycle phases

| Phase | Evidence | Interpretation |
|---|---:|---|
| Weights command | 258.071 s | Acquisition plus per-file SHA-256 and `verify_model`'s second full verification. The log confirms original FP32 bytes, 27,679,451,403 expected asset bytes. Download and hash time are not separately instrumented. |
| Model acquisition inside validator | approximately 33.8 s | Log starts at elapsed 0.2 s and completes at 34.0 s. Includes another `verify_model` hash pass, dtype/config checks and loading. Hashing and model placement are not separately timed. |
| Validator subprocess | 670.280 s | Includes preflight/model acquisition and partial measurement. Receipt elapsed is 668.949 s; neither figure should be added to its component timings. |
| Completed comparison engine | 632.493 s | 17 inputs; includes diagnostic checks/progress. Excludes initial model acquisition and outer completed-case persistence. |
| Local attempt | 1094.191 s | Provision/startup/upload/work/retrieval/teardown. Remote runner elapsed is 934.925 s; their 159.266 s difference is unclassified lifecycle overhead, not pure teardown. |
| Retrieval and teardown tail | about 27 s | Console retrieval starts 01:20:02 UTC; deletion confirmed 01:20:29. Not a separately instrumented phase timer. |

The named attempt's cleanup receipt confirms deletion with the volume retained. Saved billing readback at 01:27 UTC records both attempt pods absent. I made no fresh cloud read, so those are historical snapshot assertions, not a current account-state claim. The verified network-volume cache avoids future downloading; verification, loading and future storage still cost time/money. Past elapsed/billed work remains spent.

## Actual collector accounting and consumers

For one physical input with L requested effect layers, source gives:

| Path | Full model forwards | Suffix branches | For layers 0/7/15 |
|---|---:|---:|---|
| Native scientific effects | 2 + 8L | 0 | 26 full forwards |
| Current optimized scientific effects | 3 | 8L | 3 full forwards + 24 suffixes |
| Proposed rank-1 optimized effects | 2 | 8L | 2 full forwards + 24 suffixes |
| Native calibration/probe | 2 | 0 | One baseline and one exact cleanup |
| Diagnostic `_input_case` | 3 + 8L | 8L | 27 full forwards + 24 suffixes |

The collector's three optimized full forwards are at [measure_input](../../src/run_experiment.py#L113): pristine reference at lines 134–135, contribution-captured baseline at 136–139, and plain cleanup at 181–186. Calibration retains all input-position routes and lossless FP32 coefficient bytes; probe retains predictor routing. Both remain native. Every effect branch starts from immutable baseline data and keeps eight raw ablated NLLs and deltas.

[_replay_from_layer](../../src/run_experiment.py#L95) rebuilds the targeted local mixture, adds the stored residual, executes **full-shape** downstream decoder layers, then final norm and the last-token LM head. Layers 0/7/15 invoke respectively 15/8/0 decoder layers per branch: `8*(15+8)=184` downstream decoder calls per three-layer physical input. Layer 15 already avoids downstream decoder execution; making it a token-only expert GEMV would introduce numerical risk where little measured suffix time remains.

### Checks, synchronization and copying

- Collector CUDA synchronization sites are start, baseline completion, every expert completion, cleanup completion and the availability-based final synchronization: `4+8L`, or 28 for three effect layers, 12 for primary-only and four for routing-only. NLL `float()`, finite checks, `torch.equal`, routing `tolist()`, dynamic `where/nonzero` and tensor-dependent Python decisions add implicit synchronization; explicit counts are not total waits.
- `forward` creates device IDs/mask each full call, uses only the final predictor logits, casts them to FP64 and computes actual target log-softmax. `_replay_from_layer` also converts its NLL to a Python float. Loss arithmetic remains unchanged.
- [RoutingCapture._before/_after](../../src/interventions.py#L51) checks complete hidden/weights/output finiteness, clones predictor IDs/weights/mixture, and transfers whole-window routes only when requested. Native ablation clones weights and exactly checks locality/survivors. Calibration's `capture_window` also clones a full mixture for which collection does not currently serialize a field; this is a profiling target, not a measured bottleneck.
- Contribution capture retains hidden/IDs/weights references, full mixture clones, eight predictor expert outputs and residual references for selected layers. `_native_forward_with_capture` preserves native batched GEMMs and ordered `index_add_`. `recompose_for_target` clones the full mixture per expert and rebuilds the predictor from zero. It has small allocations and tensor-dependent branch synchronization. Collector suffixes have no downstream route-to-CPU packing hooks.
- The diagnostic additionally saves whole-window mixtures/routes for baseline and each native ablation, concatenates untargeted rows for exact comparisons, and calls recomposition twice per expert: once for checking and again inside replay. It packs full-window downstream routes via `_routes` for both paths: `2*8*(15+8)=368` calls/input, 6256 for the completed 17 inputs. Each call copies IDs/weights to CPU and packs coefficient bytes through Python lists/struct. This is absent from scientific suffix collection.
- Diagnostic `synchronized_clock` is called 104 times per complete three-layer input, plus the memory-metric synchronization and implicit waits. `_record` does 24 expert progress fsyncs, one complete-case fsync and one persistence-timing fsync: 26 per completed input. Durable expert progress must remain available; reducing validation scope is not a production optimization.

**Measured attribution still missing:** attention versus experts, expert batches/row distribution, scalar waits, each transfer/clone and production serialization/persistence. There is no justified percentage speedup from removing any one of those operations.

### Durable collection and timing interpretation

[collect](../../src/run_experiment.py#L215) checks sealed inputs/receipts/freezes, acquires a model, identifies the executor, streams token windows and handles resume/reuse. It calls `measure_input` only for missing layers. [Journal.append](../../src/artifacts.py#L231) validates and flushes/fsyncs each complete event; a three-layer miss gets three event writes and one [input timing sidecar](../../src/run_experiment.py#L393) write. Primary-only gets one of each. Cache hits still validate/read/copy/persist events; a fully reused input does no measurement forward but current `collect` acquires a model before processing hits. Failures append a durable incomplete record and propagate. Final sealing hashes every bundle file and atomically persists the manifest.

Shared baseline, cleanup and input-total values repeat on layer events. Count each **once per physical input**, plus each layer's eight-intervention time once. `measure_input` total excludes Journal persistence. The outer input sidecar total includes event persistence/reuse but is sampled **before its own sidecar fsync**. `persistence_seconds` measures event append time only. Bound sidecar writes and final sealing separately; do not assume the sidecar measures every durable write.

[execution_budget._unique_input_timings/_input_seconds](../../src/execution_budget.py#L40) deduplicates shared work and preserves unexplained elapsed time. [profiles_from_measurements](../../src/execution_budget.py#L466) requires verified sealed measurements and excludes cached/partial profiles. [forecast_execution_budget](../../src/execution_budget.py#L306) requires cohort profiles and explicit overhead/accounting, uses cohort component maxima and keeps UNKNOWN blockers. Never turn the validation JSONL into a production measurement bundle or fill unknown cost with zero.

Consumer trace verified against source: `workflow.main` calls `collect`, `collect-sequence`, validation and profile/forecast functions; `collect` calls `measure_input`, `SealedMeasurementReuse` and model acquisition; `measure_input` and `_input_case` share suffix replay; reuse calls `raw_observation_key` and compatibility/lineage builders. Calibration and offline analysis consume the raw event schema. [planning.plan_main](../../src/planning.py#L64) reads the existing shared/effect timing aliases, counts primary/secondary positions and includes C4; keep those aliases and shared-work semantics.

Native mechanics is a further **acceptance cost**, not collecting: [verify_native](../../src/verify.py#L24) performs 13 full forwards per input/layer (baseline, two repeats, noop, eight ablations, cleanup), plus ten independent expert reconstruction loops. Three layers cost 39 full forwards and 30 reconstruction loops/input. The paired full technical plans have 304 inputs, implying 11,856 full forwards and 9120 reconstruction loops before optimized validation. Missing verified acceptance time cannot be budgeted as zero.

### Current main workload, independently recomputed

I checksum-verified the combined prepared bundle and streamed its current document metadata, applying current source priority ordering and the approved review. This reproduces the **minimum candidate**, not a sufficient-pilot allocation or main freeze:

| Domain | Documents | Groups | P: primary windows | Q: windows at each secondary layer |
|---|---:|---:|---:|---:|
| Scientific | 200 | 200 | 12,755 | 3200 |
| Code | 205 | 200 | 12,957 | 3278 |
| C4 | 200 | 190 | 12,584 | 3199 |
| Total | 605 | 590 | 38,296 | 9677 |

Q windows contain all three layers; P−Q contain layer 15 only. Current main ordering is `[15,0,7]`. Source counts are:

- Native: `10P + 16Q = 537,792` full model forwards.
- Current optimized: `3P = 114,888` full forwards, `8(P+2Q) = 461,200` suffix branches and `184Q = 1,780,568` downstream decoder calls.
- Rank 1: `2P = 76,592` full forwards; the same suffix work. Saving: 38,296 full forwards.

These counts preserve all effects and shared work. Existing exact nested observation reuse and one verified model session are already implemented; their benefit is not a new optimization result. Observation reuse is not permission to merge different windows, contexts, source stages or executors. Code group sizes and actual eligibility matter; do not substitute 200 files for 200 groups.

## Ranked optimization 1 — consolidate native reference with mandatory cleanup

**First implementation slice; low to medium control-flow/evidence risk.** Locations: `src/run_experiment.py::measure_input` lines 134–146 and 179–202; collection persistence/timing lines 367–418. Preserve independent `src/optimized_validation.py::_input_case` lines 183–217 and 311–327.

For optimized effects, remove the first plain reference forward. Keep the full native-shape contribution baseline and all independent suffix branches. Run the existing final plain full cleanup and use it as the independent native reference for the captured baseline. Preserve `cleaned_nll == baseline` and `torch.equal(cleaned_logits, captured_logits)`, including the exact failure-before-return/persistence rule. The final exact comparison is stronger than the removed early tolerant predictor check. Do not replace exact cleanup with tolerant equality or omit the full cleanup forward.

Only detection timing changes: a bad captured baseline is rejected after branches instead of before them, still before any complete event becomes durable. Preserve the native/calibration/probe paths and all model arithmetic. The validator keeps two independent baselines and separately executed native ablations; its diagnostic overhead is deliberate.

Retain `shared_seconds`, cleanup, per-layer effects, physical input total and legacy aliases used by planning. Candidate `shared_seconds` measures the remaining captured baseline/setup; final reference time belongs to cleanup. Give the candidate a distinct execution mode and fresh executor receipt identity. Peak memory may fall slightly because independent reference logits/one lifetime disappear, but captured tensors remain; no fit gain is established.

**Expected impact label: source count, with conditional timing illustration only.** One full forward saved/input. Transferring the observed diagnostic components would change primary-only work from approximately 2.815 to 1.872 seconds, and three-layer work from 14.008 to 13.065 seconds. That is about 33.5% and 6.7% less component time respectively. Those are unverified transfers, not measured collector gains.

Acceptance tests for the coordinator:

1. Extend tiny packed 16-layer fixtures to compare native, preserved current optimized and candidate outputs for `[15]`, `[0,7,15]` and actual main ordering `[15,0,7]`: event IDs, routing/FP32 coefficient hex, baseline NLL and all eight ablated NLL/deltas. Exclude timing/resource differences. Count exactly two full candidate forwards and eight suffixes/requested layer; native/calibration/probe counts remain unchanged.
2. Inject a captured-baseline/logit discrepancy, including one inside the tolerant comparison bound: exact final cleanup must reject with **zero Journal.append calls for that failed input**. A cached failure cannot become a completed zero effect.
3. Inject baseline expert failure, seventh suffix failure, nonfinite output and final cleanup failure. All expert overrides, routing hooks, layer/residual hooks must be removed; parameters unchanged; the next plain forward recovers baseline. Keep partial-layer resume, torn-tail/duplicate rejection, writer-lock cleanup and immutable sealed outputs.
4. Preserve current sidecar-failure behavior: already fsynced raw events remain, missing timing stays UNKNOWN on resume. Verify shared work is deduplicated and raw reuse/executor/source/context/target mismatches fail closed.
5. Run the bounded diagnostic proposed below, then obtain fresh exact-executor native mechanics and complete full-technical optimized acceptance at both contexts before collection. CPU fixtures and a six-window benchmark cannot satisfy those gates.

The edit changes `implementation_identity`, which invalidates old native/optimized receipts, session/reuse identities and executor-bound freezes. Keep baseline code in an immutable sibling package for comparison. Do not mutate a running session, relabel old measurements or reuse old calibration under the new identity.

## Ranked optimization 2 — guarded downstream expert-batch reuse

**Conditional follow-up; high numerical/memory risk, unmeasured benefit.** Locations: `src/run_experiment.py::_replay_from_layer` lines 95–110; `src/interventions.py::RoutingCapture._native_forward_with_capture` lines 91–112, recomposition lines 114–155 and cleanup lines 203–215. Native reference: pinned `OlmoeExperts.forward` lines 314–338. Validator: `_input_case` lines 228–321.

Eight layer-0/7 suffixes measured 7.292/3.900 seconds versus 0.0196 seconds at layer 15. They execute the complete original token sequence through 15/eight downstream layers. This identifies a larger remaining work class, but it does not reveal how much belongs to experts versus eager attention or dispatch.

Profile first. Then prototype **one downstream MoE layer, initially layer 8 after a layer-7 intervention**, with per-input immutable baseline expert-batch outputs. Continue full-shape native eager attention and natural routing in every branch. Reuse an unweighted expert output only if its **complete input matrix, ordered row map, native batch shape and parameters are exactly unchanged**. Otherwise recompute the entire current native expert batch. Apply current branch coefficients and reconstruct the complete mixture from zero in native expert/scatter order. Do not reuse a weighted output under changed weights, subtract from a finished mixture, freeze routes, skip zero multiplication, or switch to predictor-only GEMV.

Mathematically only the final predictor can affect downstream predictions here. **That does not prove bitwise prefix invariance.** A changed predictor active set changes expert batch sizes; native GEMM shape changes can alter prefix rounding, propagate to later layers and defeat reuse. Guards must compare actual dependencies, with conservative misses/recomputation. Cache hit rates, guard/copy/scatter time and recomputed rows must be recorded. If guards fail or erase gains, retain the existing suffix executor.

Conditional opportunity: when prefix bytes and batches are unchanged, experts outside the union of baseline/current predictor active sets receive no changed predictor row. That union contains at most 16 of 64 experts. This suggests fewer expert GEMMs only when those experts are hit and checks pass; expert loads/batch sizes are uneven and eager attention remains. It is not a 4× latency claim.

This is an **expert-output cache within one input**, with `use_cache=False`, `past_key_values=None` and full context; no attention KV cache is proposed. Retain outputs with exact row/input dependencies, scope cache lifetime to one window, and install overrides only on candidate suffixes. Independent native comparison must never use those overrides. Expand to multiple layers only after local/GPU agreement and measured benefit.

Memory cost for unweighted outputs is approximately `4*W*8*H` bytes/layer, plus input snapshots/maps, residuals, masks and temporaries. As an explicitly conditional sizing example, H=2048 and W=4096 need 256 MiB/layer: 2 GiB for eight layers or 3.75 GiB for 15 layers **for outputs alone**. H=2048 was not independently verified from the pinned configuration here. Existing partial peaks do not establish headroom for this cache. Profile/release memory before considering a broader prototype.

Acceptance tests:

- Stable/changed predictor membership, changing expert batch row counts, predictor-only experts, ties, reordered/non-contiguous batches and repeated immutable branches. Compare complete native-shape outputs and full event payloads; exact routes/coefficients and frozen mixture/logit/NLL tolerances.
- Perturb an earlier hidden row, row mapping, shape or coefficient. Demonstrate a guarded miss/recompute or fail closed; no stale hit. Test mixed hits/misses and native/calibration/probe cache exclusion.
- Inject failures in capture, guard, GEMM, scatter and later decoder execution. Remove every override/hook, preserve parameters, release buffers and recover native baseline before any failed event persists.
- Separate full native acceptance for this candidate across both contexts/all domains/all eight experts, including natural downstream routes versus native ablations. Rank-1 acceptance cannot approve it. Retain only after actual total collector improvement beyond observed variation and measured positive memory headroom.

## Identity contracts and implementation boundaries

[implementation_identity](../../src/model.py#L75) hashes model, interventions, collector, native verification, artifacts, reuse, observation contract and optimized validation source. [VerifiedModelSession](../../src/model.py#L82) rejects changed implementation/runtime/catalog/directory/seed or parameters. Collection requires exact native-mechanics executor identity; optimized collection additionally requires a complete full-technical receipt matching execution mode, contexts/layers, source review, technical plans, complete checks and frozen tolerances. A screen or incomplete pass cannot authorize collection.

[raw_observation_key](../../src/observation_contract.py#L46) and [SealedMeasurementReuse](../../src/reuse.py#L155) bind model/tokenizer, executor/mode/device, dataset/source/content/token/window/predictor/target/layer/operator and stage. A rank-1/rank-2 candidate cannot treat previous-executor measurements as exact cache hits. Preserve fresh consumer sampling metadata, immutable source references and raw schema. Do not count historical source timing as newly measured execution. Calibration and main freezes bind executor too: settle the production executor before calibration/pilot, or handle any compatibility decision explicitly without rewriting old evidence.

First concrete slice: rank 1 scheduling + distinct candidate mode/receipt plumbing + differential/failure regressions. Do not combine it with expert cache, numerical kernel changes, validator simplification or I/O durability changes. A paired GPU diagnostic should measure this slice before adding rank 2.

## Budget assessment and reconciliation with earlier hypotheses

Complete $12/ten-hour feasibility remains **UNKNOWN**. Current wallet, reconciled unspent work allowance and clock start are unknown. The saved $13.99 user report is not a live balance. Saved attempt billing queries returned zero records; that does not establish zero charges. The volume charge of $0.0048611113 covers one saved hour only and does not capture all ongoing storage. Do not deduct prior snapshots twice or assume the entire ceiling remains.

Recomputed illustrative formula, explicitly assuming diagnostic components transfer to all main domains and windows:

`P*(reference + captured baseline + cleanup + eight layer15 suffixes) + Q*(eight layer0 + eight layer7 suffixes)`

With P=38,296 and Q=9677 this gives approximately **60.035 hours**, or **50.002 hours** after rank 1, before other stages/overheads. Under that same transfer assumption, rank 1's two shared full forwards alone are approximately 19.709 hours even if suffix work vanished. These figures show which costs require measurement/improvement; they are neither measured production forecasts nor universal lower bounds. The diagnostic engine's 37.205 seconds/input must not be substituted into this formula.

Missing: actual three-domain/two-context collector profiles and both layer-set patterns, calibration/probe serialization, reuse/read/write costs, setup/hash/load/transfer/seal/teardown, full native acceptance, host waits/storage, sufficient-pilot allocation, matched sensitivity, analysis/reporting and reserve. `forecast_execution_budget` must remain fail-closed until these are measured/bounded and accounting is reconciled. No profile here proves scientific precision or powers the final sample.

Reconciliation after independent investigation:

- O18/O19/O20 are supported: redundant reference consolidation, unchanged full-window suffix work and a paired actual-collector benchmark. The previous 50-hour illustration omitted a real forward; the corrected 60-hour illustration still relies on assumptions.
- EXECUTION-BUDGET-PLAN's original findings table describes pre-integration source: current `plan_main` includes C4 and the current validator already compares native ablations. Use current source rather than those historical rows.
- O10's rejection records no demonstrated material benefit, not proof that expert reuse is impossible. Rank 2 stays conditional on new profiles and exact dependency checks.
- Session reuse, exact nested observation reuse, streaming/lazy journals and coefficient hex are already implemented. Do not count them again as new wins.
- Full-tensor checks, scalar synchronization, unused calibration mixture copies, diagnostic route packing and CPU preparation merit attribution if profiles identify a bottleneck. None has enough isolated evidence to outrank the two candidates. Keep per-expert diagnostic durability and existing failure gates.

## One proposed paired GPU benchmark — not authorized

Target rank 1 versus the **preserved current optimized collector**, with an independent native numerical reference. Maximum **1200 seconds / $0.40 across every phase**, further reduced by reconciled remaining cash/time and fresh all-in pricing. No launch occurred. This is a diagnostic, not collection approval.

The JSON fixes six input IDs: one deterministic position from the first sealed technical group/domain, with the exact parent-matched 2048 input. Round-robin order is C4-4096, scientific-2048, code-4096, C4-2048, scientific-4096, code-2048. Preserve original complete plans; select a separate diagnostic view. For each input measure both actual main layer shape `[15,0,7]` and primary-only `[15]`, every active expert. Alternate baseline/candidate execution order. Use one paired all-layer warmup per context and account it separately. One measured pair per cohort/shape is a bounded diagnostic, not a precise statistical speedup estimate.

Run the existing independent `_input_case` on all six windows: 144 native comparisons. Compare actual candidate event outputs to its independent native baseline/routes/individual ablated NLLs as well as to current collector outputs. Keep pristine native paths free of candidate overrides. Never forge acceptance receipts to call `collect --optimized`; use a diagnostic adapter invoking the real measurement functions and existing durable-write primitives.

Source operation ledger for the whole proposal:

| Component | Full forwards | Suffix branches |
|---|---:|---:|
| Measured current/candidate pairs, both layer shapes, six inputs | 60 | 384 |
| Independent native diagnostic, six inputs | 162 | 144 |
| Paired warmup, one all-layer input per context | 10 | 96 |
| Total | 232 | 624 |

Measured pairs produce 48 raw layer events. Time hook/setup, baseline capture, final reference/cleanup, each layer's eight effects, event serialization/fsync, sidecar write, synchronized complete collector invocation and all lifecycle phases. Persist raw route bytes, baseline/eight ablated NLLs/deltas, individual discrepancy maxima, hook/parameter integrity, memory/telemetry and immutable baseline/candidate/plan/runtime identities. Use unique output paths, append-only progress and existing Journal/`append_timing_row`/`_record`/atomic-receipt patterns. Mark diagnostic profile scope honestly; sealed scientific measurement acceptance is still pending.

Phase **caps**, not duration predictions:

| Phase | Seconds |
|---|---:|
| Provision/startup/upload/preflight | 120 |
| Cached weight verification | 180 |
| Model hash/load/identity | 120 |
| Paired warmup | 90 |
| Independent native checks | 240 |
| Actual paired collector plus durable persistence | 300 |
| Finalize receipts | 30 |
| Retrieve and confirm teardown | 120 |
| Total | 1200 |

Reuse retained cached weights after required catalog/hash verification; missing/corrupt cache stops without downloading. Hashes repeated by existing acquisition/loading paths must be counted where they execute. Use the pinned runtime and reviewed future image declaration; old null-image receipts remain untouched. Proposed hardware is one L40S in the retained volume's region with host RAM at least 64 GB. Before launching, read current placement/rate and any existing verification; do not interrupt or compete with that work.

Historical $1.09/h compute plus a planning $0.02/h storage allowance gives $0.37 for 20 minutes, leaving $0.03 for other bounded charges. It is not a live quote or an invoice. Let fixed cost bound be C and total hourly burn R. Require:

`T <= min(1200, remaining_wall_seconds, 3600*(min(0.40, reconciled_allocated_cash)-C)/R)`

Unknown/nonpositive limits stop preflight. Start the lease before creating a resource; reserve retrieval/deletion time before measurement. Existing detached guard/operator must enforce deletion confirmation within the absolute deadline: an application timer alone does not stop billing. One create/attempt, no automatic retry. Stop on discrepancy/nonfinite, identity/fit failure, competing verification, cap or phase overrun threatening teardown. Preserve incomplete pairs and all spent time; never zero-fill or include unmatched pairs in speedup aggregates.

**Preparation needed before paid approval:** current `screen.py` and image-identity-v3 still package/run the old 72-window validator. They do not implement this paired collector benchmark. The coordinator must prepare/review a small diagnostic adapter and immutable successor packages, with exact baseline/candidate binding and all-phase guard. No adapter was built here, and no runnable paid command is claimed ready.

Remaining gates: fresh exact-executor native mechanics at both contexts; complete accepted full-technical native-versus-candidate coverage; actual whole-device/host fit and missing code/2048 profiles; measured full-workflow forecast including escalation/reserve with reconciled cash/time; matched calibration/probe stability, sufficient pilot and exact main freeze; explicit authorization for paid benchmarking/collection. This benchmark does not remove any gate.

## Checks actually run

- Existing local suite: `timeout 110s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/dsic3007-py314-test-deps python -m pytest -q -p no:cacheprovider` — **142 passed in 19.02 seconds**. CPU/mocked implementation checks; neither candidate has been implemented or tested.
- Raw JSON/JSONL aggregation, completed/progress count reconciliation, case/receipt hashes, seven-file frozen archive/source comparison and installed Transformers-source identity check passed.
- Read-only sealed-screen verification and combined prepared-input checksum verification passed; minimum-main census reproduced under current priority ordering.
- Output JSON/Markdown and protected-file integrity checks are recorded in the machine-readable companion. No GPU benchmark, native acceptance, remote process inspection or cloud/account action was performed.

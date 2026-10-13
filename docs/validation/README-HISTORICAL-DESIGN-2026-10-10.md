# Historical README design — 2026-10-10

Archived from the root README on 2026-10-13. This records the earlier design
and inventory snapshot; it does not define current Path B settings or readiness.
Use the [current README](../../README.md) and
[Path B settings](../../configs/research-path-b-v1.json) for the active study.

## Historical amendment — fresh-clone decisions, 2026-10-10

Arya explicitly selected the following rules for the fresh implementation.
These replace the conflicting context, calibration, pilot, and bootstrap counts
in the older summary below. Missing local documents and old code are not
authorities for this restart. Machine-readable decisions are in
[configs/research.json](../../configs/research.json); commands and limits are in
[WORKFLOW.md](../../WORKFLOW.md).

- Master seed **3007**, with separate deterministic stage streams. Calibration
  position seeds are **3008, 3009, 3010**.
- Primary inputs contain **exactly 4096 original input tokens**. The predictor
  is the last input token, with **4095 preceding tokens**; the actual next-token
  target is outside the input. Exclude positions without full context and target.
- Source groups are C4 hostnames, scientific paper IDs, and code repository
  IDs. Check duplicate content, article variants, repository aliases/known
  clones, and group overlap. Technical, calibration, probe, pilot, and main
  partitions are disjoint by source group. Reserve **12 technical groups/domain**.
- Fixed probe: **32 documents/domain, 4 positions/document**. Calibration uses
  the same nested document subsets across three seeds, at **100, 250, 500
  documents/domain**. Each document contributes one independently sampled full
  window per seed; record native routing at **all input positions**. Utilization
  is membership frequency per position, summing to eight over experts. Average
  documents equally, then average the three seed tables. M1 mixes the three
  domain tables equally; M2 uses the matched domain table.
- Stability uses **L1 drift ≤0.05** over all 64 utilization values, without
  dividing by eight, and both **top-8 and bottom-8 Jaccard ≥0.75**, passing in
  **at least 90%** of domain/layer comparisons across seeds and consecutive
  budgets. Freeze 250 after a passing 100→250 comparison, otherwise 500
  after a passing 250→500 comparison. Each candidate seed-pair and averaged-table budget comparison must
  meet the 90% rule; predecessor seed stability is not required. Keep seed/budget tables, frozen rankings, and probe selection
  agreements; report instability when no budget qualifies.
- Pilot: nested **20→40→80 scientific/C4 documents and whole code groups**, with
  **2 positions/document**. Extend if fewer than **10 independent groups** are
  represented or the 95% CI half-width exceeds **0.25 × source-group SD**.
  Report insufficiency at the final pilot. Include every eligible file in a
  selected code repository group.
- Main: **at least 200 scientific documents and 200 code groups**, **64
  positions/document** at layer 15, first **16 nested positions** at layers 0/7,
  and **200 C4 diagnostic documents**. Use **2000 pilot bootstrap draws** to
  choose the lowest measured GPU-runtime scientific/code allocation meeting
  the **0.005-nat CI half-width** target and source-unit minima. Freeze exact
  inputs and analysis before collecting main outcomes.
- Infer document effects first. Domain cluster variance is
  `G/(G−1) × Σg[Σi∈g(Di−Dbar)]²/N²`; primary variance is
  `0.25 × (V_scientific + V_code)`. Resample whole groups, preserving files and
  repeated draw slots. Final paired bootstrap-t uses **10,000 draws**; report
  degeneracy and require **≥95% valid draws**. Use linear CI quantiles and
  `(count+1)/(valid_draws+1)` tail probabilities before two-domain Holm correction.
  C4 is descriptive: hostname bootstrap intervals require at least 10 hostnames
  and ≥95% valid draws; no confirmatory test is performed. Main planning scales
  domain variance by pilot/main source-group counts and uses 2000 pilot
  bootstrap-t draws with measured technical timings.
- Technical mechanics: four deterministic positions/document (first, last, two
  interior thirds; all available if fewer), layers 0/7/15 and all eight experts.
  Routing, survivors and cleanup must be exact. FP32 mixture/logit comparisons
  use atol=1e-6, rtol=1e-5; NLL tolerance is 1e-6 nats.
- Sensitivity: freeze a source-stratified **25% of main documents**, selecting
  whole groups, including complete repositories, as close to the target count
  as possible. Preserve predictor positions and targets, shorten inputs to
  **2048 tokens**, and perform matched **2048 calibration**.

Pinned retokenization confirms only **7 C4 calibration documents** and **99 code
calibration documents** support the new full-context rule. The first calibration
budget cannot run on this snapshot. Arya selected keeping full context and
preparing an additional long-document acquisition plan. Use a new directory and
manifest; no retained corpus is replaced or concatenated.


# Fresh Path B implementation — 2026-10-13

This is a new implementation of the approved Research Master, using `config/`,
`scripts/` and `terraform/`. It does not import the previous implementation.
The README's scientific methods remain; obsolete operational commands/status and
historical census were removed from it. `WORKFLOW.md` describes current execution.

## Recovery and retention

Before replacing files, the complete non-ignored dirty working tree was committed
as `a19f7711572071fac97386036ce7af89a22417f5` and tagged
`archive/path-b-before-rebuild-2026-10-13`. No remote Git operation was performed.
Inspect/recover code in a separate checkout, preserving the new working tree:

```bash
git worktree add --detach /tmp/dsic3007-before-rebuild archive/path-b-before-rebuild-2026-10-13
```

Local code, old plans/prepared bundles, diagnostic packages, runtime receipts and
other historical implementation records were moved into
`.archive/path-b-before-rebuild-2026-10-13/`. This ignored directory is a local
backup, not part of the Git tag and not an offsite backup. No corpus, model, raw
runtime receipt or cloud volume was deleted. The retained raw corpus and tokenizer
remain usable locally; new prepared data is separate.

Retained active references: the Master, literature notes, historical Path B decision,
original FP32 asset catalog, retained-source identity catalog, and the feasibility,
optimization and paired diagnostic findings in `docs/validation/`.
Those findings are historical: their old source links resolve through the recovery
tag and their runtime/package paths through `.archive/`. They are not prerequisites
or GPU acceptance for the fresh scripts. Old decision JSON operational grants are
historical, not executable authorization or current account state.

## Retained behavior and deliberate simplification

- One captured full baseline and one exact uninstrumented cleanup per input.
  Final-layer branches reuse native-shaped expert outputs and ordered accumulation;
  all eight effects remain. No lower-layer executor or generic suffix replay.
- Independent verification uses unmodified library-native expert execution,
  repeats, no-op, all eight masks, untargeted rows, survivors and exact cleanup.
- Consecutive GPU stages share one model session. Calibration/pilot ladders reuse
  complete identical windows. No cross-stage scientific row reuse or old receipts.
- One prepared token store and one source review; no copies of the full review
  in every plan. Raw native calibration routes/FP32 coefficients use lossless blobs.
- SQLite transaction per complete input; online backup for retrieval. Configuration,
  actual source and dependency lock are saved automatically. Numerical-code/data/
  calibration matching protects reuse; narrative documents do not invalidate it.
- One execution config feeds Terraform. Prebuilt dependencies, persistent results,
  direct SSH transfer, automatic compute cleanup after the remote run; no manual
  authorization IDs, account-grant registry, sealed-package hierarchy or R2 service.

Layer/context/outcome deferrals and eight-position sampling remain the approved
scientific Path B changes. They are not presented as computational optimizations.
No BF16/TF32, quantization, selective effects, changed source weighting, or new
equivalence/depth/context claims were introduced.

## Actual local data preparation

The fresh preparation processed 34,591 retained/supplemental records and stored
122,210,954 token IDs (488,843,816 bytes). Exclusions and every source allocation
are recorded in `data/prepared/documents.jsonl` and `prepared.json`.

| Domain | Technical documents/groups | Probe documents/groups | Calibration documents/groups | Pilot documents/groups | Available main documents/groups |
|---|---:|---:|---:|---:|---:|
| C4 | 12/12 | 32/26 | 500/483 | deferred | deferred |
| Scientific | 12/12 | 32/32 | 500/500 | 80/80 | 6,542/6,542 |
| Code | 14/12 | 32/31 | 500/488 | 80/80 | 3,171/3,092 |

There are 152 technical windows, requiring 1,216 individual native-versus-optimized
expert comparisons plus the other mechanism checks. No stage allocation shortage
was reported. These are inventory/design counts, not measured expert outcomes.
The audit covers exact content and recorded aliases; it does not certify unknown
clone/variant relationships. Code groups mean whole eligible collected groups.

## Verification and remaining work

Local tests cover independent tiny OLMoE native comparisons, target alignment,
locality/survivors, cleanup/failure restoration, calibration weighting/stability,
whole-group bootstrap multiplicities, an explicitly expanded bootstrap-t reference,
ties/controls, main allocation, resume/completeness, actual input mismatch, token
corruption, online backup, transfer failures and source snapshot exclusions.
The connected tiny-model workflow executes benchmark → calibration → nested pilot
→ analysis and resume. It deliberately permits a zero/degenerate scientific result.

Terraform validates against the installed pinned provider. Two mock plans check
one-pod provisioning and compute release while retaining protected network storage.
Shell syntax and Python compilation are checked. Final local run: **32 passed in 12.83 seconds** (Python 3.14.6), **2 Terraform
mock plans passed** (Terraform 1.13.4), Terraform validation/format, shell syntax,
Python compilation, dependency-lock check and Git whitespace checks passed.
These checks are not pretrained runtime proof.

Not run: Docker image build/push, real provisioning/SSH/storage locking, original
FP32 pretrained GPU verification/fit, real calibration/probe, pilot/main effects,
loading/throughput timing or achieved precision. No paid GPU/cloud action occurred.
Existing projected speed/cost is not used as measured input to the fresh planner.

Run long builds and jobs yourself using `WORKFLOW.md`. Before paid work, configure
the actual image, data center, prepared model storage and process time limit.
Before main, the sufficient pilot, full technical timings, remaining cost/time
ledger and exact prospective selection are required. The precision fallback is
still pending; the code currently reports insufficiency rather than relax it.

The regional-volume Terraform path is independent of the previous Global Volume
setup. Existing Global Volume bytes have not been moved or reverified remotely.
SQLite persistence and the prebuilt SSH runtime must be verified on the actual
RunPod storage/image. Runtime timeout is not a hard account-spend cap; controller
or cleanup failures can leave a managed pod to inspect. Protected storage remains
billable after compute cleanup.

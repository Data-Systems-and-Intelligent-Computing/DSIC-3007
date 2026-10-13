# Working on Path B

`README.md` is the Research Master. Read it before scientific changes. Do not
import methods or execution restrictions from `.archive/` or old workspaces.

## Structure

- `config/research.json`: executable research parameters.
- `config/execution.json`: machine paths, runtime limits, remaining budget, RunPod settings.
- `scripts/prepare.py`: tokenization, recorded-source audit, disjoint partitions.
- `scripts/measurement.py`: model loading, native references, final-layer interventions.
- `scripts/benchmark.py`, `calibrate.py`, `experiment.py`: GPU research stages.
- `scripts/analyze.py`, `statistics.py`: offline planning, inference and artifacts.
- `scripts/common.py`: small I/O helpers and transactional SQLite measurements.
- `scripts/run_gpu.py`: consecutive stages in one loaded model session.
- `scripts/runpod.sh`, `export_results.py`: transfer and managed compute cleanup.
- `terraform/`: pinned provider, persistent storage, pod, prebuilt runtime and SSH bootstrap.
- `WORKFLOW.md`: practical commands. `docs/rebuild.md`: evidence and recovery.

## Development

Use ordinary functions and direct scripts. Add dependencies/files only for an
actual research or operational need. Keep parameters in config, avoid duplicate
policy checks, legacy compatibility layers, authorization IDs and artifact frameworks.
Use graph discovery when needed; index `scripts/` directly (root fast indexing
may exclude that directory). Verify graph results against current source.

Before changing shared measurement/statistical functions, inspect their consumers
and affected tests. Read files before editing. Keep raw observations; complete
inputs are SQLite transactions, and failures must never become completed samples.
Resume verifies actual inputs/targets, scientific settings and frozen calibration.
Confirmatory analysis must use the calibration and main allocation actually frozen.

Preserve the Master's intervention, numerical requirements, source units and
weighting. Optimized execution must match independent library-native ablations;
CPU/tiny-model checks are implementation evidence, not pretrained GPU acceptance,
throughput, power or scientific findings. Do not silently relax precision or scope.

Run short relevant checks:

```bash
python -m pytest -q
terraform -chdir=terraform fmt -check
terraform -chdir=terraform validate
terraform -chdir=terraform test
bash -n scripts/runpod.sh terraform/bootstrap.sh
```

Terraform tests use a mock provider and make no account changes. Real provisioning
and GPU work are separate from local implementation. Hand long builds/jobs to Arya
with commands; do not run them yourself. Routine safe local edits/checks need no
extra approval. Ask before destructive remote Git operations.

The pre-rebuild local recovery tag is `archive/path-b-before-rebuild-2026-10-13`.
Ignored `.archive/` retains previous local assets; never overwrite it casually.
Keep credentials, Terraform state, corpus/model bytes and generated results out
of Git. The declared workflow preserves network storage when releasing compute.

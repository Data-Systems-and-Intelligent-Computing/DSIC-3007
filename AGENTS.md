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
- `scripts/runpod.sh`, `export_results.py`: SSH transfer, verified backups/restore and owned compute cleanup.
- `scripts/diagnose_gpu.py`: user-run bounded load/benchmark diagnostic; API create, SSH, retrieval and owned pod deletion.
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
Preserve the native full-window residual/RMSNorm shape in optimized suffixes;
slice to the target token only after final normalization. Keep NLL failures fatal.

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
of Git. Store the model on the existing network volume selected by
`runpod.network_volume_id`, mounted at `/workspace` in its data center. Acquire
pinned FP32 bytes once on a CPU preparation pod with `runpod.sh acquire`; GPU
runs load `model_path` directly without local staging or implicit downloads.
`runpod.acquisition_python` selects the preparation interpreter. Acquisition uses
an isolated `.acquire-venv` with the Hub version from `uv.lock`; no scientific
runtime packages or CUDA are needed. GPU startup still installs nothing.
Manual pod SSH host/port live in execution config; the private key defaults to
the configured public-key path without `.pub`. Automatic diagnostics override
the connection for their own pod; users need no SSH environment exports.
Diagnostics save provider startup logs, readiness errors and subprocess output
locally before cleanup. Preserve those on failures before GPU execution too.
Use the per-run SSH control socket for SSH/rsync and record the actual assigned
address in resolved execution. Connection limits live in config; never retry
scientific execution automatically.
The old Global Volume remains untouched. Active SQLite stays under the configured
`runpod.remote_root` (`/root/path-b` by default), on local pod disk. Fetch before
stop/reset/delete; restore verified backups onto a new pod. Pods created outside
Terraform need explicit termination after retrieval except the diagnostic runner,
which deletes its own pod after successful backup and on startup failure. Failed
retrieval preserves the pod and prints recovery instructions. Diagnostic mode
(`load` or `benchmark`), window count and limits belong in execution config.
Limited benchmarks interleave domains/source groups and report verification time
separately from durable routing/effects. Runtime scenarios use actual source-group
file and window counts; estimates never freeze main or grant full acceptance.
Initialize CUDA and validate the device/allocator before model loading. `load`
reports constructor/GPU-transfer times and load allocator peaks; keep FP32 checks.
Never run paid diagnostics on Arya's behalf; hand over
`python -m scripts.diagnose_gpu`, which reads those settings directly.
`runpod.terraform_managed=true` selects owned Terraform
cleanup only after retrieval succeeds; regional storage persists independently.

# Running the fresh Path B study

Read `README.md` for the science. `config/research.json` holds its executable
parameters; `config/execution.json` holds paths, runtime limits, infrastructure,
and remaining funds/time. The scripts do not read old `src/`, plans, or runners.
Run commands from the repository root.

## Local setup and preparation

```bash
uv sync --extra dev --extra figures
uv run python -m pytest -q
uv run python -m scripts.prepare
```

Preparation is already completed locally in `data/prepared/`. Do not repeat it
for this inventory. `prepared.json` reports exclusions, partitions and shortages;
`documents.jsonl` records every included/excluded source; `tokens.bin` contains
original, untruncated token IDs. A new corpus/context/partition needs a new
`prepared_path`. Changing only analysis settings does not require retokenization.

Inputs are retained JSONL files with `text`, `source_metadata`, and optional
`duplicate_source_metadata`. The script also accepts explicit `source_group`
and `group_aliases` on records. All recorded aliases, including those on short or
excluded records, participate in transitive grouping. Scientific paper variants
get one eligible representative; pilot/main code groups retain every eligible
collected file. This is an audit of recorded relationships, not proof that
unrecorded clones/variants do not exist. No public-repository completeness claim.

The tokenizer is the retained pinned OLMoE tokenizer, loaded locally. There is
no concatenation, special-token insertion, truncation or short-context fallback.

All Python stage commands accept `--research FILE --execution FILE`.
`PATH_B_MODEL_PATH`, `PATH_B_PREPARED_PATH`, `PATH_B_RESULTS_PATH`, and
`PATH_B_TOKENIZER_PATH` override machine paths without editing scientific settings.

## Runtime and model

Build dependencies into the image once, before renting a GPU:

```bash
docker build -f terraform/Dockerfile -t YOUR_REGISTRY/path-b:VERSION .
docker push YOUR_REGISTRY/path-b:VERSION
```

Set `runpod.image` to the published image, preferably its digest. The image uses
the frozen dependency lock and starts SSH; it does not install packages, download
weights, or start experiments at boot. Torch's locked Linux wheels use CUDA 13.0.
Execution config pins the image already built and pushed by the user. SSH startup
was observed on the previous pod; successful pretrained loading remains unverified.
No image rebuild is needed for the network-volume/script changes.

## Network volume and one-time model download

Current `config/execution.json` selects A40 in `CA-MTL-1` and the existing
50 GB High-Performance network volume `2fkb3b9n7h`. The volume mounts at
`/workspace`; `model_path` is `/workspace/models/OLMoE-fp32`. A network volume
must be attached when creating the pod, in its own data center. The previous
Global Volume is not attached or deleted by this workflow.

Fill the new volume once before starting a GPU diagnostic. Use a CPU preparation
pod in that data center if available, with the configured image, private registry
credential and network volume. Expose TCP 22 and use the registered SSH key.
The current preparation pod has Python 3.10 at `/usr/bin/python3`, selected by
`runpod.acquisition_python`. It has no `/opt/path-b` runtime. This is a user-run
preparation step, separate from GPU execution.

The preparation pod's direct TCP host/port are saved as `runpod.ssh_host` and
`runpod.ssh_port` in execution config. The private key defaults to the matching
`ssh_public_key_path` without `.pub`, with `~` expanded automatically. Update
these two connection fields when replacing a manually created preparation pod.
The automatic diagnostic runner supplies its own live connection for its pod.
From the local repository, no SSH exports are needed:

```bash
bash scripts/runpod.sh push
bash scripts/runpod.sh acquire
```

`acquire` creates `.acquire-venv` under the remote repo using that interpreter and
installs the Hugging Face Hub version from `uv.lock` if missing or different.
This preparation-only environment needs no PyTorch, NumPy or CUDA. The GPU image
and its frozen research environment are unchanged; GPU startup installs nothing.
It runs `scripts.acquire_model` with the configured model path, downloads the
pinned original FP32 assets, verifies SHA256/sizes and F32 header metadata, and
writes `path-b-acquisition.json` only on success. Hugging Face reuses completed downloads after an interruption;
repeat `acquire` on the same volume. Later completed acquisitions use the saved
record and cheap size checks without repeated full-weight hashing. Alternatively,
`bash scripts/runpod.sh model /path/to/already-verified/OLMoE-fp32` transfers existing
local bytes rather than downloading again.

The preparation command does not provision or delete a pod. After successful
acquisition, terminate the preparation pod yourself; keep network storage. If
acquisition fails, leave the volume intact and correct/retry the download. Neither
long downloads nor paid preparation/GPU jobs are run by the assistant.

## User-run GPU diagnostic

Diagnostic settings live in `config/execution.json`:

```json
"diagnostic": {
  "mode": "benchmark",
  "windows": 9,
  "max_seconds": 1200,
  "ssh_ready_seconds": 420,
  "transfer_seconds": 180,
  "retrieval_seconds": 120,
  "poll_seconds": 10
}
```

After model acquisition completes, run locally from the repository root:

```bash
python -m scripts.diagnose_gpu
```

The runner reads the API key from the environment or ignored `.env`, checks the
existing volume's data center, registry credential, SSH key and current GPU stock,
then creates one pod with that network volume. It waits for direct SSH, checks
only the configured model directory against catalog file sizes and uploads current
scripts/config and prepared input. It never downloads or stages model bytes.
Stock checks do not reserve a GPU. An explicit HTTP 400 creation rejection means
no pod was allocated and needs no recovery. Catalog stock can disagree with the
allocator for the complete request; a rejection alone does not prove stock is gone.
`create-request.json` retains the provisioning payload with environment values
redacted; HTTP errors retain status and available request/trace IDs in `run.json`.
Diagnostic mode changes worker behavior only, not the provisioning request.
An ambiguous creation response still triggers lookup by the exact request name,
without repeating the POST.
The worker initializes CUDA and checks the selected device/allocator before model
loading, records `df -hT`, then calls `from_pretrained` directly on `model_path`.

`diagnostic.mode: "load"` stops after loading. Set it to `"benchmark"` to continue
with `diagnostic.windows` inputs in the same loaded session. The current nine-window
diagnostic interleaves C4/scientific/code (three each), taking different technical
source groups before revisiting a group. Full technical selection is unchanged.
Scientific settings remain FP32, 4096 tokens, layer 15 and all eight native
ablations. A limited diagnostic cannot grant full technical acceptance or run
calibration/pilot/main. The A40 network-volume load completed in 143.21 seconds
with 27.680 GB peak reserved VRAM in diagnostic `20261015T070309850673Z`.
The later single-C4-window diagnostic `20261015T073622468533Z` passed eight native
comparisons with zero reported error, with 31.350 GB peak reserved VRAM and
30.10 seconds including warm-up/verification. Its durable effects/routing timings
were 2.95/2.91 seconds. These are single-window observations, not cross-domain
throughput or full technical acceptance.

The nine-window run `20261015T080152380616Z` completed five inputs, then failed
the frozen ablated-NLL gate on its second code input. Completed mixtures had zero
reported error, but some logits differed. The optimized suffix previously applied
final RMSNorm to a single token instead of the full native window. It now keeps
the full residual/normalization shape and slices only before the LM head. A local
regression reproduces that shape mismatch and checks all eight corrected branches
against native inputs/logits; the pretrained A40 fix still requires a new run.
NLL tolerance remains `1e-6`; failures now include expert/target, both NLL values,
absolute error, tolerance and logit/mixture maxima. Earlier timings describe the
previous measurement implementation and must be remeasured after this fix.

`benchmark-diagnostic.json` reports per-domain routing/effects durable timings
separately from `verification` timing, with warm-up and model load separate.
`runtime_projection` contains work-only timing scenarios for the full technical
set, nested calibration budgets (with probe at eligible freeze rungs), pilot rungs,
the configured minimum main, and the available main inventory. Code scenarios
include every eligible file in each selected group and use actual position counts;
calibration budgets reuse previously completed inputs rather than adding all rungs.
Mean-rate and observed-max-rate estimates are extrapolations, not confidence
bounds, invoices, achieved precision or a frozen main allocation. Add one model
load per session plus startup, transfer, warm-up and remaining CPU/I/O work.
No calibration, pilot, main collection or main-plan freeze runs in this diagnostic.

`model-load.json` reports the source model path, `from_pretrained_seconds` (CPU
constructor reading the volume), synchronized `to_gpu_seconds`, `total_load_seconds`,
and PyTorch load allocator peaks `gpu_allocated_gb`/`gpu_reserved_gb` in decimal GB.
These are not forward-pass or whole-device VRAM peaks. `filesystems.txt` and
`console.log` retain mount information and execution output.

During provisioning, `startup.jsonl` records RunPod system/container log events
locally, including a final bounded read before cleanup. Log API failures are
recorded explicitly. `progress.jsonl` records stage timestamps, actual SSH
host/port and the reason for each failed readiness probe. `controller.log` keeps
SSH/rsync/job subprocess output, including failures before any GPU work starts.
These files survive deletion of the diagnostic pod. Provider logs can still be
incomplete if the API is unavailable; the recorder reports that limitation.

`runpod.ssh_connect_seconds` (10) and `runpod.ssh_attempts` (3) govern connection
establishment. Readiness probes make one attempt per poll within the existing
readiness deadline. The diagnostic uses an isolated OpenSSH control socket to
reuse the authenticated connection for SSH and rsync; the socket includes the
host/port identity and is closed at exit. The resolved execution snapshot records
the new pod's actual SSH address. Scientific jobs are never automatically replayed.

The configured 1200 seconds covers worker initialization/model loading/GPU work;
420 seconds covers waiting for SSH, 180 seconds upload and 120 seconds retrieval.
The remote timeout allows 10 seconds for forced termination. Startup/transfer are
additional billable time; the final log read can add up to ten seconds before
cleanup. This is not a total rental or account-spend cap.
The worker exits as soon as its windows finish; it does not wait out the limit.

After a completed/failed/interrupted diagnostic, the controller stops its recorded
process, retrieves a verified online backup, deletes only its pod, and confirms
deletion by a new API read. It never deletes network storage. A retrieval/cleanup
failure preserves the pod and prints `RECOVERY REQUIRED` with its recovery command.
Keep the controller terminal alive: power loss/SIGKILL can prevent cleanup.
Each run is stored separately under `results/diagnostics/`, including resolved
execution, logs, settings/source and a load report if loading completes.

Read-only preflight and explicit recovery:

```bash
python -m scripts.diagnose_gpu --plan
python -m scripts.diagnose_gpu --recover results/diagnostics/THE_PRINTED_RUN_DIRECTORY
```

The API runner uses `mounts.network` with the configured volume ID and
`registry` with the configured private-image credential. `runpod.remote_root`
remains `/root/path-b`: scripts, prepared input and active SQLite are on pod disk;
model bytes persist on the network volume. Fetch before stop/reset/delete and
restore verified backups onto a new pod. Model persistence does not make local
SQLite persistent. Full-window calibration routing may produce larger artifacts
than the roughly 481 MiB input; result size remains unmeasured.

For a manually provisioned pod with prepared model bytes:

```bash
bash scripts/runpod.sh push
# On a NEW pod only, when restoring previously retrieved observations:
bash scripts/runpod.sh restore
# Set max_seconds first; the user runs GPU stages:
bash scripts/runpod.sh run benchmark calibrate pilot --units 20
```

`run` fetches a verified online SQLite backup even after a failed stage. `fetch`
also works during an active job; issue one retrieval at a time. `restore` refuses
to replace existing remote observations. The generic wrapper does not delete an
API/console-created pod; terminate it after successful retrieval. Failed retrieval
leaves compute available for recovery. Periodic retrieval for long jobs is manual.

## Alternative: Terraform-managed pod with regional storage

Set `runpod.data_center`, `runpod.image`, and `runpod.ssh_public_key_path` in
execution config. Select actual stock and price before a paid run. Current config selects
a single A40, 64 GB host RAM and the existing 50 GB network volume; allocation and
model fit still need runtime verification. Set a compatible CUDA floor for the
built image. `network_volume_id` can reference an existing **regional** volume in
that data center. For this alternative, set `runpod.terraform_managed` to true
and `runpod.remote_root` to `/workspace/path-b`. A Global Volume ID cannot be
used as `network_volume_id`; the provider wiring here uses regional storage.

```bash
export RUNPOD_API_KEY=...
terraform -chdir=terraform init
terraform -chdir=terraform plan -var=enable_compute=true
terraform -chdir=terraform apply -var=enable_compute=true
```

Terraform reads execution config directly. Its default `enable_compute=false`
can provision storage alone; applying Terraform can therefore incur storage
cost even without a GPU. The network volume is protected from destruction.
`-var=execution_file=/absolute/path/to/execution.json` selects another config.

Set the pod's direct TCP connection in `runpod.ssh_host`/`runpod.ssh_port`, then run:

```bash
bash scripts/runpod.sh push
bash scripts/runpod.sh model /path/to/already-verified/OLMoE-fp32
bash scripts/runpod.sh run benchmark calibrate pilot --units 20
```

`push` sends current scripts/config, the prepared corpus and the model catalog.
`model` is an explicit transfer; omit it when the volume already has the model at
`/workspace/models/OLMoE-fp32`. The wrapper uses relative prepared/results paths
from execution config. It runs all selected stages in **one loaded model session**,
streams progress into the local console log, retrieves a verified SQLite backup,
then removes the Terraform-managed pod automatically after successful retrieval.
Durable storage remains. A failed retrieval leaves compute running for recovery.
Failed/interrupted inputs are not complete observations.

Set `max_seconds` before a bounded run. The remote shell timeout includes loading
and all selected GPU stages; it allows 30 seconds for termination. Provisioning,
upload, retrieval and Terraform cleanup are additional billable time, so this
is not a hard account-spend guarantee. If the local controller itself dies,
Terraform cannot clean up automatically; inspect/remove its managed pod.

Direct GPU commands, on an already provisioned machine:

```bash
python -u -m scripts.run_gpu benchmark calibrate pilot --units 20
```

Direct Python execution does not provision or remove a pod. The per-input limit
is cooperative; use shell `timeout` for a hard process deadline including loading.

## Pilot, main freeze, final analysis

```bash
uv run python -m scripts.analyze pilot
```

The report identifies pilot insufficiency and the next rung. If needed, provision
the pod again, `push`, then run `pilot --units 40` or `pilot --units 80`. Earlier
complete measurements are reused. With local pod disk, restore the retrieved
results onto a new pod first; with regional storage they persist on the volume.
A changed GPU/runtime or measurement implementation requires fresh
native benchmarking. A diagnostic `scripts.benchmark --limit N` cannot grant
full technical acceptance.

Before main, update the reconciled `remaining_usd`, `remaining_wall_hours`,
`hourly_usd`, `fixed_remaining_usd`, `remaining_overhead_seconds` and
`remaining_offline_seconds`. Account for prior study spending/time, restart,
transfer, final analysis and retained storage. These fields are initially unknown.

```bash
uv run python -m scripts.analyze pilot --freeze-main
```

The planner uses sufficient pilot cluster variance, bootstrap-t critical values,
actual whole code groups and full technical durable timing maxima. It chooses
the fastest precision-qualified allocation and checks remaining time/money with
20% reserve. This is a conditional forecast, not achieved precision or an invoice.
`main-plan.json` freezes exact documents, predictors, calibration and analysis.
Copy it to the configured remote results directory before main:

```bash
# With the pod provisioned and the usual SSH variables set:
scp -i "$PATH_B_SSH_KEY" -P "$PATH_B_PORT" results/path-b/main-plan.json "root@$PATH_B_HOST:/root/path-b/results/path-b/main-plan.json"
bash scripts/runpod.sh run main
uv run python -m scripts.analyze main
```

Use your configured relative results path in the SCP command if it differs from
the default; use `/workspace/path-b` for the regional-storage alternative. Main
freeze is refused when the pilot is insufficient, precision
does not fit the available inventory/budget, or outcomes already exist. The
Research Master's precision fallback remains pending; it is not silently relaxed.

Analysis saves `analysis.json`, document means and estimates CSVs, and optional
PDF/PNG figures. Only paired delta has the primary and two Holm-corrected secondary
tests. Other scores/controls are descriptive. M0 Monte Carlo variability is
reported separately. Degenerate bootstrap draws are disclosed; inadequate valid
draws produce no interval or equivalence claim.

## Resume, monitoring and cleanup

Repeat the same stage command with the same inputs/settings to resume. Each
complete input is a SQLite transaction with original window IDs, target,
baseline, eight ablated NLLs/effects, routing and timing. Calibration additionally
retains lossless compressed whole-window expert IDs and FP32 coefficients.
Hooks/overrides are restored before exact cleanup and persistence. Incomplete
stages are refused by analysis instead of becoming a selected complete-case sample.

Progress reports stage, completed/total windows, elapsed time, ETA, and peak
RAM/VRAM. Source bytes, resolved settings, pinned lock and frozen rankings are
saved automatically. Documentation edits do not invalidate numerical measurements.

```bash
bash scripts/runpod.sh fetch
bash scripts/runpod.sh cleanup
```

`fetch` makes an online SQLite backup and checks transferred bytes and database
integrity. `cleanup` retrieves first, then removes Terraform-managed compute or
asks you to terminate a pod created outside Terraform. The existing volume remains.
Retained storage continues to incur its own cost. Never use a broad
`terraform destroy` as routine cleanup. Terraform state stays local and ignored.

Verification status and recovery instructions: [docs/rebuild.md](docs/rebuild.md).

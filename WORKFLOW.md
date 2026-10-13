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
The image build and actual pod startup remain unverified in this session.

Use the existing verified original FP32 model bytes. If acquisition is actually
needed, do it locally or on a CPU preparation machine, not inside GPU startup:

```bash
PATH_B_MODEL_PATH=models/OLMoE-fp32 uv run python -m scripts.acquire_model
```

Acquisition downloads only the pinned catalog assets and checks their SHA256,
sizes and F32 headers once. Subsequent acquisition calls use the saved completion
record and cheap sizes. GPU loading checks presence/sizes and actual loaded FP32
parameters, without another full-weight hash pass. Previously prepared bytes
are trusted as unchanged; those checks are not fresh verification of every byte.

## Provision, upload, run, retrieve

Set `runpod.data_center`, `runpod.image`, and `runpod.ssh_public_key_path` in
execution config. Select actual stock and price before a paid run. Defaults are
a single L40S, 64 GB host RAM and 80 GB durable network storage; these are intended
settings, not confirmed availability or fit. Set a compatible CUDA floor for the
built image. `network_volume_id` can reference an existing **regional** volume in
that data center. An existing Global Volume is not a regional volume ID; migrate
its prepared bytes deliberately or use direct GPU commands on a suitable mount.

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

Copy the pod's public SSH host/port into these local environment values:

```bash
export PATH_B_HOST=...
export PATH_B_PORT=...
export PATH_B_SSH_KEY=$HOME/.ssh/id_ed25519
bash scripts/runpod.sh push
bash scripts/runpod.sh model /path/to/already-verified/OLMoE-fp32
bash scripts/runpod.sh run benchmark calibrate pilot --units 20
```

`push` sends current scripts/config, the prepared corpus and the model catalog.
`model` is an explicit transfer; omit it when the volume already has the model at
`/workspace/models/OLMoE-fp32`. The wrapper uses relative prepared/results paths
from execution config. It runs all selected stages in **one loaded model session**,
streams progress into the local console log, retrieves a verified SQLite backup,
then removes this Terraform-managed pod automatically. Durable storage remains.
Failed/interrupted inputs are not complete observations; the network volume
retains successful inputs even if retrieval fails.

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
complete measurements are reused. Calibration and benchmark results persist on
the volume. A changed GPU/runtime or measurement implementation requires fresh
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
Copy it to the persistent remote results directory before main:

```bash
# With the pod provisioned and the usual SSH variables set:
scp -i "$PATH_B_SSH_KEY" -P "$PATH_B_PORT" results/path-b/main-plan.json "root@$PATH_B_HOST:/workspace/path-b/results/path-b/main-plan.json"
bash scripts/runpod.sh run main
uv run python -m scripts.analyze main
```

Use your configured relative results path in the SCP command if it differs from
the default. Main freeze is refused when the pilot is insufficient, precision
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
integrity. `cleanup` retrieves first, then removes compute while preserving
storage. Retained storage continues to incur its own cost. Never use a broad
`terraform destroy` as routine cleanup. Terraform state stays local and ignored.

Verification status and recovery instructions: [docs/rebuild.md](docs/rebuild.md).

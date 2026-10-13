# L40S + Global Volume selection — 2026-10-13

**Current state:** model preparation completed in
`dsic-screen-20261013T053315-10a2bf48`. The following $.20 benchmark hit the120s
weights cap before native comparisons. Cleanup is confirmed. The grant is consumed.
[Successor v9](PATH-B-BENCHMARK-DEADLINE-FIX-2026-10-13.md) fixes benchmark phase
allocation and prepares a separate $.60/1800s proposal pending authorization.
The earlier readiness/grant paragraphs below are historical. Do not repeat model setup.

Arya selected L40 and plans to create a Global Volume in the console, delete the
regional network volume, and re-download the pinned model instead of copying it.
No resource creation/deletion, download or paid work was performed by the agent.
The research design, FP32 arithmetic and numerical tolerances remain unchanged.
[Execution decision](../../configs/path-b-l40-storage-decision-v1.json).

## Current readiness

Global ID/name supplied by Arya: `cmv66f04p000007l8gngvd5k4` /
`yelling_amber_shark`. Attachment not independently verified. Proposed custom
mount/model path: `/workspace-global/models/OLMoE-fp32`; confirm on the actual pod. New cache preparation is separate
from the previously authorized $.20/900s cache-only diagnostic. Its budget and
elapsed-time allowance are not yet specified. Frozen FP32 catalog has about 27.6GB
of model assets; use the actual catalog total rather than a download-time estimate.
The exact commit is `c6f6954f182ffdd9f6ab4f4d772714b7dcd6836b`.

The existing preflight binds L40S, regional volume `k6u810gze3`, and US-MO-1;
it does not implement the new L40/Global Volume choice. After Arya reported L40S
availability and a README source-hash error, a v3 successor was prepared to fix
that hash mismatch without relaxing the region or numerical gates. The live
read at 04:19:42Z still showed no L40S stock in US-MO-1. A new L40/Global Volume
package is NOT ready yet.

## Verified product and API facts

Runpod's Global Volume documentation supports GPU Pods across regions and
says that a volume alone defaults to `/workspace`; with an additional network
volume the untouched default shifts to `/workspace-global`. Confirm mount paths
at deployment. Global Volumes are object-backed and do not support atomic rename,
file locking or hard links. Weights can live there, but the research atomic
journals must use local POSIX disk. Raw results need retrieval before pod teardown;
local disk does not provide the old network volume's recovery guarantee.

Public REST v2 OpenAPI fetched on this date has no global-volume routes or mount
fields. The Pod mounts schema supports network/persistent only. Do not invent a
Global Volume mount payload or treat a global id as a regional volume id. Current
official Pod attachment instructions use the console. CLI/other API support must
be established before integrating automated create/cleanup with its cost guard.

[Overview and request pricing](https://docs.runpod.io/storage/globalvolume/overview)
[Pod attachment and mount paths](https://docs.runpod.io/storage/globalvolume/globalvolume-pods)

## Next sequence

1. Arya creates Global Volume and sends ID; inspect its actual mount/model path.
2. Prepare a bounded user-run download/verification step using the exact pinned
   manifest; reconcile setup cost before starting a billable preparation pod.
3. Adapt the L40 device gate, supported deployment path, local workdir, retrieval
   and guard ownership. Freeze a new package with explicit Global Volume identity.
4. Re-run local regressions, then one user-run diagnostic within the existing cap.
5. Use new measured timings and actual costs for subsequent budget decisions.

## Local implementation update

`runpod/global_preflight.py` reads current GPU stock across ALL data centers;
no US-MO-1 lock and no regional-volume lookup. Tests cover stock moving between
regions and reject inherited region constraints, silent GPU substitution,
over-cap prices and unavailable pinned CUDA. The default CLI selects this global
profile. Paid run/prepare fail clearly before any API action because Global
Volume attachment support is not integrated. Legacy regional execution is an
explicit flag and does not silently authorize a different storage class.

Public GraphQL schema introspection also returned INTROSPECTION_DISABLED. No
attempt was made to bypass it or invent fields. Current official console
instructions are the established attachment route; bounded automated deployment
or an explicitly verified console-created pod remains required.

Critical benchmark metrics are now explicit in the final receipt; model
acquisition is timed separately from native comparisons while retaining parent
session identity checks. Existing raw event/timing and GPU-sample files remain.
First-ablation-only timing, CPU percentage and R2 upload are absent, rather than
filled with invented values. Eight-effect timings and process RSS are recorded.
No router-suppression intervention or dtype/tolerance change was added.

Local checks: **74 affected tests passed in 18.45s**; final focused recheck **6
passed in 2.86s**. No pretrained GPU run occurred. At 04:35:26Z the live catalog
reported L40 stock NONE and L40S LOW (including multiple regions). The default
L40 check correctly stopped before any paid action; a GPU substitution remains
an explicit user choice. Verification and durable logs are recorded in
[global-preflight-implementation-2026-10-13.json](global-preflight-implementation-2026-10-13.json).

## Superseding update — user selected L40S and authorized model preparation

Arya explicitly selected L40S. Global Volume ID/name now CONFIRMED using the
official SDK GraphQL query `myself { globalStoreBuckets { id name } }`.
New official SDK source resolves the previous attachment gap: `volumeMounts`
with `volumeType=OBJECT_STORE_VOLUME`, via `podFindAndDeployOnDemand`. Source:
https://github.com/runpod/runpod-python/tree/3864d374cba523ecee09f3bc95f073f17a64b970
Earlier API-gap observations above remain historical, not current readiness.

User-run model preparation is separately authorized at **$.20/900s**. The existing
cached diagnostic authorization is also **$.20/900s**. No paid run by the agent;
no automatic retries. Package v4 binds the updated profile/source. Setup may stop
with incomplete cache; additional setup attempts require budget reconciliation.
The guard reserves $.08 fixed allowance for preparation and $.03 for diagnostic,
plus $.02/hour storage allowance, and90s cleanup. These are planning allowances,
not a hard provider billing cap or proof of Global Volume request billing.

The controller sends ALL stock-qualified regions, checks live volume and registry,
retains running unrelated pods, uses the pinned image/SSH tooling and CUDA13, and
sets both a local detached guard and `terminateAfter`. Cleanup checks the unique
pod name/image/registry/GPU/global-ID environment marker. Actual mount presence is
checked remotely; it is not in-container physical-volume attestation. No volume
is deleted. Setup downloads directly to an owned catalog directory on the global
mount, verifies prefix Range responses, size/SHA and F32 metadata. Partial assets
are retained, never silently overwritten. Atomic status/journals are local.

`--run` requires a retrieved complete preparation receipt matching global ID,
model path, catalog digest and each exact asset size/hash, then repeats normal
model verification before the24 independent native comparisons and timings.
Ephemeral results are retrieved before deletion. If retrieval fails, those local
raw files may be lost with the pod; Global Volume weights remain. This differs
from the regional volume's retained result directory and is recorded explicitly.

Commands and expected output files are in README/WORKFLOW. Deployment, mount
performance, download completion and pretrained GPU fidelity remain unmeasured
until Arya runs the commands. Source/CPU checks alone do not prove those gates.

## Superseding CUDA capacity fix — package v5

The user's 05:04:40Z setup command stopped before pod creation on the exact
CUDA13 capacity gate. At 05:05:05Z the Secure catalog reported only CUDA12.8
L40S capacity in US-TX-4; at 05:10:45Z it reported NONE. The user separately
reported a dashboard card advertising CUDA13.0. These snapshots do not identify
the same host/pool, and neither guarantees a reservation. No conclusion that
the dashboard host is incompatible follows from the catalog failure.

Setup now uses CPU-only download/checksum/F32 metadata validation without any
torch.cuda initialization and accepts advertised host versions >=12.8. The
benchmark still requires advertised host versions >=13.0; newer compatible
drivers are accepted while the container runtime stays exactly CUDA13.0.
Actual pod version is checked against the selected advertised versions after
creation. Read-only status reports setup and benchmark capacity separately.
Image, arithmetic, scientific settings, tolerances, budget and retry policy are
unchanged. Package v5 replaces v4 for these commands; old packages remain intact.
Local regression checks: 76 passed in 3.26s. No paid execution by the agent.

## First user-run Global Volume setup receipt — 05:13:30Z

Run `dsic-screen-20261013T051330-477db619` used L40S, host CUDA13.0, US-TX-4,
the pinned container, and the correct Global Volume. Actual mount check passed.
CPU-only preparation observed no GPU compute; this is not native model fidelity
or benchmark evidence. Model preparation stopped at its deliberate deadline
reserve (TimeoutError), with about9.44GB recorded of27.68GB, including one
complete SHA-verified tensor shard and a partial second shard. Final cached byte
count may exceed the last progress receipt; remote reinspection is still needed.

Elapsed overall266.86s; preparation141.69s. The $.20 allowance with $.08 fixed
reserve produces a389.19s lease, further reduced by startup and separate
cleanup/finalization reserves. Thus the maximum900s was not the effective
download budget. The measured stream rate is about66.6MB/s, conditional on this
host and destination. Remaining recorded bytes are about18.24GB; at that rate
download alone takes about274s, excluding prefix rehash, boot and final metadata.
This projection does not guarantee completion or an invoice amount.

Pod `mjvvbbgcsb5fyd` deletion is confirmed; Global Volume retained. No retry or
benchmark performed. The final console line incorrectly named historical volume
`k6u810gze3`; request/attempt/cleanup evidence confirms the global attachment.
Package v6 fixes this footer and the setup-stage/retrieval messages. Historical
raw run and prior frozen packages are unchanged. Additional paid setup needs
new authorization because the existing grant covered one attempt only.

## Superseding authorization — one model resume, package v7

In response to the explicit request for one $.30/15min resume, Arya accepted
the fresh-volume download overhead. This is recorded as one additional model
setup attempt, not a study-budget reset or expanded diagnostic authorization.
The explicit `--resume-model` command requires the matching incomplete parent
receipt and confirmed cleanup, and refuses reuse after an attempt identity has
been created. It carries $.30 only for setup; `--run` remains capped at $.20.
Same pinned Global Volume/catalog, image, F32 checks and scientific settings.
Existing prefixes are rehashed before exact Range resumption. With $.08 fixed
and $.02/hour storage allowances the lease is713.51s at1.09/h; this is a planning
guard rather than a provider billing cap or guaranteed completion. Overall
remaining study budget is still unreconciled. Paid command is run by Arya.

## Startup failure and user-requested time increase — package v8

Resume `dsic-screen-20261013T052553-60516a78` stopped at150s waiting for SSH,
despite about560s left in the713.51s cash lease. No upload/download started;
cleanup of `fb26sdim4sswdn` is confirmed. Raising the cash cap had retained the
diagnostic startup cap; this was a controller preparation mistake. Why the
host never exposed SSH before150s cannot be resolved from the retained evidence.

Arya requested increasing the time limit and preparing the continuation without
more extensive checking. Setup now has a600s startup cap and a separate90s
upload/preflight interval after SSH readiness. Both remain clipped by the same
$.30/900s total cash/time guard and90s cleanup reserve; a long startup can leave
little download time. One user-run continuation uses resume authorization v2;
the earlier consumed v1 remains in raw evidence. No scientific settings changed,
no paid agent run. Latest boot metadata is retained locally even if SSH fails.

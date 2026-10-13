# Path B benchmark deadline fix — successor v9

Local preparation only. No paid command, model preparation, or native acceptance
was performed for this fix. The proposed retry needs a new explicit budget decision.

## Confirmed failure and repeated work

Run `dsic-screen-20261013T054903-16792cdc` completed boot/upload/preflight.
The retrieved runner receipt records weights starting after preflight at
1791870647.9748483 and stopping at 1791870768.4509015: about 120.48 seconds.
Its controller work deadline was 1791871004.9291005, leaving 236.48 seconds.
The standalone Path B weights cap was 120 seconds. A separate 190-second
benchmark deadline and 125-second load/native-validation cap would have applied
later. Neither native comparisons nor timings started. Cleanup is confirmed.

The standalone process called `verify_model`, and `VerifiedModelSession.acquire`
then called `load_model`, which calls `verify_model` again. Each verification
reads all 27,679,451,403 catalog bytes for SHA256, checks F32 metadata and checks
architecture. Path B now performs the complete check once in the unchanged
verified loader, before any native forward. The remote precheck still requires
every cached file and exact size. No old setup receipt substitutes for runtime
hash verification. Native validation, warmup and timings share the same session;
executor and parameter mutation checks remain. Explicit download acquisition
also hashes assets and then calls `verify_model`, but that setup path is outside
this benchmark fix and was not changed or rerun.

## Bounded execution allocation

Proposed one-attempt envelope: **$0.60 / 1,800 seconds**, pending authorization.
At the configured guard rate of $1.09/hour, plus $0.02/hour storage and $0.03
fixed allowance, cash arithmetic permits about 1,848.65 seconds; the wall cap
therefore limits the lease to 1,800. This is planning arithmetic, not an invoice
guarantee. Live rate, host CUDA >=13.0, pinned container CUDA13.0, GPU and volume
identity checks still run before/after creation. No study-budget reset is implied.

| Phase | Allocation | Evidence |
| --- | --- | --- |
| Create/boot/SSH | up to 600s, clipped by total/cleanup deadline | failed diagnostic boot about 76s; earlier setup boot exceeded 150s |
| Upload/archive/source/runtime/plan preflight | up to 90s after SSH, clipped by total deadline | latest diagnostic completed before weights; no completion guarantee |
| Full hashes/F32, model load, 24 native comparisons, one warmup, three measured windows | shared remaining work deadline, no 120/125/190s subcaps | full hash time exceeded 120s; successful hash/load/native/timing durations on this Global Volume are UNKNOWN |
| Remote finalization | 30s reserved | operational reserve, not measured completion time |
| Retrieve raw local-disk evidence and delete only this pod | 90s reserved; first 25s reserved for retrieval when work exhausts lease | operational reserve; Global Volume retained |

After worst-case 600s startup and 90s upload/preflight, up to 990s remain for
verified loading/native validation/warmup/timing. With 76s boot and 29s remaining
setup (rounded historical observations), about 1,575s remain. These are available
intervals, not runtime estimates or evidence the diagnostic will finish.
Hashes, loading, comparisons and timings may borrow from this shared interval.
The native validator gets the remaining work interval minus 5s for its receipt;
the subprocess supervisor enforces the absolute deadline even during blocking
hash/load/GPU work. Incomplete coverage must stop before timings. Per-event raw
evidence and progress remain durable; retrieval failure can lose ephemeral results.

Three diagnostic windows remain one/domain, context4096, layer15, FP32/eager,
24 native expert comparisons, one warmup and 24 measured effects. No reduced
comparisons, relaxed tolerances, download fallback, or scientific acceptance.
The numerical implementation SHA256 remains
`52b1c9a3b2a6bfa2e0141d7795a11fa7edff0d8d9b8daa95414dc7e38e61dfbf`.

## Authorization and user command

The original $.20 diagnostic grant is consumed. Plain `--run` now refuses before
live API calls. The proposed retry requires the explicit operator release below;
its unique authorization ID can be used only once after attempt creation.
Previous cleanup must be confirmed. Default invocation remains offline.
No command here should be run until Arya accepts this separate envelope.
Actual invoices and remaining original $12/10h allowance are unreconciled.

```bash
cd /home/arya/School/TA/DSIC-3007
python -u runpod/path_b_preflight.py --run --benchmark-authorization-id global-path-b-benchmark-2026-10-13-v9
```

The controller fsyncs console output to `runpod/runs/<attempt>/console.log` and
retrieves `results/` before deletion; `runpod/runs/latest` points to the attempt.
Inspect `result.json`, `cleanup.json`, `results/runner-receipt.json`,
`results/validation.json`, `results/independent-native-validation.json`,
native case JSONL, warmup/measured-event JSONL, input timings and GPU samples.
New source/config/README edits require another immutable successor; v8 and raw
receipts stay unchanged. Frozen v9 also binds this execution note.


## Focused local verification

107 tests passed in21.44s with the installed `python`; the repository `.venv`
does not contain pytest. Command:

```bash
python -m pytest -q tests/test_runpod_screen.py tests/test_global_deployment.py tests/test_global_preflight.py tests/test_path_b.py tests/test_path_b_benchmark.py tests/test_model_session.py tests/test_model_acquisition.py
```

These are synthetic/mocked deadline, loader-integrity, lifecycle and scope checks.
They cover600s startup policy and upload90s after SSH, an1800s controller lease,
one loaded session with570s synthetic acquisition and24 native synthetic
comparisons before timings, exhausted shared-deadline refusal, partial/failed
native coverage refusal, corrupted bytes rejected before model loading, consumed
budget release and ID reuse refusal. Nine changed Python files passed AST parsing;
`git diff --check` passed. This is a focused107-test result, not a new full suite
or native GPU/runtime proof. No paid resources or live catalog reads were made.

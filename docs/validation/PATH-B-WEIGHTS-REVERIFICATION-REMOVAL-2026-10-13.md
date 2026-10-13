# Path B: remove repeated weights verification — v10

Arya explicitly instructed removal of repeated weights verification.
The benchmark now loads the completed Global Volume model without another
SHA256, F32 header or architecture verification pass. Its controller still
requires the existing successful setup receipt for the exact catalog/volume;
that receipt documents preparation, not fresh verification of current bytes.
Cheap file presence/size checks remain. No model preparation is repeated.

Only the Path B benchmark session disables loader weight verification.
Other loaders retain their default behavior. Runtime/parameter identity checks,
24 native expert comparisons before timings, FP32/eager, context4096/layer15,
all8 effects and frozen tolerances remain. Implementation identity changes,
so fresh native acceptance remains pending.

The unrequested copy/cache/proof and64GB disk changes were removed.
Weights load directly from the existing Global Volume; disk remains35GB.
The prior120/125/190s timeout corrections remain. No speedup is claimed from
local tests. The $.60 retry proposal is withdrawn; no active paid grant exists.
No paid command, live cloud read, or model preparation was performed here.
Old packages and raw receipts are preserved; v10 is a local successor only.

Default `python -u runpod/path_b_preflight.py` verifies the local source package
offline. There is no authorized paid retry command to run.

Focused local checks:95 passed in23.45s (`test_model_session`,
`test_path_b_benchmark`, `test_global_preflight`, `test_runpod_screen`,
`test_path_b`). These are CPU/synthetic/mocked checks, not native GPU evidence.
AST syntax and `git diff --check` passed.
Current implementation SHA256: `4f989f16859917d9a85073e081eaacf1f84894e7fdbdeb93ddf11f1bfe52cb81`.

# Path B README and cost-preflight package refresh — 2026-10-13

Root README now describes the active Path B design throughout. The earlier
2026-10-10 amendment and initial inventory are preserved in
[the historical archive](README-HISTORICAL-DESIGN-2026-10-10.md). WORKFLOW marks
old plans/commands as historical and includes the current user-run preflight.
No scientific settings, numerical tolerances or intervention arithmetic changed.

The preflight source archive binds README bytes. The runner's PACKAGE constant
therefore points to a new immutable successor:
`data/optimization/path-b-preflight-2026-10-13-v2/`. The v1 archive remains intact.
Same three cases, 24 native comparisons, one diagnostic attempt, $.20/900s upper
planning envelope, cash-derived deadline, cleanup reserve, cache-only weights.
This refresh does not authorize or launch full acceptance/calibration/pilot/main.

User commands:

```bash
cd /home/arya/School/TA/DSIC-3007
python -u runpod/path_b_preflight.py --check-live
# After that read-only check passes:
python -u runpod/path_b_preflight.py --run
```

Read live stock/quote at execution time; the 2026-10-12 stock snapshot is not a
current availability claim. Progress/logs are in `runpod/runs/<attempt>/` and
`runpod/runs/latest`; result/cleanup must be inspected after the paid attempt.
The budget guard is a planning bound, not a provider hard billing guarantee.
No paid resources are launched by the local refresh.

Verification results are recorded alongside this file after local checks.

Local verification: new archives and current source hashes passed; proposal and
numerical implementation identity match v1. Offline preflight command passed;
`git diff --check` passed. **66 affected tests passed in 17.14s**. The full-suite
attempt stopped during collection because `datasets` is missing in the current
Python environment; the previous 180-pass receipt remains historical, not a
newly repeated full-suite pass. No dependency install or paid run performed.

[Machine-readable refresh record](path-b-readme-refresh-2026-10-13.json) and
local durable logs: `docs/validation/path-b-readme-refresh-2026-10-13/`.

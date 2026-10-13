"""Run consecutive GPU stages in one model session. No infrastructure side effects."""
from __future__ import annotations

import time

from scripts.benchmark import benchmark
from scripts.calibrate import calibrate
from scripts.common import parser, settings
from scripts.experiment import collect
from scripts.measurement import load_model


def main():
    p = parser(__doc__)
    p.add_argument("stages", nargs="+", choices=["benchmark", "calibrate", "pilot", "main"])
    p.add_argument("--units", type=int, help="Pilot ladder rung")
    args = p.parse_args()
    research, execution = settings(args)
    started = time.monotonic()
    model = load_model(research, execution)
    loaded_seconds = time.monotonic() - started
    deadline = None if execution["max_seconds"] is None else started + execution["max_seconds"]
    for stage in args.stages:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Combined GPU execution limit reached")
            execution = {**execution, "max_seconds": remaining}
        if stage == "benchmark":
            benchmark(research, execution, model=model, loaded_seconds=loaded_seconds)
        elif stage == "calibrate":
            calibrate(research, execution, model=model)
        else:
            collect(research, execution, stage, args.units, model=model)


if __name__ == "__main__":
    main()

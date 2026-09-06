"""Main runner scaffold for DSIC-3007.

Recommended sequence:
1. Load frozen model revision.
2. Load frozen evaluation position list.
3. Recover active experts and routing weights for the token.
4. Rank active experts by M0/M1/M2/M3/M4.
5. Compute baseline next-token loss.
6. Ablate high-ranked active expert only at the target token/layer.
7. Ablate low-ranked active expert only at the same token/layer.
8. Save event-level record.
"""

from pathlib import Path
import json


def write_jsonl(path: str, record: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def causal_discrimination(delta_high: float, delta_low: float) -> float:
    return delta_high - delta_low


if __name__ == "__main__":
    raise SystemExit("Implement the model-specific runner after verification E0-E1 passes.")

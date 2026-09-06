"""Dataset loading, deterministic document split, and frozen evaluation sampling."""

from dataclasses import dataclass
from typing import Iterable, List, Dict
import random


@dataclass(frozen=True)
class DocumentRecord:
    domain: str
    document_id: str
    text: str


def deterministic_document_split(records: List[DocumentRecord], seed: int = 42, calibration_ratio: float = 0.5):
    """Split at document level. Never split token positions from the same document across calibration/evaluation."""
    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)
    cut = int(len(items) * calibration_ratio)
    return items[:cut], items[cut:]


def validate_no_document_leakage(calibration: Iterable[DocumentRecord], evaluation: Iterable[DocumentRecord]) -> None:
    c = {x.document_id for x in calibration}
    e = {x.document_id for x in evaluation}
    overlap = c & e
    if overlap:
        raise ValueError(f"Document leakage detected: {len(overlap)} overlapping documents")


def select_evaluation_positions(*args, **kwargs):
    """Implement deterministic token-position sampling after tokenization and position filtering."""
    raise NotImplementedError

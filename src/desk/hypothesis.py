from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

OPS = (">", "<")
MAX_CONDITIONS = 3
MAX_HOLD_BARS = 500


class HypothesisError(ValueError):
    """The hypothesis is malformed or uses a feature that does not exist."""


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str  # ">" or "<"
    threshold: float


@dataclass(frozen=True)
class Hypothesis:
    """A testable trading rule in a deliberately small, safe language.

    All conditions must hold at the close of a bar. The position (long if direction is +1,
    short if -1) is then held for `hold_bars` bars starting from the NEXT bar.
    Agents emit this structure as data: they never run code on the evaluator.
    """

    id: str
    asset: str
    conditions: tuple[Condition, ...]
    direction: int
    hold_bars: int
    rationale: str = ""
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Hypothesis":
        try:
            return Hypothesis(
                id=str(d["id"]),
                asset=str(d["asset"]),
                conditions=tuple(
                    Condition(str(c["feature"]), str(c["op"]), float(c["threshold"])) for c in d["conditions"]
                ),
                direction=int(d["direction"]),
                hold_bars=int(d["hold_bars"]),
                rationale=str(d.get("rationale", "")),
                sources=tuple(str(s) for s in d.get("sources", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HypothesisError(f"Cannot parse hypothesis: {exc!r}") from exc


def validate(h: Hypothesis, available_features: set[str]) -> None:
    if not h.id:
        raise HypothesisError("hypothesis needs an id")
    if h.direction not in (-1, 1):
        raise HypothesisError("direction must be +1 (long) or -1 (short)")
    if not 1 <= h.hold_bars <= MAX_HOLD_BARS:
        raise HypothesisError(f"hold_bars must be between 1 and {MAX_HOLD_BARS}")
    if not 1 <= len(h.conditions) <= MAX_CONDITIONS:
        raise HypothesisError(f"between 1 and {MAX_CONDITIONS} conditions are required")
    for c in h.conditions:
        if c.op not in OPS:
            raise HypothesisError(f"operator must be one of {OPS}, got {c.op!r}")
        if not math.isfinite(c.threshold):
            raise HypothesisError("threshold must be a finite number")
        if c.feature not in available_features:
            raise HypothesisError(f"unknown feature {c.feature!r}")

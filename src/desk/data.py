from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Market:
    """Aligned bar data for one asset. Every feature must be causal: its value at bar t
    may only use information available at the close of bar t."""

    name: str
    close: np.ndarray
    features: dict[str, np.ndarray]
    periods_per_year: float
    times: np.ndarray | None = None
    synthetic: bool = False  # label synthetic data honestly, per the track rules

    def __len__(self) -> int:
        return len(self.close)

    def sliced(self, start: int, stop: int) -> "Market":
        return Market(
            name=self.name,
            close=self.close[start:stop].copy(),
            features={k: v[start:stop].copy() for k, v in self.features.items()},
            periods_per_year=self.periods_per_year,
            times=None if self.times is None else self.times[start:stop].copy(),
            synthetic=self.synthetic,
        )


@dataclass
class SealedData:
    """Splits a market into a discovery set the agents may study and a sealed holdout only
    the evaluator touches. An embargo gap stops positions held across the boundary leaking.

    The seal is enforced by the API (agents only ever receive `discovery`) and made
    auditable by a SHA-256 commitment to the holdout, logged at start-up. It is a design
    boundary inside one process. In production the evaluator would run as a separate service.
    """

    discovery: Market
    commitment: str = field(init=False)
    embargo: int = 0
    _sealed: Market | None = field(default=None, repr=False)

    @classmethod
    def split(cls, market: Market, holdout_fraction: float = 0.4, embargo: int = 200) -> "SealedData":
        if not 0.05 <= holdout_fraction <= 0.8:
            raise ValueError("holdout_fraction must be between 0.05 and 0.8")
        total = len(market)
        cut = int(total * (1 - holdout_fraction))
        if cut + embargo >= total - 100:
            raise ValueError("market too short for this holdout and embargo")
        obj = cls(discovery=market.sliced(0, cut), embargo=embargo)
        obj._sealed = market.sliced(cut + embargo, total)
        obj.commitment = _commit(obj._sealed)
        return obj

    def __repr__(self) -> str:
        n = 0 if self._sealed is None else len(self._sealed)
        return f"SealedData(commitment={self.commitment[:16]}..., discovery_bars={len(self.discovery)}, holdout_bars={n})"

    def _holdout(self) -> Market:
        """For the evaluator only. Agent-facing code must never call this."""
        assert self._sealed is not None
        return self._sealed


def _commit(m: Market) -> str:
    h = hashlib.sha256()
    h.update(m.close.tobytes())
    for key in sorted(m.features):
        h.update(key.encode())
        h.update(m.features[key].tobytes())
    return h.hexdigest()

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Market
from .hypothesis import Hypothesis


@dataclass(frozen=True)
class Costs:
    """Per-side trading costs in basis points of notional."""

    fee_bps: float = 2.0
    slippage_bps: float = 3.0

    @property
    def per_side(self) -> float:
        return (self.fee_bps + self.slippage_bps) * 1e-4


@dataclass(frozen=True)
class BacktestResult:
    ret: np.ndarray  # asset simple returns per bar
    held: np.ndarray  # position actually held over each bar (decided one bar earlier)
    gross: np.ndarray
    net: np.ndarray
    turnover: np.ndarray
    entries: int
    exposure: float


def trigger(market: Market, h: Hypothesis) -> np.ndarray:
    """Bars where every condition holds. Missing values (NaN) never trigger."""
    out = np.ones(len(market), dtype=bool)
    with np.errstate(invalid="ignore"):
        for c in h.conditions:
            x = market.features[c.feature]
            out &= (x > c.threshold) if c.op == ">" else (x < c.threshold)
    return out


def positions_from_trigger(trig: np.ndarray, direction: int, hold: int) -> np.ndarray:
    """Position decided at the close of each bar: on if any trigger in the last `hold` bars."""
    csum = np.concatenate(([0], np.cumsum(trig.astype(np.int64))))
    t = np.arange(len(trig))
    lo = np.maximum(0, t + 1 - hold)
    active = (csum[t + 1] - csum[lo]) > 0
    return active.astype(float) * direction


def run_backtest(market: Market, h: Hypothesis, costs: Costs) -> BacktestResult:
    ret = np.zeros(len(market))
    ret[1:] = market.close[1:] / market.close[:-1] - 1.0

    pos = positions_from_trigger(trigger(market, h), h.direction, h.hold_bars)
    held = np.roll(pos, 1)  # decided at close t-1, earns the return over bar t: no lookahead
    held[0] = 0.0

    turnover = np.abs(np.diff(held, prepend=0.0))
    gross = held * ret
    net = gross - costs.per_side * turnover

    prev = np.roll(held, 1)
    prev[0] = 0.0
    entries = int(np.sum((held != 0) & (held != prev)))
    return BacktestResult(
        ret=ret,
        held=held,
        gross=gross,
        net=net,
        turnover=turnover,
        entries=entries,
        exposure=float(np.mean(held != 0)),
    )

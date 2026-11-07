"""SYNTHETIC data for tests and for the null study. Every Market built here is flagged
synthetic=True. Results on it are never to be presented as real-world performance."""
from __future__ import annotations

import numpy as np

from .data import Market
from .hypothesis import Condition, Hypothesis

WINDOWS = (6, 24, 72, 168)


def build_features(close: np.ndarray, windows: tuple[int, ...] = WINDOWS) -> dict[str, np.ndarray]:
    """Causal price features: 1-bar return, momentum and rolling volatility per window."""
    n = len(close)
    logc = np.log(close)
    ret1 = np.zeros(n)
    ret1[1:] = np.diff(logc)
    c1 = np.concatenate(([0.0], np.cumsum(ret1)))
    c2 = np.concatenate(([0.0], np.cumsum(ret1**2)))
    feats: dict[str, np.ndarray] = {"ret_1": ret1}
    for w in windows:
        mom = np.full(n, np.nan)
        mom[w:] = logc[w:] - logc[:-w]
        feats[f"mom_{w}"] = mom
        s1, s2 = c1[w:] - c1[:-w], c2[w:] - c2[:-w]  # windows ending at bars w-1 .. n-1
        var = np.maximum((s2 / w - (s1 / w) ** 2) * w / (w - 1), 0.0)
        vol = np.full(n, np.nan)
        vol[w - 1 :] = np.sqrt(var)
        feats[f"vol_{w}"] = vol
    return feats


def make_market(
    n_bars: int = 40_000,
    seed: int = 0,
    planted_beta: float = 0.0,
    event_prob: float = 1 / 72,
    hold: int = 12,
    vol: float = 0.005,
) -> Market:
    """Fat-tailed hourly-style returns. If planted_beta > 0, an observable `event_tone` feature
    (think: a model's reading of a central bank press conference) genuinely drives the
    next `hold` bars: total drift = planted_beta * tone, spread evenly over those bars."""
    rng = np.random.default_rng(seed)
    ret = vol * rng.standard_t(4, n_bars) / np.sqrt(2.0)
    events = rng.random(n_bars) < event_prob
    tone = np.full(n_bars, np.nan)
    tone[events] = rng.normal(0.0, 1.0, int(events.sum()))
    if planted_beta:
        for t in np.flatnonzero(events):
            end = min(n_bars, t + 1 + hold)
            ret[t + 1 : end] += planted_beta * tone[t] / hold
    close = 100.0 * np.exp(np.cumsum(ret))
    feats = build_features(close)
    feats["event_tone"] = tone
    times = np.datetime64("2020-01-01T00:00") + np.arange(n_bars).astype("timedelta64[h]")
    return Market(
        name="SYNTHETIC",
        close=close,
        features=feats,
        periods_per_year=24 * 365,
        times=times,
        synthetic=True,
    )


def planted_hypothesis(threshold: float = 0.5, hold: int = 12) -> Hypothesis:
    return Hypothesis(
        id="planted-long-on-positive-tone",
        asset="SYNTHETIC",
        conditions=(Condition("event_tone", ">", threshold),),
        direction=1,
        hold_bars=hold,
        rationale="Synthetic fixture: tone above threshold genuinely predicts the next bars.",
    )

"""Statistics for judging a backtest honestly. All maths lives here, apart from any model."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


def sharpe(returns: np.ndarray, periods_per_year: float = 1.0) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def moments(returns: np.ndarray) -> tuple[float, float]:
    """(skewness, raw kurtosis) with kurtosis of a normal = 3."""
    r = np.asarray(returns, dtype=float)
    d = r - r.mean()
    s2 = float((d**2).mean())
    if s2 == 0:
        return 0.0, 3.0
    return float((d**3).mean() / s2**1.5), float((d**4).mean() / s2**2)


def probabilistic_sharpe(sr: float, sr_benchmark: float, n: int, skew: float, kurt: float) -> float:
    """P(true Sharpe > benchmark), adjusting for sample size, skew and fat tails.
    All Sharpe values are per-period (not annualised). Bailey & Lopez de Prado (2012)."""
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if n < 3 or denom <= 0:
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return _N.cdf(z)


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Best Sharpe you would expect from n_trials skill-less tries, given the variance of
    per-period Sharpe estimates across trials. Bailey & Lopez de Prado (2014)."""
    if n_trials <= 1:
        return 0.0
    sd = math.sqrt(max(var_sr, 0.0))
    return sd * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(returns: np.ndarray, n_trials: int, trial_sr_var: float | None = None) -> float:
    """Probability the strategy's Sharpe beats what luck alone would produce after n_trials tries.

    The variance of Sharpe estimates across trials is floored at the pure-noise value 1/(n-1),
    so the correction is never tighter than the noise level."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 3:
        return 0.0
    floor = 1.0 / (n - 1)
    var = floor if trial_sr_var is None else max(trial_sr_var, floor)
    sr0 = expected_max_sharpe(n_trials, var)
    skew, kurt = moments(r)
    return probabilistic_sharpe(sharpe(r, 1.0), sr0, n, skew, kurt)


def block_bootstrap_sharpe_ci(
    returns: np.ndarray,
    block: int,
    n_boot: int,
    alpha: float,
    periods_per_year: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Circular block-bootstrap interval for the annualised Sharpe (keeps autocorrelation)."""
    r = np.asarray(returns, dtype=float)
    t = r.size
    b = max(1, min(block, t))
    nb = math.ceil(t / b)
    out = np.empty(n_boot)
    chunk = 100
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        starts = rng.integers(0, t, size=(m, nb))
        idx = ((starts[:, :, None] + np.arange(b)[None, None, :]) % t).reshape(m, -1)[:, :t]
        sample = r[idx]
        sd = sample.std(axis=1, ddof=1)
        mean = sample.mean(axis=1)
        safe = np.where(sd > 0, sd, 1.0)
        out[s : s + m] = np.where(sd > 0, mean / safe * math.sqrt(periods_per_year), 0.0)
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def permutation_pvalue(
    held: np.ndarray,
    ret: np.ndarray,
    cost_per_side: float,
    n_perm: int,
    rng: np.random.Generator,
    min_shift: int,
) -> float:
    """One-sided p-value for mean net return. The null circularly shifts the position series
    against the returns, which keeps its autocorrelation, exposure and turnover intact."""

    def mean_net(h: np.ndarray) -> float:
        turn = np.abs(np.diff(h, prepend=0.0))
        return float(np.mean(h * ret - cost_per_side * turn))

    t = len(held)
    lo, hi = min_shift, t - min_shift
    if hi <= lo:
        return 1.0
    observed = mean_net(held)
    shifts = rng.integers(lo, hi, size=n_perm)
    null = np.array([mean_net(np.roll(held, int(k))) for k in shifts])
    return float((1 + np.sum(null >= observed)) / (1 + n_perm))


def bonferroni(p: float, n_tests: int) -> float:
    return min(1.0, p * n_tests)

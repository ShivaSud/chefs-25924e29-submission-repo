"""Scenario simulation and the three success metrics.

Each simulated run draws an economic scenario and a set of deal assumptions (synergy size,
how much is achieved, delay, integration cost, customer churn, regulatory outcome), then asks:
  1. Value:      does the deal earn back the premium and beat the cost of capital?
  2. Resilience: can the combined company carry its debt?
  3. Fit:        does it achieve the intention the user picked?
A metric's score is the share of runs that pass it. All defaults are stated assumptions,
not facts, and every one can be changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .financials import Financials

INTENTIONS = ("reduce_debt", "market_share", "economies_of_scale", "horizontal", "vertical")

# Illustrative economic scenarios: (probability, shock to EBITDA, change in interest rates).
SCENARIOS = {
    "base": (0.5, 0.03, 0.0),
    "recession": (0.2, -0.15, -0.01),
    "rate_shock": (0.2, -0.05, 0.02),
    "boom": (0.1, 0.08, 0.005),
}


@dataclass(frozen=True)
class Assumptions:
    """Defaults are placeholders for the team to tune and cite. (low, most likely, high) ranges."""

    premium: float = 0.30  # paid over the target's market value
    debt_financed: float = 0.6  # share of the purchase price funded with new debt
    cost_of_debt: float = 0.06
    wacc: float = 0.09
    tax_rate: float = 0.25
    floating_share: float = 0.5  # share of existing debt repriced when rates move
    cost_synergy_pct_target_opex: tuple = (0.02, 0.05, 0.10)  # of target operating costs (revenue - EBITDA)
    cross_industry_overlap: float = 0.4  # cost synergies scaled down when the two firms' 2-digit SIC differs
    revenue_synergy_pct_target_rev: tuple = (0.0, 0.01, 0.04)
    churn_pct_target_rev: tuple = (0.0, 0.02, 0.06)
    achievement: tuple = (0.4, 0.8, 1.0)
    integration_cost_x_runrate: tuple = (1.0, 1.5, 2.5)
    delay_probs: tuple = (0.6, 0.3, 0.1)  # P(synergies start on time / 1 year late / 2 years late)
    phase_in: tuple = (0.25, 0.6, 1.0)  # share of run-rate synergy reached in years 1, 2, 3
    p_block: float = 0.03
    p_remedy: float = 0.10
    remedy_synergy_cut: float = 0.3
    break_cost_pct_price: float = 0.01
    horizon_years: int = 5


@dataclass(frozen=True)
class Thresholds:
    """Pass/fail lines, fixed before running and shown in every report."""

    max_leverage: float = 3.5  # net debt / EBITDA in year 2
    min_interest_cover: float = 4.0  # EBITDA / interest in year 2
    strong: float = 0.8
    moderate: float = 0.5
    debt_leverage_drop: float = 0.25  # reduce_debt: leverage must fall at least this far below standalone
    scale_cost_ratio_drop: float = 0.005  # economies_of_scale: operating costs / revenue must fall 0.5 points
    max_revenue_loss: float = 0.03  # market_share: churn above 3% of target revenue counts as failure
    vertical_min_uplift: float = 0.05  # vertical: year-3 synergies at least 5% of target EBITDA


@dataclass(frozen=True)
class Metric:
    name: str
    pass_rate: float
    level: str
    main_failure: str
    worst_scenario: str
    worst_scenario_pass_rate: float


@dataclass(frozen=True)
class SimResult:
    acquirer: str
    target: str
    intention: str
    n_runs: int
    value: Metric
    resilience: Metric
    fit: Metric
    purchase_price: float
    net_value_p10: float
    net_value_median: float
    net_value_p90: float
    breakeven_runrate: float
    expected_runrate: float
    top_driver: str
    assumptions: dict
    thresholds: dict

    def metrics(self) -> tuple[Metric, Metric, Metric]:
        return self.value, self.resilience, self.fit


def level(rate: float, th: Thresholds) -> str:
    return "Strong" if rate >= th.strong else "Moderate" if rate >= th.moderate else "Weak"


def _tri(rng: np.random.Generator, t: tuple, n: int) -> np.ndarray:
    lo, mode, hi = t
    if not lo <= mode <= hi:
        raise ValueError(f"range {t} must be ordered (low, most likely, high)")
    return np.full(n, float(lo)) if lo == hi else rng.triangular(lo, mode, hi, n)


def _phase(k: np.ndarray, phase_in: tuple) -> np.ndarray:
    """Share of run-rate synergy reached k years after synergies start (k < 0: not started)."""
    out = np.where(k < 0, 0.0, 1.0)
    for i, f in enumerate(phase_in):
        out = np.where(k == i, f, out)
    return out


def _metric(name: str, passed: np.ndarray, reasons: np.ndarray, scen: np.ndarray, th: Thresholds) -> Metric:
    rate = float(passed.mean())
    failed = reasons[~passed]
    if failed.size:
        vals, counts = np.unique(failed, return_counts=True)
        main = f"{vals[np.argmax(counts)]} ({counts.max() / passed.size:.0%} of runs)"
    else:
        main = "none"
    names = list(SCENARIOS)
    by = {names[i]: float(passed[scen == i].mean()) for i in range(len(names)) if np.any(scen == i)}
    worst = min(by, key=by.get)
    if max(by.values()) - by[worst] < 0.02:
        worst = "no scenario stands out"
    return Metric(name, rate, level(rate, th), main, worst, by[min(by, key=by.get)])


def _spearman_top(target: np.ndarray, inputs: dict[str, np.ndarray]) -> str:
    def ranks(x):
        return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)

    rt = ranks(target)
    best, best_r = "none", 0.0
    for name, x in inputs.items():
        if np.ptp(x) == 0:
            continue
        r = float(np.corrcoef(rt, ranks(x))[0, 1])
        if abs(r) > abs(best_r):
            best, best_r = name, r
    return f"{best} (rank correlation {best_r:+.2f})" if best != "none" else "none"


def breakeven_runrate(tgt: Financials, a: Assumptions) -> float:
    """Pre-tax yearly synergy needed for net value = 0, on time, with typical integration cost and churn."""
    w, tax, H = a.wacc, a.tax_rate, a.horizon_years
    frac = [_phase(np.array([t - 1]), a.phase_in)[0] for t in range(1, H + 1)]
    pv_factor = sum(f / (1 + w) ** t for t, f in enumerate(frac, 1)) + frac[-1] / w / (1 + w) ** H
    annuity = sum(1 / (1 + w) ** t for t in range(1, H + 1)) + 1 / w / (1 + w) ** H
    pv_int = a.integration_cost_x_runrate[1] * (0.6 / (1 + w) + 0.4 / (1 + w) ** 2)
    churn = a.churn_pct_target_rev[1] * tgt.revenue * (tgt.ebitda / tgt.revenue)
    per_unit = (1 - tax) * (pv_factor - pv_int)
    if per_unit <= 0:
        return float("inf")
    return (a.premium * tgt.market_cap + churn * (1 - tax) * annuity) / per_unit


def simulate(
    acq: Financials,
    tgt: Financials,
    intention: str,
    a: Assumptions = Assumptions(),
    th: Thresholds = Thresholds(),
    n: int = 10_000,
    seed: int = 0,
) -> SimResult:
    if intention not in INTENTIONS:
        raise ValueError(f"intention must be one of {INTENTIONS}")
    acq.check()
    tgt.check()
    if abs(sum(a.delay_probs) - 1) > 1e-9:
        raise ValueError("delay_probs must sum to 1")

    rng = np.random.default_rng(seed)
    names = list(SCENARIOS)
    probs = np.array([SCENARIOS[s][0] for s in names])
    scen = rng.choice(len(names), n, p=probs / probs.sum())
    shock = np.array([SCENARIOS[s][1] for s in names])[scen]
    shift = np.array([SCENARIOS[s][2] for s in names])[scen]

    margin = max(tgt.ebitda, 0.0) / tgt.revenue
    same_industry = bool(acq.sic) and acq.sic[:2] == tgt.sic[:2]
    overlap = 1.0 if same_industry else a.cross_industry_overlap
    tgt_opex = tgt.revenue - tgt.ebitda
    cost_pct = _tri(rng, a.cost_synergy_pct_target_opex, n) * overlap
    rev_pct = _tri(rng, a.revenue_synergy_pct_target_rev, n)
    churn_pct = _tri(rng, a.churn_pct_target_rev, n)
    ach = _tri(rng, a.achievement, n)
    int_x = _tri(rng, a.integration_cost_x_runrate, n)
    delay = rng.choice(len(a.delay_probs), n, p=np.array(a.delay_probs))
    u = rng.random(n)
    blocked = u < a.p_block
    remedy = (~blocked) & (u < a.p_block + a.p_remedy)

    cut = np.where(remedy, 1 - a.remedy_synergy_cut, 1.0)
    planned = cost_pct * tgt_opex + rev_pct * tgt.revenue * margin
    runrate = planned * ach * cut
    churn = churn_pct * tgt.revenue * margin
    integration = int_x * planned
    price = tgt.market_cap * (1 + a.premium)
    premium_paid = tgt.market_cap * a.premium

    # ---- 1. value ----
    w, tax, H = a.wacc, a.tax_rate, a.horizon_years
    pv = np.zeros(n)
    syn = {}
    for t in range(1, H + 1):
        syn[t] = runrate * _phase(t - 1 - delay, a.phase_in) - churn
        pv += syn[t] * (1 - tax) / (1 + w) ** t
    pv += syn[H] * (1 - tax) / w / (1 + w) ** H
    pv_integration = integration * (1 - tax) * (0.6 / (1 + w) + 0.4 / (1 + w) ** 2)
    net_value = np.where(blocked, -a.break_cost_pct_price * price, pv - premium_paid - pv_integration)

    nopat3 = tgt.operating_income * (1 + shock) * (1 - tax) + syn[min(3, H)] * (1 - tax)
    roic3 = nopat3 / (price + integration)
    value_ok = (~blocked) & (net_value > 0) & (roic3 > w)
    value_why = np.where(blocked, "deal blocked", np.where(net_value <= 0, "synergies do not cover premium and costs",
                                                          "return below cost of capital"))

    # ---- 2. resilience (year 2) ----
    new_debt = np.where(blocked, 0.0, price * a.debt_financed)
    old_debt = acq.total_debt + np.where(blocked, 0.0, tgt.total_debt)
    old_interest = acq.interest_expense + np.where(blocked, 0.0, tgt.interest_expense)
    interest = old_interest + old_debt * shift * a.floating_share + new_debt * (a.cost_of_debt + shift)
    ebitda2 = np.where(blocked, acq.ebitda * (1 + shock), (acq.ebitda + tgt.ebitda) * (1 + shock) + syn[min(2, H)])
    net_debt = old_debt + new_debt - acq.cash - np.where(blocked, 0.0, tgt.cash)
    with np.errstate(divide="ignore", invalid="ignore"):
        lev = np.where(ebitda2 > 0, net_debt / ebitda2, np.inf)
        cover = np.where(interest > 0, ebitda2 / interest, np.inf)
    lev_ok, cov_ok = lev <= th.max_leverage, cover >= th.min_interest_cover
    resilience_ok = lev_ok & cov_ok
    resilience_why = np.where(~lev_ok, f"debt above {th.max_leverage}x EBITDA",
                              f"interest cover below {th.min_interest_cover}x")

    # ---- 3. strategic fit ----
    syn3 = syn[min(3, H)]
    if intention == "reduce_debt":
        standalone = acq.net_debt / np.maximum(acq.ebitda * (1 + shock), 1e-9)
        goal = lev <= standalone - th.debt_leverage_drop
        goal_text = "leverage not reduced enough"
    elif intention in ("economies_of_scale", "horizontal"):
        opex = (acq.revenue - acq.ebitda) + tgt_opex
        revenue = acq.revenue + tgt.revenue
        cost_saved = cost_pct * tgt_opex * ach * cut * _phase(3 - 1 - delay, a.phase_in)
        lost_rev = churn_pct * tgt.revenue
        after = (opex - cost_saved - lost_rev * (1 - margin)) / (revenue - lost_rev)
        goal = opex / revenue - after >= th.scale_cost_ratio_drop
        goal_text = "cost ratio not cut enough"
    elif intention == "market_share":
        goal = (~remedy) & (churn_pct <= th.max_revenue_loss)
        goal_text = "share lost to remedies or customer churn"
    else:  # vertical
        goal = syn3 >= th.vertical_min_uplift * max(tgt.ebitda, 1e-9)
        goal_text = "integration gains too small"
    fit_ok = (~blocked) & goal
    fit_why = np.where(blocked, "deal blocked", goal_text)

    expected = float(np.mean(planned * a.achievement[1]))
    return SimResult(
        acquirer=acq.ticker,
        target=tgt.ticker,
        intention=intention,
        n_runs=n,
        value=_metric("Value creation", value_ok, value_why, scen, th),
        resilience=_metric("Financial resilience", resilience_ok, resilience_why, scen, th),
        fit=_metric("Strategic fit", fit_ok, fit_why, scen, th),
        purchase_price=float(price),
        net_value_p10=float(np.percentile(net_value, 10)),
        net_value_median=float(np.median(net_value)),
        net_value_p90=float(np.percentile(net_value, 90)),
        breakeven_runrate=float(breakeven_runrate(tgt, a)),
        expected_runrate=expected,
        top_driver=_spearman_top(net_value, {
            "cost synergy size": cost_pct, "revenue synergy size": rev_pct, "customer churn": churn_pct,
            "share of synergies achieved": ach, "integration cost": int_x, "synergy delay": delay.astype(float),
            "economic scenario": shock,
        }),
        assumptions=asdict(a),
        thresholds=asdict(th),
    )

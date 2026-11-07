from __future__ import annotations

from dataclasses import replace

from .financials import Financials
from .simulate import INTENTIONS, Assumptions, SimResult, Thresholds, simulate

# Which metric matters most for each intention (used to rank candidates, never to hide a metric).
PRIORITY = {
    "reduce_debt": ("fit", "resilience", "value"),
    "market_share": ("fit", "value", "resilience"),
    "horizontal": ("fit", "value", "resilience"),
    "economies_of_scale": ("value", "fit", "resilience"),
    "vertical": ("fit", "value", "resilience"),
}


def eligible(acq: Financials, cand: Financials, intention: str, suppliers: set[str] | None) -> bool:
    """Hard filters per intention. Plain, inspectable rules."""
    if cand.ticker == acq.ticker or cand.market_cap >= acq.market_cap:
        return False
    if intention == "reduce_debt":
        return cand.ebitda > 0 and cand.leverage < 1.5
    if intention in ("market_share", "horizontal", "economies_of_scale"):
        return bool(acq.sic) and cand.sic[:2] == acq.sic[:2]
    if intention == "vertical":
        return cand.ticker in (suppliers or set())
    return False


def shortlist(
    acq: Financials,
    universe: list[Financials],
    intention: str,
    k: int = 3,
    assumptions: Assumptions = Assumptions(),
    thresholds: Thresholds = Thresholds(),
    suppliers: set[str] | None = None,
    n: int = 5000,
    seed: int = 0,
) -> list[SimResult]:
    """Top-k targets for the chosen intention, each with all three metrics.

    Vertical deals need a supplier/customer list (from verified filing quotes or the user),
    because industry codes do not reveal supply chains."""
    if intention not in INTENTIONS:
        raise ValueError(f"intention must be one of {INTENTIONS}")
    if intention == "vertical" and not suppliers:
        raise ValueError("vertical integration needs a list of known suppliers or customers")
    if intention == "reduce_debt":
        # A deal meant to cut leverage is modelled as paid in shares. Recorded in each result's assumptions.
        assumptions = replace(assumptions, debt_financed=0.0)
    results = [
        simulate(acq, c, intention, assumptions, thresholds, n=n, seed=seed)
        for c in universe
        if eligible(acq, c, intention, suppliers)
    ]
    order = PRIORITY[intention]
    results.sort(key=lambda r: tuple(getattr(r, m).pass_rate for m in order) + (r.net_value_median,), reverse=True)
    return results[:k]

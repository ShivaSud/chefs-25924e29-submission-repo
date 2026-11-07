from dataclasses import replace

import pytest

from merger.demo_data import ACQUIRER, UNIVERSE
from merger.screen import shortlist
from merger.sec import MissingData, parse_companyfacts
from merger.simulate import Assumptions, Thresholds, level, simulate

TARGET = UNIVERSE[0]  # Bramble Snacks, same industry as the acquirer


def usd(*entries):
    return {"units": {"USD": list(entries)}}


def annual(val, year, filed, form="10-K"):
    return {"start": f"{year}-01-01", "end": f"{year}-12-31", "val": val, "form": form, "filed": filed, "accn": f"a{year}"}


def instant(val, year, filed):
    return {"end": f"{year}-12-31", "val": val, "form": "10-K", "filed": filed}


FACTS = {
    "entityName": "Example Corp",
    "facts": {
        "us-gaap": {
            "Revenues": usd(
                annual(900, 2023, "2024-02-01"),
                annual(1000, 2024, "2025-02-01"),
                {"start": "2024-10-01", "end": "2024-12-31", "val": 260, "form": "10-K", "filed": "2025-02-01"},
                annual(1100, 2025, "2025-08-01", form="10-Q"),
            ),
            "OperatingIncomeLoss": usd(annual(150, 2024, "2025-02-01")),
            "DepreciationAndAmortization": usd(annual(30, 2024, "2025-02-01")),
            "SellingGeneralAndAdministrativeExpense": usd(annual(200, 2024, "2025-02-01")),
            "CashAndCashEquivalentsAtCarryingValue": usd(instant(50, 2024, "2025-02-01")),
            "LongTermDebtNoncurrent": usd(instant(300, 2024, "2025-02-01")),
            "LongTermDebtCurrent": usd(instant(40, 2024, "2025-02-01")),
            "ShortTermBorrowings": usd(instant(10, 2024, "2025-02-01")),
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2025-01-20", "val": 100, "filed": "2025-02-01"}]}}},
    },
}


def test_parser_takes_latest_full_year_from_10k_with_provenance():
    f = parse_companyfacts(FACTS, "exm", share_price=20, sic="2000")
    assert f.revenue == 1000  # not the quarter, not the 10-Q, not the older year
    assert f.ebitda == 180 and f.total_debt == 350 and f.cash == 50
    assert f.market_cap == 2000 and f.ticker == "EXM"
    assert f.provenance["revenue"]["tag"] == "Revenues" and f.provenance["revenue"]["end"] == "2024-12-31"
    assert "assumed" in f.provenance["interest_expense"]  # missing, recorded as an assumption, not hidden
    assert f.provenance["total_debt"]["sum_of"] == ["long_term_debt_noncurrent", "long_term_debt_current", "short_term_borrowings"]


def test_parser_refuses_to_guess_required_figures():
    facts = {"facts": {"us-gaap": {k: v for k, v in FACTS["facts"]["us-gaap"].items() if k != "OperatingIncomeLoss"},
                       "dei": FACTS["facts"]["dei"]}}
    with pytest.raises(MissingData, match="operating_income"):
        parse_companyfacts(facts, "exm", 20)


def test_levels():
    th = Thresholds()
    assert (level(0.85, th), level(0.6, th), level(0.2, th)) == ("Strong", "Moderate", "Weak")


def test_simulation_is_reproducible_and_three_metrics_are_rates():
    a = simulate(ACQUIRER, TARGET, "economies_of_scale", n=3000, seed=1)
    b = simulate(ACQUIRER, TARGET, "economies_of_scale", n=3000, seed=1)
    assert a == b
    for m in a.metrics():
        assert 0.0 <= m.pass_rate <= 1.0 and m.level in ("Strong", "Moderate", "Weak")
    assert a.net_value_p10 <= a.net_value_median <= a.net_value_p90


def test_lower_premium_raises_value_score():
    cheap = simulate(ACQUIRER, TARGET, "economies_of_scale", Assumptions(premium=0.0), n=3000)
    dear = simulate(ACQUIRER, TARGET, "economies_of_scale", Assumptions(premium=0.6), n=3000)
    assert cheap.value.pass_rate > dear.value.pass_rate
    assert cheap.breakeven_runrate < dear.breakeven_runrate


def test_heavy_debt_hurts_resilience_and_rate_shocks_are_the_weak_spot():
    indebted = replace(ACQUIRER, total_debt=11_000, interest_expense=600)
    r = simulate(indebted, TARGET, "economies_of_scale", Assumptions(debt_financed=1.0), n=5000)
    base = simulate(ACQUIRER, TARGET, "economies_of_scale", n=5000)
    assert r.resilience.pass_rate < base.resilience.pass_rate
    assert r.resilience.worst_scenario in ("rate_shock", "recession")


def test_blocked_deals_always_fail_fit():
    r = simulate(ACQUIRER, TARGET, "market_share", Assumptions(p_block=1.0, p_remedy=0.0), n=1000)
    assert r.fit.pass_rate == 0.0 and r.value.pass_rate == 0.0
    assert "blocked" in r.fit.main_failure


def test_bad_inputs_rejected():
    with pytest.raises(ValueError):
        simulate(ACQUIRER, TARGET, "world_domination")
    with pytest.raises(ValueError):
        simulate(ACQUIRER, TARGET, "horizontal", Assumptions(achievement=(0.9, 0.5, 1.0)))
    with pytest.raises(ValueError):
        simulate(ACQUIRER, replace(TARGET, share_price=0), "horizontal")


def test_shortlist_rules():
    top = shortlist(ACQUIRER, UNIVERSE, "economies_of_scale", n=1000)
    assert len(top) == 3 and all(r.target in {"BRM", "HRW", "KST"} for r in top)  # same 2-digit SIC only
    assert all(r.target != ACQUIRER.ticker for r in top)
    debt = shortlist(ACQUIRER, UNIVERSE, "reduce_debt", n=1000)
    assert all(r.assumptions["debt_financed"] == 0.0 for r in debt)  # share-funded, and recorded
    assert debt[0].fit.pass_rate >= debt[-1].fit.pass_rate
    with pytest.raises(ValueError):
        shortlist(ACQUIRER, UNIVERSE, "vertical")
    assert [r.target for r in shortlist(ACQUIRER, UNIVERSE, "vertical", suppliers={"LRK"}, n=1000)] == ["LRK"]

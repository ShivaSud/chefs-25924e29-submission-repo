from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from finagent.agent import AgentContext, dispatch, run_agent
from finagent.analysis import detect_recurring, find_duplicates, variable_daily_spend
from finagent.forecast import forecast_to_payday, next_monthly_date
from finagent.ingest import load_csv
from finagent.models import normalise_merchant

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"
AS_OF = date(2026, 11, 7)
PAYDAY = date(2026, 11, 28)


@pytest.fixture(scope="module")
def txns():
    return load_csv(SAMPLE)


@pytest.fixture(scope="module")
def recurring(txns):
    return detect_recurring(txns)


def test_normalise_merchant():
    assert normalise_merchant("TESCO STORES 1234 LONDON GB") == "tesco stores"
    assert normalise_merchant("SPOTIFY P3A1B2C") == "spotify"
    assert normalise_merchant("NETFLIX.COM") == "netflix"
    assert normalise_merchant("ICLOUD 2TB") == "icloud"


def test_recurring_found_and_variable_spend_excluded(recurring):
    merchants = {r.merchant for r in recurring}
    assert {"netflix", "spotify", "puregym", "headspace", "adobe creative cloud", "icloud"} <= merchants
    assert not any(m.startswith(("tesco", "tfl", "lidl", "pret")) for m in merchants)


def test_essentials_flagged(recurring):
    by_name = {r.merchant: r for r in recurring}
    assert by_name["landlord rent standing order"].is_essential
    assert not by_name["netflix"].is_essential


def test_duplicate_found(txns):
    dups = find_duplicates(txns)
    assert any(d.merchant == "uber eats" and d.amount == 23.40 for d in dups)


def test_forecast_shortfall_and_committed(txns, recurring):
    daily = variable_daily_spend(txns, recurring, AS_OF)
    low = forecast_to_payday(100.0, AS_OF, PAYDAY, recurring, daily)
    assert low.shortfall and low.min_balance < 0
    # icloud 6.99 + netflix 10.99 + puregym 39.99 + three 18.00 + adobe 19.97 + headspace 9.99
    assert low.committed_total == pytest.approx(105.93, abs=0.01)
    assert any("netflix" in e for day in low.days for e in day.events)
    rich = forecast_to_payday(5000.0, AS_OF, PAYDAY, recurring, daily)
    assert not rich.shortfall


def test_forecast_rejects_past_payday(recurring):
    with pytest.raises(ValueError):
        forecast_to_payday(100.0, AS_OF, AS_OF, recurring, 10.0)


def test_next_monthly_date():
    assert next_monthly_date(date(2026, 11, 7), 28) == date(2026, 11, 28)
    assert next_monthly_date(date(2026, 11, 28), 28) == date(2026, 12, 28)
    assert next_monthly_date(date(2026, 1, 31), 31) == date(2026, 2, 28)


def test_dispatch_whatif_and_actions(txns):
    ctx = AgentContext(txns, balance=480.0, payday=PAYDAY, as_of=AS_OF)
    base = dispatch(ctx, "forecast_to_payday", {})
    cut = dispatch(ctx, "forecast_to_payday", {"cancel_merchants": ["headspace", "adobe"]})
    assert cut["lowest_balance"] > base["lowest_balance"]

    result = dispatch(ctx, "propose_action", {"kind": "cancel_subscription", "merchant": "headspace"})
    assert result["status"] == "pending" and "headspace" in result["summary"].lower()
    assert len(ctx.queue.pending()) == 1
    assert "error" in dispatch(ctx, "propose_action", {"kind": "cancel_subscription", "merchant": "landlord"})
    assert "error" in dispatch(ctx, "nope", {})

    ctx.queue.approve(1)
    assert ctx.queue.items[0].status == "approved"


def test_agent_loop_with_fake_client(txns):
    ctx = AgentContext(txns, balance=480.0, payday=PAYDAY, as_of=AS_OF)
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            block = SimpleNamespace(type="tool_use", id="t1", name="list_recurring", input={})
            return SimpleNamespace(stop_reason="tool_use", content=[block])
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Done.")])

    client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    text, history = run_agent(ctx, "What recurs?", client=client)
    assert text == "Done."
    assert history[-2]["content"][0]["tool_use_id"] == "t1"
    assert "netflix" in history[-2]["content"][0]["content"]

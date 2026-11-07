import json

import pytest

from desk.audit import AuditLog
from desk.risk_gate import Order, PaperBroker, RiskLimits

LIMITS = RiskLimits(
    allowed_symbols=frozenset({"BTCUSDT", "ETHUSDT"}),
    max_order_notional=10_000,
    max_position_notional=20_000,
    max_gross_notional=30_000,
    max_daily_loss=1_000,
)


def broker():
    audit = AuditLog()
    b = PaperBroker(LIMITS, audit)
    b.mark("BTCUSDT", 100.0)
    b.mark("ETHUSDT", 50.0)
    b.start_day()
    return b, audit


def test_audit_chain_detects_tampering():
    log = AuditLog(clock=lambda: "t")
    for i in range(5):
        log.append("event", {"i": i})
    assert log.verify() == (True, "ok")
    tampered = json.loads(json.dumps(log.records))
    tampered[2]["payload"]["i"] = 99
    assert not log.verify(tampered)[0]
    deleted = [r for k, r in enumerate(log.records) if k != 1]
    assert not log.verify(deleted)[0]


def test_audit_persists_and_reloads(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("a", {"x": 1})
    log.append("b", {"y": 2})
    reloaded = AuditLog(path)
    assert len(reloaded.records) == 2 and reloaded.verify()[0]
    reloaded.append("c", {"z": 3})
    assert AuditLog(path).verify()[0]
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace('"x":1', '"x":2')
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        AuditLog(path)


def test_normal_order_fills_and_is_logged():
    b, audit = broker()
    d = b.submit(Order("BTCUSDT", "buy", 50, 100.5))
    assert d.allowed and b.positions["BTCUSDT"] == 50
    assert audit.of_kind("order_decision") and audit.of_kind("fill")


@pytest.mark.parametrize(
    "order, fragment",
    [
        (Order("DOGEUSDT", "buy", 1, 100.0), "allowed list"),
        (Order("BTCUSDT", "buy", 1, 150.0), "fat-finger"),
        (Order("BTCUSDT", "buy", 500, 100.0), "per-order"),
        (Order("BTCUSDT", "buy", -3, 100.0), "positive finite"),
        (Order("BTCUSDT", "buy", float("nan"), 100.0), "positive finite"),
        (Order("BTCUSDT", "hold", 1, 100.0), "unknown side"),
    ],
)
def test_bad_orders_are_blocked_in_code(order, fragment):
    b, audit = broker()
    d = b.submit(order)
    assert not d.allowed and any(fragment in r for r in d.reasons)
    assert not b.positions and not audit.of_kind("fill")
    assert audit.of_kind("order_decision")[0]["payload"]["allowed"] is False


def test_position_and_gross_limits():
    b, _ = broker()
    assert b.submit(Order("BTCUSDT", "buy", 100, 100.0)).allowed  # 10k
    assert b.submit(Order("BTCUSDT", "buy", 100, 100.0)).allowed  # 20k, at the position limit
    d = b.submit(Order("BTCUSDT", "buy", 10, 100.0))
    assert not d.allowed and any("position limit" in r for r in d.reasons)
    assert b.submit(Order("ETHUSDT", "buy", 180, 50.0)).allowed  # 9k, gross 29k
    d = b.submit(Order("ETHUSDT", "buy", 60, 50.0))  # gross would be 32k
    assert not d.allowed and any("gross exposure" in r for r in d.reasons)


def test_kill_switch_latches_allows_reducing_orders_and_needs_named_reset():
    b, audit = broker()
    assert b.submit(Order("BTCUSDT", "buy", 100, 100.0)).allowed
    b.mark("BTCUSDT", 85.0)  # about 1.5k loss on 100 units, beyond the 1k limit
    assert b.halted and audit.of_kind("kill_switch")

    d = b.submit(Order("BTCUSDT", "buy", 1, 85.0))
    assert not d.allowed and any("kill switch" in r for r in d.reasons)
    assert b.submit(Order("BTCUSDT", "sell", 100, 85.0)).allowed  # flattening is always allowed
    assert b.positions["BTCUSDT"] == 0

    b.mark("BTCUSDT", 100.0)  # price recovering does not release the latch
    assert b.halted
    with pytest.raises(ValueError):
        b.reset_halt("  ")
    b.reset_halt("risk officer")
    assert not b.halted and audit.of_kind("kill_switch_reset")[0]["payload"]["operator"] == "risk officer"
    assert audit.verify()[0]

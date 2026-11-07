import numpy as np
import pytest

from desk.backtest import Costs, positions_from_trigger, run_backtest
from desk.data import Market
from desk.hypothesis import Condition, Hypothesis, HypothesisError, validate


def tiny_market(close, signal):
    return Market(name="T", close=np.array(close, float), features={"sig": np.array(signal, float)},
                  periods_per_year=252, synthetic=True)


def test_position_window():
    trig = np.array([0, 0, 0, 1, 0, 0, 0], dtype=bool)
    pos = positions_from_trigger(trig, 1, hold=2)
    assert list(pos) == [0, 0, 0, 1, 1, 0, 0]


def test_no_lookahead_signal_earns_next_bar_only():
    # Signal fires at bar 2. The +10% move on bar 3 is earned; the move on bar 2 is not.
    close = [100, 100, 110, 121, 121, 121]
    m = tiny_market(close, [0, 0, 1, 0, 0, 0])
    h = Hypothesis("h", "T", (Condition("sig", ">", 0.5),), 1, 1)
    res = run_backtest(m, h, Costs(0, 0))
    assert res.gross[2] == 0.0  # bar-2 return not earned despite the signal firing at its close
    assert res.gross[3] == pytest.approx(0.10)
    assert res.entries == 1


def test_costs_charged_on_entry_and_exit():
    close = [100, 100, 100, 100, 100, 100]
    m = tiny_market(close, [0, 1, 0, 0, 0, 0])
    h = Hypothesis("h", "T", (Condition("sig", ">", 0.5),), 1, 1)
    res = run_backtest(m, h, Costs(fee_bps=5, slippage_bps=5))
    assert res.net.sum() == pytest.approx(-2 * 10e-4)  # one entry plus one exit at 10 bps each


def test_nan_features_never_trigger():
    m = tiny_market([100, 101, 102, 103], [np.nan, np.nan, 2.0, np.nan])
    h = Hypothesis("h", "T", (Condition("sig", ">", 0.5),), 1, 1)
    assert run_backtest(m, h, Costs()).entries == 1


def test_validation():
    ok = Hypothesis("h", "T", (Condition("sig", ">", 1.0),), 1, 5)
    validate(ok, {"sig"})
    with pytest.raises(HypothesisError):
        validate(ok, {"other"})
    with pytest.raises(HypothesisError):
        validate(Hypothesis("h", "T", (Condition("sig", "==", 1.0),), 1, 5), {"sig"})
    with pytest.raises(HypothesisError):
        validate(Hypothesis("h", "T", (Condition("sig", ">", 1.0),), 2, 5), {"sig"})
    with pytest.raises(HypothesisError):
        validate(Hypothesis("h", "T", (Condition("sig", ">", float("nan")),), 1, 5), {"sig"})
    with pytest.raises(HypothesisError):
        validate(Hypothesis("h", "T", (), 1, 5), {"sig"})


def test_from_dict_roundtrip_and_bad_input():
    h = Hypothesis("h", "T", (Condition("sig", ">", 1.0),), -1, 5, "why", ("doc1",))
    assert Hypothesis.from_dict(h.to_dict()) == h
    with pytest.raises(HypothesisError):
        Hypothesis.from_dict({"id": "x"})

import numpy as np
import pytest

from desk.audit import AuditLog
from desk.data import SealedData
from desk.evaluator import BudgetExhausted, DuplicateHypothesis, SealedEvaluator
from desk.hypothesis import Condition, Hypothesis, HypothesisError
from desk.study import null_false_positive_study
from desk.synthetic import make_market, planted_hypothesis


@pytest.fixture(scope="module")
def planted():
    market = make_market(seed=1, planted_beta=0.008)
    sealed = SealedData.split(market)
    audit = AuditLog()
    return sealed, audit, SealedEvaluator(sealed, audit, max_trials=20)


def test_true_signal_is_accepted_wrong_way_rejected(planted):
    sealed, audit, ev = planted
    good = ev.evaluate(planted_hypothesis())
    assert good.accepted and good.metrics["net_sharpe"] > 1 and good.metrics["adj_p"] <= 0.05
    wrong = ev.evaluate(Hypothesis("wrong-way", "SYNTHETIC", (Condition("event_tone", ">", 0.5),), -1, 12))
    assert not wrong.accepted and wrong.metrics["net_sharpe"] < 0


def test_seal_hides_holdout_and_commitment_is_stable(planted):
    sealed, audit, ev = planted
    total = 40_000
    assert len(sealed.discovery) + sealed.embargo + len(sealed._holdout()) == total
    assert sealed.commitment[:16] in repr(sealed)
    assert "close" not in repr(sealed)
    assert audit.of_kind("evaluator_init")[0]["payload"]["holdout_commitment"] == sealed.commitment


def test_duplicate_budget_and_bad_feature():
    market = make_market(n_bars=20_000, seed=3)
    sealed = SealedData.split(market)
    audit = AuditLog()
    ev = SealedEvaluator(sealed, audit, max_trials=2, n_perm=50, n_boot=20)
    h = Hypothesis("a", "SYNTHETIC", (Condition("mom_24", ">", 0.0),), 1, 6)
    ev.evaluate(h)
    with pytest.raises(DuplicateHypothesis):
        ev.evaluate(h)
    with pytest.raises(HypothesisError):  # malformed ideas do not burn budget
        ev.evaluate(Hypothesis("bad", "SYNTHETIC", (Condition("nope", ">", 0.0),), 1, 6))
    assert ev.trials_used == 1
    ev.evaluate(Hypothesis("b", "SYNTHETIC", (Condition("mom_24", "<", 0.0),), 1, 6))
    with pytest.raises(BudgetExhausted):
        ev.evaluate(Hypothesis("c", "SYNTHETIC", (Condition("vol_24", ">", 0.0),), 1, 6))
    assert len(audit.of_kind("evaluation_refused")) == 2
    assert audit.verify()[0]


def test_every_verdict_is_logged_with_its_trace(planted):
    sealed, audit, ev = planted
    verdicts = audit.of_kind("verdict")
    assert len(verdicts) == ev.trials_used
    assert verdicts[0]["payload"]["hypothesis_fingerprint"] == planted_hypothesis().fingerprint()
    assert verdicts[0]["payload"]["holdout_commitment"] == sealed.commitment


def test_null_study_pipeline_beats_naive_testing():
    out = null_false_positive_study(n_hypotheses=100, seed=0)
    assert out["audit_ok"]
    assert out["naive_accepted"] >= 1  # a quick backtest finds "strategies" in pure noise
    assert out["pipeline_accepted"] == 0  # the sealed, corrected pipeline does not
    assert out["best_net_sharpe"] > 0.5  # and the lucky winner looked tempting

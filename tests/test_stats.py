import numpy as np
import pytest

from desk.stats import (
    block_bootstrap_sharpe_ci,
    bonferroni,
    deflated_sharpe,
    expected_max_sharpe,
    permutation_pvalue,
    probabilistic_sharpe,
    sharpe,
)


def test_sharpe_basic():
    assert sharpe(np.array([0.01, 0.02, 0.015, 0.012])) > 0
    assert sharpe(np.zeros(10)) == 0.0
    assert sharpe(np.array([0.01])) == 0.0
    r = np.array([0.01, -0.01, 0.02, 0.0])
    assert sharpe(r, 252) == pytest.approx(sharpe(r) * np.sqrt(252))


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1, 0.01) == 0.0
    values = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert values == sorted(values) and values[0] > 0
    assert expected_max_sharpe(50, 0.04) == pytest.approx(2 * expected_max_sharpe(50, 0.01))


def test_deflated_sharpe_falls_as_trials_rise():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.01, 4000)
    one, ten, thousand = (deflated_sharpe(r, n) for n in (1, 10, 1000))
    assert one > ten > thousand
    assert one == pytest.approx(probabilistic_sharpe(sharpe(r), 0.0, len(r), *__import__("desk.stats", fromlist=["moments"]).moments(r)))


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, 3000)
    lo, hi = block_bootstrap_sharpe_ci(r, block=10, n_boot=300, alpha=0.05, periods_per_year=252, rng=rng)
    assert lo < sharpe(r, 252) < hi


def test_permutation_pvalues_roughly_uniform_under_null():
    rng = np.random.default_rng(2)
    ps = []
    for _ in range(60):
        ret = rng.normal(0, 0.01, 1500)
        held = (rng.random(1500) < 0.2).astype(float)  # position unrelated to returns
        ps.append(permutation_pvalue(held, ret, 0.0, 200, rng, min_shift=50))
    ps = np.array(ps)
    assert 0.35 < ps.mean() < 0.65
    assert (ps < 0.05).mean() < 0.15


def test_permutation_detects_real_signal():
    rng = np.random.default_rng(3)
    ret = rng.normal(0, 0.01, 3000)
    held = (ret > 0.005).astype(float)  # position built from the same bar's return: a planted edge
    assert permutation_pvalue(held, ret, 0.0, 400, rng, min_shift=100) < 0.01


def test_bonferroni_caps_at_one():
    assert bonferroni(0.01, 20) == pytest.approx(0.2)
    assert bonferroni(0.2, 20) == 1.0

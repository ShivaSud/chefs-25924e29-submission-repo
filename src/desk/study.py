"""The null study: how often does each testing procedure 'discover' a strategy in pure noise?

Headline statistic for the deck. Runs on SYNTHETIC data with no real signal, so every
accepted idea is a false positive by construction."""
from __future__ import annotations

import numpy as np

from .audit import AuditLog
from .data import Market, SealedData
from .evaluator import AcceptanceRule, SealedEvaluator
from .hypothesis import Condition, Hypothesis
from .synthetic import make_market

NULL_FEATURES = ["ret_1", "mom_6", "mom_24", "mom_72", "mom_168", "vol_6", "vol_24", "vol_72", "vol_168"]


def random_hypothesis(rng: np.random.Generator, discovery: Market, i: int) -> Hypothesis:
    """A random but sensible rule: threshold at a discovery-set quantile so it trades often."""
    feature = str(rng.choice(NULL_FEATURES))
    q = float(rng.choice([0.1, 0.2, 0.8, 0.9]))
    threshold = float(np.nanquantile(discovery.features[feature], q))
    return Hypothesis(
        id=f"random-{i:04d}",
        asset=discovery.name,
        conditions=(Condition(feature, ">" if q > 0.5 else "<", threshold),),
        direction=int(rng.choice([-1, 1])),
        hold_bars=int(rng.choice([6, 12, 24, 48])),
        rationale="random idea for the null study",
    )


def null_false_positive_study(
    n_hypotheses: int = 100,
    n_bars: int = 40_000,
    seed: int = 0,
    n_perm: int = 200,
    n_boot: int = 100,
) -> dict:
    market = make_market(n_bars=n_bars, seed=seed, planted_beta=0.0)
    sealed = SealedData.split(market)
    audit = AuditLog()
    ev = SealedEvaluator(sealed, audit, AcceptanceRule(), max_trials=n_hypotheses, n_perm=n_perm, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed + 1)

    naive = pipeline = 0
    sharpes = []
    for i in range(n_hypotheses):
        v = ev.evaluate(random_hypothesis(rng, sealed.discovery, i))
        sharpes.append(v.metrics["net_sharpe"])
        # What a typical quick backtest does: positive net Sharpe and an unadjusted p < 0.05.
        if v.checks["enough_entries"] and v.metrics["net_sharpe"] > 0 and v.metrics["perm_p"] < 0.05:
            naive += 1
        if v.accepted:
            pipeline += 1
    return {
        "n_hypotheses": n_hypotheses,
        "naive_accepted": naive,
        "pipeline_accepted": pipeline,
        "best_net_sharpe": float(np.max(sharpes)),
        "median_net_sharpe": float(np.median(sharpes)),
        "audit_ok": audit.verify()[0],
    }

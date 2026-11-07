from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .audit import AuditLog
from .backtest import Costs, run_backtest
from .data import SealedData
from .hypothesis import Hypothesis, validate
from .stats import (
    block_bootstrap_sharpe_ci,
    bonferroni,
    deflated_sharpe,
    permutation_pvalue,
    sharpe,
)


class BudgetExhausted(RuntimeError):
    """The holdout has been used as many times as the pre-registered budget allows."""


class DuplicateHypothesis(RuntimeError):
    """A hypothesis id was submitted twice. A tweaked idea must be a new, counted trial."""


@dataclass(frozen=True)
class AcceptanceRule:
    """Fixed before any test is run and written to the audit log at start-up."""

    min_entries: int = 30
    min_net_sharpe: float = 0.0
    min_dsr: float = 0.95
    max_adj_p: float = 0.05


@dataclass(frozen=True)
class Verdict:
    hypothesis_id: str
    accepted: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]
    checks: dict[str, bool]
    trial_index: int
    trace: dict[str, Any] = field(repr=False)


class SealedEvaluator:
    """The only code that ever reads the holdout. Agents get a Verdict, never the data.

    Limits enforced in code: a fixed holdout budget (`max_trials`), no resubmitting an id,
    realistic costs, a rule fixed in advance, and an audit record for every call."""

    def __init__(
        self,
        sealed: SealedData,
        audit: AuditLog,
        rule: AcceptanceRule = AcceptanceRule(),
        costs: Costs = Costs(),
        max_trials: int = 20,
        n_perm: int = 2000,
        n_boot: int = 1000,
        seed: int = 0,
    ):
        self.sealed, self.audit, self.rule, self.costs = sealed, audit, rule, costs
        self.max_trials, self.n_perm, self.n_boot = max_trials, n_perm, n_boot
        self._rng = np.random.default_rng(seed)
        self._trials = 0
        self._seen: set[str] = set()
        self._trial_sharpes: list[float] = []
        audit.append(
            "evaluator_init",
            {
                "holdout_commitment": sealed.commitment,
                "holdout_bars": len(sealed._holdout()),
                "embargo_bars": sealed.embargo,
                "rule": asdict(rule),
                "costs": asdict(costs),
                "max_trials": max_trials,
                "synthetic_data": sealed._holdout().synthetic,
            },
        )

    @property
    def trials_used(self) -> int:
        return self._trials

    def evaluate(self, h: Hypothesis) -> Verdict:
        if self._trials >= self.max_trials:
            self.audit.append("evaluation_refused", {"id": h.id, "reason": "holdout budget exhausted"})
            raise BudgetExhausted(f"holdout budget of {self.max_trials} trials is used up")
        if h.id in self._seen:
            self.audit.append("evaluation_refused", {"id": h.id, "reason": "duplicate id"})
            raise DuplicateHypothesis(f"hypothesis {h.id!r} was already evaluated")

        holdout = self.sealed._holdout()
        validate(h, set(holdout.features))  # malformed ideas are rejected without costing a trial
        self._seen.add(h.id)
        self._trials += 1

        res = run_backtest(holdout, h, self.costs)
        ppy = holdout.periods_per_year
        sr_gross, sr_net = sharpe(res.gross, ppy), sharpe(res.net, ppy)
        self._trial_sharpes.append(sharpe(res.net, 1.0))
        trial_var = float(np.var(self._trial_sharpes, ddof=1)) if len(self._trial_sharpes) >= 2 else None

        dsr = deflated_sharpe(res.net, self._trials, trial_var)
        ci_lo, ci_hi = block_bootstrap_sharpe_ci(res.net, max(h.hold_bars, 5), self.n_boot, 0.05, ppy, self._rng)
        p = permutation_pvalue(
            res.held, res.ret, self.costs.per_side, self.n_perm, self._rng,
            min_shift=max(2 * h.hold_bars, len(res.net) // 20),
        )
        p_adj = bonferroni(p, self.max_trials)  # spend alpha across the whole pre-registered budget

        equity = np.cumprod(1.0 + res.net)
        max_dd = float(np.min(equity / np.maximum.accumulate(equity) - 1.0))

        checks = {
            "enough_entries": res.entries >= self.rule.min_entries,
            "net_sharpe_positive": sr_net > self.rule.min_net_sharpe,
            "deflated_sharpe": dsr >= self.rule.min_dsr,
            "multiple_testing_p": p_adj <= self.rule.max_adj_p,
        }
        metrics = {
            "entries": float(res.entries),
            "exposure": res.exposure,
            "gross_sharpe": sr_gross,
            "net_sharpe": sr_net,
            "net_sharpe_ci_lo": ci_lo,
            "net_sharpe_ci_hi": ci_hi,
            "cost_drag_ann": float((res.gross.mean() - res.net.mean()) * ppy),
            "perm_p": p,
            "adj_p": p_adj,
            "deflated_sharpe": dsr,
            "max_drawdown": max_dd,
            "n_trials_so_far": float(self._trials),
        }
        reasons = tuple(f"failed {name}" for name, ok in checks.items() if not ok)
        accepted = all(checks.values())
        trace = {
            "hypothesis": h.to_dict(),
            "hypothesis_fingerprint": h.fingerprint(),
            "holdout_commitment": self.sealed.commitment,
            "holdout_bars": len(res.net),
            "metrics": metrics,
            "checks": checks,
            "accepted": accepted,
        }
        self.audit.append("verdict", trace)
        return Verdict(h.id, accepted, reasons, metrics, checks, self._trials, trace)

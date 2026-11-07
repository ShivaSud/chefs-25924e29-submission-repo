# SealedDesk

AgentVerse 2026, Finance track ("Trusted Markets"). An autonomous research desk that is built not to fool itself.

Agents (Gemini) propose trading hypotheses from filings, news, charts and central-bank or earnings audio.
Plain code the agents cannot read then tests each idea on **sealed held-out data** with realistic costs and
multiple-testing corrections, enforces **risk limits**, and records every action in a **tamper-evident audit chain**.
Paper trading only. Never connected to real money.

## Status

| Piece | State |
| --- | --- |
| Hypothesis language, sealed split, cost-aware backtest | built, tested |
| Deflated Sharpe, permutation test, block-bootstrap CI, Bonferroni over a fixed trial budget | built, tested |
| Hash-chained audit log | built, tested |
| Paper broker with risk gate and kill switch | built, tested |
| Null study (false positives on pure noise) | built, run |
| Gemini hypothesis agent and skeptic agent | next |
| Multimodal central-bank / earnings signal (Gemini on audio and video) | next |
| Contamination test (mask names and dates, measure the change) | next |
| Real price data and live paper-trading loop | next |

## Quickstart

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest                      # 29 tests, about 4 seconds, no API key needed
python scripts/null_study.py --seeds 20    # the headline false-positive study, about 1 minute
```

## How it works

| Module | Job |
| --- | --- |
| `hypothesis.py` | A small, safe rule language. Agents emit rules as data and never run code on the evaluator |
| `data.py` | `SealedData` splits discovery from a sealed holdout, with an embargo gap and a SHA-256 commitment |
| `backtest.py` | Positions decided at one close earn the next bar's return (no lookahead), with per-side costs |
| `stats.py` | All the maths, kept apart from any model: deflated Sharpe, permutation p-value, bootstrap interval |
| `evaluator.py` | The only code that reads the holdout. Fixed acceptance rule, fixed trial budget, one verdict per idea |
| `audit.py` | Append-only, hash-chained log. Editing or deleting any record breaks every later hash |
| `risk_gate.py` | Position, order, gross and fat-finger limits, plus a latching daily-loss kill switch |
| `study.py` | The null study |

**Traceable reasoning.** Every verdict is logged with the hypothesis fingerprint, the holdout commitment, every metric
and each pass/fail check. The model never computes a number.

**Unseen data.** The evaluator tests only on the holdout. The accept rule is fixed in advance and logged at start-up:
at least 30 entries, positive net Sharpe, deflated Sharpe at least 0.95, and a Bonferroni-adjusted permutation p of at most 0.05
across the whole pre-registered trial budget. Retrying an idea means a new id, which costs a trial. When the budget is
spent the evaluator refuses.

**Risk controls.** Limits are enforced in the broker, whatever an agent says. A daily-loss breach latches a kill switch:
only risk-reducing orders pass until a named human resets it. Every decision, fill and reset is in the audit chain.

## Headline result (synthetic noise)

On 2,000 random ideas tested against pure noise (20 runs of 100 ideas), a naive backtest (positive net Sharpe and an
unadjusted p < 0.05) accepted 65, and would have shipped at least one "strategy" in 19 of 20 runs. The sealed pipeline accepted
0 of 2,000. The median best Sharpe among 100 random ideas was 1.43.

On synthetic data with a planted signal, the pipeline accepts the true rule (net Sharpe about 4 at the planted strength used
in the tests) and rejects the same rule traded the wrong way.

## Limits, stated plainly

- **Synthetic data.** Everything above is on synthetic data we generated (fat-tailed returns, plus a planted effect in the signal
  fixture). It shows the machinery works, not that any strategy earns money. Synthetic results are not real-world performance.
- **Zero false positives is by design, not a power claim.** Bonferroni across the trial budget controls false positives strictly,
  so it can miss weak real signals (in our probe, a weaker planted effect was accepted long but not short). A power curve is still to do.
- **The seal is a design boundary inside one process**, made auditable by the logged commitment. In production the evaluator
  would be a separate service.
- Deflated Sharpe assumes independent trials. Overlapping holds are handled by the permutation test and block bootstrap, not by DSR.
- Paper trading only. No real financial records, account details or logins are used.

"""Aggregate the null study over many seeds. Run from the repo root:

    python scripts/null_study.py --seeds 20 --ideas 100

Prints how often a naive backtest vs the sealed pipeline 'discovers' a strategy in pure
noise. SYNTHETIC data, no real signal: every acceptance is a false positive.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from desk.study import null_false_positive_study  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--ideas", type=int, default=100)
    args = ap.parse_args()

    runs = [null_false_positive_study(n_hypotheses=args.ideas, seed=s) for s in range(args.seeds)]
    total = args.seeds * args.ideas
    naive = sum(r["naive_accepted"] for r in runs)
    pipe = sum(r["pipeline_accepted"] for r in runs)
    summary = {
        "data": "SYNTHETIC pure noise, no real signal",
        "random_ideas_tested": total,
        "naive_false_positives": naive,
        "naive_false_positive_rate": naive / total,
        "pipeline_false_positives": pipe,
        "pipeline_false_positive_rate": pipe / total,
        "runs_where_pipeline_accepted_anything": sum(r["pipeline_accepted"] > 0 for r in runs),
        "runs_where_naive_accepted_anything": sum(r["naive_accepted"] > 0 for r in runs),
        "median_best_net_sharpe_per_run": sorted(r["best_net_sharpe"] for r in runs)[len(runs) // 2],
        "all_audit_chains_verified": all(r["audit_ok"] for r in runs),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

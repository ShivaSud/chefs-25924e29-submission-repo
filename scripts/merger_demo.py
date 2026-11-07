"""Run the merger screener on SYNTHETIC companies.  From the repo root:

    python scripts/merger_demo.py --intention economies_of_scale

Intentions: reduce_debt, market_share, economies_of_scale, horizontal, vertical
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from merger.demo_data import ACQUIRER, UNIVERSE  # noqa: E402
from merger.screen import shortlist  # noqa: E402
from merger.simulate import INTENTIONS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intention", choices=INTENTIONS, default="economies_of_scale")
    args = ap.parse_args()

    suppliers = {"LRK"} if args.intention == "vertical" else None
    results = shortlist(ACQUIRER, UNIVERSE, args.intention, suppliers=suppliers)
    print(f"SYNTHETIC DEMO DATA. Acquirer: {ACQUIRER.name}. Intention: {args.intention}\n")
    if not results:
        print("No eligible candidates for this intention.")
        return
    for r in results:
        name = next(c.name for c in UNIVERSE if c.ticker == r.target)
        print(f"== {name}  (price ${r.purchase_price:,.0f}M incl. premium)")
        for m in r.metrics():
            weak = (m.worst_scenario if m.worst_scenario.startswith("no ")
                    else f"weakest in {m.worst_scenario} ({m.worst_scenario_pass_rate:.0%})")
            print(f"   {m.name:<22} {m.level:<9} {m.pass_rate:>5.0%}   main failure: {m.main_failure}; {weak}")
        print(f"   Net value ($M): bad case {r.net_value_p10:,.0f} | typical {r.net_value_median:,.0f} | good case {r.net_value_p90:,.0f}")
        print(f"   Synergy needed to break even: ${r.breakeven_runrate:,.0f}M/yr vs typical expected ${r.expected_runrate:,.0f}M/yr")
        print(f"   Biggest driver of value: {r.top_driver}\n")


if __name__ == "__main__":
    main()

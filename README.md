# Merger screener

AgentVerse 2026, Finance track ("Trusted Markets").

Enter a company and choose why you want a merger (cut debt, gain market share, economies of scale,
horizontal or vertical integration). The tool finds three candidate targets and scores each on three separate
measures across thousands of simulated economic scenarios:

| Metric | Question | Passes a run when |
| --- | --- | --- |
| **Value creation** | Is it worth doing? | Synergies, after tax and integration costs, beat the premium, and year-3 return beats the cost of capital |
| **Financial resilience** | Can the combined company carry it? | Year-2 net debt / EBITDA ≤ 3.5x and EBITDA / interest ≥ 4x |
| **Strategic fit** | Does it achieve the chosen goal? | The intention's own target is met and the deal isn't blocked |

Each metric is shown as Strong (80% of runs or more), Moderate (50 to 80%) or Weak, with the underlying percentage,
the most common reason it fails, and the scenario where it is weakest. Metrics are never averaged into one score, so a
deal that can't carry its debt can't hide behind good synergies.

## Quickstart

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest                                        # 38 tests, no API key or network needed
python scripts/merger_demo.py --intention economies_of_scale  # SYNTHETIC example companies
```

Intentions: `reduce_debt`, `market_share`, `economies_of_scale`, `horizontal`, `vertical`.

## Where each number comes from

- **Company figures**: the SEC's free XBRL API (`src/merger/sec.py`). Every figure records its tag, period and filing.
  Required figures that are missing raise an error rather than being guessed, and anything assumed (such as zero interest
  when none is reported) is labelled. Set `SEC_USER_AGENT="Team contact@example.com"` before fetching. Share price is an input.
- **Assumptions**: synergy size, how much is achieved, delays, integration cost, churn, regulatory outcomes and the four
  economic scenarios are all in `Assumptions` and `SCENARIOS` in `src/merger/simulate.py`. **The defaults are placeholders
  to be tuned and cited**, not established facts.
- **Thresholds**: fixed in `Thresholds` before any run and returned with every result.

## How it works

| Module | Job |
| --- | --- |
| `merger/financials.py` | One company's figures, with provenance |
| `merger/sec.py` | Fetch and parse SEC filings: latest full-year 10-K values only |
| `merger/simulate.py` | Scenario simulation, the three metrics, break-even synergy, main value driver |
| `merger/screen.py` | Hard eligibility rules per intention, then ranking by the metric that matters for that intention |
| `merger/demo_data.py` | Invented companies for the demo, all flagged synthetic |
| `desk/audit.py`, `desk/stats.py` | Hash-chained audit log and statistics reused from our earlier work |

## Still to build

- Gemini layer: summarise each company and propose supplier lists and synergy evidence, with every claim tied to a quoted
  passage that code checks against the filing.
- Front end with the company box and intention dropdown.
- Test on past deals: run the tool on completed deals using only pre-announcement data, then compare its scores with what
  happened (realised savings, goodwill write-downs, returns).
- Tune the default assumptions against published deal evidence, with sources.

## Limits, stated plainly

- The percentages are **robustness scores under our assumptions**, not real-world probabilities of success.
- The demo runs on invented companies. It shows the machinery, not a finding about any real deal.
- Market share is approximated (no market-size data yet); vertical deals need a supplier list because industry codes
  don't reveal supply chains.
- Not investment advice. Any real decision needs a human and full due diligence.

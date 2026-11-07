# Finance track brief (transcribed from slides, 7 Nov 2026)

**Track:** Finance, "Trusted Markets" (with Jane Street). Agents you would trust near real money.

## The challenge
Help people make better money decisions. Pick a decision-maker (a compliance analyst facing 400 alerts, a trader testing a hunch,
a small business owner wondering if payroll will clear), find a decision they get wrong or make too slowly, and build an agent that helps,
without hiding the reasoning or the risk.

Where to aim: risk and compliance, market agents, financial inclusion, scam and fraud, regulation, or your own idea.

"Would you let it near your own money?" Keep a human in the loop for anything that moves money or affects a person, and show your
reasoning so a professional could check it.

## Data and tools
Agentverse finance kit (synthetic cases, real price history, hidden test mode); Binance Vision and the Binance public market-data API;
FRED, Bank of England, ECB Data Portal, ONS; SEC EDGAR, Companies House API; IBM AMLSim, PaySim, Elliptic, ULB card fraud;
Polymarket public API; World Bank Findex, FCA Financial Lives, ONS; FCA Handbook, legislation.gov.uk, BIS.

**Synthetic data is allowed if you:** label it and say how you made it; never present synthetic results or backtests as real-world performance;
use no real financial records, account details or logins; paper trade only and never connect to real money; reference every dataset and check its licence.

## What to submit
- **Code:** a public repo with a short README, ideally with a demo that runs, even a rough one.
- **Pitch deck:** the problem, who it helps, your data, how you evaluated against a benchmark, the agent's limits and risk controls, and your key statistics.
- **Live pitch:** top three teams only. Up to 4 minutes to the judges, plus 2 minutes for questions.

Judges score every team from the deck, so put the evidence there.

## Judging
| Weight | Criterion | What a top score looks like |
| --- | --- | --- |
| 25% | Traceable reasoning | Any decision traces from raw data to output, with maths kept apart |
| 25% | Testing on unseen data | Unseen data, real costs, a fitting benchmark and stated uncertainty |
| 15% | Risk controls and oversight | Limits enforced in code, every action logged, one control shown live |
| 20% | Relevance and impact | A specific user, with the failure that would hurt them handled on screen |
| 15% | Ambition and technical quality | Does something a conventional program couldn't, built to a high standard |

No leaderboard. An honest result that misses its benchmark can still score top marks.

**Also on the day:** "Best use of Gemini" (Google DeepMind is the exclusive AI partner across all four tracks): onsite credits, cash prizes, swag,
a possible London office visit and Foundry VC intros.

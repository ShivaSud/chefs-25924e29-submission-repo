# FinAgent

AgentVerse 2026 (UCL AI Society), finance track.

A personal finance agent for the cost-of-living squeeze. It reads transactions, finds wasted spend
and forgotten subscriptions, models cash flow to payday, and takes action, with a human approving
every action before anything happens.

## Quickstart

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # only needed for the chat agent
pytest                              # core logic and the agent loop run offline
streamlit run app.py                # demo UI on the sample statement
```

CSV format: `date, description, amount[, category]`. Money out is negative. Dates as `2026-11-07` or `07/11/2026`.
Regenerate the sample statement with `python scripts/make_sample_data.py`.

## How it works

| Module | Job |
| --- | --- |
| `ingest.py` | CSV to transactions |
| `models.py` | Transaction and merchant-name normalisation |
| `analysis.py` | Recurring-charge detection (cadence and amount stability), duplicate charges, category spend |
| `forecast.py` | Day-by-day balance to payday, with what-ifs |
| `actions.py` | Approval queue and drafted emails (cancel a subscription, dispute a duplicate) |
| `agent.py` | Claude tool-use loop over the above |

The numbers come from deterministic code and the model only orchestrates it, so figures are never
guessed. The agent cannot execute anything: `propose_action` queues a draft, and `ActionQueue.executor`
is the single place to wire in real side effects (sending the email, calling a bank API).

## Assumptions

- Recurring means 3+ charges at a steady weekly, monthly or annual cadence, each within 10% of the typical amount.
- Everyday spend is modelled as a flat daily average of the last 30 days of non-recurring spend.
- A recurring charge already due in the data is assumed paid. Duplicate charges are candidates, not verdicts.
- Essentials (housing, utilities, insurance, debt) are excluded from cancellation suggestions.

## If the track brief changes

The problem statement is revealed at the event. Core pieces (ingest, analysis, forecast, approval queue)
are independent of the UI and the model, so most of this carries over to an adjacent brief.

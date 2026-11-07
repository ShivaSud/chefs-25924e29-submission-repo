"""Streamlit demo UI.  Run:  streamlit run app.py"""
import os
import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from finagent.agent import AgentContext, run_agent  # noqa: E402
from finagent.analysis import detect_recurring, find_duplicates, spend_by_category, variable_daily_spend  # noqa: E402
from finagent.forecast import forecast_to_payday, next_monthly_date  # noqa: E402
from finagent.ingest import load_csv, parse_csv_text  # noqa: E402

SAMPLE = Path(__file__).parent / "data" / "sample_transactions.csv"

st.set_page_config(page_title="FinAgent", layout="wide")
st.title("FinAgent")
st.caption("Finds wasted spend, models cash flow to payday, and drafts actions for you to approve.")

with st.sidebar:
    upload = st.file_uploader("Transactions CSV (date, description, amount, category)", type="csv")
    balance = st.number_input("Current balance (£)", value=400.0, step=10.0)
    as_of = st.date_input("As-of date", value=date.today())
    payday = st.date_input("Next payday", value=next_monthly_date(as_of, 28))

txns = parse_csv_text(upload.getvalue().decode("utf-8")) if upload else load_csv(SAMPLE)
recurring = detect_recurring(txns)
daily = variable_daily_spend(txns, recurring, as_of)

if "ctx" not in st.session_state or st.session_state.get("ctx_key") != (len(txns), balance, as_of, payday):
    st.session_state.ctx = AgentContext(txns, balance, payday, as_of)
    st.session_state.ctx_key = (len(txns), balance, as_of, payday)
    st.session_state.history = []
    st.session_state.transcript = []
ctx: AgentContext = st.session_state.ctx

left, right = st.columns(2)
with left:
    st.subheader("Recurring charges")
    st.dataframe(
        [
            {
                "merchant": r.merchant,
                "cadence": r.cadence,
                "amount": r.amount,
                "per month": r.monthly_cost,
                "next due": r.next_due,
                "essential": r.is_essential,
            }
            for r in recurring
        ],
        use_container_width=True,
    )
    dups = find_duplicates(txns)
    if dups:
        st.warning("Possible duplicate charges: " + ", ".join(f"{d.merchant} {d.amount:.2f} on {d.second}" for d in dups))
    st.subheader("Spend by category")
    st.bar_chart(spend_by_category(txns))

with right:
    st.subheader("Balance until payday")
    if payday > as_of:
        fc = forecast_to_payday(balance, as_of, payday, recurring, daily)
        st.line_chart({d.date: d.balance for d in fc.days})
        if fc.shortfall:
            st.error(f"Projected to hit {fc.min_balance:.2f} on {fc.min_date}: you would not make it to payday.")
        else:
            st.success(f"Lowest projected balance {fc.min_balance:.2f} on {fc.min_date}.")
    else:
        st.info("Pick a payday after the as-of date.")

st.divider()
st.subheader("Ask FinAgent")
for role, text in st.session_state.transcript:
    st.chat_message(role).write(text)

if prompt := st.chat_input("e.g. Will I make it to payday? What can I cut?"):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("Set ANTHROPIC_API_KEY to talk to the agent.")
    else:
        st.session_state.transcript.append(("user", prompt))
        reply, st.session_state.history = run_agent(ctx, prompt, st.session_state.history)
        st.session_state.transcript.append(("assistant", reply))
        st.rerun()

if ctx.queue.items:
    st.subheader("Proposed actions")
    for action in ctx.queue.items:
        with st.expander(f"#{action.id} {action.summary} [{action.status}]", expanded=action.status == "pending"):
            st.text(f"{action.payload.get('subject', '')}\n\n{action.payload.get('body', '')}")
            if action.status == "pending":
                a, b = st.columns(2)
                if a.button("Approve", key=f"ok{action.id}"):
                    ctx.queue.approve(action.id)
                    st.rerun()
                if b.button("Reject", key=f"no{action.id}"):
                    ctx.queue.reject(action.id)
                    st.rerun()

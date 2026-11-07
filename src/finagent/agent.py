from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .actions import ActionQueue, draft_cancellation, draft_duplicate_dispute
from .analysis import (
    cancellation_candidates,
    detect_recurring,
    find_duplicates,
    spend_by_category,
    variable_daily_spend,
)
from .forecast import forecast_to_payday
from .models import Transaction

DEFAULT_MODEL = os.environ.get("FINAGENT_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are FinAgent, a personal finance assistant for someone feeling the cost-of-living squeeze.
Always ground answers in the tools: never guess figures. Typical flow: check the forecast to payday, find
recurring charges and duplicates, run a what-if with the savings, then propose actions.
You cannot send, cancel or pay anything yourself. propose_action only queues a draft for the user to approve.
Be concise, use plain language, and give amounts in pounds."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "spend_summary",
        "description": "Total spend per category, optionally limited to the last N days.",
        "input_schema": {
            "type": "object",
            "properties": {"since_days": {"type": "integer", "description": "Only count the last N days"}},
        },
    },
    {
        "name": "list_recurring",
        "description": "Recurring charges (subscriptions, bills) with cadence, amount, monthly cost and next due date. "
        "Each is flagged essential or not.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "find_duplicate_charges",
        "description": "Same-merchant, same-amount charges within a few days of each other.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "forecast_to_payday",
        "description": "Day-by-day balance until payday. Supports what-ifs: cancel some recurring merchants "
        "and/or cut everyday spend, and optionally override balance or payday.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cancel_merchants": {"type": "array", "items": {"type": "string"}},
                "daily_spend_cut": {"type": "number", "description": "Pounds per day to trim from everyday spend"},
                "balance": {"type": "number"},
                "payday": {"type": "string", "description": "ISO date, e.g. 2026-11-28"},
            },
        },
    },
    {
        "name": "propose_action",
        "description": "Queue a drafted action for the user to approve. Does not execute anything.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["cancel_subscription", "dispute_duplicate"]},
                "merchant": {"type": "string"},
            },
            "required": ["kind", "merchant"],
        },
    },
]


@dataclass
class AgentContext:
    transactions: list[Transaction]
    balance: float
    payday: date
    as_of: date
    queue: ActionQueue = field(default_factory=ActionQueue)

    def recurring(self):
        return detect_recurring(self.transactions)


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _match(merchant: str, candidates):
    needle = merchant.lower().strip()
    for c in candidates:
        if needle == c.merchant or needle in c.merchant or c.merchant in needle:
            return c
    return None


def dispatch(ctx: AgentContext, name: str, args: dict[str, Any]) -> Any:
    """Run one tool call. Returns JSON-serialisable data; errors come back as {"error": ...}."""
    try:
        if name == "spend_summary":
            since = ctx.as_of - timedelta(days=int(args["since_days"])) if args.get("since_days") else None
            return spend_by_category(ctx.transactions, since)

        if name == "list_recurring":
            return [{**_jsonable(r), "is_essential": r.is_essential} for r in ctx.recurring()]

        if name == "find_duplicate_charges":
            return _jsonable(find_duplicates(ctx.transactions))

        if name == "forecast_to_payday":
            recurring = ctx.recurring()
            cancelled = {m.lower() for m in args.get("cancel_merchants", [])}
            if cancelled:
                recurring = [r for r in recurring if not any(c in r.merchant for c in cancelled)]
            daily = variable_daily_spend(ctx.transactions, ctx.recurring(), ctx.as_of)
            daily = max(0.0, daily - float(args.get("daily_spend_cut", 0.0)))
            payday = date.fromisoformat(args["payday"]) if args.get("payday") else ctx.payday
            balance = float(args["balance"]) if "balance" in args else ctx.balance
            fc = forecast_to_payday(balance, ctx.as_of, payday, recurring, daily)
            return {
                "start_balance": fc.start_balance,
                "payday": fc.payday.isoformat(),
                "daily_everyday_spend": daily,
                "committed_before_payday": fc.committed_total,
                "everyday_spend_before_payday": fc.variable_total,
                "lowest_balance": fc.min_balance,
                "lowest_balance_date": fc.min_date.isoformat(),
                "shortfall": fc.shortfall,
                "days": [{"date": d.date.isoformat(), "balance": d.balance, "events": list(d.events)} for d in fc.days],
            }

        if name == "propose_action":
            kind, merchant = args["kind"], args["merchant"]
            if kind == "cancel_subscription":
                rec = _match(merchant, cancellation_candidates(ctx.recurring()))
                if rec is None:
                    return {"error": f"No non-essential recurring charge matches {merchant!r}"}
                summary, payload = draft_cancellation(rec)
                key = rec.merchant
            elif kind == "dispute_duplicate":
                dup = _match(merchant, find_duplicates(ctx.transactions))
                if dup is None:
                    return {"error": f"No duplicate charge found for {merchant!r}"}
                summary, payload = draft_duplicate_dispute(dup)
                key = dup.merchant
            else:
                return {"error": f"Unknown action kind {kind!r}"}
            action = ctx.queue.propose(kind, key, summary, payload)
            return {**_jsonable(action), "note": "Queued for user approval. Nothing has been sent."}

        return {"error": f"Unknown tool {name!r}"}
    except Exception as exc:  # tool errors go back to the model, not up the stack
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_agent(
    ctx: AgentContext,
    user_message: str,
    history: list[dict] | None = None,
    client: Any = None,
    model: str | None = None,
    max_turns: int = 8,
) -> tuple[str, list[dict]]:
    """One user turn through the tool-use loop. Returns (reply_text, updated_history)."""
    if client is None:
        import anthropic  # imported lazily so analysis/tests need no API key

        client = anthropic.Anthropic()

    messages = list(history or []) + [{"role": "user", "content": user_message}]
    system = f"{SYSTEM_PROMPT}\nAs-of date: {ctx.as_of.isoformat()}. Payday: {ctx.payday.isoformat()}."

    for _ in range(max_turns):
        response = client.messages.create(
            model=model or DEFAULT_MODEL,
            max_tokens=1500,
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            return text, messages

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = dispatch(ctx, block.name, dict(block.input))
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(output, default=str)}
                )
        messages.append({"role": "user", "content": results})

    return "I hit my step limit before finishing. Try asking a narrower question.", messages

from __future__ import annotations

import math
from dataclasses import dataclass

from .audit import AuditLog


@dataclass(frozen=True)
class RiskLimits:
    allowed_symbols: frozenset[str]
    max_order_notional: float
    max_position_notional: float
    max_gross_notional: float
    max_daily_loss: float
    max_price_deviation: float = 0.02  # limit price may sit at most 2% from the mark


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    limit_price: float

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side == "buy" else -self.qty


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reasons: tuple[str, ...]


class PaperBroker:
    """A paper-only broker that every order must pass through. It never connects to real money.

    Limits are enforced here, in code, whatever an agent says. Each decision and fill is
    written to the audit log. A daily-loss breach latches a kill switch: from then on only
    risk-reducing orders are accepted until a named human resets it."""

    def __init__(self, limits: RiskLimits, audit: AuditLog, cash: float = 100_000.0, slippage_bps: float = 3.0):
        self.limits, self.audit = limits, audit
        self.cash = cash
        self.slippage = slippage_bps * 1e-4
        self.positions: dict[str, float] = {}
        self.marks: dict[str, float] = {}
        self.halted = False
        self.halt_reason = ""
        self._day_start_equity: float | None = None

    def equity(self) -> float:
        return self.cash + sum(q * self.marks.get(s, 0.0) for s, q in self.positions.items())

    def gross_notional(self) -> float:
        return sum(abs(q) * self.marks.get(s, 0.0) for s, q in self.positions.items())

    def start_day(self) -> None:
        self._day_start_equity = self.equity()
        self.audit.append("day_start", {"equity": self._day_start_equity})

    def mark(self, symbol: str, price: float) -> None:
        self.marks[symbol] = price
        if self.halted or self._day_start_equity is None:
            return
        loss = self._day_start_equity - self.equity()
        if loss >= self.limits.max_daily_loss:
            self.halted = True
            self.halt_reason = f"daily loss {loss:.2f} reached limit {self.limits.max_daily_loss:.2f}"
            self.audit.append("kill_switch", {"reason": self.halt_reason, "equity": self.equity()})

    def reset_halt(self, operator: str) -> None:
        if not operator.strip():
            raise ValueError("a named operator is required to reset the kill switch")
        self.halted = False
        self.halt_reason = ""
        self._day_start_equity = self.equity()
        self.audit.append("kill_switch_reset", {"operator": operator, "equity": self._day_start_equity})

    def check(self, order: Order) -> Decision:
        lim = self.limits
        reasons: list[str] = []

        if order.symbol not in lim.allowed_symbols:
            reasons.append(f"symbol {order.symbol!r} is not on the allowed list")
        if order.side not in ("buy", "sell"):
            reasons.append(f"unknown side {order.side!r}")
        qty_ok = math.isfinite(order.qty) and order.qty > 0
        if not qty_ok:
            reasons.append("quantity must be a positive finite number")
        mark = self.marks.get(order.symbol)
        if mark is None or mark <= 0:
            reasons.append("no valid mark price for this symbol")

        if qty_ok and mark and mark > 0 and order.side in ("buy", "sell"):
            if not (math.isfinite(order.limit_price) and order.limit_price > 0):
                reasons.append("limit price must be a positive finite number")
            else:
                dev = abs(order.limit_price - mark) / mark
                if dev > lim.max_price_deviation:
                    reasons.append(f"limit price is {dev:.1%} from the mark, over the {lim.max_price_deviation:.1%} fat-finger limit")

            if order.qty * mark > lim.max_order_notional:
                reasons.append(f"order notional {order.qty * mark:,.0f} exceeds the {lim.max_order_notional:,.0f} per-order limit")

            current = self.positions.get(order.symbol, 0.0)
            new_pos = current + order.signed_qty
            if abs(new_pos) * mark > lim.max_position_notional:
                reasons.append(f"resulting position {abs(new_pos) * mark:,.0f} exceeds the {lim.max_position_notional:,.0f} position limit")
            others = sum(abs(q) * self.marks.get(s, 0.0) for s, q in self.positions.items() if s != order.symbol)
            if others + abs(new_pos) * mark > lim.max_gross_notional:
                reasons.append(f"resulting gross exposure exceeds the {lim.max_gross_notional:,.0f} limit")
            if self.halted and abs(new_pos) >= abs(current):
                reasons.append(f"kill switch engaged ({self.halt_reason}): only risk-reducing orders are accepted")
        elif self.halted:
            reasons.append(f"kill switch engaged ({self.halt_reason})")

        return Decision(allowed=not reasons, reasons=tuple(reasons))

    def submit(self, order: Order, source: str = "agent") -> Decision:
        decision = self.check(order)
        self.audit.append(
            "order_decision",
            {"source": source, "order": order.__dict__, "allowed": decision.allowed, "reasons": list(decision.reasons)},
        )
        if decision.allowed:
            mark = self.marks[order.symbol]
            fill = mark * (1 + self.slippage) if order.side == "buy" else mark * (1 - self.slippage)
            self.cash -= order.signed_qty * fill
            self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + order.signed_qty
            self.audit.append("fill", {"symbol": order.symbol, "qty": order.signed_qty, "price": fill, "equity": self.equity()})
        return decision

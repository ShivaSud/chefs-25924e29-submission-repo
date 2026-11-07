from __future__ import annotations

import calendar
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median

from .models import Transaction

# Median gap (days) that qualifies a charge stream as a given cadence.
_CADENCE_WINDOWS = {"weekly": (6, 8), "monthly": (27, 34), "annual": (355, 375)}
_PER_MONTH = {"weekly": 52 / 12, "monthly": 1.0, "annual": 1 / 12}
_AMOUNT_TOLERANCE = 0.10  # amounts must sit within 10% of the median to count as a subscription

ESSENTIAL_CATEGORIES = {"housing", "utilities", "insurance", "debt", "council tax", "childcare"}


@dataclass(frozen=True)
class Recurring:
    merchant: str
    cadence: str
    amount: float
    monthly_cost: float
    occurrences: int
    last_date: date
    next_due: date
    category: str

    @property
    def is_essential(self) -> bool:
        return self.category.lower() in ESSENTIAL_CATEGORIES


@dataclass(frozen=True)
class DuplicateCharge:
    merchant: str
    amount: float
    first: date
    second: date


def advance(cadence: str, d: date) -> date:
    """The next occurrence after d for a given cadence."""
    if cadence == "weekly":
        return d + timedelta(days=7)
    if cadence == "annual":
        try:
            return d.replace(year=d.year + 1)
        except ValueError:  # 29 Feb
            return d.replace(year=d.year + 1, day=28)
    year = d.year + (1 if d.month == 12 else 0)
    month = d.month % 12 + 1
    return d.replace(year=year, month=month, day=min(d.day, calendar.monthrange(year, month)[1]))


def detect_recurring(txns: list[Transaction], min_occurrences: int = 3) -> list[Recurring]:
    """Find charge streams with a steady cadence and a steady amount."""
    streams: dict[str, list[Transaction]] = defaultdict(list)
    for t in txns:
        if t.is_spend:
            streams[t.merchant].append(t)

    found: list[Recurring] = []
    for merchant, items in streams.items():
        if len(items) < min_occurrences:
            continue
        items.sort(key=lambda t: t.date)
        amounts = [-t.amount for t in items]
        typical = median(amounts)
        if any(abs(a - typical) > _AMOUNT_TOLERANCE * typical for a in amounts):
            continue
        gap = median([(b.date - a.date).days for a, b in zip(items, items[1:])])
        cadence = next((n for n, (lo, hi) in _CADENCE_WINDOWS.items() if lo <= gap <= hi), None)
        if cadence is None:
            continue
        categories = [t.category for t in items if t.category]
        found.append(
            Recurring(
                merchant=merchant,
                cadence=cadence,
                amount=round(typical, 2),
                monthly_cost=round(typical * _PER_MONTH[cadence], 2),
                occurrences=len(items),
                last_date=items[-1].date,
                next_due=advance(cadence, items[-1].date),
                category=Counter(categories).most_common(1)[0][0] if categories else "",
            )
        )
    return sorted(found, key=lambda r: r.monthly_cost, reverse=True)


def cancellation_candidates(recurring: list[Recurring]) -> list[Recurring]:
    """Recurring charges that are not essentials, i.e. worth a second look."""
    return [r for r in recurring if not r.is_essential]


def occurrences_between(rec: Recurring, after: date, until: date) -> list[date]:
    """Due dates of a recurring charge in the window (after, until]."""
    out: list[date] = []
    d = rec.next_due
    while d <= until:
        if d > after:
            out.append(d)
        d = advance(rec.cadence, d)
    return out


def find_duplicates(txns: list[Transaction], window_days: int = 3) -> list[DuplicateCharge]:
    """Same merchant, same amount, within a few days. Candidates, not verdicts."""
    seen: dict[tuple[str, float], list[date]] = defaultdict(list)
    for t in txns:
        if t.is_spend:
            seen[(t.merchant, round(-t.amount, 2))].append(t.date)
    out: list[DuplicateCharge] = []
    for (merchant, amount), dates in seen.items():
        dates.sort()
        for first, second in zip(dates, dates[1:]):
            if (second - first).days <= window_days:
                out.append(DuplicateCharge(merchant, amount, first, second))
    return sorted(out, key=lambda d: d.second)


def spend_by_category(txns: list[Transaction], since: date | None = None) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for t in txns:
        if t.is_spend and (since is None or t.date >= since):
            totals[t.category or "Uncategorised"] += -t.amount
    return dict(sorted(((k, round(v, 2)) for k, v in totals.items()), key=lambda kv: kv[1], reverse=True))


def variable_daily_spend(
    txns: list[Transaction], recurring: list[Recurring], as_of: date, lookback_days: int = 30
) -> float:
    """Average daily spend on things that are not recurring charges."""
    fixed = {r.merchant for r in recurring}
    start = as_of - timedelta(days=lookback_days)
    total = sum(-t.amount for t in txns if t.is_spend and start < t.date <= as_of and t.merchant not in fixed)
    return round(total / lookback_days, 2)

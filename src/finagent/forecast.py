from __future__ import annotations

import calendar
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from .analysis import Recurring, occurrences_between


@dataclass(frozen=True)
class ForecastDay:
    date: date
    balance: float
    events: tuple[str, ...]


@dataclass(frozen=True)
class Forecast:
    start_balance: float
    payday: date
    days: tuple[ForecastDay, ...]
    min_balance: float
    min_date: date
    shortfall: bool
    committed_total: float  # recurring charges landing before payday
    variable_total: float  # modelled everyday spend before payday


def next_monthly_date(as_of: date, day: int) -> date:
    """The next date strictly after as_of that falls on `day` of a month (clamped to month end)."""
    for offset in (0, 1):
        month_index = as_of.month - 1 + offset
        year, month = as_of.year + month_index // 12, month_index % 12 + 1
        candidate = date(year, month, min(day, calendar.monthrange(year, month)[1]))
        if candidate > as_of:
            return candidate
    raise AssertionError("unreachable")


def forecast_to_payday(
    balance: float,
    as_of: date,
    payday: date,
    recurring: list[Recurring],
    daily_variable: float,
    buffer: float = 0.0,
) -> Forecast:
    """Project the balance day by day until the day before payday.

    Assumes recurring charges repeat on their cadence and everyday spend is a flat daily
    average. A recurring charge already past due in the data is assumed to have been paid.
    """
    if payday <= as_of:
        raise ValueError("payday must be after the as-of date")

    due: dict[date, list[Recurring]] = defaultdict(list)
    for rec in recurring:
        for d in occurrences_between(rec, as_of, payday - timedelta(days=1)):
            due[d].append(rec)

    running = balance
    committed = 0.0
    days: list[ForecastDay] = []
    d = as_of + timedelta(days=1)
    while d < payday:
        events = []
        for rec in due.get(d, []):
            running -= rec.amount
            committed += rec.amount
            events.append(f"{rec.merchant} -{rec.amount:.2f}")
        running -= daily_variable
        days.append(ForecastDay(d, round(running, 2), tuple(events)))
        d += timedelta(days=1)

    if days:
        low = min(days, key=lambda x: x.balance)
        min_balance, min_date = low.balance, low.date
    else:
        min_balance, min_date = round(balance, 2), as_of

    return Forecast(
        start_balance=round(balance, 2),
        payday=payday,
        days=tuple(days),
        min_balance=min_balance,
        min_date=min_date,
        shortfall=min_balance < buffer,
        committed_total=round(committed, 2),
        variable_total=round(daily_variable * len(days), 2),
    )

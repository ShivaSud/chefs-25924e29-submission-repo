from __future__ import annotations

import csv
import io
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from .models import Transaction

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")


def _parse_date(raw: str) -> date:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date: {raw!r}")


def _rows_to_transactions(reader: Iterable[dict]) -> list[Transaction]:
    txns: list[Transaction] = []
    for raw in reader:
        row = {(k or "").strip().lower(): (v or "") for k, v in raw.items()}
        txns.append(
            Transaction(
                date=_parse_date(row["date"]),
                description=row["description"].strip(),
                amount=float(row["amount"].replace(",", "")),
                category=row.get("category", "").strip(),
            )
        )
    return sorted(txns, key=lambda t: t.date)


def load_csv(path: str | Path) -> list[Transaction]:
    """Load transactions from a CSV with columns: date, description, amount[, category]."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return _rows_to_transactions(csv.DictReader(f))


def parse_csv_text(text: str) -> list[Transaction]:
    """Same as load_csv but for in-memory text (e.g. a Streamlit upload)."""
    return _rows_to_transactions(csv.DictReader(io.StringIO(text.lstrip("﻿"))))

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

# Words that appear in bank descriptions but say nothing about the merchant.
_NOISE = {"card", "payment", "pos", "dd", "ref", "ltd", "limited", "uk", "gb", "london", "www", "com"}


def normalise_merchant(description: str) -> str:
    """Collapse a raw bank description to a stable merchant key.

    "SPOTIFY P3A1B2C" -> "spotify", "TESCO STORES 1234 LONDON GB" -> "tesco stores".
    """
    tokens = re.split(r"[^a-z0-9&]+", description.lower())
    kept = [
        t
        for t in tokens
        if len(t) > 1 and t not in _NOISE and not any(c.isdigit() for c in t)
    ]
    return " ".join(kept) or description.lower().strip()


@dataclass(frozen=True)
class Transaction:
    date: dt.date
    description: str
    amount: float  # negative = money out, positive = money in
    category: str = ""

    @property
    def merchant(self) -> str:
        return normalise_merchant(self.description)

    @property
    def is_spend(self) -> bool:
        return self.amount < 0

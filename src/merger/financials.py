from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Financials:
    """One company's latest annual figures, in USD. `provenance` records where each figure came from
    (XBRL tag, period end, filing accession), or that it was assumed, so every input is traceable."""

    name: str
    ticker: str
    revenue: float
    operating_income: float
    da: float  # depreciation and amortisation
    sga: float  # selling, general and administrative: the overhead cost synergies come from
    interest_expense: float
    total_debt: float
    cash: float
    shares: float
    share_price: float
    sic: str = ""
    synthetic: bool = False
    provenance: dict = field(default_factory=dict, compare=False)

    @property
    def ebitda(self) -> float:
        return self.operating_income + self.da

    @property
    def market_cap(self) -> float:
        return self.shares * self.share_price

    @property
    def net_debt(self) -> float:
        return self.total_debt - self.cash

    @property
    def leverage(self) -> float:
        """Net debt / EBITDA. Infinite if EBITDA is not positive."""
        return self.net_debt / self.ebitda if self.ebitda > 0 else math.inf

    def check(self) -> None:
        if self.revenue <= 0:
            raise ValueError(f"{self.ticker}: revenue must be positive")
        if self.market_cap <= 0:
            raise ValueError(f"{self.ticker}: needs a positive share count and share price")
        if self.sga < 0 or self.total_debt < 0 or self.cash < 0 or self.interest_expense < 0:
            raise ValueError(f"{self.ticker}: negative balance or cost figure")

"""Pull a company's figures from the SEC's free XBRL API (no key needed).

The SEC asks automated clients to identify themselves: set SEC_USER_AGENT to something like
"TeamName contact@example.com" before fetching. Parsing is separate from fetching, so it can be
tested offline and every figure carries its source.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import date

from .financials import Financials

FLOW_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "operating_income": ["OperatingIncomeLoss"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
}
INSTANT_TAGS = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "long_term_debt": ["LongTermDebt"],
    "long_term_debt_noncurrent": ["LongTermDebtNoncurrent"],
    "long_term_debt_current": ["LongTermDebtCurrent"],
    "short_term_borrowings": ["ShortTermBorrowings"],
    "commercial_paper": ["CommercialPaper"],
}
REQUIRED = ("revenue", "operating_income", "sga")


class MissingData(ValueError):
    """A figure the model needs was not found in the filings. We never guess it."""


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _latest(entries: list[dict], annual_flow: bool) -> dict | None:
    """Latest 10-K value. Flow items must cover a full year (so quarters inside a 10-K are skipped)."""
    picked = []
    for e in entries:
        if not str(e.get("form", "")).startswith("10-K"):
            continue
        if annual_flow:
            if "start" not in e or not 350 <= _days(e["start"], e["end"]) <= 380:
                continue
        elif "start" in e:
            continue
        picked.append(e)
    if not picked:
        return None
    return max(picked, key=lambda e: (e["end"], e.get("filed", "")))


def _find(gaap: dict, tags: list[str], annual_flow: bool) -> tuple[float, dict] | None:
    for tag in tags:
        usd = gaap.get(tag, {}).get("units", {}).get("USD")
        if not usd:
            continue
        e = _latest(usd, annual_flow)
        if e is not None:
            return float(e["val"]), {"tag": tag, "end": e["end"], "filed": e.get("filed"), "accn": e.get("accn")}
    return None


def parse_companyfacts(facts: dict, ticker: str, share_price: float, sic: str = "") -> Financials:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    values: dict[str, float] = {}
    prov: dict[str, dict] = {}

    for key, tags in FLOW_TAGS.items():
        hit = _find(gaap, tags, annual_flow=True)
        if hit:
            values[key], prov[key] = hit
    for key, tags in INSTANT_TAGS.items():
        hit = _find(gaap, tags, annual_flow=False)
        if hit:
            values[key], prov[key] = hit

    missing = [k for k in REQUIRED if k not in values]
    if missing:
        raise MissingData(f"{ticker}: not found in filings: {', '.join(missing)}")

    for key in ("da", "interest_expense", "cash"):
        if key not in values:
            values[key] = 0.0
            prov[key] = {"assumed": "zero, not reported under the tags searched"}

    if "long_term_debt" in values:
        debt = values["long_term_debt"]
        parts = ["long_term_debt"]
    else:
        debt = values.get("long_term_debt_noncurrent", 0.0) + values.get("long_term_debt_current", 0.0)
        parts = [k for k in ("long_term_debt_noncurrent", "long_term_debt_current") if k in values]
    for k in ("short_term_borrowings", "commercial_paper"):
        if k in values:
            debt += values[k]
            parts.append(k)
    prov["total_debt"] = {"sum_of": parts} if parts else {"assumed": "zero, no debt tags found"}

    shares_entries = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares")
    if not shares_entries:
        raise MissingData(f"{ticker}: shares outstanding not found")
    s = max(shares_entries, key=lambda e: (e["end"], e.get("filed", "")))
    prov["shares"] = {"tag": "dei:EntityCommonStockSharesOutstanding", "end": s["end"], "filed": s.get("filed")}
    prov["share_price"] = {"input": "supplied by user or market data, not from filings"}

    return Financials(
        name=str(facts.get("entityName", ticker)),
        ticker=ticker.upper(),
        revenue=values["revenue"],
        operating_income=values["operating_income"],
        da=values["da"],
        sga=values["sga"],
        interest_expense=values["interest_expense"],
        total_debt=debt,
        cash=values["cash"],
        shares=float(s["val"]),
        share_price=float(share_price),
        sic=str(sic),
        provenance=prov,
    )


# ---- network (run on your own laptop; set SEC_USER_AGENT first) ----

def _get(url: str) -> dict:
    agent = os.environ.get("SEC_USER_AGENT")
    if not agent:
        raise RuntimeError('Set SEC_USER_AGENT, e.g. "TeamName contact@example.com", as the SEC requires')
    req = urllib.request.Request(url, headers={"User-Agent": agent, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def lookup_cik(ticker: str) -> str:
    table = _get("https://www.sec.gov/files/company_tickers.json")
    for row in table.values():
        if row["ticker"].upper() == ticker.upper():
            return f"{int(row['cik_str']):010d}"
    raise MissingData(f"ticker {ticker!r} not found in the SEC list")


def fetch(ticker: str, share_price: float) -> Financials:
    cik = lookup_cik(ticker)
    facts = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    sic = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").get("sic", "")
    return parse_companyfacts(facts, ticker, share_price, sic)

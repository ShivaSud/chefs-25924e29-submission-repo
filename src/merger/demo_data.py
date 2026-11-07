"""SYNTHETIC example companies (all figures invented, in $ millions) for demos and tests.
They are not real firms and results on them say nothing about real deals."""
from .financials import Financials


def _co(name, ticker, sic, rev, oi, da, sga, interest, debt, cash, shares, price):
    return Financials(f"{name} (synthetic)", ticker, rev, oi, da, sga, interest, debt, cash, shares, price,
                      sic=sic, synthetic=True, provenance={"source": "invented for the demo"})


ACQUIRER = _co("Northwind Foods", "NWF", "2000", 20_000, 2_400, 600, 3_000, 250, 6_000, 1_500, 500, 60)

UNIVERSE = [
    _co("Bramble Snacks", "BRM", "2060", 4_000, 480, 120, 800, 40, 700, 300, 100, 40),
    _co("Harrow Dairy", "HRW", "2020", 6_000, 420, 200, 1_100, 120, 2_400, 200, 150, 35),
    _co("Kestrel Beverages", "KST", "2080", 3_000, 450, 90, 500, 20, 200, 400, 80, 75),
    _co("Larkspur Packaging", "LRK", "2650", 2_500, 300, 100, 300, 30, 500, 100, 60, 45),
    _co("Osprey Software", "OSP", "7372", 1_500, 400, 50, 600, 5, 0, 800, 40, 150),
]

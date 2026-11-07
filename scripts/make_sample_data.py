"""Generate a deterministic synthetic bank statement for demos and tests.

Run from the repo root:  python scripts/make_sample_data.py
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"

# (day of month, description, amount, category)
MONTHLY = [
    (1, "LANDLORD RENT STANDING ORDER", -1150.00, "Housing"),
    (3, "SPOTIFY P3A1B2C", -11.99, "Entertainment"),
    (6, "OCTOPUS ENERGY DD", -92.00, "Utilities"),
    (9, "ICLOUD 2TB", -6.99, "Software"),
    (12, "NETFLIX.COM", -10.99, "Entertainment"),
    (15, "PUREGYM LTD", -39.99, "Health & Fitness"),
    (18, "THREE MOBILE DD", -18.00, "Utilities"),
    (20, "ADOBE CREATIVE CLOUD", -19.97, "Software"),
    (22, "HEADSPACE", -9.99, "Health & Fitness"),
    (28, "ACME LTD SALARY", 2650.00, "Income"),
]


def main() -> None:
    rng = random.Random(7)
    start, end = date(2026, 8, 1), date(2026, 11, 6)
    rows = []
    d = start
    while d <= end:
        for day, desc, amount, category in MONTHLY:
            if d.day == day:
                rows.append((d, desc, amount, category))
        if d.weekday() < 5 and rng.random() < 0.8:
            rows.append((d, f"TFL TRAVEL CH {rng.randint(1000, 9999)}", -rng.uniform(2.8, 8.4), "Transport"))
        if rng.random() < 0.35:
            shop = rng.choice(["TESCO STORES", "SAINSBURYS S/MKTS", "LIDL GB LONDON"])
            rows.append((d, f"{shop} {rng.randint(1000, 9999)}", -rng.uniform(9, 48), "Groceries"))
        if rng.random() < 0.22:
            place = rng.choice(["PRET A MANGER", "DELIVEROO", "UBER EATS", "COSTA COFFEE"])
            rows.append((d, f"{place} {rng.randint(1000, 9999)}", -rng.uniform(4, 28), "Eating out"))
        d += timedelta(days=1)

    # A double charge to find.
    dup_day = date(2026, 10, 14)
    rows += [(dup_day, "UBER EATS 8841", -23.40, "Eating out")] * 2

    rows.sort(key=lambda r: r[0])
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "description", "amount", "category"])
        for day, desc, amount, category in rows:
            w.writerow([day.isoformat(), desc, f"{amount:.2f}", category])
    print(f"Wrote {len(rows)} transactions to {OUT}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

GENESIS = "0" * 64


def _default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _digest(prev: str, seq: int, ts: str, kind: str, payload: Any) -> str:
    return hashlib.sha256((prev + _canonical([seq, ts, kind, payload])).encode()).hexdigest()


class AuditLog:
    """Append-only, hash-chained log. Editing, deleting or reordering any past record
    breaks every later hash, so tampering is detectable by `verify`."""

    def __init__(self, path: str | Path | None = None, clock: Callable[[], str] | None = None):
        self.path = Path(path) if path else None
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.records: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            self.records = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
            ok, msg = self.verify()
            if not ok:
                raise ValueError(f"existing audit log failed verification: {msg}")

    def append(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        prev = self.records[-1]["hash"] if self.records else GENESIS
        seq = len(self.records)
        ts = self._clock()
        payload = json.loads(_canonical(payload))  # normalise numpy types and key order
        record = {"seq": seq, "ts": ts, "kind": kind, "payload": payload, "prev": prev,
                  "hash": _digest(prev, seq, ts, kind, payload)}
        self.records.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(_canonical(record) + "\n")
        return record

    def verify(self, records: list[dict[str, Any]] | None = None) -> tuple[bool, str]:
        recs = self.records if records is None else records
        prev = GENESIS
        for i, r in enumerate(recs):
            if r["seq"] != i:
                return False, f"record {i}: sequence number is {r['seq']}"
            if r["prev"] != prev:
                return False, f"record {i}: chain broken"
            if r["hash"] != _digest(prev, r["seq"], r["ts"], r["kind"], r["payload"]):
                return False, f"record {i}: content does not match its hash"
            prev = r["hash"]
        return True, "ok"

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["kind"] == kind]

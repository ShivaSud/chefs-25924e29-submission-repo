from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .analysis import DuplicateCharge, Recurring


@dataclass
class ProposedAction:
    id: int
    kind: str  # "cancel_subscription" | "dispute_duplicate"
    merchant: str
    summary: str
    payload: dict
    status: str = "pending"  # pending | approved | rejected


@dataclass
class ActionQueue:
    """Actions the agent proposes. Nothing runs until a human approves it.

    `executor` is the hook for real side effects (send the email, call a bank API).
    Left as None the approval is recorded only, which is the safe default for a demo.
    """

    executor: Callable[[ProposedAction], None] | None = None
    items: list[ProposedAction] = field(default_factory=list)

    def propose(self, kind: str, merchant: str, summary: str, payload: dict) -> ProposedAction:
        for existing in self.items:
            if existing.kind == kind and existing.merchant == merchant and existing.status == "pending":
                return existing
        action = ProposedAction(len(self.items) + 1, kind, merchant, summary, payload)
        self.items.append(action)
        return action

    def pending(self) -> list[ProposedAction]:
        return [a for a in self.items if a.status == "pending"]

    def approve(self, action_id: int) -> ProposedAction:
        action = self._get(action_id)
        if self.executor is not None:
            self.executor(action)
        action.status = "approved"
        return action

    def reject(self, action_id: int) -> ProposedAction:
        action = self._get(action_id)
        action.status = "rejected"
        return action

    def _get(self, action_id: int) -> ProposedAction:
        for a in self.items:
            if a.id == action_id:
                return a
        raise KeyError(f"No action with id {action_id}")


def draft_cancellation(rec: Recurring) -> tuple[str, dict]:
    name = rec.merchant.title()
    summary = f"Cancel {name} ({rec.amount:.2f} {rec.cadence}, saves about {rec.monthly_cost:.2f}/month)"
    payload = {
        "subject": f"Request to cancel my {name} subscription",
        "body": (
            f"Hello,\n\nPlease cancel my {name} subscription, billed {rec.cadence} at {rec.amount:.2f}. "
            "The account is registered to [ACCOUNT EMAIL].\n\n"
            "Please confirm the cancellation in writing and the date of the final charge. "
            "I do not authorise any further payments after that date.\n\nThank you,\n[YOUR NAME]"
        ),
    }
    return summary, payload


def draft_duplicate_dispute(dup: DuplicateCharge) -> tuple[str, dict]:
    name = dup.merchant.title()
    summary = f"Dispute duplicate {name} charge of {dup.amount:.2f} ({dup.first} and {dup.second})"
    payload = {
        "subject": f"Possible duplicate charge from {name}",
        "body": (
            f"Hello,\n\nI was charged {dup.amount:.2f} twice, on {dup.first} and {dup.second}. "
            "If this was not two separate orders, please refund the second charge.\n\n"
            "Thank you,\n[YOUR NAME]"
        ),
    }
    return summary, payload

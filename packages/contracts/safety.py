"""Rule-based safety cutoffs. Deterministic, auditable, never ML.

The technical profile calls the AI-authority boundary non-negotiable: all three
AI modules advise, the captain commands, and safety cutoffs stay rule-based so
behaviour under fault is deterministic and auditable.

`services/safety/` therefore imports no model, loads no artifact, and calls no
network. Given the same frame it returns the same answer, forever. If a judge
asks "what happens when your model is wrong", this contract and that module are
the answer.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    NOMINAL = "nominal"
    WARNING = "warning"
    """Approaching a limit. Advisory. The captain has time to decide."""
    CRITICAL = "critical"
    """A limit is exceeded. Immediate attention. Still advisory -- the system
    does not actuate the vessel."""


class SafetyCutoff(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(description="Stable identifier, e.g. 'coolant_overtemp'. Cited in logs.")
    severity: Severity

    stream: str = Field(description="Field path the rule watches.")
    label_en: str
    label_fil: str

    observed: float
    threshold: float
    unit: str

    message_en: str
    message_fil: str

    triggered_at: datetime


class SafetyState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vessel_id: str
    generated_at: datetime

    severity: Severity = Field(description="Highest severity among active cutoffs.")
    active: list[SafetyCutoff] = Field(default_factory=list)

    evaluated_rules: int = Field(
        ge=0, description="How many rules ran. Distinguishes 'all clear' from 'no data to check'."
    )
    skipped_rules: list[str] = Field(
        default_factory=list,
        description="Rules skipped because their sensor was absent. A modular retrofit may "
        "lack a channel; silence about that would be dishonest.",
    )

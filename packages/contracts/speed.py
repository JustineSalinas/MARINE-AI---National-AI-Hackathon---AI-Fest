"""AI Speed Optimization output contract.

The product is not the recommended RPM. The product is the *delta* -- the
difference between what the captain is doing now and what the fuel model says
is cheaper for the same arrival time. Every field here exists to make that
delta legible and trustworthy on a bridge display read in under two seconds.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SpeedRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vessel_id: str
    generated_at: datetime

    # What to do
    recommended_rpm: float = Field(ge=0, description="The setting the captain should move to.")
    recommended_speed_kn: float = Field(ge=0, description="Expected speed through water at that RPM.")

    # What is happening now
    current_rpm: float | None = Field(None, ge=0)
    current_burn_lph: float | None = Field(None, ge=0, description="Model estimate at current RPM.")

    # The delta -- the actual product
    predicted_burn_lph: float = Field(ge=0, description="Model estimate at recommended RPM.")
    savings_lph: float = Field(
        description="current_burn_lph - predicted_burn_lph. Negative means the captain "
        "is already more efficient than the recommendation; show it honestly."
    )
    savings_php_per_hour: float | None = Field(
        None, description="savings_lph * diesel price. Operators budget in pesos, not litres."
    )

    # Why the captain should believe it
    model_confidence: float = Field(
        ge=0, le=1, description="Widens as inputs drift from the training distribution."
    )
    eta_impact_minutes: float = Field(
        description="Change in arrival time if the recommendation is followed. "
        "Near zero by construction -- the optimizer holds ETA and minimises burn."
    )

    # The sentence the captain actually reads. Claude-generated, template fallback.
    advisory_en: str
    advisory_fil: str
    advisory_source: str = Field(
        "template",
        description="'claude' or 'template'. The display never blocks on the API; "
        "if Claude is slow or down, the deterministic template ships instead.",
    )

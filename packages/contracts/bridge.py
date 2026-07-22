"""The composite state the bridge display consumes. One screen, one payload.

Two structural claims from the technical profile are encoded here:

1. "A failure or delay in one module never blocks the other two." Each module is
   wrapped in its own `ModuleStatus` carrying its own freshness and error. One
   module returning `state="error"` still ships a complete, renderable frame.

2. "Predictive Maintenance runs at the edge so it works with zero signal."
   `connectivity` plus the per-module `computed_at_edge` flag let the display
   show maintenance live while speed and route visibly age. Losing signal is a
   designed state, not an error screen.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from packages.contracts.maintenance import MaintenanceStatus
from packages.contracts.route import RouteRecommendation
from packages.contracts.safety import SafetyState
from packages.contracts.speed import SpeedRecommendation

T = TypeVar("T")


class Connectivity(StrEnum):
    ONLINE = "online"
    DEGRADED = "degraded"
    """Reachable but stale -- last cloud response older than its refresh interval."""
    OFFLINE = "offline"
    """No signal. Edge modules continue; cloud modules serve their last cached value."""


class ModuleStatus(BaseModel, Generic[T]):
    """One AI module's contribution, with its own freshness and failure envelope."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["ok", "stale", "error", "unavailable"]
    data: T | None = None

    generated_at: datetime | None = None
    age_seconds: float | None = Field(
        None, ge=0, description="Rendered verbatim on the display. Never hide the age of advice."
    )
    computed_at_edge: bool = Field(
        description="True for maintenance and safety, false for speed and route. "
        "Explains to the captain why one panel is live while another is frozen."
    )
    error: str | None = Field(
        None, description="Plain language, shown to the captain. Not a stack trace."
    )


class BridgeState(BaseModel):
    """Everything the captain's screen needs, in one frame."""

    model_config = ConfigDict(extra="forbid")

    vessel_id: str
    voyage_id: str | None = None
    generated_at: datetime
    connectivity: Connectivity

    speed: ModuleStatus[SpeedRecommendation]
    route: ModuleStatus[RouteRecommendation]
    maintenance: ModuleStatus[MaintenanceStatus]
    safety: ModuleStatus[SafetyState]

    # Emissions layer -- same telemetry, no extra sensors (profile, Problem 3)
    voyage_fuel_used_l: float | None = Field(None, ge=0)
    voyage_co2_kg: float | None = Field(None, ge=0)
    voyage_co2_avoided_kg: float | None = Field(
        None, description="Against the vessel's own pre-Marine-AI baseline burn."
    )

    language: Literal["en", "fil"] = "en"

    advisory_only: Literal[True] = Field(
        True,
        description="Constant by design. Marine-AI never overrides the captain and never "
        "actuates the vessel. Rendered persistently on the display, not buried in a settings page.",
    )

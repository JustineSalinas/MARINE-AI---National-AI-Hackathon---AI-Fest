"""Telemetry contracts — the three input systems from the technical profile.

Throttling, Routing, and Electro-Mechanical each map to one frame model. A
`TelemetryFrame` is one timestamped observation carrying all three plus the
operator context in force at that moment.

Every sensor field is Optional. A retrofit kit is modular: a vessel with only
the Phase 1 core sensor set must produce valid frames without the recommended
additions. Downstream code must handle `None` rather than assume presence --
this is the difference between a demo and a system that survives a real install.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Latitude = Annotated[float, Field(ge=-90, le=90)]
Longitude = Annotated[float, Field(ge=-180, le=180)]
Compass = Annotated[float, Field(ge=0, lt=360)]


class HullType(str, Enum):
    FIBERGLASS_MONOHULL = "fiberglass_monohull"
    FIBERGLASS_OUTRIGGER = "fiberglass_outrigger"  # bangka / pumpboat
    STEEL_MONOHULL = "steel_monohull"


class ThrottlingFrame(BaseModel):
    """Fuel and engine-load channels. Feeds AI Speed Optimization."""

    model_config = ConfigDict(extra="forbid")

    # Core sensor set
    fuel_level_l: float | None = Field(None, ge=0, description="Tank remaining, litres")
    engine_rpm: float | None = Field(None, ge=0, le=6000)
    fuel_flow_lph: float | None = Field(
        None, ge=0, description="Ground truth burn rate. Trains and validates the fuel model."
    )
    throttle_position_pct: float | None = Field(
        None, ge=0, le=100, description="Actual captain input. Closes the advice/action feedback loop."
    )

    # Recommended additions (profile 3.2)
    engine_torque_nm: float | None = Field(
        None, ge=0, description="Stronger fuel-burn predictor than RPM alone."
    )
    speed_through_water_kn: float | None = Field(
        None, ge=0, description="Differenced against GPS ground speed to isolate current."
    )

    # Cloud-sourced environment at the vessel's position
    wind_speed_kn: float | None = Field(None, ge=0)
    wind_direction_deg: Compass | None = Field(None, description="Direction wind is coming FROM.")
    current_speed_kn: float | None = Field(None, ge=0)
    current_direction_deg: Compass | None = Field(None, description="Direction current is flowing TOWARD.")
    tide_level_m: float | None = None


class RoutingFrame(BaseModel):
    """Position and navigational constraints. Feeds AI Route Optimization."""

    model_config = ConfigDict(extra="forbid")

    latitude: Latitude | None = None
    longitude: Longitude | None = None
    heading_deg: Compass | None = None
    speed_over_ground_kn: float | None = Field(None, ge=0)

    depth_m: float | None = Field(
        None, ge=0, description="Under-keel depth. Hard safety constraint on any route."
    )
    wave_height_m: float | None = Field(None, ge=0)

    # Recommended addition. Absent in the hackathon build -- no free live AIS feed.
    nearby_vessel_count: int | None = Field(None, ge=0)


class ElectroMechanicalFrame(BaseModel):
    """Engine health channels. Feeds AI Predictive Maintenance. Edge-resident."""

    model_config = ConfigDict(extra="forbid")

    # Core sensor set
    coolant_temp_c: float | None = Field(None, ge=-20, le=200)
    oil_pressure_kpa: float | None = Field(None, ge=0, le=1500)
    battery_voltage_v: float | None = Field(None, ge=0, le=60)

    # 6-axis IMU -- vibration signature for bearing wear and shaft misalignment
    accel_x_g: float | None = None
    accel_y_g: float | None = None
    accel_z_g: float | None = None
    gyro_x_dps: float | None = None
    gyro_y_dps: float | None = None
    gyro_z_dps: float | None = None

    exhaust_co2_pct: float | None = Field(None, ge=0, le=20)
    exhaust_nox_ppm: float | None = Field(None, ge=0)

    # Recommended additions (profile 3.3)
    oil_particulate_ppm: float | None = Field(None, ge=0)
    exhaust_gas_temp_c: float | None = Field(None, ge=-20, le=900)

    engine_hours: float | None = Field(
        None, ge=0, description="Cumulative run-hours. The denominator of every RUL calculation."
    )


class VesselProfile(BaseModel):
    """Operator one-time input. Calibrates the fuel model to this specific boat."""

    model_config = ConfigDict(extra="forbid")

    vessel_id: str
    name: str
    hull_type: HullType
    length_overall_m: float = Field(gt=0)
    draft_m: float = Field(gt=0)
    displacement_kg: float = Field(gt=0)
    engine_make_model: str
    engine_rated_kw: float = Field(gt=0)
    engine_rated_rpm: float = Field(gt=0)
    passenger_capacity: int = Field(ge=0)


class OperatorContext(BaseModel):
    """Operator per-trip input. Load changes displacement, which changes the optimal throttle."""

    model_config = ConfigDict(extra="forbid")

    passenger_count: int | None = Field(None, ge=0)
    cargo_estimate_kg: float | None = Field(None, ge=0)
    scheduled_arrival: datetime | None = Field(
        None, description="The ETA constraint the route optimizer must respect."
    )


class TelemetryFrame(BaseModel):
    """One timestamped observation from one vessel. The atomic unit of the system."""

    model_config = ConfigDict(extra="forbid")

    vessel_id: str
    ts: datetime = Field(description="Timezone-aware UTC. Validated on ingest.")
    voyage_id: str | None = None

    throttling: ThrottlingFrame = Field(default_factory=ThrottlingFrame)
    routing: RoutingFrame = Field(default_factory=RoutingFrame)
    electro_mechanical: ElectroMechanicalFrame = Field(default_factory=ElectroMechanicalFrame)
    operator: OperatorContext = Field(default_factory=OperatorContext)

    source: str = Field(
        "simulator",
        description="Provenance. 'simulator' for every frame in the hackathon build -- "
        "no hardware was used. Never silently defaults to 'sensor'.",
    )

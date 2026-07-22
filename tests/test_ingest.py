"""Ingest validation tests.

The governing rule under test: never silently repair. Bad data is rejected with
a reason or flagged, never quietly fixed.
"""

from datetime import UTC, datetime, timedelta

import pytest

from packages.contracts import (
    ElectroMechanicalFrame,
    TelemetryFrame,
    ThrottlingFrame,
)
from packages.ingest.validate import (
    FlagCode,
    VesselValidator,
    Verdict,
    validate_voyage,
)

VESSEL = "MV-TEST-01"
T0 = datetime(2026, 7, 22, 4, 0, 0, tzinfo=UTC)


def frame(offset_s: float = 0, **sections) -> TelemetryFrame:
    return TelemetryFrame(vessel_id=VESSEL, ts=T0 + timedelta(seconds=offset_s), **sections)


def nominal(offset_s: float, *, coolant: float | None = None, rpm: float = 1800.0) -> TelemetryFrame:
    """A healthy running frame, with the small sensor jitter real telemetry has.

    The jitter is not decoration. A perfectly constant channel is exactly what
    the freeze detector is built to catch, so noiseless fixtures would make these
    tests assert the opposite of production behaviour.
    """
    i = int(offset_s)
    return frame(
        offset_s,
        throttling=ThrottlingFrame(
            engine_rpm=rpm + (i % 5) - 2, fuel_flow_lph=11.0 + (i % 3) * 0.05
        ),
        electro_mechanical=ElectroMechanicalFrame(
            coolant_temp_c=coolant if coolant is not None else 82.0 + (i % 4) * 0.25,
            oil_pressure_kpa=310.0 + (i % 6) - 3,
            battery_voltage_v=13.8 + (i % 3) * 0.05,
        ),
    )


def frozen(offset_s: float) -> TelemetryFrame:
    """A running engine whose coolant sensor has stuck at a plausible value."""
    i = int(offset_s)
    return frame(
        offset_s,
        throttling=ThrottlingFrame(engine_rpm=1800 + (i % 5) - 2, fuel_flow_lph=11.0),
        electro_mechanical=ElectroMechanicalFrame(
            coolant_temp_c=82.0,  # stuck
            oil_pressure_kpa=310.0 + (i % 6) - 3,
            battery_voltage_v=13.8 + (i % 3) * 0.05,
        ),
    )


def moored(offset_s: float) -> TelemetryFrame:
    """Engine off at the pier. Constant readings here are correct, not a fault."""
    return frame(
        offset_s,
        throttling=ThrottlingFrame(engine_rpm=0.0, fuel_flow_lph=0.0),
        electro_mechanical=ElectroMechanicalFrame(
            coolant_temp_c=31.0, oil_pressure_kpa=0.0, battery_voltage_v=12.6
        ),
    )


@pytest.fixture
def validator() -> VesselValidator:
    return VesselValidator(vessel_id=VESSEL)


class TestTimestamps:
    def test_naive_timestamp_rejected(self, validator):
        bad = TelemetryFrame(vessel_id=VESSEL, ts=datetime(2026, 7, 22, 4, 0, 0))
        result = validator.validate(bad, now=T0)
        assert result.verdict is Verdict.REJECT
        assert result.flags[0].code is FlagCode.TIMESTAMP_NAIVE

    def test_future_timestamp_rejected(self, validator):
        result = validator.validate(frame(3600), now=T0)
        assert result.verdict is Verdict.REJECT
        assert result.flags[0].code is FlagCode.TIMESTAMP_FUTURE

    def test_duplicate_timestamp_rejected(self, validator):
        now = T0 + timedelta(hours=1)
        assert validator.validate(nominal(0), now=now).verdict is Verdict.ACCEPT
        result = validator.validate(nominal(0), now=now)
        assert result.verdict is Verdict.REJECT
        assert result.flags[0].code is FlagCode.TIMESTAMP_DUPLICATE

    def test_out_of_order_rejected_not_reordered(self, validator):
        """Silently sorting would hide a clock fault that corrupts every RUL figure."""
        now = T0 + timedelta(hours=1)
        validator.validate(nominal(10), now=now)
        result = validator.validate(nominal(5), now=now)
        assert result.verdict is Verdict.REJECT
        assert result.flags[0].code is FlagCode.TIMESTAMP_REGRESSED

    def test_gap_is_flagged_but_frame_still_usable(self, validator):
        """Signal loss is expected on these routes. Flag it; do not throw it away."""
        now = T0 + timedelta(hours=1)
        validator.validate(nominal(0), now=now)
        result = validator.validate(nominal(120), now=now)
        assert result.verdict is Verdict.ACCEPT_FLAGGED
        assert result.usable_for_display
        assert not result.usable_for_training
        assert any(f.code is FlagCode.GAP for f in result.flags)


class TestPhysicalPlausibility:
    def test_fuel_flow_with_stopped_engine_flagged(self, validator):
        result = validator.validate(
            frame(0, throttling=ThrottlingFrame(engine_rpm=0, fuel_flow_lph=8.0)),
            now=T0 + timedelta(hours=1),
        )
        assert any(f.code is FlagCode.OUT_OF_RANGE for f in result.flags)

    def test_impossible_coolant_jump_flagged(self, validator):
        """In range, but no engine heats 40 C in one second."""
        now = T0 + timedelta(hours=1)
        validator.validate(nominal(0, coolant=82.0), now=now)
        result = validator.validate(nominal(1, coolant=122.0), now=now)
        assert any(
            f.code is FlagCode.IMPLAUSIBLE_RATE
            and f.stream == "electro_mechanical.coolant_temp_c"
            for f in result.flags
        )

    def test_gradual_warmup_not_flagged(self, validator):
        """The rate check must not fire on normal operation."""
        now = T0 + timedelta(hours=1)
        for i in range(120):
            result = validator.validate(nominal(i, coolant=70.0 + i * 0.1), now=now)
            assert not any(f.code is FlagCode.IMPLAUSIBLE_RATE for f in result.flags)


class TestSensorHealth:
    def test_frozen_sensor_detected(self, validator):
        """A stuck sensor reads as nominal to every model downstream."""
        now = T0 + timedelta(hours=2)
        result = None
        for i in range(90):
            result = validator.validate(frozen(i), now=now)
        assert any(
            f.code is FlagCode.SENSOR_FROZEN
            and f.stream == "electro_mechanical.coolant_temp_c"
            for f in result.flags
        )

    def test_moored_vessel_is_not_a_frozen_sensor(self):
        """Engine off: constant RPM of zero and a flat battery are correct.
        Flagging them would train the crew to ignore the alert that matters."""
        validator = VesselValidator(vessel_id=VESSEL)
        now = T0 + timedelta(hours=2)
        result = None
        for i in range(120):
            result = validator.validate(moored(i), now=now)
        assert not any(f.code is FlagCode.SENSOR_FROZEN for f in result.flags)

    def test_drift_detected_against_own_baseline(self, validator):
        now = T0 + timedelta(hours=2)
        for i in range(120):
            validator.validate(nominal(i, coolant=82.0 + (i % 3) * 0.4), now=now)
        result = validator.validate(nominal(121, coolant=95.0), now=now)
        assert any(f.code is FlagCode.SENSOR_DRIFT for f in result.flags)

    def test_baseline_is_per_vessel(self):
        """A fleet-wide baseline would flatten the vessel-specific wear signature
        the maintenance module exists to learn."""
        a = VesselValidator(vessel_id="MV-A")
        foreign = TelemetryFrame(vessel_id="MV-B", ts=T0)
        assert a.validate(foreign, now=T0 + timedelta(hours=1)).verdict is Verdict.REJECT


class TestVoyageReport:
    def test_clean_voyage_is_not_an_outlier(self):
        frames = [nominal(i) for i in range(200)]
        _, report = validate_voyage(VESSEL, frames, voyage_id="V1")
        assert report.rejected == 0
        assert not report.is_outlier
        assert report.reason is None

    def test_voyage_with_heavy_dropout_flagged_as_outlier(self):
        """Flagged, not silently averaged -- the profile's stated commitment."""
        frames = [nominal(i * 300) for i in range(40)]  # every frame a 5 min gap
        _, report = validate_voyage(VESSEL, frames, voyage_id="V2")
        assert report.is_outlier
        assert "flagged" in report.reason

    def test_empty_voyage_is_an_outlier(self):
        _, report = validate_voyage(VESSEL, [], voyage_id="V3")
        assert report.is_outlier
        assert report.reason == "no frames"

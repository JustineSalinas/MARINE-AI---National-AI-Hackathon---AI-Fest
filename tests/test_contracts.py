"""Contract tests. These guard the two commitments that are easy to erode.

1. Phase 1 predictive maintenance cannot make component-level claims.
2. A modular retrofit missing sensors still produces valid frames.
"""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from packages.contracts import (
    MaintenancePhase,
    MaintenanceStatus,
    TelemetryFrame,
    ThrottlingFrame,
)


def _base_status(**overrides):
    payload = {
        "vessel_id": "MV-TEST-01",
        "generated_at": datetime.now(UTC),
        "phase": MaintenancePhase.PHASE_1_COLD_START,
        "anomaly_score": 0.12,
        "is_anomalous": False,
        "observed_hours": 40.0,
        "baseline_confidence": 0.3,
        "advisory_en": "All engine readings normal.",
        "advisory_fil": "Normal ang lahat ng reading ng makina.",
    }
    payload.update(overrides)
    return payload


class TestPhase1Honesty:
    """The cold-start fairness commitment, enforced rather than trusted."""

    def test_phase_1_nominal_status_is_valid(self):
        status = MaintenanceStatus(**_base_status())
        assert status.phase is MaintenancePhase.PHASE_1_COLD_START
        assert status.likely_component is None

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("likely_component", "Raw water pump impeller"),
            ("recommended_maintenance_date", datetime.now(UTC).date()),
            ("remaining_useful_life_days", 47.0),
            ("required_parts", ["impeller-3857"]),
            ("estimated_downtime_hours", 6.0),
        ],
    )
    def test_phase_1_rejects_every_component_level_field(self, field, value):
        with pytest.raises(ValidationError, match="cannot make component-level claims"):
            MaintenanceStatus(**_base_status(**{field: value}))

    def test_phase_1_error_names_the_offending_fields(self):
        with pytest.raises(ValidationError) as exc:
            MaintenanceStatus(
                **_base_status(
                    likely_component="Injector #2",
                    remaining_useful_life_days=12.0,
                )
            )
        message = str(exc.value)
        assert "likely_component" in message
        assert "remaining_useful_life_days" in message

    def test_phase_2_may_make_component_level_claims(self):
        status = MaintenanceStatus(
            **_base_status(
                phase=MaintenancePhase.PHASE_2_MATURE,
                observed_hours=9_000.0,
                baseline_confidence=0.91,
                likely_component="Raw water pump impeller",
                likely_component_fil="Impeller ng raw water pump",
                remaining_useful_life_days=47.0,
                rul_confidence_interval_days=(40.0, 55.0),
                recommended_maintenance_date=(datetime.now(UTC) + timedelta(days=44)).date(),
                required_parts=["impeller-3857", "gasket-114"],
                estimated_downtime_hours=6.0,
            )
        )
        assert status.remaining_useful_life_days == 47.0


class TestModularRetrofit:
    """Phase 1 core sensors only -- the recommended additions are absent."""

    def test_frame_valid_with_core_sensors_only(self):
        frame = TelemetryFrame(
            vessel_id="MV-TEST-01",
            ts=datetime.now(UTC),
            throttling=ThrottlingFrame(engine_rpm=1800, fuel_flow_lph=11.4),
        )
        assert frame.throttling.engine_torque_nm is None
        assert frame.routing.latitude is None
        assert frame.electro_mechanical.coolant_temp_c is None

    def test_frame_valid_with_no_sensors_at_all(self):
        frame = TelemetryFrame(vessel_id="MV-TEST-01", ts=datetime.now(UTC))
        assert frame.source == "simulator"

    def test_provenance_defaults_to_simulator_not_sensor(self):
        """No hardware was used in this build. The default must never claim otherwise."""
        frame = TelemetryFrame(vessel_id="MV-TEST-01", ts=datetime.now(UTC))
        assert frame.source == "simulator"

    def test_out_of_range_sensor_value_is_rejected(self):
        with pytest.raises(ValidationError):
            ThrottlingFrame(engine_rpm=-5)
        with pytest.raises(ValidationError):
            ThrottlingFrame(throttle_position_pct=140)

    def test_unknown_field_is_rejected(self):
        """extra='forbid' catches typos and schema drift at the boundary."""
        with pytest.raises(ValidationError):
            ThrottlingFrame(engine_rpms=1800)

"""End-to-end tests for the advisory API.

These are the tests that stand between the demo and a repeat of the prototype's
central flaw: an interface where the weather sliders move the readouts and never
the boat. Every claim the bridge display makes is asserted here against a real
HTTP round trip.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app

CALM = {"wind_speed_kn": 0.0, "wave_height_m": 0.0}
HEAD_SEA = {
    "wind_speed_kn": 20.0,
    "wind_direction_deg": 0.0,
    "wave_height_m": 1.5,
    "wave_direction_deg": 0.0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def advise(client, **kwargs) -> dict:
    body = {"heading_deg": 0.0, "distance_remaining_nm": 2.0, **kwargs}
    r = client.post("/advise", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_health_declares_the_advisory_boundary(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["advisory_only"] is True


def test_advise_works_with_no_body_at_all(client):
    """Every field has a defensible default so the simulator can boot before the
    operator has entered a vessel profile."""
    r = client.post("/advise", json={})
    assert r.status_code == 200
    assert r.json()["recommendation"]["recommended_rpm"] > 0


def test_bad_input_is_rejected_not_coerced(client):
    assert client.post("/advise", json={"heading_deg": 999}).status_code == 422
    assert client.post("/advise", json={"distance_remaining_nm": -1}).status_code == 422
    assert client.post("/advise", json={"nonsense": True}).status_code == 422


# --- The claim the whole rewrite exists to make true ------------------------


def test_weather_slows_the_boat_at_the_same_throttle(client):
    """Same RPM, worse weather, slower boat.

    The prototype computed speed from the throttle slider alone. If this ever
    stops holding, the environmental controls are decorative again.
    """
    calm = advise(client, sea=CALM, current_rpm=2400.0)
    rough = advise(client, sea=HEAD_SEA, current_rpm=2400.0)

    assert rough["achievable_speed_kn"] < calm["achievable_speed_kn"]
    assert calm["achievable_speed_kn"] - rough["achievable_speed_kn"] > 0.3


def test_weather_raises_the_cost_of_holding_a_schedule(client):
    """Same distance, same deadline, worse weather -> more fuel per hour."""
    calm = advise(client, sea=CALM, minutes_available=20.0)
    rough = advise(client, sea=HEAD_SEA, minutes_available=20.0)

    assert (
        rough["recommendation"]["predicted_burn_lph"]
        > calm["recommendation"]["predicted_burn_lph"]
    )


def test_head_sea_is_itemised_so_the_advice_can_explain_itself(client):
    rough = advise(client, sea=HEAD_SEA, minutes_available=20.0)
    power = rough["power"]

    assert power["wave_kw"] > 0
    assert power["wind_kw"] > 0
    assert power["environmental_penalty_pct"] > 0
    assert power["total_kw"] == pytest.approx(
        power["calm_water_kw"] + power["wind_kw"] + power["wave_kw"], rel=0.01
    )


def test_following_sea_is_cheaper_than_head_sea(client):
    following = dict(HEAD_SEA, wind_direction_deg=180.0, wave_direction_deg=180.0)
    head = advise(client, sea=HEAD_SEA, minutes_available=25.0)
    follow = advise(client, sea=following, minutes_available=25.0)

    assert (
        follow["recommendation"]["predicted_burn_lph"]
        < head["recommendation"]["predicted_burn_lph"]
    )


def test_a_foul_current_shows_up_as_speed_through_water(client):
    """A vessel making 8 kn over ground against 2 kn of current is driving its
    hull at 10 kn, and paying 10-knot fuel for 8-knot progress."""
    foul = advise(
        client,
        sea={"current_speed_kn": 2.0, "current_direction_deg": 180.0},
        minutes_available=30.0,
    )
    rec = foul["recommendation"]
    assert rec["recommended_speed_kn"] > foul["power"]["speed_through_water_kn"] - 1e-6


# --- Load ------------------------------------------------------------------


def test_a_full_boat_is_a_thirstier_boat(client):
    empty = advise(client, minutes_available=20.0, passenger_count=0)
    full = advise(client, minutes_available=20.0, passenger_count=60)

    assert (
        full["recommendation"]["predicted_burn_lph"]
        > empty["recommendation"]["predicted_burn_lph"]
    )


# --- Engine condition: the Problem 1 -> Problem 2 link ----------------------


def test_a_worn_engine_costs_money_per_hour(client):
    healthy = advise(client, minutes_available=20.0, egt_excess_ratio=1.0)
    worn = advise(client, minutes_available=20.0, egt_excess_ratio=1.06)

    assert worn["wear"]["multiplier"] >= healthy["wear"]["multiplier"]
    assert worn["wear"]["penalty_lph"] >= healthy["wear"]["penalty_lph"]
    if worn["model_trained"]:
        assert worn["wear"]["multiplier"] > 1.0
        assert worn["wear"]["penalty_php_per_hour"] > 0


def test_unknown_engine_condition_assumes_health_and_lowers_confidence(client):
    unknown = advise(client, minutes_available=20.0)
    assert unknown["wear"]["multiplier"] == 1.0
    assert unknown["recommendation"]["model_confidence"] < 1.0
    assert any("assumed healthy" in n for n in unknown["notes"])


# --- Schedule --------------------------------------------------------------


def test_impossible_schedule_is_flagged_not_faked(client):
    hard = advise(
        client,
        sea={"wind_speed_kn": 40.0, "wind_direction_deg": 0.0, "wave_height_m": 3.0},
        distance_remaining_nm=20.0,
        minutes_available=15.0,
    )
    assert hard["feasible"] is False
    assert "late" in hard["recommendation"]["advisory_en"].lower()


def test_recommendation_never_exceeds_rated_power(client):
    hard = advise(client, sea=HEAD_SEA, distance_remaining_nm=20.0, minutes_available=15.0)
    assert hard["recommendation"]["recommended_rpm"] <= 2800.0 * 1.001
    assert hard["power"]["total_kw"] <= 90.0 * 1.001


def test_more_time_is_never_more_fuel(client):
    tight = advise(client, minutes_available=12.0)
    loose = advise(client, minutes_available=30.0)
    assert (
        loose["recommendation"]["predicted_burn_lph"]
        <= tight["recommendation"]["predicted_burn_lph"]
    )


# --- Contract guarantees the display depends on -----------------------------


def test_advisory_is_bilingual_and_never_imperative(client):
    body = advise(client, minutes_available=20.0, current_rpm=2600.0)
    rec = body["recommendation"]

    assert rec["advisory_en"] and rec["advisory_fil"]
    assert rec["advisory_en"] != rec["advisory_fil"]
    assert rec["advisory_source"] == "template"
    for banned in ("reduce", "increase", "slow down", "throttle back"):
        assert banned not in rec["advisory_en"].lower()


def test_savings_in_pesos_track_savings_in_litres(client):
    body = advise(client, minutes_available=20.0, current_rpm=2600.0, php_per_litre=70.0)
    rec = body["recommendation"]
    assert rec["savings_php_per_hour"] == pytest.approx(rec["savings_lph"] * 70.0)


def test_curve_is_returned_so_the_browser_never_computes_physics(client):
    body = advise(client, minutes_available=20.0)
    curve = body["curve"]

    assert len(curve) > 5
    speeds = [p["speed_kn"] for p in curve]
    assert speeds == sorted(speeds)
    assert all(p["litres_per_hour"] > 0 for p in curve)
    assert max(p["speed_kn"] for p in curve) <= body["max_speed_kn"] + 0.51


def test_emissions_come_from_the_same_burn_figure(client):
    body = advise(client, minutes_available=20.0)
    lph = body["recommendation"]["predicted_burn_lph"]
    assert body["emissions"]["co2_kg_per_hour"] == pytest.approx(lph * 2.68, rel=1e-6)

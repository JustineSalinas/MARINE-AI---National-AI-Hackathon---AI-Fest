"""Tests for the throttle optimizer.

The load-bearing test in this file is
`test_head_sea_slows_the_boat_at_fixed_throttle`. The prototype simulator
computed speed from the throttle slider alone, so weather moved the readouts and
never the vessel. That made the product's central claim decorative. If that test
ever fails, the demo is lying again.
"""

from __future__ import annotations

import pytest

from services.speed.fuel import EngineSpec, FuelMap
from services.speed.optimizer import (
    ThrottleAdvice,
    advisory_sentences,
    as_recommendation,
    optimise_throttle,
    performance_curve,
    power_for_rpm,
    rpm_for_power,
    speed_for_power_kn,
)
from services.speed.resistance import SeaState, VesselHull, required_shaft_power_kw

HULL = VesselHull(length_waterline_m=11.5, beam_m=2.8, draft_m=1.1, displacement_kg=8500.0)
SPEC = EngineSpec(rated_kw=90.0, rated_rpm=2800.0)
CALM = SeaState()


@pytest.fixture
def fuel_map():
    return FuelMap(SPEC)


# --- speed_for_power: the inverse solve ------------------------------------


def test_speed_for_power_round_trips_against_the_forward_model():
    """Solve for speed, feed it back, and the power must come out where it started."""
    for target_kw in (10.0, 25.0, 50.0, 90.0):
        speed = speed_for_power_kn(HULL, target_kw, CALM, 0.0)
        back = required_shaft_power_kw(HULL, speed, CALM, 0.0).total_kw
        assert back == pytest.approx(target_kw, rel=0.02)


def test_more_power_never_means_less_speed():
    speeds = [speed_for_power_kn(HULL, kw, CALM, 0.0) for kw in range(5, 95, 5)]
    assert speeds == sorted(speeds)


def test_zero_power_means_stopped():
    assert speed_for_power_kn(HULL, 0.0, CALM, 0.0) == 0.0
    assert speed_for_power_kn(HULL, -5.0, CALM, 0.0) == 0.0


def test_head_sea_slows_the_boat_at_fixed_throttle():
    """The test the whole simulator rewrite exists for.

    Same engine, same throttle, worse weather -> slower boat. The prototype had
    speed = f(throttle) only, so this relationship did not exist and every
    environmental slider was decorative.
    """
    rough = SeaState(wind_speed_kn=20.0, wind_direction_deg=0.0, wave_height_m=1.5)

    calm_speed = speed_for_power_kn(HULL, 60.0, CALM, 0.0)
    rough_speed = speed_for_power_kn(HULL, 60.0, rough, heading_deg=0.0)

    assert rough_speed < calm_speed
    # Not a rounding artifact -- this must be a difference a captain would notice.
    assert calm_speed - rough_speed > 0.5


def test_following_sea_is_faster_than_head_sea_for_the_same_power():
    wind = 20.0
    head = SeaState(wind_speed_kn=wind, wind_direction_deg=0.0, wave_height_m=1.2)
    following = SeaState(wind_speed_kn=wind, wind_direction_deg=180.0, wave_height_m=1.2)

    assert speed_for_power_kn(HULL, 60.0, following, 0.0) > speed_for_power_kn(
        HULL, 60.0, head, 0.0
    )


def test_foul_current_costs_speed_over_ground():
    """A current on the bow means driving the hull faster than progress made."""
    foul = SeaState(current_speed_kn=2.0, current_direction_deg=180.0)
    assert speed_for_power_kn(HULL, 50.0, foul, heading_deg=0.0) < speed_for_power_kn(
        HULL, 50.0, CALM, heading_deg=0.0
    )


# --- RPM mapping ------------------------------------------------------------


def test_rpm_power_round_trip():
    for kw in (5.0, 30.0, 90.0):
        assert power_for_rpm(SPEC, rpm_for_power(SPEC, kw)) == pytest.approx(kw)


def test_rated_power_is_rated_rpm():
    assert rpm_for_power(SPEC, SPEC.rated_kw) == pytest.approx(SPEC.rated_rpm)


def test_propeller_law_is_cubic():
    """Half the rated RPM should absorb an eighth of the power."""
    assert power_for_rpm(SPEC, SPEC.rated_rpm / 2) == pytest.approx(SPEC.rated_kw / 8)


def test_zero_and_negative_are_handled():
    assert rpm_for_power(SPEC, 0.0) == 0.0
    assert power_for_rpm(SPEC, 0.0) == 0.0
    assert rpm_for_power(SPEC, -1.0) == 0.0


# --- the sweep --------------------------------------------------------------


def test_performance_curve_burn_rises_with_speed(fuel_map):
    curve = performance_curve(HULL, SPEC, fuel_map, CALM, 0.0, max_speed_kn=12.0)
    burns = [o.litres_per_hour for o in curve]
    assert burns == sorted(burns)
    assert len(curve) > 10


def test_cheapest_speed_per_mile_is_an_interior_optimum(fuel_map):
    """Litres per hour alone would recommend idling; litres per mile does not.

    Crawling burns little per hour but burns it for hours, and the idle governor
    sets a floor below which fuel flow stops falling. Going flat out pays the
    cubic law. So the per-mile minimum sits strictly between the extremes --
    which is the property that makes the optimizer's answer non-trivial.
    """
    curve = performance_curve(HULL, SPEC, fuel_map, CALM, 0.0, min_speed_kn=1.0, max_speed_kn=12.0)
    per_mile = [o.litres_for_distance(10.0) for o in curve]
    best = per_mile.index(min(per_mile))

    assert 0 < best < len(curve) - 1, "optimum landed on an endpoint"
    assert curve[best].speed_kn == pytest.approx(4.5, abs=2.0)


def test_idle_floor_makes_crawling_expensive_per_mile(fuel_map):
    """Without the governor floor, one knot looks nearly free and the optimizer
    concludes the cheapest crossing is to barely move."""
    curve = performance_curve(HULL, SPEC, fuel_map, CALM, 0.0, min_speed_kn=1.0, max_speed_kn=8.0)
    crawl = curve[0]
    assert crawl.litres_per_hour == pytest.approx(SPEC.idle_burn_lph)
    # 10 nm at 1 knot is ten hours of idling.
    assert crawl.litres_for_distance(10.0) > 10.0


# --- the optimizer ----------------------------------------------------------


def test_recommendation_meets_the_schedule(fuel_map):
    advice = optimise_throttle(
        HULL, SPEC, fuel_map, CALM, 0.0, distance_remaining_nm=6.0, minutes_available=45.0
    )
    required_kn = 6.0 / (45.0 / 60.0)
    assert advice.feasible
    assert advice.recommended.speed_kn >= required_kn - 0.01


def test_never_recommends_more_than_the_engine_can_deliver(fuel_map):
    """In heavy weather the optimizer must not advise a speed the boat cannot hold."""
    rough = SeaState(wind_speed_kn=35.0, wind_direction_deg=0.0, wave_height_m=2.5)
    advice = optimise_throttle(
        HULL, SPEC, fuel_map, rough, 0.0, distance_remaining_nm=6.0, minutes_available=30.0
    )
    assert advice.recommended.power.total_kw <= SPEC.rated_kw * 1.001
    assert advice.recommended.rpm <= SPEC.rated_rpm * 1.001


def test_impossible_schedule_is_reported_not_faked(fuel_map):
    """An unmeetable ETA must surface as infeasible, with a note."""
    rough = SeaState(wind_speed_kn=40.0, wind_direction_deg=0.0, wave_height_m=3.0)
    advice = optimise_throttle(
        HULL, SPEC, fuel_map, rough, 0.0, distance_remaining_nm=20.0, minutes_available=20.0
    )
    assert not advice.feasible
    assert any("all this engine can hold" in n for n in advice.notes)


def test_a_looser_schedule_is_never_more_expensive(fuel_map):
    """More time available cannot cost more fuel per voyage."""
    tight = optimise_throttle(
        HULL, SPEC, fuel_map, CALM, 0.0, distance_remaining_nm=6.0, minutes_available=40.0
    )
    loose = optimise_throttle(
        HULL, SPEC, fuel_map, CALM, 0.0, distance_remaining_nm=6.0, minutes_available=70.0
    )
    assert loose.recommended.litres_for_distance(6.0) <= tight.recommended.litres_for_distance(6.0)


def test_savings_are_reported_against_current_throttle(fuel_map):
    advice = optimise_throttle(
        HULL,
        SPEC,
        fuel_map,
        CALM,
        0.0,
        distance_remaining_nm=6.0,
        minutes_available=60.0,
        current_rpm=2600.0,
    )
    assert advice.current is not None
    assert advice.savings_lph > 0
    assert advice.eta_impact_minutes > 0  # slower arrival is the trade


def test_negative_savings_are_shown_honestly(fuel_map):
    """A captain already running leaner than needed must not be told he is wasting."""
    advice = optimise_throttle(
        HULL,
        SPEC,
        fuel_map,
        CALM,
        0.0,
        distance_remaining_nm=6.0,
        minutes_available=40.0,
        current_rpm=900.0,
    )
    assert advice.savings_lph < 0
    en, _ = advisory_sentences(advice)
    assert "cheaper" in en


def test_no_schedule_optimises_fuel_per_mile(fuel_map):
    advice = optimise_throttle(
        HULL, SPEC, fuel_map, CALM, 0.0, distance_remaining_nm=6.0, minutes_available=None
    )
    assert any("no arrival time" in n for n in advice.notes)


# --- the sentence -----------------------------------------------------------


def test_advisory_is_never_imperative(fuel_map):
    """PRODUCT.md: '1650 RPM saves 2.1 L/h' is right; 'Reduce to 1650 RPM' is wrong."""
    advice = optimise_throttle(
        HULL,
        SPEC,
        fuel_map,
        CALM,
        0.0,
        distance_remaining_nm=6.0,
        minutes_available=60.0,
        current_rpm=2600.0,
    )
    en, fil = advisory_sentences(advice)
    for banned in ("reduce", "increase", "set ", "throttle back", "slow down"):
        assert banned not in en.lower()
    assert en and fil
    assert en != fil  # Filipino is a real translation, not a copy


def test_advisory_includes_pesos_when_a_price_is_known(fuel_map):
    advice = optimise_throttle(
        HULL,
        SPEC,
        fuel_map,
        CALM,
        0.0,
        distance_remaining_nm=6.0,
        minutes_available=60.0,
        current_rpm=2600.0,
    )
    en, fil = advisory_sentences(advice, php_per_litre=70.0)
    assert "PHP" in en and "PHP" in fil


def test_infeasible_advisory_says_the_arrival_will_be_late(fuel_map):
    advice = ThrottleAdvice(
        recommended=performance_curve(HULL, SPEC, fuel_map, CALM, 0.0, max_speed_kn=8.0)[-1],
        current=None,
        savings_lph=0.0,
        eta_impact_minutes=0.0,
        feasible=False,
    )
    en, fil = advisory_sentences(advice)
    assert "late" in en.lower()
    assert "mahuhuli" in fil.lower()


# --- the wire contract ------------------------------------------------------


def test_flattens_into_the_speed_contract(fuel_map):
    advice = optimise_throttle(
        HULL,
        SPEC,
        fuel_map,
        CALM,
        0.0,
        distance_remaining_nm=6.0,
        minutes_available=60.0,
        current_rpm=2600.0,
    )
    rec = as_recommendation(advice, vessel_id="MV-TEST-01", php_per_litre=70.0)

    assert rec.vessel_id == "MV-TEST-01"
    assert rec.recommended_rpm > 0
    assert rec.predicted_burn_lph > 0
    assert rec.savings_php_per_hour == pytest.approx(rec.savings_lph * 70.0)
    assert rec.advisory_source == "template"
    assert 0 <= rec.model_confidence <= 1

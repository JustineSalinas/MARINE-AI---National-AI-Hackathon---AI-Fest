"""Hull resistance tests.

Physics with a sign error does not crash. It quietly tells a captain to speed up
into a head sea and bills it as a saving. These tests pin down direction and
magnitude, not just "returns a number".

Reference vessel: a 12 m fiberglass passenger boat of the kind these routes run.
"""


import pytest

from services.speed.resistance import (
    HULL_SPEED_SLR,
    SeaState,
    VesselHull,
    admiralty_power_kw,
    calibrate_admiralty,
    calm_water_power_kw,
    required_shaft_power_kw,
    speed_length_ratio,
    speed_through_water_kn,
    wave_resistance_kw,
    wind_resistance_kw,
)

HULL = VesselHull(
    length_waterline_m=11.5,
    beam_m=3.2,
    draft_m=0.9,
    displacement_kg=8_500.0,
)

CALM = SeaState()
CRUISE_KN = 12.0
NORTH = 0.0


class TestCalmWater:
    def test_zero_speed_costs_nothing(self):
        assert calm_water_power_kw(HULL, 0.0) == 0.0

    def test_bare_admiralty_term_is_cubic_in_speed(self):
        """The cubic law, isolated from the semi-displacement correction."""
        assert admiralty_power_kw(HULL, 20.0) / admiralty_power_kw(HULL, 10.0) == pytest.approx(
            8.0, rel=1e-6
        )

    def test_hull_speed_is_about_eight_knots_for_this_hull(self):
        """11.5 m waterline. The routes are run above this, which is the whole
        reason the semi-displacement correction exists."""
        assert speed_length_ratio(HULL, 8.2) == pytest.approx(HULL_SPEED_SLR, abs=0.02)

    def test_no_wave_making_penalty_below_hull_speed(self):
        assert calm_water_power_kw(HULL, 7.0) == pytest.approx(
            admiralty_power_kw(HULL, 7.0), rel=1e-9
        )

    def test_wave_making_penalty_applies_above_hull_speed(self):
        assert calm_water_power_kw(HULL, 12.0) > admiralty_power_kw(HULL, 12.0)

    def test_one_knot_slower_saves_more_than_the_cubic_law_alone(self):
        """12 -> 11 kn is an 8% speed cut. The cubic alone gives 23%; with the
        hump penalty it is over 30%. This is the largest fuel lever in the
        product, and it would be invisible without the correction."""
        fast = calm_water_power_kw(HULL, 12.0)
        slow = calm_water_power_kw(HULL, 11.0)
        saving = (fast - slow) / fast
        assert saving > 0.29
        assert saving > (
            admiralty_power_kw(HULL, 12.0) - admiralty_power_kw(HULL, 11.0)
        ) / admiralty_power_kw(HULL, 12.0)

    def test_added_passengers_increase_power(self):
        """Load changes displacement, which changes the optimal throttle."""
        empty = calm_water_power_kw(HULL, CRUISE_KN)
        laden = calm_water_power_kw(HULL, CRUISE_KN, added_load_kg=30 * 65.0)
        assert laden > empty

    @pytest.mark.parametrize(
        ("speed_kn", "low_kw", "high_kw"),
        [
            (8.0, 20.0, 45.0),  # at hull speed, economical
            (10.0, 45.0, 90.0),  # pushing past it
            (12.0, 100.0, 190.0),  # typical scheduled cruise, expensive
        ],
    )
    def test_power_is_plausible_for_a_12m_passenger_boat(self, speed_kn, low_kw, high_kw):
        """Sanity anchors against the engines these boats actually carry.

        This test is the reason the model has a wave-making term at all: the bare
        Admiralty cubic put 12 knots at 22 kW, which no 8.5 tonne hull achieves.
        """
        assert low_kw < calm_water_power_kw(HULL, speed_kn) < high_kw


class TestCurrent:
    def test_foul_current_means_driving_the_hull_faster_than_you_travel(self):
        """The correction captains most often misjudge: 10 kn over ground against
        a 2 kn foul current is 12 kn through the water, at 12-knot fuel cost."""
        sea = SeaState(current_speed_kn=2.0, current_direction_deg=180.0)  # flows south
        stw = speed_through_water_kn(10.0, sea, heading_deg=NORTH)  # steaming north
        assert stw == pytest.approx(12.0, abs=1e-6)

    def test_fair_current_is_free_speed(self):
        sea = SeaState(current_speed_kn=2.0, current_direction_deg=0.0)  # flows north
        stw = speed_through_water_kn(10.0, sea, heading_deg=NORTH)
        assert stw == pytest.approx(8.0, abs=1e-6)

    def test_beam_current_barely_affects_along_track_speed(self):
        sea = SeaState(current_speed_kn=2.0, current_direction_deg=90.0)
        stw = speed_through_water_kn(10.0, sea, heading_deg=NORTH)
        assert stw == pytest.approx(10.0, abs=1e-6)

    def test_current_never_produces_negative_speed(self):
        sea = SeaState(current_speed_kn=8.0, current_direction_deg=0.0)
        assert speed_through_water_kn(3.0, sea, heading_deg=NORTH) >= 0.0


class TestWind:
    def test_headwind_costs_power(self):
        head = SeaState(wind_speed_kn=20.0, wind_direction_deg=0.0)  # from the north
        assert wind_resistance_kw(HULL, CRUISE_KN, head, heading_deg=NORTH) > 0.0

    def test_tailwind_returns_some_power(self):
        tail = SeaState(wind_speed_kn=20.0, wind_direction_deg=180.0)
        assert wind_resistance_kw(HULL, CRUISE_KN, tail, heading_deg=NORTH) < 0.0

    def test_headwind_costs_more_than_a_tailwind_returns(self):
        """Thrust recovery downwind is inefficient; the penalty is asymmetric."""
        head = SeaState(wind_speed_kn=20.0, wind_direction_deg=0.0)
        tail = SeaState(wind_speed_kn=20.0, wind_direction_deg=180.0)
        cost = wind_resistance_kw(HULL, CRUISE_KN, head, heading_deg=NORTH)
        gain = abs(wind_resistance_kw(HULL, CRUISE_KN, tail, heading_deg=NORTH))
        assert cost > gain

    def test_calm_air_still_costs_something_from_the_vessel_own_motion(self):
        """Even in still air the boat pushes through it at 12 knots."""
        assert wind_resistance_kw(HULL, CRUISE_KN, CALM, heading_deg=NORTH) > 0.0

    def test_stronger_wind_costs_more(self):
        light = SeaState(wind_speed_kn=5.0, wind_direction_deg=0.0)
        gale = SeaState(wind_speed_kn=35.0, wind_direction_deg=0.0)
        assert wind_resistance_kw(HULL, CRUISE_KN, gale, heading_deg=NORTH) > wind_resistance_kw(
            HULL, CRUISE_KN, light, heading_deg=NORTH
        )


class TestWaves:
    def test_waves_never_help(self):
        for bearing in range(0, 360, 30):
            sea = SeaState(wave_height_m=1.2, wave_direction_deg=float(bearing))
            assert wave_resistance_kw(HULL, CRUISE_KN, sea, heading_deg=NORTH) >= 0.0

    def test_flat_water_costs_nothing(self):
        assert wave_resistance_kw(HULL, CRUISE_KN, CALM, heading_deg=NORTH) == 0.0

    def test_cost_scales_with_the_square_of_wave_height(self):
        """1.5 m seas cost ~9x what 0.5 m seas cost. This is why sea state
        dominates the throttle recommendation on exposed legs."""
        small = SeaState(wave_height_m=0.5, wave_direction_deg=0.0)
        big = SeaState(wave_height_m=1.5, wave_direction_deg=0.0)
        ratio = wave_resistance_kw(HULL, CRUISE_KN, big, heading_deg=NORTH) / wave_resistance_kw(
            HULL, CRUISE_KN, small, heading_deg=NORTH
        )
        assert ratio == pytest.approx(9.0, rel=1e-6)

    def test_head_seas_cost_more_than_following_seas(self):
        head = SeaState(wave_height_m=1.2, wave_direction_deg=0.0)
        following = SeaState(wave_height_m=1.2, wave_direction_deg=180.0)
        assert wave_resistance_kw(HULL, CRUISE_KN, head, heading_deg=NORTH) > wave_resistance_kw(
            HULL, CRUISE_KN, following, heading_deg=NORTH
        )


class TestTotalPower:
    def test_total_is_never_negative_even_in_a_strong_following_wind(self):
        sea = SeaState(wind_speed_kn=40.0, wind_direction_deg=180.0)
        assert required_shaft_power_kw(HULL, 6.0, sea, heading_deg=NORTH).total_kw >= 0.0

    def test_rough_weather_costs_more_than_calm(self):
        rough = SeaState(
            wind_speed_kn=25.0, wind_direction_deg=0.0, wave_height_m=1.4, wave_direction_deg=0.0
        )
        assert (
            required_shaft_power_kw(HULL, CRUISE_KN, rough, heading_deg=NORTH).total_kw
            > required_shaft_power_kw(HULL, CRUISE_KN, CALM, heading_deg=NORTH).total_kw
        )

    def test_breakdown_sums_to_total(self):
        """The itemisation is shown to the captain, so it must actually add up."""
        sea = SeaState(wind_speed_kn=18.0, wind_direction_deg=45.0, wave_height_m=1.0)
        b = required_shaft_power_kw(HULL, CRUISE_KN, sea, heading_deg=NORTH)
        assert b.calm_water_kw + b.wind_kw + b.wave_kw == pytest.approx(b.total_kw, rel=1e-9)

    def test_environmental_penalty_is_reported_as_a_share(self):
        rough = SeaState(
            wind_speed_kn=28.0, wind_direction_deg=0.0, wave_height_m=1.5, wave_direction_deg=0.0
        )
        penalty = required_shaft_power_kw(HULL, CRUISE_KN, rough, heading_deg=NORTH)
        assert 0.0 < penalty.environmental_penalty_pct < 100.0
        assert penalty.environmental_penalty_pct > required_shaft_power_kw(
            HULL, CRUISE_KN, CALM, heading_deg=NORTH
        ).environmental_penalty_pct

    def test_foul_current_raises_power_at_the_same_ground_speed(self):
        """Same progress over the ground, more fuel. The captain cannot see this
        without the system; it is one of the clearest wins in the product."""
        foul = SeaState(current_speed_kn=2.0, current_direction_deg=180.0)
        assert (
            required_shaft_power_kw(HULL, 10.0, foul, heading_deg=NORTH).total_kw
            > required_shaft_power_kw(HULL, 10.0, CALM, heading_deg=NORTH).total_kw
        )


class TestCalibration:
    def test_recovers_a_known_coefficient(self):
        """Calibration is what makes this a model of one specific boat."""
        truth = 385.0
        hull = VesselHull(
            length_waterline_m=11.5,
            beam_m=3.2,
            draft_m=0.9,
            displacement_kg=8_500.0,
            admiralty_coefficient=truth,
        )
        observations = [
            (speed, calm_water_power_kw(hull, speed)) for speed in (8.0, 10.0, 12.0, 14.0)
        ]
        assert calibrate_admiralty(hull, observations) == pytest.approx(truth, rel=1e-6)

    def test_median_fit_survives_a_bad_observation(self):
        truth = 385.0
        hull = VesselHull(
            length_waterline_m=11.5,
            beam_m=3.2,
            draft_m=0.9,
            displacement_kg=8_500.0,
            admiralty_coefficient=truth,
        )
        observations = [(s, calm_water_power_kw(hull, s)) for s in (8.0, 10.0, 12.0, 14.0, 16.0)]
        observations.append((12.0, 5.0))  # a plainly broken fuel-flow reading
        assert calibrate_admiralty(hull, observations) == pytest.approx(truth, rel=0.05)

    def test_refuses_to_calibrate_from_nothing(self):
        with pytest.raises(ValueError, match="no usable"):
            calibrate_admiralty(HULL, [(0.0, 0.0), (-1.0, 5.0)])


class TestHullValidation:
    def test_rejects_impossible_hull(self):
        with pytest.raises(ValueError):
            VesselHull(length_waterline_m=0.0, beam_m=3.2, draft_m=0.9, displacement_kg=8500.0)
        with pytest.raises(ValueError):
            VesselHull(
                length_waterline_m=11.5,
                beam_m=3.2,
                draft_m=0.9,
                displacement_kg=8500.0,
                propulsive_efficiency=1.4,
            )

    def test_superstructure_area_is_estimated_when_absent(self):
        assert HULL.effective_superstructure_area_m2 == pytest.approx(
            3.2 * (0.6 * 0.9 + 1.8), rel=1e-9
        )

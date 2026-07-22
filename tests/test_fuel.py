"""Tests for the fuel map: the diesel BSFC curve and the learned wear penalty.

The dataset tests are skipped rather than failed when data/raw is absent -- the
repository does not commit data, and a fresh clone must still get a green run.
"""

from __future__ import annotations

import math

import pytest

from services.speed.fuel import (
    ENRICHMENT_K,
    FRICTION_K,
    FUEL_DENSITY_KG_PER_L,
    MIN_MODELLED_LOAD_FRACTION,
    OPTIMAL_LOAD_FRACTION,
    BurnEstimate,
    EngineSpec,
    FuelMap,
    diesel_bsfc_ratio,
)

# A representative Philippine short-haul passenger boat: ~11.5 m fiberglass
# monohull on a small high-speed diesel.
BANGKA = EngineSpec(rated_kw=90.0, rated_rpm=2800.0)


# --- The diesel part-load curve --------------------------------------------


def test_bsfc_is_minimised_at_the_stated_optimal_load():
    """The curve's minimum must actually sit where the constant claims it does."""
    ratios = {x / 100: diesel_bsfc_ratio(x / 100) for x in range(10, 101)}
    best_load = min(ratios, key=ratios.get)
    assert best_load == pytest.approx(OPTIMAL_LOAD_FRACTION, abs=0.02)
    assert diesel_bsfc_ratio(OPTIMAL_LOAD_FRACTION) == pytest.approx(1.0)


def test_part_load_penalty_stays_in_the_diesel_band_not_the_turbine_band():
    """This is the assertion the gas-turbine caveat rests on.

    If these bounds ever loosen towards the turbine's ~7x penalty at 10% load,
    the model has started overstating the savings from slowing down and the
    whole product claim inflates with it.
    """
    # Bounds are published part-load spreads for small high-speed marine
    # diesels, not the current implementation's output.
    assert 1.35 < diesel_bsfc_ratio(0.10) < 1.60
    assert 1.10 < diesel_bsfc_ratio(0.25) < 1.18
    assert 1.02 < diesel_bsfc_ratio(0.50) < 1.06
    assert 1.00 < diesel_bsfc_ratio(1.00) < 1.03


def test_curve_is_clamped_below_the_modelled_floor():
    """Below the floor the 1/x term would run away to infinity."""
    assert diesel_bsfc_ratio(0.001) == diesel_bsfc_ratio(MIN_MODELLED_LOAD_FRACTION)
    assert math.isfinite(diesel_bsfc_ratio(0.0))


def test_enrichment_coefficient_is_tied_to_the_stated_optimum():
    """The two coefficients are not independent; drifting one must break this.

    ENRICHMENT_K is derived from FRICTION_K so the curve turns around exactly at
    OPTIMAL_LOAD_FRACTION. If someone later hand-tunes either constant, the
    named optimum and the actual optimum silently diverge and every "run at 80%
    load" recommendation becomes wrong.
    """
    derived = FRICTION_K / (2 * OPTIMAL_LOAD_FRACTION**3)
    assert abs(ENRICHMENT_K - derived) < 1e-12

    # Curve rises on both sides of the optimum.
    assert diesel_bsfc_ratio(OPTIMAL_LOAD_FRACTION - 0.01) > 1.0
    assert diesel_bsfc_ratio(OPTIMAL_LOAD_FRACTION + 0.01) > 1.0


# --- FuelMap without a trained artifact ------------------------------------


def test_fuel_map_works_with_no_trained_model():
    """A fresh clone must be able to boot the API before anyone runs training."""
    fm = FuelMap(BANGKA)
    est = fm.estimate(45.0, egt_excess_ratio=1.05)
    assert not fm.has_wear_model
    assert est.wear_multiplier == 1.0
    assert est.wear_penalty_lph == 0.0
    assert est.confidence < 1.0
    assert any("not trained" in c for c in est.caveats)


def test_burn_is_dimensionally_correct():
    """Hand-check the arithmetic end to end, independent of the implementation."""
    fm = FuelMap(BANGKA)
    shaft_kw = BANGKA.rated_kw * OPTIMAL_LOAD_FRACTION  # exactly the sweet spot
    est = fm.estimate(shaft_kw)

    expected_kg_h = shaft_kw * BANGKA.best_bsfc_g_per_kwh / 1000.0
    expected_lph = expected_kg_h / FUEL_DENSITY_KG_PER_L

    assert est.bsfc_g_per_kwh == pytest.approx(BANGKA.best_bsfc_g_per_kwh)
    assert est.litres_per_hour == pytest.approx(expected_lph)
    # ~18 L/h for a 72 kW load. Sanity: these boats burn 15-25 L/h in service.
    assert 15.0 < est.litres_per_hour < 25.0


def test_idle_governor_sets_a_floor_under_burn():
    """A running diesel burns fuel at no load. Without this the model says a
    vessel crawling at one knot consumes almost nothing, and any optimiser
    reading that curve concludes the cheapest crossing is to barely move."""
    fm = FuelMap(BANGKA)
    assert fm.estimate(0.05).litres_per_hour == pytest.approx(BANGKA.idle_burn_lph)
    assert fm.estimate(0.0).litres_per_hour == pytest.approx(BANGKA.idle_burn_lph)
    # The floor must not distort the normal operating band.
    assert fm.estimate(72.0).litres_per_hour > BANGKA.idle_burn_lph * 5


def test_reported_bsfc_always_matches_reported_burn():
    """The two are derived from one number, so they cannot drift apart -- including
    where the idle floor binds and specific consumption goes very high."""
    fm = FuelMap(BANGKA)
    for kw in (0.5, 5.0, 45.0, 90.0):
        est = fm.estimate(kw)
        implied = est.bsfc_g_per_kwh * kw / 1000.0 / FUEL_DENSITY_KG_PER_L
        assert implied == pytest.approx(est.litres_per_hour)


def test_negative_idle_burn_rejected():
    with pytest.raises(ValueError):
        EngineSpec(rated_kw=90.0, rated_rpm=2800.0, idle_burn_lph=-1.0)


def test_burn_rises_monotonically_with_shaft_power():
    fm = FuelMap(BANGKA)
    burns = [fm.estimate(kw).litres_per_hour for kw in range(10, 91, 10)]
    assert burns == sorted(burns)


def test_overload_and_idle_are_flagged_not_silently_extrapolated():
    fm = FuelMap(BANGKA)
    over = fm.estimate(BANGKA.rated_kw * 1.2)
    assert over.confidence < 0.5
    assert any("exceeds" in c for c in over.caveats)

    idle = fm.estimate(BANGKA.rated_kw * 0.01)
    assert any("idle" in c for c in idle.caveats)


def test_negative_power_is_rejected():
    with pytest.raises(ValueError):
        FuelMap(BANGKA).estimate(-1.0)


def test_engine_spec_validates_its_inputs():
    with pytest.raises(ValueError):
        EngineSpec(rated_kw=0.0, rated_rpm=2800.0)
    with pytest.raises(ValueError):
        EngineSpec(rated_kw=90.0, rated_rpm=2800.0, best_bsfc_g_per_kwh=-1.0)


# --- The wear model ---------------------------------------------------------


class _StubWearModel:
    """Stands in for the trained regressor so wear handling is testable without
    data on disk.

    Linear in EGT excess, with an optional bias at the healthy point so the
    anchoring behaviour in `wear_multiplier` is observable.
    """

    def __init__(self, slope: float, bias: float = 0.0):
        self.slope = slope
        self.bias = bias

    def predict(self, X):
        return [1.0 + self.bias + self.slope * (row[1] - 1.0) for row in X]


def test_wear_multiplier_is_clamped_to_never_reward_a_worn_engine():
    """A boundary artifact must not hand the captain a fuel discount for wear."""
    fm = FuelMap(BANGKA, wear_model=_StubWearModel(slope=-1.0))
    assert fm.wear_multiplier(0.5, 1.05) == 1.0

    fm_high = FuelMap(BANGKA, wear_model=_StubWearModel(slope=50.0))
    assert fm_high.wear_multiplier(0.5, 1.30) == 1.5


def test_healthy_exhaust_temperature_means_exactly_no_penalty():
    """The anchoring property: model bias at the healthy point is divided out.

    Without this, a brand-new engine is billed for wear it does not have, and
    the savings the product reports drift by the size of the model's own error.
    """
    biased = FuelMap(BANGKA, wear_model=_StubWearModel(slope=1.0, bias=0.05))
    assert biased.wear_multiplier(0.5, 1.0) == pytest.approx(1.0)
    assert biased.estimate(45.0, egt_excess_ratio=1.0).wear_penalty_lph == pytest.approx(0.0)


def test_wear_penalty_is_reported_in_litres():
    """The Problem-1 to Problem-2 bridge: engine condition, priced per hour."""
    # slope 4/3 over 0.06 of EGT excess -> an 8% fuel penalty.
    fm = FuelMap(BANGKA, wear_model=_StubWearModel(slope=4.0 / 3.0))
    est = fm.estimate(72.0, egt_excess_ratio=1.06)

    assert est.wear_multiplier == pytest.approx(1.08)
    healthy = fm.estimate(72.0).litres_per_hour
    assert est.litres_per_hour == pytest.approx(healthy * 1.08)
    assert est.wear_penalty_lph == pytest.approx(healthy * 0.08)


def test_missing_exhaust_temperature_assumes_health_and_says_so():
    fm = FuelMap(BANGKA, wear_model=_StubWearModel(slope=4.0 / 3.0))
    est = fm.estimate(72.0)
    assert est.wear_multiplier == 1.0
    assert est.confidence < 1.0
    assert any("assumed healthy" in c for c in est.caveats)


# --- Physics -> ML handoff --------------------------------------------------


def test_resistance_output_feeds_the_fuel_map():
    """The two halves of the hybrid model must actually compose.

    This is the integration the whole architecture rests on: conditions in one
    end, litres per hour out the other, with no dimensional mismatch in between.
    """
    from services.speed.resistance import SeaState, VesselHull, required_shaft_power_kw

    hull = VesselHull(
        length_waterline_m=11.5, beam_m=2.8, draft_m=1.1, displacement_kg=8500.0
    )
    fm = FuelMap(BANGKA)

    calm = required_shaft_power_kw(hull, 10.0, SeaState(), heading_deg=0.0)
    rough = required_shaft_power_kw(
        hull,
        10.0,
        SeaState(wind_speed_kn=20.0, wind_direction_deg=0.0, wave_height_m=1.2),
        heading_deg=0.0,
    )

    calm_burn = fm.estimate(calm.total_kw).litres_per_hour
    rough_burn = fm.estimate(rough.total_kw).litres_per_hour

    assert rough.total_kw > calm.total_kw
    assert rough_burn > calm_burn
    assert 0 < calm_burn < 60


def test_slowing_down_saves_fuel_superlinearly():
    """The core product claim, asserted as a test rather than a slide.

    A 1-knot reduction must save materially more than 10% of burn, otherwise the
    advisory has nothing worth showing a captain.
    """
    from services.speed.resistance import SeaState, VesselHull, required_shaft_power_kw

    hull = VesselHull(
        length_waterline_m=11.5, beam_m=2.8, draft_m=1.1, displacement_kg=8500.0
    )
    fm = FuelMap(BANGKA)
    sea = SeaState()

    def burn(speed_kn: float) -> float:
        p = required_shaft_power_kw(hull, speed_kn, sea, heading_deg=0.0)
        return fm.estimate(p.total_kw).litres_per_hour

    saving = (burn(12.0) - burn(11.0)) / burn(12.0)
    assert saving > 0.10


# --- Dataset construction ---------------------------------------------------


@pytest.fixture(scope="module")
def cbm():
    dataset = pytest.importorskip("services.speed.dataset")
    try:
        return dataset.build_dataset()
    except FileNotFoundError:
        pytest.skip("uci-cbm not downloaded; run python -m data.download uci-cbm")


def test_dataset_is_the_expected_factorial_grid(cbm):
    """7 remaining lever positions x 51 x 26 wear states."""
    assert len(cbm.frame) == 7 * 51 * 26
    assert cbm.frame.degradation_state_id.nunique() == 51 * 26
    assert cbm.rows_dropped == 2 * 51 * 26


def test_unstable_idle_rows_are_excluded(cbm):
    assert not cbm.frame.ship_speed_kn.isin([3.0, 6.0]).any()


def test_source_plant_power_is_a_credible_gas_turbine(cbm):
    """Guards the N m / kN m unit correction. A 27 GW frigate would be a nuclear
    aircraft carrier a thousand times over."""
    assert 20_000 < cbm.max_shaft_kw < 35_000


def test_targets_are_dimensionless_and_bounded(cbm):
    """Nothing with units may leak from a gas turbine into a diesel.

    The 1e-4 tolerance is simulator round-off: a handful of wear states come out
    a few parts in 10^5 below their own healthy baseline. That is noise in the
    source data, not a worn engine saving fuel, and `FuelMap.wear_multiplier`
    clamps it away at inference.
    """
    assert cbm.target.min() == pytest.approx(1.0, abs=1e-4)
    assert 1.05 < cbm.target.max() < 1.20
    assert cbm.frame.egt_excess_ratio.min() == pytest.approx(1.0, abs=1e-4)


def test_wear_raises_both_fuel_burn_and_exhaust_temperature(cbm):
    """The physical premise behind using EGT as the observable wear proxy."""
    at_load = cbm.frame[cbm.frame.ship_speed_kn == 15.0]
    assert at_load.sfc_ratio.corr(at_load.egt_excess_ratio) > 0.95
    assert at_load.sfc_ratio.corr(at_load.compressor_decay) < -0.5


def test_features_exclude_unobservable_ground_truth(cbm):
    """The decay coefficients are simulator internals no vessel can measure.

    If they ever appear as features the model will score beautifully and be
    undeployable.
    """
    from services.speed.dataset import FEATURE_COLUMNS

    assert "compressor_decay" not in FEATURE_COLUMNS
    assert "turbine_decay" not in FEATURE_COLUMNS
    assert set(FEATURE_COLUMNS) == {"load_fraction", "egt_excess_ratio"}


def test_burn_estimate_is_immutable():
    est = FuelMap(BANGKA).estimate(50.0)
    assert isinstance(est, BurnEstimate)
    # Frozen dataclasses raise FrozenInstanceError, a subclass of AttributeError.
    with pytest.raises(AttributeError):
        est.litres_per_hour = 0.0

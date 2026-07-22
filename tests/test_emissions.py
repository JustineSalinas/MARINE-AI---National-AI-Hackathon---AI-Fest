"""Tests for the emissions layer (Problem 3).

Short, because the module is deliberately arithmetic. The tests that matter are
the ones guarding the *reporting* honesty: the factor must stay a constant of
chemistry, and a bad month must be allowed to report as a bad month.
"""

from __future__ import annotations

import pytest

from services.emissions import (
    DIESEL_CO2_KG_PER_L,
    DIESEL_CO2E_KG_PER_L,
    VoyageEmissions,
    co2_avoided_kg,
    co2_kg,
)


def test_factor_matches_carbon_mass_balance():
    """The factor is derivable, not tuned. Diesel ~0.845 kg/L, ~86.2% carbon,
    CO2/C mass ratio 44/12."""
    derived = 0.845 * 0.862 * (44.0 / 12.0)
    assert abs(DIESEL_CO2_KG_PER_L - derived) < 0.02


def test_co2e_is_slightly_above_co2():
    """Methane and N2O slip. Mixing the two silently is how reporting goes wrong."""
    assert DIESEL_CO2E_KG_PER_L > DIESEL_CO2_KG_PER_L
    assert DIESEL_CO2E_KG_PER_L / DIESEL_CO2_KG_PER_L < 1.05


def test_co2_is_linear_in_litres():
    assert co2_kg(0.0) == 0.0
    assert co2_kg(100.0) == pytest.approx(268.0, abs=1.0)
    assert co2_kg(50.0) == pytest.approx(co2_kg(100.0) / 2)


def test_co2e_flag_selects_the_other_factor():
    assert co2_kg(100.0, co2e=True) > co2_kg(100.0)


def test_negative_litres_rejected():
    with pytest.raises(ValueError):
        co2_kg(-1.0)


def test_avoided_is_positive_when_burning_less_than_baseline():
    assert co2_avoided_kg(120.0, 100.0) == pytest.approx(20.0 * DIESEL_CO2_KG_PER_L)


def test_a_bad_month_reports_as_a_bad_month():
    """Never floor at zero. A month that burned more than baseline is a real
    month, and a report that hides it is not auditable."""
    assert co2_avoided_kg(100.0, 130.0) < 0


def test_voyage_accumulates_a_rate_over_time():
    v = VoyageEmissions()
    v.accumulate(litres_per_hour=20.0, seconds=1800)  # half an hour
    assert v.fuel_used_l == pytest.approx(10.0)
    assert v.co2_kg == pytest.approx(co2_kg(10.0))


def test_voyage_avoided_uses_the_supplied_baseline():
    v = VoyageEmissions()
    v.accumulate(litres_per_hour=18.0, seconds=3600, baseline_lph=24.0)
    assert v.fuel_used_l == pytest.approx(18.0)
    assert v.baseline_fuel_l == pytest.approx(24.0)
    assert v.co2_avoided_kg == pytest.approx(6.0 * DIESEL_CO2_KG_PER_L)


def test_without_a_baseline_nothing_is_claimed_as_avoided():
    """No counterfactual means no avoided-emissions claim. Zero, not a guess."""
    v = VoyageEmissions()
    v.accumulate(litres_per_hour=20.0, seconds=3600)
    assert v.co2_avoided_kg == pytest.approx(0.0)


def test_accumulation_is_additive():
    v = VoyageEmissions()
    for _ in range(4):
        v.accumulate(litres_per_hour=20.0, seconds=900)
    assert v.fuel_used_l == pytest.approx(20.0)
    assert v.seconds == pytest.approx(3600)


def test_negative_duration_rejected():
    with pytest.raises(ValueError):
        VoyageEmissions().accumulate(litres_per_hour=20.0, seconds=-1)


def test_pesos_are_available_because_operators_budget_in_them():
    v = VoyageEmissions()
    v.accumulate(litres_per_hour=20.0, seconds=3600)
    assert v.php_spent(70.0) == pytest.approx(1400.0)

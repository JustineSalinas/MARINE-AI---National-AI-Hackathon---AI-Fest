"""Problem 3: the auditable emissions layer.

The technical profile gives emissions equal billing with fuel and maintenance,
and promises the operator "a monthly CO2-avoided report — exportable evidence
for LGU / MARINA compliance and ESG-linked green financing", with **zero extra
sensors beyond what Problems 1 and 2 already require**.

That last clause is the whole design. There is no emissions model here and there
should not be one: combustion CO2 is fixed stoichiometry, not something to
predict. Burn a litre of diesel and you emit a known mass of CO2. So this module
is arithmetic on the output of `services/speed/fuel.py`, and the entire accuracy
of the emissions report is inherited from the accuracy of the fuel model.

Stating that plainly is better than dressing it up. An "AI emissions model" that
multiplied litres by a constant and called itself a prediction would be the
least defensible thing in the repository.

What is *not* trivial, and is where the honesty lives, is the counterfactual:
"avoided" emissions require a baseline of what the vessel would have burned
without the advice. See `co2_avoided_kg`.
"""

from __future__ import annotations

from dataclasses import dataclass

DIESEL_CO2_KG_PER_L = 2.68
"""Kilograms of CO2 per litre of diesel burned.

Well-to-tank is excluded; this is tank-to-wake combustion only, which is the
convention for operational vessel reporting and the only part the vessel
controls. The figure follows from carbon mass balance: diesel is ~86.2% carbon
by mass at ~0.845 kg/L, and each carbon atom leaves as CO2 (44/12 mass ratio),
giving 0.845 x 0.862 x 3.667 = 2.67. Published factors (IMO Fourth GHG Study,
UK DEFRA conversion factors) sit at 2.68 for marine gas oil.

It is a constant of chemistry, not a tuned parameter. It does not need
calibration and must not be adjusted to make a report look better."""

DIESEL_CO2E_KG_PER_L = 2.73
"""CO2-equivalent, including the methane and nitrous oxide slip that combustion
also produces. Roughly 2% above the CO2-only figure for marine diesel.

Reported separately because compliance regimes differ on which they require, and
silently mixing the two is how emissions reporting goes wrong."""


def co2_kg(litres: float, *, co2e: bool = False) -> float:
    """CO2 emitted by burning `litres` of diesel."""
    if litres < 0:
        raise ValueError("litres cannot be negative")
    factor = DIESEL_CO2E_KG_PER_L if co2e else DIESEL_CO2_KG_PER_L
    return litres * factor


def co2_avoided_kg(baseline_litres: float, actual_litres: float, *, co2e: bool = False) -> float:
    """CO2 not emitted, against a baseline of what the vessel would have burned.

    **The baseline is the entire claim.** A number like "we avoided 400 kg of
    CO2 this month" means nothing without saying what it is measured against,
    and it is trivially inflatable by choosing a wasteful counterfactual. This
    system uses exactly one baseline and names it on every report: **the
    vessel's own recorded burn on the same route before Marine-AI was fitted.**

    Not a fleet average, not a published efficiency benchmark, not a modelled
    "typical" vessel. The boat versus its own past, on its own route. That is
    the only comparison an auditor can check and the only one an operator will
    believe.

    Negative results are returned as negative. A month where the vessel burned
    more than baseline -- heavier loads, worse weather, a fouled hull -- is a
    real month, and a report that silently floors at zero is not auditable.
    """
    return co2_kg(baseline_litres, co2e=co2e) - co2_kg(actual_litres, co2e=co2e)


@dataclass
class VoyageEmissions:
    """Running total for one voyage. Feeds `BridgeState.voyage_co2_*`.

    Accumulates from burn-rate samples, because that is what the system has: a
    litres-per-hour estimate updated every advisory cycle. Integrating a rate
    over time is the only available route to a total, and its error is the fuel
    model's error, carried forward honestly rather than re-estimated.
    """

    fuel_used_l: float = 0.0
    baseline_fuel_l: float = 0.0
    seconds: float = 0.0

    def accumulate(
        self, litres_per_hour: float, seconds: float, *, baseline_lph: float | None = None
    ) -> None:
        """Add one interval at a given burn rate."""
        if seconds < 0:
            raise ValueError("seconds cannot be negative")
        hours = seconds / 3600.0
        self.fuel_used_l += max(0.0, litres_per_hour) * hours
        baseline = baseline_lph if baseline_lph is not None else litres_per_hour
        self.baseline_fuel_l += max(0.0, baseline) * hours
        self.seconds += seconds

    @property
    def co2_kg(self) -> float:
        return co2_kg(self.fuel_used_l)

    @property
    def co2_avoided_kg(self) -> float:
        return co2_avoided_kg(self.baseline_fuel_l, self.fuel_used_l)

    def php_spent(self, php_per_litre: float) -> float:
        """Operators budget in pesos. Litres are an engineering unit."""
        return self.fuel_used_l * php_per_litre


__all__ = [
    "DIESEL_CO2E_KG_PER_L",
    "DIESEL_CO2_KG_PER_L",
    "VoyageEmissions",
    "co2_avoided_kg",
    "co2_kg",
]

"""The throttle optimizer: conditions in, a recommended RPM out.

This is the piece that turns two models into a product. `resistance.py` says how
much shaft power a speed costs in these conditions; `fuel.py` says how many
litres that power burns. Neither answers the question the captain actually has,
which is *what should I set the throttle to*. This module does.

The optimisation is deliberately a brute-force sweep rather than a solver. The
search space is one-dimensional and about thirty candidates wide, the cost
function is a millisecond, and a sweep cannot land in a local minimum or fail to
converge on stage. A gradient method here would be sophistication bought at the
price of the demo.

The governing constraint is arrival time, not speed. Slowing down always saves
fuel -- that is just the cubic law, and a recommendation to go slower with no
regard for the schedule would be worthless and immediately ignored. The
optimizer holds the ETA the operator committed to and finds the cheapest way to
meet it, which is why `eta_impact_minutes` on the contract is near zero by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from services.speed.fuel import BurnEstimate, EngineSpec, FuelMap
from services.speed.resistance import (
    PowerBreakdown,
    SeaState,
    VesselHull,
    required_shaft_power_kw,
)

MAX_SPEED_SEARCH_KN = 25.0
"""Upper bound of every speed search. No boat in this class approaches it; it
exists so bisection has a bracket that is guaranteed to contain the answer."""

SPEED_TOLERANCE_KN = 0.005
BISECTION_MAX_ITERATIONS = 60


def speed_for_power_kn(
    hull: VesselHull,
    available_shaft_kw: float,
    sea: SeaState,
    heading_deg: float,
    *,
    added_load_kg: float = 0.0,
) -> float:
    """Fastest speed over ground this much shaft power can sustain, in these conditions.

    The inverse of `required_shaft_power_kw`, solved by bisection because the
    forward function has no closed-form inverse -- it composes a cubic, a
    quadratic hump penalty, an apparent-wind term with its own speed dependence,
    and a current correction.

    **This function is why the simulator is honest.** Without it, displayed speed
    is a function of the throttle alone, and a headwind changes the readouts
    while the boat sails on unaffected. With it, the same throttle in a 20-knot
    head sea produces a slower boat, which is the entire premise of the product.

    Bisection is safe here because required power is monotonically increasing in
    speed: every term rises with speed, so there is exactly one crossing.
    """
    if available_shaft_kw <= 0:
        return 0.0

    lo, hi = 0.0, MAX_SPEED_SEARCH_KN
    if required_shaft_power_kw(
        hull, hi, sea, heading_deg, added_load_kg=added_load_kg
    ).total_kw <= available_shaft_kw:
        return hi  # more power than the search bracket can absorb

    for _ in range(BISECTION_MAX_ITERATIONS):
        mid = 0.5 * (lo + hi)
        needed = required_shaft_power_kw(
            hull, mid, sea, heading_deg, added_load_kg=added_load_kg
        ).total_kw
        if needed > available_shaft_kw:
            hi = mid
        else:
            lo = mid
        if hi - lo < SPEED_TOLERANCE_KN:
            break
    return lo


def rpm_for_power(spec: EngineSpec, shaft_kw: float) -> float:
    """Shaft power -> engine RPM, via the propeller law.

    For a fixed-pitch propeller, absorbed power varies as the cube of shaft
    speed, so `rpm = rated_rpm * (kw / rated_kw)^(1/3)`.

    This is an approximation and is stated as one. It assumes a fixed-pitch prop
    turning in undisturbed water, ignores gearbox ratio changes and slip, and
    will be a few percent off on a fouled or ventilating propeller. Like the
    Admiralty coefficient in `resistance.py`, it is the kind of relationship that
    should be replaced by a per-vessel fit against observed (RPM, power) pairs
    once a boat has logged a few voyages.

    It exists because the captain sets an RPM, not a kilowatt. Every number this
    system computes is in power or litres; exactly one of them has to be
    translated into the unit on the physical gauge, and this is that translation.
    """
    if shaft_kw <= 0:
        return 0.0
    return spec.rated_rpm * (shaft_kw / spec.rated_kw) ** (1.0 / 3.0)


def power_for_rpm(spec: EngineSpec, rpm: float) -> float:
    """Inverse of `rpm_for_power`. Used to turn a throttle position into power."""
    if rpm <= 0:
        return 0.0
    return spec.rated_kw * (rpm / spec.rated_rpm) ** 3.0


@dataclass(frozen=True)
class SpeedOption:
    """One candidate speed, fully costed. The sweep is a list of these."""

    speed_kn: float
    rpm: float
    power: PowerBreakdown
    burn: BurnEstimate

    @property
    def litres_per_hour(self) -> float:
        return self.burn.litres_per_hour

    def litres_for_distance(self, distance_nm: float) -> float:
        """Total burn to cover a distance. The number that actually matters.

        Litres per hour alone is misleading: idling burns very little per hour
        and never arrives. Fuel per voyage is the quantity the operator pays.
        """
        if self.speed_kn <= 0:
            return float("inf")
        return self.burn.litres_per_hour * (distance_nm / self.speed_kn)


def evaluate_speed(
    hull: VesselHull,
    spec: EngineSpec,
    fuel_map: FuelMap,
    speed_kn: float,
    sea: SeaState,
    heading_deg: float,
    *,
    added_load_kg: float = 0.0,
    egt_excess_ratio: float | None = None,
) -> SpeedOption:
    """Cost a single candidate speed through both halves of the model."""
    power = required_shaft_power_kw(
        hull, speed_kn, sea, heading_deg, added_load_kg=added_load_kg
    )
    burn = fuel_map.estimate(power.total_kw, egt_excess_ratio=egt_excess_ratio)
    return SpeedOption(
        speed_kn=speed_kn,
        rpm=rpm_for_power(spec, power.total_kw),
        power=power,
        burn=burn,
    )


def performance_curve(
    hull: VesselHull,
    spec: EngineSpec,
    fuel_map: FuelMap,
    sea: SeaState,
    heading_deg: float,
    *,
    added_load_kg: float = 0.0,
    egt_excess_ratio: float | None = None,
    min_speed_kn: float = 1.0,
    max_speed_kn: float = 14.0,
    step_kn: float = 0.5,
) -> list[SpeedOption]:
    """The whole speed/burn curve for these conditions.

    Returned to the browser so a 60fps render loop can interpolate locally
    instead of calling the API every frame -- and, more importantly, instead of
    reimplementing the physics in JavaScript. There is one fuel model in this
    system and it is written in Python. The display consumes its output; it never
    computes its own.
    """
    options: list[SpeedOption] = []
    speed = min_speed_kn
    while speed <= max_speed_kn + 1e-9:
        options.append(
            evaluate_speed(
                hull,
                spec,
                fuel_map,
                speed,
                sea,
                heading_deg,
                added_load_kg=added_load_kg,
                egt_excess_ratio=egt_excess_ratio,
            )
        )
        speed += step_kn
    return options


@dataclass(frozen=True)
class ThrottleAdvice:
    """The optimizer's full answer, before it is flattened into the contract.

    Richer than `SpeedRecommendation` on purpose: the simulator and the shore
    dashboard want the breakdown, while the bridge display wants one sentence
    and one number.
    """

    recommended: SpeedOption
    current: SpeedOption | None
    savings_lph: float
    eta_impact_minutes: float
    feasible: bool
    """False when the engine cannot make the required speed in these conditions.
    The advice is then the best available, and the ETA cannot be met."""

    notes: tuple[str, ...] = ()


def optimise_throttle(
    hull: VesselHull,
    spec: EngineSpec,
    fuel_map: FuelMap,
    sea: SeaState,
    heading_deg: float,
    *,
    distance_remaining_nm: float,
    minutes_available: float | None = None,
    current_rpm: float | None = None,
    added_load_kg: float = 0.0,
    egt_excess_ratio: float | None = None,
) -> ThrottleAdvice:
    """Cheapest throttle that still meets the schedule.

    `minutes_available` is the ETA constraint. When it is None the optimizer has
    no schedule to honour and returns the most fuel-efficient speed per nautical
    mile -- correct for a vessel with no deadline, and rarely the real case.
    """
    notes: list[str] = []

    max_option = evaluate_speed(
        hull,
        spec,
        fuel_map,
        speed_for_power_kn(
            hull, spec.rated_kw, sea, heading_deg, added_load_kg=added_load_kg
        ),
        sea,
        heading_deg,
        added_load_kg=added_load_kg,
        egt_excess_ratio=egt_excess_ratio,
    )
    top_speed = max_option.speed_kn

    curve = performance_curve(
        hull,
        spec,
        fuel_map,
        sea,
        heading_deg,
        added_load_kg=added_load_kg,
        egt_excess_ratio=egt_excess_ratio,
        max_speed_kn=max(1.0, top_speed),
    )
    # Never recommend a speed the engine cannot hold in these conditions.
    curve = [o for o in curve if o.speed_kn <= top_speed + 1e-9] or [max_option]

    feasible = True
    if minutes_available is not None and minutes_available > 0:
        required_kn = distance_remaining_nm / (minutes_available / 60.0)
        candidates = [o for o in curve if o.speed_kn >= required_kn - 1e-9]
        if candidates:
            # Cheapest option that still arrives on time. Because burn per mile
            # rises with speed, this is nearly always the slowest of them -- but
            # it is selected on cost, not assumed.
            best = min(candidates, key=lambda o: o.litres_for_distance(distance_remaining_nm))
        else:
            feasible = False
            best = max_option
            notes.append(
                f"{required_kn:.1f} kn needed to arrive on time; "
                f"{top_speed:.1f} kn is all this engine can hold in these conditions"
            )
    else:
        best = min(curve, key=lambda o: o.litres_for_distance(distance_remaining_nm))
        notes.append("no arrival time set; optimising for fuel per nautical mile alone")

    current: SpeedOption | None = None
    if current_rpm is not None and current_rpm > 0:
        current_speed = speed_for_power_kn(
            hull,
            power_for_rpm(spec, current_rpm),
            sea,
            heading_deg,
            added_load_kg=added_load_kg,
        )
        current = evaluate_speed(
            hull,
            spec,
            fuel_map,
            current_speed,
            sea,
            heading_deg,
            added_load_kg=added_load_kg,
            egt_excess_ratio=egt_excess_ratio,
        )

    savings = 0.0
    eta_impact = 0.0
    if current is not None:
        savings = current.litres_per_hour - best.litres_per_hour
        if current.speed_kn > 0 and best.speed_kn > 0:
            now_min = 60.0 * distance_remaining_nm / current.speed_kn
            then_min = 60.0 * distance_remaining_nm / best.speed_kn
            eta_impact = then_min - now_min
        if savings < 0:
            # The captain is already more efficient than the recommendation.
            # Say so rather than hiding it; the contract requires the honest sign.
            notes.append("current throttle is already cheaper than the schedule requires")

    return ThrottleAdvice(
        recommended=best,
        current=current,
        savings_lph=savings,
        eta_impact_minutes=eta_impact,
        feasible=feasible,
        notes=tuple(notes),
    )


# --- Rendering the advice as a sentence -------------------------------------
#
# Claude writes this sentence in production (`advisory_source="claude"`). The
# template below is the deterministic fallback, and it ships as the default
# because the display must never block on an API call. Per PRODUCT.md the
# phrasing is never imperative: "1650 RPM saves 2.1 L/h", not "reduce to 1650".


def advisory_sentences(
    advice: ThrottleAdvice, *, php_per_litre: float | None = None
) -> tuple[str, str]:
    """(English, Filipino) plain-language advisory. Never imperative."""
    rpm = round(advice.recommended.rpm)

    if not advice.feasible:
        return (
            f"{rpm} RPM is all the engine holds in this weather. Arrival will be late.",
            f"{rpm} RPM na lang ang kaya ng makina sa panahong ito. Mahuhuli ang dating.",
        )

    if advice.current is None:
        lph = advice.recommended.litres_per_hour
        return (
            f"{rpm} RPM meets the schedule at {lph:.1f} L/h.",
            f"{rpm} RPM ay sapat sa iskedyul sa {lph:.1f} L/h.",
        )

    saved = advice.savings_lph
    if abs(saved) < 0.05:
        return (
            f"{rpm} RPM. Current throttle is already about right.",
            f"{rpm} RPM. Tama na ang kasalukuyang throttle.",
        )
    if saved < 0:
        return (
            f"Current throttle is {abs(saved):.1f} L/h cheaper than {rpm} RPM, but arrives later.",
            f"Ang kasalukuyang throttle ay {abs(saved):.1f} L/h na mas mura kaysa {rpm} RPM, "
            "pero mas mahuhuli ang dating.",
        )

    money = ""
    money_fil = ""
    if php_per_litre:
        money = f" — about PHP {saved * php_per_litre:,.0f} per hour"
        money_fil = f" — humigit-kumulang PHP {saved * php_per_litre:,.0f} kada oras"
    return (
        f"{rpm} RPM saves {saved:.1f} L/h{money}.",
        f"{rpm} RPM ay nakakatipid ng {saved:.1f} L/h{money_fil}.",
    )


def as_recommendation(
    advice: ThrottleAdvice,
    *,
    vessel_id: str,
    php_per_litre: float | None = None,
    generated_at: datetime | None = None,
):
    """Flatten `ThrottleAdvice` into the wire contract the display consumes."""
    from packages.contracts.speed import SpeedRecommendation

    en, fil = advisory_sentences(advice, php_per_litre=php_per_litre)
    return SpeedRecommendation(
        vessel_id=vessel_id,
        generated_at=generated_at or datetime.now(UTC),
        recommended_rpm=advice.recommended.rpm,
        recommended_speed_kn=advice.recommended.power.speed_through_water_kn,
        current_rpm=None if advice.current is None else rpm_of(advice.current),
        current_burn_lph=None if advice.current is None else advice.current.litres_per_hour,
        predicted_burn_lph=advice.recommended.litres_per_hour,
        savings_lph=advice.savings_lph,
        savings_php_per_hour=(
            None if php_per_litre is None else advice.savings_lph * php_per_litre
        ),
        model_confidence=advice.recommended.burn.confidence,
        eta_impact_minutes=advice.eta_impact_minutes,
        advisory_en=en,
        advisory_fil=fil,
        advisory_source="template",
    )


def rpm_of(option: SpeedOption) -> float:
    return option.rpm

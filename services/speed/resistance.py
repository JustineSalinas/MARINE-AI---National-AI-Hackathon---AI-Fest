"""Hull resistance: sea conditions and load -> shaft power required.

This is the physics half of the hybrid fuel model. See docs/DATA.md for why it
is physics and not machine learning: the only public dataset pairing shaft
torque, RPM and ground-truth fuel flow (UCI CBM) holds wind, current, wave
height and displacement completely constant. Those effects cannot be learned
from it, so they are computed from established naval architecture instead.

The division of labour:

    conditions + load  --[this module, physics]-->  required shaft power
    shaft power + RPM  --[XGBoost on UCI CBM]   -->  fuel flow

Every coefficient below is named, sourced, and adjustable. That is the point: a
mis-specified coefficient is inspectable and calibratable against a vessel's own
fuel-flow meter, whereas a neural network that silently ignored wind because
wind never varied in training is not.

Method: Holtrop-Mennen is the industry standard but needs hull-form coefficients
(prismatic coefficient, wetted surface, transom area) no operator can supply.
We use an Admiralty-coefficient calm-water baseline calibrated per vessel from
its own telemetry, plus separate added-resistance terms for waves and wind. This
is the standard reduced-order approach for ship performance monitoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants
RHO_SEAWATER = 1025.0
"""kg/m^3 at 25 C, 35 PSU. Philippine coastal waters."""

RHO_AIR = 1.20
"""kg/m^3 at 30 C. Tropical, warmer and thinner than the 1.225 standard."""

G = 9.81
KNOTS_TO_MS = 0.514444


@dataclass(frozen=True)
class VesselHull:
    """Hull parameters an operator can actually supply or a surveyor can measure.

    Deliberately excludes hull-form coefficients that require lines plans. If a
    parameter cannot be obtained during a one-week retrofit install, it is not
    in this model.
    """

    length_waterline_m: float
    beam_m: float
    draft_m: float
    displacement_kg: float

    admiralty_coefficient: float = 70.0
    """Calm-water efficiency constant. P = disp^(2/3) * V^3 / C_adm.

    Textbook values of 300-600 are for large merchant displacement ships. Small
    craft are far less efficient per tonne, and the default here (70) is scaled
    to put an 8.5 t, 11.5 m hull at roughly 30 kW at 8 knots, which matches the
    engines these boats actually carry.

    THIS IS THE PRIMARY CALIBRATION HANDLE. Fit it per vessel from observed
    (speed, shaft power) pairs via `calibrate_admiralty`. Do not ship the
    default into production; it is a starting point, not a measurement."""

    semi_displacement_k: float = 1.0
    """Wave-making penalty past hull speed. See `_hump_factor`.

    A reduced-order fit, not a first-principles constant. Calibrated alongside
    the Admiralty coefficient from the vessel's own high-speed runs."""

    superstructure_area_m2: float | None = None
    """Transverse projected area above waterline, for wind resistance. Estimated
    from beam and freeboard when not supplied."""

    propulsive_efficiency: float = 0.55
    """Shaft power to effective towing power. Covers propeller open-water
    efficiency, hull and relative-rotative efficiency, and shaft losses.
    0.50-0.65 is typical for small craft with fixed-pitch propellers."""

    def __post_init__(self) -> None:
        if self.length_waterline_m <= 0 or self.displacement_kg <= 0:
            raise ValueError("length_waterline_m and displacement_kg must be positive")
        if not 0 < self.propulsive_efficiency <= 1:
            raise ValueError("propulsive_efficiency must be in (0, 1]")

    @property
    def effective_superstructure_area_m2(self) -> float:
        if self.superstructure_area_m2 is not None:
            return self.superstructure_area_m2
        # Freeboard approximated as 0.6 x draft for a small passenger hull; the
        # transverse area is then beam x (freeboard + cabin height ~1.8 m).
        return self.beam_m * (0.6 * self.draft_m + 1.8)


@dataclass(frozen=True)
class SeaState:
    """Conditions at the vessel. Angles are compass degrees.

    Sign convention, chosen to match how the sensors report and stated because
    getting it backwards silently inverts every recommendation:
      - wind_direction_deg: the direction the wind blows FROM (meteorological)
      - current_direction_deg: the direction the current flows TOWARD (oceanographic)
    """

    wind_speed_kn: float = 0.0
    wind_direction_deg: float = 0.0
    current_speed_kn: float = 0.0
    current_direction_deg: float = 0.0
    wave_height_m: float = 0.0
    wave_direction_deg: float | None = None
    """Defaults to the wind direction when absent -- wind-driven seas dominate
    on short coastal routes."""


def _relative_angle_deg(heading_deg: float, other_deg: float) -> float:
    """Angle between vessel heading and a direction vector, 0-180 degrees."""
    return abs((other_deg - heading_deg + 180.0) % 360.0 - 180.0)


HULL_SPEED_SLR = 1.34
"""Speed-length ratio (knots / sqrt(waterline feet)) at which a displacement
hull's own bow and stern waves reach its length. Past this the hull must climb
its own bow wave, and resistance rises much faster than the cubic law predicts."""

METRES_TO_FEET = 3.28084


def speed_length_ratio(hull: VesselHull, speed_kn: float) -> float:
    return speed_kn / math.sqrt(hull.length_waterline_m * METRES_TO_FEET)


def _hump_factor(hull: VesselHull, speed_kn: float) -> float:
    """Wave-making penalty for driving a semi-displacement hull past hull speed.

    The plain Admiralty cubic is derived for displacement ships operating below
    hull speed. These boats do not: an 11.5 m waterline has a hull speed near 8
    knots, and the routes are run at 10-12. Applying the bare cubic there
    under-predicts required power by a factor of three or more, which would make
    the whole product understate its own savings.

    Modelled as a quadratic penalty in the excess speed-length ratio:

        factor = 1 + k * max(0, SLR - 1.34)^2

    This matters more than it looks. It is why slowing from 12 to 11 knots saves
    roughly 32% of shaft power rather than the 23% the cubic alone would give --
    the single largest fuel lever the system has, and it would be invisible
    without this term.

    Full planing (SLR > ~3) is deliberately not modelled. These hulls and engines
    do not go there, and a curve fitted outside its regime is worse than an
    honest range limit.
    """
    excess = speed_length_ratio(hull, speed_kn) - HULL_SPEED_SLR
    if excess <= 0:
        return 1.0
    return 1.0 + hull.semi_displacement_k * excess**2


def admiralty_power_kw(hull: VesselHull, speed_kn: float, *, added_load_kg: float = 0.0) -> float:
    """Bare Admiralty cubic, without the semi-displacement correction.

    Exposed separately so the cubic law and the hump penalty can be tested and
    calibrated independently.
    """
    if speed_kn <= 0:
        return 0.0
    displacement_tonnes = (hull.displacement_kg + max(0.0, added_load_kg)) / 1000.0
    return (displacement_tonnes ** (2.0 / 3.0)) * (speed_kn**3) / hull.admiralty_coefficient


def calm_water_power_kw(hull: VesselHull, speed_kn: float, *, added_load_kg: float = 0.0) -> float:
    """Shaft power to hold `speed_kn` through still water at a given load.

    Admiralty cubic plus the semi-displacement wave-making penalty. The
    combination is why speed optimization pays: power rises with the cube of
    speed, and faster still once the hull is pushed past its own wave.
    """
    if speed_kn <= 0:
        return 0.0
    return admiralty_power_kw(hull, speed_kn, added_load_kg=added_load_kg) * _hump_factor(
        hull, speed_kn
    )


def wind_resistance_kw(
    hull: VesselHull, speed_kn: float, sea: SeaState, heading_deg: float
) -> float:
    """Added shaft power to push through apparent wind. Negative in a following wind.

    Uses apparent wind (vessel motion plus true wind), a drag coefficient that
    falls with off-bow angle, and the standard 0.5 * rho * A * Cd * V^2 form.
    """
    if hull.effective_superstructure_area_m2 <= 0:
        return 0.0

    boat_ms = speed_kn * KNOTS_TO_MS
    wind_ms = sea.wind_speed_kn * KNOTS_TO_MS
    angle = math.radians(_relative_angle_deg(heading_deg, sea.wind_direction_deg))

    # Apparent wind: headwind component adds to the vessel's own motion.
    head_component = wind_ms * math.cos(angle)
    cross_component = wind_ms * math.sin(angle)
    apparent_ms = math.hypot(boat_ms + head_component, cross_component)

    # Cd ~1.0 head-on, falling to ~0.4 abeam and ~0.6 astern for a boxy
    # superstructure. Cosine interpolation is the usual reduced-order fit.
    cd = 0.4 + 0.6 * math.cos(angle) if math.cos(angle) > 0 else 0.4 + 0.2 * abs(math.cos(angle))

    drag_n = 0.5 * RHO_AIR * hull.effective_superstructure_area_m2 * cd * apparent_ms**2
    if head_component < 0:  # following wind pushes the vessel along
        drag_n = -abs(drag_n) * 0.5  # thrust recovery is inefficient; halve it

    effective_kw = drag_n * boat_ms / 1000.0
    return effective_kw / hull.propulsive_efficiency


def wave_resistance_kw(
    hull: VesselHull, speed_kn: float, sea: SeaState, heading_deg: float
) -> float:
    """Added shaft power from waves. Always positive -- waves never help.

    Added resistance scales with the square of significant wave height, which is
    why a modest sea state matters so much: 1.5 m seas cost roughly nine times
    what 0.5 m seas cost.
    """
    if sea.wave_height_m <= 0 or speed_kn <= 0:
        return 0.0

    wave_dir = sea.wave_direction_deg if sea.wave_direction_deg is not None else sea.wind_direction_deg
    angle = math.radians(_relative_angle_deg(heading_deg, wave_dir))

    # Head seas are worst; following seas cost about a fifth as much.
    heading_factor = 0.2 + 0.8 * max(0.0, math.cos(angle))

    # R_aw ~ rho * g * H^2 * B^2 / L, the standard scaling for added resistance
    # in waves. The 0.12 coefficient is the reduced-order fit for small craft.
    resistance_n = (
        0.12
        * RHO_SEAWATER
        * G
        * (sea.wave_height_m**2)
        * (hull.beam_m**2)
        / hull.length_waterline_m
    ) * heading_factor

    effective_kw = resistance_n * (speed_kn * KNOTS_TO_MS) / 1000.0
    return effective_kw / hull.propulsive_efficiency


def speed_through_water_kn(
    speed_over_ground_kn: float, sea: SeaState, heading_deg: float
) -> float:
    """Ground speed corrected for current set.

    This is the single most important environmental correction, and the one
    captains most often misjudge. A 2-knot foul current means a vessel making 10
    knots over the ground is driving its hull at 12 knots through the water --
    and paying 12-knot fuel for 10-knot progress.
    """
    if sea.current_speed_kn <= 0:
        return speed_over_ground_kn

    # Current direction is where it flows TOWARD, so a current on the bow opposes.
    angle = math.radians(_relative_angle_deg(heading_deg, sea.current_direction_deg))
    along_track = sea.current_speed_kn * math.cos(angle)
    return max(0.0, speed_over_ground_kn - along_track)


@dataclass(frozen=True)
class PowerBreakdown:
    """Shaft power required, itemised. The breakdown is shown to the captain so
    a recommendation to slow down is explainable ("you are punching a 1.4 m head
    sea") rather than arbitrary."""

    total_kw: float
    calm_water_kw: float
    wind_kw: float
    wave_kw: float
    speed_through_water_kn: float

    @property
    def environmental_penalty_pct(self) -> float:
        """Share of shaft power spent purely on wind and waves."""
        if self.total_kw <= 0:
            return 0.0
        return 100.0 * (self.wind_kw + self.wave_kw) / self.total_kw


def required_shaft_power_kw(
    hull: VesselHull,
    speed_over_ground_kn: float,
    sea: SeaState,
    heading_deg: float,
    *,
    added_load_kg: float = 0.0,
) -> PowerBreakdown:
    """Total shaft power to make `speed_over_ground_kn` in these conditions.

    This is the function the throttle optimizer sweeps. Feed its `total_kw` and a
    candidate RPM to the XGBoost fuel map to get litres per hour.
    """
    stw = speed_through_water_kn(speed_over_ground_kn, sea, heading_deg)

    calm = calm_water_power_kw(hull, stw, added_load_kg=added_load_kg)
    wind = wind_resistance_kw(hull, stw, sea, heading_deg)
    wave = wave_resistance_kw(hull, stw, sea, heading_deg)

    # Wind can be negative (following), but total shaft power cannot be.
    total = max(0.0, calm + wind + wave)

    return PowerBreakdown(
        total_kw=total,
        calm_water_kw=calm,
        wind_kw=wind,
        wave_kw=wave,
        speed_through_water_kn=stw,
    )


def calibrate_admiralty(
    hull: VesselHull, observations: list[tuple[float, float]], *, added_load_kg: float = 0.0
) -> float:
    """Fit the Admiralty coefficient from this vessel's own (speed_kn, shaft_kw) data.

    Run on calm-weather voyages only, so wind and wave terms stay near zero and
    the coefficient absorbs hull form rather than sea state. This is what turns a
    generic default into a model of one specific boat -- the profile's
    "calibrates the fuel model to this specific boat", implemented.

    Returns the median of per-observation fits: robust to the handful of bad
    points every real dataset contains, unlike a least-squares fit.
    """
    displacement_tonnes = (hull.displacement_kg + max(0.0, added_load_kg)) / 1000.0
    # Divide the hump penalty back out before fitting. Without this, observations
    # taken above hull speed would push the coefficient low and the model would
    # then over-predict power everywhere below it.
    fits = [
        (displacement_tonnes ** (2.0 / 3.0)) * (speed**3) * _hump_factor(hull, speed) / power
        for speed, power in observations
        if speed > 0 and power > 0
    ]
    if not fits:
        raise ValueError("no usable (speed, power) observations to calibrate from")

    fits.sort()
    middle = len(fits) // 2
    return fits[middle] if len(fits) % 2 else 0.5 * (fits[middle - 1] + fits[middle])

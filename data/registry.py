"""Dataset registry — the single place a data source is declared.

Every entry carries its licence and source URL. `data/download.py` refuses to
fetch anything not declared here. The submission rules require licensed or
public datasets only; making the licence a required field is how that stays
true after the fourth late-night dataset addition.

`docs/DATA.md` is the human-facing version of this file and must be kept in
sync by hand when an entry changes.

Nothing downloaded here is committed to the repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    url: str
    licence: str
    citation: str
    purpose: str
    archive: bool = False
    """True if the download is a zip that must be extracted."""

    caveats: list[str] = field(default_factory=list)
    """Honest limitations. Rendered into docs/DATA.md and expected to appear in
    the pitch deck. A caveat recorded here and nowhere else is a caveat a judge
    finds before you mention it."""


REGISTRY: dict[str, Dataset] = {
    "uci-cbm": Dataset(
        key="uci-cbm",
        name="Condition Based Maintenance of Naval Propulsion Plants",
        url=(
            "https://archive.ics.uci.edu/static/public/316/"
            "condition+based+maintenance+of+naval+propulsion+plants.zip"
        ),
        licence="CC BY 4.0 (UCI Machine Learning Repository)",
        citation=(
            "Coraddu, A., Oneto, L., Ghio, A., Savio, S., Anguita, D., & Figari, M. (2014). "
            "Machine learning approaches for improving condition-based maintenance of naval "
            "propulsion plants. UCI Machine Learning Repository."
        ),
        purpose=(
            "Trains the fuel-consumption model: given lever position, ship speed, shaft "
            "torque and shaft RPM, predict fuel flow."
        ),
        archive=True,
        caveats=[
            "This is a GAS TURBINE frigate propulsion plant, not a diesel engine. It is used "
            "as a documented proxy because it is the only public dataset pairing shaft "
            "torque, RPM, ship speed and ground-truth fuel flow at this resolution.",
            "The transfer assumption is that the SHAPE of the load-to-burn relationship "
            "(burn rises super-linearly with shaft power, and the efficient band sits below "
            "maximum rated RPM) holds across both prime movers. The absolute litres-per-hour "
            "values do not transfer and are rescaled to the target vessel's rated power.",
            "STRUCTURE VERIFIED 2026-07-22: 11,934 rows are a complete factorial grid of "
            "9 lever positions x 51 compressor-decay states x 26 turbine-decay states. "
            "There are only 9 distinct ship speeds, and speed is fully determined by lever "
            "position. Ambient inlet temperature (T1) and pressure (P1) are constant and "
            "carry no information.",
            "CONSEQUENCE: this dataset cannot support learning the effect of wind, current, "
            "wave height or passenger load on fuel burn, because none of those variables "
            "vary in it. It supports exactly one thing: the engine's fuel map — shaft torque "
            "and RPM, modulated by degradation state, to fuel flow. Environmental and load "
            "effects are therefore computed by an explicit hull-resistance model rather than "
            "pretended to be learned. See docs/DATA.md.",
            "Simulator-generated, not measured at sea.",
        ],
    ),
    "nasa-cmapss": Dataset(
        key="nasa-cmapss",
        name="NASA C-MAPSS Turbofan Engine Degradation Simulation",
        url=(
            "https://phm-datasets.s3.amazonaws.com/NASA/"
            "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
        ),
        licence="U.S. Government work, public domain (NASA Open Data)",
        citation=(
            "Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation "
            "modeling for aircraft engine run-to-failure simulation. IEEE PHM 2008."
        ),
        purpose=(
            "Pretrains the Phase 1 anomaly detector on run-to-failure degradation patterns "
            "before fine-tuning on vessel engine channels."
        ),
        archive=True,
        caveats=[
            "Turbofan, not marine diesel. Used for the SHAPE of gradual multi-sensor "
            "degradation, not for any component-level claim about a boat engine.",
            "Provides the pretraining signal that makes cold-start anomaly detection "
            "possible at all; it is explicitly not a source of RUL predictions here.",
        ],
    ),
    # FEMTO / PRONOSTIA is named in the technical profile but is NOT used here.
    # Two reasons, recorded so the omission is a decision rather than an oversight:
    #
    #   1. Its NASA PCoE download key could not be resolved on 2026-07-22. The
    #      data.nasa.gov entry 404s and the phm-datasets S3 bucket denies listing.
    #      We do not cite a source we could not download.
    #
    #   2. More fundamentally, it would not have helped. FEMTO is bench-rig bearing
    #      vibration sampled at 25.6 kHz. Bearing defect signatures live in the
    #      kilohertz band. The retrofit IMU logs at ~1 Hz alongside the other
    #      electro-mechanical channels, which is three to four orders of magnitude
    #      too slow to resolve them. Pretraining on FEMTO and applying it to a 1 Hz
    #      IMU stream would imply a diagnostic capability the sensor cannot deliver.
    #
    # What the IMU is genuinely good for at 1 Hz -- sustained vibration energy
    # trending upward, shock events, changes in mounting rigidity -- is learned
    # from the vessel's own baseline instead. See docs/DEVIATIONS.md.
    "natural-earth-coastline": Dataset(
        key="natural-earth-coastline",
        name="Natural Earth 10m Physical Coastline",
        url="https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip",
        licence="Public domain (Natural Earth)",
        citation="Natural Earth. Free vector and raster map data, naturalearthdata.com.",
        purpose=(
            "Chart geometry for the bridge display. Rendered as a drawn nautical chart "
            "rather than a tile basemap: no API key, no quota, nothing to fail on stage."
        ),
        archive=True,
    ),
}


def get(key: str) -> Dataset:
    try:
        return REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown dataset {key!r}. Declared datasets: {known}") from None

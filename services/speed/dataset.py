"""UCI CBM -> the modelling frame for the fuel map.

Loading is separated from training so the cleaning decisions below are testable
on their own, and so anyone auditing the model can read what was thrown away
before reading what was fitted.

Three things about this dataset drove every choice here, all verified by direct
inspection (see docs/DATA.md):

1. It is a complete factorial grid, 9 lever positions x 51 compressor-decay x
   26 turbine-decay states. A random train/test split would put a row's
   near-identical neighbours on both sides and report a meaningless score. We
   split by whole degradation state instead -- see `degradation_state_id`.
2. Shaft RPM is functionally constant within a lever position (standard
   deviation 0.01 rpm at most positions). RPM carries no information the load
   fraction does not already carry, so it is not a feature. The profile calls
   RPM a fuel predictor; in this data it is a relabelling of load.
3. The plant is a 27 MW gas turbine, not a 30 kW diesel. Nothing dimensional
   survives the transfer. Every column produced here is a ratio.

Units note: `Features.txt` labels shaft torque as kN m. It is N m -- the stated
units would put the plant at 27 GW. The correction is applied in `SHAFT_TORQUE`
handling below and is the reason `MAX_SHAFT_KW` comes out at a credible 27 MW,
which is an LM2500, which is what a frigate of this description carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/uci-cbm/UCI CBM Dataset/data.txt")

RAW_COLUMNS = [
    "lever_position",
    "ship_speed_kn",
    "shaft_torque_nm",
    "shaft_rpm",
    "gas_generator_rpm",
    "propeller_torque_stbd",
    "propeller_torque_port",
    "turbine_exit_temp_c",  # T48 -- the wear signal
    "compressor_inlet_temp_c",  # T1 -- constant, dropped
    "compressor_outlet_temp_c",
    "turbine_exit_pressure_bar",
    "compressor_inlet_pressure_bar",  # P1 -- constant, dropped
    "compressor_outlet_pressure_bar",
    "exhaust_pressure_bar",
    "turbine_injection_control_pct",
    "fuel_flow_kg_s",
    "compressor_decay",  # kMc, 1.000 healthy -> 0.950 worst
    "turbine_decay",  # kMt, 1.000 healthy -> 0.975 worst
]

EXCLUDED_LEVER_SPEEDS_KN = (3.0, 6.0)
"""The two lowest lever positions are dropped as not-at-steady-state.

Not a convenience cut. At these positions the simulator's fuel ratio ranges from
0.79 to 7.22 relative to the healthy baseline and moves non-monotonically with
degradation -- a worn engine sometimes burning *less* than a new one. Every
other lever position is cleanly monotonic and bounded within 1.00-1.15. The
controller has not settled at idle, and the rows are governor transients rather
than the steady-state operating points the dataset documents itself as holding.

Keeping them would let a 7x outlier at 3 knots dominate the loss and corrupt the
fit everywhere else. They are excluded loudly here rather than silently
winsorised, per the ingest policy in docs/DATA.md: outliers are flagged, never
smoothed.
"""

HEALTHY_COMPRESSOR_DECAY = 1.0
HEALTHY_TURBINE_DECAY = 1.0


@dataclass(frozen=True)
class FuelMapDataset:
    """The modelling frame, plus what is needed to interpret it honestly."""

    frame: pd.DataFrame
    max_shaft_kw: float
    """Rated power of the source plant. Used only to form the load fraction --
    it is deliberately divided out and never propagated to the target vessel."""

    rows_dropped: int
    excluded_speeds_kn: tuple[float, ...]

    @property
    def features(self) -> pd.DataFrame:
        return self.frame[list(FEATURE_COLUMNS)]

    @property
    def target(self) -> pd.Series:
        return self.frame[TARGET_COLUMN]

    @property
    def groups(self) -> pd.Series:
        return self.frame["degradation_state_id"]


FEATURE_COLUMNS = ("load_fraction", "egt_excess_ratio")
"""Both are dimensionless and both are observable on a retrofit install.

`load_fraction` comes from the hull resistance model at inference time, not from
a sensor -- that is the physics-to-ML handoff. `egt_excess_ratio` comes from an
exhaust-manifold thermocouple, which is in the contract as
`ElectroMechanicalFrame.exhaust_gas_temp_c`.

Deliberately absent: shaft RPM (collinear with load, see module docstring), and
the decay coefficients kMc/kMt themselves. The decay coefficients are the
strongest possible features and are exactly what a real vessel cannot supply --
they are simulator ground truth for internal component wear. Training on them
would produce a model that scores beautifully and cannot be deployed."""

TARGET_COLUMN = "sfc_ratio"
"""Specific fuel consumption as a multiple of this plant's own healthy SFC at
the same load. The dimensionless form is what makes gas-turbine data usable at
all: the *level* of a turbine's fuel curve does not transfer to a diesel, but
the proportional penalty for running a worn engine at a given load does."""


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read data.txt as-is, with names and the torque unit correction applied."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it with: python -m data.download uci-cbm"
        )
    return pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS)


def shaft_power_kw(torque_nm: pd.Series, rpm: pd.Series) -> pd.Series:
    """P = tau * omega. The only place the torque unit correction is applied."""
    return torque_nm * (2.0 * np.pi * rpm / 60.0) / 1000.0


def build_dataset(path: Path | str = DATA_PATH) -> FuelMapDataset:
    """Clean, derive features, and normalise against the healthy baseline.

    The healthy baseline is the nine rows where both decay coefficients are 1.0
    -- one per lever position. Every ratio in the output frame is measured
    against the row at the *same* lever position, so load effects are divided
    out and what remains is purely the cost of wear.
    """
    raw = load_raw(path)
    n_raw = len(raw)

    df = raw[~raw.ship_speed_kn.isin(EXCLUDED_LEVER_SPEEDS_KN)].copy()

    df["shaft_kw"] = shaft_power_kw(df.shaft_torque_nm, df.shaft_rpm)
    df["sfc_g_per_kwh"] = df.fuel_flow_kg_s * 3600.0 * 1000.0 / df.shaft_kw

    healthy = df[
        (df.compressor_decay == HEALTHY_COMPRESSOR_DECAY)
        & (df.turbine_decay == HEALTHY_TURBINE_DECAY)
    ]
    if len(healthy) != df.ship_speed_kn.nunique():
        raise ValueError(
            f"expected one healthy row per lever position, found {len(healthy)} "
            f"for {df.ship_speed_kn.nunique()} positions"
        )
    baseline = healthy.set_index("ship_speed_kn")

    df["sfc_ratio"] = df.sfc_g_per_kwh / df.ship_speed_kn.map(baseline.sfc_g_per_kwh)
    df["egt_excess_ratio"] = df.turbine_exit_temp_c / df.ship_speed_kn.map(
        baseline.turbine_exit_temp_c
    )

    max_shaft_kw = float(df.shaft_kw.max())
    df["load_fraction"] = df.shaft_kw / max_shaft_kw

    # One group per (kMc, kMt) pair. Holding whole wear states out of training is
    # the only split that answers the question the product asks: does this
    # generalise to an engine worn in a way we have not seen?
    df["degradation_state_id"] = (
        df.compressor_decay.round(4).astype(str) + "_" + df.turbine_decay.round(4).astype(str)
    )

    return FuelMapDataset(
        frame=df.reset_index(drop=True),
        max_shaft_kw=max_shaft_kw,
        rows_dropped=n_raw - len(df),
        excluded_speeds_kn=EXCLUDED_LEVER_SPEEDS_KN,
    )

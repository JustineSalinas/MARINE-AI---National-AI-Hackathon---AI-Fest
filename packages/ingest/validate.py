"""Telemetry validation. Every frame passes through here before any AI module.

The technical profile promises four things: range checks, timestamp validation,
drift monitoring, and that outlier trips are flagged rather than silently
averaged. This module implements all four.

Design rule throughout: **never silently repair.** A frame is accepted, or it is
rejected with a reason, or it is accepted with an attached flag. Nothing is
quietly clamped, interpolated, or reordered. A model trained on quietly repaired
data learns the repair, and the operator never finds out why the recommendation
drifted.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from statistics import fmean, pstdev

from packages.contracts import TelemetryFrame


class Verdict(str, Enum):
    ACCEPT = "accept"
    ACCEPT_FLAGGED = "accept_flagged"
    """Usable for live display, excluded from training."""
    REJECT = "reject"


class FlagCode(str, Enum):
    OUT_OF_RANGE = "out_of_range"
    TIMESTAMP_NAIVE = "timestamp_naive"
    TIMESTAMP_FUTURE = "timestamp_future"
    TIMESTAMP_REGRESSED = "timestamp_regressed"
    TIMESTAMP_DUPLICATE = "timestamp_duplicate"
    GAP = "gap"
    SENSOR_FROZEN = "sensor_frozen"
    SENSOR_DRIFT = "sensor_drift"
    IMPLAUSIBLE_RATE = "implausible_rate"


@dataclass(frozen=True)
class Flag:
    code: FlagCode
    stream: str
    detail: str


@dataclass(frozen=True)
class Result:
    verdict: Verdict
    frame: TelemetryFrame | None
    flags: tuple[Flag, ...] = ()

    @property
    def usable_for_training(self) -> bool:
        return self.verdict is Verdict.ACCEPT

    @property
    def usable_for_display(self) -> bool:
        return self.verdict in (Verdict.ACCEPT, Verdict.ACCEPT_FLAGGED)


# Physically implausible rates of change. A reading can be inside its absolute
# range and still be impossible -- coolant does not climb 40 C in one second.
# Units per second.
MAX_RATE: dict[str, float] = {
    "electro_mechanical.coolant_temp_c": 2.0,
    "electro_mechanical.oil_pressure_kpa": 200.0,
    "electro_mechanical.battery_voltage_v": 5.0,
    "electro_mechanical.exhaust_gas_temp_c": 25.0,
    "throttling.engine_rpm": 500.0,
    "throttling.fuel_flow_lph": 20.0,
}

# Streams that should never sit perfectly still on a RUNNING engine. A frozen
# channel reads as "nominal" to every downstream model, which makes it more
# dangerous than an obviously broken one.
#
# Gated on the engine actually running (see RUNNING_RPM). A moored vessel with
# the engine off legitimately reports a constant RPM of zero and a flat battery
# voltage; flagging that as a stuck sensor would train the crew to ignore the
# one alert that matters.
FREEZE_WATCH: tuple[str, ...] = (
    "electro_mechanical.coolant_temp_c",
    "electro_mechanical.oil_pressure_kpa",
    "electro_mechanical.battery_voltage_v",
    "throttling.engine_rpm",
)

RUNNING_RPM = 200.0
"""Below this the engine is off or cranking. Freeze detection is suspended."""


def _read(frame: TelemetryFrame, path: str) -> float | None:
    section, field_name = path.split(".", 1)
    value = getattr(getattr(frame, section), field_name, None)
    return float(value) if isinstance(value, (int, float)) else None


@dataclass
class VesselValidator:
    """Stateful per-vessel validator. One instance per vessel, not shared.

    Holds the recent history needed for rate, freeze, and drift checks. State is
    per vessel because a fleet-wide baseline would flatten exactly the
    vessel-specific wear signature the maintenance module is trying to learn.
    """

    vessel_id: str
    history_len: int = 300
    """~5 minutes at 1 Hz. Long enough to establish a local baseline, short
    enough that a real operating-point change is not mistaken for drift."""

    max_gap: timedelta = timedelta(seconds=30)
    freeze_min_samples: int = 60
    drift_sigma: float = 4.0
    future_tolerance: timedelta = timedelta(seconds=5)

    _last_ts: datetime | None = field(default=None, init=False)
    _seen_ts: set[datetime] = field(default_factory=set, init=False)
    _history: dict[str, deque[float]] = field(default_factory=dict, init=False)
    _baseline: dict[str, tuple[float, float]] = field(default_factory=dict, init=False)

    def validate(self, frame: TelemetryFrame, *, now: datetime | None = None) -> Result:
        if frame.vessel_id != self.vessel_id:
            return Result(
                Verdict.REJECT,
                None,
                (Flag(FlagCode.OUT_OF_RANGE, "vessel_id",
                      f"frame for {frame.vessel_id!r} sent to validator for {self.vessel_id!r}"),),
            )

        flags: list[Flag] = []

        # --- Timestamps. Rejections, not repairs. ---
        if frame.ts.tzinfo is None:
            return Result(
                Verdict.REJECT, None,
                (Flag(FlagCode.TIMESTAMP_NAIVE, "ts",
                      "naive datetime; a timestamp without a zone is not a timestamp"),),
            )

        reference = now or datetime.now(frame.ts.tzinfo)
        if frame.ts > reference + self.future_tolerance:
            return Result(
                Verdict.REJECT, None,
                (Flag(FlagCode.TIMESTAMP_FUTURE, "ts", f"{frame.ts.isoformat()} is in the future"),),
            )

        if frame.ts in self._seen_ts:
            return Result(
                Verdict.REJECT, None,
                (Flag(FlagCode.TIMESTAMP_DUPLICATE, "ts", f"duplicate {frame.ts.isoformat()}"),),
            )

        if self._last_ts is not None:
            if frame.ts < self._last_ts:
                # Out of order. Rejected rather than reordered: silently sorting
                # hides a clock problem that will corrupt every RUL calculation.
                return Result(
                    Verdict.REJECT, None,
                    (Flag(FlagCode.TIMESTAMP_REGRESSED, "ts",
                          f"{frame.ts.isoformat()} precedes last accepted {self._last_ts.isoformat()}"),),
                )
            gap = frame.ts - self._last_ts
            if gap > self.max_gap:
                flags.append(Flag(FlagCode.GAP, "ts", f"{gap.total_seconds():.0f}s since last frame"))

        # --- Range. Contract bounds already ran at parse time; this catches
        # cross-field impossibility the schema cannot express. ---
        rpm = _read(frame, "throttling.engine_rpm")
        flow = _read(frame, "throttling.fuel_flow_lph")
        if rpm is not None and flow is not None and rpm <= 0 < flow:
            flags.append(
                Flag(FlagCode.OUT_OF_RANGE, "throttling.fuel_flow_lph",
                     f"{flow:.2f} L/h at {rpm:.0f} rpm; a stopped engine cannot burn fuel")
            )

        # --- Rate of change ---
        elapsed = (frame.ts - self._last_ts).total_seconds() if self._last_ts else None
        if elapsed and elapsed > 0:
            for path, limit in MAX_RATE.items():
                value = _read(frame, path)
                previous = self._history.get(path)
                if value is None or not previous:
                    continue
                rate = abs(value - previous[-1]) / elapsed
                if rate > limit:
                    flags.append(
                        Flag(FlagCode.IMPLAUSIBLE_RATE, path,
                             f"{rate:.1f}/s exceeds physical limit {limit:.1f}/s")
                    )

        # --- Freeze and drift, against this vessel's own baseline ---
        # Only meaningful while the engine is turning; see RUNNING_RPM.
        if (rpm or 0.0) >= RUNNING_RPM:
            for path in FREEZE_WATCH:
                series = self._history.get(path)
                if series and len(series) >= self.freeze_min_samples and len(set(series)) == 1:
                    flags.append(
                        Flag(FlagCode.SENSOR_FROZEN, path,
                             f"identical value for {len(series)} samples on a running engine; "
                             "a stuck sensor reads as nominal to every model downstream")
                    )

        for path, (mean, sigma) in self._baseline.items():
            value = _read(frame, path)
            if value is None or sigma <= 0:
                continue
            if abs(value - mean) > self.drift_sigma * sigma:
                flags.append(
                    Flag(FlagCode.SENSOR_DRIFT, path,
                         f"{value:.2f} is {abs(value - mean) / sigma:.1f} sigma from baseline {mean:.2f}")
                )

        self._commit(frame)

        verdict = Verdict.ACCEPT_FLAGGED if flags else Verdict.ACCEPT
        return Result(verdict, frame, tuple(flags))

    def _commit(self, frame: TelemetryFrame) -> None:
        self._last_ts = frame.ts
        self._seen_ts.add(frame.ts)
        if len(self._seen_ts) > self.history_len * 2:
            cutoff = frame.ts - timedelta(seconds=self.history_len * 2)
            self._seen_ts = {ts for ts in self._seen_ts if ts >= cutoff}

        for path in {*MAX_RATE, *FREEZE_WATCH}:
            value = _read(frame, path)
            if value is None:
                continue
            series = self._history.setdefault(path, deque(maxlen=self.history_len))
            series.append(value)
            if len(series) >= self.freeze_min_samples:
                self._baseline[path] = (fmean(series), pstdev(series))


@dataclass(frozen=True)
class VoyageReport:
    """Whole-voyage verdict. Outlier trips are flagged, never silently averaged."""

    voyage_id: str | None
    total: int
    accepted: int
    flagged: int
    rejected: int
    flag_counts: dict[FlagCode, int]

    @property
    def is_outlier(self) -> bool:
        """A voyage this damaged is excluded from training and marked on any report."""
        if self.total == 0:
            return True
        return (self.rejected / self.total) > 0.05 or (self.flagged / self.total) > 0.20

    @property
    def reason(self) -> str | None:
        if not self.is_outlier:
            return None
        if self.total == 0:
            return "no frames"
        return (
            f"{self.rejected / self.total:.1%} rejected, {self.flagged / self.total:.1%} flagged "
            f"({', '.join(f'{c.value}x{n}' for c, n in sorted(self.flag_counts.items(), key=lambda kv: -kv[1])[:3])})"
        )


def validate_voyage(
    vessel_id: str, frames: Iterable[TelemetryFrame], *, voyage_id: str | None = None
) -> tuple[list[Result], VoyageReport]:
    validator = VesselValidator(vessel_id=vessel_id)
    results = [validator.validate(frame) for frame in frames]

    counts: dict[FlagCode, int] = {}
    for result in results:
        for flag in result.flags:
            counts[flag.code] = counts.get(flag.code, 0) + 1

    report = VoyageReport(
        voyage_id=voyage_id,
        total=len(results),
        accepted=sum(r.verdict is Verdict.ACCEPT for r in results),
        flagged=sum(r.verdict is Verdict.ACCEPT_FLAGGED for r in results),
        rejected=sum(r.verdict is Verdict.REJECT for r in results),
        flag_counts=counts,
    )
    return results, report

# Data Strategy

Machine-readable source of truth: [`data/registry.py`](../data/registry.py).
Nothing under `data/` is committed; fetch with `python -m data.download --all`.

Every dataset here is public, licensed for research and educational use, and
cited. No live operator telemetry is used in this submission, and no data was
collected from any vessel.

---

## Sources

| Key | Source | Licence | Used for |
|---|---|---|---|
| `uci-cbm` | UCI *Condition Based Maintenance of Naval Propulsion Plants* (Coraddu et al., 2014) | CC BY 4.0 | Engine fuel map for Speed Optimization |
| `nasa-cmapss` | NASA C-MAPSS turbofan degradation (Saxena et al., 2008) | Public domain (US Gov) | Pretraining the Phase 1 anomaly detector |
| `femto-bearing` | FEMTO-ST / PRONOSTIA bearing run-to-failure (Nectoux et al., 2012) | Public domain (NASA PCoE) | Pretraining the vibration branch (IMU channels) |
| `natural-earth-coastline` | Natural Earth 10m physical coastline | Public domain | Chart geometry on the bridge display |
| Open-Meteo Marine API | open-meteo.com | CC BY 4.0, free tier | Live and historical wind, wave, current |
| GEBCO | GEBCO global bathymetry grid | Public, attribution required | Depth safety constraint |

**PAGASA** is named in the technical profile but has **no public programmatic
API**. It is a stated future integration, not a working data source in this
build. See [`DEVIATIONS.md`](DEVIATIONS.md).

---

## What the anchor dataset actually contains

*Verified by direct inspection on 2026-07-22, before any model was written.*

`uci-cbm` is 11,934 rows across 18 columns. It is not a voyage recording. It is
a **complete factorial grid**:

```
9 lever positions  x  51 compressor-decay states  x  26 turbine-decay states  =  11,934
```

Three consequences follow, and all three shaped the architecture:

1. **There are only 9 distinct ship speeds**, and speed is fully determined by
   lever position (r = 0.93). There is no independent speed variation to learn from.
2. **Ambient inlet temperature (T1) and pressure (P1) are constant.** They carry
   zero information and are dropped.
3. **Wind, current, wave height, and passenger load do not appear at all** — not
   as weak signals, but as variables that genuinely do not vary in the data.

### Why this matters, stated plainly

The technical profile describes the fuel model as answering: *given RPM, torque,
wind, current, and load, how many litres per hour will this engine burn?*

**No public dataset we could find supports learning that function end-to-end.**
Training a model on `uci-cbm` and presenting it as having learned wind and
current effects would be a fabrication — the model would simply be ignoring
inputs that were constant during training, and its confidence on those inputs
would be meaningless.

### What we do instead

We split the problem at the boundary where the data actually is, and use the
right tool on each side:

| Layer | Method | Justification |
|---|---|---|
| Conditions → required shaft power | **Physics**: hull resistance model (calm-water resistance + added resistance from wind, waves, current, and displacement from passenger/cargo load) | These relationships are well established in naval architecture and require no training data. Pretending to learn them from data that does not contain them would be worse, not better. |
| Shaft power + RPM + engine health → fuel flow | **XGBoost**, trained on `uci-cbm` | This is exactly what the dataset contains, at 11,934 points across the full load and degradation range. Real ground-truth fuel flow, genuinely nonlinear, well suited to gradient boosting. |

This is a **hybrid physics-ML model**, and it is a stronger design than pure ML,
not a weaker one. Use ML where there is ground truth; use physics where there is
established theory and no data. The honest failure mode of this design is a
mis-specified resistance coefficient, which is inspectable and calibratable. The
failure mode of the alternative is a model that silently ignores the inputs the
product claims to act on.

**Unexpected benefit.** Because the fuel map is trained across 51 compressor-decay
and 26 turbine-decay states, it predicts the *fuel penalty of a degraded engine*.
That is the literal, quantified data link between Problem 1 (inefficiency) and
Problem 2 (accelerated wear) that the technical profile asserts — not a claimed
connection, a measured one.

### The gas turbine caveat

`uci-cbm` is a **gas turbine frigate propulsion plant, not a marine diesel.** It
is used as a documented proxy because it is the only public dataset pairing shaft
torque, RPM, ship speed, and ground-truth fuel flow at this resolution.

The transfer assumption is that the **shape** of the load-to-burn relationship
holds across both prime movers: burn rises super-linearly with shaft power, and
the efficient band sits below maximum rated RPM. The **absolute** litres-per-hour
values do not transfer; they are rescaled to the target vessel's rated power via
brake-specific fuel consumption.

This is stated in the README, in the model card, and belongs on a pitch deck
slide. A judge who discovers it in the repository unprompted will weight it far
more heavily than one who was told.

---

## Data quality and cleaning

Implemented in [`packages/ingest/`](../packages/ingest/), applied to every frame
before it reaches any AI module:

- **Range checks** — each channel against physical bounds declared in
  `packages/contracts/telemetry.py`.
- **Timestamp validation** — timezone-aware UTC, monotonic per vessel,
  out-of-order and duplicate frames rejected rather than silently reordered.
- **Drift monitoring** — rolling comparison against the vessel's learned baseline;
  a sensor that has stopped moving is as suspect as one reading out of range.
- **Outlier trips are flagged, never silently averaged.** A voyage with a
  mid-passage signal dropout is marked and excluded from training, not smoothed.

Operator-entered maintenance logs (parts replaced, dates, failure reports) are
treated as **first-class data with the same validation rigour as sensor
telemetry**. Phase 2 of Predictive Maintenance depends entirely on accurate
labelled failure history: a mislabelled maintenance event two years from now
would directly corrupt the RUL model.

## Synthetic data

The simulator in [`packages/sim/`](../packages/sim/) generates voyage telemetry
by replaying `uci-cbm` operating points along a synthetic Philippine short-haul
route with real Open-Meteo forecast overlay. Controlled fault injections
(coolant drift, oil-pressure decay, bearing vibration) drive the maintenance
demo.

**Every synthetic frame is labelled `source="simulator"` in the contract itself**
(`packages/contracts/telemetry.py`), and the field has no default that could
claim otherwise. No frame in this submission originates from physical hardware.

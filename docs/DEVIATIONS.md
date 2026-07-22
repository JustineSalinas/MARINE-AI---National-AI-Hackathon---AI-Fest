# Deviations from the Technical Profile

The submitted technical profile (*Marine-AI by SOLMATE*, National AI Hackathon
2026, Blue Economy / Clean Energy track) is the specification this repository is
judged against. This page lists every place the build departs from it, why, and
what the departure costs.

It exists because a judge will find these anyway. Finding them here, with the
reasoning attached, is a very different experience from finding them by reading
the source and discovering the paper overstated something.

Nothing below was changed to make the build easier. Each item was forced by a
verified fact about a dataset, an API, or a sensor — the check that produced it
is named in every case, and every check is reproducible.

**Status: reviewed against the submitted PDF on 2026-07-22.**

| # | Profile says | Build does | Class |
|---|---|---|---|
| 1 | Fuel burn learned end-to-end by XGBoost from RPM, torque, wind, current, load | Hybrid: physics for conditions→power, XGBoost for wear→fuel penalty | **Architecture** |
| 2 | "Public marine-diesel engine performance datasets" | UCI CBM, a **gas turbine**, as a documented proxy | **Data** |
| 3 | FEMTO/PRONOSTIA pretrains the anomaly detector | Dropped; NASA C-MAPSS only | **Data** |
| 4 | RPM is a core fuel-model input | Not a feature — collinear with load | **Modelling** |
| 5 | TensorFlow Lite for edge inference | ONNX Runtime | Stack |
| 6 | TimescaleDB for time-series storage | Supabase Postgres | Stack |
| 7 | PAGASA and OpenWeather marine forecasts | Open-Meteo Marine | Stack |
| 8 | Sonar hardware for depth | Charted bathymetry intended; **not yet integrated** | Sensing |
| 9 | AIS receiver for traffic avoidance | Omitted | Sensing |
| 10 | (not in profile) | Sentinel-2 satellite basemap; Google/Bing/Esri rejected on licence | Data |
| 11 | Docker for edge deployment | Container at deploy; not used locally | Minor |

---

## 1. The fuel model is a hybrid, not end-to-end XGBoost

**Profile (§3.1):** *"A XGBoost regression model trained to answer one question:
given RPM, torque, wind, current, and load, how many liters per hour will this
engine burn?"*

**Build:** two layers. Conditions and load → required shaft power is **physics**
(`services/speed/resistance.py`). Shaft power and engine condition → litres per
hour is a published diesel BSFC curve plus an **XGBoost** wear model
(`services/speed/fuel.py`, `services/speed/train.py`).

**Why.** No public dataset supports learning that function end-to-end. Verified
by direct inspection before any model was written: the anchor dataset is a
complete factorial grid of 9 lever positions × 51 compressor-decay × 26
turbine-decay states. Wind, current, wave height and passenger load **do not
vary in it at all** — not as weak signals, as constants.

A model trained on that data and presented as having learned wind and current
would be ignoring the inputs the product acts on, while reporting confidence on
them. The failure would be invisible. A mis-specified physics coefficient is
inspectable, and calibratable against the vessel's own fuel-flow meter.

**What it costs.** The environmental terms are theory-driven rather than
fitted, so they are only as good as their coefficients. Every coefficient is
named, sourced, and exposed for per-vessel calibration
(`calibrate_admiralty`). This is the honest trade and we think it is the better
one.

## 2. The anchor dataset is a gas turbine

**Profile (§4):** *"public marine-diesel engine performance datasets for the
fuel/RPM curve."*

**Build:** UCI *Condition Based Maintenance of Naval Propulsion Plants* — a
27 MW frigate **gas turbine**. We could not find a public marine-diesel dataset
pairing shaft torque, RPM and ground-truth fuel flow at usable resolution.

**Why this is survivable, and exactly how far the proxy is trusted.** Only the
**dimensionless wear penalty** is taken from it. Not the fuel level, and — the
important part — **not the part-load curve either.**

Measured, not assumed: the turbine burns ~**7×** its best-point specific fuel
consumption at 10% load. A marine diesel burns ~**1.5×**. Borrowing the
turbine's part-load shape would have overstated the savings from slowing down by
roughly **five times, in our own favour**. So healthy burn comes from published
diesel part-load data, and the dataset answers only the question it uniquely
can: *at the same shaft load, how much more does a worn engine burn than a
healthy one?* — a question whose **form** is prime-mover-agnostic, sampled here
across 1,326 wear states, validated against wear states held out of training.

There is a regression test pinning the diesel curve inside published part-load
bands specifically so it can never drift back toward the turbine's.

## 3. FEMTO/PRONOSTIA was dropped

**Profile (§3.1, §4):** named alongside NASA C-MAPSS as Phase 1 pretraining.

**Build:** C-MAPSS only.

**Why.** FEMTO is bench-rig bearing vibration sampled at **25.6 kHz**. The
retrofit IMU in the profile's own parameter list logs at roughly **1 Hz** —
three to four orders of magnitude too slow to resolve bearing defect
frequencies. Pretraining on it would imply a diagnostic capability the specified
sensor physically cannot deliver.

Dropping it is a smaller error than keeping it. **`README.md` and `docs/DATA.md`
must not list FEMTO as a source.**

## 4. Shaft RPM is not a fuel-model feature

**Profile (§3.2):** *"Engine RPM sensor — core input to the fuel-consumption
model."*

**Build:** not a feature. In the training data, RPM varies by **0.01 rpm** within
a lever position — it is a relabelling of load, not an independent predictor.
Including it would add a feature that looks informative and carries nothing.

RPM is still ingested, still displayed, and still the unit the recommendation is
delivered in (`SpeedRecommendation.recommended_rpm`) — a captain sets a throttle,
not a kilowatt. It is the *model input* that changed, not the interface.

## 5. ONNX Runtime instead of TensorFlow Lite

**Profile (§3.1 Tools):** TensorFlow Lite for edge inference.

**Build:** ONNX Runtime, and it is **in use, not planned**. The trained model
is exported by `services/speed/train.py` to `models/fuel_degradation.onnx`, and
that file is what the API loads — `services/speed/fuel.py` imports no xgboost.

TFLite targets TensorFlow graphs; exporting gradient-boosted trees through it
means a conversion no one should be debugging during a sprint. ONNX is the
native export path for both XGBoost and scikit-learn estimators, and runs on
the same class of edge hardware.

The decision earned its keep sooner than expected. Serving through xgboost
requires xgboost, scikit-learn, scipy and pandas — **358 MB of runtime for a
363 kB tree ensemble**. ONNX Runtime plus numpy is **99 MB**, which is what
lets the advisory API run as a serverless function inside a 500 MB limit
instead of needing a dedicated host. The same argument applies unchanged to a
control unit aboard a vessel.

The export is gated: the trainer refuses to write an ONNX file whose
predictions differ from the trained model by more than `1e-4`, so the deployed
model is provably the one the metrics describe. Observed drift is `1.5e-6`,
which is float32 rounding.

## 6. Supabase Postgres instead of TimescaleDB

**Profile (§3.1 Tools):** TimescaleDB.

**Build:** Supabase Postgres. Supabase *is* Postgres, so the SQL and the schema
are portable to Timescale later; what is lost is hypertable partitioning and
native continuous aggregates, neither of which matters at hackathon data
volumes. What is gained is a managed free tier, connection pooling, and a public
URL — and the submission requires a **publicly reachable** prototype.

Firebase and Convex were considered and rejected: training needs bulk SQL reads,
and Firestore free-tier read quotas would break a single training run.

## 7. Open-Meteo instead of PAGASA and OpenWeather

**Profile (§4):** *"API feeds: PAGASA and OpenWeather marine forecasts."*

**Build:** Open-Meteo Marine.

**PAGASA has no public programmatic API.** It publishes bulletins for human
readers. It is a stated future integration — appropriate for a Philippine
product and worth pursuing through a data-sharing agreement — but it is not a
working data source in this build and the profile should not have implied it was.

Open-Meteo is preferred over OpenWeather for the marine variables actually
needed: wave height, wave direction and ocean current are in its free tier and
behind OpenWeather's paid tier. It is CC BY 4.0 and needs no API key, so a judge
can clone and run without registering anywhere.

## 8. Charted bathymetry instead of sonar — and not yet built

**Profile (§2.1, §3.1):** sonar hardware supplies the depth safety constraint.

**Build:** charted depth from GEBCO is the intended substitute, forced by the
no-hardware declaration — there is no sonar transducer because there is no
vessel. Charted depth is not live: it misses uncharted obstructions and silting,
which is exactly why the profile specifies sonar for production. The constraint
would be real; the sensing is not.

**Status as of 2026-07-22: not integrated.** The depth constraint belongs to
Route Optimization, which is not built yet. GEBCO is therefore deliberately
*absent* from `data/registry.py`, so `data/download.py` refuses to fetch it and
no part of the repository can imply a bathymetry source is in use when it is
not. The chart the display draws is Natural Earth coastline only, extracted by
`data/build_chart.py`, and it carries its own scale caveat in the output file.

Recording an unbuilt thing as unbuilt is cheaper than being asked about it.

## 9. AIS is omitted

**Profile (§3.2):** AIS receiver, marked *recommended addition*, feeding traffic
avoidance in the MPC loop.

**Build:** omitted. No free live AIS feed covers Philippine coastal waters at
usable rates, and fabricating vessel traffic would make the collision-avoidance
constraint a demo of our own random number generator.

`RoutingFrame.nearby_vessel_count` exists in the contract and stays `None`, so
the field is reserved and the omission is visible in the data model rather than
hidden. It was a recommended addition, not baseline.

## 10. Satellite imagery: Sentinel-2, not Google

Not a deviation from the profile — the profile says nothing about basemaps — but
recorded here because it is the kind of decision a judge should be able to check.

The display shows **real satellite imagery of the Iloilo Strait**: Sentinel-2
cloudless 2020 by EOX, CC BY 4.0, containing modified Copernicus Sentinel data.
10 m ground resolution, no API key, attributed on screen as the licence requires.

**Google, Bing and Esri satellite tiles were considered and rejected.** Their
terms permit use only through their own APIs and forbid redistributing or
re-hosting imagery in another application; a screenshot of a web map is neither
licensed nor attributable. The brief grades "use only licensed or public
datasets", so a Google Maps capture is a scoring risk before it is anything
else. An earlier prototype used exactly that and it was removed.

The same imagery does double duty: `data/build_chart.py` classifies it into a
land/water mask, and the helm view ray-casts that mask to build its horizon. So
the shoreline under the vessel and the silhouette on the skyline come from one
source and cannot contradict each other.

Its limits are recorded in `data/registry.py`: it is imagery, not a chart. No
depth, no aids to navigation, and an annual composite showing no particular day.

## 11. Docker

**Profile (§3.1 Tools):** Docker for edge deployment.

Accurate for deployment — the API ships to Fly.io as a container. It is simply
not used for local development on the build machine. Listed only for
completeness.

---

## Not deviations

Worth stating, because they are the claims most likely to be doubted:

- **No physical hardware, and it is declared.** Every frame carries
  `source="simulator"`, and the contract has no default that could claim
  otherwise (`packages/contracts/telemetry.py`).
- **The Phase 1 / Phase 2 maintenance maturity curve is real and enforced in
  code**, not just described. The contract refuses to emit a component-level
  prediction while a vessel is in Phase 1 — the profile's honesty commitment is
  a validator, not a promise.
- **The AI-authority boundary holds.** Safety cutoffs are rule-based and
  independent of every model (`packages/contracts/safety.py`). No module
  actuates anything.
- **Claude phrases, it does not decide.** The advisory layer never produces a
  number, a threshold, or a recommendation; there is a deterministic template
  fallback and the source is labelled in the payload
  (`SpeedRecommendation.advisory_source`).

# Marine-AI

**Team SOLMATE — National AI Hackathon 2026, Blue Economy / Clean Energy track**

A retrofittable IoT and AI advisory system for traditional diesel fiberglass
passenger boats in the Philippines. Three sensor systems feed three parallel AI
modules — Speed Optimization, Route Optimization, and Predictive Maintenance —
which converge on a single bridge display showing a live waypoint route and a
recommended throttle setting.

No new vessel. No engine replacement. It installs onto boats already in service.

- **Live demo:** _(deployed D11 — see `docs/DEPLOY.md`)_
- **Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- **Data sources and licences:** [`docs/DATA.md`](docs/DATA.md)
- **Ethics and bias mitigation:** [`docs/ETHICS.md`](docs/ETHICS.md)
- **Deviations from the technical profile:** [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md)

---

## Hardware declaration

**No physical hardware was used in this submission.**

Every telemetry frame is produced by the documented sensor simulator in
`packages/sim/`, which replays public research datasets along a synthetic
Philippine short-haul route and overlays live marine forecast data. Frames carry
`source="simulator"` and the contract default never claims otherwise
(`packages/contracts/telemetry.py`).

The sensor set the simulator emits is exactly the set a physical retrofit kit
would install, so the pipeline from ingest to display is the one that would run
on a vessel. What is simulated is the sensing, not the system.

## External models and datasets

All external work is cited in [`docs/DATA.md`](docs/DATA.md) with licence,
source URL, and retrieval date. Summary:

| Component | Origin |
|---|---|
| Fuel-consumption model | Trained by us (XGBoost) on the UCI *Condition Based Maintenance of Naval Propulsion Plants* dataset. **Gas turbine data used as a documented proxy for diesel** — see `docs/DATA.md`. |
| Weather / wave / current forecasting | Trained by us (Temporal Fusion Transformer) on Open-Meteo Marine historical reanalysis. |
| Anomaly detection | Trained by us (autoencoder + Isolation Forest), pretrained on the NASA C-MAPSS run-to-failure dataset. FEMTO/PRONOSTIA is named in the technical profile but **is not used** — it is 25.6 kHz bench-rig vibration and the specified retrofit IMU logs at ~1 Hz. See [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md). |
| Natural-language advisory | Anthropic Claude API (`claude-sonnet-5`), used for phrasing only. It never produces a number, a threshold, or a recommendation. |
| Chart geometry | Natural Earth coastline (public domain), GEBCO bathymetry. |

No pretrained model weights from third parties are shipped in this repository.

---

## Setup

Requires **Python 3.11+** and **Node 20+**. Docker is optional (local edge stack only).

```bash
git clone https://github.com/<org>/marine-ai.git
cd marine-ai
```

### 1. Python services

```bash
pip install uv                      # if you don't have it
uv venv --python 3.11
uv pip install -e ".[train,dev]"    # omit [train] to skip torch (~2 GB)
```

Activate the venv: `source .venv/bin/activate` (macOS/Linux) or
`.venv\Scripts\activate` (Windows).

### 2. Fetch the datasets

Nothing under `data/` is committed. Every source is public and fetched by script:

```bash
python -m data.download --all       # or --dataset uci-cbm
```

Each download asserts its licence and records a card in `docs/DATA.md`.

### 3. Train the models

```bash
python -m services.speed.train          # ~30s   XGBoost fuel model
python -m services.route.train          # ~15min TFT forecaster
python -m services.maintenance.train    # ~5min  autoencoder + IsolationForest
```

Artifacts land in `models/`, each beside a `MODEL_CARD.md` with held-out metrics.
Metrics are computed on **voyage-wise** splits, not row-wise — row-wise splitting
leaks on time series and inflates the numbers.

### 4. Run

```bash
cp .env.example .env                # add ANTHROPIC_API_KEY (optional)
uvicorn apps.api.main:app --reload  # http://localhost:8000
python -m packages.sim.run          # feeds a synthetic voyage into the API
```

In a second terminal:

```bash
cd apps/bridge
bun install
bun dev                             # http://localhost:3000
```

The advisory layer runs without `ANTHROPIC_API_KEY`. It falls back to
deterministic templates, and the display marks the source. Nothing in the demo
path blocks on a network call.

### 5. Test

```bash
pytest                              # contracts, ingest validation, safety rules
cd apps/bridge && bun run typecheck
```

---

## Repository layout

```
apps/bridge/        Next.js bridge display — the captain's screen
apps/api/           FastAPI: ingest, three module routers, SSE stream
services/speed/     XGBoost fuel model + throttle optimizer
services/route/     TFT forecaster + MPC waypoint solver
services/maintenance/  Autoencoder + Isolation Forest (Phase 1); RSF scaffold (Phase 2)
services/safety/    Rule-based cutoffs. Deterministic, no ML, no network.
packages/contracts/ Pydantic models -> generated TypeScript. Single source of truth.
packages/sim/       Vessel and sensor simulator, with fault injection
packages/ingest/    Range checks, timestamp validation, drift monitoring
data/               Download scripts. No data committed.
models/             Trained artifacts + model cards
infra/              Dockerfile, compose (TimescaleDB + Mosquitto), fly.toml
docs/               Architecture, data, ethics, deviations
```

## The AI-authority boundary

Marine-AI never overrides the captain and never autonomously actuates the
vessel. All three modules produce recommendations; the captain acts on them.

Safety cutoffs — over-temperature, over-pressure, critical battery voltage —
are **rule-based, not ML-based**, so behaviour under fault is deterministic and
auditable. `services/safety/` imports no model, loads no artifact, and makes no
network call. Given the same input it returns the same answer, forever.

This is simultaneously an ethics safeguard, a legal necessity under maritime
liability, and the correct engineering answer for advisory AI in a high-stakes
physical domain.

## Maturity: Predictive Maintenance is honest about what it cannot yet do

Predictive maintenance cannot be precise on day one. It needs labelled failure
history, not just sensors.

- **Phase 1 (months 0–24):** unsupervised anomaly detection. Flags *that*
  something is deviating and *which sensor stream*. Cannot name a component or
  a date.
- **Phase 2 (after ~24 months):** with accumulated labelled maintenance history,
  supervised remaining-useful-life models per component.

This build ships **Phase 1 only** as a live prediction, because that is the
honest state of any newly installed unit. The constraint is enforced in code:
`packages/contracts/maintenance.py` raises a validation error if a Phase 1 status
carries a component name or a maintenance date. The Phase 2 model exists in
`services/maintenance/rul_scaffold.py` as roadmap evidence, trained on clearly
labelled synthetic data, and is not wired to the display.

## Licence

MIT — see [`LICENSE`](LICENSE).

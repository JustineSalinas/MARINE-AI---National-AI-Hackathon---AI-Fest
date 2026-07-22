"""Marine-AI advisory API.

    uvicorn apps.api.main:app --reload

One endpoint carries the product: `POST /advise` takes conditions and returns
the cheapest throttle that still meets the schedule, with the reasoning
itemised. Everything the bridge display shows comes from here.

Why the display calls an API rather than computing anything itself: there is one
fuel model in this system and it is in Python. A JavaScript reimplementation --
even a faithful one -- becomes a second model the moment either is edited, and
the version the judges see would drift from the version the tests cover. The
browser renders; it does not decide.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.schemas import (
    AdviseRequest,
    AdviseResponse,
    CurvePoint,
    EmissionsOut,
    PowerOut,
    WearOut,
)
from services.emissions import co2_kg
from services.speed.fuel import EngineSpec, FuelMap
from services.speed.optimizer import (
    as_recommendation,
    optimise_throttle,
    performance_curve,
    power_for_rpm,
    speed_for_power_kn,
)
from services.speed.resistance import SeaState, VesselHull

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained wear model once, not per request.

    `FuelMap.load` degrades gracefully when the artifact is missing -- wear stays
    at 1.0 and confidence drops -- so a fresh clone boots and serves before
    anyone has run the trainer. `model_trained` on every response says which
    mode is active rather than leaving the client to guess.
    """
    _state["fuel_map_loaded"] = FuelMap.load(EngineSpec(rated_kw=90.0, rated_rpm=2800.0))
    yield
    _state.clear()


app = FastAPI(
    title="Marine-AI Advisory API",
    version="0.1.0",
    summary="Conditions in, recommended throttle out. Advisory only.",
    lifespan=lifespan,
)

# Local dev ports for the bridge display. 3000 is the Next default; the others
# are there because a developer machine frequently already has something on it.
_DEFAULT_ORIGINS = [
    f"http://{host}:{port}"
    for host in ("localhost", "127.0.0.1")
    for port in (3000, 3100, 3001)
]
_origins = [
    origin.strip()
    for origin in os.environ.get("MARINE_AI_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

# Preview deployments get a fresh hostname every push, so pinning the deployed
# display by exact origin means the preview link silently stops working. The
# regex covers them. This API is public, unauthenticated and read-only -- it
# holds no session and returns the same answer to anyone -- so a permissive
# origin policy gives away nothing that GET /advise does not already give away.
_origin_regex = os.environ.get("MARINE_AI_CORS_ORIGIN_REGEX") or r"https://.*\.vercel\.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or _DEFAULT_ORIGINS,
    allow_origin_regex=_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _fuel_map_for(spec: EngineSpec) -> FuelMap:
    """Reuse the loaded wear model with this request's engine spec.

    The wear model is dimensionless -- it predicts a fuel *ratio*, not litres --
    so it transfers across engine specs unchanged. Only the BSFC baseline and
    the idle floor are per-vessel, and those live on `EngineSpec`.
    """
    loaded = _state.get("fuel_map_loaded")
    wear_model = getattr(loaded, "_wear_model", None)
    load_range = getattr(loaded, "_load_range", None)
    return FuelMap(spec, wear_model=wear_model, load_range=load_range)


@app.get("/health")
async def health() -> dict:
    loaded = _state.get("fuel_map_loaded")
    return {
        "status": "ok",
        "wear_model_loaded": bool(getattr(loaded, "has_wear_model", False)),
        "advisory_only": True,
    }


@app.post("/advise", response_model=AdviseResponse)
async def advise(req: AdviseRequest) -> AdviseResponse:
    hull = VesselHull(
        length_waterline_m=req.vessel.length_waterline_m,
        beam_m=req.vessel.beam_m,
        draft_m=req.vessel.draft_m,
        displacement_kg=req.vessel.displacement_kg,
        admiralty_coefficient=req.vessel.admiralty_coefficient,
    )
    spec = EngineSpec(
        rated_kw=req.vessel.rated_kw,
        rated_rpm=req.vessel.rated_rpm,
        best_bsfc_g_per_kwh=req.vessel.best_bsfc_g_per_kwh,
        idle_burn_lph=req.vessel.idle_burn_lph,
    )
    fuel_map = _fuel_map_for(spec)

    sea = SeaState(
        wind_speed_kn=req.sea.wind_speed_kn,
        wind_direction_deg=req.sea.wind_direction_deg,
        current_speed_kn=req.sea.current_speed_kn,
        current_direction_deg=req.sea.current_direction_deg,
        wave_height_m=req.sea.wave_height_m,
        wave_direction_deg=req.sea.wave_direction_deg,
    )
    load = req.added_load_kg

    advice = optimise_throttle(
        hull,
        spec,
        fuel_map,
        sea,
        req.heading_deg,
        distance_remaining_nm=req.distance_remaining_nm,
        minutes_available=req.minutes_available,
        current_rpm=req.current_rpm,
        added_load_kg=load,
        egt_excess_ratio=req.egt_excess_ratio,
    )

    max_speed = speed_for_power_kn(hull, spec.rated_kw, sea, req.heading_deg, added_load_kg=load)

    # The honest speed: what the vessel actually makes at the throttle the
    # captain is holding, in the weather it is actually in.
    achievable = None
    if req.current_rpm:
        achievable = speed_for_power_kn(
            hull, power_for_rpm(spec, req.current_rpm), sea, req.heading_deg, added_load_kg=load
        )

    curve = performance_curve(
        hull,
        spec,
        fuel_map,
        sea,
        req.heading_deg,
        added_load_kg=load,
        egt_excess_ratio=req.egt_excess_ratio,
        max_speed_kn=max(1.0, max_speed),
    )

    best = advice.recommended
    php = req.php_per_litre

    return AdviseResponse(
        recommendation=as_recommendation(
            advice, vessel_id=req.vessel.vessel_id, php_per_litre=php
        ),
        power=PowerOut(
            total_kw=best.power.total_kw,
            calm_water_kw=best.power.calm_water_kw,
            wind_kw=best.power.wind_kw,
            wave_kw=best.power.wave_kw,
            speed_through_water_kn=best.power.speed_through_water_kn,
            environmental_penalty_pct=best.power.environmental_penalty_pct,
        ),
        wear=WearOut(
            multiplier=best.burn.wear_multiplier,
            penalty_lph=best.burn.wear_penalty_lph,
            penalty_php_per_hour=None if php is None else best.burn.wear_penalty_lph * php,
        ),
        emissions=EmissionsOut(
            co2_kg_per_hour=co2_kg(best.litres_per_hour),
            co2_kg_per_nm=(
                co2_kg(best.litres_per_hour / best.speed_kn) if best.speed_kn > 0 else None
            ),
        ),
        curve=[
            CurvePoint(
                speed_kn=o.speed_kn,
                rpm=o.rpm,
                shaft_kw=o.power.total_kw,
                litres_per_hour=o.litres_per_hour,
            )
            for o in curve
        ],
        achievable_speed_kn=achievable,
        max_speed_kn=max_speed,
        feasible=advice.feasible,
        notes=list(advice.notes) + list(best.burn.caveats),
        model_trained=fuel_map.has_wear_model,
    )

// Client for the Marine-AI advisory API.
//
// This file is the only place the display learns anything about fuel, power, or
// achievable speed. Nothing in the browser computes physics.
//
// That rule is deliberate and worth stating where someone will read it. There is
// one fuel model in this system, it lives in services/speed/, it is trained
// against held-out engine wear states and covered by tests. A JavaScript
// reimplementation -- however faithful on the day it is written -- becomes a
// second model the moment either side is edited, and the version on stage would
// silently diverge from the version the tests defend.
//
// The cost of the rule is latency, and it is paid for by `curve`: the API
// returns the whole speed/burn relationship for the current conditions, so the
// 60fps loop interpolates real model output locally instead of guessing.

import type { AdviseRequest, AdviseResponse, CurvePoint } from "./contracts";

export type { AdviseRequest, AdviseResponse, CurvePoint };

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiUnavailable extends Error {}

export async function advise(
  body: AdviseRequest,
  signal?: AbortSignal,
): Promise<AdviseResponse> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/advise`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (cause) {
    throw new ApiUnavailable("advisory service unreachable", { cause });
  }
  if (!response.ok) {
    throw new ApiUnavailable(`advisory service returned ${response.status}`);
  }
  return (await response.json()) as AdviseResponse;
}

/**
 * Speed the vessel makes at a given RPM, read off the model's own curve.
 *
 * Between API calls the throttle still has to feel live, so the curve is
 * interpolated rather than re-derived. Both RPM and speed increase monotonically
 * along it, which makes a linear search safe and a binary search unnecessary at
 * this size.
 *
 * Returns null when there is no curve yet -- the caller must then show the
 * vessel as stationary rather than inventing a speed, which is precisely the
 * failure this whole rewrite exists to remove.
 */
export function speedForRpm(curve: CurvePoint[] | null, rpm: number): number | null {
  if (!curve || curve.length === 0) return null;
  if (rpm <= curve[0].rpm) return curve[0].speed_kn * (rpm / Math.max(1, curve[0].rpm));

  for (let i = 1; i < curve.length; i++) {
    const a = curve[i - 1];
    const b = curve[i];
    if (rpm <= b.rpm) {
      const span = b.rpm - a.rpm;
      const t = span <= 0 ? 0 : (rpm - a.rpm) / span;
      return a.speed_kn + (b.speed_kn - a.speed_kn) * t;
    }
  }
  return curve[curve.length - 1].speed_kn;
}

/** Burn at a given RPM, interpolated from the same curve. */
export function burnForRpm(curve: CurvePoint[] | null, rpm: number): number | null {
  if (!curve || curve.length === 0) return null;
  if (rpm <= curve[0].rpm) return curve[0].litres_per_hour;

  for (let i = 1; i < curve.length; i++) {
    const a = curve[i - 1];
    const b = curve[i];
    if (rpm <= b.rpm) {
      const span = b.rpm - a.rpm;
      const t = span <= 0 ? 0 : (rpm - a.rpm) / span;
      return a.litres_per_hour + (b.litres_per_hour - a.litres_per_hour) * t;
    }
  }
  return curve[curve.length - 1].litres_per_hour;
}

/** Throttle percent -> RPM. The captain's control is a lever, not a tachometer. */
export function rpmForThrottle(throttlePct: number, ratedRpm: number): number {
  return (Math.max(0, Math.min(100, throttlePct)) / 100) * ratedRpm;
}

// Land/water mask, and the ray-cast that turns it into a horizon.
//
// The mask is derived from the same Sentinel-2 imagery the chart displays
// (`data/build_chart.py`), which is the point: the silhouette on the horizon and
// the coastline under the vessel come from one source and cannot disagree.
//
// This is what makes the helm view show *Guimaras* rather than a generic lump of
// land. Turn the boat and the shape on the horizon changes correctly, because it
// is being sampled from real geography at real bearings.

import { DEG, clamp, normaliseDeg } from "./nautical";
import type { Vec } from "./nautical";

export interface LandMask {
  width: number;
  height: number;
  /** One byte per cell, 1 = land. */
  data: Uint8Array;
  /** Nautical miles spanned by the full mask, per axis. They differ: a degree of
   *  longitude is shorter than a degree of latitude at this latitude. */
  widthNm: number;
  heightNm: number;
}

export async function loadLandMask(
  url: string,
  widthNm: number,
  heightNm: number,
): Promise<LandMask | null> {
  try {
    const image = new Image();
    image.src = url;
    await image.decode();

    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return null;

    ctx.drawImage(image, 0, 0);
    const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;

    const data = new Uint8Array(canvas.width * canvas.height);
    for (let i = 0; i < data.length; i++) data[i] = rgba[i * 4] > 127 ? 1 : 0;

    return { width: canvas.width, height: canvas.height, data, widthNm, heightNm };
  } catch {
    // No mask is a survivable state: the helm view simply shows open horizon.
    return null;
  }
}

/** Is this normalised chart position on land? Off-mask counts as land, so a ray
 *  that leaves the charted window terminates rather than running forever. */
export function isLand(mask: LandMask, nx: number, ny: number): boolean {
  const x = Math.floor(nx * mask.width);
  const y = Math.floor(ny * mask.height);
  if (x < 0 || y < 0 || x >= mask.width || y >= mask.height) return true;
  return mask.data[y * mask.width + x] === 1;
}

export interface HorizonSample {
  /** Compass bearing of this sample. */
  bearingDeg: number;
  /** Distance to the shoreline along that bearing, nautical miles. */
  distanceNm: number;
  /** False when the ray reached the edge of the charted window without hitting
   *  land -- open sea, and nothing should be drawn on the horizon. */
  land: boolean;
}

const STEP_NM = 0.05;
const MAX_RANGE_NM = 14;

/**
 * Cast a ray from the vessel along a compass bearing until it meets land.
 *
 * Marching in nautical miles rather than pixels is what keeps bearings honest:
 * the mask's cells are not square in ground distance, so stepping in cell units
 * would stretch every direction except due north.
 */
export function rayToShore(mask: LandMask, from: Vec, bearingDeg: number): HorizonSample {
  const rad = normaliseDeg(bearingDeg) * DEG;
  // Compass: 0 is north, which is -y on a screen-oriented chart.
  const dxNm = Math.sin(rad);
  const dyNm = -Math.cos(rad);

  // A berth is on the shore, so at the start and end of every crossing the
  // vessel's own position classifies as land and every ray would report a hit
  // at the first step -- a wall of terrain in all directions. What the eye
  // actually sees from a wharf is the far shore across the water, so the ray
  // must clear its own shoreline before it starts looking for the next one.
  let clearedOwnShore = false;

  for (let travelled = STEP_NM; travelled <= MAX_RANGE_NM; travelled += STEP_NM) {
    const nx = from.x + (dxNm * travelled) / mask.widthNm;
    const ny = from.y + (dyNm * travelled) / mask.heightNm;

    if (nx < 0 || ny < 0 || nx > 1 || ny > 1) {
      return { bearingDeg, distanceNm: travelled, land: false };
    }

    const land = isLand(mask, nx, ny);
    if (!land) {
      clearedOwnShore = true;
    } else if (clearedOwnShore) {
      return { bearingDeg, distanceNm: travelled, land: true };
    }
  }
  return { bearingDeg, distanceNm: MAX_RANGE_NM, land: false };
}

/**
 * The full horizon across the field of view, one sample per screen column band.
 *
 * `columns` trades detail against cost. At 90 columns over a 70-degree field
 * each sample is about three quarters of a degree, which is finer than a
 * coastline three miles away can be resolved anyway.
 */
export function horizonProfile(
  mask: LandMask,
  from: Vec,
  headingDeg: number,
  fovDeg: number,
  columns = 90,
): HorizonSample[] {
  const out: HorizonSample[] = [];
  for (let i = 0; i <= columns; i++) {
    const offset = (i / columns - 0.5) * fovDeg;
    out.push(rayToShore(mask, from, headingDeg + offset));
  }
  return out;
}

const METRES_PER_NM = 1852;

/**
 * Deterministic terrain height for a bearing, metres.
 *
 * The mask carries no elevation -- it is a land/water classification, nothing
 * more -- so this is honest invention, and it is bounded to the range Guimaras
 * and the Iloilo shore actually occupy (roughly 60-240 m). Its job is to stop
 * the skyline being a perfectly straight edge, which is the single thing that
 * makes a rendered horizon read as fake.
 *
 * Deterministic in bearing rather than random in time: the same hill must be in
 * the same place on the next frame, or the coast shimmers.
 */
export function terrainHeightM(bearingDeg: number): number {
  const b = normaliseDeg(bearingDeg) * DEG;
  const ridge =
    Math.sin(b * 3.1) * 0.5 + Math.sin(b * 7.3 + 1.2) * 0.3 + Math.sin(b * 13.7 + 2.9) * 0.2;
  return 150 + ridge * 90;
}

/**
 * Apparent height of a shoreline on the horizon, in pixels.
 *
 * Proper angular subtense -- `atan(height / distance)` scaled by the display's
 * pixels-per-degree -- rather than an inverse-distance fudge. That matters
 * because it is what makes closing the land feel right: at two miles Guimaras
 * is a low green strip, at half a mile it fills the windscreen, and the rate it
 * grows between the two is the rate a captain's eye expects.
 */
export function shoreElevation(
  distanceNm: number,
  bearingDeg: number,
  pixelsPerDegree: number,
  screenHeight: number,
): number {
  const distanceM = Math.max(250, distanceNm * METRES_PER_NM);
  const angleDeg = (Math.atan(terrainHeightM(bearingDeg) / distanceM) * 180) / Math.PI;
  // Capped at a third of the frame. Past that the view stops being a horizon
  // and becomes a wall, and the sea -- which is what the captain is actually
  // reading -- disappears off the bottom of the screen.
  return clamp(angleDeg * pixelsPerDegree, 0, screenHeight * 0.34);
}

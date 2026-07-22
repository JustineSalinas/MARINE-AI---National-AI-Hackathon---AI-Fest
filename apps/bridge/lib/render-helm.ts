// Helm view: the horizon from behind the wheel.
//
// Procedural, on the same 2D canvas as the chart. No game engine, no 3D assets,
// no extra dependency -- and, more to the point, no decoration. Every motion in
// this view is derived from a modelled quantity:
//
//   roll amplitude    wave height x |sin(relative wave angle)|   beam seas roll worst
//   pitch amplitude   wave height x |cos(relative wave angle)|   head seas pitch worst
//   motion period     encounter period, which shortens as you steam into a sea
//   sea scroll rate   the vessel's actual speed from the fuel model's curve
//   spray             appears when pitching hard into a head sea
//
// So turning the boat changes how it moves, at the same wave height, in the way
// a captain would expect. That is the difference between a horizon that looks
// like the sea and one that behaves like it -- and it is defensible under
// questioning, which a rendered asset would not be.
//
// The convention for relative wave angle matches lib/nautical.ts and
// services/speed/resistance.py: 0 is a head sea.

import {
  DEG,
  clamp,
  encounterPeriodSeconds,
  normaliseDeg,
  relativeAngle,
  relativeBearing,
} from "./nautical";
import { type HorizonSample, shoreElevation } from "./landmask";


export interface HelmScene {
  width: number;
  height: number;
  timeSeconds: number;

  headingDeg: number;
  speedKn: number;
  recommendedRpm: number | null;
  currentRpm: number;

  waveHeightM: number;
  waveDirectionDeg: number;
  windSpeedKn: number;
  windDirectionDeg: number;
  gloom: number;
  rain: number;

  /** Shoreline distance per bearing, ray-cast from the Sentinel-2 land mask.
   *  This is why the horizon shows the real shape of Guimaras rather than a
   *  generic landmass, and why turning the vessel changes it correctly. */
  horizon: HorizonSample[];
}

export interface VesselMotion {
  rollDeg: number;
  pitchDeg: number;
  /** Seconds between wave encounters. Shown in the HUD; it is a real quantity. */
  encounterPeriod: number;
}

const MAX_ROLL_DEG = 22;
const MAX_PITCH_DEG = 9;
const HORIZON_FRACTION = 0.46;
export const HELM_FOV_DEG = 70;
/** Horizontal field of view. Exported because the horizon ray-cast in
 *  lib/landmask.ts must sample across exactly this span: if the two disagree,
 *  the shoreline silhouette slides against the vessel's actual heading and the
 *  view stops being a view of anywhere. */

/**
 * Roll and pitch at this instant, from the sea state and the vessel's own motion.
 *
 * Two harmonics rather than one: real vessel motion is not a clean sinusoid, and
 * a single sine reads as a metronome within about five seconds of watching.
 */
export function vesselMotion(scene: HelmScene): VesselMotion {
  const { waveHeightM: H, headingDeg, waveDirectionDeg, speedKn, timeSeconds: t } = scene;

  if (H <= 0.02) return { rollDeg: 0, pitchDeg: 0, encounterPeriod: 6 };

  const relative = relativeAngle(headingDeg, waveDirectionDeg) * DEG;
  const period = encounterPeriodSeconds(H, speedKn, headingDeg, waveDirectionDeg);
  const phase = (2 * Math.PI * t) / period;

  // Which side the sea is on decides which way the first roll goes.
  const side = Math.sin(relativeBearing(headingDeg, waveDirectionDeg) * DEG) >= 0 ? 1 : -1;

  const rollAmp = clamp(H * 6.5 * Math.abs(Math.sin(relative)), 0, MAX_ROLL_DEG);
  const pitchAmp = clamp(H * 3.2 * Math.abs(Math.cos(relative)), 0, MAX_PITCH_DEG);

  const roll = side * rollAmp * (0.82 * Math.sin(phase) + 0.18 * Math.sin(phase * 2.3 + 1.1));
  const pitch = pitchAmp * (0.85 * Math.sin(phase * 1.05 + 0.6) + 0.15 * Math.sin(phase * 2.7));

  return { rollDeg: roll, pitchDeg: pitch, encounterPeriod: period };
}

export function drawHelm(ctx: CanvasRenderingContext2D, scene: HelmScene): void {
  const { width: w, height: h } = scene;
  const motion = vesselMotion(scene);

  ctx.save();
  ctx.clearRect(0, 0, w, h);

  // Pitch raises and lowers the horizon; roll tilts it. Rotating about the
  // screen centre keeps the geometry stable at large roll angles.
  const horizonY = h * HORIZON_FRACTION + (motion.pitchDeg / MAX_PITCH_DEG) * h * 0.10;

  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.rotate(motion.rollDeg * DEG);
  // Overdraw well beyond the viewport so rotation never exposes a corner.
  ctx.translate(-w / 2, -h / 2);
  const bleed = Math.max(w, h);

  drawSky(ctx, scene, horizonY, bleed);
  drawLandmarks(ctx, scene, horizonY);
  drawSea(ctx, scene, horizonY, bleed, motion);

  ctx.restore();

  drawSpray(ctx, scene, motion, horizonY);
  drawRain(ctx, scene);

  if (scene.gloom > 0) {
    ctx.fillStyle = `rgba(6, 14, 26, ${scene.gloom})`;
    ctx.fillRect(0, 0, w, h);
  }

  drawBow(ctx, scene, motion);
  drawHud(ctx, scene, motion);
  ctx.restore();
}

function drawSky(ctx: CanvasRenderingContext2D, scene: HelmScene, horizonY: number, bleed: number) {
  const { width: w } = scene;
  const overcast = clamp(scene.gloom * 1.6, 0, 1);

  const sky = ctx.createLinearGradient(0, -bleed, 0, horizonY);
  sky.addColorStop(0, mix("#1e3a8a", "#0f172a", overcast));
  sky.addColorStop(0.7, mix("#7dd3fc", "#475569", overcast));
  sky.addColorStop(1, mix("#dbeafe", "#64748b", overcast));
  ctx.fillStyle = sky;
  ctx.fillRect(-bleed, -bleed, w + bleed * 2, horizonY + bleed);
}

function drawLandmarks(ctx: CanvasRenderingContext2D, scene: HelmScene, horizonY: number) {
  const { width: w, height: h } = scene;
  const samples = scene.horizon;
  if (samples.length < 2) return;

  // One continuous silhouette rather than separate blobs. Each ray gives the
  // distance to the shore on that bearing; nearer shore stands higher, and the
  // gaps where a ray found open water sit flat on the horizon.
  //
  // Atmospheric haze is applied per band by distance, which is what actually
  // sells depth at sea: the far shore of a strait is paler than the near one.
  const pixelsPerDegree = w / HELM_FOV_DEG;
  const bands: { x: number; top: number; distanceNm: number; land: boolean }[] = [];
  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    const x = (i / (samples.length - 1)) * w;
    const top = sample.land
      ? horizonY - shoreElevation(sample.distanceNm, sample.bearingDeg, pixelsPerDegree, h)
      : horizonY;
    bands.push({ x, top, distanceNm: sample.distanceNm, land: sample.land });
  }

  // Fill in one pass, then wash the near bands darker.
  ctx.beginPath();
  ctx.moveTo(0, horizonY);
  for (const band of bands) ctx.lineTo(band.x, band.top);
  ctx.lineTo(w, horizonY);
  ctx.closePath();

  const landBands = bands.filter((b) => b.land);
  const nearest = landBands.length ? Math.min(...landBands.map((b) => b.distanceNm)) : 12;
  // Aerial perspective: the further the shore, the more it takes the colour of
  // the sky. Two miles of tropical haze is a lot, and leaving it out is what
  // makes a drawn coastline look pasted on.
  const haze = clamp(nearest / 5, 0.08, 0.85);
  const top = Math.min(...bands.map((b) => b.top), horizonY);

  const gradient = ctx.createLinearGradient(0, top, 0, horizonY);
  gradient.addColorStop(0, `rgba(96, 122, 104, ${0.72 - haze * 0.28})`);
  gradient.addColorStop(0.45, `rgba(52, 78, 58, ${0.85 - haze * 0.30})`);
  gradient.addColorStop(1, `rgba(22, 40, 30, ${0.92 - haze * 0.25})`);
  ctx.fillStyle = gradient;
  ctx.fill();

  // A thin lighter lip where land meets water reads as the surf line.
  ctx.beginPath();
  ctx.moveTo(0, horizonY);
  for (const band of bands) ctx.lineTo(band.x, band.top);
  ctx.strokeStyle = `rgba(150, 180, 155, ${0.35 - haze * 0.2})`;
  ctx.lineWidth = 1.2;
  ctx.stroke();
}

function drawSea(
  ctx: CanvasRenderingContext2D,
  scene: HelmScene,
  horizonY: number,
  bleed: number,
  motion: VesselMotion,
) {
  const { width: w, height: h, timeSeconds: t } = scene;

  const sea = ctx.createLinearGradient(0, horizonY, 0, h + bleed);
  sea.addColorStop(0, mix("#1e4e6b", "#0b1f2e", scene.gloom));
  sea.addColorStop(1, mix("#04121d", "#020a12", scene.gloom));
  ctx.fillStyle = sea;
  ctx.fillRect(-bleed, horizonY, w + bleed * 2, h + bleed * 2);

  // Crest lines laid out in perspective: rows near the horizon are compressed,
  // rows near the viewer are far apart. Scroll rate is the vessel's real speed.
  const rows = 46;
  const depth = h + bleed - horizonY;
  const scroll = (t * Math.max(0.4, scene.speedKn)) / 9;
  const relative = relativeAngle(scene.headingDeg, scene.waveDirectionDeg) * DEG;
  // A beam sea shows crests running across the view; a head sea shows them square on.
  const skew = Math.sin(relative) * 0.55;

  ctx.lineWidth = 1.4;
  for (let i = 1; i <= rows; i++) {
    const f = i / rows;
    const perspective = f * f; // foreshortening
    const y = horizonY + perspective * depth;
    const phase = (perspective * 7 - scroll) % 1;

    const amplitude = clamp(scene.waveHeightM, 0, 4) * perspective * 10;
    // Crest contrast rises with sea state: a glassy calm should look glassy, and
    // a 2 m sea should be legible through the gloom of a storm overlay.
    const seaContrast = 0.35 + clamp(scene.waveHeightM / 2.5, 0, 1) * 0.85;
    const alpha =
      clamp(0.05 + perspective * 0.30, 0, 0.45) *
      seaContrast *
      (0.55 + 0.45 * Math.sin(phase * Math.PI * 2));
    if (alpha <= 0.02) continue;

    ctx.beginPath();
    const steps = 26;
    for (let s = 0; s <= steps; s++) {
      const px = (s / steps) * (w + bleed * 2) - bleed;
      const wobble =
        Math.sin((px / w) * Math.PI * 3 + t * 1.6 / Math.max(0.5, motion.encounterPeriod) * 6 + i) *
        amplitude;
      const py = y + wobble + skew * (px - w / 2) * perspective * 0.03;
      if (s === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = `rgba(186, 230, 253, ${alpha})`;
    ctx.stroke();
  }

  // The horizon itself, drawn last so crests never cross it.
  ctx.beginPath();
  ctx.moveTo(-bleed, horizonY);
  ctx.lineTo(w + bleed, horizonY);
  ctx.strokeStyle = "rgba(226, 232, 240, 0.5)";
  ctx.lineWidth = 1.2;
  ctx.stroke();
}

function drawSpray(
  ctx: CanvasRenderingContext2D,
  scene: HelmScene,
  motion: VesselMotion,
  horizonY: number,
) {
  // Spray comes over the bow when pitching into a sea at speed, and not otherwise.
  const relative = relativeAngle(scene.headingDeg, scene.waveDirectionDeg);
  const heading_into = relative < 70;
  const intensity = heading_into ? scene.waveHeightM * (scene.speedKn / 10) : 0;
  if (intensity < 0.35) return;

  const { width: w, height: h, timeSeconds: t } = scene;
  const burst = Math.max(0, Math.sin((2 * Math.PI * t) / motion.encounterPeriod));
  const count = Math.floor(clamp(intensity * 40, 0, 90) * burst);

  ctx.fillStyle = `rgba(224, 242, 254, ${clamp(0.25 * burst, 0, 0.4)})`;
  for (let i = 0; i < count; i++) {
    const seed = (i * 97.13 + Math.floor(t * 3) * 13.7) % 1;
    const x = w * 0.5 + (seed - 0.5) * w * 0.75;
    const y = horizonY + h * 0.28 + ((seed * 7919) % 1) * h * 0.3 - burst * h * 0.18;
    ctx.fillRect(x, y, 2.5, 2.5);
  }
}

function drawRain(ctx: CanvasRenderingContext2D, scene: HelmScene) {
  if (scene.rain <= 0) return;
  const { width: w, height: h, timeSeconds: t } = scene;
  const count = Math.min(420, scene.rain);
  const lean = clamp(scene.windSpeedKn / 40, 0, 0.8);

  ctx.strokeStyle = "rgba(203, 225, 255, 0.3)";
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  for (let i = 0; i < count; i++) {
    const seed = (i * 0.6180339887) % 1;
    const x = ((seed * w * 3 + t * 260 * lean) % (w + 200)) - 100;
    const y = ((seed * 7919 + t * 900) % (h + 120)) - 60;
    ctx.moveTo(x, y);
    ctx.lineTo(x - lean * 22, y + 26);
  }
  ctx.stroke();
}

function drawBow(ctx: CanvasRenderingContext2D, scene: HelmScene, motion: VesselMotion) {
  const { width: w, height: h } = scene;
  // The bow is fixed to the camera: the world moves, the boat does not. That is
  // what makes the view read as "from the helm" rather than "above the boat".
  const lift = (motion.pitchDeg / MAX_PITCH_DEG) * h * 0.015;

  ctx.save();
  ctx.translate(0, lift);

  // A foredeck seen in perspective: narrow at the stem, widening to the
  // gunwales at the bottom of frame. Sitting low leaves the sea visible, which
  // is the point of the view.
  ctx.beginPath();
  ctx.moveTo(w * 0.5, h * 0.925);
  ctx.quadraticCurveTo(w * 0.64, h * 0.945, w * 0.78, h * 0.99);
  ctx.lineTo(w * 0.86, h + 12);
  ctx.lineTo(w * 0.14, h + 12);
  ctx.lineTo(w * 0.22, h * 0.99);
  ctx.quadraticCurveTo(w * 0.36, h * 0.945, w * 0.5, h * 0.925);
  ctx.closePath();
  ctx.fillStyle = "#e8eef5";
  ctx.fill();
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Centreline, the reference a helmsman actually steers by.
  ctx.beginPath();
  ctx.moveTo(w * 0.5, h * 0.925);
  ctx.lineTo(w * 0.5, h + 12);
  ctx.strokeStyle = "rgba(100, 116, 139, 0.65)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.restore();
}

function drawHud(ctx: CanvasRenderingContext2D, scene: HelmScene, motion: VesselMotion) {
  const { width: w } = scene;
  ctx.font = "600 13px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.fillStyle = "rgba(248, 250, 252, 0.92)";
  ctx.textAlign = "center";

  const heading = String(Math.round(normaliseDeg(scene.headingDeg))).padStart(3, "0");
  ctx.fillText(
    `SPEED ${scene.speedKn.toFixed(1)} kts     HDG ${heading}°     ` +
      `ROLL ${Math.abs(motion.rollDeg).toFixed(0)}°     PERIOD ${motion.encounterPeriod.toFixed(1)}s`,
    w / 2,
    26,
  );

  if (scene.recommendedRpm != null) {
    ctx.font = "600 12px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "rgba(249, 115, 22, 0.95)";
    ctx.fillText(
      `ADVISORY  ${Math.round(scene.recommendedRpm)} RPM   (now ${Math.round(scene.currentRpm)})`,
      w / 2,
      46,
    );
  }
}

/** Blend two hex colours. Used to darken sky and sea with the weather. */
function mix(a: string, b: string, t: number): string {
  const pa = parseInt(a.slice(1), 16);
  const pb = parseInt(b.slice(1), 16);
  const f = clamp(t, 0, 1);
  const r = Math.round((pa >> 16) * (1 - f) + (pb >> 16) * f);
  const g = Math.round(((pa >> 8) & 255) * (1 - f) + ((pb >> 8) & 255) * f);
  const bl = Math.round((pa & 255) * (1 - f) + (pb & 255) * f);
  return `rgb(${r}, ${g}, ${bl})`;
}

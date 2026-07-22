"use client";

// The bridge display, plus the diagnostic detail the simulator needs.
//
// The top block is the captain's screen and follows PRODUCT.md: throttle first
// and largest, then route and ETA, then health, with a permanent trust bar
// carrying data freshness and ADVISORY ONLY. Everything below the divider is
// simulator instrumentation and would not appear on a real bridge.
//
// Two rules from PRODUCT.md are enforced here rather than merely intended:
//   1. Never hide the age of advice. `age` is always on screen.
//   2. Never use imperative language. The advisory sentence comes from the API
//      already phrased as a consequence, and this file does not rewrite it.

import type { Snapshot } from "./Simulator";

const DIESEL_CO2_KG_PER_L = 2.68;

export default function TelemetryPanel({ snapshot }: { snapshot: Snapshot | null }) {
  const response = snapshot?.api.response ?? null;
  const rec = response?.recommendation ?? null;
  const stale = (snapshot?.api.ageSeconds ?? 0) > 4;
  const offline = Boolean(snapshot?.api.error);

  const saved = snapshot ? snapshot.fuelUsedL - snapshot.advisedFuelL : 0;

  return (
    <aside className="flex w-96 shrink-0 flex-col border-l border-slate-800 bg-slate-900/60">
      {/* ---------- the captain's display ---------- */}
      <div className="border-b border-slate-800 p-4">
        <div className="mb-3 grid grid-cols-2 gap-3">
          <Tile
            label="Recommended"
            value={rec ? Math.round(rec.recommended_rpm).toString() : "—"}
            unit="rpm"
            emphasis
          />
          <Tile
            label="Speed"
            value={snapshot ? snapshot.speedKn.toFixed(1) : "—"}
            unit="kn"
          />
        </div>

        <p className="mb-3 min-h-[2.5rem] text-sm leading-snug text-slate-100">
          {rec?.advisory_en ?? "Waiting for the advisory service."}
        </p>
        {rec?.advisory_fil && (
          <p className="mb-3 text-xs italic leading-snug text-slate-400">{rec.advisory_fil}</p>
        )}

        <div className="space-y-1.5 text-xs">
          <Row
            label="Saves"
            value={
              rec
                ? `${rec.savings_lph >= 0 ? "" : "−"}${Math.abs(rec.savings_lph).toFixed(1)} L/h` +
                  (rec.savings_php_per_hour != null
                    ? `  ·  ₱${Math.abs(rec.savings_php_per_hour).toFixed(0)}/h`
                    : "")
                : "—"
            }
            tone={rec ? (rec.savings_lph > 0.05 ? "good" : "neutral") : "neutral"}
          />
          <Row
            label="Burn now"
            value={rec?.current_burn_lph != null ? `${rec.current_burn_lph.toFixed(1)} L/h` : "—"}
          />
          <Row
            label="Arrival impact"
            value={rec ? `${rec.eta_impact_minutes >= 0 ? "+" : ""}${rec.eta_impact_minutes.toFixed(1)} min` : "—"}
          />
          <Row
            label="Engine"
            value={
              response
                ? response.wear.multiplier > 1.005
                  ? `+${((response.wear.multiplier - 1) * 100).toFixed(1)}% fuel` +
                    (response.wear.penalty_php_per_hour != null
                      ? `  ·  ₱${response.wear.penalty_php_per_hour.toFixed(0)}/h`
                      : "")
                  : "nominal"
                : "—"
            }
            tone={response && response.wear.multiplier > 1.02 ? "warn" : "neutral"}
          />
        </div>

        {/* Trust bar. Permanent, not dismissible. */}
        <div className="mt-3 flex items-center justify-between border-t border-slate-800 pt-2 text-[10px]">
          <span
            className={
              offline ? "text-red-400" : stale ? "text-amber-400" : "text-emerald-400"
            }
          >
            {offline
              ? "ADVISORY OFFLINE — LAST KNOWN"
              : stale
                ? `STALE ${snapshot?.api.ageSeconds.toFixed(0)}s`
                : `LIVE ${snapshot?.api.ageSeconds.toFixed(1)}s`}
          </span>
          <span className="font-semibold tracking-wider text-slate-400">
            ADVISORY ONLY — CAPTAIN COMMANDS
          </span>
        </div>
      </div>

      {/* ---------- simulator instrumentation ---------- */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 border-b border-slate-800 p-4 text-[11px]">
        <Row label="Heading" value={snapshot ? `${Math.round(snapshot.headingDeg)}°` : "—"} />
        <Row label="Roll" value={snapshot ? `${Math.abs(snapshot.rollDeg).toFixed(0)}°` : "—"} />
        <Row
          label="Wind rel."
          value={snapshot ? `${Math.round(snapshot.relativeWindDeg)}°` : "—"}
        />
        <Row
          label="Sea rel."
          value={snapshot ? `${Math.round(snapshot.relativeWaveDeg)}°` : "—"}
        />
        <Row
          label="Encounter"
          value={snapshot ? `${snapshot.encounterPeriod.toFixed(1)} s` : "—"}
        />
        <Row
          label="Max speed"
          value={response ? `${response.max_speed_kn.toFixed(1)} kn` : "—"}
        />
        <Row
          label="Shaft"
          value={response ? `${response.power.total_kw.toFixed(0)} kW` : "—"}
        />
        <Row
          label="Weather cost"
          value={response ? `${response.power.environmental_penalty_pct.toFixed(0)}%` : "—"}
        />
      </div>

      {/* ---------- voyage + emissions ---------- */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 border-b border-slate-800 p-4 text-[11px]">
        <Row label="Fuel used" value={snapshot ? `${snapshot.fuelUsedL.toFixed(2)} L` : "—"} />
        <Row
          label="CO₂"
          value={snapshot ? `${(snapshot.fuelUsedL * DIESEL_CO2_KG_PER_L).toFixed(1)} kg` : "—"}
        />
        <Row
          label="If advised"
          value={snapshot ? `${snapshot.advisedFuelL.toFixed(2)} L` : "—"}
        />
        <Row
          label="CO₂ vs advised"
          value={
            snapshot
              ? `${saved >= 0 ? "−" : "+"}${Math.abs(saved * DIESEL_CO2_KG_PER_L).toFixed(1)} kg`
              : "—"
          }
          tone={saved > 0.01 ? "good" : "neutral"}
        />
        <Row label="Progress" value={snapshot ? `${(snapshot.progress * 100).toFixed(0)}%` : "—"} />
        <Row
          label="Elapsed"
          value={
            snapshot
              ? `${Math.floor(snapshot.elapsedSeconds / 60)}:${String(
                  Math.floor(snapshot.elapsedSeconds % 60),
                ).padStart(2, "0")}`
              : "—"
          }
        />
      </div>

      {/* ---------- log ---------- */}
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="shrink-0 px-4 py-2 text-[10px] font-semibold uppercase tracking-widest text-slate-500">
          Event log
        </div>
        <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto px-4 pb-4 font-mono text-[10px]">
          {(snapshot?.log ?? []).map((entry) => (
            <div
              key={entry.id}
              className={`border-l-2 pl-2 ${
                entry.kind === "alert"
                  ? "border-red-500 text-red-300"
                  : entry.kind === "warn"
                    ? "border-amber-500 text-amber-300"
                    : entry.kind === "advisory"
                      ? "border-orange-500 text-orange-300"
                      : "border-slate-700 text-slate-400"
              }`}
            >
              <span className="mr-1.5 text-slate-600">{entry.time}</span>
              {entry.message}
            </div>
          ))}
        </div>
      </div>

      {response && !response.model_trained && (
        <p className="shrink-0 border-t border-amber-900/50 bg-amber-950/30 px-4 py-2 text-[10px] text-amber-300">
          Wear model not trained — engine assumed healthy. Run{" "}
          <code>python -m services.speed.train</code>.
        </p>
      )}
    </aside>
  );
}

function Tile({
  label,
  value,
  unit,
  emphasis = false,
}: {
  label: string;
  value: string;
  unit: string;
  emphasis?: boolean;
}) {
  return (
    <div className="rounded border border-slate-700 bg-slate-950/70 p-2.5">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="flex items-baseline gap-1">
        <span
          className={`font-mono font-semibold ${emphasis ? "text-3xl text-white" : "text-2xl text-slate-200"}`}
        >
          {value}
        </span>
        <span className="text-xs text-orange-500">{unit}</span>
      </div>
    </div>
  );
}

function Row({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "warn";
}) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-slate-500">{label}</span>
      <span
        className={`font-mono ${
          tone === "good" ? "text-emerald-400" : tone === "warn" ? "text-amber-400" : "text-slate-200"
        }`}
      >
        {value}
      </span>
    </div>
  );
}

# Marine-AI — Product Definition

> Read by `/impeccable` before any interface work. This file settles what the
> display is for, so design decisions are arguments about the product rather
> than about taste.

**Register:** product. Design serves the product. This is a bridge instrument,
not a landing page.

## What it is

A retrofittable IoT and AI advisory system for traditional diesel fiberglass
passenger boats in the Philippines. It installs onto boats already in service —
no new vessel, no engine replacement. Three sensor systems feed three parallel
AI modules; all three converge on one bridge-mounted display.

## Who uses it

**Primary: the captain.** Twenty years at the helm, throttling on instinct built
from experience. Not an engineer. May read Filipino more comfortably than
English. Does not want to be taught, and has no reason to trust a computer that
has been on his boat for three weeks.

Secondary: the boat owner or cooperative (monthly savings and emissions
reports — a different surface, not this screen). Tertiary: LGU / MARINA / ESG
lenders (exported reports, not a screen at all).

## The scene that determines the design

*A captain on an open fiberglass boat at 05:40. Pre-dawn, running lights on,
salt spray on the glass. One hand on the throttle, the other on the wheel.
He looks at the screen for under two seconds at a time, maybe twice a minute.*

That scene forces every major decision:

- **Dark theme.** Not a preference. A bright screen at 05:40 destroys night
  vision, and night vision is a safety asset.
- **Very high contrast, oversized type.** Read at arm's length, in motion,
  through spray, possibly through salt-crusted glasses.
- **No hover-dependent information.** There is no mouse. There may be gloves.
- **No scrolling.** One screen. If it does not fit, it is not important enough.
- **Glance-sized answers.** A number and a direction, not a chart to interpret.
- **Degraded states are normal.** Signal drops on most of these routes. Offline
  is a designed state, not an error screen.

## What the captain must be able to answer in two seconds

1. **Should I change the throttle, and which way?** — the primary zone
2. **Where am I going next, and when do I arrive?** — the secondary zone
3. **Is my engine okay?** — quiet until it is not

Nothing else belongs on this screen.

## Zones, by glance priority

1. **THROTTLE** — largest element on the display. Recommended RPM, the delta
   from current, one plain-language sentence, and the litres-per-hour saved.
2. **ROUTE** — drawn nautical chart, next waypoint, ETA.
3. **HEALTH** — a single status strip. Expands only when something is wrong.

**Persistent trust bar** across all three: data freshness, connectivity state,
and `ADVISORY ONLY — CAPTAIN COMMANDS`. That last line is a legal and ethical
requirement under maritime liability, and it is a permanent design element, not
a dismissible banner.

## Non-negotiables

- **The system never overrides the captain and never actuates the vessel.**
  Every module advises. The captain decides. The interface must never use
  imperative language that implies otherwise ("Reduce to 1650 RPM" is wrong;
  "1650 RPM saves 2.1 L/h" is right).
- **Never hide the age of advice.** A stale recommendation displayed as fresh is
  worse than no recommendation. Freshness is always on screen.
- **Never overclaim maintenance.** During the ~24-month Phase 1 window the system
  says "coolant temperature is drifting", never "your impeller will fail in 40
  days". This is enforced in `packages/contracts/maintenance.py`.
- **Filipino is a first-class language,** not a translation afterthought.
  Defaults to Filipino when the device language is set to it.
- **Plain language throughout.** Nothing about the interface may assume an
  engineering background.

## Explicitly out of scope for this screen

Configuration, historical charts, model internals, fleet views, emissions
reports, maintenance logs. All of these belong on the owner's phone or the
shore-side dashboard. Putting any of them here costs a second of the captain's
attention that he does not have.

## How we know it works

The captain looks at it for two seconds and changes the throttle, or doesn't,
and is right either way. Measured as: litres of diesel per nautical mile down,
pesos per voyage down, unscheduled downtime converted to scheduled dry-docking.

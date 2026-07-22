"""Vercel serverless entrypoint for the advisory API.

Vercel's Python runtime looks for a module-level ASGI application in `api/`, so
this file exists only to make `apps.api.main:app` importable from the project
root and reachable at every path. All the logic is in `apps/api/`; nothing
behavioural belongs here.

**Why this fits in a serverless function at all.** Serving the wear model
through XGBoost would need xgboost, scikit-learn, scipy and pandas -- 358 MB of
runtime for a 363 kB tree ensemble, against Vercel's 500 MB ceiling on a
platform whose Linux wheels are larger than the Windows ones this was measured
on. `services/speed/train.py` exports the model to ONNX instead, and
`requirements.txt` here installs onnxruntime and nothing else from that stack:
roughly 77 MB, with headroom that does not depend on a wheel size we cannot
control.

The rest of the system is unchanged -- same `optimise_throttle`, same physics,
same tests. Only the format the trained model is read from is different, and
the trainer refuses to emit an ONNX file whose predictions drift from the
model those tests validated.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The function's working directory is not guaranteed to be the project root, and
# `services/` and `packages/` are imported as top-level packages.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.api.main import app  # noqa: E402

__all__ = ["app"]

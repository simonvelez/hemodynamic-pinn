"""Evaluate a run: per-severity errors, Δp curve, field plots, inference time.

    python scripts/evaluate.py --run runs/hybrid
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinn.evaluate import evaluate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory containing model.pt")
    ap.add_argument("--data-dir", default=None, help="override the CFD dataset folder")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    evaluate(a.run, a.data_dir, plots=not a.no_plots)

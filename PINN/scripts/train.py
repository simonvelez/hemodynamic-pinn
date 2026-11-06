"""Train the parametric hybrid PINN.  All Config fields are flags, e.g.

    python scripts/train.py --out-dir runs/hybrid --steps 30000
    python scripts/train.py --out-dir runs/physics_only --lambda-data 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinn.config import Config  # noqa: E402
from pinn.train import train  # noqa: E402

if __name__ == "__main__":
    train(Config.from_cli(description=__doc__))

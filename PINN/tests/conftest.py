import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn.config import Config  # noqa: E402

DATA_DIR = Path(Config().data_dir)


@pytest.fixture(scope="session")
def data_dir():
    if not list(DATA_DIR.glob("pinn2d_severity_*.npz")):
        pytest.skip(f"no CFD datasets in {DATA_DIR}")
    return DATA_DIR

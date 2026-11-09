"""Steady solve with a symmetry-breaking imperfection (asym_eps) picks the stable branch:
symmetric below the planar bifurcation, bent jet above it (run in fenicsx-env):

    cd ns_cfd && python -m pytest -q tests
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("dolfinx")

from mesh_2d import build_stenosis_mesh  # noqa: E402
from solver_2d import run_simulation  # noqa: E402


def _asym(severity, u_max, asym_eps):
    dom, ft = build_stenosis_mesh(severity=severity, length=0.05, radius=0.0025, x_center=0.01,
                                  half_length=0.004, nx=125, ny=20, diagonal="crossed")
    _, _, info, _ = run_simulation(dom, ft, u_max=u_max, steady_solve=True, asym_eps=asym_eps, verbose=False)
    return info["asym_history"][-1][1]


def test_stable_case_returns_to_symmetric():
    """50 % at Re≈301 is below the threshold: the imperfection continued to 0 leaves no asymmetry."""
    assert _asym(0.50, 0.2, 0.3) < 1e-8


def test_unstable_case_lands_on_bent_branch():
    """70 % at Re≈301 is above it: the bent jet (centreline |v| ~ 0.4 u_max) is found, while the
    plain continuation returns the unstable symmetric state."""
    assert _asym(0.70, 0.2, 0.3) > 0.1
    assert _asym(0.70, 0.2, 0.0) < 1e-8

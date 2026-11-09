"""CFD verification against exact planar Poiseuille flow (run in fenicsx-env):

    cd ns_cfd && python -m pytest -q tests
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("dolfinx")

from verify_poiseuille import ACCEPT, poiseuille_errors  # noqa: E402


@pytest.mark.parametrize("nx,ny", [(125, 20)])
def test_taylor_hood_reproduces_poiseuille(nx, ny):
    """P2/P1 + gradient-form viscous term contains the exact solution: errors at round-off."""
    r = poiseuille_errors("P2P1-grad", nx, ny)
    for k, tol in ACCEPT.items():
        assert r[k] <= tol, f"{k} = {r[k]:.3e} > {tol:.0e}"
    assert r["u_rel_l2"] < 1e-10 and r["p_rel_l2"] < 1e-10
    assert r["mass_error"] < 1e-10


def test_symmetric_viscous_form_distorts_outlet():
    """Documents the legacy defect: sigma.n = 0 is incompatible with Poiseuille outflow."""
    r = poiseuille_errors("P1P1-sym", 125, 20)
    assert r["max_v_rel"] > 1e-2           # spurious cross-flow near the outlet
    g = poiseuille_errors("P1P1-grad", 125, 20)
    assert g["max_v_rel"] < 0.2 * r["max_v_rel"]

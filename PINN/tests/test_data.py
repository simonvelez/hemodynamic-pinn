import json
import warnings

import numpy as np
import pytest

from pinn.data import INLET, load_cases, sensor_tensors, split_cases
from pinn.geometry import inlet_profile


def test_nondim_round_trip_and_inlet(data_dir):
    cases, sc, geo = load_cases(data_dir)
    c = cases[0]
    raw = np.load(data_dir / f"pinn2d_severity_{round(c.severity * 100)}.npz")
    assert np.allclose(c.xy[:, 0] * sc.D, raw["x"])
    assert np.allclose(c.uvp[:, 0] * sc.U, raw["u"])
    assert np.allclose(c.uvp[:, 2] * sc.P, raw["p"])
    # CFD inlet nodes carry the nondimensional Poiseuille profile used by the PINN loss
    i = c.flag == INLET
    assert np.allclose(c.uvp[i, 0], inlet_profile(c.xy[i, 1], geo), atol=1e-8)
    assert 250 < sc.Re < 350


def test_rejects_bifurcated_cases(data_dir):
    metas = {f.name: json.loads(str(np.load(f)["meta"])) for f in data_dir.glob("pinn2d_severity_*.npz")}
    bad = [m for m in metas.values() if not m["steady"] or m["symmetry_error"] > 0.05]
    if not bad:
        pytest.skip("no bifurcated case in the dataset folder")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cases, _, _ = load_cases(data_dir, max_symmetry_error=0.05)
    kept = {round(c.severity, 4) for c in cases}
    assert all(round(m["geometry"]["severity"], 4) not in kept for m in bad)


def test_split_has_no_leakage(data_dir):
    cases, _, _ = load_cases(data_dir)
    test_s = [cases[len(cases) // 2].severity]
    tr, te = split_cases(cases, test_s)
    assert len(tr) + len(te) == len(cases) and len(te) == 1
    assert not {c.severity for c in tr} & {c.severity for c in te}

    x, y = sensor_tensors(tr, 50, np.random.default_rng(0))
    assert x.shape[1] == 3 and y.shape[1] == 2
    assert not np.isin(np.round(x[:, 2].numpy(), 6), np.round(test_s, 6)).any()

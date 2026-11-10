"""Shared metric helpers (common/metrics.py), guide §7.3 area weighting."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from metrics import node_areas, rel_l2_weighted  # noqa: E402


def test_node_areas_sum_to_lumen_area(data_dir):
    """Lumped nodal areas of a CFD case add up to the lumen area integral of 2 r(x) dx."""
    from pinn.data import load_cases
    from pinn.geometry import wall_radius
    cases, _, geo = load_cases(data_dir)
    c = cases[-1]
    x = np.linspace(0.0, geo.length, 200001)
    exact = np.trapezoid(2 * wall_radius(x, c.severity, geo), x)
    assert abs(node_areas(c.xy, c.triangles).sum() / exact - 1) < 1e-4   # polygonal wall vs cosine


def test_weighted_l2_scale_invariant_and_uniform_limit():
    rng = np.random.default_rng(0)
    ref, pred, w = rng.normal(size=(50, 2)), rng.normal(size=(50, 2)), rng.random(50)
    e = rel_l2_weighted(pred, ref, w)
    assert np.isclose(rel_l2_weighted(3 * pred, 3 * ref, w), e)
    assert np.isclose(rel_l2_weighted(pred, ref, 7 * w), e)
    assert np.isclose(rel_l2_weighted(pred, ref, np.ones(50)), np.linalg.norm(pred - ref) / np.linalg.norm(ref))

import numpy as np
import torch

from pinn.data import WALL, load_cases
from pinn.geometry import sample_interior, sample_line, sample_wall, wall_radius
from pinn.scaling import Geometry

GEO = Geometry(length=10.0, R=0.5, x_center=2.0, half_length=0.8)


def test_wall_radius_matches_cfd_wall_nodes(data_dir):
    cases, _, geo = load_cases(data_dir, max_symmetry_error=1.0)
    for c in cases:
        w = c.flag == WALL
        r = wall_radius(c.xy[w, 0], c.severity, geo)
        assert np.max(np.abs(np.abs(c.xy[w, 1]) - r)) < 1e-9, c.severity


def test_throat_and_healthy_radius():
    assert np.isclose(wall_radius(np.array(2.0), 0.5, GEO), 0.25)
    assert np.isclose(wall_radius(np.array(0.0), 0.5, GEO), 0.5)
    assert np.isclose(wall_radius(np.array(9.0), 0.7, GEO), 0.5)


def test_samples_inside_domain():
    g = torch.Generator().manual_seed(1)
    s_range = (0.2, 0.7)
    pts = sample_interior(5000, GEO, s_range, g, focus_frac=0.3)
    x, y, s = pts.T
    assert (x >= 0).all() and (x <= GEO.length).all()
    assert (s >= 0.2).all() and (s <= 0.7).all()
    assert (y.abs() <= wall_radius(x, s, GEO) + 1e-7).all()

    w = sample_wall(1000, GEO, s_range, g)
    assert torch.allclose(w[:, 1].abs(), wall_radius(w[:, 0], w[:, 2], GEO))

    inl = sample_line(100, 0.0, GEO, s_range, g)
    assert (inl[:, 0] == 0).all() and (inl[:, 1].abs() <= GEO.R).all()

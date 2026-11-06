"""Analytic stenosis geometry and point samplers (nondimensional).

Same cosine profile as ns_cfd/mesh_2d.py:wall_radius, re-implemented so the
PINN does not depend on dolfinx. Works for numpy arrays and torch tensors.
All samplers return inputs as (n, 3) tensors with columns (x*, y*, s).
"""

import math

import numpy as np
import torch


def wall_radius(x, s, geo):
    lib = torch if torch.is_tensor(x) else np
    xi = lib.clip((x - geo.x_center) / geo.half_length, -1.0, 1.0)
    return geo.R * (1.0 - 0.5 * s * (1.0 + lib.cos(math.pi * xi)))


def _uniform(n, lo, hi, gen):
    return lo + (hi - lo) * torch.rand(n, 1, generator=gen)


def sample_interior(n, geo, s_range, gen, focus_frac=0.0):
    """Points inside the lumen; a share is concentrated around throat + wake."""
    s = _uniform(n, *s_range, gen)
    n_focus = int(round(focus_frac * n))
    x = torch.cat([
        _uniform(n - n_focus, 0.0, geo.length, gen),
        _uniform(n_focus, max(geo.x_center - geo.half_length, 0.0),
                 min(geo.x_center + 4.0 * geo.half_length, geo.length), gen),
    ])
    y = wall_radius(x, s, geo) * (2.0 * torch.rand(n, 1, generator=gen) - 1.0)
    return torch.cat([x, y, s], dim=1)


def sample_wall(n, geo, s_range, gen):
    s = _uniform(n, *s_range, gen)
    x = _uniform(n, 0.0, geo.length, gen)
    sign = torch.where(torch.rand(n, 1, generator=gen) < 0.5, -1.0, 1.0)
    return torch.cat([x, sign * wall_radius(x, s, geo), s], dim=1)


def sample_line(n, x_value, geo, s_range, gen):
    """Inlet (x=0) or outlet (x=length) cross-section; both are unconstricted."""
    s = _uniform(n, *s_range, gen)
    y = _uniform(n, -geo.R, geo.R, gen)
    return torch.cat([torch.full((n, 1), float(x_value)), y, s], dim=1)


def inlet_profile(y, geo):
    """Nondimensional Poiseuille inlet: u* = 1 - (y/R)^2 (peak 1 = U)."""
    return 1.0 - (y / geo.R) ** 2

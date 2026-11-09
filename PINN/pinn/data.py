"""Load ns_cfd 2D datasets (pinn2d_severity_XX.npz) in nondimensional form."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .scaling import Geometry, Scales

# Node flags written by ns_cfd/generate_dataset_2d.py
INTERIOR, INLET, OUTLET, WALL = 0, 1, 2, 3


@dataclass
class Case:
    severity: float
    xy: np.ndarray         # (N, 2) nondimensional coordinates
    uvp: np.ndarray        # (N, 3) nondimensional u, v, p
    flag: np.ndarray       # (N,)
    triangles: np.ndarray  # (M, 3)
    meta: dict
    wall_idx: np.ndarray = None   # node indices of the wall (datasets with WSS only)
    tau_w: np.ndarray = None      # nondimensional WSS tau / (rho U^2) at wall_idx, signed

    def inputs(self):
        """(N, 3) tensor of (x*, y*, s) in torch's default dtype."""
        s = np.full((len(self.xy), 1), self.severity)
        return torch.tensor(np.hstack([self.xy, s]), dtype=torch.get_default_dtype())

    def dp(self):
        """Nondimensional pressure drop: mean p on inlet nodes - mean p on outlet nodes."""
        return self.uvp[self.flag == INLET, 2].mean() - self.uvp[self.flag == OUTLET, 2].mean()


def load_cases(data_dir, max_symmetry_error=0.05):
    """Return (cases sorted by severity, Scales, Geometry). Rejects unusable cases."""
    files = sorted(Path(data_dir).glob("pinn2d_severity_*.npz"))
    if not files:
        raise FileNotFoundError(f"no pinn2d_severity_*.npz in {data_dir} - run ns_cfd/generate_dataset_2d.py")

    cases, scales, geo = [], None, None
    for f in files:
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        sev = float(meta["geometry"]["severity"])
        # Only stable symmetric flows are used: cases past the planar symmetry-breaking
        # bifurcation (branch "asymmetric", bent jet) are CFD-only and never reach the PINN.
        if meta.get("branch") == "asymmetric":
            warnings.warn(f"skipping {f.name}: asymmetric (bent-jet) branch, outside the symmetric range")
            continue
        if not meta["steady"] or meta["symmetry_error"] > max_symmetry_error:
            warnings.warn(f"skipping {f.name}: steady={meta['steady']}, "
                          f"symmetry_error={meta['symmetry_error']:.2e} (bifurcated / not converged)")
            continue

        sc = Scales.from_meta(meta)
        g = Geometry.from_meta(meta, sc)
        if scales is None:
            scales, geo = sc, g
        elif sc != scales or g != geo:
            raise ValueError(f"{f.name}: physical scales/geometry differ from other cases; "
                             "all cases must share everything except severity")

        xy = np.column_stack([d["x"], d["y"]]) / sc.D
        uvp = np.column_stack([d["u"] / sc.U, d["v"] / sc.U, d["p"] / sc.P])
        wall_idx = d["wall_idx"] if "wall_idx" in d else None
        tau_w = d["tau_w"] / sc.P if "tau_w" in d else None
        cases.append(Case(sev, xy, uvp, d["flag"], d["triangles"], meta, wall_idx, tau_w))

    if not cases:
        raise RuntimeError("all CFD cases were rejected")
    cases.sort(key=lambda c: c.severity)
    return cases, scales, geo


def split_cases(cases, test_severities, tol=1e-6, data_s_range=None):
    """(train, test). A case is test if its severity is in test_severities, or, when
    data_s_range = (lo, hi) is given, if it lies outside [lo, hi] (extrapolation)."""
    def is_test(c):
        if any(abs(c.severity - t) < tol for t in test_severities):
            return True
        return bool(data_s_range) and not (data_s_range[0] - tol <= c.severity <= data_s_range[1] + tol)
    return [c for c in cases if not is_test(c)], [c for c in cases if is_test(c)]


def sensor_tensors(cases, n_sensors, rng, with_pressure=False, noise=0.0, noise_rng=None):
    """Random interior CFD nodes per case -> (inputs (n,3), targets (n,2|3)).

    noise: std of additive Gaussian noise on the targets, as a fraction of the
    velocity scale U (targets are nondimensional, so U = 1; pressure targets get the
    same std in units of rho U^2). Drawn from noise_rng (default: rng)."""
    xs, ys = [], []
    for c in cases:
        idx = np.flatnonzero(c.flag == INTERIOR)
        idx = rng.choice(idx, size=min(n_sensors, len(idx)), replace=False)
        y = c.uvp[idx, : 3 if with_pressure else 2].copy()
        if noise > 0:
            y += noise * (noise_rng or rng).standard_normal(y.shape)
        xs.append(c.inputs()[idx])
        ys.append(torch.tensor(y, dtype=torch.get_default_dtype()))
    return torch.cat(xs), torch.cat(ys)

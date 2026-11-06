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

    def inputs(self):
        """(N, 3) float32 tensor of (x*, y*, s)."""
        s = np.full((len(self.xy), 1), self.severity)
        return torch.tensor(np.hstack([self.xy, s]), dtype=torch.float32)

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
        cases.append(Case(sev, xy, uvp, d["flag"], d["triangles"], meta))

    if not cases:
        raise RuntimeError("all CFD cases were rejected")
    cases.sort(key=lambda c: c.severity)
    return cases, scales, geo


def split_cases(cases, test_severities, tol=1e-6):
    is_test = lambda c: any(abs(c.severity - t) < tol for t in test_severities)
    return [c for c in cases if not is_test(c)], [c for c in cases if is_test(c)]


def sensor_tensors(cases, n_sensors, rng, with_pressure=False):
    """Random interior CFD nodes per case -> (inputs (n,3), targets (n,2|3))."""
    xs, ys = [], []
    for c in cases:
        idx = np.flatnonzero(c.flag == INTERIOR)
        idx = rng.choice(idx, size=min(n_sensors, len(idx)), replace=False)
        xs.append(c.inputs()[idx])
        ys.append(torch.tensor(c.uvp[idx, : 3 if with_pressure else 2], dtype=torch.float32))
    return torch.cat(xs), torch.cat(ys)

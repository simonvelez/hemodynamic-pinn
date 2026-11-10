"""Error metrics shared by the CFD cost study (ns_cfd/) and the PINN evaluation (PINN/).

Guide §7.3: relative L2 errors should be weighted by cell area, because a plain node sum
over-weights the regions where the mesh is finest (here the throat, where the warped cells are
smallest). The nodal weight is the lumped P1 mass: a third of the area of each adjacent triangle,
so sum(w) is the domain area and sum(w f^2) is the trapezoidal-rule integral of f^2.
"""

import numpy as np


def node_areas(xy, triangles):
    """Lumped P1 nodal areas for a triangulation (xy: (N, 2), triangles: (M, 3) node indices)."""
    xy = np.asarray(xy, dtype=float)
    t = np.asarray(triangles)
    a, b, c = xy[t[:, 0]], xy[t[:, 1]], xy[t[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    w = np.zeros(len(xy))
    np.add.at(w, t.ravel(), np.repeat(area / 3.0, 3))
    return w


def rel_l2_weighted(pred, ref, w):
    """sqrt(sum w |pred - ref|^2) / sqrt(sum w |ref|^2); pred/ref (N,) or (N, k) (vector norm per node)."""
    pred, ref, w = np.asarray(pred, float), np.asarray(ref, float), np.asarray(w, float)
    if pred.ndim == 1:
        pred, ref = pred[:, None], ref[:, None]
    num = np.sum(w * np.sum((pred - ref) ** 2, axis=1))
    den = np.sum(w * np.sum(ref ** 2, axis=1))
    return float(np.sqrt(num / max(den, 1e-300)))

"""Wall shear stress (WSS) and recirculation from a CFD solution (DOLFINx).

WSS on the stenosed channel walls, signed in the flow direction:

    tau_w = mu * t . (grad u) . n_in

with n_in the unit normal pointing INTO the fluid and t the unit tangent with a
positive x-component. On a no-slip wall the u_n contribution vanishes, so this
equals the tangential viscous traction. Positive = forward flow at the wall,
negative = reversed flow (recirculation). For Poiseuille flow tau_w = 2 mu U / R.

The velocity gradient is evaluated element-wise from the finite-element solution
(DG_{k-1} for P_k velocity) and averaged over the cells that touch each wall node,
so no finite differencing of interpolated nodal data is involved (guide §5.3 item 7).
The wall normal is analytic, from the same cosine profile as mesh_2d.wall_radius.

Separation and reattachment are the sign changes of tau_w on the upper wall
downstream of the throat; L_recirc = x_reattach - x_separation.
"""

import numpy as np
import ufl
from dolfinx import fem, geometry


def wall_slope(x, severity, radius, x_center, half_length):
    """dr/dx of the cosine stenosis r(x) (mesh_2d.wall_radius)."""
    xi = (x - x_center) / half_length
    inside = np.abs(xi) <= 1.0
    return np.where(inside, radius * severity * np.pi / (2.0 * half_length) * np.sin(np.pi * xi), 0.0)


def _cell_average_eval(fn, pts, domain):
    """Evaluate fn at pts, averaging over every cell that contains the point
    (wall nodes sit on several cells, and DG fields are discontinuous there)."""
    tdim = domain.topology.dim
    tree = geometry.bb_tree(domain, tdim)
    p3 = np.zeros((len(pts), 3))
    p3[:, :2] = pts
    cands = geometry.compute_collisions_points(tree, p3)
    coll = geometry.compute_colliding_cells(domain, cands, p3)
    out = None
    for i in range(len(pts)):
        cells = coll.links(i)
        if len(cells) == 0:
            raise RuntimeError(f"wall point {pts[i]} not found in the mesh")
        vals = fn.eval(np.repeat(p3[i:i + 1], len(cells), axis=0), cells)
        avg = vals.mean(axis=0)
        if out is None:
            out = np.zeros((len(pts), avg.size))
        out[i] = avg
    return out


def wall_shear_stress(domain, u_h, mu, wall_xy, geo):
    """tau_w [Pa] at wall points wall_xy (N, 2). geo: severity, radius, x_center, half_length."""
    k = u_h.function_space.ufl_element().embedded_superdegree
    T = fem.functionspace(domain, ("DG", max(k - 1, 0), (2, 2)))
    G = fem.Function(T)
    G.interpolate(fem.Expression(ufl.grad(u_h), T.element.interpolation_points))
    g = _cell_average_eval(G, wall_xy, domain).reshape(-1, 2, 2)   # g[:, i, j] = d u_i / d x_j

    x, y = wall_xy[:, 0], wall_xy[:, 1]
    rp = wall_slope(x, geo["severity"], geo["radius"], geo["x_center"], geo["half_length"])
    sgn = np.sign(y)                          # +1 upper wall, -1 lower wall
    # upper wall y = r(x): fluid where r - y > 0 -> n_in ∝ (r', -1); lower wall y = -r(x): n_in ∝ (r', +1)
    n_in = np.stack([rp, -sgn], axis=1)
    t = np.stack([np.ones_like(rp), sgn * rp], axis=1)
    n_in /= np.linalg.norm(n_in, axis=1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return mu * np.einsum("ni,nij,nj->n", t, g, n_in)


def recirculation_from_wss(x, tau, x_from):
    """(x_separation, x_reattachment) [same units as x] from sign changes of tau(x) on one
    wall for x >= x_from, or (nan, nan) if the flow never reverses."""
    o = np.argsort(x)
    xs, ts = x[o], tau[o]
    m = xs >= x_from
    xs, ts = xs[m], ts[m]
    neg = np.flatnonzero(ts < 0)
    if len(neg) == 0:
        return np.nan, np.nan

    def crossing(i):
        return xs[i - 1] - ts[i - 1] * (xs[i] - xs[i - 1]) / (ts[i] - ts[i - 1])

    i_sep = neg[0]
    x_sep = crossing(i_sep) if i_sep > 0 else xs[0]
    pos_after = np.flatnonzero((ts >= 0) & (np.arange(len(ts)) > i_sep))
    x_re = crossing(pos_after[0]) if len(pos_after) else np.nan
    return float(x_sep), float(x_re)


def wall_qois(x_wall, y_wall, tau, x_center):
    """Upper-wall QoIs: peak |WSS| and its x, separation, reattachment, recirculation length."""
    up = y_wall > 0
    xu, tu = x_wall[up], tau[up]
    i = int(np.argmax(np.abs(tu)))
    x_sep, x_re = recirculation_from_wss(xu, tu, x_center)
    return dict(wss_peak_pa=float(tu[i]), wss_peak_x=float(xu[i]),
                x_sep=x_sep, x_reatt=x_re, recirc_len=float(x_re - x_sep))

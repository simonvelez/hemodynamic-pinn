"""Stability check of the labmate's 75 % stenosis geometry (Zuluaga 2026, thesis §3.4 / §3.8.1).

His planar geometry, built with our mirror-symmetric `crossed` mesh and solver (P2/P1):
    linear taper R_in = 1.57 mm -> R_out = 1.20 mm over L = 138 mm, lesion centre x_m = 30 mm,
    R_min = (1 - eta) R_taper(x_m), half-length d = (R_taper(x_m) - R_min) / alpha,
    cubic Bezier wall (radial h(t) = h_sten (3t^2 - 2t^3), axial control points with tension 0.5).
Flow: his target Q = v_mean * h = 280 mm/s * 2.41 mm = 674.8 mm^2/s (per unit depth; Re ≈ 204 on
distal height + mean velocity), imposed here as a parabolic inlet with the same Q (his inlet is
pressure-driven and his 75 % cases lose 11-22 % of the flow: use --q-scale 0.78 for that bound).

    python zuluaga_check.py steady --alpha 0.3                 # symmetric state vs bent branch (Newton)
    python zuluaga_check.py stability --alpha 0.3              # perturbed transient: does a(t) grow?
    python zuluaga_check.py stability --perturb-eps 0          # his procedure: from rest, no perturbation
    python zuluaga_check.py report

Outputs in ./zuluaga_check/.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
from dolfinx import mesh

from mesh_2d import INLET_TAG, OUTLET_TAG, WALL_TAG

OUT = Path(__file__).resolve().parent / "zuluaga_check"
RHO, MU = 1060.0, 0.0035                   # his values
L, R_IN, R_OUT, X_M, TENSION = 0.138, 0.00157, 0.00120, 0.030, 0.5
Q_TARGET = 0.280 * 0.00241                 # m^2/s per unit depth


def r_taper(x):
    return R_IN + (R_OUT - R_IN) * x / L


def wall_radius(x, eta, alpha):
    """His cubic-Bezier stenosis on the linear taper (Bezier inverted numerically in t)."""
    x = np.asarray(x, dtype=float)
    h_sten = eta * r_taper(X_M)
    d = h_sten / alpha
    r = r_taper(x)
    t = np.linspace(0.0, 1.0, 4001)
    # proximal half: x0 = x_m - d, x1 = x0 + tau d, x2 = x_m - tau d, x3 = x_m (monotone for tau <= 0.5)
    x0, x1, x2, x3 = X_M - d, X_M - d + TENSION * d, X_M - TENSION * d, X_M
    xt = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * x1 + 3 * (1 - t) * t ** 2 * x2 + t ** 3 * x3
    ht = h_sten * (3 * t ** 2 - 2 * t ** 3)
    dist = np.abs(x - X_M)
    inside = dist < d
    h = np.interp(X_M - dist[inside], xt, ht)      # distal half is the mirror image about x_m
    r[inside] -= h
    return r


def build_mesh(eta, alpha, nx, ny):
    dom = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, -R_IN]), np.array([L, R_IN])], [nx, ny],
                                cell_type=mesh.CellType.triangle, diagonal=mesh.DiagonalType.crossed)
    x = dom.geometry.x
    x[:, 1] *= wall_radius(x[:, 0], eta, alpha) / R_IN
    fdim = dom.topology.dim - 1
    dom.topology.create_connectivity(fdim, dom.topology.dim)
    tol = 1e-10 * L
    inlet = mesh.locate_entities_boundary(dom, fdim, lambda p: np.isclose(p[0], 0.0, atol=tol))
    outlet = mesh.locate_entities_boundary(dom, fdim, lambda p: np.isclose(p[0], L, atol=tol))
    wall = np.setdiff1d(mesh.exterior_facet_indices(dom.topology), np.union1d(inlet, outlet))
    facets = np.concatenate([inlet, outlet, wall]).astype(np.int32)
    vals = np.concatenate([np.full(len(inlet), INLET_TAG), np.full(len(outlet), OUTLET_TAG),
                           np.full(len(wall), WALL_TAG)]).astype(np.int32)
    o = np.argsort(facets)
    return dom, mesh.meshtags(dom, fdim, facets[o], vals[o])


def u_max_for(q_scale):
    # planar parabola on the inlet height 2 R_IN: Q = (2/3) u_max * 2 R_IN
    return 1.5 * q_scale * Q_TARGET / (2 * R_IN)


def tag(a):
    return f"eta{round(a.eta * 100)}_a{a.alpha:g}_q{a.q_scale:g}_{a.nx}x{a.ny}"


def save(a, kind, dom, ft, u_h, p_h, info):
    from generate_dataset_2d import nodal_arrays, plot_case, symmetry_error
    xy, uv, p, flag, tri = nodal_arrays(dom, ft, u_h, p_h)
    q_in = info["flux_in"]
    info.update(kind=kind, args=vars(a), symmetry_error=symmetry_error(xy, uv),
                Re_mean_distal=RHO * (q_in / (2 * R_OUT)) * (2 * R_OUT) / MU,
                Re_mean_inlet=RHO * q_in / MU)
    OUT.mkdir(exist_ok=True)
    name = f"{kind}_{tag(a)}"
    np.savez_compressed(OUT / f"{name}.npz", x=xy[:, 0], y=xy[:, 1], u=uv[:, 0], v=uv[:, 1], p=p,
                        triangles=tri, meta=json.dumps(info, default=float))
    plot_case(OUT / f"{name}.png", xy, uv, p, tri, a.eta)
    return info


def cmd_steady(a):
    from solver_2d import run_simulation
    dom, ft = build_mesh(a.eta, a.alpha, a.nx, a.ny)
    for eps in (0.0, 0.3):
        t0 = time.time()
        u_h, p_h, info, _ = run_simulation(dom, ft, rho=RHO, mu=MU, u_max=u_max_for(a.q_scale),
                                           steady_solve=True, asym_eps=eps, asym_x=(X_M, X_M + 0.015),
                                           verbose=a.verbose)
        info = save(a, f"steady_eps{eps:g}", dom, ft, u_h, p_h, info)
        print(f"[steady] {tag(a)} asym_eps={eps}: centreline asym={info['asym_history'][-1][1]:.3e} "
              f"sym_err={info['symmetry_error']:.2e} dp={info['dp']:.1f} Pa Re_mean(distal h)="
              f"{info['Re_mean_distal']:.0f} ({time.time() - t0:.0f} s)", flush=True)


def cmd_stability(a):
    from solver_2d import run_simulation
    from stability_test import EPS, MIN_TIME, PERTURB_UNTIL, growth_rate, verdict
    dom, ft = build_mesh(a.eta, a.alpha, a.nx, a.ny)
    t0 = time.time()
    eps = EPS if a.perturb_eps is None else a.perturb_eps
    u_h, p_h, info, _ = run_simulation(dom, ft, rho=RHO, mu=MU, u_max=u_max_for(a.q_scale), dt=a.dt,
                                       t_final=a.t_final, perturb_eps=eps, perturb_until=PERTURB_UNTIL,
                                       min_time=MIN_TIME, verbose=a.verbose)
    sigma, npts = growth_rate(info["asym_history"], PERTURB_UNTIL + 0.3)
    a_final = info["asym_history"][-1][1]
    info.update(sigma=sigma, fit_points=npts, a_final=a_final,
                a_max=max(v for _, v in info["asym_history"]), verdict=verdict(sigma, a_final),
                wall_time_s=time.time() - t0)
    info = save(a, "stability" if eps else "unperturbed", dom, ft, u_h, p_h, info)
    print(f"[stability] {tag(a)}: sigma={sigma:+.3f} 1/s a_final={a_final:.2e} "
          f"sym_err={info['symmetry_error']:.2e} t_end={info['t_end']:.2f}s -> {info['verdict']} "
          f"({info['wall_time_s'] / 60:.0f} min)", flush=True)


def cmd_report(_):
    rows = []
    for f in sorted(OUT.glob("*.npz")):
        m = json.loads(str(np.load(f)["meta"]))
        rows.append(f"| {f.stem} | {m['asym_history'][-1][1]:.3e} | {m['symmetry_error']:.2e} | "
                    f"{m['dp']:.1f} | {m['Re_mean_distal']:.0f} | {m.get('sigma', float('nan')):+.3f} | "
                    f"{m.get('verdict', '-')} |")
    text = ("| run | centreline asym | symmetry err | Δp (Pa) | Re_mean (distal h) | σ (1/s) | verdict |\n"
            "|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    print(text)
    (OUT / "report.md").write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("steady", "stability"):
        s = sub.add_parser(name)
        s.add_argument("--eta", type=float, default=0.75)
        s.add_argument("--alpha", type=float, default=0.3)
        s.add_argument("--q-scale", type=float, default=1.0)
        s.add_argument("--nx", type=int, default=690)      # 0.2 mm cells
        s.add_argument("--ny", type=int, default=24)
        s.add_argument("--dt", type=float, default=0.002)
        s.add_argument("--t-final", type=float, default=6.0)
        s.add_argument("--perturb-eps", type=float, default=None,
                       help="stability: inlet perturbation amplitude (default stability_test.EPS; 0 = his "
                            "procedure: start from rest, no perturbation)")
        s.add_argument("--verbose", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args()
    dict(steady=cmd_steady, stability=cmd_stability, report=cmd_report)[a.cmd](a)


if __name__ == "__main__":
    main()

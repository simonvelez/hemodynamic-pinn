"""Verify the CFD solver against exact planar Poiseuille flow (straight channel, s = 0).

With severity 0 the domain is a straight channel and the inlet profile is already the
fully developed one, so the exact solution holds everywhere:

    u = u_max (1 - (y/R)^2),  v = 0,  p = (2 mu u_max / R^2) (L - x)     (p = 0 at the outlet)

Any deviation is discretisation or boundary-condition error. The check separates two
known defects of the original scheme (P1/P1 + PSPG, symmetric viscous form):
  * outlet:  2 mu eps(u):eps(v) has the natural condition sigma.n = 0, which Poiseuille
             flow violates (wall shear at the outlet) -> spurious v, non-uniform outlet p;
  * PSPG:    the P1 strong residual has no viscous term -> pressure error at inlet/outlet.

Usage (conda env fenicsx-env):
    python verify_poiseuille.py                 # all variants, 125x20 and 250x40 (steady solves)
    python verify_poiseuille.py --transient --out verification/poiseuille_transient.json
    python verify_poiseuille.py --variants P2P1-grad --meshes 125x20
"""

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from mesh_2d import build_stenosis_mesh
from solver_2d import run_simulation
from generate_dataset_2d import nodal_arrays, INLET, OUTLET

RHO, MU, U_MAX, R, L = 1055.0, 0.0035, 0.2, 0.0025, 0.05

VARIANTS = {
    "P1P1-sym": dict(element="P1P1", viscous_form="sym"),     # original scheme
    "P1P1-grad": dict(element="P1P1", viscous_form="grad"),   # outlet fixed, PSPG kept
    "P2P1-grad": dict(element="P2P1", viscous_form="grad"),   # new default
}

# Acceptance for the new default (P2P1-grad); the P1 variants are reported, not judged.
ACCEPT = dict(dp_rel=1e-3, outlet_p_spread_Pa=1e-3, max_v_rel=1e-4)


def poiseuille_errors(variant, nx, ny, u_max=U_MAX, steady_solve=True, t_final=1.0):
    dom, ft = build_stenosis_mesh(severity=0.0, length=L, radius=R, nx=nx, ny=ny)
    t0 = time.time()
    u_h, p_h, info, _ = run_simulation(dom, ft, rho=RHO, mu=MU, u_max=u_max, t_final=t_final,
                                       steady_solve=steady_solve, verbose=False, **VARIANTS[variant])
    wall = time.time() - t0
    xy, uv, p, flag, _ = nodal_arrays(dom, ft, u_h, p_h)
    x, y = xy[:, 0], xy[:, 1]
    G = 2.0 * MU * u_max / R**2                 # -dp/dx
    ue, pe = u_max * (1.0 - (y / R) ** 2), G * (L - x)
    c = np.abs(y) < 1e-12
    mid = c & (x > 0.2 * L) & (x < 0.8 * L)
    return dict(
        variant=variant, mesh=f"{nx}x{ny}", solve_mode=info["solve_mode"], steady=bool(info["steady"]),
        wall_s=wall, n_dofs=info["n_dofs"],
        u_rel_l2=float(np.linalg.norm(uv[:, 0] - ue) / np.linalg.norm(ue)),
        max_v_rel=float(np.abs(uv[:, 1]).max() / u_max),
        p_rel_l2=float(np.linalg.norm(p - pe) / np.linalg.norm(pe)),
        p_max_err_Pa=float(np.abs(p - pe).max()),
        dp_Pa=float(info["dp"]), dp_exact_Pa=G * L,
        dp_rel=float(abs(info["dp"] - G * L) / (G * L)),
        dpdx_rel=float(abs(np.polyfit(x[mid], p[mid], 1)[0] + G) / G),
        outlet_p_spread_Pa=float(np.ptp(p[flag == OUTLET])),
        inlet_p_spread_Pa=float(np.ptp(p[flag == INLET])),
        mass_error=float(info["mass_error"]),
    )


def passes(r):
    return all(r[k] <= v for k, v in ACCEPT.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--meshes", nargs="+", default=["125x20", "250x40"])
    ap.add_argument("--transient", action="store_true",
                    help="time-march to steady state instead of the direct steady solve")
    ap.add_argument("--out", default="verification/poiseuille.json")
    args = ap.parse_args()

    rows = []
    for v in args.variants:
        for m in args.meshes:
            nx, ny = map(int, m.split("x"))
            r = poiseuille_errors(v, nx, ny, steady_solve=not args.transient)
            r["accept"] = passes(r) if v == "P2P1-grad" else None
            rows.append(r)
            print(f"{v:10s} {m:7s} u L2={r['u_rel_l2']:.2e}  max|v|/U={r['max_v_rel']:.2e}  "
                  f"p L2={r['p_rel_l2']:.2e}  max|p err|={r['p_max_err_Pa']:.3f} Pa  "
                  f"dp={r['dp_Pa']:.4f}/{r['dp_exact_Pa']:.4f} ({r['dp_rel']:.2e})  "
                  f"outlet p spread={r['outlet_p_spread_Pa']:.2e} Pa  {r['wall_s']:.0f}s"
                  + ("" if r["accept"] is None else f"  -> {'PASS' if r['accept'] else 'FAIL'}"), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(acceptance=ACCEPT, rows=rows), f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

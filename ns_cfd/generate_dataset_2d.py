"""2D CFD -> PINN dataset pipeline.

    conda activate fenicsx-env
    python generate_dataset_2d.py                      # 30-70 % severities (literature range)
    python generate_dataset_2d.py --severities 0.5 --nx 300 --ny 48

For each severity writes to ./output_2d/:
    pinn2d_severity_XX.npz   nodal x, y, u, v, p, boundary flag, triangles, wall shear stress
                             (wall_idx, tau_w), metadata (incl. qoi: dp, peak WSS, recirculation)
    pinn2d_severity_XX_run_manifest.json / _cost.json   provenance and stage timings
    pinn2d_severity_XX.png   velocity-magnitude and pressure fields
    pinn2d_severity_XX_mesh.png  mesh troubleshooting sheet (tags, quality, spacing)

Runs in serial (the meshes are small; the export relies on serial dof ordering).
"""

import argparse
import json
import os
import sys

# Threaded BLAS inside MUMPS oversubscribes the CPU on these small systems
# (observed ~9x CPU time vs wall time); default to 1 thread unless overridden.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from mpi4py import MPI
from dolfinx import fem

from mesh_2d import INLET_TAG, OUTLET_TAG, WALL_TAG, build_stenosis_mesh
from mesh_plots import mesh_arrays, plot_mesh
from solver_2d import run_simulation
from wall_shear import wall_qois, wall_shear_stress

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from manifest import RunLog  # noqa: E402

# Node flags stored in the dataset
INTERIOR, INLET, OUTLET, WALL = 0, 1, 2, 3


def nodal_arrays(domain, facet_tags, u_h, p_h):
    """Interpolate (u, p) to a common P1 layout and return node-ordered arrays.

    Velocity and pressure dofs in a mixed space do NOT share ordering with
    domain.geometry.x, so we interpolate both into plain P1 spaces built on the
    same mesh, whose dofmaps coincide, and read coordinates from that dofmap.
    """
    S = fem.functionspace(domain, ("Lagrange", 1))
    Vv = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    assert np.allclose(S.tabulate_dof_coordinates(), Vv.tabulate_dof_coordinates())

    u_p1, p_p1 = fem.Function(Vv), fem.Function(S)
    u_p1.interpolate(u_h)
    p_p1.interpolate(p_h)

    xy = S.tabulate_dof_coordinates()[:, :2]
    uv = u_p1.x.array.reshape(-1, 2)
    flag = np.full(len(xy), INTERIOR, dtype=np.int8)
    fdim = domain.topology.dim - 1
    for tag, val in ((WALL_TAG, WALL), (INLET_TAG, INLET), (OUTLET_TAG, OUTLET)):  # inlet/outlet win at corners
        flag[fem.locate_dofs_topological(S, fdim, facet_tags.find(tag))] = val
    triangles = S.dofmap.list.copy()
    return xy, uv, p_p1.x.array.copy(), flag, triangles


def symmetry_error(xy, uv):
    """max |u(x,y) - u(x,-y)| / max|u| (0 for a symmetric flow, grows if it bifurcates)."""
    key = lambda x, y: (round(x, 9), round(y, 9))
    lut = {key(x, y): i for i, (x, y) in enumerate(xy)}
    mirror = np.array([lut.get(key(x, -y), -1) for x, y in xy])
    ok = mirror >= 0
    du = np.abs(uv[ok, 0] - uv[mirror[ok], 0])
    return float(du.max() / np.abs(uv[:, 0]).max())


def plot_case(path, xy, uv, p, triangles, severity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import TwoSlopeNorm

    tri = mtri.Triangulation(xy[:, 0] * 1e3, xy[:, 1] * 1e3, triangles)
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.5), dpi=150, sharex=True)
    # Gauge pressure (outlet = 0): diverging map centred at 0, so the low-pressure
    # zone behind the throat (blue) is visibly below the outlet pressure.
    p_norm = TwoSlopeNorm(0.0, min(p.min(), -1e-9), max(p.max(), 1e-9))
    for ax, field, label, cmap, kw in (
        (axes[0], np.linalg.norm(uv, axis=1), "|u| (m/s)", "inferno", {}),
        (axes[1], p, "gauge p (Pa, outlet = 0)", "RdBu_r", dict(norm=p_norm)),
    ):
        c = ax.tripcolor(tri, field, shading="gouraud", cmap=cmap, **kw)
        fig.colorbar(c, ax=ax, label=label, pad=0.01)
        ax.set_aspect("equal")
        ax.set_ylabel("y (mm)")
    axes[0].set_title(f"2D stenosis, {int(round(severity * 100))}% diameter reduction")
    axes[1].set_xlabel("x (mm)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


SYMMETRIC_TOL = 1e-6   # symmetry_error below this -> symmetric branch


def run_case(severity, args):
    tag = f"severity_{int(round(severity * 100))}"
    rlog = RunLog(args.output_dir, kind="cfd", prefix=f"pinn2d_{tag}_",
                  config=dict(vars(args), severity=severity))
    with rlog.stage("mesh"):
        domain, facet_tags = build_stenosis_mesh(
            severity=severity, length=args.length, radius=args.radius,
            x_center=args.x_center, half_length=args.half_length, nx=args.nx, ny=args.ny,
            diagonal=args.diagonal,
        )
    # Mesh troubleshooting sheet, written before solving so it exists even if the solve fails.
    with rlog.stage("plot"):
        xy0, tri0, flag0 = mesh_arrays(domain, facet_tags)
        geo = dict(severity=severity, length=args.length, radius=args.radius,
                   x_center=args.x_center, half_length=args.half_length)
        mstats = plot_mesh(os.path.join(args.output_dir, f"pinn2d_{tag}_mesh.png"), xy0, tri0, flag0, geo,
                           f"Mesh {args.nx}x{args.ny} ({args.diagonal}), severity {severity:.0%}")
    with rlog.stage("solve"):
        u_h, p_h, info, snapshots = run_simulation(
            domain, facet_tags, u_max=args.u_max, dt=args.dt, t_final=args.t_final,
            steady_tol=args.steady_tol, snapshot_every=args.snapshot_every,
            element=args.element, viscous_form=args.viscous_form,
            steady_solve=(args.solve == "steady"),
            asym_eps=args.asym_eps if args.solve == "steady" else 0.0,
        )
    rlog.counters.update(n_steps=info["steps"], n_dofs=info.get("n_dofs"), t_end=info["t_end"],
                         steady=info["steady"])

    with rlog.stage("export"):
        xy, uv, p, flag, triangles = nodal_arrays(domain, facet_tags, u_h, p_h)
    info["symmetry_error"] = symmetry_error(xy, uv)
    # Stable symmetric solutions are mirror-symmetric to round-off (~1e-12); the bent-jet
    # branch (planar Coanda bifurcation) is O(1e-2) or more. PINN training uses only "symmetric".
    info["branch"] = "symmetric" if info["symmetry_error"] < SYMMETRIC_TOL else "asymmetric"
    info["u_max_field"] = float(np.linalg.norm(uv, axis=1).max())
    info["geometry"] = dict(severity=severity, length=args.length, radius=args.radius,
                            x_center=args.x_center, half_length=args.half_length,
                            nx=args.nx, ny=args.ny, diagonal=args.diagonal)
    info["mesh"] = mstats

    # Wall shear stress at the wall nodes (signed; Pa) and upper-wall QoIs.
    with rlog.stage("post"):
        wall_idx = np.flatnonzero(flag == WALL)
        tau_w = wall_shear_stress(domain, u_h, info["params"]["mu"], xy[wall_idx],
                                  dict(severity=severity, radius=args.radius, x_center=args.x_center,
                                       half_length=args.half_length))
        info["qoi"] = dict(dp_pa=float(info["dp"]), **wall_qois(xy[wall_idx, 0], xy[wall_idx, 1], tau_w,
                                                                args.x_center))

    data = dict(x=xy[:, 0], y=xy[:, 1], u=uv[:, 0], v=uv[:, 1], p=p,
                flag=flag, triangles=triangles, wall_idx=wall_idx, tau_w=tau_w,
                meta=json.dumps(info, default=float))

    snaps = list(snapshots())
    if snaps:
        fields = [nodal_arrays(domain, facet_tags, us, ps) for _, us, ps in snaps]
        data["t_snap"] = np.array([ts for ts, _, _ in snaps])
        data["u_snap"] = np.stack([f[1][:, 0] for f in fields])
        data["v_snap"] = np.stack([f[1][:, 1] for f in fields])
        data["p_snap"] = np.stack([f[2] for f in fields])

    with rlog.stage("export"):
        np.savez_compressed(os.path.join(args.output_dir, f"pinn2d_{tag}.npz"), **data)
    with rlog.stage("plot"):
        plot_case(os.path.join(args.output_dir, f"pinn2d_{tag}.png"), xy, uv, p, triangles, severity)
    rlog.finish(metrics=dict(mass_error=info["mass_error"], symmetry_error=info["symmetry_error"],
                             **info["qoi"]))

    print(f"[dataset] {tag}: {len(xy)} nodes | steady={info['steady']} t_end={info['t_end']:.3f}s | "
          f"mass err={info['mass_error']:.1e} | dp={info['dp']:.2f} Pa | "
          f"max|u|={info['u_max_field']:.3f} m/s | symmetry err={info['symmetry_error']:.1e} | "
          f"branch={info['branch']} | peak WSS={info['qoi']['wss_peak_pa']:.2f} Pa | L_recirc={info['qoi']['recirc_len'] * 1e3:.2f} mm",
          flush=True)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--severities", type=float, nargs="+",
                    default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    ap.add_argument("--output-dir", default="./output_2d")
    ap.add_argument("--length", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=0.0025)
    ap.add_argument("--x-center", type=float, default=0.01)
    ap.add_argument("--half-length", type=float, default=0.004)
    ap.add_argument("--nx", type=int, default=250)
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--diagonal", default="crossed", help="triangle split: crossed (symmetric) | right | left")
    ap.add_argument("--element", default="P2P1", choices=["P2P1", "P1P1"],
                    help="P2P1 Taylor-Hood (default) | P1P1 + SUPG/PSPG/LSIC (legacy, cheaper)")
    ap.add_argument("--viscous-form", default="grad", choices=["grad", "sym"],
                    help="grad: Poiseuille-compatible do-nothing outlet (default) | sym: legacy")
    ap.add_argument("--solve", default="steady", choices=["steady", "transient"],
                    help="steady: direct Newton + continuation (default, seconds); transient: time-march "
                         "to steady state (needed for --snapshot-every; stays on the symmetric branch on the "
                         "symmetric mesh, so check stability_test.py above the bifurcation)")
    ap.add_argument("--asym-eps", type=float, default=0.3,
                    help="steady solve: symmetry-breaking imperfection continued to 0, so the result is the "
                         "physically stable steady state (bent jet above the bifurcation, symmetric below). "
                         "0 = plain symmetric continuation (returns the unstable symmetric state above it)")
    ap.add_argument("--u-max", type=float, default=0.2)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--t-final", type=float, default=2.0)
    ap.add_argument("--steady-tol", type=float, default=1e-6)
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also store transient fields every N steps (for time-dependent PINNs)")
    args = ap.parse_args()

    if MPI.COMM_WORLD.size > 1:
        raise SystemExit("Run in serial: the export assumes a single process.")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.snapshot_every and args.solve == "steady":
        raise SystemExit("--snapshot-every needs --solve transient")

    for s in args.severities:
        print(f"\n===== severity {int(round(s * 100))}% =====", flush=True)
        run_case(s, args)


if __name__ == "__main__":
    main()

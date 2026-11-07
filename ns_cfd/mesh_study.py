"""Mesh-independence study for the 2D stenosis solver.

Three nested meshes (refinement ratio 2): 125x20, 250x40 (production), 500x80.
The y-warp is pointwise, so every coarse node is also a fine node and fields
are compared node-by-node without interpolation. Discretisation error is
estimated with the Grid Convergence Index (Roache 1994; Celik et al. 2008,
ASME J. Fluids Eng. 130:078001).

    python mesh_study.py mesh  --severity 0.5 --nx 250 --ny 40     # mesh plots only (no solve)
    python mesh_study.py run   --severity 0.5 --u-max 0.2 --nx 250 --ny 40
    python mesh_study.py batch --workers 6                         # full study, resumable
    python mesh_study.py report                                    # tables + plots

Outputs in ./mesh_study/: runs/<tag>.npz (fields + JSON meta),
plots/mesh_<tag>.png, plots/fields_<tag>.png, report/*.{csv,md,png}.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "mesh_study"
MESHES = [(125, 20), (250, 40), (500, 80)]          # coarse, medium (production), fine

# --------------------------------------------------------------------- study
# GCI families: 3 meshes each. Threshold checks: does the steady/symmetric
# verdict near the bifurcation depend on mesh size or triangle orientation?
# dt check: SUPG/PSPG tau contains (2/dt)^2, so the steady solution depends on dt.
STUDY = (
    [dict(severity=s, u_max=u, t_final=tf, nx=nx, ny=ny)
     for s, u, tf in ((0.25, 0.2, 2.0), (0.50, 0.2, 2.0), (0.60, 0.1, 4.0))
     for nx, ny in MESHES]
    + [dict(severity=s, u_max=u, t_final=tf, nx=nx, ny=ny, diagonal=d)
       for s, u, tf in ((0.55, 0.2, 2.0), (0.65, 0.1, 4.0))
       for (nx, ny), d in (((250, 40), "right"), ((250, 40), "crossed"), ((500, 80), "right"))]
    + [dict(severity=0.50, u_max=0.2, t_final=2.0, nx=250, ny=40, dt=0.001)]
)
# The study itself was run with the old "right" default; keep it explicit so tags stay valid.
DEFAULTS = dict(diagonal="right", dt=0.002, t_final=2.0)


def tag_of(c):
    c = {**DEFAULTS, **c}
    return (f"s{round(c['severity'] * 100):02d}_u{round(c['u_max'] * 100):02d}_{c['nx']}x{c['ny']}"
            f"_{c['diagonal']}_dt{round(c['dt'] * 1e4):02d}")


from mesh_plots import mesh_arrays, plot_mesh  # noqa: E402


# ------------------------------------------------------------------- QoIs
def node_index(xy, x, y):
    return int(np.argmin((xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2))


def near_wall_u(xy, uv, flag):
    """u on the wall-adjacent node row of the upper wall, as a function of x."""
    wall_up = (flag == 3) & (xy[:, 1] > 0)
    xs, us = [], []
    for i in np.flatnonzero(wall_up):
        col = np.isclose(xy[:, 0], xy[i, 0], atol=1e-12) & (xy[:, 1] < xy[i, 1]) & (xy[:, 1] > 0)
        if col.any():
            j = np.flatnonzero(col)[np.argmax(xy[col, 1])]
            xs.append(xy[i, 0])
            us.append(uv[j, 0])
    o = np.argsort(xs)
    return np.array(xs)[o], np.array(us)[o]


def recirculation(xy, uv, flag, x_center):
    """Separation / reattachment x [m] downstream of the throat (upper wall), or nan."""
    xs, us = near_wall_u(xy, uv, flag)
    m = xs >= x_center
    xs, us = xs[m], us[m]
    neg = np.flatnonzero(us < 0)
    if len(neg) == 0:
        return np.nan, np.nan

    def crossing(i):  # linear zero crossing between i-1 and i
        return xs[i - 1] - us[i - 1] * (xs[i] - xs[i - 1]) / (us[i] - us[i - 1])

    i_sep = neg[0]
    x_sep = crossing(i_sep) if i_sep > 0 else xs[0]
    pos_after = np.flatnonzero((us >= 0) & (np.arange(len(us)) > i_sep))
    x_re = crossing(pos_after[0]) if len(pos_after) else np.nan
    return float(x_sep), float(x_re)


def qois(xy, uv, p, flag, info, geo):
    x_sep, x_re = recirculation(xy, uv, flag, geo["x_center"])
    it = node_index(xy, geo["x_center"], 0.0)
    return dict(
        dp_pa=float(info["dp"]),
        umax=float(np.linalg.norm(uv, axis=1).max()),
        u_throat=float(uv[it, 0]),
        p_throat=float(p[it]),
        x_sep_mm=x_sep * 1e3,
        x_reatt_mm=x_re * 1e3,
        recirc_len_mm=(x_re - x_sep) * 1e3,
    )


# ------------------------------------------------------------------ modes
def build(args):
    from mesh_2d import build_stenosis_mesh
    geo = dict(severity=args.severity, length=0.05, radius=0.0025, x_center=0.01, half_length=0.004)
    domain, ft = build_stenosis_mesh(**geo, nx=args.nx, ny=args.ny, diagonal=args.diagonal)
    return domain, ft, geo


def cmd_mesh(args):
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    domain, ft, geo = build(args)
    xy, tri, flag = mesh_arrays(domain, ft)
    t = f"s{round(args.severity * 100):02d}_{args.nx}x{args.ny}_{args.diagonal}"
    st = plot_mesh(OUT / "plots" / f"mesh_{t}.png", xy, tri, flag, geo,
                   f"Mesh {args.nx}x{args.ny} ({args.diagonal}), severity {args.severity:.0%}")
    print(json.dumps(st))


def cmd_run(args):
    from generate_dataset_2d import nodal_arrays, plot_case, symmetry_error
    from solver_2d import run_simulation

    c = dict(severity=args.severity, u_max=args.u_max, nx=args.nx, ny=args.ny,
             diagonal=args.diagonal, dt=args.dt, t_final=args.t_final)
    tag = tag_of(c)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "plots").mkdir(parents=True, exist_ok=True)

    domain, ft, geo = build(args)
    xy0, tri0, flag0 = mesh_arrays(domain, ft)
    mstats = plot_mesh(OUT / "plots" / f"mesh_{tag}.png", xy0, tri0, flag0, geo,
                       f"Mesh {args.nx}x{args.ny} ({args.diagonal}), severity {args.severity:.0%}")

    t0 = time.time()
    u_h, p_h, info, _ = run_simulation(domain, ft, u_max=args.u_max, dt=args.dt,
                                       t_final=args.t_final, verbose=True)
    wall = time.time() - t0
    xy, uv, p, flag, tri = nodal_arrays(domain, ft, u_h, p_h)
    info.update(symmetry_error=symmetry_error(xy, uv), wall_time_s=wall, mesh=mstats,
                case=c, geometry=geo, qoi=qois(xy, uv, p, flag, info, geo))
    np.savez_compressed(OUT / "runs" / f"{tag}.npz", x=xy[:, 0], y=xy[:, 1], u=uv[:, 0], v=uv[:, 1],
                        p=p, flag=flag, triangles=tri, meta=json.dumps(info, default=float))
    plot_case(OUT / "plots" / f"fields_{tag}.png", xy, uv, p, tri, args.severity)
    q = info["qoi"]
    print(f"[run] {tag}: {wall:.0f}s steady={info['steady']} sym={info['symmetry_error']:.1e} "
          f"dp={q['dp_pa']:.3f}Pa umax={q['umax']:.4f} recirc={q['recirc_len_mm']:.2f}mm", flush=True)


def cmd_batch(args):
    todo = [c for c in STUDY if not (OUT / "runs" / f"{tag_of(c)}.npz").exists()]
    # longest first: fine meshes, then crossed, then the rest
    todo.sort(key=lambda c: -(c["nx"] * c["ny"] * (2 if c.get("diagonal") == "crossed" else 1)
                              * c.get("t_final", 2.0)))
    print(f"[batch] {len(STUDY) - len(todo)} done, {len(todo)} to run with {args.workers} workers", flush=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    def launch(c):
        c = {**DEFAULTS, **c}
        cmd = [sys.executable, str(Path(__file__).resolve()), "run",
               "--severity", str(c["severity"]), "--u-max", str(c["u_max"]), "--nx", str(c["nx"]),
               "--ny", str(c["ny"]), "--diagonal", c["diagonal"], "--dt", str(c["dt"]),
               "--t-final", str(c["t_final"])]
        with open(OUT / "logs" / f"{tag_of(c)}.log", "w") as log:
            r = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        last = (OUT / "logs" / f"{tag_of(c)}.log").read_text().strip().splitlines()[-1:]
        print(f"[batch] {'OK ' if r.returncode == 0 else 'FAIL'} {tag_of(c)} | {last[0] if last else ''}", flush=True)
        return r.returncode

    with ThreadPoolExecutor(args.workers) as ex:
        codes = list(ex.map(launch, todo))
    print(f"[batch] finished: {codes.count(0)} ok, {len(codes) - codes.count(0)} failed", flush=True)


# ------------------------------------------------------------------ report
def load_run(tag):
    f = OUT / "runs" / f"{tag}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return dict(xy=np.column_stack([d["x"], d["y"]]), uv=np.column_stack([d["u"], d["v"]]),
                p=d["p"], flag=d["flag"], tri=d["triangles"], meta=json.loads(str(d["meta"])))


def gci(f1, f2, f3, r=2.0):
    """Celik et al. (2008) with constant ratio. f1 fine, f2 medium, f3 coarse."""
    e21, e32 = f2 - f1, f3 - f2
    out = dict(f_fine=f1, f_med=f2, f_coarse=f3, p=np.nan, f_ext=np.nan,
               gci_fine=np.nan, gci_med=np.nan, err_med_vs_ext=np.nan, convergence="")
    if not np.all(np.isfinite([f1, f2, f3])):
        out["convergence"] = "n/a"
        return out
    if abs(e21) < 1e-14 * max(abs(f1), 1e-30):
        out.update(convergence="converged (no change)", p=np.inf, f_ext=f1, gci_fine=0.0, gci_med=0.0,
                   err_med_vs_ext=abs(f2 - f1) / max(abs(f1), 1e-30))
        return out
    ratio = e32 / e21
    if ratio < 0:
        out["convergence"] = "oscillatory"
        out["gci_fine"] = 1.25 * abs(e21 / f1)  # conservative bound, no order available
        return out
    p = abs(np.log(abs(ratio))) / np.log(r)
    f_ext = (r ** p * f1 - f2) / (r ** p - 1)
    out.update(p=p, f_ext=f_ext, convergence="monotonic",
               gci_fine=1.25 * abs(e21 / f1) / (r ** p - 1),
               gci_med=1.25 * abs(e32 / f2) / (r ** p - 1),
               err_med_vs_ext=abs(f2 - f_ext) / abs(f_ext))
    return out


def shared_node_errors(a, b):
    """Rel-L2 of (u,v) and p of run a vs run b on the nodes they share (a coarser)."""
    key = lambda xy: [(round(x, 10), round(y, 10)) for x, y in xy]
    lut = {k: i for i, k in enumerate(key(b["xy"]))}
    ia, ib = zip(*[(i, lut[k]) for i, k in enumerate(key(a["xy"])) if k in lut])
    ia, ib = np.array(ia), np.array(ib)
    rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
    return dict(shared_nodes=len(ia), l2_vel=rel(a["uv"][ia], b["uv"][ib]), l2_p=rel(a["p"][ia], b["p"][ib]),
                max_abs_u=float(np.abs(a["uv"][ia] - b["uv"][ib]).max()),
                max_abs_p=float(np.abs(a["p"][ia] - b["p"][ib]).max()))


QOI_LABELS = dict(dp_pa="Δp (Pa)", umax="max|u| (m/s)", u_throat="u throat centre (m/s)",
                  p_throat="p throat centre (Pa)", recirc_len_mm="recirculation length (mm)",
                  x_reatt_mm="reattachment x (mm)")


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = OUT / "report"
    rep.mkdir(parents=True, exist_ok=True)
    md = ["# Mesh-independence study — results\n"]
    csv_rows = []

    families = [((0.25, 0.2), "Re≈301"), ((0.50, 0.2), "Re≈301"), ((0.60, 0.1), "Re≈150")]
    for (s, u), re_lab in families:
        runs = [load_run(tag_of(dict(severity=s, u_max=u, nx=nx, ny=ny))) for nx, ny in MESHES]
        name = f"{round(s * 100)}% {re_lab}"
        if any(r is None for r in runs):
            md.append(f"\n## {name}: incomplete ({sum(r is not None for r in runs)}/3 meshes)\n")
            continue
        h = np.array([0.05 / nx for nx, _ in MESHES]) * 1e3      # axial spacing (mm)
        md.append(f"\n## {name} (severity {s:.2f}, u_max {u} m/s)\n")
        md.append("| mesh | cells | steady | symmetry err | wall time |\n|---|---|---|---|---|")
        for (nx, ny), r in zip(MESHES, runs):
            m = r["meta"]
            md.append(f"| {nx}×{ny} | {m['mesh']['cells']} | {m['steady']} | {m['symmetry_error']:.1e} "
                      f"| {m['wall_time_s'] / 60:.1f} min |")
        md.append("\n| QoI | coarse | medium (production) | fine | order p | extrapolated | "
                  "GCI fine | GCI medium | medium vs extrap. | convergence |\n|---|---|---|---|---|---|---|---|---|---|")
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), dpi=130)
        for ax, (k, lab) in zip(axes.flat, QOI_LABELS.items()):
            f3, f2, f1 = (r["meta"]["qoi"][k] for r in runs)
            g = gci(f1, f2, f3)
            csv_rows.append(dict(case=name, qoi=k, **g))
            fmt = lambda v: "–" if not np.isfinite(v) else f"{v:.4g}"
            pct = lambda v: "–" if not np.isfinite(v) else f"{v:.2%}"
            md.append(f"| {lab} | {fmt(f3)} | {fmt(f2)} | {fmt(f1)} | {fmt(g['p'])} | {fmt(g['f_ext'])} | "
                      f"{pct(g['gci_fine'])} | {pct(g['gci_med'])} | {pct(g['err_med_vs_ext'])} | {g['convergence']} |")
            ax.plot(h, [f3, f2, f1], "o-", label="CFD")
            if np.isfinite(g["f_ext"]):
                ax.axhline(g["f_ext"], color="C3", ls="--", lw=1, label="Richardson extrapolation")
            ax.axvline(h[1], color="0.6", ls=":", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("axial spacing h (mm)")
            ax.set_title(lab, fontsize=9)
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Grid convergence: {name}  (dotted = production mesh)")
        fig.tight_layout()
        fig.savefig(rep / f"gci_{round(s * 100)}_{re_lab.replace('≈', '')}.png")
        plt.close(fig)

        # node-wise field differences vs the fine mesh
        md.append("\n| field difference vs fine mesh (shared nodes) | rel-L2 velocity | rel-L2 pressure | max abs Δu | max abs Δp |\n|---|---|---|---|---|")
        for lab, r in (("coarse 125×20", runs[0]), ("medium 250×40", runs[1])):
            e = shared_node_errors(r, runs[2])
            md.append(f"| {lab} ({e['shared_nodes']} nodes) | {e['l2_vel']:.2%} | {e['l2_p']:.2%} | "
                      f"{e['max_abs_u']:.4f} m/s | {e['max_abs_p']:.3f} Pa |")

        # profiles: centreline u, p and near-wall u
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), dpi=130, sharex=True)
        for (nx, ny), r in zip(MESHES, runs):
            xy, uv, p = r["xy"], r["uv"], r["p"]
            c = np.isclose(xy[:, 1], 0.0, atol=1e-12)
            o = np.argsort(xy[c, 0])
            axes[0].plot(xy[c, 0][o] * 1e3, uv[c, 0][o], label=f"{nx}×{ny}", lw=1)
            axes[1].plot(xy[c, 0][o] * 1e3, p[c][o], lw=1)
            xs, us = near_wall_u(xy, uv, r["flag"])
            axes[2].plot(xs * 1e3, us, lw=1)
        axes[0].set_ylabel("centreline u (m/s)")
        axes[1].set_ylabel("centreline p (Pa)")
        axes[2].set_ylabel("near-wall u (m/s)")
        axes[2].axhline(0, color="k", lw=0.5)
        axes[2].set_xlabel("x (mm)")
        axes[0].legend()
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle(f"Profiles on the three meshes: {name}")
        fig.tight_layout()
        fig.savefig(rep / f"profiles_{round(s * 100)}_{re_lab.replace('≈', '')}.png")
        plt.close(fig)

    # symmetry-breaking threshold checks
    md.append("\n## Symmetry-breaking threshold: mesh and triangle-orientation dependence\n")
    md.append("| case | mesh | diagonal | steady | symmetry err | Δp (Pa) | verdict (usable?) |\n|---|---|---|---|---|---|---|")
    for s, u, re_lab in ((0.55, 0.2, "Re≈301"), (0.65, 0.1, "Re≈150")):
        for (nx, ny), d in (((250, 40), "right"), ((250, 40), "crossed"), ((500, 80), "right")):
            r = load_run(tag_of(dict(severity=s, u_max=u, nx=nx, ny=ny, diagonal=d)))
            if r is None:
                md.append(f"| {round(s * 100)}% {re_lab} | {nx}×{ny} | {d} | – | – | – | not run |")
                continue
            m = r["meta"]
            ok = m["steady"] and m["symmetry_error"] <= 0.05
            md.append(f"| {round(s * 100)}% {re_lab} | {nx}×{ny} | {d} | {m['steady']} | "
                      f"{m['symmetry_error']:.2e} | {m['qoi']['dp_pa']:.3f} | {'yes' if ok else 'no'} |")

    # dt sensitivity
    md.append("\n## Time-step sensitivity of the steady solution (50 %, Re≈301, 250×40)\n")
    a = load_run(tag_of(dict(severity=0.5, u_max=0.2, nx=250, ny=40)))
    b = load_run(tag_of(dict(severity=0.5, u_max=0.2, nx=250, ny=40, dt=0.001)))
    if a and b:
        md.append("| QoI | dt = 2 ms | dt = 1 ms | rel. diff |\n|---|---|---|---|")
        for k, lab in QOI_LABELS.items():
            fa, fb = a["meta"]["qoi"][k], b["meta"]["qoi"][k]
            md.append(f"| {lab} | {fa:.4g} | {fb:.4g} | {abs(fa - fb) / max(abs(fb), 1e-30):.2%} |")
        e = shared_node_errors(a, b)
        md.append(f"\nField difference (all nodes): velocity {e['l2_vel']:.3%}, pressure {e['l2_p']:.3%}.")
    else:
        md.append("not run")

    (rep / "report.md").write_text("\n".join(md) + "\n")
    import csv
    with open(rep / "gci_table.csv", "w", newline="") as f:
        if csv_rows:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mesh", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--severity", type=float, required=True)
        sp.add_argument("--nx", type=int, default=250)
        sp.add_argument("--ny", type=int, default=40)
        sp.add_argument("--diagonal", default="crossed")
        if name == "run":
            sp.add_argument("--u-max", type=float, default=0.2)
            sp.add_argument("--dt", type=float, default=0.002)
            sp.add_argument("--t-final", type=float, default=2.0)
    sb = sub.add_parser("batch")
    sb.add_argument("--workers", type=int, default=6)
    sub.add_parser("report")
    args = ap.parse_args()
    dict(mesh=cmd_mesh, run=cmd_run, batch=cmd_batch, report=cmd_report)[args.cmd](args)


if __name__ == "__main__":
    main()

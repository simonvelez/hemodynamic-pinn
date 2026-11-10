"""CFD accuracy-vs-cost study: the CFD side of the CFD-vs-PINN comparison.

For a fair speed comparison the CFD baseline must be the *cheapest* configuration that
reaches the PINN's accuracy, not the fine reference (McGreivy & Hakim 2024; guide §5.5).
This script times a matrix of (mesh, element, solve mode) per case, measures each run's
error against a fine P2/P1 reference, and writes an error-vs-wall-clock table (Pareto front).

    python cost_study.py list                 # the planned matrix
    python cost_study.py batch [--repeats 3]  # runs SEQUENTIALLY (clean timings); skips finished runs
    python cost_study.py report               # cost_study/report/{pareto.csv, pareto_*.png, report.md}

Timing hygiene: one process at a time on an otherwise idle machine, OMP_NUM_THREADS=1, each
configuration repeated (median reported); mesh generation, solve and export are timed separately.
Errors are evaluated at the reference mesh nodes (candidate P1 export linearly interpolated), with
the same metrics as the PINN evaluation: rel-L2 of velocity and pressure, Δp, peak WSS and
recirculation length. Run on the machine that also trains the PINN (A4000 box).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "cost_study"

# 0.35 at u_max 0.1 is a held-out PINN test severity (campaign RE150 split): the comparison on a case the
# PINN never saw. The others are PINN training severities (range ends and the hardest symmetric case).
CASES = [dict(severity=0.25, u_max=0.2), dict(severity=0.50, u_max=0.2), dict(severity=0.35, u_max=0.1),
         dict(severity=0.60, u_max=0.1)]
MESHES = [(100, 16), (125, 20), (175, 28), (250, 40), (350, 56), (500, 80)]   # r ≈ 1.4 steps
SCHEMES = [("P2P1", "steady"), ("P1P1", "steady"), ("P2P1", "transient"), ("P1P1", "transient")]
REFERENCE = dict(nx=700, ny=112, element="P2P1", solve="steady")
T_FINAL = {0.2: 2.0, 0.1: 4.0}           # transient runs: same budgets as the datasets


def matrix():
    runs = []
    for c in CASES:
        runs.append(dict(c, **REFERENCE, ref=True))
        for el, solve in SCHEMES:
            for nx, ny in MESHES:
                if solve == "transient" and nx * ny > 250 * 40 and el == "P2P1":
                    continue        # hours per run; dominated by the steady solve anyway
                runs.append(dict(c, nx=nx, ny=ny, element=el, solve=solve, ref=False))
    return runs


def tag_of(r, rep=None):
    t = f"s{round(r['severity'] * 100):02d}_u{round(r['u_max'] * 100):02d}_{r['nx']}x{r['ny']}_{r['element']}_{r['solve']}"
    return t if rep is None else f"{t}_rep{rep}"


# ----------------------------------------------------------------------- run
def cmd_run(a):
    from mesh_2d import build_stenosis_mesh
    from solver_2d import run_simulation
    from generate_dataset_2d import nodal_arrays, WALL
    from wall_shear import wall_qois, wall_shear_stress

    r = dict(severity=a.severity, u_max=a.u_max, nx=a.nx, ny=a.ny, element=a.element, solve=a.solve)
    geo = dict(severity=a.severity, length=0.05, radius=0.0025, x_center=0.01, half_length=0.004)
    t = {}
    t0 = time.perf_counter()
    dom, ft = build_stenosis_mesh(**geo, nx=a.nx, ny=a.ny)
    t["mesh_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    u_h, p_h, info, _ = run_simulation(dom, ft, u_max=a.u_max, element=a.element, viscous_form="grad",
                                       steady_solve=a.solve == "steady", t_final=T_FINAL[a.u_max],
                                       verbose=False)
    t["solve_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    xy, uv, p, flag, tri = nodal_arrays(dom, ft, u_h, p_h)
    w = np.flatnonzero(flag == WALL)
    tau = wall_shear_stress(dom, u_h, info["params"]["mu"], xy[w], geo)
    q = dict(dp_pa=info["dp"], **wall_qois(xy[w, 0], xy[w, 1], tau, geo["x_center"]))
    t["post_s"] = time.perf_counter() - t0
    t["total_s"] = sum(t.values())
    meta = dict(run=r, timing=t, qoi=q, n_dofs=info["n_dofs"], steps=info["steps"],
                newton=info["newton_iterations"], steady=info["steady"], t_end=info["t_end"])
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "runs" / f"{tag_of(r, a.rep)}.npz", x=xy[:, 0], y=xy[:, 1], u=uv[:, 0],
                        v=uv[:, 1], p=p, triangles=tri, wall_idx=w, tau_w=tau,
                        meta=json.dumps(meta, default=float))
    print(f"[run] {tag_of(r, a.rep)}: solve {t['solve_s']:.1f}s total {t['total_s']:.1f}s dofs {info['n_dofs']}",
          flush=True)


def cmd_list(a):
    for r in matrix():
        print(tag_of(r), "(reference)" if r["ref"] else "")
    print(f"{len(matrix())} configurations x repeats")


def cmd_batch(a):
    sys.path.insert(0, str(HERE.parent / "common"))
    from manifest import RunLog
    log = RunLog(OUT, kind="cfd_cost_study", config=dict(repeats=a.repeats))
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    for r in matrix():
        for rep in range(1 if r["ref"] else a.repeats):
            if (OUT / "runs" / f"{tag_of(r, rep)}.npz").exists():
                continue
            cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--severity", str(r["severity"]),
                   "--u-max", str(r["u_max"]), "--nx", str(r["nx"]), "--ny", str(r["ny"]),
                   "--element", r["element"], "--solve", r["solve"], "--rep", str(rep)]
            with open(OUT / "logs" / f"{tag_of(r, rep)}.log", "w") as f:
                rc = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT).returncode
            print(f"[batch] {'OK ' if rc == 0 else 'FAIL'} {tag_of(r, rep)}", flush=True)
    log.finish()


# -------------------------------------------------------------------- report
def _load(tag):
    f = OUT / "runs" / f"{tag}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return dict(d=d, meta=json.loads(str(d["meta"])))


def _interp(src, xy_dst):
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(src["x"], src["y"], src["triangles"])
    out = []
    for k in ("u", "v", "p"):
        f = mtri.LinearTriInterpolator(tri, src[k])(xy_dst[:, 0], xy_dst[:, 1])
        out.append(np.ma.filled(f, np.nan))
    return np.column_stack(out)


def cmd_report(a):
    import csv
    sys.path.insert(0, str(HERE.parent / "common"))
    from metrics import node_areas, rel_l2_weighted
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep_dir = OUT / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in CASES:
        ref = _load(tag_of(dict(c, **REFERENCE), 0))
        if ref is None:
            print(f"no reference for {c}")
            continue
        rd, rq = ref["d"], ref["meta"]["qoi"]
        xy = np.column_stack([rd["x"], rd["y"]])
        u_ref, p_ref = np.column_stack([rd["u"], rd["v"]]), rd["p"]
        w_ref = node_areas(xy, rd["triangles"])     # area weights for the l2_*_area keys (guide §7.3)
        for r in matrix():
            if r["ref"] or r["severity"] != c["severity"] or r["u_max"] != c["u_max"]:
                continue
            reps = [x for x in (_load(tag_of(r, k)) for k in range(a.repeats)) if x is not None]
            if not reps:
                continue
            f = _interp(reps[0]["d"], xy)
            ok = np.isfinite(f).all(axis=1)
            q = reps[0]["meta"]["qoi"]
            rel = lambda est, ref_: abs(est - ref_) / abs(ref_) if np.isfinite(ref_) else np.nan
            rows.append(dict(
                case=f"s{round(c['severity'] * 100)}_u{round(c['u_max'] * 100)}", mesh=f"{r['nx']}x{r['ny']}",
                element=r["element"], solve=r["solve"], n_dofs=reps[0]["meta"]["n_dofs"],
                solve_s_median=float(np.median([x["meta"]["timing"]["solve_s"] for x in reps])),
                total_s_median=float(np.median([x["meta"]["timing"]["total_s"] for x in reps])),
                n_repeats=len(reps),
                l2_vel=float(np.linalg.norm(f[ok, :2] - u_ref[ok]) / np.linalg.norm(u_ref[ok])),
                l2_p=float(np.linalg.norm(f[ok, 2] - p_ref[ok]) / np.linalg.norm(p_ref[ok])),
                l2_vel_area=rel_l2_weighted(f[ok, :2], u_ref[ok], w_ref[ok]),
                l2_p_area=rel_l2_weighted(f[ok, 2], p_ref[ok], w_ref[ok]),
                dp_rel_err=rel(q["dp_pa"], rq["dp_pa"]),
                wss_peak_rel_err=rel(q["wss_peak_pa"], rq["wss_peak_pa"]),
                recirc_rel_err=rel(q["recirc_len"], rq["recirc_len"])))
    if not rows:
        print("nothing to report")
        return
    with open(rep_dir / "pareto.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = ["# CFD accuracy vs cost\n", f"Reference: {REFERENCE}. Times are medians over repeats.\n"]
    for case in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case]
        md += [f"\n## {case}\n", "| element | solve | mesh | dofs | solve (s) | vel L2 | p L2 | Δp err | peak WSS err |",
               "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {r['element']} | {r['solve']} | {r['mesh']} | {r['n_dofs']} | {r['solve_s_median']:.1f} | "
               f"{r['l2_vel']:.2%} | {r['l2_p']:.2%} | {r['dp_rel_err']:.2%} | {r['wss_peak_rel_err']:.2%} |"
               for r in sorted(sub, key=lambda r: (r["element"], r["solve"], r["n_dofs"]))]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=130)
        for ax, key, lab in zip(axes, ("l2_vel", "l2_p", "dp_rel_err"), ("velocity rel-L2", "pressure rel-L2", "Δp rel. error")):
            for el, solve in SCHEMES:
                pts = sorted((r["total_s_median"], r[key]) for r in sub if r["element"] == el and r["solve"] == solve)
                if pts:
                    ax.loglog(*zip(*pts), "o-", label=f"{el} {solve}")
            ax.set_xlabel("wall-clock per case (s)")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3, which="both")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"CFD error vs cost, {case} (add the PINN points from its metrics/cost.json)")
        fig.tight_layout()
        fig.savefig(rep_dir / f"pareto_{case}.png")
        plt.close(fig)
    (rep_dir / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sb = sub.add_parser("batch")
    sb.add_argument("--repeats", type=int, default=3)
    sr = sub.add_parser("report")
    sr.add_argument("--repeats", type=int, default=3)
    r = sub.add_parser("run")
    r.add_argument("--severity", type=float, required=True)
    r.add_argument("--u-max", type=float, required=True)
    r.add_argument("--nx", type=int, required=True)
    r.add_argument("--ny", type=int, required=True)
    r.add_argument("--element", choices=["P1P1", "P2P1"], required=True)
    r.add_argument("--solve", choices=["steady", "transient"], required=True)
    r.add_argument("--rep", type=int, default=0)
    a = ap.parse_args()
    dict(list=cmd_list, batch=cmd_batch, report=cmd_report, run=cmd_run)[a.cmd](a)


if __name__ == "__main__":
    main()

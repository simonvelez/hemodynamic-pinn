"""Stability of the symmetric steady flow: perturb, then watch the asymmetry grow or decay.

On the symmetric `crossed` mesh the flow stays symmetric to round-off even when
the symmetric state is unstable, because nothing breaks the symmetry. Here a
small antisymmetric, zero-flux component eps*eta*(1-eta^2) (eta = y/R) is added
to the inlet for t < perturb_until; afterwards the inlet is symmetric again and
the asymmetry a(t) = max|v| on the centreline / u_max is tracked:

    a(t) ~ exp(sigma t):  sigma < 0  -> symmetric flow is stable (disturbance decays)
                          sigma > 0  -> unstable: physical symmetry breaking

    python stability_test.py run --severity 0.6 --u-max 0.2     # one case
    python stability_test.py batch --workers 6                    # all cases, resumable
    python stability_test.py report                               # table + plots

Outputs in ./stability/: runs/<tag>.npz (+ meta), plots/{mesh,fields}_<tag>.png, report/.
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
OUT = HERE / "stability"

EPS = 0.01            # peak disturbance 0.385*EPS ≈ 0.4 % of u_max (small: linear regime)
PERTURB_UNTIL = 0.2   # s (inlet ramp ends at 0.1 s)
MIN_TIME = 3.0        # s: never stop as "steady" before this, so slow growth is visible
T_FINAL = 6.0         # s
LINEAR_MAX = 5e-2     # a(t) above this is treated as nonlinear (saturating) growth
FLOOR = 1e-12         # a(t) below this is round-off
MARGINAL = 0.1        # |sigma| < this [1/s] -> "marginal"

STUDY = (
    [dict(severity=s, u_max=0.2) for s in (0.50, 0.55, 0.60, 0.65, 0.70)]    # Re≈301 (0.50 = control)
    + [dict(severity=s, u_max=0.1) for s in (0.60, 0.65, 0.70)]             # Re≈150 (0.60 = control)
    # Threshold check with the corrected scheme (P2/P1 Taylor-Hood, gradient-form viscous term):
    # On 125x20: with P2 velocity it has the same nodes as the P1 250x40 runs and is within
    # 0.3-0.4 % (velocity) of the 500x80 solution (mesh study); 250x40 P2 time-marching takes ~8 h/case.
    + [dict(severity=s, u_max=u, element="P2P1", viscous_form="grad", nx=125, ny=20)
       for s, u in ((0.50, 0.2), (0.55, 0.2), (0.65, 0.2), (0.70, 0.2), (0.60, 0.1), (0.65, 0.1), (0.70, 0.1))]
)
# Original runs used P1/P1 + SUPG/PSPG/LSIC with the symmetric viscous form (legacy tags).
DEFAULTS = dict(nx=250, ny=40, diagonal="crossed", eps=EPS, element="P1P1", viscous_form="sym")
LEGACY_SCHEME = ("P1P1", "sym")


def tag_of(c):
    c = {**DEFAULTS, **c}
    scheme = "" if (c["element"], c["viscous_form"]) == LEGACY_SCHEME else f"_{c['element']}{c['viscous_form']}"
    return (f"s{round(c['severity'] * 100):02d}_u{round(c['u_max'] * 100):02d}_{c['nx']}x{c['ny']}"
            f"_{c['diagonal']}_eps{round(c['eps'] * 1000):03d}{scheme}")


def growth_rate(hist, t_start):
    """Late-time exponential rate of a(t) = exp(c + sigma*t).

    Right after the perturbation every case first decays (transient), so a fit
    over the whole window biases sigma. Fit only after the minimum of a(t)
    (t > t_start), on the linear regime FLOOR < a < LINEAR_MAX. If a(t) is
    still decreasing at the end (stable), fit the last second instead.
    """
    t, a = np.asarray(hist, dtype=float).T
    post = t >= t_start
    if post.sum() < 5:
        return np.nan, 0
    i_min = np.flatnonzero(post)[np.argmin(a[post])]
    lin = (a > FLOOR) & (a < LINEAR_MAX)
    m = lin & (t >= t[i_min] + 0.2)
    if m.sum() < 5:                       # minimum at the end: decaying -> use the last second
        m = lin & (t >= t[-1] - 1.0)
    if m.sum() < 5:
        return np.nan, 0
    sigma, _ = np.polyfit(t[m], np.log(a[m]), 1)
    return float(sigma), int(m.sum())


def verdict(sigma, a_final):
    if np.isfinite(sigma):
        if sigma < -MARGINAL:
            return "stable (decays)"
        if sigma > MARGINAL:
            return "UNSTABLE (grows)"
        return "marginal"
    if a_final >= LINEAR_MAX:
        return "UNSTABLE (saturated)"
    return "stable (at round-off)" if a_final <= FLOOR else "undetermined"


def cmd_run(args):
    from generate_dataset_2d import nodal_arrays, plot_case, symmetry_error
    from mesh_2d import build_stenosis_mesh
    from mesh_plots import mesh_arrays, plot_mesh
    from solver_2d import run_simulation

    c = dict(severity=args.severity, u_max=args.u_max, nx=args.nx, ny=args.ny,
             diagonal=args.diagonal, eps=args.eps, element=args.element, viscous_form=args.viscous_form)
    tag = tag_of(c)
    for d in ("runs", "plots"):
        (OUT / d).mkdir(parents=True, exist_ok=True)

    geo = dict(severity=args.severity, length=0.05, radius=0.0025, x_center=0.01, half_length=0.004)
    domain, ft = build_stenosis_mesh(**geo, nx=args.nx, ny=args.ny, diagonal=args.diagonal)
    xy0, tri0, flag0 = mesh_arrays(domain, ft)
    mstats = plot_mesh(OUT / "plots" / f"mesh_{tag}.png", xy0, tri0, flag0, geo,
                       f"Mesh {args.nx}x{args.ny} ({args.diagonal}), severity {args.severity:.0%}")

    t0 = time.time()
    u_h, p_h, info, _ = run_simulation(domain, ft, u_max=args.u_max, dt=0.002, t_final=args.t_final,
                                       perturb_eps=args.eps, perturb_until=PERTURB_UNTIL,
                                       min_time=MIN_TIME, element=args.element,
                                       viscous_form=args.viscous_form, verbose=True)
    xy, uv, p, flag, tri = nodal_arrays(domain, ft, u_h, p_h)
    sigma, npts = growth_rate(info["asym_history"], PERTURB_UNTIL + 0.3)
    a_final = info["asym_history"][-1][1]
    info.update(case=c, geometry=geo, mesh=mstats, wall_time_s=time.time() - t0,
                symmetry_error=symmetry_error(xy, uv), sigma=sigma, fit_points=npts,
                a_final=a_final, a_max=max(a for _, a in info["asym_history"]),
                verdict=verdict(sigma, a_final))
    np.savez_compressed(OUT / "runs" / f"{tag}.npz", x=xy[:, 0], y=xy[:, 1], u=uv[:, 0], v=uv[:, 1],
                        p=p, flag=flag, triangles=tri, meta=json.dumps(info, default=float))
    plot_case(OUT / "plots" / f"fields_{tag}.png", xy, uv, p, tri, args.severity)
    print(f"[run] {tag}: {info['wall_time_s'] / 60:.0f} min | sigma={sigma:+.3f} 1/s | "
          f"a_final={a_final:.2e} | sym_err={info['symmetry_error']:.1e} | {info['verdict']}", flush=True)


def cmd_batch(args):
    todo = [c for c in STUDY if not (OUT / "runs" / f"{tag_of(c)}.npz").exists()]
    print(f"[batch] {len(STUDY) - len(todo)} done, {len(todo)} to run with {args.workers} workers", flush=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    def launch(c):
        c = {**DEFAULTS, **c}
        cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--severity", str(c["severity"]),
               "--u-max", str(c["u_max"]), "--nx", str(c["nx"]), "--ny", str(c["ny"]),
               "--diagonal", c["diagonal"], "--eps", str(c["eps"]),
               "--element", c["element"], "--viscous-form", c["viscous_form"]]
        log = OUT / "logs" / f"{tag_of(c)}.log"
        with open(log, "w") as f:
            r = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
        last = log.read_text().strip().splitlines()[-1:]
        print(f"[batch] {'OK ' if r.returncode == 0 else 'FAIL'} {tag_of(c)} | {last[0] if last else ''}", flush=True)
        return r.returncode

    with ThreadPoolExecutor(args.workers) as ex:
        codes = list(ex.map(launch, todo))
    print(f"[batch] finished: {codes.count(0)} ok, {len(codes) - codes.count(0)} failed", flush=True)


def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = OUT / "report"
    rep.mkdir(parents=True, exist_ok=True)
    md = ["# Stability of the symmetric 2D stenosis flow\n",
          f"Antisymmetric inlet perturbation eps = {EPS} (peak ≈ {0.385 * EPS:.1%} of u_max) for t < {PERTURB_UNTIL} s, "
          f"then symmetric inlet; mesh 250×40 crossed (P1P1) or 125×20 crossed (P2P1); growth rate σ = late-time slope of log a(t) "
          f"(after the post-perturbation minimum, {FLOOR:g} < a < {LINEAR_MAX:g}). "
          f"Runs stop as 'steady' after {MIN_TIME} s if the change per step is < 1e-6, so slowly "
          f"growing cases end before saturating.\n",
          "| Re | severity | scheme | σ (1/s) | max a(t) | final a | final symmetry err | steady | t_end (s) | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=130, sharey=True)
    for c in STUDY:
        f = OUT / "runs" / f"{tag_of(c)}.npz"
        re_lab = "≈301" if c["u_max"] == 0.2 else "≈150"
        cc = {**DEFAULTS, **c}
        scheme = f"{cc['element']}/{cc['viscous_form']}"
        if not f.exists():
            md.append(f"| {re_lab} | {c['severity']:.0%} | {scheme} | – | – | – | – | – | – | not run |")
            continue
        m = json.loads(str(np.load(f)["meta"]))
        m["sigma"], _ = growth_rate(m["asym_history"], PERTURB_UNTIL + 0.3)
        m["verdict"] = verdict(m["sigma"], m["a_final"])
        sig = "–" if not np.isfinite(m["sigma"]) else f"{m['sigma']:+.3f}"
        md.append(f"| {re_lab} | {c['severity']:.0%} | {scheme} | {sig} | {m['a_max']:.2e} | {m['a_final']:.2e} | "
                  f"{m['symmetry_error']:.1e} | {m['steady']} | {m['t_end']:.2f} | **{m['verdict']}** |")
        t, a = np.asarray(m["asym_history"]).T
        ax = axes[0 if c["u_max"] == 0.2 else 1]
        ax.semilogy(t, np.maximum(a, 1e-17), ls="-" if (cc["element"], cc["viscous_form"]) == LEGACY_SCHEME else "--",
                    label=f"{c['severity']:.0%} {scheme} ({m['verdict']})")
    for ax, title in zip(axes, ("Re ≈ 301 (u_max 0.2 m/s)", "Re ≈ 150 (u_max 0.1 m/s)")):
        ax.axvspan(0, PERTURB_UNTIL, color="0.9", label="perturbation on")
        ax.axhline(LINEAR_MAX, color="0.5", ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("asymmetry a(t) = max|v|centreline / u_max")
    fig.tight_layout()
    fig.savefig(rep / "asymmetry_vs_time.png")
    plt.close(fig)
    (rep / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sr = sub.add_parser("run")
    sr.add_argument("--severity", type=float, required=True)
    sr.add_argument("--u-max", type=float, default=0.2)
    sr.add_argument("--nx", type=int, default=250)
    sr.add_argument("--ny", type=int, default=40)
    sr.add_argument("--diagonal", default="crossed")
    sr.add_argument("--eps", type=float, default=EPS)
    sr.add_argument("--element", default=DEFAULTS["element"], choices=["P1P1", "P2P1"])
    sr.add_argument("--viscous-form", default=DEFAULTS["viscous_form"], choices=["sym", "grad"])
    sr.add_argument("--t-final", type=float, default=T_FINAL)
    sb = sub.add_parser("batch")
    sb.add_argument("--workers", type=int, default=6)
    sub.add_parser("report")
    args = ap.parse_args()
    dict(run=cmd_run, batch=cmd_batch, report=cmd_report)[args.cmd](args)


if __name__ == "__main__":
    main()

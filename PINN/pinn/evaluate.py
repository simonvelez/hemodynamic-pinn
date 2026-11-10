"""Evaluate a trained run against every CFD case (train and held-out)."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_cases, split_cases
from .geometry import wall_frame
from .train import load_checkpoint, predict, rel_l2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from manifest import RunLog  # noqa: E402
from metrics import node_areas, rel_l2_weighted  # noqa: E402


def pinn_wss(model, case, geo, Re, device="cpu"):
    """Nondimensional signed WSS tau/(rho U^2) = (1/Re) t.(grad u).n_in at the CFD wall nodes,
    by automatic differentiation (same sign convention as ns_cfd/wall_shear.py)."""
    xy = case.xy[case.wall_idx]
    pts = torch.tensor(np.column_stack([xy, np.full(len(xy), case.severity)]),
                       dtype=torch.get_default_dtype(), device=device, requires_grad=True)
    out = model(pts)
    gu = torch.autograd.grad(out[:, 0].sum(), pts, retain_graph=True)[0][:, :2]
    gv = torch.autograd.grad(out[:, 1].sum(), pts)[0][:, :2]
    g = torch.stack([gu, gv], dim=1).detach().cpu().numpy()          # g[:, i, j] = du_i/dx_j
    t, n = wall_frame(xy[:, 0], xy[:, 1], case.severity, geo)
    return np.einsum("ni,nij,nj->n", t, g, n) / Re


def recirculation(x, tau, x_from):
    """(x_sep, x_reattach) from sign changes of tau(x) for x >= x_from (nan if none);
    same rule as ns_cfd/wall_shear.recirculation_from_wss."""
    o = np.argsort(x)
    xs, ts = x[o], tau[o]
    m = xs >= x_from
    xs, ts = xs[m], ts[m]
    neg = np.flatnonzero(ts < 0)
    if len(neg) == 0:
        return np.nan, np.nan
    cross = lambda i: xs[i - 1] - ts[i - 1] * (xs[i] - xs[i - 1]) / (ts[i] - ts[i - 1])
    i_sep = neg[0]
    x_sep = cross(i_sep) if i_sep > 0 else xs[0]
    after = np.flatnonzero((ts >= 0) & (np.arange(len(ts)) > i_sep))
    return float(x_sep), float(cross(after[0])) if len(after) else np.nan


def wss_metrics(case, tau_pinn, geo):
    """WSS errors on the upper wall (the lower is its mirror image), in CFD units of rho U^2.
    l2_wss: rel-L2 along the wall; peak and recirculation-length errors in % and in D."""
    xw, yw = case.xy[case.wall_idx, 0], case.xy[case.wall_idx, 1]
    up = yw > 0
    x, tc, tp = xw[up], case.tau_w[up], tau_pinn[up]
    ic, ip = np.argmax(np.abs(tc)), np.argmax(np.abs(tp))
    sc, rc = recirculation(x, tc, geo.x_center)
    sp, rp = recirculation(x, tp, geo.x_center)
    Lc, Lp = rc - sc, rp - sp
    return dict(l2_wss=float(np.linalg.norm(tp - tc) / np.linalg.norm(tc)),
                wss_peak_rel_err=float(abs(tp[ip] - tc[ic]) / abs(tc[ic])),
                wss_peak_x_err_D=float(abs(x[ip] - x[ic])),
                recirc_len_cfd_D=float(Lc), recirc_len_pinn_D=float(Lp),
                recirc_len_err_D=float(abs(Lp - Lc)) if np.isfinite(Lc) and np.isfinite(Lp) else
                (0.0 if not (np.isfinite(Lc) or np.isfinite(Lp)) else float("nan")))


def _pinn_dp(model, geo, severities, device="cpu", n=201):
    """Nondimensional Δp = mean p(inlet) - mean p(outlet) along the cross-sections."""
    y = torch.linspace(-geo.R, geo.R, n, device=device)
    out = []
    for s in severities:
        line = lambda x: torch.stack([torch.full_like(y, x), y, torch.full_like(y, s)], dim=1)
        out.append(predict(model, line(0.0))[:, 2].mean() - predict(model, line(geo.length))[:, 2].mean())
    return np.array(out)


def pressure_metrics(case, pred):
    """Pressure errors that do not depend on how the gauge is chosen.

    l2_p_meanfree : rel-L2 of p minus its node mean (removes a constant offset)
    p_rms_over_dp : RMS(p_PINN - p_CFD) / dp_CFD  (error in units of the pressure drop)
    p_max_over_dp : max |p_PINN - p_CFD| / dp_CFD
    p_offset_over_dp : mean(p_PINN - p_CFD) / dp_CFD (a pure gauge shift shows up here)
    """
    pc, pp = case.uvp[:, 2], pred[:, 2]
    dp = abs(case.dp())
    e = pp - pc
    a, b = pp - pp.mean(), pc - pc.mean()
    return dict(l2_p_meanfree=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                p_rms_over_dp=float(np.sqrt(np.mean(e ** 2)) / dp),
                p_max_over_dp=float(np.abs(e).max() / dp),
                p_offset_over_dp=float(e.mean() / dp))


def _plot_case(path, case, pred, scales):
    """CFD / PINN / error maps. Pressure is gauge pressure (outlet = 0), so it is drawn
    with a diverging map centred at 0: blue = below outlet pressure (the low-pressure
    zone behind the throat), red = above. The pressure error is signed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import TwoSlopeNorm

    tri = mtri.Triangulation(case.xy[:, 0], case.xy[:, 1], case.triangles)
    speed = lambda a: np.linalg.norm(a[:, :2], axis=1) * scales.U
    s_cfd, s_pinn = speed(case.uvp), speed(pred)
    p_cfd, p_pinn = case.uvp[:, 2] * scales.P, pred[:, 2] * scales.P
    p_norm = lambda: TwoSlopeNorm(0.0, min(p_cfd.min(), -1e-9), max(p_cfd.max(), 1e-9))
    e_p = p_pinn - p_cfd
    e_lim = max(np.abs(e_p).max(), 1e-12)
    panels = [  # (row, col, field, title, cmap, kwargs)
        (0, 0, s_cfd, "CFD  |u| (m/s)", "viridis", dict(vmin=s_cfd.min(), vmax=s_cfd.max())),
        (1, 0, s_pinn, "PINN  |u| (m/s)", "viridis", dict(vmin=s_cfd.min(), vmax=s_cfd.max())),
        (2, 0, np.abs(s_pinn - s_cfd), "|error|  |u| (m/s)", "magma", {}),
        (0, 1, p_cfd, "CFD  gauge p (Pa, outlet = 0)", "RdBu_r", dict(norm=p_norm())),
        (1, 1, p_pinn, "PINN  gauge p (Pa, outlet = 0)", "RdBu_r", dict(norm=p_norm())),
        (2, 1, e_p, "error  p_PINN - p_CFD (Pa)", "PuOr_r", dict(vmin=-e_lim, vmax=e_lim)),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15, 5.2), dpi=140, sharex=True, sharey=True)
    for i, j, f, title, cmap, kw in panels:
        ax = axes[i, j]
        c = ax.tripcolor(tri, f, shading="gouraud", cmap=cmap, **kw)
        fig.colorbar(c, ax=ax, pad=0.01, fraction=0.03)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("x / D")
    fig.suptitle(f"Severity {case.severity:.0%}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_pressure_lines(path, case, pred, scales, tau_pinn=None):
    """Gauge pressure along the centreline and the (upper) wall, CFD vs PINN, plus
    the upper-wall shear stress when the dataset has it. The clearest pressure figure:
    viscous drop, throat minimum, partial recovery."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = case.xy[:, 0], case.xy[:, 1]
    lines = {"centreline (y = 0)": np.abs(y) < 1e-9,
             "upper wall": (case.flag == 3) & (y > 0)}
    has_wss = tau_pinn is not None and case.tau_w is not None
    fig, axes = plt.subplots(3 if has_wss else 2, 1, figsize=(8, 8 if has_wss else 5.5), dpi=140, sharex=True)
    if has_wss:
        xw, yw = x[case.wall_idx], y[case.wall_idx]
        up = yw > 0
        o = np.argsort(xw[up])
        ax = axes[2]
        ax.plot(xw[up][o], case.tau_w[up][o] * scales.P, "k-", lw=1.6, label="CFD")
        ax.plot(xw[up][o], tau_pinn[up][o] * scales.P, "C3--", lw=1.4, label="PINN")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_ylabel("WSS (Pa)")
        ax.set_title("upper-wall shear stress (negative = reversed flow)", fontsize=9)
        ax.grid(alpha=0.3)
    for ax, (name, m) in zip(axes, lines.items()):
        o = np.argsort(x[m])
        xs = x[m][o]
        ax.plot(xs, case.uvp[m, 2][o] * scales.P, "k-", lw=1.6, label="CFD")
        ax.plot(xs, pred[m, 2][o] * scales.P, "C3--", lw=1.4, label="PINN")
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.set_ylabel("gauge p (Pa)")
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("x / D")
    fig.suptitle(f"Pressure along the vessel, severity {case.severity:.0%} (outlet = 0 Pa)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_dp(path, s_grid, dp_grid, cases, test_s, scales):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.plot(np.array(s_grid) * 100, dp_grid * scales.P, label="PINN (continuous)")
    for c in cases:
        held = any(abs(c.severity - t) < 1e-6 for t in test_s)
        ax.plot(c.severity * 100, c.dp() * scales.P, "s" if held else "o",
                color="C3" if held else "C1", label="CFD held-out" if held else "CFD train")
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys())
    ax.set_xlabel("severity (% diameter reduction)")
    ax.set_ylabel("Δp inlet→outlet (Pa)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def evaluate(run_dir, data_dir=None, plots=True, device=None):
    run = Path(run_dir)
    ck_cfg = torch.load(run / "model.pt", map_location="cpu", weights_only=False)["config"]
    device = device or ck_cfg.get("device", "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model, scales, geo, ck = load_checkpoint(run / "model.pt", device=device)
    cfg = ck["config"]
    rlog = RunLog(run, kind="pinn_eval", prefix="eval_", config=dict(data_dir=data_dir, device=str(device)))
    with rlog.stage("load"):
        cases, _, _ = load_cases(data_dir or cfg["data_dir"], cfg["max_symmetry_error"])
    _, test_c = split_cases(cases, cfg["test_severities"], data_s_range=cfg.get("data_s_range") or None)
    test_s = [c.severity for c in test_c]

    rows = []
    for c in cases:
        with rlog.stage("metrics"):
            m = rel_l2(model, c, device)
            pred = predict(model, c.inputs().to(device))
            tau_p = pinn_wss(model, c, geo, scales.Re, device) if c.tau_w is not None else None
            wm = wss_metrics(c, tau_p, geo) if tau_p is not None else {}
        dp_cfd = c.dp() * scales.P
        dp_pinn = (pred[c.flag == 1, 2].mean() - pred[c.flag == 2, 2].mean()) * scales.P
        # Area-weighted counterparts of l2_vel / l2_p (guide §7.3; docs/DEVIATIONS.md D-009).
        w = node_areas(c.xy, c.triangles)
        m_area = dict(l2_vel_area=rel_l2_weighted(pred[:, :2], c.uvp[:, :2], w),
                      l2_p_area=rel_l2_weighted(pred[:, 2], c.uvp[:, 2], w))
        rows.append(dict(severity=c.severity, split="test" if c.severity in test_s else "train",
                         **{f"l2_{k}": v for k, v in m.items()}, **m_area,
                         **pressure_metrics(c, pred), **wm,
                         dp_cfd_pa=float(dp_cfd), dp_pinn_pa=float(dp_pinn),
                         dp_rel_err=float(abs(dp_pinn - dp_cfd) / abs(dp_cfd))))
        if plots:
            with rlog.stage("plots"):
                tag = round(c.severity * 100)
                _plot_case(run / f"fields_severity_{tag}.png", c, pred, scales)
                _plot_pressure_lines(run / f"pressure_lines_severity_{tag}.png", c, pred, scales, tau_p)

    s_grid = np.linspace(*ck["s_range"], 51)
    if plots:
        _plot_dp(run / "dp_vs_severity.png", s_grid, _pinn_dp(model, geo, s_grid, device), cases, test_s, scales)

    # Inference timing on one full CFD node set: warm-up, then the median of 10 repeats
    # (CUDA synchronised so asynchronous kernels are not under-counted).
    x = cases[0].inputs().to(device)
    predict(model, x)
    _sync(device)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        predict(model, x)
        _sync(device)
        times.append(time.perf_counter() - t0)
    t_inf = float(np.median(times))
    # WSS by autodiff on one case (the derivative costs extra; guide §7.9 item 8)
    t_wss = float("nan")
    if cases[0].tau_w is not None:
        t0 = time.perf_counter()
        pinn_wss(model, cases[0], geo, scales.Re, device)
        _sync(device)
        t_wss = time.perf_counter() - t0
    rlog.counters.update(inference_s_median=t_inf, inference_s_all=times, wss_autodiff_s=t_wss,
                         nodes=len(x), device=str(device))

    summary = dict(rows=rows, inference_s=t_inf, inference_wss_s=t_wss, nodes=len(x), device=str(device),
                   step=ck["step"], lambdas=ck["lambdas"])
    (run / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"{'sev':>5} {'split':>5} {'vel L2':>8} {'u L2':>8} {'v L2':>8} {'p L2':>8} {'p-mean':>8} "
          f"{'rms/dp':>7} {'off/dp':>7} {'dp CFD':>9} {'dp PINN':>9} {'dp err':>7}")
    for r in rows:
        print(f"{r['severity']:5.2f} {r['split']:>5} {r['l2_vel']:8.2%} {r['l2_u']:8.2%} {r['l2_v']:8.2%} "
              f"{r['l2_p']:8.2%} {r['l2_p_meanfree']:8.2%} {r['p_rms_over_dp']:7.2%} {r['p_offset_over_dp']:+7.2%} "
              f"{r['dp_cfd_pa']:8.2f}Pa {r['dp_pinn_pa']:8.2f}Pa {r['dp_rel_err']:7.1%}")
    if any("l2_wss" in r for r in rows):
        print(f"{'sev':>5} {'split':>5} {'WSS L2':>8} {'peak err':>9} {'L_rec CFD':>10} {'L_rec PINN':>10}  (L in D)")
        for r in rows:
            if "l2_wss" in r:
                print(f"{r['severity']:5.2f} {r['split']:>5} {r['l2_wss']:8.2%} {r['wss_peak_rel_err']:9.2%} "
                      f"{r['recirc_len_cfd_D']:10.3f} {r['recirc_len_pinn_D']:10.3f}")
    print(f"PINN inference on {len(x)} nodes ({device}): {t_inf * 1e3:.1f} ms (median of 10); "
          f"WSS autodiff: {t_wss * 1e3:.1f} ms")
    rlog.finish(metrics=dict(n_cases=len(rows)))
    return summary

"""Evaluate a trained run against every CFD case (train and held-out)."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_cases, split_cases
from .train import load_checkpoint, predict, rel_l2


def _pinn_dp(model, geo, severities, n=201):
    """Nondimensional Δp = mean p(inlet) - mean p(outlet) along the cross-sections."""
    y = torch.linspace(-geo.R, geo.R, n)
    out = []
    for s in severities:
        line = lambda x: torch.stack([torch.full_like(y, x), y, torch.full_like(y, s)], dim=1)
        out.append(predict(model, line(0.0))[:, 2].mean() - predict(model, line(geo.length))[:, 2].mean())
    return np.array(out)


def _plot_case(path, case, pred, scales):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(case.xy[:, 0], case.xy[:, 1], case.triangles)
    speed = lambda a: np.linalg.norm(a[:, :2], axis=1) * scales.U
    fields = {
        "|u| (m/s)": (speed(case.uvp), speed(pred)),
        "p (Pa)": (case.uvp[:, 2] * scales.P, pred[:, 2] * scales.P),
    }
    fig, axes = plt.subplots(3, 2, figsize=(15, 5.2), dpi=140, sharex=True, sharey=True)
    for j, (name, (cfd, pinn)) in enumerate(fields.items()):
        lo, hi = cfd.min(), cfd.max()
        for i, (lab, f, kw) in enumerate((("CFD", cfd, dict(vmin=lo, vmax=hi)),
                                          ("PINN", pinn, dict(vmin=lo, vmax=hi)),
                                          ("|error|", np.abs(pinn - cfd), {}))):
            ax = axes[i, j]
            c = ax.tripcolor(tri, f, shading="gouraud", cmap="magma" if i == 2 else "viridis", **kw)
            fig.colorbar(c, ax=ax, pad=0.01, fraction=0.03)
            ax.set_aspect("equal")
            ax.set_title(f"{lab}  {name}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("x / D")
    fig.suptitle(f"Severity {case.severity:.0%}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
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


def evaluate(run_dir, data_dir=None, plots=True):
    run = Path(run_dir)
    model, scales, geo, ck = load_checkpoint(run / "model.pt")
    cfg = ck["config"]
    cases, _, _ = load_cases(data_dir or cfg["data_dir"], cfg["max_symmetry_error"])
    _, test_c = split_cases(cases, cfg["test_severities"])
    test_s = [c.severity for c in test_c]

    rows = []
    for c in cases:
        m = rel_l2(model, c)
        pred = predict(model, c.inputs())
        dp_cfd = c.dp() * scales.P
        dp_pinn = (pred[c.flag == 1, 2].mean() - pred[c.flag == 2, 2].mean()) * scales.P
        rows.append(dict(severity=c.severity, split="test" if c.severity in test_s else "train",
                         **{f"l2_{k}": v for k, v in m.items()},
                         dp_cfd_pa=float(dp_cfd), dp_pinn_pa=float(dp_pinn),
                         dp_rel_err=float(abs(dp_pinn - dp_cfd) / abs(dp_cfd))))
        if plots:
            _plot_case(run / f"fields_severity_{round(c.severity * 100)}.png", c, pred, scales)

    s_grid = np.linspace(*ck["s_range"], 51)
    if plots:
        _plot_dp(run / "dp_vs_severity.png", s_grid, _pinn_dp(model, geo, s_grid), cases, test_s, scales)

    # Inference timing on one full CFD node set (PINN_mvp phase-3 style benchmark).
    x = cases[0].inputs()
    predict(model, x)
    t0 = time.perf_counter()
    for _ in range(10):
        predict(model, x)
    t_inf = (time.perf_counter() - t0) / 10

    summary = dict(rows=rows, inference_s=t_inf, nodes=len(x), step=ck["step"], lambdas=ck["lambdas"])
    (run / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"{'sev':>5} {'split':>5} {'vel L2':>8} {'u L2':>8} {'v L2':>8} {'p L2':>8} "
          f"{'dp CFD':>9} {'dp PINN':>9} {'dp err':>7}")
    for r in rows:
        print(f"{r['severity']:5.2f} {r['split']:>5} {r['l2_vel']:8.2%} {r['l2_u']:8.2%} {r['l2_v']:8.2%} "
              f"{r['l2_p']:8.2%} {r['dp_cfd_pa']:8.2f}Pa {r['dp_pinn_pa']:8.2f}Pa {r['dp_rel_err']:7.1%}")
    print(f"PINN inference on {len(x)} nodes: {t_inf * 1e3:.1f} ms")
    return summary

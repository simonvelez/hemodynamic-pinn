"""Data-efficiency study: PINN accuracy against the number of CFD runs used for training (Re ≈ 150).

Answers "how many CFD runs does the PINN need?". Runs come from scripts/campaign.py:
  K = 0      physonly                 (physics only, no CFD data)
  K = 1..5   ncases_K{K}_s{seed}      hybrid (data + Navier-Stokes residual), 3 seeds
             ncases_K{K}_dataonly     same network and data, no physics (lambda_pde = 0)
  K = 7      base_re150_s* / dataonly (the default split)
0.35 and 0.55 are held out in every run, so the "common test" columns compare all K on the same cases.
"All held-out" is the worst case over every non-training symmetric severity (for K = 1 this includes
0.20 / 0.60, outside the data but inside the physics range).

Per run, the error on a set of severities is the mean over those severities (worst case: max).
Across seeds: median, with min-max in brackets. C_data is the logged wall time of the CFD runs used
(dataset *_cost.json); C_train is the PINN training wall time (cost.json).

    python scripts/data_efficiency.py                              # runs/campaign
    python scripts/data_efficiency.py --runs-dir runs/campaign_smoke
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfd_vs_pinn import dataset_info  # noqa: E402
from compare_runs import load  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMMON_TEST = (0.35, 0.55)
# (metrics key, label); dp_rel_err is taken in absolute value
QOIS = [("l2_vel", "velocity L2"), ("l2_vel_area", "velocity L2 (area)"), ("dp_rel_err", "Δp error"),
        ("l2_p_meanfree", "pressure L2 (mean-free)"), ("l2_wss", "WSS L2")]
PLOT_QOIS = [("l2_vel", "Velocity L2 error"), ("dp_rel_err", "Δp error"), ("l2_wss", "WSS L2 error")]
COLORS = {"hybrid": "#2a78d6", "data-only": "#eb6834", "physics-only": "#6b6a64"}   # reference palette slots 1, 2, neutral


def groups(rd):
    """{(method, K): [run dirs]} for every pattern that has evaluated runs."""
    pats = {("physics-only", 0): "physonly", ("hybrid", 7): "base_re150_s*", ("data-only", 7): "dataonly"}
    for k in range(1, 6):
        pats[("hybrid", k)] = f"ncases_K{k}_s*"
        pats[("data-only", k)] = f"ncases_K{k}_dataonly"
    out = {}
    for key, pat in pats.items():
        runs = sorted(p for p in glob.glob(str(rd / pat)) if (Path(p) / "metrics.json").exists())
        if runs:
            out[key] = runs
    return out


def run_summary(run):
    rows = load(run)
    cfg = json.loads((Path(run) / "config.json").read_text())
    cost = json.loads((Path(run) / "cost.json").read_text())
    train = sorted(s for s, r in rows.items() if r["split"] == "train")
    used = cfg["lambda_data"] > 0 and cfg["n_sensors"] > 0
    _, data_cost, _ = dataset_info(cfg["data_dir"])
    c_data = sum(data_cost.get(round(s, 2), np.nan) for s in train) if used else 0.0
    monitor = cost["stages"].get("monitor_eval", {}).get("wall_s", 0.0)
    out = dict(k=len(train) if used else 0, train=train if used else [], c_data=c_data,
               c_train=cost["total_wall_s"] - monitor)
    held = [s for s, r in rows.items() if r["split"] == "test"]
    common = [s for s in held if any(abs(s - t) < 1e-6 for t in COMMON_TEST)]
    for key, _ in QOIS:
        def val(s):
            v = rows[s].get(key)
            return np.nan if v is None else abs(float(v))
        c = [val(s) for s in common]
        a = [val(s) for s in held]
        out[key + "_common"] = float(np.mean(c)) if c and np.isfinite(c).all() else np.nan
        out[key + "_worst"] = float(np.max(a)) if a and np.isfinite(a).all() else np.nan
    return out


def stat(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if not len(v):
        return np.nan, np.nan, np.nan
    return float(np.median(v)), float(v.min()), float(v.max())


def fmt(vals):
    med, lo, hi = stat(vals)
    if not np.isfinite(med):
        return "–"
    return f"{med:.2%}" if lo == hi else f"{med:.2%} [{lo:.2%}–{hi:.2%}]"


def plot(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
    fig, axes = plt.subplots(1, len(PLOT_QOIS), figsize=(4.2 * len(PLOT_QOIS), 3.6), constrained_layout=True)
    for ax, (key, title) in zip(np.atleast_1d(axes), PLOT_QOIS):
        for method, marker in (("hybrid", "o"), ("data-only", "s")):
            pts = sorted((k, stat([r[key + "_common"] for r in rs])) for (m, k), rs in table.items() if m == method)
            pts = [(k, s) for k, s in pts if np.isfinite(s[0])]
            if not pts:
                continue
            k = np.array([p[0] for p in pts])
            med = np.array([p[1][0] for p in pts])
            err = np.array([[p[1][0] - p[1][1], p[1][2] - p[1][0]] for p in pts]).T
            ax.errorbar(k, med, yerr=err, color=COLORS[method], marker=marker, ms=7, lw=2, capsize=3,
                        mec="white", mew=1.5, label=method)
        if ("physics-only", 0) in table:   # no CFD data: a single point, not part of either line
            med = stat([r[key + "_common"] for r in table[("physics-only", 0)]])[0]
            if np.isfinite(med):
                ax.plot([0], [med], marker="D", ms=7, ls="none", color=COLORS["physics-only"], mec="white",
                        mew=1.5, label="physics-only")
        ax.set_yscale("log")
        ax.set_xticks([0, 1, 2, 3, 4, 5, 7])
        ax.set_xlim(-0.4, 8.2)
        ax.set_xlabel("CFD runs used for training (K)", color=muted)
        ax.set_title(title + " on held-out 35 % / 55 %", fontsize=10, color=ink, loc="left")
        pct = matplotlib.ticker.FuncFormatter(lambda y, _: f"{100 * y:.3g} %")
        ax.yaxis.set_major_formatter(pct)
        ax.yaxis.set_minor_formatter(pct)
        ax.grid(True, which="major", color=grid, lw=0.8)
        ax.tick_params(which="both", colors=muted, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9)
    fig.text(0.0, -0.04, "Re ≈ 150. Markers: median over seeds; bars: min–max. K = 0 is the physics-only PINN.",
             fontsize=8, color=muted)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(ROOT / "runs" / "campaign"))
    ap.add_argument("--out", default=None, help="markdown path (default <runs-dir>/data_efficiency.md); "
                                                "the plot goes next to it as .png")
    a = ap.parse_args()
    rd = Path(a.runs_dir)
    if not rd.is_absolute():
        rd = (ROOT / rd).resolve()
    g = groups(rd)
    if not g:
        raise SystemExit(f"no evaluated data-efficiency runs under {rd}")
    table = {key: [run_summary(r) for r in runs] for key, runs in g.items()}

    lines = ["# Data efficiency: PINN error against the number of CFD runs (Re ≈ 150)", "",
             f"Generated by `PINN/scripts/data_efficiency.py` from `{rd}`. Common test = 0.35 and 0.55 (held out "
             "in every run; mean over the two). Worst = max over all held-out symmetric severities. Median over "
             "seeds, [min–max]. Physics covers 0.20–0.60 in every run except data-only.", ""]
    head = "| K | method (runs) | training severities | " + " | ".join(
        f"{lab} common | {lab} worst" for _, lab in QOIS) + " | C_data (s) | C_train (s) |"
    lines += [head, "|" + "---|" * (3 + 2 * len(QOIS) + 2)]
    order = sorted(table, key=lambda km: (km[1], ["physics-only", "hybrid", "data-only"].index(km[0])))
    for method, k in order:
        rs = table[(method, k)]
        train = ", ".join(f"{s:.2f}" for s in rs[0]["train"]) or "none"
        cells = []
        for key, _ in QOIS:
            cells += [fmt([r[key + "_common"] for r in rs]), fmt([r[key + "_worst"] for r in rs])]
        c_data = np.median([r["c_data"] for r in rs])
        c_train = np.median([r["c_train"] for r in rs])
        lines.append(f"| {k} | {method} ({len(rs)}) | {train} | " + " | ".join(cells)
                     + f" | {c_data:.0f} | {c_train:.0f} |")
    text = "\n".join(lines)
    print(text)
    out = Path(a.out) if a.out else rd / "data_efficiency.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    plot(table, out.with_suffix(".png"))
    print(f"\nwrote {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()

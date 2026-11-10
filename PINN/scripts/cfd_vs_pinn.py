"""Accuracy-matched CFD baseline and break-even query count (guide §7.7-7.9; thesis objective 4).

For every case in the CFD cost study (ns_cfd/cost_study.py report -> pareto.csv) and every PINN group,
and for each quantity (velocity L2, area-weighted velocity L2, Δp error, peak-WSS error):

  1. PINN error at that case: median over the group's runs (seeds), from metrics.json.
  2. t_B: wall-clock of the CHEAPEST CFD configuration whose error is <= the PINN error
     (mesh + solve + export, median of repeats). That is the fair baseline (McGreivy & Hakim 2024),
     not the fine reference. If even the coarsest CFD cell is more accurate than the PINN, t_B is the
     cheapest cell's time (the PINN cannot be matched from below: a lower bound on CFD's advantage).
     If the PINN beats every cell, t_B is undefined within the matrix.
  3. N* = (C_data + C_train) / (t_B - t_inf), defined only if t_B > t_inf, where
     C_data  = n_train x the cost-study time of the dataset configuration (250x40, P2P1, steady) at the
               same u_max: same machine and procedure as t_B. The dataset's own *_cost.json sum is shown
               as a separate column (other machine; its runs also include the symmetry-breaking
               imperfection continuation, so they are slower). Falls back to it if the cell is missing,
     C_train = PINN training wall time (cost.json total minus monitoring evaluations),
     t_inf   = PINN inference on one full node set (+ WSS autodiff for the WSS quantity).
     Tuning cost (C_tune) is not included; the guide's conservative form adds it.
     When even the cheapest CFD cell is more accurate than the PINN, the true equal-accuracy t_B is at
     most that cell's time, so N* is a lower bound and printed as ">= N*".

    python scripts/cfd_vs_pinn.py                                   # default groups under runs/campaign
    python scripts/cfd_vs_pinn.py --group hybrid=runs/campaign/base_re150_s* --group dataonly=runs/campaign/dataonly
    python scripts/cfd_vs_pinn.py --pareto ../ns_cfd/cost_study/report/pareto.csv --out runs/campaign/cfd_vs_pinn.md
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUPS = ["hybrid=runs/campaign/base_re*_s*", "data-only=runs/campaign/dataonly",
                  "physics-only=runs/campaign/physonly"]
# (PINN metrics key, pareto.csv key, label, PINN inference includes WSS autodiff)
DATASET_CELL = dict(mesh="250x40", element="P2P1", solve="steady")
QOIS = [("l2_vel", "l2_vel", "velocity L2 (node)", False),
        ("l2_vel_area", "l2_vel_area", "velocity L2 (area)", False),
        ("dp_rel_err", "dp_rel_err", "Δp error", False),
        ("wss_peak_rel_err", "wss_peak_rel_err", "peak WSS error", True)]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_pareto(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        sev, um = r["case"].split("_")            # e.g. "s35_u10"
        r["severity"], r["u_max"] = int(sev[1:]) / 100, int(um[1:]) / 100
    return rows


def dataset_info(data_dir):
    """u_max of a dataset directory and per-severity CFD wall time (its *_cost.json files)."""
    d = Path(data_dir)
    if not d.is_absolute():
        d = (ROOT / d).resolve()
    npz = sorted(d.glob("pinn2d_severity_*.npz"))
    meta = json.loads(str(np.load(npz[0])["meta"]))
    cost, cpu = {}, None
    for f in d.glob("pinn2d_severity_*_cost.json"):
        c = json.loads(f.read_text())
        cost[int(f.name.split("_")[2]) / 100] = c["total_wall_s"]
        cpu = c.get("cpu_model", cpu)
    return meta["params"]["u_max"], cost, cpu


def load_run(run):
    run = Path(run)
    m = json.loads((run / "metrics.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    cost = json.loads((run / "cost.json").read_text())
    u_max, data_cost, data_cpu = dataset_info(cfg["data_dir"])
    train_s = [r["severity"] for r in m["rows"] if r["split"] == "train"]
    used = cfg["lambda_data"] > 0 and cfg["n_sensors"] > 0
    c_data = sum(data_cost.get(round(s, 2), float("nan")) for s in train_s) if used else 0.0
    n_train = len(train_s) if used else 0
    monitor = cost["stages"].get("monitor_eval", {}).get("wall_s", 0.0)
    return dict(name=run.name, u_max=u_max, rows={round(r["severity"], 4): r for r in m["rows"]},
                t_inf=m["inference_s"], t_wss=_f(m.get("inference_wss_s")), c_train=cost["total_wall_s"] - monitor,
                c_data_dataset=c_data, n_train=n_train, pinn_hw=f"{cost.get('cpu_model')} / GPU {cost.get('gpu')} ({m['device']})",
                data_cpu=data_cpu)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pareto", default=str(ROOT.parent / "ns_cfd" / "cost_study" / "report" / "pareto.csv"))
    ap.add_argument("--group", action="append", default=None, help="NAME=GLOB of PINN run dirs (pooled)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "campaign" / "cfd_vs_pinn.md"))
    a = ap.parse_args()

    pareto = load_pareto(a.pareto)
    groups = {}
    for g in a.group or DEFAULT_GROUPS:
        name, pattern = g.split("=", 1)
        pattern = pattern if Path(pattern).is_absolute() else str(ROOT / pattern)
        runs = [load_run(p) for p in sorted(glob.glob(pattern)) if (Path(p) / "metrics.json").exists()]
        if runs:
            groups[name] = runs
    if not groups:
        raise SystemExit("no evaluated PINN runs found")

    lines = ["# CFD vs PINN at equal accuracy, and break-even", "",
             "Generated by `PINN/scripts/cfd_vs_pinn.py` from `" + a.pareto + "` and the PINN runs' "
             "`metrics.json` / `cost.json`. Errors: PINN vs the CFD dataset mesh (250×40); CFD cells vs the "
             "700×112 reference (docs/DEVIATIONS.md D-015). PINN numbers are medians over each group's runs.", ""]
    hw = {r["pinn_hw"] for rs in groups.values() for r in rs}
    cfd_hw = {r["data_cpu"] for rs in groups.values() for r in rs}
    lines += [f"- PINN hardware: {'; '.join(sorted(map(str, hw)))}",
              f"- CFD dataset hardware (C_data): {'; '.join(sorted(map(str, cfd_hw)))}",
              "- CFD cost-study timings: single core (OMP_NUM_THREADS=1), on the machine where `cost_study.py batch` ran",
              "- N* = (C_data + C_train) / (t_B − t_inf); C_tune not included. C_data = n_train × the cost-study "
              "time of the dataset configuration (250×40, P2P1, steady) at the same u_max, so it shares t_B's "
              "machine and procedure; 'C_data (dataset)' is the logged time of the actual dataset runs (other "
              "machine, includes the imperfection continuation). '≥' marks a lower bound (the cheapest CFD "
              "cell is already more accurate than the PINN).",
              "- Units: wall-clock seconds; the PINN and the CFD run on different hardware (state this with any "
              "speed-up claim).", ""]
    lines += ["| case | PINN group (runs) | split | quantity | PINN error | CFD baseline (mesh, element, solve) | "
              "CFD error | t_B (s) | t_inf (s) | C_data (s) | C_data (dataset) (s) | C_train (s) | N* |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    cases = sorted({(r["severity"], r["u_max"]) for r in pareto})
    for sev, um in cases:
        cells = [r for r in pareto if r["severity"] == sev and r["u_max"] == um]
        ds_cells = [_f(r["total_s_median"]) for r in pareto
                    if r["u_max"] == um and all(r[k] == v for k, v in DATASET_CELL.items())]
        t_case = float(np.median(ds_cells)) if ds_cells else float("nan")
        for gname, runs in groups.items():
            rs = [r for r in runs if abs(r["u_max"] - um) < 1e-12 and round(sev, 4) in r["rows"]]
            if not rs:
                continue
            split = rs[0]["rows"][round(sev, 4)]["split"]
            c_data_ds = float(np.median([r["c_data_dataset"] for r in rs]))
            c_data = float(np.median([r["n_train"] for r in rs])) * t_case
            if not np.isfinite(c_data):
                c_data = c_data_ds         # dataset cell missing from the cost study
            c_train = float(np.median([r["c_train"] for r in rs]))
            for key_p, key_c, label, wss in QOIS:
                errs = [_f(r["rows"][round(sev, 4)].get(key_p)) for r in rs]
                if not np.isfinite(errs).any():
                    continue
                e_pinn = float(np.nanmedian(errs))
                t_inf = float(np.median([r["t_inf"] + (r["t_wss"] if wss else 0.0) for r in rs]))
                ok = [c for c in cells if np.isfinite(_f(c.get(key_c)))]
                if not ok:
                    continue
                match = [c for c in ok if _f(c[key_c]) <= e_pinn]
                if match:
                    best, note = min(match, key=lambda c: _f(c["total_s_median"])), ""
                else:
                    best, note = None, "PINN more accurate than every CFD cell"
                if best is None:
                    lines.append(f"| s{sev:.2f} u{um:g} | {gname} ({len(rs)}) | {split} | {label} | {e_pinn:.2%} | "
                                 f"{note} | – | – | {t_inf:.3g} | {c_data:.0f} | {c_data_ds:.0f} | {c_train:.0f} | undefined |")
                    continue
                t_b = _f(best["total_s_median"])
                bound = best is min(ok, key=lambda c: _f(c["total_s_median"]))
                if bound:
                    note = " (cheapest cell already more accurate)"
                n_star = (c_data + c_train) / (t_b - t_inf) if t_b > t_inf else float("nan")
                lines.append(
                    f"| s{sev:.2f} u{um:g} | {gname} ({len(rs)}) | {split} | {label} | {e_pinn:.2%} | "
                    f"{best['mesh']}, {best['element']}, {best['solve']}{note} | {_f(best[key_c]):.2%} | {t_b:.3g} | "
                    f"{t_inf:.3g} | {c_data:.0f} | {c_data_ds:.0f} | {c_train:.0f} | "
                    f"{(('≥ ' if bound else '') + f'{n_star:,.0f}') if np.isfinite(n_star) else 'undefined (t_inf ≥ t_B)'} |")
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()

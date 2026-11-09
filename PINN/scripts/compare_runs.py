"""Compare evaluated runs: per-severity table, mean ± std over groups (e.g. seeds).

    python scripts/compare_runs.py runs/full_re150_lmax10 runs/physics_only_re150
    python scripts/compare_runs.py --group hybrid=runs/full_re150_lmax10* --group physics=runs/physics_only_re150
    python scripts/compare_runs.py ... --out runs/comparison_re150.md

Each run directory must contain metrics.json (written by scripts/evaluate.py).
With --group NAME=GLOB, runs matching GLOB are pooled and shown as mean ± std.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

COLS = [("l2_vel", "vel L2"), ("l2_p", "p L2"), ("p_rms_over_dp", "p RMS/Δp"), ("dp_rel_err", "Δp err"),
        ("l2_wss", "WSS L2")]   # metrics missing from older runs are shown as "–"


def load(run):
    rows = json.loads((Path(run) / "metrics.json").read_text())["rows"]
    return {round(r["severity"], 4): r for r in rows}


def fmt(vals):
    vals = np.asarray(vals) * 100
    if len(vals) == 1:
        return f"{vals[0]:.2f} %"
    return f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f} %"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories (each becomes its own group)")
    ap.add_argument("--group", action="append", default=[], help="NAME=GLOB (pooled runs)")
    ap.add_argument("--out", help="write the markdown table to this file")
    a = ap.parse_args()

    groups = {Path(r).name: [r] for r in a.runs}
    for g in a.group:
        name, pattern = g.split("=", 1)
        groups[name] = sorted(p for p in glob.glob(pattern) if (Path(p) / "metrics.json").exists())
    groups = {k: v for k, v in groups.items() if v}
    if not groups:
        raise SystemExit("no evaluated runs found")

    data = {k: [load(r) for r in v] for k, v in groups.items()}
    sevs = sorted(set().union(*[set(m) for ms in data.values() for m in ms]))
    lines = ["Groups: " + "; ".join(f"**{k}** = {len(v)} run(s): {', '.join(Path(r).name for r in v)}"
                                    for k, v in groups.items()), ""]
    header = "| severity | split | " + " | ".join(f"{k}: {lab}" for k in groups for _, lab in COLS) + " |"
    lines += [header, "|" + "---|" * (2 + len(groups) * len(COLS))]
    for s in sevs:
        split = next((m[s]["split"] for ms in data.values() for m in ms if s in m), "")
        cells = []
        for k in groups:
            for key, _ in COLS:
                vals = [m[s][key] for m in data[k] if s in m and m[s].get(key) is not None]
                cells.append(fmt(vals) if vals else "–")
        lines.append(f"| {s:.2f} | {'**test**' if split == 'test' else split} | " + " | ".join(cells) + " |")
    # overall means per split
    for split in ("train", "test"):
        cells = []
        for k in groups:
            for key, _ in COLS:
                per_run = [[m[s][key] for s in m if m[s]["split"] == split and m[s].get(key) is not None]
                           for m in data[k]]
                per_run = [np.mean(v) for v in per_run if v]
                cells.append(fmt(per_run) if per_run else "–")
        lines.append(f"| mean | {split} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()

"""Comparison + robustness campaign (to run on the GPU machine, e.g. NVIDIA A4000).

Every run is a named set of train.py flags. The campaign is idempotent: a run whose
model.pt and metrics.json both exist is skipped, so it can be stopped and restarted.
Each run is trained, then evaluated, and all results are summarised by compare_runs.py.

    python scripts/campaign.py --dry-run                 # list runs and commands
    python scripts/campaign.py --device cuda             # run everything not yet done
    python scripts/campaign.py --only base_re150 noise   # run groups whose name starts with these
    python scripts/campaign.py --smoke                   # 200-step version of every run (pipeline check)

The CFD side of the comparison is ns_cfd/cost_study.py (run it on the same machine, CPU, while
the GPU is idle, so the timings don't interfere).

Groups (Re ≈ 150 unless stated; default config otherwise, see pinn/config.py):
  base_re150_sK   5 seeds (init + sensors + collocation): seed variance of the headline numbers
  base_re301_sK   3 seeds at Re ≈ 301
  sensors_N       250 / 100 / 50 sensors per training case (default 500)
  noise_P         1 / 5 / 10 % Gaussian noise on the sensor velocities
  dataonly        lambda_pde = 0: same network, data and BCs, no Navier-Stokes residual
  physonly        lambda_data = 0: standard physics-only PINN
  extrap_*        data only at 0.25-0.50; tests 0.20 / 0.55 / 0.60 are outside the data.
                  'extrap_physin'  physics still enforced over 0.20-0.60 (PINN advantage)
                  'extrap_physout' physics also restricted to 0.25-0.50 (pure extrapolation)
  f64             float64 throughout (precision check, guide §6.3)
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "campaign"
RE150 = ["--data-dir", "../ns_cfd/output_2d_re150", "--test-severities", "0.35", "0.55"]
RE301 = ["--data-dir", "../ns_cfd/output_2d", "--test-severities", "0.35", "0.45"]


def campaign():
    runs = {}
    for k in range(5):
        runs[f"base_re150_s{k}"] = RE150 + ["--seed", str(k)]
    for k in range(3):
        runs[f"base_re301_s{k}"] = RE301 + ["--seed", str(k)]
    for n in (250, 100, 50):
        runs[f"sensors_{n}"] = RE150 + ["--n-sensors", str(n)]
    for pct in (1, 5, 10):
        runs[f"noise_{pct}pct"] = RE150 + ["--sensor-noise", str(pct / 100)]
    runs["dataonly"] = RE150 + ["--lambda-pde", "0"]
    runs["physonly"] = RE150 + ["--lambda-data", "0"]
    ext = ["--data-dir", "../ns_cfd/output_2d_re150", "--test-severities", "0.35",
           "--data-s-range", "0.25", "0.50"]
    runs["extrap_physin"] = ext
    runs["extrap_physout"] = ext + ["--physics-s-range", "0.25", "0.50"]
    runs["f64"] = RE150 + ["--dtype", "float64"]
    return runs


def done(out):
    return (out / "model.pt").exists() and (out / "metrics.json").exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="run names starting with any of these")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="200 Adam + 20 L-BFGS steps, into runs/campaign_smoke")
    a = ap.parse_args()

    base = ROOT / "runs" / ("campaign_smoke" if a.smoke else "campaign")
    todo = {k: v for k, v in campaign().items() if not a.only or any(k.startswith(p) for p in a.only)}
    py = sys.executable
    for name, flags in todo.items():
        out = base / name
        extra = ["--steps", "200", "--lbfgs-steps", "20", "--eval-every", "100"] if a.smoke else []
        train = [py, "scripts/train.py", "--out-dir", str(out), "--device", a.device,
                 "--threads", str(a.threads), *flags, *extra]
        ev = [py, "scripts/evaluate.py", "--run", str(out), "--device", a.device]
        status = "done" if done(out) else "todo"
        print(f"[{status}] {name}\n    {shlex.join(train)}\n    {shlex.join(ev)}", flush=True)
        if a.dry_run or status == "done":
            continue
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "train.log", "w") as log:
            rc = subprocess.run(train, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc == 0:
                rc = subprocess.run(ev, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        msg = "OK" if rc == 0 else f"FAILED (exit {rc}), see {out / 'train.log'}"
        print(f"    -> {msg}", flush=True)

    if not a.dry_run:
        groups = [("hybrid (seeds)", "base_re150_s*"), ("data-only", "dataonly"), ("physics-only", "physonly"),
                  ("noise 1%", "noise_1pct"), ("noise 5%", "noise_5pct"), ("noise 10%", "noise_10pct"),
                  ("250 sensors", "sensors_250"), ("100 sensors", "sensors_100"), ("50 sensors", "sensors_50")]
        args = [py, "scripts/compare_runs.py", "--out", str(base / "comparison_re150.md")]
        for label, pattern in groups:
            if any(done(p) for p in base.glob(pattern)):
                args += ["--group", f"{label}={base / pattern}"]
        if "--group" in args:
            subprocess.run(args, cwd=ROOT)


if __name__ == "__main__":
    main()

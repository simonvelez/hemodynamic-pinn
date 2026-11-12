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
  ncases_K*       data-efficiency study: K = 1-5 CFD runs (severities) with data, 0.35 / 0.55 always
                  held out, physics over 0.20-0.60. 'ncases_K{K}_s{seed}' hybrid (3 seeds),
                  'ncases_K{K}_dataonly' without physics. K = 7 is base_re150_s*, K = 0 is physonly.
                  Summary: scripts/data_efficiency.py
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
# symmetric Re ≈ 150 cases (0.65 / 0.70 are the asymmetric branch and are never loaded)
RE150_SYM = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
COMMON_TEST = [0.35, 0.55]
# training severities per number of CFD runs K (data-efficiency study)
NCASES = {1: [0.40], 2: [0.20, 0.60], 3: [0.20, 0.40, 0.60], 4: [0.20, 0.30, 0.45, 0.60],
          5: [0.20, 0.30, 0.40, 0.50, 0.60]}
assert all(not set(v) & set(COMMON_TEST) and len(v) == k for k, v in NCASES.items())


def ncases_flags(train):
    test = [s for s in RE150_SYM if all(abs(s - t) > 1e-9 for t in train)]
    return ["--data-dir", "../ns_cfd/output_2d_re150", "--test-severities", *[f"{s:.2f}" for s in test]]


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
    for k, train in NCASES.items():
        for seed in range(3):
            runs[f"ncases_K{k}_s{seed}"] = ncases_flags(train) + ["--seed", str(seed)]
        runs[f"ncases_K{k}_dataonly"] = ncases_flags(train) + ["--lambda-pde", "0"]
    runs.update(round2())
    return runs


# Round 2 (2026-11-12), in priority order: questions raised by the round-1 results.
# Run with:  python scripts/campaign.py --only r2_ --device cuda
LONG = ["--lbfgs-steps", "10000", "--lbfgs-rounds", "5"]                 # 5x L-BFGS, fresh points per round
ADAM_LONG = ["--steps", "60000", "--lr-step", "12000"] + LONG             # 3x Adam on top
PLACE = {2: [0.45, 0.60], 3: [0.30, 0.45, 0.60]}                          # CFD runs placed at high severity


def round2():
    r = {}
    # 1. Fair baselines: both networks also see pressure at the sensors (round 1's data-only never did).
    for s in range(3):
        r[f"r2_press_K7_hybrid_s{s}"] = RE150 + ["--data-pressure", "--seed", str(s)]
        r[f"r2_press_K7_dataonly_s{s}"] = RE150 + ["--data-pressure", "--lambda-pde", "0", "--seed", str(s)]
    for k in (3, 4):
        r[f"r2_press_K{k}_hybrid_s0"] = ncases_flags(NCASES[k]) + ["--data-pressure"]
        r[f"r2_press_K{k}_dataonly_s0"] = ncases_flags(NCASES[k]) + ["--data-pressure", "--lambda-pde", "0"]
    # 2. Training budget: were the sparse-data and physics-only runs simply under-trained?
    for s in range(2):
        r[f"r2_budget_physonly_long_s{s}"] = RE150 + ["--lambda-data", "0", "--seed", str(s)] + LONG
    for k in (1, 2, 3):
        for s in range(3):
            r[f"r2_budget_K{k}_long_s{s}"] = ncases_flags(NCASES[k]) + ["--seed", str(s)] + LONG
    r["r2_budget_K7_long_s0"] = RE150 + LONG
    r["r2_budget_K3_adamlong_s0"] = ncases_flags(NCASES[3]) + ADAM_LONG
    r["r2_budget_physonly_adamlong_s0"] = RE150 + ["--lambda-data", "0"] + ADAM_LONG
    # 3. Physics weight with sparse data: every round-1 lambda sat at the cap of 10.
    for lm in ("1", "3", "30"):
        r[f"r2_lmax{lm}_K3_s0"] = ncases_flags(NCASES[3]) + ["--lambda-max", lm]
    # 4. Placement: the same K, with the CFD runs where the physics struggles (high severity).
    for k, train in PLACE.items():
        for s in range(3):
            r[f"r2_place_K{k}_s{s}"] = ncases_flags(train) + ["--seed", str(s)]
        r[f"r2_place_K{k}_dataonly"] = ncases_flags(train) + ["--lambda-pde", "0"]
    # 5. 60 % edge (WSS ~11 %): Fourier features, SIREN, more throat collocation.
    r["r2_edge_fourier_s0"] = RE150 + ["--fourier-features", "64", "--fourier-sigma", "1.0"]
    r["r2_edge_sine_s0"] = RE150 + ["--activation", "sine"]
    r["r2_edge_focus05_s0"] = RE150 + ["--focus-frac", "0.5"]
    # 6. Does the physics help more with noisy data?
    for pct in (5, 10):
        r[f"r2_noise_{pct}pct_dataonly"] = RE150 + ["--sensor-noise", str(pct / 100), "--lambda-pde", "0"]
    # 7. Two more seeds for round 1's single-run groups.
    for s in (1, 2):
        for n in (250, 100, 50):
            r[f"r2_sensors_{n}_s{s}"] = RE150 + ["--n-sensors", str(n), "--seed", str(s)]
        for pct in (1, 5, 10):
            r[f"r2_noise_{pct}pct_s{s}"] = RE150 + ["--sensor-noise", str(pct / 100), "--seed", str(s)]
        ext = ["--data-dir", "../ns_cfd/output_2d_re150", "--test-severities", "0.35",
               "--data-s-range", "0.25", "0.50", "--seed", str(s)]
        r[f"r2_extrap_physin_s{s}"] = ext
        r[f"r2_extrap_physout_s{s}"] = ext + ["--physics-s-range", "0.25", "0.50"]
        r[f"r2_dataonly_s{s}"] = RE150 + ["--lambda-pde", "0", "--seed", str(s)]
    return r


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
        if any(done(p) for p in base.glob("ncases_*")):
            subprocess.run([py, "scripts/data_efficiency.py", "--runs-dir", str(base)], cwd=ROOT)


if __name__ == "__main__":
    main()

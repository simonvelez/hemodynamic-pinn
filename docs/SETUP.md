# Setting up a new Linux machine (e.g. the A4000 workstation)

This guide sets up everything needed to run the CFD (`ns_cfd/`) and PINN (`PINN/`) code on a fresh
Linux machine, checks it, and starts the comparison campaign. It assumes a terminal, `git`, internet
access and an NVIDIA GPU. Root is not needed.

## 1. What runs where

| Part | Needs | Device |
|---|---|---|
| CFD: `ns_cfd/` (datasets, mesh study, cost study) | FEniCSx (dolfinx 0.11), PETSc 3.25 + MUMPS, mpi4py | CPU, one thread (the scripts set `OMP_NUM_THREADS=1` themselves) |
| PINN: `PINN/` (training, evaluation, campaign) | PyTorch 2.12 with CUDA | GPU (`--device cuda`) |
| Shared: `common/` (run manifests, metrics) | numpy | – |

**One conda environment, `fenicsx-env`, runs everything.** It is defined in `environment.yml` at the repo
root, with the same versions as the laptop environment that produced the datasets.

**Use this environment, not the FEniCS already installed on the machine.**
- The CFD datasets and the mesh study were made with dolfinx 0.11 / PETSc 3.25. The same versions keep the
  CFD numbers reproducible and make the cost-study timings comparable.
- A system FEniCS may be legacy FEniCS (2019, `import fenics`), which this code cannot use. Even a system
  dolfinx usually has a different version.
- Fallback, only if conda is not allowed on the machine: see section 9.

## 2. Check the machine

```bash
nvidia-smi                 # GPU visible? note the "Driver Version" and "CUDA Version"
git --version
df -h ~                    # about 5 GB for the environment, plus a few GB for runs
nproc; lscpu | grep "Model name"

# Which FEniCS is already installed (for information only; it will not be used):
python3 -c "import dolfinx; print('dolfinx', dolfinx.__version__, dolfinx.__file__)" 2>/dev/null \
  || python3 -c "import fenics; print('legacy FEniCS', fenics.__version__)" 2>/dev/null \
  || echo "no FEniCS on the system python"
echo "PYTHONPATH=$PYTHONPATH"   # a system FEniCS often sets this; see section 5
```

The default PyTorch wheel is built for CUDA 13 and needs an NVIDIA driver ≥ 580. With an older driver,
use the CUDA 12.6 wheel (section 5).

## 3. Install conda (Miniforge), if `conda` is not available

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash
exec bash                                   # reload the shell so `conda` is on PATH
conda config --set channel_priority strict
conda --version
```

## 4. Get the code

**Commit and push from the laptop first.** The clone only contains what is on GitHub; uncommitted work
(e.g. new scripts) will be missing.

```bash
cd ~ && git clone https://github.com/simonvelez/hemodynamic-pinn.git
cd hemodynamic-pinn
git log -1 --oneline                        # check it is the commit you expect
ls ns_cfd/output_2d ns_cfd/output_2d_re150  # the CFD datasets are in git (about 31 MB)
```

`thesis_docs/` and `PINN/runs/` are not in git. Copy `thesis_docs/` only if you need it on this machine.

## 5. Create the environment

```bash
conda env create -f environment.yml         # about 5–15 min
conda activate fenicsx-env
```

**Keep a system FEniCS out of the environment.** Add these two lines to `~/.bashrc` (after the conda block),
or run them in every new shell before working:

```bash
unset PYTHONPATH            # a system FEniCS may put its own dolfinx on PYTHONPATH
export PYTHONNOUSERSITE=1   # ignore packages pip-installed into ~/.local
```

**Check that the GPU works from PyTorch:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

If it prints `False` and `nvidia-smi` works, the wheel does not match the driver. Install the CUDA 12.6 build:

```bash
pip install --force-reinstall torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
```

## 6. Verify

All commands are run from the repo root with `fenicsx-env` active.

```bash
# Versions, and that dolfinx comes from the conda env (its path must contain envs/fenicsx-env)
python -c "
import sys, dolfinx, petsc4py, mpi4py, numpy, torch
print('python ', sys.version.split()[0])
print('dolfinx', dolfinx.__version__, dolfinx.__file__)
print('petsc4py', petsc4py.__version__, '| mpi4py', mpi4py.__version__, '| numpy', numpy.__version__)
print('torch  ', torch.__version__, '| CUDA', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
"

# CFD tests (Poiseuille verification + symmetric/bent branch; a few minutes)
cd ns_cfd && python -m pytest -q tests && cd ..

# PINN tests (15 tests, seconds)
cd PINN && python -m pytest -q tests && cd ..

# One short PINN run on the GPU (200 steps, about a minute) into runs/campaign_smoke/
cd PINN && python scripts/campaign.py --smoke --only base_re150_s0 --device cuda && cd ..
python -c "import json; c=json.load(open('PINN/runs/campaign_smoke/base_re150_s0/cost.json')); print(c['gpu'], c['cpu_model'], round(c['total_wall_s']), 's')"
```

The last line must name the A4000. Expected versions: python 3.11, dolfinx 0.11.0, petsc4py 3.25.x,
torch 2.12.1.

## 7. Run the campaign

Use `tmux` so runs survive closing the terminal or SSH session (`tmux new -s campaign`; detach with
`Ctrl-b d`; reattach with `tmux attach -t campaign`). If the machine suspends when idle, wrap long runs in
`systemd-inhibit --what=sleep:idle <command>`.

**Run the CFD cost study and the PINN campaign one after the other, never at the same time**: the cost
study measures wall-clock times, and the comparison is only fair on an otherwise idle machine.

```bash
conda activate fenicsx-env

# 1. CFD cost study (CPU, sequential; 92 configurations × 3 repeats plus a 700×112 reference per case).
#    Long: the transient and fine-mesh runs dominate. Resumable: finished runs are skipped.
cd ns_cfd
python cost_study.py list
python cost_study.py batch 2>&1 | tee cost_study_batch.log
python cost_study.py report          # cost_study/report/{pareto.csv, report.md, plots}
cd ..

# 2. PINN campaign (GPU). Resumable the same way. List first, then run;
#    "--only ncases" runs just the data-efficiency study (20 runs).
cd PINN
python scripts/campaign.py --dry-run
python scripts/campaign.py --device cuda 2>&1 | tee runs/campaign.log
#   writes runs/campaign/comparison_re150.md and runs/campaign/data_efficiency.{md,png}

# 3. CFD vs PINN at equal accuracy, and the break-even N*
python scripts/cfd_vs_pinn.py        # runs/campaign/cfd_vs_pinn.md
```

Time estimate: one full PINN run took about 2.3 h on the laptop CPU. On the A4000 this is a guess,
20–40 min per run; the first run's `cost.json` gives the real number.

## 8. Copy results back to the laptop

`PINN/runs/` is git-ignored, so copy it with `rsync`. Run this on the laptop, from the repo root, replacing
`user@a4000` with the workstation's login:

```bash
rsync -av user@a4000:~/hemodynamic-pinn/PINN/runs/campaign/ PINN/runs/campaign/
rsync -av user@a4000:~/hemodynamic-pinn/ns_cfd/cost_study/  ns_cfd/cost_study/
```

Every run writes a `run_manifest.json` with the CPU, GPU, library versions and git commit. Use it to fill
`hardware.campaign` in `docs/lab_to_fill_in.yaml`.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `dolfinx.__file__` is not inside `envs/fenicsx-env` | The system FEniCS is on `PYTHONPATH` or `~/.local`: `unset PYTHONPATH; export PYTHONNOUSERSITE=1`, then reactivate the env |
| `ImportError` mentioning `libpetsc`, `libmpi` or MPI symbols | System MPI/PETSc libraries are leaking in: `unset LD_LIBRARY_PATH`, reactivate the env. Never mix the system `mpirun` with the env's mpich |
| `torch.cuda.is_available()` is `False` | Driver older than the wheel's CUDA: reinstall the cu126 wheel (section 5). Check `nvidia-smi` works at all |
| `conda env create` is very slow or fails to solve | `conda config --set channel_priority strict`; Miniforge already uses the faster libmamba solver |
| CFD runs much slower than expected | Something else is using the CPU (check `htop`). The scripts set `OMP_NUM_THREADS=1`; do not override it |
| `campaign.py` says a run is `done` but you changed something | Runs with `model.pt` and `metrics.json` are skipped: delete that run's folder to redo it |
| `import cadquery` / `gmsh` missing | Not needed: the 2D pipeline does not use them. Only the old 3D code in `old/` does (`notes/CFD.md` §8) |

**Fallback without conda** (only if the system has dolfinx **0.11**; record this in `docs/DEVIATIONS.md`,
because the versions then differ from the laptop):

```bash
python3 -m venv --system-site-packages ~/venvs/hemo     # sees the system dolfinx/petsc4py/mpi4py
source ~/venvs/hemo/bin/activate
pip install torch==2.12.1 matplotlib pytest             # add --index-url .../whl/cu126 if needed
```

Then run the checks in section 6.

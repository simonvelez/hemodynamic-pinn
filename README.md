# hemodynamic-pinn

CFD vs physics-informed neural networks (PINNs) for 2D coronary stenosis hemodynamics: a methodological
benchmark of accuracy, computational cost and feasibility (thesis, Universidad del Rosario).

## Layout

| Folder / file | What it holds |
|---|---|
| `ns_cfd/` | 2D CFD: warped `crossed` mesh, P2/P1 steady Navier–Stokes (FEniCSx 0.11), dataset generator, and the verification studies (mesh/GCI, stability, Poiseuille, cost study, labmate comparison). Datasets: `output_2d/` (Re ≈ 301), `output_2d_re150/` (Re ≈ 150) |
| `PINN/` | Parametric hybrid PINN (x, y, severity) → (u, v, p), evaluation, and the comparison campaign (`scripts/campaign.py`, `cfd_vs_pinn.py`, `data_efficiency.py`) |
| `common/` | Shared by both: run manifests and cost ledger (`manifest.py`), area-weighted metrics (`metrics.py`) |
| `docs/` | `SETUP.md` (new machine setup and run order), `DEVIATIONS.md` (deviations from the proposal and the guide), `lab_to_fill_in.yaml` (model parameters) |
| `notes/` | Design notes and results: `CFD.md`, `PINN.md` |
| `old/` | Retired code, datasets and runs, kept for reference (`old/README.md`) |
| `thesis_docs/` | Proposal, guide and references (not in git) |
| `environment.yml` | Conda environment `fenicsx-env` for everything |

## Pipeline

1. CFD datasets: `ns_cfd/generate_dataset_2d.py` (already in git; see `notes/CFD.md`).
2. PINN training and evaluation: `PINN/scripts/train.py`, `evaluate.py` (see `PINN/README.md`, `notes/PINN.md`).
3. Comparison on the A4000: `ns_cfd/cost_study.py`, then `PINN/scripts/campaign.py`, then `PINN/scripts/cfd_vs_pinn.py`.

Setup on a new Linux machine (conda env, GPU check, campaign order): [docs/SETUP.md](docs/SETUP.md).

# PINN — parametric hybrid PINN for 2D stenosis flow

One network `(x, y, severity) → (u, v, p)` trained on the steady 2D Navier–Stokes
datasets produced by `ns_cfd/generate_dataset_2d.py`, and usable at severities it
never saw in the CFD data.

**Hybrid** loss: Navier–Stokes residual + boundary conditions + sparse CFD velocity
"sensors". Pressure is not supervised by default; it is inferred from the physics.

```
L = L_pde + λ_wall·L_wall + λ_inlet·L_inlet + λ_outlet·L_outlet + λ_data·L_data
```

| term   | where                         | condition                                  |
|--------|-------------------------------|--------------------------------------------|
| pde    | random interior points (x,y,s) | (u·∇)u + ∇p − Re⁻¹∇²u = 0,  ∇·u = 0         |
| wall   | analytic stenosis wall         | u = v = 0                                  |
| inlet  | x = 0                          | u = 1 − (y/R)², v = 0                      |
| outlet | x = L                          | traction −p + u_x/Re = 0, v_x = 0 (matches the CFD outlet) |
| data   | `n_sensors` CFD nodes per training severity | u, v (+ p with `--data-pressure`) |

The λ's adapt during training using gradient statistics (Wang, Teng & Perdikaris 2021),
as in CardiovascularPINNs. Setting `--lambda-data 0` turns off the data term and gives a
standard physics-only PINN for comparison.

All quantities are nondimensional: `x* = x/H`, `u* = u/u_max`, `p* = p/(ρ u_max²)`, with
Re = ρ u_max H/μ: ≈ 301 for `ns_cfd/output_2d` (the config default) and ≈ 150 for `ns_cfd/output_2d_re150`
(used by the campaign). See `docs/lab_to_fill_in.yaml` for the other Re conventions. Collocation points are drawn
continuously over severity, so the physics also constrains severities that have no
CFD data.

## Layout
```
pinn/config.py     all hyperparameters (dataclass); every field is a CLI flag
pinn/scaling.py    nondimensionalisation, nondimensional geometry
pinn/geometry.py   analytic wall r(x, s) (same as ns_cfd/mesh_2d.py) + point samplers
pinn/data.py       load/validate npz cases (skips asymmetric ones), train/test split, sensor sampling
pinn/models.py     MLP (tanh | SIREN, optional Fourier features), input scaling built in
pinn/physics.py    NS residuals, outlet traction and wall shear stress with autograd
pinn/losses.py     loss terms + AdaptiveWeights
pinn/train.py      Adam + StepLR, then L-BFGS; checkpoints, losses.csv, run_manifest.json, cost.json
pinn/evaluate.py   per-severity metrics (node- and area-weighted L2, Δp, WSS, recirculation), plots, timing
scripts/train.py, scripts/evaluate.py   command-line entry points
scripts/campaign.py       the comparison and robustness campaign (A4000), resumable
scripts/compare_runs.py   per-severity table, mean ± std over groups of runs
scripts/data_efficiency.py  error vs number of CFD runs used for training (campaign group ncases_*)
scripts/cfd_vs_pinn.py    equal-accuracy CFD baseline (ns_cfd cost study) and break-even N*
tests/             pytest (15): Poiseuille residual/traction/WSS, autodiff vs FD, geometry vs CFD wall,
                   data split, area-weighted metrics
runs/              outputs (git-ignored): p2_re150, p2_re301 (current), campaign/ (A4000), campaign_smoke/
```
Shared helpers live in `../common/` (`manifest.py`: run provenance and cost ledger; `metrics.py`:
area-weighted L2). Older runs on retired datasets are in `../old/pinn_runs/`.

## Usage (environment `fenicsx-env`; setup in `../docs/SETUP.md`)
The CFD datasets are in git (`ns_cfd/output_2d`, `ns_cfd/output_2d_re150`); regenerate them only if the
CFD changes (`ns_cfd/generate_dataset_2d.py`, see `notes/CFD.md`).

```bash
cd PINN
python -m pytest -q tests

# One run: Re ≈ 150, 0.35 and 0.55 held out (as in the campaign); add --device cuda on a GPU
python scripts/train.py --out-dir runs/my_run --data-dir ../ns_cfd/output_2d_re150 --test-severities 0.35 0.55
python scripts/evaluate.py --run runs/my_run      # metrics.json + plots in runs/my_run/

# Baselines: physics-only (--lambda-data 0) and data-only (--lambda-pde 0)
python scripts/train.py --out-dir runs/physics_only --lambda-data 0

# Full campaign (list first; --smoke = 200-step pipeline check)
python scripts/campaign.py --dry-run
python scripts/campaign.py --device cuda
python scripts/cfd_vs_pinn.py                     # after ns_cfd/cost_study.py batch + report
```

The loader skips cases whose `meta` reports `steady=False`, `branch: asymmetric`, or
`symmetry_error > --max-symmetry-error`. Design decisions and results: `../notes/PINN.md`.

## References
- `old/PINN_mvp/juanes_pinn.py`: the hybrid data+physics pipeline and parameter-vector inputs this builds on.
- A. Aghaee, O. Khan, *CardiovascularPINNs*: https://github.com/Owais-Khan/CardiovascularPINNs
  (single MLP for (u, v, p), tanh/SIREN, adaptive λ, velocity-only sensors).
- S. Wang, Y. Teng, P. Perdikaris, "Understanding and mitigating gradient flow pathologies in
  physics-informed neural networks", SIAM J. Sci. Comput. 2021.

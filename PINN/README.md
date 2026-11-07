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
| outlet | x = L                          | p = 0 (matches the CFD do-nothing outlet)  |
| data   | `n_sensors` CFD nodes per training severity | u, v (+ p with `--data-pressure`) |

The λ's adapt during training using gradient statistics (Wang, Teng & Perdikaris 2021),
as in CardiovascularPINNs. Setting `--lambda-data 0` turns off the data term and gives a
standard physics-only PINN for comparison.

All quantities are nondimensional: `x* = x/D`, `u* = u/U`, `p* = p/(ρU²)`, with
Re = ρUD/μ ≈ 301 for the default CFD settings. Collocation points are drawn
continuously over severity, so the physics also constrains severities that have no
CFD data.

## Layout
```
pinn/config.py     all hyperparameters (dataclass); every field is a CLI flag
pinn/scaling.py    nondimensionalisation, nondimensional geometry
pinn/geometry.py   analytic wall r(x, s) (same as ns_cfd/mesh_2d.py) + point samplers
pinn/data.py       load/validate npz cases, train/test split, sensor sampling
pinn/models.py     MLP (tanh | SIREN, optional Fourier features), input scaling built in
pinn/physics.py    NS residuals with autograd
pinn/losses.py     loss terms + AdaptiveWeights
pinn/train.py      Adam + StepLR (+ optional L-BFGS), checkpoints, losses.csv
pinn/evaluate.py   rel-L2 per severity, Δp vs severity, CFD/PINN/error plots, timing
scripts/           train.py / evaluate.py command-line entry points
tests/             pytest: Poiseuille residual, geometry vs CFD wall, data split
```

## Usage (environment `fenicsx-env`, CPU)
```bash
# 1. CFD data (symmetric `crossed` mesh by default; see notes/CFD.md for the usable ranges)
cd ns_cfd && python generate_dataset_2d.py --severities 0.20 0.25 0.30 0.35 0.40 0.45 0.50    # Re≈301
python generate_dataset_2d.py --u-max 0.1 --t-final 4.0 --output-dir output_2d_re150 \
    --severities 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60                               # Re≈150

# 2. Tests
cd PINN && python -m pytest -q tests

# 3. Train (0.35 and 0.45 are held out by default) and evaluate
python scripts/train.py --out-dir runs/hybrid
python scripts/evaluate.py --run runs/hybrid      # metrics.json + plots in runs/hybrid/

# Physics-only baseline for comparison
python scripts/train.py --out-dir runs/physics_only --lambda-data 0
```

The loader skips cases whose `meta` reports `steady=False` or
`symmetry_error > --max-symmetry-error`.

## References
- `old/PINN_mvp/juanes_pinn.py`: the hybrid data+physics pipeline and parameter-vector inputs this builds on.
- A. Aghaee, O. Khan, *CardiovascularPINNs*: https://github.com/Owais-Khan/CardiovascularPINNs
  (single MLP for (u, v, p), tanh/SIREN, adaptive λ, velocity-only sensors).
- S. Wang, Y. Teng, P. Perdikaris, "Understanding and mitigating gradient flow pathologies in
  physics-informed neural networks", SIAM J. Sci. Comput. 2021.

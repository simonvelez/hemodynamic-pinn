# CFD pipeline (`ns_cfd/`) — design notes

These notes explain what the CFD code in `ns_cfd/` does and why it was built the way it was.
Its job is to produce reference blood-flow fields for training and testing the PINN (see `PINN.md`).

---

## 1. What is in the folder

| File | Status | Role |
|---|---|---|
| `mesh_2d.py` | **in use** | 2D stenosed-channel mesh built directly in DOLFINx, boundaries tagged (default: symmetric `crossed` triangles) |
| `solver_2d.py` | **in use** | Incompressible Navier–Stokes solver (FEniCSx / DOLFINx 0.11) |
| `generate_dataset_2d.py` | **in use** | Command-line driver: mesh → solve → export `.npz` + field plot + **mesh sheet** for each severity |
| `mesh_plots.py` | **in use** | Mesh inspection: element quality and the one-page mesh troubleshooting sheet |
| `mesh_study.py`, `mesh_study/` | **in use** | Mesh-independence study (§9) and its results |
| `output_2d/` | data | Datasets at Re ≈ 301 (u_max = 0.2 m/s), severities 20–50 %, `crossed` mesh |
| `output_2d_re150/` | data | Datasets at Re ≈ 150 (u_max = 0.1 m/s), severities 20–60 %, `crossed` mesh |

Retired code and data live in `old/` (see `old/README.md`): the first 2D prototype, the 3D skeleton,
`PINN_mvp`, and the first datasets made on the asymmetric `right`-diagonal mesh.

Environment: conda env **`fenicsx-env`** (Python 3.11, DOLFINx 0.11, PETSc 3.25 with MUMPS).

---

## 2. Pipeline

```mermaid
flowchart LR
    A["generate_dataset_2d.py<br/>CLI: severities, mesh, flow, time"] --> B["mesh_2d.build_stenosis_mesh<br/>structured rectangle → warp walls<br/>→ tag inlet / outlet / wall"]
    B --> C["solver_2d.run_simulation<br/>P1/P1 + SUPG/PSPG/LSIC<br/>implicit midpoint, Newton + MUMPS"]
    C -->|"time loop until steady<br/>(or t_final)"| C
    C --> D["diagnostics<br/>mass balance, Δp, symmetry"]
    D --> E["nodal_arrays<br/>interpolate to P1, node-ordered<br/>x, y, u, v, p, flag, triangles"]
    E --> F["pinn2d_severity_XX.npz<br/>+ pinn2d_severity_XX.png"]
    F --> G["PINN/ (training & validation)"]
```

Run:
```bash
conda activate fenicsx-env
cd ns_cfd
python generate_dataset_2d.py --severities 0.20 0.25 0.30 0.35 0.40 0.45 0.50   # Re ≈ 301
python generate_dataset_2d.py --u-max 0.1 --t-final 4.0 --output-dir output_2d_re150 \
       --severities 0.20 0.25 ... 0.60                          # Re ≈ 150
```

---

## 3. Geometry and mesh (`mesh_2d.py`)

**Geometry.** A straight 2D channel with a smooth, symmetric cosine stenosis:

```
r(x) = R · [1 − (s/2)·(1 + cos(π·(x − x_c)/L_s))]   for |x − x_c| ≤ L_s,   r = R elsewhere
```

`s` is the **severity, defined as the diameter reduction at the throat** (s = 0.5 means the throat
diameter is half the healthy diameter).

| Parameter | Default | Why |
|---|---|---|
| `radius` R | 2.5 mm (D = 5 mm) | Same healthy diameter as the original prototype; coronary/carotid order of magnitude |
| `length` L | 50 mm (10 D) | The original 30 mm domain was too short for the flow behind the stenosis to recover |
| `x_center` x_c | 10 mm (2 D) | Keeps the inlet 1.2 D before the stenosis starts; the original had only 5 mm |
| `half_length` L_s | 4 mm | Stenosis spans 6–14 mm, giving smooth walls without kinks |
| `nx × ny` | 250 × 40 | h ≈ 0.2 mm axially and 0.125 mm radially in the healthy section; finer at the throat. Verified by the mesh study (§9) |
| `diagonal` | `crossed` | 4 triangles per cell, mirror-symmetric about y = 0. The former `right` default (2 triangles per cell) seeded artificial asymmetry (§9) |

**Mesh construction.** A structured triangular rectangle `[0, L] × [−R, R]` is generated (each cell split into 4 triangles by its diagonals: 40,000 triangles, 20,291 nodes), and each
node's y-coordinate is scaled by `r(x)/R`. Inlet, outlet and wall facets are tagged **after** the warp,
so the wall tag follows the curved boundary (inlet = 1, outlet = 2, wall = 3).

**Decision: no CadQuery / Gmsh for 2D.**
- In `fenicsx-env`, CadQuery cannot be imported (§8), and Gmsh is not installed.
- A warped structured mesh is enough in 2D, and its symmetric nodes make the symmetry check (§5) exact.
- The same analytic `r(x)` is re-implemented in the PINN. A unit test checks that it matches the CFD wall nodes to 1e-9.

---

## 4. Solver (`solver_2d.py`)

### Physics
Incompressible, Newtonian Navier–Stokes:

| Parameter | Value | Note |
|---|---|---|
| ρ | 1055 kg/m³ | blood density |
| μ | 0.0035 Pa·s | Newtonian blood viscosity (non-Newtonian effects ignored) |
| u_max | 0.2 m/s (also 0.1 m/s) | **peak** inlet velocity |
| Re = ρ·u_max·D/μ | **≈ 301** (≈ 150 for u_max = 0.1) | ⚠ based on the *peak* velocity; the mean-velocity Re is 2/3 of this (≈ 201 / 100) |

### Discretisation
- **Elements:** P1/P1 (equal-order) mixed velocity–pressure space. These are cheap, but they need stabilisation.
- **Stabilisation:** SUPG + PSPG + LSIC (residual-based, Tezduyar-type):
  - τ_M = [(2/Δt)² + (2|u|/h)² + (4ν/h²)²]^(−1/2), and τ_C = h|u|/2.
  - The LSIC term is multiplied by ρ. This fixes the original prototype, where the missing ρ made LSIC about 1000× too weak.
  - The viscous term is dropped from the strong residual, because second derivatives vanish for P1.
- **Time integration:** implicit midpoint for momentum, using u_mid = ½(uⁿ⁺¹ + uⁿ).
  - Continuity is imposed at n+1.
  - Pressure is treated as the n+½ Lagrange multiplier. The prototype averaged pressure in time, which can cause pressure oscillations.
- **Quadrature degree capped at 4.** UFL's automatic estimate for the `sqrt` inside τ_M is huge. Without the cap, Jacobian assembly took 0.8 s per call (about 10× slower overall), with no accuracy benefit for P1 fields.

### Boundary conditions
| Boundary | Condition | Why |
|---|---|---|
| Inlet (x = 0) | Parabolic Poiseuille profile u = u_max(1 − (y/R)²), **ramped** as ½(1 − cos(πt/t_ramp)), t_ramp = 0.1 s | Avoids the impulsive start of the prototype, which needs more Newton iterations and gives unphysical transients |
| Walls | No-slip u = 0 | Rigid vessel |
| Outlet (x = L) | "Do-nothing" (traction-free, so p ≈ 0) + backflow stabilisation −½ρ·min(u·n, 0)·(u·v) | No pressure Dirichlet condition is needed. The backflow term prevents blow-up if a vortex reaches the outlet |

### Nonlinear / linear solver
- Monolithic Newton through DOLFINx `NonlinearProblem` (PETSc SNES `newtonls`, basic line search).
  Tolerances: rtol 1e-8, atol 1e-10, at most 25 iterations, and an error is raised if it does not converge.
- Direct LU with MUMPS. The 2D systems are small (about 31k dofs), so a direct solver is the robust choice.
- `generate_dataset_2d.py` sets `OMP_NUM_THREADS=1`. With threaded BLAS inside MUMPS the CPU was
  oversubscribed: CPU time was about 9× wall time and runs were slower.

### Time loop and steady-state criterion
- Δt = 2 ms. Implicit midpoint is unconditionally stable, and SUPG's τ includes Δt.
- The run stops when the **relative change in velocity per step is below 1e-6** after the ramp has ended.
  Otherwise it stops at `t_final` (2 s by default; 4 s for Re ≈ 150, where the flow takes longer to develop).
- Steadiness is measured on **velocity only**. Pressure is close to zero when the ramp ends, so including
  it gave misleading relative changes.
- The dataset is the **steady** solution. Time snapshots are optional (`--snapshot-every N`), for future time-dependent PINNs.

---

## 5. Diagnostics stored with every case
| Quantity | Meaning |
|---|---|
| `mass_error` | \|Q_in − Q_out\| / Q_in from boundary flux integrals (about 1e-16 in all runs) |
| `dp` | Boundary-averaged pressure inlet − outlet [Pa] |
| `symmetry_error` | max\|u(x, y) − u(x, −y)\| / max\|u\| over mirrored nodes. About 1e-3 for symmetric flow, O(1) once the jet bends to one wall |
| `steady`, `t_end`, `final_rel_change` | Whether and when the steady criterion was met |
| `u_max_field` | Peak velocity in the domain |

---

## 6. Export format (`generate_dataset_2d.py`)
Per severity: `pinn2d_severity_XX.npz`, the field plot `pinn2d_severity_XX.png` and the mesh sheet
`pinn2d_severity_XX_mesh.png` (same sheet as the mesh study, §9). The `.npz` contains:

| Key | Content |
|---|---|
| `x, y` | Node coordinates [m] |
| `u, v, p` | Velocity [m/s] and pressure [Pa] at the nodes |
| `flag` | 0 = interior, 1 = inlet, 2 = outlet, 3 = wall (inlet/outlet take precedence at corners) |
| `triangles` | Mesh connectivity (for plotting / interpolation) |
| `meta` | JSON: all solver parameters, geometry, diagnostics (§5) |
| `t_snap, u_snap, …` | Only when `--snapshot-every` is used |

**Decision: interpolate to plain P1 spaces before export.** In a mixed space the solver's value
ordering does **not** match `domain.geometry.x`. The prototype's plot assumed it did: 496 of 561
values were drawn at the wrong node, so `old/ns_cfd_legacy_2d/stenosis_50_velocity.png` does not show the real flow.
The exporter interpolates u and p into P1 spaces that share one dofmap and takes coordinates from
that dofmap. Export is serial-only, since the meshes are small.

---

## 7. Results

> These tables come from the **first datasets on the `right`-diagonal mesh** (now in
> `old/datasets_right_diagonal/`). The symmetry thresholds below are affected by that mesh (§9).

### Re ≈ 301 (u_max = 0.2 m/s)
| Severity | Steady | Δp [Pa] | max\|u\| [m/s] | Symmetry error | Usable |
|---|---|---|---|---|---|
| 20 % | ✔ | 12.21 | 0.223 | 2.0e-3 | ✔ |
| 25 % | ✔ | 12.81 | 0.233 | 1.9e-3 | ✔ |
| 30 % | ✔ | 13.67 | 0.244 | 2.5e-3 | ✔ |
| 35 % | ✔ | 14.89 | 0.259 | 3.3e-3 | ✔ |
| 40 % | ✔ | 16.66 | 0.276 | 4.0e-3 | ✔ |
| 45 % | ✔ | 19.27 | 0.298 | 4.8e-3 | ✔ |
| 50 % | ✔ | 23.19 | 0.324 | 6.8e-3 | ✔ |
| 55 % | ✘ | 29.21 | 0.358 | 6.0e-2 | ✘ |
| 60–75 % | ✘ | 40–149 | 0.40–0.64 | 0.78–0.92 | ✘ |

### Re ≈ 150 (u_max = 0.1 m/s, t_final = 4 s)
| Severity | Steady | Δp [Pa] | Symmetry error | Usable |
|---|---|---|---|---|
| 20–55 % | ✔ | 6.03 → 11.37 | 1.5e-3 – 3.5e-3 | ✔ |
| 60 % | ✔ (t = 1.64 s) | 14.07 | 5.9e-3 | ✔ |
| 65 % | ✘ | 18.36 | 7.3e-2 | ✘ |
| 70 % | ✘ | 26.30 | 0.72 | ✘ |

Sanity checks on the 50 % case at Re ≈ 301:
- The wall velocity is exactly 0.
- The velocity profile recovers to Poiseuille by the outlet (peak 0.209 m/s against 0.2 at the inlet).
- Backflow (u < 0) is captured behind the throat.

### Symmetry breaking: a physical limit, not a solver bug
Above a critical Re and severity, the 2D jet behind a stenosis bends towards one wall (a Coanda-type
bifurcation). The symmetric solution still exists but is unstable, so the solution is non-unique.
At Re ≈ 301 this starts at about **55 %**; at Re ≈ 150, at about **65 %**.
The PINN loader therefore rejects cases with `steady = False` or `symmetry_error > 0.05`.
Lowering Re is how the usable severity range was extended.

Runtime: a few minutes per case on one core. Higher severities take longer to reach steady state.

---

## 8. Legacy and unfinished code (moved to `old/`)

**`old/ns_cfd_legacy_2d/original_solver.py` (prototype).** It runs under DOLFINx 0.11, but its outputs should not be used:
- the plot misplaces nodal values (§6);
- it runs only 0.03 s after an impulsive start, while one pass of fluid through the domain takes about 0.2 s, and `early_stop_tolerance` is never used;
- the inlet and outlet are too close to the stenosis;
- the LSIC term is missing ρ.

**3D pipeline (`old/ns_cfd_3d/`: `mesh_generator.py`, `solver.py`, `generate_dataset.py`).** These are skeletons:
- `build_cadquery_solid` is undefined, and the O-grid meshing steps are only comments;
- `solver.py` never solves for pressure, applies no inlet BC, has its stabilisation commented out, and uses APIs that are removed or nonexistent in DOLFINx 0.11 (`FunctionSpace`, `gmshio`, `dolfinx.io.get_petsc_options`);
- `generate_dataset.py` passes point indices where DOLFINx expects cell indices to `Function.eval`.

**CadQuery cannot be imported in `fenicsx-env`.** `import cadquery` fails with
`libvtkWrappingPythonCore3.11.so: cannot open shared object file`:
- pip's `cadquery-ocp` requires `vtk==9.6.2`;
- conda's `vtk-base 9.6.2` satisfied that requirement on paper, so pip never installed its own VTK wheel;
- OCP links against the wheel's unversioned library names (`libvtkCommonCore.so`), but conda ships only versioned ones (`…-9.6.so`).

Recommended fix: a separate geometry environment (`mamba create -n cq-env -c conda-forge cadquery python-gmsh`)
that writes `.msh` files for `fenicsx-env` to read. Force-reinstalling pip's VTK over conda's is not recommended.

---

## 9. Mesh-independence study (`mesh_study.py`)

**How the production mesh was chosen.** The 250 × 40 mesh was picked by judgement: about 40 cells
across D and h ≈ 0.1–0.2 mm, with a few minutes of run time per case. It had not been verified.
Three specific concerns:
- **Axial spacing is uniform.** Δx = 0.2 mm everywhere, so the shear layer behind the throat
  (thickness roughly D/√Re ≈ 0.3 mm) spans only 1–2 cells axially.
- **The (former) default `right` diagonal makes the triangulation asymmetric about y = 0,** even though the
  nodes are symmetric. The aspect-ratio map in the mesh plots shows this: elongated cells sit
  upper-left in the converging part and lower-right in the diverging part. This could seed the
  symmetry breaking.
- **The steady discrete solution depends slightly on Δt,** because the stabilisation τ_M contains (2/Δt)².

**Method.**
- Three nested meshes, 125×20, 250×40 and 500×80 (refinement ratio 2).
- The warp is applied node by node, so every coarse node is also a fine node. Fields are compared
  node by node with no interpolation error.
- Discretisation error is estimated with the **Grid Convergence Index** (Roache; Celik et al. 2008):
  observed order p, Richardson extrapolation, and GCI = 1.25·|e|/(rᵖ − 1).
- Quantities of interest: Δp, max |u|, centreline u and p at the throat, separation and reattachment
  points, recirculation length (from where the wall-adjacent row of u changes sign on the upper wall),
  and node-wise velocity/pressure differences against the fine mesh.

| Study part | Cases |
|---|---|
| GCI | 25 % and 50 % at Re ≈ 301; 60 % at Re ≈ 150. Three meshes each |
| Threshold check | 55 % at Re ≈ 301 and 65 % at Re ≈ 150, on 250×40 `right`, 250×40 `crossed` (symmetric, 4 triangles per cell) and 500×80 `right` |
| Δt check | 50 % at Re ≈ 301, 250×40, Δt = 1 ms against 2 ms |

```bash
python mesh_study.py mesh --severity 0.5 --nx 250 --ny 40 --diagonal right   # mesh sheet only, no solve
python mesh_study.py batch --workers 6     # all runs; skips finished runs, so it can be resumed
python mesh_study.py report                # mesh_study/report/report.md, gci_table.csv, plots
```

**Mesh troubleshooting sheets** (`mesh_study/plots/mesh_<tag>.png`, written before every solve):
- the full mesh with boundary-tag colouring and the analytic wall overlaid;
- zooms on the stenosis and on the throat wall;
- maps of minimum angle and aspect ratio;
- grid spacing along x (Δx, and Δy of the wall-adjacent row);
- quality histograms;
- a header with cell and node counts, h_min/h_max, minimum angle, maximum aspect ratio and the number of inverted cells.

A field plot (`fields_<tag>.png`) is also saved for every run.

### Results (full tables in `ns_cfd/mesh_study/report/report.md`)

**Production mesh error estimates** (GCI on the 250×40 mesh; "vs fine" = node-wise relative-L2 difference against 500×80):

| Quantity | 25 % Re≈301 | 50 % Re≈301 | 60 % Re≈150 | Verdict |
|---|---|---|---|---|
| Δp | GCI 1.1 % | GCI 1.6 % | GCI 0.3 % | ✔ meets the < 2 % target |
| Velocity field vs fine | 0.05 % | 0.24 % | 0.32 % | ✔ far below PINN errors (0.5–5 %) |
| Pressure field vs fine | 0.18 % | 0.91 % (max 0.68 Pa at the throat) | 0.67 % | ✔ for PINN use |
| Pressure at throat centre | — | GCI 6.0 % | GCI 4.8 % | ⚠ local throat pressure is less converged |
| Recirculation length | (none) | GCI 10.8 % (9.0 mm against 9.8 extrapolated) | GCI 10.3 % | ✘ underpredicted by about 8 % |
| Reattachment point | — | GCI 3.7 % | GCI 3.4 % | ≈ acceptable |

- **Conclusion for the PINN:** the production mesh is adequate. The CFD reference error in velocity and
  pressure is an order of magnitude below the current PINN errors, so the existing datasets stay valid.
- **For wake studies** (recirculation length, wall shear) use the 500×80 mesh or refine Δx in the wake.
  Part of the recirculation-length change comes from the measurement itself: the wall-adjacent row
  moves closer to the wall on finer meshes.
- **The observed order for Δp at Re ≈ 301 is low (p ≈ 0.3–0.4)** while the differences are tiny
  (about 0.1 Pa). The solution is probably not yet in the asymptotic range, possibly because of the
  O(h) stabilisation terms. Treat the extrapolated values as indicative.
- **Δt also matters.** Halving Δt (2 → 1 ms) changes the steady Δp by 0.7 % and the recirculation
  length by 2.9 %, comparable to the mesh error. This comes from the (2/Δt)² term in τ_M. Use
  Δt = 1 ms for high-accuracy reference data.

**The symmetry-breaking thresholds depend on the triangulation:**

| Case | 250×40 `right` | 250×40 `crossed` (symmetric) | 500×80 `right` |
|---|---|---|---|
| 55 % Re≈301 | unsteady, symmetry error 0.060 | **steady, perfectly symmetric (8e-16)** | unsteady, 0.043 |
| 65 % Re≈150 | unsteady, 0.073 | **steady, perfectly symmetric (8e-16)** | unsteady, 0.19 |

- With a symmetric mesh, the cases previously rejected converge to a symmetric steady state.
- With the asymmetric `right` mesh they don't, even after refinement. On coarser `right` meshes the
  baseline asymmetry is larger (50 %: 0.022 coarse, 0.0068 medium, 0.0039 fine), so the triangle
  orientation clearly feeds an asymmetric perturbation into the flow.
- **What this does not settle** (resolved by the stability test below): whether the symmetric state is *stable* at 55 % / 65 %. A symmetric
  mesh only perturbs the flow at round-off level, which may not grow within 2–4 s even if the flow is
  unstable. The decisive test is the `crossed` mesh plus a small explicit asymmetric perturbation
  (for example in the inlet profile or initial condition). If the perturbation decays, the thresholds
  were mesh artefacts and the PINN severity range can be extended; if it grows, the bifurcation is physical.

### Stability test (`stability_test.py`): the symmetry breaking is physical

The question left open by the mesh study was whether the symmetric flow at 55–70 % is stable or only
looked steady because the symmetric mesh never disturbs it. To test it:
- An antisymmetric, zero-flux disturbance ε·η(1 − η²) (η = y/R, ε = 0.01, peak 0.4 % of u_max) was
  added to the inlet for t < 0.2 s. The inlet was then symmetric again.
- The asymmetry a(t) = max|v| on the centreline / u_max was tracked.
- σ is the late-time growth rate, fitted after the post-disturbance minimum.
- Mesh 250×40 `crossed`, t ≤ 6 s.

| Re | Severity | σ (1/s) | Verdict |
|---|---|---|---|
| ≈ 301 | 50 % (control) | −1.91 | **stable**: the disturbance decays |
| ≈ 301 | 55 % | +0.86 | **unstable** |
| ≈ 301 | 60 % | +3.21 | **unstable** |
| ≈ 301 | 65 %, 70 % | +4.6, +2.1 | **unstable**: saturates into the deflected jet (symmetry error 0.9) |
| ≈ 150 | 60 % (control) | −1.17 | **stable** |
| ≈ 150 | 65 % | +0.85 | **unstable** |
| ≈ 150 | 70 % | +2.71 | **unstable**: saturates (symmetry error 0.72) |

- **The critical severity lies between 50 and 55 % at Re ≈ 301, and between 60 and 65 % at Re ≈ 150.**
  These are the same limits found originally with the `right` mesh. The old mesh did not create the
  instability; it only *triggered* it, the way any real disturbance would.
- The steady symmetric `crossed` solutions at 55 % / 65 % in the mesh study were **unstable
  equilibria**: they persisted only because nothing perturbed them.
- **Consequence for the PINN:** the training ranges stay 20–50 % (Re ≈ 301) and 20–60 % (Re ≈ 150).
  Above them the physical steady flow is the asymmetric (Coanda) state, which has two mirror-image
  solutions, so it cannot be represented by a symmetric-in-severity surrogate.
- Caveats:
  - The cases growing at about 0.9/s stopped at 3 s when the steady criterion triggered (the change
    per step was below 1e-6 while growing slowly), so they did not reach saturation. That doesn't affect the verdict.
  - Growth rates of the saturating cases (65–70 % at Re ≈ 301) are only indicative.
  - The threshold could be refined further with runs at 52.5 % and 62.5 %.

Results: `ns_cfd/stability/report/report.md` and `asymmetry_vs_time.png`. Mesh and field plots per case are in `ns_cfd/stability/plots/`.

## 10. Known limitations / open items
- Rigid walls, Newtonian blood, steady inflow. No pulsatile waveform yet; `--snapshot-every` makes a time-dependent dataset possible.
- At high severities the off-centre jet still reaches the outlet. A longer domain (`--length 0.08`) would be needed there.
- Re is quoted with the peak inlet velocity. State this clearly when comparing with literature, which often uses the mean velocity.

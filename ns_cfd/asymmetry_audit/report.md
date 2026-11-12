# Asymmetry root-cause audit (`ns_cfd/asymmetry_audit.py`)

Steady Newton solves (P1P1, P2P1; imperfection continuation 0.3 → 0 unless the imperfection column says 0). Re_peak = ρ u_max H / μ with ρ = 1055 kg/m³, μ = 0.0035 Pa s, H = 5 mm. Centreline asym = max|v| on y = 0 / u_max. Branch: asymmetric if the node-wise symmetry error ≥ 1e-06. Cases sharing a solve show the same row.

Hardware: AMD Ryzen 7 PRO 5850U with Radeon Graphics (fedora, OMP threads 1).

## Mesh mirror symmetry (every mesh used)

| mesh | DS | L (mm) | lesion centre (mm) | nodes mirrored | max node deviation (m) | triangles mirrored | tagged facets mirrored | symmetric |
|---|---|---|---|---|---|---|---|---|
| 500x40 | 50% | 100 | 10 | 40541/40541 | 4.3e-19 | 80000/80000 | 1080/1080 | yes |
| 125x20 | 50% | 50 | 10 | 5146/5146 | 4.3e-19 | 10000/10000 | 290/290 | yes |
| 250x40 | 50% | 50 | 10 | 20291/20291 | 4.3e-19 | 40000/40000 | 580/580 | yes |
| 350x40 | 50% | 70 | 10 | 28391/28391 | 4.3e-19 | 56000/56000 | 780/780 | yes |
| 350x40 | 50% | 70 | 30 | 28391/28391 | 4.3e-19 | 56000/56000 | 780/780 | yes |
| 500x40 | 55% | 100 | 10 | 40541/40541 | 4.3e-19 | 80000/80000 | 1080/1080 | yes |
| 125x20 | 55% | 50 | 10 | 5146/5146 | 4.3e-19 | 10000/10000 | 290/290 | yes |
| 250x40 | 55% | 50 | 10 | 20291/20291 | 4.3e-19 | 40000/40000 | 580/580 | yes |
| 350x40 | 55% | 70 | 10 | 28391/28391 | 4.3e-19 | 56000/56000 | 780/780 | yes |
| 350x40 | 55% | 70 | 30 | 28391/28391 | 4.3e-19 | 56000/56000 | 780/780 | yes |
| 250x40 | 70% | 50 | 10 | 20291/20291 | 4.3e-19 | 40000/40000 | 580/580 | yes |

## Unseeded vs seeded steady solve (55 %, Re 301)

| case | DS | Re_peak | L (mm) | lesion centre (mm) | mesh | element | imperfection | centreline asym | symmetry err | branch | Δp (Pa) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s55_re301_noseed | 55% | 301 | 50 | 10 | 250×40 | P2P1 | 0 | 0.0000 | 5.5e-15 | symmetric | 28.54 | 30 |
| s55_re301_seeded | 55% | 301 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.0709 | 3.8e-01 | asymmetric | 28.76 | 74 |

## Downstream length

| case | DS | Re_peak | L (mm) | lesion centre (mm) | mesh | element | imperfection | centreline asym | symmetry err | branch | Δp (Pa) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s50_L50 | 50% | 301 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.0000 | 2.6e-10 | symmetric | 22.87 | 72 |
| s50_L100 | 50% | 301 | 100 | 10 | 500×40 | P2P1 | 0.3 | 0.0000 | 2.6e-10 | symmetric | 33.39 | 148 |
| s55_L50 | 55% | 301 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.0709 | 3.8e-01 | asymmetric | 28.76 | 74 |
| s55_L100 | 55% | 301 | 100 | 10 | 500×40 | P2P1 | 0.3 | 0.0709 | 3.8e-01 | asymmetric | 38.89 | 150 |

## Upstream length

| case | DS | Re_peak | L (mm) | lesion centre (mm) | mesh | element | imperfection | centreline asym | symmetry err | branch | Δp (Pa) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s50_xc10 | 50% | 301 | 70 | 10 | 350×40 | P2P1 | 0.3 | 0.0000 | 2.6e-10 | symmetric | 26.80 | 107 |
| s50_xc30 | 50% | 301 | 70 | 30 | 350×40 | P2P1 | 0.3 | 0.0000 | 2.6e-10 | symmetric | 27.35 | 106 |
| s55_xc10 | 55% | 301 | 70 | 10 | 350×40 | P2P1 | 0.3 | 0.0709 | 3.8e-01 | asymmetric | 32.38 | 107 |
| s55_xc30 | 55% | 301 | 70 | 30 | 350×40 | P2P1 | 0.3 | 0.0709 | 3.8e-01 | asymmetric | 33.24 | 109 |

## Mesh and element

| case | DS | Re_peak | L (mm) | lesion centre (mm) | mesh | element | imperfection | centreline asym | symmetry err | branch | Δp (Pa) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s50_125x20_P1P1 | 50% | 301 | 50 | 10 | 125×20 | P1P1 | 0.3 | 0.0000 | 2.8e-11 | symmetric | 24.40 | 8 |
| s50_125x20_P2P1 | 50% | 301 | 50 | 10 | 125×20 | P2P1 | 0.3 | 0.0000 | 1.9e-10 | symmetric | 22.74 | 16 |
| s50_250x40_P1P1 | 50% | 301 | 50 | 10 | 250×40 | P1P1 | 0.3 | 0.0000 | 3.1e-12 | symmetric | 23.55 | 25 |
| s50_250x40_P2P1 | 50% | 301 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.0000 | 2.6e-10 | symmetric | 22.87 | 72 |
| s55_125x20_P1P1 | 55% | 301 | 50 | 10 | 125×20 | P1P1 | 0.3 | 0.1241 | 6.2e-01 | asymmetric | 31.93 | 6 |
| s55_125x20_P2P1 | 55% | 301 | 50 | 10 | 125×20 | P2P1 | 0.3 | 0.0655 | 3.5e-01 | asymmetric | 28.51 | 17 |
| s55_250x40_P1P1 | 55% | 301 | 50 | 10 | 250×40 | P1P1 | 0.3 | 0.1007 | 5.2e-01 | asymmetric | 30.23 | 25 |
| s55_250x40_P2P1 | 55% | 301 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.0709 | 3.8e-01 | asymmetric | 28.76 | 74 |

## Re scan at 70 %

| case | DS | Re_peak | L (mm) | lesion centre (mm) | mesh | element | imperfection | centreline asym | symmetry err | branch | Δp (Pa) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s70_re50 | 70% | 50 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.0000 | 4.7e-10 | symmetric | 5.67 | 46 |
| s70_re75 | 70% | 75 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.0000 | 9.2e-12 | symmetric | 9.48 | 62 |
| s70_re100 | 70% | 100 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.0958 | 2.4e-01 | asymmetric | 13.97 | 75 |
| s70_re150 | 70% | 150 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.2606 | 7.1e-01 | asymmetric | 25.47 | 70 |

## Re scan at 50 %

| case | DS | Re_peak | L (mm) | lesion centre (mm) | mesh | element | imperfection | centreline asym | symmetry err | branch | Δp (Pa) | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s50_re300 | 50% | 300 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.0000 | 2.3e-10 | symmetric | 22.73 | 71 |
| s50_re400 | 50% | 400 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.0000 | 1.9e-11 | symmetric | 34.26 | 84 |
| s50_re500 | 50% | 500 | 50 | 10 | 250×40 | P2P1 | 0.3 | 0.0809 | 5.4e-01 | asymmetric | 49.49 | 79 |

# Mesh-independence study — results


## 25% Re≈301 (severity 0.25, u_max 0.2 m/s)

| mesh | cells | steady | symmetry err | wall time |
|---|---|---|---|---|
| 125×20 | 5000 | True | 7.9e-03 | 0.8 min |
| 250×40 | 20000 | True | 1.9e-03 | 3.9 min |
| 500×80 | 80000 | True | 1.0e-03 | 16.4 min |

| QoI | coarse | medium (production) | fine | order p | extrapolated | GCI fine | GCI medium | medium vs extrap. | convergence |
|---|---|---|---|---|---|---|---|---|---|
| Δp (Pa) | 12.78 | 12.81 | 12.85 | 0.3086 | 12.99 | 1.40% | 1.14% | 1.37% | monotonic |
| max|u| (m/s) | 0.2326 | 0.2327 | 0.2328 | 0.3551 | 0.2333 | 0.25% | 0.20% | 0.26% | monotonic |
| u throat centre (m/s) | 0.2301 | 0.2302 | 0.2303 | 2.231 | 0.2304 | 0.03% | 0.01% | 0.10% | monotonic |
| p throat centre (Pa) | 3.688 | 3.7 | 3.68 | – | – | 0.68% | – | – | oscillatory |
| recirculation length (mm) | – | – | – | – | – | – | – | – | n/a |
| reattachment x (mm) | – | – | – | – | – | – | – | – | n/a |

| field difference vs fine mesh (shared nodes) | rel-L2 velocity | rel-L2 pressure | max abs Δu | max abs Δp |
|---|---|---|---|---|
| coarse 125×20 (2646 nodes) | 0.16% | 0.39% | 0.0033 m/s | 0.221 Pa |
| medium 250×40 (10291 nodes) | 0.05% | 0.18% | 0.0015 m/s | 0.088 Pa |

## 50% Re≈301 (severity 0.50, u_max 0.2 m/s)

| mesh | cells | steady | symmetry err | wall time |
|---|---|---|---|---|
| 125×20 | 5000 | True | 2.2e-02 | 1.4 min |
| 250×40 | 20000 | True | 6.8e-03 | 4.8 min |
| 500×80 | 80000 | True | 3.9e-03 | 26.9 min |

| QoI | coarse | medium (production) | fine | order p | extrapolated | GCI fine | GCI medium | medium vs extrap. | convergence |
|---|---|---|---|---|---|---|---|---|---|
| Δp (Pa) | 23.3 | 23.19 | 23.12 | 0.4273 | 22.9 | 1.18% | 1.58% | 1.28% | monotonic |
| max|u| (m/s) | 0.3242 | 0.3243 | 0.3243 | 2.746 | 0.3243 | 0.00% | 0.01% | 0.00% | monotonic |
| u throat centre (m/s) | 0.3151 | 0.3154 | 0.3159 | 0.519 | 0.317 | 0.44% | 0.31% | 0.50% | monotonic |
| p throat centre (Pa) | -10.44 | -10.62 | -10.87 | 0.4397 | -11.56 | 7.99% | 6.03% | 8.15% | monotonic |
| recirculation length (mm) | 7.669 | 9.023 | 9.518 | 1.451 | 9.804 | 3.75% | 10.82% | 7.97% | monotonic |
| reattachment x (mm) | 19.48 | 20.61 | 21 | 1.525 | 21.21 | 1.24% | 3.65% | 2.84% | monotonic |

| field difference vs fine mesh (shared nodes) | rel-L2 velocity | rel-L2 pressure | max abs Δu | max abs Δp |
|---|---|---|---|---|
| coarse 125×20 (2646 nodes) | 0.86% | 2.17% | 0.0079 m/s | 1.733 Pa |
| medium 250×40 (10291 nodes) | 0.24% | 0.91% | 0.0017 m/s | 0.682 Pa |

## 60% Re≈150 (severity 0.60, u_max 0.1 m/s)

| mesh | cells | steady | symmetry err | wall time |
|---|---|---|---|---|
| 125×20 | 5000 | True | 3.1e-02 | 3.5 min |
| 250×40 | 20000 | True | 5.9e-03 | 9.5 min |
| 500×80 | 80000 | True | 3.9e-03 | 35.5 min |

| QoI | coarse | medium (production) | fine | order p | extrapolated | GCI fine | GCI medium | medium vs extrap. | convergence |
|---|---|---|---|---|---|---|---|---|---|
| Δp (Pa) | 14.14 | 14.07 | 14.05 | 1.543 | 14.04 | 0.10% | 0.30% | 0.24% | monotonic |
| max|u| (m/s) | 0.21 | 0.2096 | 0.2099 | – | – | 0.15% | – | – | oscillatory |
| u throat centre (m/s) | 0.2025 | 0.2025 | 0.203 | 4.629 | 0.203 | 0.01% | 0.00% | 0.28% | monotonic |
| p throat centre (Pa) | -3.496 | -3.583 | -3.724 | 0.7011 | -3.95 | 7.57% | 4.84% | 9.28% | monotonic |
| recirculation length (mm) | 7.345 | 8.527 | 8.968 | 1.42 | 9.232 | 3.67% | 10.34% | 7.64% | monotonic |
| reattachment x (mm) | 19.06 | 20.07 | 20.42 | 1.52 | 20.61 | 1.16% | 3.37% | 2.63% | monotonic |

| field difference vs fine mesh (shared nodes) | rel-L2 velocity | rel-L2 pressure | max abs Δu | max abs Δp |
|---|---|---|---|---|
| coarse 125×20 (2646 nodes) | 1.17% | 1.86% | 0.0040 m/s | 0.824 Pa |
| medium 250×40 (10291 nodes) | 0.32% | 0.67% | 0.0014 m/s | 0.316 Pa |

## 25% Re≈301 — P2/P1, grad, crossed, steady solve (severity 0.25, u_max 0.2 m/s)

| mesh | cells | steady | symmetry err | wall time |
|---|---|---|---|---|
| 125×20 | 10000 | True | 1.1e-15 | 0.1 min |
| 250×40 | 40000 | True | 1.8e-15 | 0.4 min |
| 500×80 | 160000 | True | 3.3e-15 | 1.9 min |

| QoI | coarse | medium (production) | fine | order p | extrapolated | GCI fine | GCI medium | medium vs extrap. | convergence |
|---|---|---|---|---|---|---|---|---|---|
| Δp (Pa) | 12.92 | 12.93 | 12.93 | 1.972 | 12.93 | 0.01% | 0.03% | 0.02% | monotonic |
| max|u| (m/s) | 0.2328 | 0.2329 | 0.2329 | 0.9206 | 0.233 | 0.03% | 0.05% | 0.04% | monotonic |
| u throat centre (m/s) | 0.2305 | 0.2306 | 0.2306 | 1.542 | 0.2306 | 0.01% | 0.02% | 0.02% | monotonic |
| p throat centre (Pa) | 3.684 | 3.674 | 3.671 | 1.783 | 3.67 | 0.04% | 0.14% | 0.11% | monotonic |
| recirculation length (mm) | – | – | – | – | – | – | – | – | n/a |
| reattachment x (mm) | – | – | – | – | – | – | – | – | n/a |

| field difference vs fine mesh (shared nodes) | rel-L2 velocity | rel-L2 pressure | max abs Δu | max abs Δp |
|---|---|---|---|---|
| coarse 125×20 (5146 nodes) | 0.05% | 0.19% | 0.0010 m/s | 0.058 Pa |
| medium 250×40 (20291 nodes) | 0.01% | 0.04% | 0.0003 m/s | 0.015 Pa |

## 50% Re≈301 — P2/P1, grad, crossed, steady solve (severity 0.50, u_max 0.2 m/s)

| mesh | cells | steady | symmetry err | wall time |
|---|---|---|---|---|
| 125×20 | 10000 | True | 1.6e-15 | 0.1 min |
| 250×40 | 40000 | True | 4.8e-15 | 0.4 min |
| 500×80 | 160000 | True | 7.2e-15 | 2.0 min |

| QoI | coarse | medium (production) | fine | order p | extrapolated | GCI fine | GCI medium | medium vs extrap. | convergence |
|---|---|---|---|---|---|---|---|---|---|
| Δp (Pa) | 22.75 | 22.87 | 22.91 | 1.937 | 22.92 | 0.06% | 0.25% | 0.20% | monotonic |
| max|u| (m/s) | 0.3229 | 0.3235 | 0.3237 | 1.372 | 0.3238 | 0.06% | 0.15% | 0.12% | monotonic |
| u throat centre (m/s) | 0.3158 | 0.3162 | 0.3164 | 1.584 | 0.3164 | 0.03% | 0.09% | 0.07% | monotonic |
| p throat centre (Pa) | -11.23 | -11.26 | -11.27 | 1.427 | -11.27 | 0.06% | 0.16% | 0.13% | monotonic |
| recirculation length (mm) | 7.374 | 8.704 | 9.254 | 1.274 | 9.642 | 5.24% | 13.47% | 9.73% | monotonic |
| reattachment x (mm) | 19.28 | 20.3 | 20.72 | 1.262 | 21.02 | 1.83% | 4.48% | 3.46% | monotonic |

| field difference vs fine mesh (shared nodes) | rel-L2 velocity | rel-L2 pressure | max abs Δu | max abs Δp |
|---|---|---|---|---|
| coarse 125×20 (5146 nodes) | 0.28% | 1.13% | 0.0058 m/s | 0.413 Pa |
| medium 250×40 (20291 nodes) | 0.06% | 0.24% | 0.0016 m/s | 0.105 Pa |

## 60% Re≈150 — P2/P1, grad, crossed, steady solve (severity 0.60, u_max 0.1 m/s)

| mesh | cells | steady | symmetry err | wall time |
|---|---|---|---|---|
| 125×20 | 10000 | True | 1.5e-15 | 0.1 min |
| 250×40 | 40000 | True | 3.3e-15 | 0.4 min |
| 500×80 | 160000 | True | 6.5e-15 | 2.1 min |

| QoI | coarse | medium (production) | fine | order p | extrapolated | GCI fine | GCI medium | medium vs extrap. | convergence |
|---|---|---|---|---|---|---|---|---|---|
| Δp (Pa) | 13.83 | 13.94 | 13.96 | 1.925 | 13.97 | 0.09% | 0.35% | 0.28% | monotonic |
| max|u| (m/s) | 0.2087 | 0.2098 | 0.21 | 2.516 | 0.21 | 0.02% | 0.14% | 0.11% | monotonic |
| u throat centre (m/s) | 0.203 | 0.2035 | 0.2037 | 1.566 | 0.2038 | 0.06% | 0.18% | 0.14% | monotonic |
| p throat centre (Pa) | -3.978 | -3.977 | -3.978 | – | – | 0.03% | – | – | oscillatory |
| recirculation length (mm) | 7.463 | 8.429 | 8.833 | 1.26 | 9.122 | 4.09% | 10.27% | 7.59% | monotonic |
| reattachment x (mm) | 19.24 | 19.96 | 20.26 | 1.271 | 20.47 | 1.30% | 3.19% | 2.49% | monotonic |

| field difference vs fine mesh (shared nodes) | rel-L2 velocity | rel-L2 pressure | max abs Δu | max abs Δp |
|---|---|---|---|---|
| coarse 125×20 (5146 nodes) | 0.40% | 1.19% | 0.0051 m/s | 0.222 Pa |
| medium 250×40 (20291 nodes) | 0.08% | 0.25% | 0.0013 m/s | 0.064 Pa |

## Symmetry-breaking threshold: mesh and triangle-orientation dependence

| case | mesh | diagonal | steady | symmetry err | Δp (Pa) | verdict (usable?) |
|---|---|---|---|---|---|---|
| 55% Re≈301 | 250×40 | right | False | 5.96e-02 | 29.214 | no |
| 55% Re≈301 | 250×40 | crossed | True | 7.77e-16 | 29.061 | yes |
| 55% Re≈301 | 500×80 | right | False | 4.26e-02 | 29.048 | no |
| 65% Re≈150 | 250×40 | right | False | 7.29e-02 | 18.363 | no |
| 65% Re≈150 | 250×40 | crossed | True | 7.54e-16 | 18.281 | yes |
| 65% Re≈150 | 500×80 | right | False | 1.88e-01 | 18.330 | no |

## Time-step sensitivity of the steady solution (50 %, Re≈301, 250×40)

| QoI | dt = 2 ms | dt = 1 ms | rel. diff |
|---|---|---|---|
| Δp (Pa) | 23.19 | 23.03 | 0.71% |
| max|u| (m/s) | 0.3243 | 0.3238 | 0.13% |
| u throat centre (m/s) | 0.3154 | 0.3157 | 0.09% |
| p throat centre (Pa) | -10.62 | -10.89 | 2.48% |
| recirculation length (mm) | 9.023 | 8.771 | 2.87% |
| reattachment x (mm) | 20.61 | 20.35 | 1.26% |

Field difference (all nodes): velocity 0.351%, pressure 1.782%.

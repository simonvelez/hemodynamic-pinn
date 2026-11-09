"""All hyperparameters in one dataclass; every field is also a CLI flag."""

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # --- data ---------------------------------------------------------------
    data_dir: str = str(REPO / "ns_cfd" / "output_2d")
    out_dir: str = "runs/default"
    test_severities: list = field(default_factory=lambda: [0.35, 0.45])  # held out entirely
    max_symmetry_error: float = 0.05   # reject bifurcated (asymmetric) CFD cases
    n_sensors: int = 500               # CFD velocity "sensors" per training case
    data_pressure: bool = False        # also supervise p (default: pressure inferred from physics)
    sensor_noise: float = 0.0          # Gaussian noise std on sensor targets, fraction of U (robustness study)
    sensor_seed: int = -1              # seed for sensor locations + noise (-1: use `seed`)
    data_s_range: list = field(default_factory=list)     # [lo, hi]: only cases inside get sensors; others -> test
    physics_s_range: list = field(default_factory=list)  # [lo, hi]: severities where physics is enforced
                                                         # (default: all loaded cases incl. held-out)

    # --- model --------------------------------------------------------------
    activation: str = "tanh"           # tanh | sine
    width: int = 128
    depth: int = 5                     # hidden layers
    omega0: float = 30.0               # first-layer frequency for sine (SIREN)
    fourier_features: int = 0          # >0: random Fourier embedding of (x, y)
    fourier_sigma: float = 1.0

    # --- collocation sampling (re-drawn every step) -------------------------
    n_interior: int = 4096
    focus_frac: float = 0.3            # share of interior points around throat / recirculation
    n_wall: int = 512
    n_inlet: int = 256
    n_outlet: int = 256
    outlet_bc: str = "traction"        # traction (-p + u_x/Re = 0, v_x = 0; matches CFD) | pressure (p = 0, legacy)

    # --- loss weighting -----------------------------------------------------
    adaptive: bool = True              # gradient-based adaptive lambdas (Wang et al. / CardiovascularPINNs)
    adapt_every: int = 10
    adapt_ema: float = 0.9
    lambda_max: float = 10.0           # validated: 1e3 starved the physics term (pressure off by 20-44 %)
    lambda_wall: float = 1.0           # initial (adaptive) or fixed values
    lambda_inlet: float = 1.0
    lambda_outlet: float = 1.0
    lambda_data: float = 1.0           # 0 -> standard (physics-only) PINN
    lambda_pde: float = 1.0            # 0 -> data-only network baseline (no NS residual; BC weights fixed)

    # --- optimisation -------------------------------------------------------
    steps: int = 20000
    lr: float = 1e-3
    lr_step: int = 4000
    lr_gamma: float = 0.5
    lbfgs_steps: int = 2000            # L-BFGS polish after Adam (0 = off)

    # --- bookkeeping --------------------------------------------------------
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    seed: int = 0
    device: str = "cpu"                # cpu | cuda | cuda:0 ...
    dtype: str = "float32"             # float32 | float64 (whole run; float64 helps L-BFGS)
    threads: int = 0                   # >0: torch.set_num_threads

    def save(self, path):
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def from_cli(cls, argv=None, description=None):
        ap = argparse.ArgumentParser(description=description)
        for f in dataclasses.fields(cls):
            default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
            flag = "--" + f.name.replace("_", "-")
            if isinstance(default, bool):
                ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
            elif isinstance(default, list):
                ap.add_argument(flag, type=float, nargs="*", default=default)
            else:
                ap.add_argument(flag, type=type(default), default=default)
        return cls(**vars(ap.parse_args(argv)))

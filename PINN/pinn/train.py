"""Training loop for the parametric hybrid PINN."""

import csv
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import load_cases, sensor_tensors, split_cases
from .geometry import sample_interior, sample_line, sample_wall
from .losses import TERMS, AdaptiveWeights, compute_losses
from .models import PINN
from .scaling import Geometry, Scales


# ---------------------------------------------------------------- checkpoints
def build_model(cfg, geo, s_range):
    kwargs = dict(lower=[0.0, -geo.R, s_range[0]], upper=[geo.length, geo.R, s_range[1]],
                  width=cfg.width, depth=cfg.depth, activation=cfg.activation, omega0=cfg.omega0,
                  fourier_features=cfg.fourier_features, fourier_sigma=cfg.fourier_sigma, seed=cfg.seed)
    return PINN(**kwargs), kwargs


def save_checkpoint(path, model, model_kwargs, cfg, scales, geo, s_range, step, weights):
    torch.save(dict(state_dict=model.state_dict(), model_kwargs=model_kwargs,
                    config=dataclasses.asdict(cfg), scales=dataclasses.asdict(scales),
                    geometry=dataclasses.asdict(geo), s_range=list(s_range), step=step,
                    lambdas=dict(weights.lam)), path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = PINN(**ck["model_kwargs"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, Scales(**ck["scales"]), Geometry(**ck["geometry"]), ck


# -------------------------------------------------------------------- metrics
@torch.no_grad()
def predict(model, inputs, chunk=20000):
    return torch.cat([model(inputs[i:i + chunk]) for i in range(0, len(inputs), chunk)]).cpu().numpy()


def rel_l2(model, case, device="cpu"):
    """Relative L2 errors on all CFD nodes: velocity vector, u, v, p."""
    pred = predict(model, case.inputs().to(device))
    t = case.uvp
    err = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
    return dict(vel=err(pred[:, :2], t[:, :2]), u=err(pred[:, 0], t[:, 0]),
                v=err(pred[:, 1], t[:, 1]), p=err(pred[:, 2], t[:, 2]))


# ----------------------------------------------------------------------- train
def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    dev = torch.device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")

    cases, scales, geo = load_cases(cfg.data_dir, cfg.max_symmetry_error)
    train_c, test_c = split_cases(cases, cfg.test_severities)
    if not train_c:
        raise RuntimeError("no training cases left after the test split")
    s_all = [c.severity for c in cases]
    s_range = (min(s_all), max(s_all))   # physics covers held-out severities too
    print(f"[data] Re={scales.Re:.1f}  train s={[c.severity for c in train_c]}  "
          f"test s={[c.severity for c in test_c]}  s_range={s_range}", flush=True)

    model, model_kwargs = build_model(cfg, geo, s_range)
    model.to(dev)

    use_data = cfg.lambda_data > 0 and cfg.n_sensors > 0
    sx = sy = None
    if use_data:
        sx, sy = sensor_tensors(train_c, cfg.n_sensors, rng, cfg.data_pressure)
        sx, sy = sx.to(dev), sy.to(dev)
        print(f"[data] {len(sx)} sensor points ({'u,v,p' if cfg.data_pressure else 'u,v'})", flush=True)
    else:
        print("[data] lambda_data=0 -> standard physics-only PINN", flush=True)

    weights = AdaptiveWeights(
        dict(wall=cfg.lambda_wall, inlet=cfg.lambda_inlet, outlet=cfg.lambda_outlet,
             data=cfg.lambda_data if use_data else 0.0),
        adaptive=cfg.adaptive, ema=cfg.adapt_ema, lam_max=cfg.lambda_max)

    def sample_batch():
        b = dict(
            interior=sample_interior(cfg.n_interior, geo, s_range, gen, cfg.focus_frac),
            wall=sample_wall(cfg.n_wall, geo, s_range, gen),
            inlet=sample_line(cfg.n_inlet, 0.0, geo, s_range, gen),
            outlet=sample_line(cfg.n_outlet, geo.length, geo, s_range, gen),
        )
        b = {k: v.to(dev) for k, v in b.items()}
        b["sensor_x"], b["sensor_y"] = sx, sy
        return b

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_step, gamma=cfg.lr_gamma)

    cols = ["step", "total", "pde", *TERMS, *[f"lam_{k}" for k in TERMS], "lr", "time_s",
            "train_vel_l2", "test_vel_l2"]
    log_f = open(out / "losses.csv", "w", newline="")
    log = csv.DictWriter(log_f, fieldnames=cols)
    log.writeheader()

    def evaluate_now():
        model.eval()
        tr = np.mean([rel_l2(model, c, dev)["vel"] for c in train_c])
        te = np.mean([rel_l2(model, c, dev)["vel"] for c in test_c]) if test_c else float("nan")
        model.train()
        return tr, te

    def log_row(step, L, total, t0, ev=None):
        row = dict(step=step, total=total.item(), lr=opt.param_groups[0]["lr"], time_s=time.time() - t0)
        row.update({k: float(v) for k, v in L.items()})
        row.update({f"lam_{k}": weights.lam.get(k, 0.0) for k in TERMS})
        if ev is not None:
            row["train_vel_l2"], row["test_vel_l2"] = ev
        log.writerow(row)
        log_f.flush()

    t0 = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        L = compute_losses(model, sample_batch(), scales.Re, geo)
        if step == 1 or step % cfg.adapt_every == 0:
            weights.update(model, L)
        total = weights.total(L)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        sched.step()

        ev = evaluate_now() if (cfg.eval_every and step % cfg.eval_every == 0) else None
        if step % cfg.log_every == 0 or step == 1 or ev is not None:
            log_row(step, L, total, t0, ev)
        if ev is not None or step == 1:
            lam = " ".join(f"{k}={weights.lam.get(k, 0):.3g}" for k in TERMS)
            evs = f" | vel-L2 train={ev[0]:.3%} test={ev[1]:.3%}" if ev else ""
            print(f"[adam] {step:6d} total={total.item():.3e} pde={L['pde'].item():.2e} "
                  f"wall={L['wall'].item():.2e} data={L.get('data', torch.tensor(0.)).item():.2e} "
                  f"| lam {lam}{evs} | {time.time() - t0:.0f}s", flush=True)
        if cfg.save_every and step % cfg.save_every == 0:
            save_checkpoint(out / "model.pt", model, model_kwargs, cfg, scales, geo, s_range, step, weights)

    if cfg.lbfgs_steps > 0:
        # Full-batch polish on one fixed sample, weights frozen at their final values.
        fixed = sample_batch()
        lb = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=cfg.lbfgs_steps,
                               history_size=50, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad(set_to_none=True)
            loss = weights.total(compute_losses(model, fixed, scales.Re, geo))
            loss.backward()
            return loss

        lb.step(closure)
        L = compute_losses(model, fixed, scales.Re, geo)
        ev = evaluate_now()
        log_row(cfg.steps + cfg.lbfgs_steps, L, weights.total(L), t0, ev)
        print(f"[lbfgs] total={weights.total(L).item():.3e} | vel-L2 train={ev[0]:.3%} test={ev[1]:.3%}", flush=True)

    save_checkpoint(out / "model.pt", model, model_kwargs, cfg, scales, geo, s_range,
                    cfg.steps + cfg.lbfgs_steps, weights)
    log_f.close()
    print(f"[done] {time.time() - t0:.0f}s -> {out / 'model.pt'}", flush=True)
    return model

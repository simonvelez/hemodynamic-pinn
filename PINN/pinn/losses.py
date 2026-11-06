"""Hybrid loss terms and gradient-based adaptive weighting."""

import torch

from .geometry import inlet_profile
from .physics import ns_residuals

TERMS = ("wall", "inlet", "outlet", "data")


def compute_losses(model, batch, Re, geo):
    """batch: dict with 'interior', 'wall', 'inlet', 'outlet' inputs and optional
    'sensor_x', 'sensor_y'. Returns dict of scalar MSE losses (incl. 'pde')."""
    r_x, r_y, r_c = ns_residuals(model, batch["interior"], Re)
    L = {"pde": (r_x ** 2).mean() + (r_y ** 2).mean() + (r_c ** 2).mean()}

    L["wall"] = (model(batch["wall"])[:, :2] ** 2).mean()

    xin = batch["inlet"]
    out_in = model(xin)
    L["inlet"] = ((out_in[:, 0:1] - inlet_profile(xin[:, 1:2], geo)) ** 2).mean() + (out_in[:, 1:2] ** 2).mean()

    L["outlet"] = (model(batch["outlet"])[:, 2:3] ** 2).mean()   # p* = 0 (do-nothing outlet)

    if batch.get("sensor_x") is not None:
        target = batch["sensor_y"]
        L["data"] = ((model(batch["sensor_x"])[:, : target.shape[1]] - target) ** 2).mean()
    return L


class AdaptiveWeights:
    """lambda_k <- ema*lambda_k + (1-ema) * max|grad_theta L_pde| / mean|grad_theta L_k|

    (learning-rate annealing of Wang, Teng & Perdikaris 2021, as used in
    CardiovascularPINNs/train.py). With adaptive=False the initial values are kept.
    A term whose initial weight is 0 stays switched off (e.g. lambda_data=0 ->
    standard physics-only PINN).
    """

    def __init__(self, init, adaptive=True, ema=0.9, lam_max=1e3):
        self.lam = {k: float(v) for k, v in init.items()}
        self.adaptive, self.ema, self.lam_max = adaptive, ema, lam_max

    def update(self, model, losses):
        if not self.adaptive:
            return
        params = [p for p in model.parameters() if p.requires_grad]

        def flat_grad(loss):
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            return torch.cat([x.reshape(-1) for x in g if x is not None]).abs()

        g_pde_max = flat_grad(losses["pde"]).max()
        for k in TERMS:
            if k not in losses or self.lam.get(k, 0.0) == 0.0:
                continue
            ratio = (g_pde_max / (flat_grad(losses[k]).mean() + 1e-12)).item()
            new = self.ema * self.lam[k] + (1.0 - self.ema) * ratio
            self.lam[k] = float(min(max(new, 1e-3), self.lam_max))

    def total(self, losses):
        return losses["pde"] + sum(self.lam.get(k, 0.0) * losses[k] for k in TERMS if k in losses)

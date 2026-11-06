import torch

from pinn.physics import ns_residuals

RE = 301.4


class Poiseuille(torch.nn.Module):
    """Exact steady NS solution in a straight channel |y| <= 1/2 (nondimensional):
    u = 1 - 4y^2, v = 0, p = -8x/Re."""

    def forward(self, xys):
        x, y = xys[:, 0:1], xys[:, 1:2]
        return torch.cat([1 - 4 * y ** 2, torch.zeros_like(x), -8 * x / RE], dim=1)


def _points(n=500):
    g = torch.Generator().manual_seed(0)
    return torch.cat([10 * torch.rand(n, 1, generator=g), torch.rand(n, 1, generator=g) - 0.5,
                      torch.rand(n, 1, generator=g)], dim=1).double()


def test_poiseuille_residual_vanishes():
    r = ns_residuals(Poiseuille(), _points(), RE)
    for res in r:
        assert res.abs().max() < 1e-12


def test_wrong_pressure_gradient_is_detected():
    class Bad(Poiseuille):
        def forward(self, xys):
            out = super().forward(xys)
            return torch.cat([out[:, :2], 8 * xys[:, 0:1] / RE], dim=1)   # sign flipped

    r_x, _, _ = ns_residuals(Bad(), _points(), RE)
    assert torch.allclose(r_x, torch.full_like(r_x, 16 / RE))


def test_residuals_differentiable_wrt_parameters():
    from pinn.models import PINN
    m = PINN([0, -0.5, 0.2], [10, 0.5, 0.7], width=16, depth=2)
    r = ns_residuals(m, _points(64).float(), RE)
    sum((x ** 2).mean() for x in r).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

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


def test_poiseuille_satisfies_traction_outlet():
    """Fully developed outflow with p = 0 at x = 10 meets -p + u_x/Re = 0, v_x = 0 exactly."""
    from pinn.physics import outlet_traction

    class OutletGauge(Poiseuille):
        def forward(self, xys):
            out = super().forward(xys)
            return torch.cat([out[:, :2], 8 * (10 - xys[:, 0:1]) / RE], dim=1)

    pts = _points()
    pts[:, 0] = 10.0
    t_x, t_y = outlet_traction(OutletGauge(), pts, RE)
    assert t_x.abs().max() < 1e-12 and t_y.abs().max() < 1e-12


def test_traction_outlet_detects_pressure_offset():
    from pinn.physics import outlet_traction
    pts = _points()
    pts[:, 0] = 10.0
    t_x, _ = outlet_traction(Poiseuille(), pts, RE)   # p = -80/Re at x = 10, not 0
    assert torch.allclose(t_x, torch.full_like(t_x, 80 / RE))


# ---- L0 (guide §7.5): WSS operator and autodiff derivatives on exact / smooth fields ----------------

def _straight_channel_case(n=200):
    """A stand-in for data.Case: straight channel (s = 0), wall nodes on y = ±1/2."""
    from types import SimpleNamespace

    import numpy as np
    from pinn.scaling import Geometry
    x = np.linspace(0.0, 10.0, n)
    xy = np.vstack([np.column_stack([x, np.full(n, 0.5)]), np.column_stack([x, np.full(n, -0.5)])])
    case = SimpleNamespace(xy=xy, wall_idx=np.arange(2 * n), severity=0.0)
    return case, Geometry(length=10.0, R=0.5, x_center=2.0, half_length=0.8)


def test_wss_operator_exact_on_poiseuille():
    """u = 1 - 4y^2 gives tau/(rho U^2) = (1/Re) |du/dy|_wall = 4/Re on both walls, positive for forward flow."""
    from pinn.evaluate import pinn_wss
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        case, geo = _straight_channel_case()
        tau = pinn_wss(Poiseuille(), case, geo, RE)

        class Reversed(Poiseuille):
            def forward(self, xys):
                return -super().forward(xys)

        tau_rev = pinn_wss(Reversed(), case, geo, RE)
    finally:
        torch.set_default_dtype(prev)
    assert abs(tau - 4 / RE).max() < 1e-12
    assert abs(tau_rev + 4 / RE).max() < 1e-12     # reversed flow -> negative WSS (recirculation sign)


def test_residual_autodiff_matches_finite_differences():
    """ns_residuals on a random smooth network equals the same operator built from central differences."""
    from pinn.models import PINN
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        m = PINN([0, -0.5, 0.2], [10, 0.5, 0.7], width=16, depth=2, seed=1)
        pts = _points(32)
        r_x, r_y, r_c = ns_residuals(m, pts, RE)
        h = 1e-4

        def f(dx=0.0, dy=0.0):
            q = pts.clone()
            q[:, 0] += dx
            q[:, 1] += dy
            with torch.no_grad():
                return m(q)

        f0 = f()
        dfx, dfy = (f(h) - f(-h)) / (2 * h), (f(0, h) - f(0, -h)) / (2 * h)
        lap = (f(h) + f(-h) + f(0, h) + f(0, -h) - 4 * f0) / h ** 2
        u, v = f0[:, 0:1], f0[:, 1:2]
        fd_x = u * dfx[:, 0:1] + v * dfy[:, 0:1] + dfx[:, 2:3] - lap[:, 0:1] / RE
        fd_y = u * dfx[:, 1:2] + v * dfy[:, 1:2] + dfy[:, 2:3] - lap[:, 1:2] / RE
        fd_c = dfx[:, 0:1] + dfy[:, 1:2]
    finally:
        torch.set_default_dtype(prev)
    for ad, fd in ((r_x, fd_x), (r_y, fd_y), (r_c, fd_c)):
        assert (ad.detach() - fd).abs().max() < 1e-6 * max(1.0, fd.abs().max().item())

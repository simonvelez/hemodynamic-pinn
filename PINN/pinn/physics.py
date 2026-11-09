"""Steady incompressible Navier-Stokes residuals (nondimensional) via autograd."""

import torch


def _grad(f, x):
    return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]


def ns_residuals(model, xys, Re):
    """Return (r_mom_x, r_mom_y, r_cont), each (n, 1), at points xys = (x*, y*, s).

        r_mom = (u.grad)u + grad p - (1/Re) lap u
        r_cont = div u
    Derivatives are spatial only (columns 0, 1); s is a parameter.
    """
    xys = xys.detach().requires_grad_(True)
    out = model(xys)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    gu, gv, gp = _grad(u, xys), _grad(v, xys), _grad(p, xys)
    u_x, u_y, v_x, v_y = gu[:, 0:1], gu[:, 1:2], gv[:, 0:1], gv[:, 1:2]
    u_xx = _grad(u_x, xys)[:, 0:1]
    u_yy = _grad(u_y, xys)[:, 1:2]
    v_xx = _grad(v_x, xys)[:, 0:1]
    v_yy = _grad(v_y, xys)[:, 1:2]

    r_x = u * u_x + v * u_y + gp[:, 0:1] - (u_xx + u_yy) / Re
    r_y = u * v_x + v * v_y + gp[:, 1:2] - (v_xx + v_yy) / Re
    r_c = u_x + v_y
    return r_x, r_y, r_c


def outlet_traction(model, xys, Re):
    """Outlet do-nothing residuals for the gradient-form viscous term, normal n = +x:

        -p + (1/Re) du/dx = 0,      dv/dx = 0

    This is the natural boundary condition of the CFD weak form
    (mu grad u : grad v, ns_cfd/solver_2d.py, viscous_form="grad"). Fully
    developed Poiseuille outflow satisfies it with p = 0 at the outlet.
    """
    xys = xys.detach().requires_grad_(True)
    out = model(xys)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    u_x = _grad(u, xys)[:, 0:1]
    v_x = _grad(v, xys)[:, 0:1]
    return -p + u_x / Re, v_x

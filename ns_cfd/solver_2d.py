"""Incompressible Navier-Stokes in a 2D stenosis (DOLFINx 0.11).

P1/P1 mixed elements, implicit-midpoint time stepping, SUPG/PSPG/LSIC
stabilisation, monolithic Newton (PETSc SNES + MUMPS LU).

Boundary conditions
  inlet  : parabolic Poiseuille profile, smoothly ramped from 0 over t_ramp
  wall   : no-slip
  outlet : do-nothing (traction-free, p ~ 0) + backflow stabilisation
The run stops when the flow is steady (relative change per step < steady_tol)
or when t_final is reached.

Stability testing: with perturb_eps > 0 an odd (antisymmetric, zero-flux)
component eps*eta*(1-eta^2), eta = y/R, is added to the inlet profile for
t < perturb_until. The asymmetry a(t) = max|v| on the centreline / u_max
(zero for a symmetric flow) is recorded in info["asym_history"].
"""

import numpy as np
import ufl
import basix.ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem.petsc import NonlinearProblem

from mesh_2d import INLET_TAG, OUTLET_TAG, WALL_TAG

DEFAULTS = dict(
    rho=1055.0,       # blood density [kg/m^3]
    mu=0.0035,        # dynamic viscosity [Pa s]
    u_max=0.2,        # peak inlet velocity [m/s]
    dt=0.002,         # time step [s]
    t_ramp=0.1,       # inlet ramp-up time [s]
    t_final=2.0,      # hard stop [s]
    steady_tol=1e-6,  # relative change per step that counts as steady
    snapshot_every=0, # >0: also store (u, v, p) every N steps for unsteady PINNs
    perturb_eps=0.0,      # amplitude of the antisymmetric inlet perturbation (0 = off)
    perturb_until=0.2,    # perturbation active for t < perturb_until [s]
    min_time=0.0,         # never declare steady before this time [s]
    verbose=True,
)


def run_simulation(domain, facet_tags, **kwargs):
    prm = {**DEFAULTS, **kwargs}
    comm = domain.comm
    gdim = domain.geometry.dim
    fdim = domain.topology.dim - 1
    log = (lambda *a: print(*a, flush=True)) if (prm["verbose"] and comm.rank == 0) else (lambda *a: None)

    # ---- Function spaces -------------------------------------------------
    cell = domain.topology.cell_name()
    v_el = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))
    p_el = basix.ufl.element("Lagrange", cell, 1)
    W = fem.functionspace(domain, basix.ufl.mixed_element([v_el, p_el]))
    V, u_map = W.sub(0).collapse()   # u_map[i]: mixed-vector index of collapsed velocity entry i
    u_map = np.asarray(u_map).reshape(-1)  # DOLFINx 0.11 returns [array] (shape 1xN)
    u_dofs = u_map[u_map < W.dofmap.index_map.size_local * W.dofmap.index_map_bs]
    # centreline v entries (blocked layout: entry 2*node+1 is v) for the asymmetry monitor
    vx = V.tabulate_dof_coordinates()
    n_owned = V.dofmap.index_map.size_local
    centre_nodes = np.flatnonzero(np.isclose(vx[:n_owned, 1], 0.0, atol=1e-12))
    centre_v = u_map[2 * centre_nodes + 1]

    w = fem.Function(W)      # unknown (u^{n+1}, p^{n+1/2})
    w_n = fem.Function(W)    # previous step
    u, p = ufl.split(w)
    u_n, _ = ufl.split(w_n)
    v, q = ufl.TestFunctions(W)

    # ---- Boundary conditions ---------------------------------------------
    # Healthy half-height = largest |y| in the mesh (the inlet is unconstricted).
    R_in = comm.allreduce(float(np.max(np.abs(domain.geometry.x[:, 1]))), op=MPI.MAX)

    u_in = fem.Function(V)
    def inlet_profile(scale, eps=0.0):
        def f(x):
            eta = x[1] / R_in
            return np.vstack((scale * prm["u_max"] * ((1.0 - eta ** 2) + eps * eta * (1.0 - eta ** 2)),
                              np.zeros_like(x[0])))
        return f

    def asymmetry():
        local = np.max(np.abs(w.x.array[centre_v])) if len(centre_v) else 0.0
        return comm.allreduce(float(local), op=MPI.MAX) / prm["u_max"]

    u_wall = fem.Function(V)  # zero
    inlet_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facet_tags.find(INLET_TAG))
    wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facet_tags.find(WALL_TAG))
    bcs = [
        fem.dirichletbc(u_in, inlet_dofs, W.sub(0)),
        fem.dirichletbc(u_wall, wall_dofs, W.sub(0)),
    ]

    # ---- Variational form --------------------------------------------------
    dt = fem.Constant(domain, prm["dt"])
    rho = fem.Constant(domain, prm["rho"])
    mu = fem.Constant(domain, prm["mu"])
    n = ufl.FacetNormal(domain)
    # Cap quadrature: UFL's estimate for the sqrt() in tau_M is huge and makes
    # Jacobian assembly ~100x slower with no accuracy gain for P1 fields.
    qd = {"quadrature_degree": 4}
    dx = ufl.Measure("dx", domain=domain, metadata=qd)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, metadata=qd)

    u_mid = 0.5 * (u + u_n)
    eps = lambda a: ufl.sym(ufl.grad(a))
    conv = ufl.grad(u_mid) * u_mid                    # (u.grad)u

    F = (
        rho * ufl.inner((u - u_n) / dt, v) * dx
        + rho * ufl.inner(conv, v) * dx
        + 2.0 * mu * ufl.inner(eps(u_mid), eps(v)) * dx
        - p * ufl.div(v) * dx
        + q * ufl.div(u) * dx
    )

    # Backflow stabilisation on the outlet (only active where u.n < 0).
    F += -0.5 * rho * ufl.min_value(ufl.dot(u_mid, n), 0.0) * ufl.inner(u_mid, v) * ds(OUTLET_TAG)

    # SUPG / PSPG / LSIC
    h = ufl.CellDiameter(domain)
    u_norm = ufl.sqrt(ufl.inner(u_mid, u_mid) + 1e-12)
    nu = mu / rho
    tau_M = 1.0 / ufl.sqrt((2.0 / dt) ** 2 + (2.0 * u_norm / h) ** 2 + (4.0 * nu / h**2) ** 2)
    tau_C = 0.5 * h * u_norm
    R_M = rho * ((u - u_n) / dt + conv) + ufl.grad(p)   # viscous term ~0 for P1
    F += tau_M * ufl.inner(ufl.grad(v) * u_mid, R_M) * dx          # SUPG
    F += (tau_M / rho) * ufl.inner(ufl.grad(q), R_M) * dx          # PSPG
    F += rho * tau_C * ufl.div(u) * ufl.div(v) * dx                # LSIC

    problem = NonlinearProblem(
        F, w, bcs=bcs, petsc_options_prefix="ns2d_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "basic",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_max_it": 25,
            "snes_error_if_not_converged": True,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    # ---- Time loop ---------------------------------------------------------
    t, step, steady = 0.0, 0, False
    rel = np.inf
    snapshots = []
    asym_history = []
    t_steady_check = max(prm["t_ramp"], prm["perturb_until"] if prm["perturb_eps"] else 0.0, prm["min_time"])
    n_max = int(round(prm["t_final"] / prm["dt"]))
    log(f"[solver] dofs={W.dofmap.index_map.size_global * W.dofmap.index_map_bs}, dt={prm['dt']}, up to {n_max} steps")

    while step < n_max:
        step += 1
        t += prm["dt"]
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t / prm["t_ramp"], 1.0)))
        eps_now = prm["perturb_eps"] if t < prm["perturb_until"] else 0.0
        u_in.interpolate(inlet_profile(ramp, eps_now))

        problem.solve()
        its = problem.solver.getIterationNumber()

        # steadiness measured on velocity only (pressure norm is ~0 near end of ramp)
        du = w.x.array[u_dofs] - w_n.x.array[u_dofs]
        diff = comm.allreduce(np.sum(du ** 2), op=MPI.SUM)
        norm = comm.allreduce(np.sum(w.x.array[u_dofs] ** 2), op=MPI.SUM)
        rel = np.sqrt(diff / max(norm, 1e-30))
        w_n.x.array[:] = w.x.array

        if prm["snapshot_every"] and step % prm["snapshot_every"] == 0:
            snapshots.append((t, w.x.array.copy()))
        if step % 10 == 0 or step == 1:
            asym_history.append((t, asymmetry()))
        if step % 25 == 0 or step == 1:
            log(f"[solver] step {step:5d}  t={t:.4f}s  newton={its}  rel_change={rel:.2e}  asym={asym_history[-1][1]:.2e}")
        if t > t_steady_check and rel < prm["steady_tol"]:
            steady = True
            break

    log(f"[solver] finished at t={t:.4f}s after {step} steps, steady={steady} (rel_change={rel:.2e})")

    # ---- Diagnostics -------------------------------------------------------
    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()
    def integ(expr):
        return comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)
    one = fem.Constant(domain, 1.0)
    info = dict(
        t_end=t, steps=step, steady=steady, final_rel_change=rel,
        flux_in=-integ(ufl.dot(u_h, n) * ds(INLET_TAG)),
        flux_out=integ(ufl.dot(u_h, n) * ds(OUTLET_TAG)),
        dp=integ(p_h * ds(INLET_TAG)) / integ(one * ds(INLET_TAG))
           - integ(p_h * ds(OUTLET_TAG)) / integ(one * ds(OUTLET_TAG)),
        params={k: v for k, v in prm.items() if k != "verbose"},
        asym_history=asym_history,
    )
    info["mass_error"] = abs(info["flux_in"] - info["flux_out"]) / abs(info["flux_in"])

    # Snapshots -> (u, p) Functions on demand
    def snapshot_fields():
        for ts, arr in snapshots:
            tmp = fem.Function(W)
            tmp.x.array[:] = arr
            yield ts, tmp.sub(0).collapse(), tmp.sub(1).collapse()

    return u_h, p_h, info, snapshot_fields

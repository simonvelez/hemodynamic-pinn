"""Incompressible Navier-Stokes in a 2D stenosis (DOLFINx 0.11).

Elements (``element``):
  "P2P1" (default) Taylor-Hood, inf-sup stable: no pressure stabilisation.
  "P1P1"           equal order + SUPG/PSPG/LSIC (the original, cheaper scheme).
Implicit-midpoint time stepping, monolithic Newton (PETSc SNES + MUMPS LU).

Viscous term (``viscous_form``):
  "grad" (default) mu grad(u):grad(v). Its natural outlet condition is
         -p n + mu du/dn = 0, which fully developed (Poiseuille) outflow
         satisfies exactly, so the outlet pressure is uniform (= 0).
  "sym"  2 mu eps(u):eps(v) (legacy). Natural condition sigma.n = 0 is NOT
         satisfied by Poiseuille flow (wall shear at the outlet), which distorts
         v and p near the outlet. Kept only to reproduce the old datasets.

Boundary conditions
  inlet  : parabolic Poiseuille profile, smoothly ramped from 0 over t_ramp
  wall   : no-slip
  outlet : do-nothing (natural condition of the chosen viscous form) + backflow stabilisation
Two solution modes (``steady_solve``):
  False (default) time marching (implicit midpoint) until the flow is steady
        (relative change per step < steady_tol) or t_final is reached. Needed
        for stability tests and unsteady snapshots.
  True  direct steady Newton solve with continuation in the inlet amplitude
        (0 -> u_max, adaptive increments). Same discrete steady equations
        without the time derivative; minutes -> seconds. Note: it converges to
        the symmetric steady solution even where that solution is unstable
        (above the bifurcation), so it must only be used inside the stable range.

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
    element="P2P1",       # "P2P1" (Taylor-Hood) | "P1P1" (+ SUPG/PSPG/LSIC)
    viscous_form="grad",  # "grad" (Poiseuille-compatible outlet) | "sym" (legacy)
    stabilise=None,       # None: SUPG/PSPG/LSIC for P1P1, none for P2P1; True/False forces it
    steady_solve=False,   # True: direct steady Newton with continuation (no time marching)
    continuation_step=0.25,  # initial inlet-amplitude increment for steady_solve (halved on failure)
    asym_eps=0.0,         # steady_solve only: >0 follows the asymmetric (bent-jet) branch. The amplitude
                          # continuation is done with a small transverse body force (symmetry-breaking
                          # imperfection, f_y = eps rho u_max^2 / R on the post-throat region asym_x),
                          # which is then continued to 0. Above the symmetry-breaking threshold this lands
                          # on the bent steady state; below it, back on the symmetric one.
    asym_x=(0.01, 0.02),  # x-range [m] of the imperfection force (default: throat to ~2.5 D downstream)
    verbose=True,
)


def run_simulation(domain, facet_tags, **kwargs):
    prm = {**DEFAULTS, **kwargs}
    comm = domain.comm
    gdim = domain.geometry.dim
    fdim = domain.topology.dim - 1
    log = (lambda *a: print(*a, flush=True)) if (prm["verbose"] and comm.rank == 0) else (lambda *a: None)

    # ---- Function spaces -------------------------------------------------
    if prm["element"] not in ("P1P1", "P2P1"):
        raise ValueError(f"unknown element {prm['element']!r}")
    if prm["viscous_form"] not in ("grad", "sym"):
        raise ValueError(f"unknown viscous_form {prm['viscous_form']!r}")
    k_u = 2 if prm["element"] == "P2P1" else 1
    stabilise = (k_u == 1) if prm["stabilise"] is None else bool(prm["stabilise"])
    cell = domain.topology.cell_name()
    v_el = basix.ufl.element("Lagrange", cell, k_u, shape=(gdim,))
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
    # Jacobian assembly ~100x slower with no accuracy gain. Degree 4 is exact
    # for the P1 Galerkin terms; P2 convection (u.grad u).v has degree 5 -> 6.
    qd = {"quadrature_degree": 4 if k_u == 1 else 6}
    dx = ufl.Measure("dx", domain=domain, metadata=qd)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, metadata=qd)

    steady_solve = bool(prm["steady_solve"])
    # u_mid: velocity at which momentum is evaluated (midpoint in time, or u itself when steady)
    u_mid = u if steady_solve else 0.5 * (u + u_n)
    eps = lambda a: ufl.sym(ufl.grad(a))
    conv = ufl.grad(u_mid) * u_mid                    # (u.grad)u
    accel = 0 if steady_solve else (u - u_n) / dt     # time derivative (absent when steady)

    F = (
        rho * ufl.inner(conv, v) * dx
        + (2.0 * mu * ufl.inner(eps(u_mid), eps(v)) if prm["viscous_form"] == "sym"
           else mu * ufl.inner(ufl.grad(u_mid), ufl.grad(v))) * dx
        - p * ufl.div(v) * dx
        + q * ufl.div(u) * dx
    )
    if not steady_solve:
        F += rho * ufl.inner(accel, v) * dx
    # Symmetry-breaking imperfection (steady asymmetric-branch continuation only; 0 otherwise).
    f_asym = fem.Constant(domain, 0.0)
    if prm["asym_eps"]:
        xs = ufl.SpatialCoordinate(domain)
        x0, x1 = prm["asym_x"]
        window = ufl.conditional(ufl.And(ufl.ge(xs[0], x0), ufl.le(xs[0], x1)), 1.0, 0.0)
        F += -f_asym * window * v[1] * dx

    # Backflow stabilisation on the outlet (only active where u.n < 0).
    F += -0.5 * rho * ufl.min_value(ufl.dot(u_mid, n), 0.0) * ufl.inner(u_mid, v) * ds(OUTLET_TAG)

    # SUPG / PSPG / LSIC (residual-based). For P1 the viscous term of the strong
    # residual vanishes elementwise, which makes PSPG slightly inconsistent where
    # grad p balances viscosity (pressure error at inlet/outlet); for P2 it is kept.
    if stabilise:
        h = ufl.CellDiameter(domain)
        u_norm = ufl.sqrt(ufl.inner(u_mid, u_mid) + 1e-12)
        nu = mu / rho
        tau_t = 0.0 if steady_solve else (2.0 / dt) ** 2
        tau_M = 1.0 / ufl.sqrt(tau_t + (2.0 * u_norm / h) ** 2 + (4.0 * nu / h**2) ** 2)
        tau_C = 0.5 * h * u_norm
        R_M = rho * (accel + conv) + ufl.grad(p)
        if k_u > 1:
            R_M = R_M - mu * ufl.div(ufl.grad(u_mid))
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
    log(f"[solver] {prm['element']} viscous={prm['viscous_form']} stabilised={stabilise} "
        f"dofs={W.dofmap.index_map.size_global * W.dofmap.index_map_bs}, "
        + ("steady Newton + continuation" if steady_solve else f"dt={prm['dt']}, up to {n_max} steps"))

    newton_total = 0
    if steady_solve:
        # Continuation in the inlet amplitude: 0 -> 1, increment halved after a failed
        # Newton solve (state restored) and grown 1.5x after a success.
        def continuation(set_param, start, end, inc, name):
            """Newton continuation of one parameter start -> end; step halved on failure, x1.5 on success."""
            nonlocal step, newton_total
            level, sign = start, np.sign(end - start)
            while sign * (end - level) > 1e-12:
                target = end if sign * (end - level) <= inc else level + sign * inc
                backup = w.x.array.copy()
                set_param(target)
                try:
                    problem.solve()
                except Exception as exc:   # PETSc raises when SNES diverges
                    w.x.array[:] = backup
                    inc *= 0.5
                    log(f"[solver] {name} {target:.4f} failed ({type(exc).__name__}); increment -> {inc:.5f}")
                    if inc < 1e-3 * max(abs(end - start), 1e-12):
                        raise RuntimeError(f"steady {name} continuation stalled") from exc
                    continue
                its = problem.solver.getIterationNumber()
                newton_total += its
                step += 1
                level = target
                log(f"[solver] {name} {level:.4f}  newton={its}  asym={asymmetry():.2e}")
                inc *= 1.5

        # Inlet amplitude 0 -> 1 (with the imperfection, if any), then imperfection -> 0.
        e0 = float(prm["asym_eps"])
        f_scale = prm["rho"] * prm["u_max"] ** 2 / R_in

        def set_amplitude(a):
            u_in.interpolate(inlet_profile(a))
            f_asym.value = e0 * f_scale * a ** 2

        def set_imperfection(e):
            f_asym.value = e * f_scale

        continuation(set_amplitude, 0.0, 1.0, float(prm["continuation_step"]), "amplitude")
        if e0:
            continuation(set_imperfection, e0, 0.0, 0.1 * e0, "imperfection")
        steady, rel = True, 0.0
        asym_history.append((0.0, asymmetry()))
        n_max = 0   # skip the time loop

    while step < n_max:
        step += 1
        t += prm["dt"]
        ramp = 0.5 * (1.0 - np.cos(np.pi * min(t / prm["t_ramp"], 1.0)))
        eps_now = prm["perturb_eps"] if t < prm["perturb_until"] else 0.0
        u_in.interpolate(inlet_profile(ramp, eps_now))

        problem.solve()
        its = problem.solver.getIterationNumber()
        newton_total += its

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

    if steady_solve:
        log(f"[solver] steady solve done: {step} continuation levels, {newton_total} Newton iterations")
    else:
        log(f"[solver] finished at t={t:.4f}s after {step} steps, steady={steady} (rel_change={rel:.2e})")

    # ---- Diagnostics -------------------------------------------------------
    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()
    def integ(expr):
        return comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)
    one = fem.Constant(domain, 1.0)
    info = dict(
        t_end=t, steps=step, steady=steady, final_rel_change=rel, newton_iterations=newton_total,
        solve_mode="steady" if steady_solve else "transient",
        n_dofs=int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs),
        flux_in=-integ(ufl.dot(u_h, n) * ds(INLET_TAG)),
        flux_out=integ(ufl.dot(u_h, n) * ds(OUTLET_TAG)),
        dp=integ(p_h * ds(INLET_TAG)) / integ(one * ds(INLET_TAG))
           - integ(p_h * ds(OUTLET_TAG)) / integ(one * ds(OUTLET_TAG)),
        params={**{k: v for k, v in prm.items() if k != "verbose"}, "stabilise": stabilise},
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

import dolfinx
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_topological
from dolfinx.io import gmshio
from mpi4py import MPI
import numpy as np
import ufl

# Import tags established in the mesh generator
INLET_TAG, OUTLET_TAG, WALL_TAG = 1, 2, 3


def run_cfd_simulation(msh_file, dt=0.001, t_final=0.1, output_dir="./output"):
    # 1. Import domain and boundary markers from Gmsh mesh file
    domain, cell_markers, facet_markers = gmshio.read_from_msh(
        msh_file, MPI.COMM_WORLD, rank=0, gdim=3
    )

    # 2. Define Function Spaces for velocity (V) and pressure (Q)
    # Using P1/P1 elements combined with stabilization terms
    V = dolfinx.fem.functionspace(
        domain, ("Lagrange", 1, (domain.geometry.dim,))
    )
    Q = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

    # 3. Boundary Conditions
    # No-slip condition on arterial walls (u = 0)
    wall_facets = facet_markers.find(WALL_TAG)
    wall_dofs = locate_dofs_topological(V, domain.topology.dim - 1, wall_facets)
    u_zero = np.array([0.0, 0.0, 0.0], dtype=dolfinx.default_scalar_type)
    bc_wall = dirichletbc(u_zero, wall_dofs, V)

    # Parabolic Poiseuille profile prescribed at the inlet
    inlet_facets = facet_markers.find(INLET_TAG)
    # Define parabolic velocity expression on inlet_facets...

    bcs = [bc_wall]

    # 4. Weak Formulation with Implicit Midpoint & Stabilization
    u = Function(V)
    p = Function(Q)
    v = ufl.TestFunction(V)
    q = ufl.TestFunction(Q)

    u_n = Function(V)  # Previous time step velocity
    u_mid = 0.5 * (u + u_n)  # Fully implicit midpoint evaluation

    # Fluid physical properties
    rho = 1055.0  # Blood density (kg/m^3)
    mu = 0.0035  # Dynamic viscosity (Pa*s)

    # Standard Navier-Stokes variational form
    F_momentum = (
        rho * ufl.dot((u - u_n) / dt, v) * ufl.dx
        + rho * ufl.dot(ufl.grad(u_mid) * u_mid, v) * ufl.dx
        + ufl.inner(2 * mu * ufl.sym(ufl.grad(u_mid)), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
    )
    F_continuity = ufl.div(u_mid) * q * ufl.dx

    # Add SUPG, PSPG, and LSIC stabilization terms for P1/P1 stability
    # F_stabilization = SUPG_term + PSPG_term + LSIC_term
    F = F_momentum + F_continuity  # + F_stabilization

    # 5. Monolithic Newton Solver setup with PETSc
    problem = dolfinx.fem.petsc.NonlinearProblem(F, u, bcs)
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)

    # Configure Schur complement preconditioning for the saddle-point system
    opts = dolfinx.io.get_petsc_options()
    # Apply PETSc solver configurations (e.g., KSP/PC options)

    # 6. Time Integration Loop
    t = 0.0
    while t < t_final:
        t += dt
        u_n.x.array[:] = u.x.array  # Update previous solution step
        num_iters, converged = solver.solve(u)

        print(f"Time t={t:.4f}s | Newton Iterations: {num_iters}")

    return domain, u, p
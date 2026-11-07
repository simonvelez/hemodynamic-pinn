import numpy as np
import ufl
import basix.ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.fem import petsc

class Simulation:
    def __init__(self, name, simulation, solver, T, dt, output_dir, early_stop_tolerance, **kwargs):
        self.name = name
        self.T = T
        self.dt = dt
        self.mu = kwargs.get("mu", 0.0035)   # Dynamic viscosity (Pa.s)[cite: 1]
        self.rho = kwargs.get("rho", 1055.0) # Density (kg/m^3)[cite: 1]
        self.p_grade = kwargs.get("p_grade", 1) # P1/P1 equal-order elements[cite: 1]
        
        # 1. Mesh setup
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([0.03, 0.005])], 
            [50, 10], cell_type=mesh.CellType.triangle
        )
        self.t = 0.0
        
        # Execution flow
        self._setup_function_spaces()
        self._setup_boundary_conditions()
        self._setup_variational_form()
        self._setup_solver()  

    def _setup_function_spaces(self):
        v_cg = basix.ufl.element("Lagrange", self.domain.topology.cell_name(), self.p_grade, shape=(self.domain.geometry.dim,))
        p_cg = basix.ufl.element("Lagrange", self.domain.topology.cell_name(), self.p_grade)
        
        mixed_el = basix.ufl.mixed_element([v_cg, p_cg])
        self.W = fem.functionspace(self.domain, mixed_el)
        
        self.w = fem.Function(self.W)
        self.w_n = fem.Function(self.W)
        
        self.u, self.p = ufl.split(self.w)
        self.u_n, self.p_n = ufl.split(self.w_n)
        self.v, self.q = ufl.TestFunctions(self.W)

    def _setup_boundary_conditions(self):
        V, _ = self.W.sub(0).collapse()
        Q, _ = self.W.sub(1).collapse()
        self.bcs = []
        
        # No-slip condition on vessel walls
        u_noslip = fem.Function(V)
        u_noslip.x.array[:] = 0.0
        wall_dofs = fem.locate_dofs_geometrical(
            (self.W.sub(0), V), 
            lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 0.005))
        )
        self.bcs.append(fem.dirichletbc(u_noslip, wall_dofs, self.W.sub(0)))

        # INLET CONDITION: Parabolic Poiseuille velocity profile at x = 0
        u_inlet = fem.Function(V)
        # Peak velocity of 0.2 m/s centered at y = 0.0025 m
        u_inlet.interpolate(
            lambda x: np.vstack((
                0.2 * (1.0 - ((x[1] - 0.0025) / 0.0025) ** 2),
                np.zeros_like(x[0])
            ))
        )
        inlet_dofs = fem.locate_dofs_geometrical((self.W.sub(0), V), lambda x: np.isclose(x[0], 0.0))
        self.bcs.append(fem.dirichletbc(u_inlet, inlet_dofs, self.W.sub(0)))
        
        # Zero pressure at outlet
        p_out = fem.Function(Q)
        p_out.x.array[:] = 0.0
        outlet_dofs = fem.locate_dofs_geometrical((self.W.sub(1), Q), lambda x: np.isclose(x[0], 0.03))
        self.bcs.append(fem.dirichletbc(p_out, outlet_dofs, self.W.sub(1)))

    def _setup_variational_form(self):
        dt = fem.Constant(self.domain, self.dt)
        mu = fem.Constant(self.domain, self.mu)
        rho = fem.Constant(self.domain, self.rho)
        
        # Implicit Midpoint Scheme[cite: 1]
        u_mid = 0.5 * (self.u + self.u_n)
        p_mid = 0.5 * (self.p + self.p_n)
        
        def sigma(u, p):
            return mu * (ufl.grad(u) + ufl.grad(u).T) - p * ufl.Identity(len(u))
        
        # Standard Galerkin weak form[cite: 1]
        F_galerkin = (
            rho * ufl.dot((self.u - self.u_n) / dt, self.v) * ufl.dx +
            rho * ufl.dot(ufl.dot(u_mid, ufl.nabla_grad(u_mid)), self.v) * ufl.dx +
            ufl.inner(sigma(u_mid, p_mid), ufl.grad(self.v)) * ufl.dx +
            ufl.dot(ufl.div(u_mid), self.q) * ufl.dx
        )
        
        # Stabilization parameters: SUPG, PSPG, and LSIC[cite: 1]
        h = ufl.CellDiameter(self.domain)
        u_norm = ufl.sqrt(ufl.dot(u_mid, u_mid) + 1e-10)
        tau_M = 1.0 / ufl.sqrt((2.0 / dt)**2 + (2.0 * u_norm / h)**2 + (4.0 * mu / (rho * h**2))**2)
        tau_C = h * u_norm
        
        R_M = rho * (self.u - self.u_n) / dt + rho * ufl.dot(u_mid, ufl.nabla_grad(u_mid)) + ufl.grad(p_mid)
        
        F_supg = ufl.dot(tau_M * ufl.dot(u_mid, ufl.nabla_grad(self.v)), R_M) * ufl.dx
        F_pspg = ufl.dot(tau_M / rho * ufl.grad(self.q), R_M) * ufl.dx
        F_lsic = tau_C * ufl.div(u_mid) * ufl.div(self.v) * ufl.dx
        
        # Total Residual Form
        self.F = F_galerkin + F_supg + F_pspg + F_lsic
        
        # Directional derivative (Jacobian matrix)
        dw = ufl.TrialFunction(self.W)
        self.J = ufl.derivative(self.F, self.w, dw)

    def _setup_solver(self):
        # 1. Compile UFL forms
        self.residual_form = fem.form(self.F)
        self.jacobian_form = fem.form(self.J)

        # 2. Create PETSc structures
        self.b = petsc.create_vector(self.W)
        self.A = petsc.create_matrix(self.jacobian_form)

        # 3. Define callback functions for PETSc SNES
        def form_residual(snes, x, b):
            x.copy(self.w.x.petsc_vec)
            self.w.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            with b.localForm() as b_local:
                b_local.set(0.0)

            petsc.assemble_vector(b, self.residual_form)
            
            # Positionally passing alpha=-1.0 to ensure compatibility across DOLFINx versions
            petsc.apply_lifting(b, [self.jacobian_form], [self.bcs], [x], -1.0)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            petsc.set_bc(b, self.bcs, x, -1.0)

        def form_jacobian(snes, x, A, P):
            A.zeroEntries()
            petsc.assemble_matrix(A, self.jacobian_form, bcs=self.bcs)
            A.assemble()

            if P != A:
                P.zeroEntries()
                petsc.assemble_matrix(P, self.jacobian_form, bcs=self.bcs)
                P.assemble()

        # 4. Instantiate and configure PETSc SNES solver
        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        self.solver.setOptionsPrefix("nls_")
        self.solver.setFunction(form_residual, self.b)
        self.solver.setJacobian(form_jacobian, self.A)
        self.solver.setTolerances(rtol=1e-6, atol=1e-8, max_it=10)

        # Linear Krylov solver setup
        ksp = self.solver.getKSP()
        ksp.setType("gmres")
        ksp.getPC().setType("bjacobi")
        self.solver.setFromOptions()

    def run(self):
        num_steps = int(self.T / self.dt)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Starting simulation over {num_steps} steps...")
            
        for n in range(num_steps):
            self.t += self.dt
            
            # Solve non-linear system
            self.solver.solve(None, self.w.x.petsc_vec)
            
            num_its = self.solver.getIterationNumber()
            converged = self.solver.getConvergedReason() > 0
            
            # Update solution vector for next time step
            self.w_n.x.array[:] = self.w.x.array
            
            if MPI.COMM_WORLD.rank == 0:
                print(f"Step {n+1}/{num_steps}, Time: {self.t:.4f}, Newton iterations: {num_its}, Converged: {converged}")
        return 0


# - - - - - - - - - - - -  - - - - - -  - - - - - - 
# Pequeño test del código
#  - - - - - -  - - - - - -  - - - - - -  - - - - -

# 50% Stenosis test: simulate low acceleration through the narrow throat and post-stenotic flow separation.

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Must be called BEFORE importing pyplot for headless environments
import matplotlib.pyplot as plt

# 1. Instantiate simulation
sim = Simulation(
    name="stenosis_50",
    simulation="coronary_50",
    solver="snes",
    T=0.03,                      # 6 time steps
    dt=0.005,
    output_dir="./results",
    early_stop_tolerance=1e-5
)

# 2. Warp rectangular mesh geometry to create 50% stenosis at x = 0.005 m
x_coords = sim.domain.geometry.x
x_center, width, reduction = 0.005, 0.0015, 0.25  
bump = reduction * np.exp(-((x_coords[:, 0] - x_center) / width) ** 2)
y_mid = 0.0025
x_coords[:, 1] = y_mid + (x_coords[:, 1] - y_mid) * (1.0 - bump)

# 3. Execute solver loop
sim.run()

# 4. Extract velocity field for plotting
W_vel, vel_map = sim.W.sub(0).collapse()
u_sol = sim.w.sub(0).collapse()

mesh_nodes = sim.domain.geometry.x
cells = sim.domain.topology.connectivity(sim.domain.topology.dim, 0).array.reshape(-1, 3)

u_array = u_sol.x.array.reshape(-1, 2)
u_mag = np.linalg.norm(u_array, axis=1)

# 5. Render and save figure to disk
plt.figure(figsize=(12, 3), dpi=150)
contour = plt.tricontourf(
    mesh_nodes[:, 0], mesh_nodes[:, 1], cells, 
    u_mag[:len(mesh_nodes)], levels=50, cmap="inferno"
)
plt.colorbar(contour, label="Velocity Magnitude (m/s)")
plt.title("Hemodynamic Flow Across 50% Coronary Stenosis")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.gca().set_aspect('equal')
plt.tight_layout()

# Save image file directly
output_file = "stenosis_50_velocity.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n[SUCCESS] Velocity contour saved to {output_file}")
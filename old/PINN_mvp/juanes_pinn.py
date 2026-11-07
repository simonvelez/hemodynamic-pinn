import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Intentar importar FEniCS (silenciando warnings de JIT si es posible)
try:
    from dolfinx import *
    import logging
    logging.getLogger('FFC').setLevel(logging.WARNING)
    logging.getLogger('UFL').setLevel(logging.WARNING)
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    print("Advertencia: FEniCS no esta disponible en este entorno. Solo se ejecutara PyTorch.")

# ==========================================
# CONFIGURACION GENERAL
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILE = "hybrid_dataset.npz"
MODEL_FILE = "pinn_hybrid_master.pth"

# ==========================================
# FASE 1: GENERACION DE DATOS (FEniCS)
# ==========================================
def generate_fenics_data(num_samples=10, AR=2.0, f_m=-1.0):
    if not FENICS_AVAILABLE:
        return

    print("\n--- [FASE 1] Generando Golden Dataset con FEniCS ---")
    data_x, data_p, data_u = [], [], []

    # Parametros fisicos constantes para este lote
    E, nu = 1.0, 0.3
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    start_t = time.time()

    for i in range(num_samples):
        # Escenario tipo Cantilever: Fijo a la izquierda (2), Carga arriba (0)
        W, H = 1.0 * AR, 1.0
        mesh = RectangleMesh(Point(0, 0), Point(W, H), 40, 20)
        V = VectorFunctionSpace(mesh, 'P', 1)

        # Dirichlet BC (Fijo en X=0)
        def left_boundary(x, on_boundary):
            return on_boundary and near(x[0], 0)
        bc = DirichletBC(V, Constant((0, 0)), left_boundary)

        # Neumann BC (Carga en Y=H)
        class TopBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[1], H)

        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        TopBoundary().mark(boundaries, 1)
        ds = Measure("ds")(subdomain_data=boundaries)

        # Carga aleatoria cercana a f_m para dar variedad
        f_current = f_m + np.random.normal(0, 0.1)
        f_vec = Constant((0, f_current))

        u, v = TrialFunction(V), TestFunction(V)
        d = u.geometric_dimension()
        epsilon = sym(grad(u))
        sigma = lambda_ * tr(epsilon) * Identity(d) + 2 * mu * epsilon

        a = inner(sigma, grad(v)) * dx
        L = dot(f_vec, v) * ds(1)

        u_sol = Function(V)
        solve(a == L, u_sol, bc)

        # Extraer datos
        coords = mesh.coordinates()
        u_vals = u_sol.compute_vertex_values(mesh).reshape(2, -1).T

        # Normalizar coordenadas para la PINN [0, 1]
        coords_norm = coords.copy()
        coords_norm[:, 0] /= AR

        # Vector de parametros: [AR, f_edge, f_c, f_w, f_mag, fix_edge, fix_c, fix_w, E]
        p_vec = [AR, 0, 0.5, 1.0, f_current, 2, 0.5, 1.0, E]

        data_x.append(coords_norm)
        data_p.append(np.tile(p_vec, (len(coords_norm), 1)))
        data_u.append(u_vals)

    np.savez(DATA_FILE,
             x=np.vstack(data_x),
             p=np.vstack(data_p),
             u=np.vstack(data_u))

    print(f"Generados {num_samples} escenarios en {time.time()-start_t:.2f}s")
    print(f"Guardado en {DATA_FILE}")

# ==========================================
# FASE 2: ENTRENAMIENTO HIBRIDO (PyTorch)
# ==========================================
class HardConstraintPINN(nn.Module):
    def __init__(self, layers=[11, 128, 128, 128, 128, 2], steepness=10.0):
        super().__init__()
        self.depth = len(layers) - 1
        self.activation = nn.Tanh()
        self.steepness = steepness
        layer_list = []
        for i in range(self.depth):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < self.depth - 1:
                layer_list.append(self.activation)
        self.net = nn.Sequential(*layer_list)

    def _create_dynamic_mask(self, x, y, fix_edge, fix_c, fix_w):
        is_top = torch.exp(-100 * (fix_edge - 0)**2)
        is_bot = torch.exp(-100 * (fix_edge - 1)**2)
        is_left = torch.exp(-100 * (fix_edge - 2)**2)
        is_right = torch.exp(-100 * (fix_edge - 3)**2)
        dist_normal = (torch.abs(1.0 - y) * is_top + torch.abs(y) * is_bot +
                       torch.abs(x) * is_left + torch.abs(1.0 - x) * is_right)
        is_on_edge = torch.tanh(self.steepness * dist_normal)
        return 1.0 - (1.0 - is_on_edge) # Simplificado para el ejemplo

    def forward(self, x_in):
        u_v_raw = self.net(x_in)
        fix_edge, fix_c, fix_w = x_in[:, 7:8], x_in[:, 8:9], x_in[:, 9:10]
        mask = self._create_dynamic_mask(x_in[:, 0:1], x_in[:, 1:2], fix_edge, fix_c, fix_w)
        return u_v_raw * mask

def compute_pde_loss(model, x_in, nu=0.3):
    x_in.requires_grad_(True)
    disp = model(x_in)
    u, v = disp[:, 0:1], disp[:, 1:2]
    AR, E = x_in[:, 2:3], x_in[:, 10:11]

    grads_u = torch.autograd.grad(u, x_in, torch.ones_like(u), create_graph=True)[0]
    grads_v = torch.autograd.grad(v, x_in, torch.ones_like(v), create_graph=True)[0]
    u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2] * AR
    v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2] * AR

    factor = E / (1 - nu**2)
    sigma_xx = factor * (u_x + nu * v_y)
    sigma_yy = factor * (v_y + nu * u_x)
    sigma_xy = (E / (2 * (1 + nu))) * (u_y + v_x)

    sxx_g = torch.autograd.grad(sigma_xx, x_in, torch.ones_like(sigma_xx), create_graph=True)[0]
    syy_g = torch.autograd.grad(sigma_yy, x_in, torch.ones_like(sigma_yy), create_graph=True)[0]
    sxy_g = torch.autograd.grad(sigma_xy, x_in, torch.ones_like(sigma_xy), create_graph=True)[0]

    f_x = sxx_g[:, 0:1] + sxy_g[:, 1:2] * AR
    f_y = sxy_g[:, 0:1] + syy_g[:, 1:2] * AR
    return torch.mean(f_x*2 + f_y*2)

def train_hybrid_pinn(epochs=2000):
    print(f"\n--- [FASE 2] Entrenando PINN Hibrida en {DEVICE} ---")
    model = HardConstraintPINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Cargar datos FEM si existen
    if os.path.exists(DATA_FILE):
        data = np.load(DATA_FILE)
        x_data = torch.tensor(data['x'], dtype=torch.float32).to(DEVICE)
        p_data = torch.tensor(data['p'], dtype=torch.float32).to(DEVICE)
        u_true = torch.tensor(data['u'], dtype=torch.float32).to(DEVICE)
        has_data = True
    else:
        has_data = False

    start_t = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1. Perdida Supervisada (Datos FEniCS)
        loss_data = torch.tensor(0.0).to(DEVICE)
        if has_data:
            idx = torch.randperm(len(x_data))[:500]
            in_data = torch.cat([x_data[idx], p_data[idx]], dim=1)
            u_pred = model(in_data)
            loss_data = torch.mean((u_pred - u_true[idx])**2)

        # 2. Perdida de la Fisica (Collocation points aleatorios)
        x_col = torch.rand(1000, 1).to(DEVICE)
        y_col = torch.rand(1000, 1).to(DEVICE)
        p_col = torch.tensor([[2.0, 0, 0.5, 1.0, -1.0, 2, 0.5, 1.0, 1.0]]).repeat(1000, 1).to(DEVICE)
        in_phys = torch.cat([x_col, y_col, p_col], dim=1)

        loss_pde = compute_pde_loss(model, in_phys)

        # 3. Funcion de Costo Hibrida
        total_loss = loss_pde + (100.0 * loss_data if has_data else 0.0)
        total_loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Total: {total_loss.item():.6f} | PDE: {loss_pde.item():.6f} | Data: {loss_data.item():.6f}")

    print(f"Entrenamiento completado en {time.time() - start_t:.2f}s")
    torch.save(model.state_dict(), MODEL_FILE)
    return model

# ==========================================
# FASE 3: BENCHMARK Y VALIDACION
# ==========================================
def benchmark_model(num_tests=10):
    print("\n--- [FASE 3] Benchmark de Rendimiento ---")
    model = HardConstraintPINN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()

    if not FENICS_AVAILABLE:
        print("FEniCS no esta disponible. No se puede ejecutar la comparativa de tiempos.")
        return

    print(f"Ejecutando {num_tests} pruebas para calcular promedios (se omitira la primera por el compilador JIT)...")

    fem_times = []
    pinn_times = []

    E, nu = 1.0, 0.3
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    for i in range(num_tests + 1):
        AR = 2.0
        W, H = 1.0 * AR, 1.0

        # --- Medir Tiempo FEM ---
        start_fem = time.time()

        mesh = RectangleMesh(Point(0, 0), Point(W, H), 40, 20)
        V = VectorFunctionSpace(mesh, 'P', 1)

        def left_boundary(x, on_boundary): return on_boundary and near(x[0], 0)
        bc = DirichletBC(V, Constant((0, 0)), left_boundary)

        class TopBoundary(SubDomain):
            def inside(self, x, on_boundary): return on_boundary and near(x[1], H)

        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        TopBoundary().mark(boundaries, 1)
        ds = Measure("ds")(subdomain_data=boundaries)

        f_vec = Constant((0, -1.0))
        u, v = TrialFunction(V), TestFunction(V)
        d = u.geometric_dimension()
        epsilon = sym(grad(u))
        sigma = lambda_ * tr(epsilon) * Identity(d) + 2 * mu * epsilon

        a = inner(sigma, grad(v)) * dx
        L = dot(f_vec, v) * ds(1)

        u_sol = Function(V)
        solve(a == L, u_sol, bc)

        end_fem = time.time()

        # --- Preparar datos exactos para la PINN ---
        coords = mesh.coordinates()
        coords_norm = coords.copy()
        coords_norm[:, 0] /= AR
        p_vec = [AR, 0, 0.5, 1.0, -1.0, 2, 0.5, 1.0, E]

        in_tensor = torch.cat([
            torch.tensor(coords_norm, dtype=torch.float32),
            torch.tensor(np.tile(p_vec, (len(coords_norm), 1)), dtype=torch.float32)
        ], dim=1).to(DEVICE)

        # --- Medir Tiempo PINN ---
        # Sincronizamos los hilos si usas aceleracion por hardware
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start_pinn = time.time()

        with torch.no_grad():
            _ = model(in_tensor)

        if torch.cuda.is_available(): torch.cuda.synchronize()
        end_pinn = time.time()

        # Guardar tiempos descartando el calentamiento inicial (i=0)
        if i > 0:
            fem_times.append(end_fem - start_fem)
            pinn_times.append(end_pinn - start_pinn)

    # Resultados y estadisticas
    avg_fem = np.mean(fem_times)
    avg_pinn = np.mean(pinn_times)
    diff = avg_fem - avg_pinn
    speedup = avg_fem / avg_pinn if avg_pinn > 0 else 0

    print("-" * 40)
    print(f"Nodos evaluados por prueba: {len(coords)}")
    print(f"Tiempo promedio FEM:   {avg_fem:.6f} segundos")
    print(f"Tiempo promedio PINN:  {avg_pinn:.6f} segundos")
    print(f"Diferencia promedio:   {diff:.6f} segundos")
    print(f"Aceleracion (Speedup): {speedup:.2f}x veces mas rapido")
    print("-" * 40)

    # Calculo de Error L2
    if os.path.exists(DATA_FILE):
        data = np.load(DATA_FILE)
        x_val = torch.tensor(data['x'], dtype=torch.float32).to(DEVICE)
        p_val = torch.tensor(data['p'], dtype=torch.float32).to(DEVICE)
        u_true = data['u']

        with torch.no_grad():
            u_pred = model(torch.cat([x_val, p_val], dim=1)).cpu().numpy()

        l2_error = np.linalg.norm(u_true - u_pred) / np.linalg.norm(u_true)
        print(f"Error Relativo L2 (vs FEniCS): {l2_error:.4%}")

# ==========================================
# EJECUCION DEL PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Iniciando Pipeline Universal PINN-FEM...")

    generate_fenics_data(num_samples=1000)
    train_hybrid_pinn(epochs=200000)
    benchmark_model()
import os
import numpy as np
from mesh_generator import generate_stenosis_mesh
from solver import run_cfd_simulation


def export_pinn_dataset(domain, u, p, severity, output_dir="./output"):
    """Extracts nodal coordinates, velocity vectors, and pressure values into standard NumPy arrays."""
    coords = domain.geometry.x  # Coordinates (N, 3)
    velocity_values = u.eval(
        coords, np.arange(len(coords))
    )  # Velocity field (N, 3)
    pressure_values = p.eval(
        coords, np.arange(len(coords))
    )  # Pressure field (N, 1)

    np.savez(
        os.path.join(output_dir, f"pinn_dataset_severity_{int(severity*100)}.npz"),
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        u=velocity_values[:, 0],
        v=velocity_values[:, 1],
        w=velocity_values[:, 2],
        p=pressure_values,
    )
    print(
        f"[INFO] PINN dataset saved for severity {severity*100}% with {len(coords)} spatial points."
    )


def main():
    severities = [0.25, 0.50, 0.75]  # Lab benchmark severities
    output_dir = "./output"

    for severity in severities:
        print(
            f"\n================ Running Workflow: Stenosis Severity {int(severity*100)}% ================"
        )

        # Step 1: Generate Mesh
        msh_file = generate_stenosis_mesh(severity=severity, output_dir=output_dir)

        # Step 2: Run FEniCSx CFD Solver
        domain, u, p = run_cfd_simulation(msh_file=msh_file, dt=0.001, t_final=0.05)

        # Step 3: Extract and Save Data for Neural Network Training
        export_pinn_dataset(domain, u, p, severity, output_dir=output_dir)


if __name__ == "__main__":
    main()
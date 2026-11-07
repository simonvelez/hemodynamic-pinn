import argparse
import math
import os
import cadquery as cq
import gmsh
import numpy as np

# Physical boundary tags
INLET_TAG = 1
OUTLET_TAG = 2
WALL_TAG = 3
FLUID_TAG = 4


def generate_stenosis_mesh(
    start=(0.0, 0.0, 0.0),
    end=(0.03, 0.0, 0.0),
    radius_in=0.002,
    radius_out=0.002,
    severity=0.5,
    slope=0.5,
    output_dir="./output",
):
    """Generates 3D stenosis geometry and outputs a structured O-grid .msh file."""
    os.makedirs(output_dir, exist_ok=True)
    msh_path = os.path.join(output_dir, f"stenosis_{int(severity*100)}.msh")
    brep_path = os.path.join(output_dir, f"stenosis_{int(severity*100)}.brep")

    min_radius = (1.0 - severity) * ((radius_in + radius_out) / 2.0)

    # 1. CadQuery Geometry Construction
    # (Uses spline-based taper and 360-degree revolve)
    solid = build_cadquery_solid(
        start, end, radius_in, radius_out, min_radius, slope
    )
    cq.exporters.export(solid, brep_path)

    # 2. Gmsh O-Grid Butterfly Decomposition & Transfinite Subdivision
    gmsh.initialize()
    gmsh.model.add("Stenosis_O_Grid")
    gmsh.model.occ.importShapes(brep_path)
    gmsh.model.occ.synchronize()

    # ... Apply cutting planes, fragment volumes, set transfinite curves/surfaces ...

    gmsh.model.mesh.generate(3)
    gmsh.write(msh_path)
    gmsh.finalize()

    return msh_path
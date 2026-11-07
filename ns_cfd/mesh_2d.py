"""2D stenosed-channel mesh built directly in DOLFINx (no CadQuery / Gmsh needed).

A structured rectangle [0, L] x [-R, R] is warped in y so the wall follows a
smooth cosine stenosis centred at ``x_center``. ``severity`` is the diameter
reduction at the throat (0.5 -> throat diameter is 50% of the healthy one).
Boundary facets are tagged INLET / OUTLET / WALL for the solver.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh

INLET_TAG, OUTLET_TAG, WALL_TAG = 1, 2, 3


def wall_radius(x, severity, radius, x_center, half_length):
    """Local channel half-height r(x) of the cosine stenosis."""
    xi = np.clip((x - x_center) / half_length, -1.0, 1.0)
    return radius * (1.0 - 0.5 * severity * (1.0 + np.cos(np.pi * xi)))


def build_stenosis_mesh(
    severity=0.5,
    length=0.05,
    radius=0.0025,
    x_center=0.01,
    half_length=0.004,
    nx=250,
    ny=40,
    diagonal="crossed",
    comm=MPI.COMM_WORLD,
):
    """Return (domain, facet_tags) for a 2D symmetric stenosis.

    diagonal: how each rectangle is split into triangles. "crossed" (default,
    4 triangles per cell) is mirror-symmetric about y=0. "right"/"left"
    (2 triangles per cell) have symmetric nodes but asymmetric triangles, which
    seeds an artificial asymmetry in the flow (see ns_cfd/mesh_study.py).
    """
    if not 0.0 <= severity < 1.0:
        raise ValueError("severity must be in [0, 1)")

    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, -radius]), np.array([length, radius])],
        [nx, ny],
        cell_type=mesh.CellType.triangle,
        diagonal=mesh.DiagonalType[diagonal],
    )

    # Warp: scale every y by r(x)/R so the walls follow the stenosis profile.
    x = domain.geometry.x
    x[:, 1] *= wall_radius(x[:, 0], severity, radius, x_center, half_length) / radius

    # Tag boundary facets (after warping, so walls are found on the curved boundary).
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    tol = 1e-10 * length
    inlet = mesh.locate_entities_boundary(domain, fdim, lambda p: np.isclose(p[0], 0.0, atol=tol))
    outlet = mesh.locate_entities_boundary(domain, fdim, lambda p: np.isclose(p[0], length, atol=tol))
    all_bnd = mesh.exterior_facet_indices(domain.topology)
    wall = np.setdiff1d(all_bnd, np.union1d(inlet, outlet))

    facets = np.concatenate([inlet, outlet, wall]).astype(np.int32)
    values = np.concatenate([
        np.full(len(inlet), INLET_TAG),
        np.full(len(outlet), OUTLET_TAG),
        np.full(len(wall), WALL_TAG),
    ]).astype(np.int32)
    order = np.argsort(facets)
    facet_tags = mesh.meshtags(domain, fdim, facets[order], values[order])

    return domain, facet_tags

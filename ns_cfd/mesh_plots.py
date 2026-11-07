"""Mesh inspection helpers: node/triangle/flag arrays, element quality, and a
one-page mesh troubleshooting sheet (used by generate_dataset_2d.py and mesh_study.py)."""

import numpy as np


def mesh_arrays(domain, facet_tags):
    """Node coordinates, triangles and boundary flags on the P1 layout."""
    from dolfinx import fem
    from mesh_2d import INLET_TAG, OUTLET_TAG, WALL_TAG
    S = fem.functionspace(domain, ("Lagrange", 1))
    xy = S.tabulate_dof_coordinates()[:, :2]
    tri = S.dofmap.list.copy()
    flag = np.zeros(len(xy), dtype=np.int8)
    fdim = domain.topology.dim - 1
    for tag, val in ((WALL_TAG, 3), (INLET_TAG, 1), (OUTLET_TAG, 2)):
        flag[fem.locate_dofs_topological(S, fdim, facet_tags.find(tag))] = val
    return xy, tri, flag


def element_quality(xy, tri):
    """Per-triangle min angle [deg], aspect ratio (1 = equilateral), longest edge."""
    p = xy[tri]                                              # (M, 3, 2)
    e = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]], axis=1)
    L = np.linalg.norm(e, axis=2)
    area = 0.5 * np.abs(e[:, 0, 0] * (-e[:, 2, 1]) - e[:, 0, 1] * (-e[:, 2, 0]))
    ang = []
    for i in range(3):
        a, b = -e[:, (i + 2) % 3], e[:, i]
        cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        ang.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    min_ang = np.min(ang, axis=0)
    aspect = L.max(axis=1) * L.sum(axis=1) / (4.0 * np.sqrt(3.0) * area)
    return min_ang, aspect, L.max(axis=1), area


def plot_mesh(path, xy, tri, flag, geo, title):
    """Mesh troubleshooting sheet: full mesh, throat zoom, quality maps, spacing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    from mesh_2d import wall_radius

    X, Y = xy[:, 0] * 1e3, xy[:, 1] * 1e3
    T = mtri.Triangulation(X, Y, tri)
    min_ang, aspect, hmax, area = element_quality(xy, tri)
    xc, Ls, R = geo["x_center"] * 1e3, geo["half_length"] * 1e3, geo["radius"] * 1e3

    fig = plt.figure(figsize=(15, 11), dpi=130)
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.5, 1, 1.1])

    ax = fig.add_subplot(gs[0, :])
    ax.triplot(T, lw=0.15, color="0.4")
    for val, col, lab in ((1, "C0", "inlet"), (2, "C3", "outlet"), (3, "C2", "wall")):
        m = flag == val
        ax.plot(X[m], Y[m], ".", ms=1.5, color=col, label=f"{lab} ({m.sum()} nodes)")
    xs = np.linspace(0, geo["length"], 600)
    rw = wall_radius(xs, geo["severity"], geo["radius"], geo["x_center"], geo["half_length"]) * 1e3
    ax.plot(xs * 1e3, rw, "k-", lw=0.6, alpha=0.6)
    ax.plot(xs * 1e3, -rw, "k-", lw=0.6, alpha=0.6, label="analytic wall")
    ax.set_aspect("equal")
    ax.set_title("Full mesh and boundary tags")
    ax.legend(loc="upper right", fontsize=7, ncol=4, markerscale=6)

    ax = fig.add_subplot(gs[1, 0])
    ax.triplot(T, lw=0.3, color="0.2")
    ax.set_xlim(xc - Ls - 1, xc + Ls + 1)
    ax.set_ylim(0, R * 1.05)
    ax.set_aspect("equal")
    ax.set_title("Zoom: upper half of the stenosis")

    ax = fig.add_subplot(gs[1, 1])
    ax.triplot(T, lw=0.4, color="0.2")
    w = (xc - 0.6, xc + 0.6)
    rt = R * (1 - geo["severity"])
    ax.set_xlim(*w)
    ax.set_ylim(rt - 0.35, rt + 0.05)
    ax.set_aspect("equal")
    ax.set_title("Zoom: throat wall (first cells)")

    ax = fig.add_subplot(gs[2, 0])
    c = ax.tripcolor(T, facecolors=min_ang, cmap="viridis_r", vmin=0, vmax=60)
    fig.colorbar(c, ax=ax, label="min angle (deg)", fraction=0.03)
    ax.set_xlim(xc - 2 * Ls, xc + 3 * Ls)
    ax.set_aspect("equal")
    ax.set_title("Element quality: minimum angle")

    ax = fig.add_subplot(gs[2, 1])
    c = ax.tripcolor(T, facecolors=aspect, cmap="magma", vmin=1, vmax=max(3, np.percentile(aspect, 99)))
    fig.colorbar(c, ax=ax, label="aspect ratio", fraction=0.03)
    ax.set_xlim(xc - 2 * Ls, xc + 3 * Ls)
    ax.set_aspect("equal")
    ax.set_title("Element quality: aspect ratio (1 = equilateral)")

    # spacing along x: axial spacing and radial spacing of the wall-adjacent row
    ax = fig.add_subplot(gs[3, 0])
    wall_up = (flag == 3) & (xy[:, 1] > 0)
    xw = np.sort(np.unique(np.round(xy[wall_up, 0], 12)))
    dy_wall = []
    for xv in xw:
        col = np.isclose(xy[:, 0], xv, atol=1e-12)
        yy = np.sort(xy[col & (xy[:, 1] >= 0), 1])
        dy_wall.append(yy[-1] - yy[-2] if len(yy) > 1 else np.nan)
    ax.plot(xw * 1e3, np.array(dy_wall) * 1e3, label="Δy, wall-adjacent row")
    ax.plot(0.5 * (xw[1:] + xw[:-1]) * 1e3, np.diff(xw) * 1e3, label="Δx")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("spacing (mm)")
    ax.set_title("Grid spacing along the channel")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[3, 1])
    ax.hist(min_ang, bins=60, alpha=0.7, label="min angle (deg)")
    ax.set_xlabel("min angle (deg)")
    ax.set_ylabel("cells")
    ax2 = ax.twiny()
    ax2.hist(aspect, bins=60, alpha=0.4, color="C1")
    ax2.set_xlabel("aspect ratio", color="C1")
    ax.set_title("Quality histograms", pad=28)

    stats = (f"{len(tri)} triangles, {len(xy)} nodes | h_max {hmax.max() * 1e3:.3f} mm, "
             f"h_min {hmax.min() * 1e3:.3f} mm | min angle {min_ang.min():.1f}°, "
             f"max aspect {aspect.max():.2f} | inverted cells: {(area <= 0).sum()}")
    fig.suptitle(f"{title}\n{stats}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return dict(cells=int(len(tri)), nodes=int(len(xy)), h_max_mm=float(hmax.max() * 1e3),
                h_min_mm=float(hmax.min() * 1e3), min_angle_deg=float(min_ang.min()),
                max_aspect=float(aspect.max()), inverted=int((area <= 0).sum()))

"""Root-cause audit of the planar symmetry breaking (notes/CFD.md §12): model, implementation or mesh?

Reproducible version of the 2026-11-08 checks, which were first run with throw-away scripts.
Every case is a steady Newton solve (P2/P1 unless the case says P1/P1). Unless stated, it uses the
imperfection continuation (asym_eps = 0.3 -> 0), with the imperfection force from the throat to 2 D
downstream; it lands on the bent-jet branch only if the symmetric state is unstable.

Checks (case groups):
    mesh        every distinct mesh used by the cases is exactly mirror-symmetric: nodes, triangles,
                tagged boundary facets
    noseed      plain steady solve (asym_eps = 0) at 55 % Re 301 stays symmetric, although unstable
    downstream  threshold vs downstream length: L = 50 vs 100 mm (55 % and 50 % at Re 301)
    upstream    threshold vs upstream length: lesion centre at 2 D vs 6 D in a 70 mm channel (55 % and 50 %
                at Re 301). Downstream length then differs (56 vs 36 mm), so the clean comparison for the
                6 D case is the 50 mm baseline (same 36 mm downstream)
    mesh_elem   threshold vs mesh and element: 125x20 / 250x40, P1/P1 / P2/P1 (55 % and 50 % at Re 301)
    re70        Re scan at 70 %: Re_peak 50, 75, 100, 150
    re50        Re scan at 50 %: Re_peak 300, 400, 500

Not covered (recorded only in notes/CFD.md from the 2026-11-08 runs): unperturbed *transient* runs; those
are in stability_test.py / zuluaga_check.py.

    python asymmetry_audit.py list                  # cases, the solve each one uses, and the mesh checks
    python asymmetry_audit.py run                   # everything not yet done (resumable)
    python asymmetry_audit.py run --only re70
    python asymmetry_audit.py run --retry-failed    # also retry solves recorded as failed
    python asymmetry_audit.py report                # asymmetry_audit/report.md

Several cases share one solve (e.g. the 55 % baseline appears in four groups); each distinct solve is run
once and stored as runs/<solve id>.json. A solve that raises is stored with status "failed" and reported.

Re_peak = rho u_max H / mu (H = channel height), the code convention (docs/lab_to_fill_in.yaml). rho, mu,
H and the symmetric/asymmetric threshold are read from solver_2d, mesh_2d and generate_dataset_2d.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import inspect
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # local modules, however the script is invoked

from mesh_2d import build_stenosis_mesh  # noqa: E402
from solver_2d import DEFAULTS  # noqa: E402
from generate_dataset_2d import SYMMETRIC_TOL, nodal_arrays, symmetry_error  # noqa: E402

OUT = HERE / "asymmetry_audit"
RUNS = OUT / "runs"
RHO, MU = DEFAULTS["rho"], DEFAULTS["mu"]
H = 2.0 * inspect.signature(build_stenosis_mesh).parameters["radius"].default


def u_for_re(re):
    return re * MU / (RHO * H)


def case(group, name, severity, re=301.4, length=0.05, x_center=0.01, nx=None, ny=40,
         element="P2P1", asym_eps=0.3, note=""):
    nx = nx if nx is not None else int(round(250 * length / 0.05))
    return dict(group=group, name=name, severity=severity, re=re, u_max=u_for_re(re), length=length,
                x_center=x_center, nx=nx, ny=ny, element=element, asym_eps=asym_eps, note=note)


def solve_id(c):
    """Identifier of the solve a case needs; cases with the same parameters share it."""
    return (f"s{c['severity'] * 100:g}_re{c['re']:g}_L{c['length'] * 1e3:g}_xc{c['x_center'] * 1e3:g}"
            f"_{c['nx']}x{c['ny']}_{c['element']}_eps{c['asym_eps']:g}")


def mesh_id(c):
    return f"s{c['severity'] * 100:g}_L{c['length'] * 1e3:g}_xc{c['x_center'] * 1e3:g}_{c['nx']}x{c['ny']}"


def cases():
    c = []
    c.append(case("noseed", "s55_re301_noseed", 0.55, asym_eps=0.0,
                  note="no imperfection: should stay symmetric although the state is unstable"))
    c.append(case("noseed", "s55_re301_seeded", 0.55, note="same case with the imperfection continuation"))
    for s in (0.50, 0.55):
        tag = f"s{int(round(s * 100))}"
        c.append(case("downstream", f"{tag}_L50", s, note="baseline length 50 mm"))
        c.append(case("downstream", f"{tag}_L100", s, length=0.10, note="downstream length doubled"))
        c.append(case("upstream", f"{tag}_xc10", s, length=0.07, x_center=0.01,
                      note="lesion centre 2 D from inlet, L 70 mm"))
        c.append(case("upstream", f"{tag}_xc30", s, length=0.07, x_center=0.03,
                      note="lesion centre 6 D from inlet, L 70 mm (36 mm downstream, as the 50 mm baseline)"))
        for nx, ny in ((125, 20), (250, 40)):
            for el in ("P1P1", "P2P1"):
                c.append(case("mesh_elem", f"{tag}_{nx}x{ny}_{el}", s, nx=nx, ny=ny, element=el))
    for re in (50, 75, 100, 150):
        c.append(case("re70", f"s70_re{re}", 0.70, re=re))
    for re in (300, 400, 500):
        c.append(case("re50", f"s50_re{re}", 0.50, re=re))
    return c


def host():
    cpu = platform.processor()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return dict(hostname=platform.node(), cpu=cpu, omp_threads=os.environ.get("OMP_NUM_THREADS"))


def mirror_check(c):
    """Exact mirror symmetry of the mesh a case uses: nodes, triangles and tagged boundary facets."""
    from mpi4py import MPI
    from dolfinx.mesh import entities_to_geometry
    domain, ft = build_stenosis_mesh(severity=c["severity"], length=c["length"], x_center=c["x_center"],
                                     nx=c["nx"], ny=c["ny"], comm=MPI.COMM_SELF)
    x = domain.geometry.x[:, :2]
    scale = np.abs(x).max()
    key = lambda p: (round(p[0] / scale, 12), round(p[1] / scale, 12) + 0.0)
    index = {key(p): i for i, p in enumerate(x)}
    mirror = np.array([index.get(key((p[0], -p[1])), -1) for p in x])
    has = mirror >= 0
    node_dev = float(np.max(np.abs(x[mirror[has], 1] + x[has, 1]))) if has.any() else float("nan")

    def mirrored(nodes):
        """Sorted mirror image of a node set, or None if any node has no mirror (no -1 indexing)."""
        m = mirror[np.asarray(nodes)]
        return None if (m < 0).any() else tuple(sorted(m))

    cells = domain.geometry.dofmaps[0]
    tri = {tuple(sorted(t)) for t in cells}
    tri_ok = sum(1 for t in cells if (m := mirrored(t)) is not None and m in tri)
    domain.topology.create_connectivity(1, 2)
    fgeom = entities_to_geometry(domain, 1, np.asarray(ft.indices, dtype=np.int32))  # geometry node ids
    facets = {tuple(sorted(g)): int(tag) for g, tag in zip(fgeom, ft.values)}
    fac_ok = sum(1 for k, tag in facets.items() if (m := mirrored(k)) is not None and facets.get(m) == tag)
    ok = bool(has.all() and tri_ok == len(cells) and fac_ok == len(facets))
    return dict(mesh_id=mesh_id(c), severity=c["severity"], length=c["length"], x_center=c["x_center"],
                mesh=f"{c['nx']}x{c['ny']}", nodes=len(x), mirrored_nodes=int(has.sum()),
                max_node_mirror_dev_m=node_dev, triangles=len(cells), mirrored_triangles=int(tri_ok),
                tagged_facets=len(facets), mirrored_facets_same_tag=int(fac_ok), symmetric=ok)


def run_solve(c):
    from mpi4py import MPI
    from solver_2d import run_simulation
    t0 = time.perf_counter()
    domain, ft = build_stenosis_mesh(severity=c["severity"], length=c["length"], x_center=c["x_center"],
                                     nx=c["nx"], ny=c["ny"], comm=MPI.COMM_SELF)
    u_h, p_h, info, _ = run_simulation(domain, ft, u_max=c["u_max"], element=c["element"], steady_solve=True,
                                       asym_eps=c["asym_eps"], verbose=False,
                                       # imperfection from the throat to 2 D downstream, wherever the lesion is
                                       asym_x=(c["x_center"], c["x_center"] + 0.01))
    xy, uv, p, _, _ = nodal_arrays(domain, ft, u_h, p_h)
    sym = symmetry_error(xy, uv)
    return dict(status="ok", symmetry_error=sym, centreline_asym=float(info["asym_history"][-1][1]),
                branch="symmetric" if sym < SYMMETRIC_TOL else "asymmetric", dp_pa=float(info["dp"]),
                mass_error=float(info["mass_error"]), newton_iterations=int(info["newton_iterations"]),
                wall_s=time.perf_counter() - t0)


def solve_params(c):
    return {k: c[k] for k in ("severity", "re", "u_max", "length", "x_center", "nx", "ny", "element", "asym_eps")}


def load_mesh_checks():
    p = OUT / "mesh_check.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    data = data if isinstance(data, list) else []  # older single-mesh format: recheck everything
    return {m["mesh_id"]: m for m in data if "mesh_id" in m}


def report():
    by = {p.stem: json.loads(p.read_text()) for p in RUNS.glob("*.json")}
    hosts = sorted({f"{r['host']['cpu']} ({r['host']['hostname']}, OMP threads {r['host']['omp_threads']})"
                    for r in by.values() if "host" in r})
    elements = sorted({r["element"] for r in by.values()})
    lines = ["# Asymmetry root-cause audit (`ns_cfd/asymmetry_audit.py`)", "",
             f"Steady Newton solves ({', '.join(elements)}; imperfection continuation 0.3 → 0 unless the "
             f"imperfection column says 0). Re_peak = ρ u_max H / μ with ρ = {RHO:g} kg/m³, μ = {MU:g} Pa s, "
             f"H = {H * 1e3:g} mm. Centreline asym = max|v| on y = 0 / u_max. Branch: asymmetric if the node-wise "
             f"symmetry error ≥ {SYMMETRIC_TOL:g}. Cases sharing a solve show the same row.", "",
             f"Hardware: {'; '.join(hosts) if hosts else 'not recorded'}.", ""]
    checks = load_mesh_checks()
    if checks:
        lines += ["## Mesh mirror symmetry (every mesh used)", "",
                  "| mesh | DS | L (mm) | lesion centre (mm) | nodes mirrored | max node deviation (m) "
                  "| triangles mirrored | tagged facets mirrored | symmetric |", "|---|---|---|---|---|---|---|---|---|"]
        for m in sorted(checks.values(), key=lambda m: m["mesh_id"]):
            lines.append(f"| {m['mesh']} | {m['severity']:.0%} | {m['length'] * 1e3:g} | {m['x_center'] * 1e3:g} "
                         f"| {m['mirrored_nodes']}/{m['nodes']} | {m['max_node_mirror_dev_m']:.1e} "
                         f"| {m['mirrored_triangles']}/{m['triangles']} "
                         f"| {m['mirrored_facets_same_tag']}/{m['tagged_facets']} | {'yes' if m['symmetric'] else '**NO**'} |")
        lines.append("")
    titles = dict(noseed="Unseeded vs seeded steady solve (55 %, Re 301)",
                  downstream="Downstream length", upstream="Upstream length",
                  mesh_elem="Mesh and element", re70="Re scan at 70 %", re50="Re scan at 50 %")
    allc = cases()
    for g, title in titles.items():
        group = [c for c in allc if c["group"] == g]
        lines += [f"## {title}", "", "| case | DS | Re_peak | L (mm) | lesion centre (mm) | mesh | element | imperfection "
                  "| centreline asym | symmetry err | branch | Δp (Pa) | time (s) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in group:
            r = by.get(solve_id(c))
            head = (f"| {c['name']} | {c['severity']:.0%} | {c['re']:.0f} | {c['length'] * 1e3:.0f} "
                    f"| {c['x_center'] * 1e3:.0f} | {c['nx']}×{c['ny']} | {c['element']} | {c['asym_eps']:g} ")
            if r is None:
                lines.append(head + "| not run | | | | |")
            elif r["status"] != "ok":
                lines.append(head + f"| **FAILED**: {r['error'][:80]} | | | | |")
            else:
                lines.append(head + f"| {r['centreline_asym']:.4f} | {r['symmetry_error']:.1e} | {r['branch']} "
                             f"| {r['dp_pa']:.2f} | {r['wall_s']:.0f} |")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "run", "report"])
    ap.add_argument("--only", nargs="*", help="case groups or names (\"mesh\" = the mesh checks only)")
    ap.add_argument("--retry-failed", action="store_true", help="rerun solves recorded as failed")
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    allc = cases()
    sel = [c for c in allc if not a.only or c["group"] in a.only or c["name"] in a.only]
    do_mesh = not a.only or "mesh" in a.only
    mesh_cases = {mesh_id(c): c for c in (allc if do_mesh else [])}
    checks = load_mesh_checks()

    if a.cmd == "report":
        report()
        return
    if a.cmd == "list":
        for mid in mesh_cases:
            m = checks.get(mid)
            st = "todo" if m is None else ("done" if m["symmetric"] else "ASYMMETRIC")
            print(f"[{st}] {'mesh':10s} {mid}")
        for c in sel:
            sid = solve_id(c)
            p = RUNS / f"{sid}.json"
            st = "todo" if not p.exists() else json.loads(p.read_text())["status"].replace("ok", "done")
            print(f"[{st}] {c['group']:10s} {c['name']:18s} solve {sid}  {c['note']}")
        return

    # run
    for mid, c in mesh_cases.items():
        if mid in checks:
            continue
        checks[mid] = mirror_check(c)
        m = checks[mid]
        print(f"[mesh] {mid}: {'symmetric' if m['symmetric'] else 'NOT SYMMETRIC'} "
              f"({m['mirrored_triangles']}/{m['triangles']} triangles)", flush=True)
        (OUT / "mesh_check.json").write_text(json.dumps(list(checks.values()), indent=2))
    todo = {}
    for c in sel:
        todo.setdefault(solve_id(c), c)
    for sid, c in todo.items():
        out = RUNS / f"{sid}.json"
        if out.exists() and (json.loads(out.read_text())["status"] == "ok" or not a.retry_failed):
            continue
        print(f"[run] {sid} ...", flush=True)
        try:
            r = run_solve(c)
        except Exception as e:  # a failed Newton solve is itself a result: record it
            r = dict(status="failed", error=f"{type(e).__name__}: {e}")
            print(f"[run] {sid} FAILED: {r['error']}", flush=True)
        out.write_text(json.dumps({"solve_id": sid, **solve_params(c), **r, "host": host()}, indent=2))
        if r["status"] == "ok":
            print(f"[run] {sid}: branch={r['branch']} asym={r['centreline_asym']:.4f} "
                  f"sym_err={r['symmetry_error']:.1e} dp={r['dp_pa']:.2f} Pa ({r['wall_s']:.0f} s)", flush=True)
    report()


if __name__ == "__main__":
    main()

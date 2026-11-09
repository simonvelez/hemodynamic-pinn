"""Run provenance and cost logging shared by the CFD (ns_cfd/) and the PINN (PINN/).

Every run writes two JSON files next to its outputs:

  run_manifest.json  what ran where: git commit + dirty flag, command line, host,
                     CPU/GPU model, thread settings, package versions, seeds, start/end.
  cost.json          how long each stage took (wall clock and process CPU time),
                     peak resident memory, plus free-form counters (steps, dofs, ...).

Both sides log the same fields so the CFD-vs-PINN cost comparison (break-even,
accuracy-matched cost) can be computed mechanically from the files.

Usage:
    from manifest import RunLog
    log = RunLog(out_dir, kind="cfd", config=dict(...), seed=0)
    with log.stage("solve"):
        ...
    log.counters.update(n_steps=..., n_dofs=...)
    log.finish(metrics={...})          # writes both files (also on failure via finish(failed=True))

No third-party dependencies; torch/dolfinx/petsc versions are recorded only if importable.
"""

import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_info():
    """GPU name(s) via torch if available, else nvidia-smi, else None."""
    try:
        import torch
        if torch.cuda.is_available():
            return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.splitlines() or None
    except Exception:
        return None


def package_versions():
    vers = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "torch", "dolfinx", "basix", "ufl", "petsc4py", "mpi4py", "matplotlib"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "?")
        except Exception:
            continue
    try:
        from petsc4py import PETSc
        vers["petsc"] = ".".join(map(str, PETSc.Sys.getVersion()))
    except Exception:
        pass
    try:
        import torch
        vers["torch_cuda"] = torch.version.cuda
        vers["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    return vers


def peak_rss_mb():
    """Peak resident set size of this process (Linux reports KiB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024.0 if sys.platform.startswith("linux") else r / 2**20


class RunLog:
    def __init__(self, out_dir, kind, config=None, seed=None, prefix=""):
        """prefix: prepended to the file names (several runs sharing one folder,
        e.g. 'pinn2d_severity_50_' -> pinn2d_severity_50_cost.json)."""
        self.out = Path(out_dir)
        self.prefix = prefix
        self.out.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        self.t_start = time.time()
        self.cpu_start = time.process_time()
        self.stages = {}          # name -> {"wall_s": float, "cpu_s": float, "calls": int}
        self.counters = {}
        dirty = _git("status", "--porcelain")
        self.manifest = dict(
            kind=kind,
            git_commit=_git("rev-parse", "HEAD"),
            git_dirty=bool(dirty),
            command=" ".join(sys.argv),
            cwd=os.getcwd(),
            hostname=socket.gethostname(),
            platform=platform.platform(),
            cpu_model=cpu_model(),
            n_cpus=os.cpu_count(),
            gpu=gpu_info(),
            threads_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                      "OPENBLAS_NUM_THREADS") if os.environ.get(k)},
            versions=package_versions(),
            seed=seed,
            config=config or {},
            t_start=datetime.now(timezone.utc).isoformat(),
        )
        self._write("run_manifest.json", {**self.manifest, "status": "running"})

    @contextmanager
    def stage(self, name):
        """Time a stage (accumulates if entered several times)."""
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            st = self.stages.setdefault(name, dict(wall_s=0.0, cpu_s=0.0, calls=0))
            st["wall_s"] += time.perf_counter() - w0
            st["cpu_s"] += time.process_time() - c0
            st["calls"] += 1

    def finish(self, metrics=None, failed=False):
        wall = time.time() - self.t_start
        cost = dict(kind=self.kind, total_wall_s=wall, total_cpu_s=time.process_time() - self.cpu_start,
                    stages=self.stages, peak_rss_mb=peak_rss_mb(), counters=self.counters,
                    cpu_model=self.manifest["cpu_model"], gpu=self.manifest["gpu"],
                    threads_env=self.manifest["threads_env"])
        self._write("cost.json", cost)
        self._write("run_manifest.json", {**self.manifest, "status": "failed" if failed else "done",
                                          "t_end": datetime.now(timezone.utc).isoformat(),
                                          "metrics": metrics or {}})
        return cost

    def _write(self, name, obj):
        (self.out / f"{self.prefix}{name}").write_text(json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)

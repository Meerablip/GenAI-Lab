"""Shared helpers for Practical 4: environment snapshot, resource monitor, JSON output."""
from __future__ import annotations

import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import psutil

GITHUB_ENV_KEYS = [
    "GITHUB_REPOSITORY", "GITHUB_WORKFLOW", "GITHUB_JOB", "GITHUB_RUN_ID",
    "GITHUB_RUN_NUMBER", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA", "GITHUB_REF_NAME",
    "GITHUB_SERVER_URL", "GITHUB_ACTOR", "RUNNER_OS", "RUNNER_ARCH",
    "RUNNER_NAME", "ImageOS", "ImageVersion",
]


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def snapshot_env(packages: list[str]) -> dict:
    """Describe the machine and software stack the experiment ran on."""
    vm = psutil.virtual_memory()
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / 1024**3, 2),
        "ram_available_gb": round(vm.available / 1024**3, 2),
        "disk_free_gb": round(psutil.disk_usage("/").free / 1024**3, 1),
        "packages": {p: package_version(p) for p in packages},
        "github": {k: os.environ[k] for k in GITHUB_ENV_KEYS if k in os.environ},
    }
    try:
        import torch

        env["torch"] = {
            "version": torch.__version__,
            "threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        }
    except ImportError:
        pass
    gh = env["github"]
    if {"GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"} <= gh.keys():
        env["run_url"] = f"{gh['GITHUB_SERVER_URL']}/{gh['GITHUB_REPOSITORY']}/actions/runs/{gh['GITHUB_RUN_ID']}"
    return env


class ResourceMonitor:
    """Context manager: wall time, peak process RSS, peak system RAM and mean CPU use.

    A background thread samples every ``interval`` seconds. ``cpu_cores_used`` is
    the process CPU time divided by wall time, i.e. the average number of busy cores.
    """

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.result: dict = {}

    def _sample(self) -> None:
        while not self._stop.is_set():
            self._peak_rss = max(self._peak_rss, self.proc.memory_info().rss)
            self._peak_sys = max(self._peak_sys, psutil.virtual_memory().percent)
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        self._peak_rss = self.proc.memory_info().rss
        self._peak_sys = psutil.virtual_memory().percent
        self._cpu0 = sum(self.proc.cpu_times()[:2])
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        wall = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join()
        cpu = sum(self.proc.cpu_times()[:2]) - self._cpu0
        self._peak_rss = max(self._peak_rss, self.proc.memory_info().rss)
        self.result = {
            "wall_s": round(wall, 3),
            "peak_rss_gb": round(self._peak_rss / 1024**3, 3),
            "peak_system_ram_pct": round(self._peak_sys, 1),
            "cpu_cores_used": round(cpu / wall, 2) if wall > 0 else 0.0,
        }


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalise(text: str) -> str:
    """Lower-case, drop thousands separators and collapse whitespace (for answer matching)."""
    import re

    text = text.lower().replace("’", "'")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_key(answer: str, key: str) -> bool:
    """True if ``key`` occurs in ``answer`` as a whole token or phrase (case-insensitive)."""
    import re

    a, k = normalise(answer), normalise(key)
    pattern = r"(?<![\w.:,])" + re.escape(k) + r"(?!\w|[.,:]\d)"
    return re.search(pattern, a) is not None

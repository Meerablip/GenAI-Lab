"""Shared utilities for Practical 6: Multi-Agent Marketing & Finance Business Workflow.

Provides environment inspection, background resource monitoring, LLM client interfacing
with Ollama (with offline/smoke-test deterministic fallback), and JSON serialization helpers.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import psutil
import requests

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("prac06")

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


def snapshot_env(packages: list[str]) -> dict[str, Any]:
    """Describe host system, CPU/RAM specs, installed Python libraries, and GitHub runner context."""
    vm = psutil.virtual_memory()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_logical_cores": os.cpu_count() or 1,
        "cpu_physical_cores": psutil.cpu_count(logical=False) or 1,
        "ram_total_gb": round(vm.total / 1024**3, 2),
        "ram_available_gb": round(vm.available / 1024**3, 2),
        "disk_free_gb": round(psutil.disk_usage("/").free / 1024**3, 1),
        "packages": {p: package_version(p) for p in packages},
        "github": {k: os.environ[k] for k in GITHUB_ENV_KEYS if k in os.environ},
    }


class ResourceMonitor:
    """Sample CPU utilization (%) and RAM usage (GB) in background at fixed intervals."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.cpu_samples: list[float] = []
        self.ram_samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self.cpu_samples.clear()
        self.ram_samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                self.cpu_samples.append(proc.cpu_percent())
                self.ram_samples.append(proc.memory_info().rss / 1024**3)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        c = self.cpu_samples or [0.0]
        r = self.ram_samples or [0.0]
        return {
            "cpu_mean_pct": round(sum(c) / len(c), 1),
            "cpu_max_pct": round(max(c), 1),
            "ram_mean_gb": round(sum(r) / len(r), 2),
            "ram_max_gb": round(max(r), 2),
            "sample_count": len(c),
        }


class LLMClient:
    """Interacts with local Ollama server if available; falls back to structured synthesis if offline."""

    def __init__(self, model: str = "llama3.2:1b", host: str = "http://127.0.0.1:11434", smoke: bool = False):
        self.model = model
        self.host = host.rstrip("/")
        self.smoke = smoke
        self.is_connected = False
        if not smoke:
            self._check_connection()

    def _check_connection(self):
        try:
            resp = requests.get(f"{self.host}/api/version", timeout=1.5)
            if resp.status_code == 200:
                self.is_connected = True
                logger.info(f"Ollama server connected at {self.host} (version: {resp.json().get('version')})")
            else:
                self.is_connected = False
        except Exception:
            self.is_connected = False
            logger.info(f"Ollama server not reachable at {self.host}; using intelligent deterministic synthesizer.")

    def generate(self, prompt: str, system: str = "", temperature: float = 0.2) -> tuple[str, dict[str, Any]]:
        """Generate response via Ollama or structured fallback, returning (text, telemetry)."""
        t0 = time.perf_counter()
        if self.is_connected and not self.smoke:
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": 768,
                    },
                }
                res = requests.post(f"{self.host}/api/generate", json=payload, timeout=60.0)
                if res.status_code == 200:
                    data = res.json()
                    elapsed = time.perf_counter() - t0
                    out = data.get("response", "").strip()
                    telemetry = {
                        "engine": "ollama",
                        "model": self.model,
                        "elapsed_s": round(elapsed, 3),
                        "eval_count": data.get("eval_count", 0),
                        "eval_duration_ms": round(data.get("eval_duration", 0) / 1e6, 1),
                        "prompt_eval_count": data.get("prompt_eval_count", 0),
                    }
                    return out, telemetry
            except Exception as e:
                logger.warning(f"Ollama call failed ({e}); falling back to deterministic synthesizer")

        # Fallback synthesis
        time.sleep(0.05)
        elapsed = time.perf_counter() - t0
        telemetry = {
            "engine": "synthesizer_fallback",
            "model": self.model,
            "elapsed_s": round(elapsed, 3),
            "eval_count": len(prompt.split()),
            "eval_duration_ms": round(elapsed * 1000, 1),
            "prompt_eval_count": len(prompt.split()),
        }
        return prompt, telemetry


def extract_json_block(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract a valid JSON object or list from LLM output, parsing code fences if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidate = match.group(1).strip() if match else text
    try:
        return json.loads(candidate)
    except Exception:
        # Try finding outermost { ... }
        s = candidate.find("{")
        e = candidate.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(candidate[s : e + 1])
            except Exception:
                pass
    return None


def write_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(val: Any, default: float = 0.0) -> float:
    """Robustly parse float from numbers or currency/percentage strings."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = re.sub(r"[^\d.-]", "", val)
        try:
            return float(cleaned)
        except Exception:
            return default
    return default


def clean_string_list(items: Any, default: list[str] | None = None) -> list[str]:
    """Ensure a list of items is cleanly converted to a list of strings."""
    if not isinstance(items, list):
        return default or []
    out = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.append(item.get("title") or item.get("name") or item.get("pillar") or str(item))
        else:
            out.append(str(item))
    return out or (default or [])


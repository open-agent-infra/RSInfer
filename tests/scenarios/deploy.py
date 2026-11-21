"""The engine as an operator starts it, ``deploy/agxinfer.py serve``, for the end-to-end tests
(``test_m2a.py`` on the CPU, ``tests/gpu/test_qwen38_serve.py`` on a GPU)."""

from __future__ import annotations

import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "deploy" / "agxinfer.py"
ENV = {**os.environ, "PYTHONPATH": str(REPO / "engine")}


def cli(*args: str, timeout: float = 120) -> str:
    out = subprocess.run([sys.executable, str(CLI), *args], env=ENV, capture_output=True, text=True, timeout=timeout)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def free_port() -> int:
    """A port free now, below the kernel's ephemeral range (32768 and up): a port ``bind(0)`` hands
    out may be taken as the local end of some other connection before the engine binds it."""
    for _ in range(200):
        port = random.randrange(20000, 32000)
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 20000..32000")


@contextmanager
def served(run: Path, *args: str, ready_s: float = 90) -> Generator[str]:
    """``serve`` with ``args`` until the frontend answers ``/health``; its URL. SIGTERM at the end."""
    port = free_port()
    log_path = run.parent / f"{run.name}.log"
    log = open(log_path, "wb")  # noqa: SIM115 - the child writes to it
    proc = subprocess.Popen([sys.executable, str(CLI), "serve", "--port", str(port), "--run", str(run), *args],
                            env=ENV, stdout=log, stderr=log, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + ready_s
        while b"engine up" not in log_path.read_bytes():  # the CLI says so once the start-up change committed
            assert proc.poll() is None, log_path.read_text()[-4000:]
            assert time.monotonic() < deadline, "the engine did not come up"
            time.sleep(0.3)
        with urllib.request.urlopen(url + "/v1/models", timeout=10) as r:  # and it is this engine on that port
            assert json.loads(r.read())["data"][0]["owned_by"] == "rsinfer", f"another server on port {port}"
        yield url
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def post(url: str, body: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise AssertionError(f"{e.code}: {e.read().decode()}") from e

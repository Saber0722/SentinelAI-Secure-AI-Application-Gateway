"""Starts an isolated gateway and UI for the browser tests: uv run pytest tests/e2e

Uses a temporary audit database, the scripted vulnerable model and the rules only, so results are repeatable and no
real data is touched. Add --headed to watch, --video on to record, or set SENTINEL_E2E_URL to test a running site.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from chat_page import ChatPage

sys.path.insert(0, str(Path(__file__).parents[1] / "security"))  # shares corpus.py with the security tests

CLI = Path(sys.executable).parent / "sentinelai"
ADMIN_TOKEN = "e2e-admin-token-0123456789"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(url: str, process: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{process.args} exited early:\n{log.read_text()}")
        try:
            urllib.request.urlopen(url, timeout=2).close()
            return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"{url} did not come up:\n{log.read_text()}")


@pytest.fixture(scope="session")
def base_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    if external := os.environ.get("SENTINEL_E2E_URL"):
        yield external
        return
    workdir = tmp_path_factory.mktemp("e2e")
    api_port, ui_port = _free_port(), _free_port()
    env = {
        **os.environ,
        "SENTINEL_MODEL_BACKEND": "mock",
        "SENTINEL_CLASSIFIER_ENABLED": "false",
        "SENTINEL_VAR_DIR": str(workdir),
        "SENTINEL_DB_PATH": str(workdir / "sentinel.db"),
        "SENTINEL_ADMIN_TOKEN": ADMIN_TOKEN,
        "SENTINEL_API_URL": f"http://127.0.0.1:{api_port}",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }
    commands = {
        "api": ([str(CLI), "serve", "--host", "127.0.0.1", "--port", str(api_port)], f"http://127.0.0.1:{api_port}/health"),
        "ui": ([str(CLI), "ui", "--host", "127.0.0.1", "--port", str(ui_port)], f"http://127.0.0.1:{ui_port}/_stcore/health"),
    }
    processes = []
    try:
        for name, (command, health) in commands.items():
            log = workdir / f"{name}.log"
            with log.open("w") as out:
                process = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT)
            processes.append(process)
            _wait_for(health, process, log)
        yield f"http://127.0.0.1:{ui_port}"
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=20)


@pytest.fixture
def chat(page, base_url: str) -> ChatPage:
    return ChatPage(page, base_url).open()

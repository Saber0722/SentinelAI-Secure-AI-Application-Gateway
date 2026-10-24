"""Command-line entry point: ``uv run sentinelai serve|ui|check-env``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sentinelai.config import PROJECT_ROOT, get_settings


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "sentinelai.api.app:create_app",
        factory=True,
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        log_level="info",
    )


def _ui(args: argparse.Namespace) -> None:
    app = Path(__file__).parent / "ui" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
    ]
    # Replace this process with Streamlit so signals (Ctrl+C, kill) reach it directly.
    os.chdir(PROJECT_ROOT)
    os.execv(sys.executable, cmd)


def _check_env(_: argparse.Namespace) -> None:
    from sentinelai.environment import summarize_environment

    print(json.dumps(summarize_environment(PROJECT_ROOT), indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sentinelai", description="SentinelAI security gateway")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the FastAPI gateway")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.set_defaults(func=_serve)
    ui = sub.add_parser("ui", help="run the Streamlit dashboard")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8501)
    ui.set_defaults(func=_ui)
    sub.add_parser("check-env", help="print hardware/runtime information").set_defaults(func=_check_env)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

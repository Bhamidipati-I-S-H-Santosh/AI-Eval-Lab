"""Local dashboard launcher.

    python -m evalkit ui                 start the dashboard on http://127.0.0.1:8765 and open it
    python -m evalkit ui --stop          stop a dashboard started in the background
    python -m evalkit run CONFIG         opens the dashboard on the new run automatically

The local dashboard is the same server as `evalkit serve`, bound to 127.0.0.1
only, with sign-in disabled (it refuses any other address in that mode). It
reads the same results folder the CLI writes to, so every run appears there
while it is still running.

Automatic opening is skipped in CI (the CI environment variable), with
`--no-ui`, or when EVALKIT_NO_UI=1.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

DEFAULT_PORT = int(os.environ.get("EVALKIT_UI_PORT", "8765"))
HOST = "127.0.0.1"


def url(port: int = DEFAULT_PORT, path: str = "") -> str:
    return f"http://{HOST}:{port}/{path}"


def disabled() -> bool:
    return bool(os.environ.get("EVALKIT_NO_UI") or os.environ.get("CI"))


def is_up(port: int = DEFAULT_PORT, timeout: float = 0.5) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url(port, "healthz"), timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def local_server_config(root: str, configs_dir: str = "examples", datasets_dir: str = "data") -> dict:
    return {"storage": {"type": "local", "root": str(root)}, "configs_dir": str(configs_dir),
            "datasets_dir": str(datasets_dir), "auth": {"mode": "none"}, "max_parallel_runs": 1}


def _state_dir(root: str | Path) -> Path:
    d = Path(root) / "_ui"
    d.mkdir(parents=True, exist_ok=True)
    return d


def start_background(root: str | Path, port: int = DEFAULT_PORT, configs_dir: str = "examples",
                     wait_s: float = 20.0) -> bool:
    """Start the local dashboard as a detached process; True once it answers."""
    if is_up(port):
        return True
    state = _state_dir(root)
    log = open(state / "server.log", "a", encoding="utf-8")
    cmd = [sys.executable, "-m", "evalkit", "ui", "--no-browser", "--port", str(port),
           "--root", str(root), "--configs", configs_dir]
    kw: dict = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "cwd": os.getcwd()}
    if os.name == "nt":
        kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, env={**os.environ, "PYTHONUTF8": "1"}, **kw)
    (state / "server.pid").write_text(str(proc.pid), encoding="utf-8")
    end = time.time() + wait_s
    while time.time() < end:
        if is_up(port):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.3)
    return False


def open_run(run_id: str, root: str | Path, port: int = DEFAULT_PORT, log=print) -> None:
    """Called by `evalkit run` when a run starts: show it in the dashboard."""
    if disabled():
        return
    if not start_background(root, port):
        log(f"  (dashboard did not start; see {_state_dir(root) / 'server.log'})")
        return
    link = url(port, f"#/runs/{run_id}")
    log(f"  Dashboard: {link}")
    try:
        webbrowser.open(link, new=2)
    except Exception:
        pass


def stop(root: str | Path) -> str:
    pid_file = _state_dir(root) / "server.pid"
    if not pid_file.exists():
        return "no background dashboard recorded"
    pid = int(pid_file.read_text(encoding="utf-8").strip() or 0)
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
        else:
            os.kill(pid, 15)
    finally:
        pid_file.unlink(missing_ok=True)
    return f"stopped dashboard (pid {pid})"


def serve_local(root: str, port: int = DEFAULT_PORT, configs_dir: str = "examples", browser: bool = True):
    """Run the local dashboard in the foreground."""
    import uvicorn

    from . import audit
    from .server.app import create_app
    from dotenv import load_dotenv
    load_dotenv()
    if is_up(port):
        print(f"Dashboard already running at {url(port)}")
        if browser:
            webbrowser.open(url(port), new=2)
        return
    app = create_app(local_server_config(root, configs_dir))
    audit.emit("ui.started", port=port, root=str(root))
    print(f"evalkit dashboard: {url(port)}  (local only; Ctrl+C to stop)")
    if browser:
        import threading
        threading.Timer(1.2, lambda: webbrowser.open(url(port), new=2)).start()
    uvicorn.run(app, host=HOST, port=port, log_level="warning")

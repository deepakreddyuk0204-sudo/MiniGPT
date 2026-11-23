"""Start the local chat model and Streamlit. Press Ctrl-C to stop newly started services."""

from contextlib import ExitStack
import fcntl
import http.client
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minigpt import local_chat
from scripts.download_chat_model import MANIFEST, verify
from scripts.setup_llama import TOOLS, VERSION

HOST = "127.0.0.1"
UI_PORT = 8501


def listener(port):
    """Find the process that accepted an IPv4 loopback connection on this port."""
    try:
        with socket.create_connection((HOST, port), timeout=1):
            pass
    except ConnectionRefusedError:
        return None
    query = subprocess.run(
        ["lsof", "-nP", "-a", f"-iTCP@{HOST}:{port}", "-sTCP:LISTEN", "-Fpn"],
        text=True, capture_output=True,
    )
    lines = query.stdout.splitlines()
    pids = [line[1:] for line in lines if line.startswith("p")]
    if len(pids) != 1 or f"n{HOST}:{port}" not in lines:
        raise RuntimeError(f"Port {port} is busy. Could not identify a loopback-only listener.")
    return psutil.Process(int(pids[0]))


def same_ui(process, backend_port):
    args = process.cmdline()
    if "run" not in args:
        return False
    run = args.index("run")
    if run == 0 or run + 1 >= len(args) or Path(args[run - 1]).name != "streamlit":
        return False
    app = Path(process.cwd()) / args[run + 1]
    env_port = process.environ().get("MINIGPT_CHAT_PORT", "8091")
    return app.resolve() == ROOT / "app.py" and env_port == str(backend_port)


def ui_ready():
    conn = http.client.HTTPConnection(HOST, UI_PORT, timeout=1)
    try:
        conn.request("GET", "/_stcore/health")
        response = conn.getresponse()
        return response.status == 200 and response.read(128).strip() == b"ok"
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


def stop(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def start_or_reuse(stack, label, port, command, matches, ready, log_dir):
    existing = listener(port)
    if existing and not matches(existing):
        raise RuntimeError(f"Port {port} already serves another application. Free it before starting {label}.")
    if existing:
        process = existing
    else:
        log = log_dir / f"{label}.log"
        with log.open("ab") as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=stream, start_new_session=True)
        stack.callback(stop, child)
        process = psutil.Process(child.pid)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            break
        if ready():
            owner = listener(port)
            if owner and owner.pid == process.pid:
                print(f"{'Reusing' if existing else 'Started'} {label} on {HOST}:{port}", flush=True)
                return process, not existing
        time.sleep(0.25)
    raise RuntimeError(f"{label} failed to become ready on port {port}. See {log_dir}/{label}.log.")


def run():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model = ROOT / "models" / manifest["filename"]
    binary = TOOLS / VERSION / "llama-server"
    if not binary.is_file():
        raise RuntimeError("Install the runtime with .venv/bin/python scripts/setup_llama.py")
    if not model.is_file():
        raise RuntimeError("Download and verify the model with .venv/bin/python scripts/download_chat_model.py")
    backend_port = local_chat.port()
    if not 1 <= backend_port <= 65535 or backend_port == UI_PORT:
        raise ValueError("MINIGPT_CHAT_PORT must be a valid TCP port different from the UI port 8501.")
    backend = [
        str(binary), "-m", str(model), "--host", HOST, "--port", str(backend_port),
        "--alias", local_chat.MODEL_ID, "--ctx-size", str(local_chat.CONTEXT_TOKENS),
        "--parallel", "1", "--gpu-layers", "all", "--jinja", "--reasoning", "off",
        "--chat-template-kwargs", '{"enable_thinking":false}', "--offline", "--no-webui",
        "--verbosity", "2",
    ]
    ui = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
        "--server.address", HOST, "--server.port", str(UI_PORT),
        "--server.headless", "true", "--browser.gatherUsageStats", "false",
    ]
    log_dir = ROOT / "runs" / "local-chat"
    log_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as services:
        with (log_dir / ".startup.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # Hash before loading a new copy. A reused live server already maps its weights.
            if listener(backend_port) is None:
                verify(model, manifest)
            infer, own_infer = start_or_reuse(
                services, "llama-server", backend_port, backend,
                lambda process: process.cmdline() == backend, local_chat.ready, log_dir,
            )
            web, own_web = start_or_reuse(
                services, "streamlit", UI_PORT, ui,
                lambda process: same_ui(process, backend_port), ui_ready, log_dir,
            )
        print(f"Open http://{HOST}:{UI_PORT}", flush=True)
        if own_infer or own_web:
            print("Press Ctrl-C to stop services started by this launcher.", flush=True)
            while all(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in (infer, web)):
                time.sleep(1)
            raise RuntimeError(f"A chat service exited. See logs under {log_dir}.")


def interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, psutil.Error) as exc:
        raise SystemExit(str(exc))

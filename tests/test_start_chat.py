from contextlib import ExitStack
import socket
import sys
from types import SimpleNamespace

import pytest

from scripts import start_chat as launcher


def test_streamlit_reuse_matches_checkout_and_backend_port(tmp_path):
    def process(cwd, port, args=None):
        return SimpleNamespace(
            cmdline=lambda: args or ["python", "-m", "streamlit", "run", "app.py"],
            cwd=lambda: str(cwd),
            environ=lambda: {"MINIGPT_CHAT_PORT": port},
        )

    assert launcher.same_ui(process(launcher.ROOT, "8091"), 8091)
    assert not launcher.same_ui(process(tmp_path, "8091"), 8091)
    assert not launcher.same_ui(process(launcher.ROOT, "8092"), 8091)
    assert not launcher.same_ui(process(launcher.ROOT, "8091", ["python", "app.py"]), 8091)


def test_listener_ownership_and_shutdown(tmp_path):
    # A tiny loopback HTTP process exercises real startup/reuse/cleanup without model files.
    with socket.socket() as reserved:
        reserved.bind((launcher.HOST, 0))
        port = reserved.getsockname()[1]
    command = [sys.executable, "-m", "http.server", str(port), "--bind", launcher.HOST]

    def ready():
        try:
            with socket.create_connection((launcher.HOST, port), timeout=1):
                return True
        except ConnectionRefusedError:
            return False

    def same(process):
        return process.cmdline() == command

    with ExitStack() as owner:
        process, created = launcher.start_or_reuse(owner, "test", port, command, same, ready, tmp_path)
        assert created and process.is_running()
        with ExitStack() as second:
            reused, created = launcher.start_or_reuse(second, "test", port, command, same, ready, tmp_path)
            assert reused.pid == process.pid and not created
            with pytest.raises(RuntimeError, match="another application"):
                launcher.start_or_reuse(second, "test", port, command, lambda _: False, ready, tmp_path)
        assert process.is_running()
    assert launcher.listener(port) is None

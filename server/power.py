"""System power manager: one web click starts/stops the whole two-device
system -- the Pi streamer (via SSH) and the local analysis pipeline.

The dashboard process itself stays featherweight when off: no ORT session,
no GPU memory, just this manager waiting for a click.
"""

from __future__ import annotations

import os
import subprocess
import threading

from server.pipeline import StreamingPipeline

PI_USER = os.environ.get("PI_USER", "saltz")
PI_HOST = os.environ.get("PI_HOST", "100.93.84.79")
PI_KEY = os.environ.get("PI_KEY", r"C:\Users\ZHENG QIAORI\.ssh\id_rsa")
PI_DIR = os.environ.get("PI_DIR", "edgefit")

_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=8", "-i", PI_KEY, f"{PI_USER}@{PI_HOST}"]

# single fixed production profile; keepalive keeps the dashboard alive in
# event mode, mode is hot-switchable from the web page.
# NOTE the [.] trick in pgrep/pkill patterns: a plain "edge_main" pattern
# also matches THIS ssh command's own remote shell cmdline, which can kill
# the shell mid-pkill and leak the real process (observed once).
_EDGE_START = (
    f"cd ~/{PI_DIR} && (nohup .venv/bin/python -u -m edge.edge_main "
    f"--mode event --source camera --bind tcp://0.0.0.0:5555 --keepalive 30 "
    f"> edge.log 2>&1 &) && sleep 2 && "
    f"(pgrep -f 'edge[.]edge_main' >/dev/null && echo RUNNING || echo FAILED)")
_EDGE_STOP = ("pkill -f 'edge[.]edge_main'; sleep 1; "
              "pgrep -f 'edge[.]edge_main' >/dev/null && echo STILL || echo STOPPED")


def _ssh(cmd: str, timeout: int = 20) -> str:
    """Run a command on the Pi; returns tail of stdout (or error text)."""
    try:
        r = subprocess.run(_SSH + [cmd], capture_output=True, text=True,
                           timeout=timeout)
        return (r.stdout or r.stderr).strip()[-200:]
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"SSH_ERROR: {e}"


class PowerManager:
    def __init__(self, connect: str, model: str = "models/yolo11n-pose.onnx",
                 backend: str = "dml") -> None:
        self.connect, self.model, self.backend = connect, model, backend
        self._lock = threading.Lock()
        self._pipe: StreamingPipeline | None = None

    # ---------------------------------------------------------------- api --

    def pipeline(self) -> StreamingPipeline | None:
        with self._lock:
            return self._pipe

    def state(self) -> dict:
        pipe = self.pipeline()
        return {
            "power": "on" if pipe is not None else "off",
            "pipeline_alive": bool(pipe and pipe.is_alive()),
            "pipeline_error": str(pipe.error) if pipe and pipe.error else None,
        }

    def on(self) -> dict:
        with self._lock:
            if self._pipe is not None and self._pipe.is_alive():
                return {"power": "on", "detail": "already on"}
        out = _ssh(_EDGE_START)
        if "RUNNING" not in out:
            return {"power": "off", "detail": f"edge failed to start: {out}"}
        pipe = StreamingPipeline(self.connect, self.model, self.backend,
                                 stats_every=1.0)
        pipe.start()
        with self._lock:
            self._pipe = pipe
        return {"power": "on", "detail": "edge RUNNING, pipeline started"}

    def off(self) -> dict:
        with self._lock:
            pipe, self._pipe = self._pipe, None
        if pipe is not None:
            pipe.stop()
            pipe.join(timeout=5)   # frees the ORT session / GPU memory
        out = _ssh(_EDGE_STOP)
        return {"power": "off",
                "detail": f"pipeline stopped, edge {out}"}

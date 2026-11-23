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
from server.session import TrainingSession

PI_USER = os.environ.get("PI_USER", "saltz")
PI_HOST = os.environ.get("PI_HOST", "100.93.84.79")
PI_KEY = os.environ.get("PI_KEY", r"C:\Users\ZHENG QIAORI\.ssh\id_rsa")
PI_DIR = os.environ.get("PI_DIR", "edgefit")

_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=8", "-i", PI_KEY, f"{PI_USER}@{PI_HOST}"]

# Pre-kill runs in its OWN ssh call: the start command's own remote cmdline
# literally contains "edge.edge_main" (the nohup payload), so a pkill in the
# same command would kill its own shell mid-flight (rc=255, silent). The
# bracketed pattern below matches real streamers but not its own cmdline.
_EDGE_PREKILL = "pkill -f 'edge[.]edge_main'; sleep 0.5; echo K"
# single fixed production profile; keepalive keeps the dashboard alive in
# event mode, mode is hot-switchable from the web page. The pidfile makes
# the RUNNING verdict OUR process: pgrep alone would happily report someone
# else's streamer that grabbed the port first (S2).
_EDGE_START = (
    f"cd ~/{PI_DIR} && (nohup .venv/bin/python -u -m edge.edge_main "
    f"--mode event --source camera --bind tcp://0.0.0.0:5555 --keepalive 30 "
    f"> edge.log 2>&1 & echo $! > edge.pid) && sleep 2 && "
    f"(kill -0 $(cat edge.pid) 2>/dev/null && echo RUNNING || echo FAILED)")
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
        self._pipe_lock = threading.Lock()   # guards _pipe ONLY (fast; the
        # event loop takes it in /api/status and the WS loop -- it must never
        # be held across SSH)
        self._op_lock = threading.Lock()     # serializes on/off sequences
        self._pipe: StreamingPipeline | None = None
        self._zombie: StreamingPipeline | None = None
        # survives power cycles: a training plan outlives a stream pause
        self.session = TrainingSession()

    # ---------------------------------------------------------------- api --

    def pipeline(self) -> StreamingPipeline | None:
        with self._pipe_lock:
            return self._pipe

    def state(self) -> dict:
        pipe = self.pipeline()
        return {
            "power": "on" if pipe is not None else "off",
            "pipeline_alive": bool(pipe and pipe.is_alive()),
            "pipeline_error": str(pipe.error) if pipe and pipe.error else None,
        }

    def on(self) -> dict:
        with self._op_lock:
            pipe = self.pipeline()
            if pipe is not None and pipe.is_alive():
                return {"power": "on", "detail": "already on"}
            if pipe is not None or self._zombie is not None:
                # reap a zombie left by a timed-out off() (S1): two live
                # pipelines would both subscribe the stream and double-write
                # any recording
                z = self._zombie or pipe
                print("[power] reaping leftover pipeline thread ...", flush=True)
                z.stop()
                z.join(timeout=8)
                if z.is_alive():
                    return {"power": "off",
                            "detail": "previous pipeline thread stuck; retry in a moment"}
                self._zombie = None
            # slow SSH sequence runs WITHOUT _pipe_lock: readers on the event
            # loop must stay responsive during boot (B4)
            _ssh(_EDGE_PREKILL)
            out = _ssh(_EDGE_START)
            if "RUNNING" not in out:
                return {"power": "off", "detail": f"edge failed to start: {out}"}
            new_pipe = StreamingPipeline(self.connect, self.model, self.backend,
                                         stats_every=1.0, session=self.session)
            if getattr(self, "record_path", ""):
                new_pipe.record_to(self.record_path)
            new_pipe.start()
            with self._pipe_lock:
                self._pipe = new_pipe
            return {"power": "on", "detail": "edge RUNNING, pipeline started"}

    def reset_stats(self) -> dict:
        pipe = self.pipeline()
        if pipe is None or not pipe.is_alive():
            return {"detail": "system is powered off"}
        pipe.request_reset()
        return {"detail": "reset requested"}

    def off(self) -> dict:
        with self._op_lock:
            with self._pipe_lock:
                pipe, self._pipe = self._pipe, None
            if pipe is not None:
                pipe.stop()
                pipe.join(timeout=5)   # frees the ORT session / GPU memory
                if pipe.is_alive():
                    # keep the reference so on() can reap it (S1)
                    self._zombie = pipe
                    print("[power] WARNING: pipeline thread did not exit in "
                          "5s (still holds ORT session)", flush=True)
            out = _ssh(_EDGE_STOP)
            return {"power": "off", "detail": f"pipeline stopped, edge {out}"}

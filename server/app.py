"""Edge server B web dashboard: FastAPI + WebSocket + POWER MANAGEMENT.

The dashboard process itself is featherweight and always-on-able: it holds
no model and no GPU memory while the system is OFF.  The header power
button (or /api/power/*, or scripts/*_system.bat) turns the whole two-device
system on/off -- Pi streamer via SSH + the local analysis pipeline thread.

    .venv/Scripts/python server/app.py                 # defaults
    EDGE_CONNECT=tcp://100.93.84.79:5555 ...           # env overrides
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.protocol import VALID_MODES  # noqa: E402
from server.history import HistoryStore  # noqa: E402
from server.power import PowerManager  # noqa: E402

CONNECT = os.environ.get("EDGE_CONNECT", "tcp://100.93.84.79:5555")
MODEL = os.environ.get("EDGE_MODEL", "models/yolo11n-pose.onnx")
BACKEND = os.environ.get("EDGE_BACKEND", "dml")
RECORD = os.environ.get("EDGE_RECORD", "")

power = PowerManager(CONNECT, MODEL, BACKEND)
store = HistoryStore(os.environ.get("EDGE_DB", "data/edgefit.db"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if RECORD:
        Path(RECORD).parent.mkdir(parents=True, exist_ok=True)
        power.record_path = RECORD
    yield   # nothing runs until the user powers the system on


app = FastAPI(title="EdgeFit Coach", lifespan=lifespan)

# vendored static assets (Chart.js) -- no CDN: the recording machine may be
# offline and dead charts would sink the demo video
_VENDOR = Path(__file__).resolve().parents[1] / "dashboard" / "vendor"
app.mount("/vendor", StaticFiles(directory=str(_VENDOR)), name="vendor")


@app.get("/")
async def index():
    return FileResponse(Path(__file__).resolve().parents[1] / "dashboard" / "index.html")


@app.post("/api/power/on")
async def power_on():
    return await asyncio.get_running_loop().run_in_executor(None, power.on)


@app.post("/api/power/off")
async def power_off():
    return await asyncio.get_running_loop().run_in_executor(None, power.off)


@app.post("/api/session/start")
async def session_start(sets: int = 3, reps: int = 8, rest: float = 60.0):
    try:
        pipe = power.pipeline()
        if pipe is None or not pipe.is_alive():
            r = await asyncio.get_running_loop().run_in_executor(None, power.on)
            if r["power"] != "on":
                raise HTTPException(502, f"cannot power on: {r['detail']}")
            pipe = power.pipeline()
        total = pipe.status().reps if pipe else 0
        power.session.start(total, sets=sets, reps_per_set=reps, rest_sec=rest)
        store.begin(sets_planned=sets, reps_per_set=reps, rest_sec=rest)
        return {"detail": "session started", **power.session.snapshot()}
    except ValueError as e:
        raise HTTPException(400, str(e))


def _finish_open_session(pipe) -> None:
    if store.current_id is None:
        return
    store.finish(pipe.status() if pipe else {},
                 power.session.snapshot())


@app.post("/api/session/stop")
async def session_stop():
    pipe = power.pipeline()
    power.session.stop()
    _finish_open_session(pipe)
    return {"detail": "session stopped"}


@app.post("/api/session/skip_rest")
async def session_skip_rest():
    power.session.skip_rest()
    return {"detail": "ok"}


@app.post("/api/stats/reset")
async def stats_reset():
    power.session.stop()
    return await asyncio.get_running_loop().run_in_executor(
        None, power.reset_stats)


@app.get("/api/power")
async def power_state():
    return power.state()


@app.post("/api/mode/{mode}")
async def set_mode(mode: str):
    if mode not in VALID_MODES:
        raise HTTPException(400, f"unknown mode {mode!r}")
    pipe = power.pipeline()
    if pipe is None or not pipe.is_alive():
        raise HTTPException(409, "system is powered off")
    reply = await asyncio.get_running_loop().run_in_executor(
        None, pipe.send_control, {"cmd": "set_mode", "arg": {"mode": mode}})
    if reply is None:
        raise HTTPException(504, "edge control channel timeout")
    if str(reply.get("result", "")).startswith("error"):
        raise HTTPException(409, reply["result"])
    return {"mode": reply.get("mode"), "result": reply.get("result")}


@app.post("/api/rate_limit/{kbps}")
async def set_rate_limit(kbps: float):
    """Simulate a constrained uplink ON DEVICE A (closed-loop seed)."""
    if kbps < 0:
        raise HTTPException(400, "kbps must be >= 0")
    pipe = power.pipeline()
    if pipe is None or not pipe.is_alive():
        raise HTTPException(409, "system is powered off")
    reply = await asyncio.get_running_loop().run_in_executor(
        None, pipe.send_control, {"cmd": "set_rate_limit",
                                  "arg": {"kbps": kbps}})
    if reply is None:
        raise HTTPException(504, "edge control channel timeout")
    return {"result": reply.get("result")}


@app.get("/api/status")
async def status():
    st = power.state()
    pipe = power.pipeline()
    s = pipe.status() if pipe else None
    return {"power": st, "status": s, "events": pipe.recent_events() if pipe else []}


@app.get("/api/history")
async def history(limit: int = 20):
    return store.recent_sessions(limit)


@app.get("/api/history/reps")
async def history_reps(limit: int = 300):
    return store.recent_reps(limit)


@app.get("/api/history/last")
async def history_last():
    return store.last_session("curl")


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            pipe = power.pipeline()
            if pipe is None or not pipe.is_alive():
                await ws.send_text(json.dumps({"power": "off"}))
                await asyncio.sleep(1.0)
                continue
            meta = pipe.status()
            payload = pipe.latest_jpeg()
            events = pipe.recent_events()
            snap = power.session.snapshot()
            if store.current_id is not None:
                store.feed(events)
                if snap["phase"] == "done":
                    _finish_open_session(pipe)
            await ws.send_text(json.dumps({
                "power": "on",
                "session": power.session.snapshot(),
                "mode": meta.mode,
                "reps": meta.reps, "full": meta.full_reps,
                "half": meta.half_reps, "suppressed": meta.suppressed,
                "phase": meta.phase, "warnings": meta.warnings,
                "last_frame_age": round(meta.last_frame_age, 2),
                "rx_fps": round(meta.rx_fps, 1),
                "a_eff_fps": round(meta.a_eff_fps, 1),
                "kbps": round(meta.kbps, 1),
                "loss": meta.loss,
                "e2e_ms": round(meta.e2e_p50_ms, 0) or None,
                "infer_ms": round(meta.infer_p50_ms, 1),
                "jpeg_q": meta.jpeg_q,
                "motion": round(meta.motion, 2),
                "elbow_l": round(meta.elbow_l, 1),
                "elbow_r": round(meta.elbow_r, 1),
                "view_ok": meta.view_ok,
                "running": meta.running,
                "events": events[-5:],
            }))
            if payload:
                await ws.send_bytes(payload)
            await asyncio.sleep(0.05)   # ~20 Hz UI refresh cap
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

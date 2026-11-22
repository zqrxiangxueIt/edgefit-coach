"""Edge server B web dashboard: FastAPI + WebSocket.

Serves the single-page dashboard and pushes the shared StreamingPipeline's
annotated frames (binary JPEG) and metadata (JSON) over one WebSocket,
using the dual-message pattern.  Mode switching relays to the edge device
over the ZMQ control channel.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.protocol import VALID_MODES  # noqa: E402
from server.pipeline import StreamingPipeline  # noqa: E402

CONNECT = os.environ.get("EDGE_CONNECT", "tcp://100.93.84.79:5555")
MODEL = os.environ.get("EDGE_MODEL", "models/yolo11n-pose.onnx")
BACKEND = os.environ.get("EDGE_BACKEND", "dml")
RECORD = os.environ.get("EDGE_RECORD", "")

pipe = StreamingPipeline(CONNECT, MODEL, BACKEND, stats_every=1.0)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if RECORD:
        Path(RECORD).parent.mkdir(parents=True, exist_ok=True)
        pipe.record_to(RECORD)
    pipe.start()
    yield
    pipe.stop()
    pipe.join(timeout=5)


app = FastAPI(title="EdgeFit Coach", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).resolve().parents[1] / "dashboard" / "index.html")


@app.post("/api/mode/{mode}")
async def set_mode(mode: str):
    if mode not in VALID_MODES:
        raise HTTPException(400, f"unknown mode {mode!r}")
    reply = await asyncio.get_running_loop().run_in_executor(
        None, pipe.send_control, {"cmd": "set_mode", "arg": {"mode": mode}})
    if reply is None:
        raise HTTPException(504, "edge control channel timeout")
    if str(reply.get("result", "")).startswith("error"):
        raise HTTPException(409, reply["result"])
    return {"mode": reply.get("mode"), "result": reply.get("result")}


@app.get("/api/status")
async def status():
    s = pipe.status()
    return {"status": s, "events": pipe.recent_events(),
            "pipeline_error": str(pipe.error) if pipe.error else None}


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            meta = pipe.status()
            payload = pipe.latest_jpeg()
            await ws.send_text(json.dumps({
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
                "running": meta.running,
                "events": pipe.recent_events()[-5:],
            }))
            if payload:
                await ws.send_bytes(payload)
            await asyncio.sleep(0.05)   # ~20 Hz UI refresh cap
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

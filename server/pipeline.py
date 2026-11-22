"""Shared streaming pipeline: ZMQ SUB -> pose -> counter -> annotated output.

One loop, three frontends:
  * server/receiver.py  (console stats / mp4 recording)
  * server/app.py       (web dashboard over WebSocket)
  * tests               (loopback integration)

The pipeline runs in its own thread and exposes a thread-safe snapshot
(status()) plus the latest annotated JPEG for pull-style consumers.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import zmq

from common.protocol import decode_frame
from server.analysis.curl_counter import CurlCounter
from server.analysis.filters import KeypointSmoother
from server.analysis.visualize import draw
from server.inference.pose_engine import PoseEngine


def estimate_offset(control_endpoint: str, n: int = 5) -> tuple[float, float] | None:
    """NTP-style clock offset (A-clock minus B-clock) over the ZMQ control
    channel.  Returns (offset_seconds, min_rtt_ms), best sample by RTT;
    None when the channel is dead (caller must NOT pretend offset is 0 --
    on the real pair A-B is ~-779ms, so a fake 0 poisons every e2e number).
    """
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, 2000)
    s.setsockopt(zmq.SNDTIMEO, 2000)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(control_endpoint)
    best: tuple[float, float] | None = None
    try:
        for _ in range(n):
            t0 = time.time()
            try:
                s.send_json({"cmd": "ping", "t0": t0})
                rep = s.recv_json()
            except zmq.ZMQError:
                break
            t3 = time.time()
            if rep.get("echo", {}).get("cmd") == "ping":
                rtt, off = (t3 - t0) * 1000, rep["t_reply"] - (t0 + t3) / 2
                if best is None or rtt < best[1]:
                    best = (off, rtt)
            time.sleep(0.05)
    finally:
        s.close()
    return best


@dataclass
class PipelineStatus:
    """Immutable-per-snapshot view of the pipeline for UI/console."""
    running: bool = False
    mode: str = "?"            # last header mode seen from A
    reps: int = 0
    full_reps: int = 0
    half_reps: int = 0
    suppressed: int = 0
    phase: str = "idle"
    warnings: list = field(default_factory=list)
    # rolling metrics (per stats window)
    rx_fps: float = 0.0
    kbps: float = 0.0
    loss: int = 0
    a_eff_fps: float = 0.0
    infer_p50_ms: float = 0.0
    infer_p95_ms: float = 0.0
    e2e_p50_ms: float = 0.0    # 0 == unavailable (no clock offset)
    last_frame_age: float = 1e9  # s since last frame; distinguishes "thread
                                 # alive" from "link delivering" (N1)
    offset_ms: float = 0.0
    frames: int = 0
    jpeg_q: int = 0
    motion: float = 0.0


class StreamingPipeline(threading.Thread):
    def __init__(self, connect: str, model: str = "models/yolo11n-pose.onnx",
                 backend: str = "dml", clock_offset: float | None = None,
                 stats_every: float = 2.0, session=None) -> None:
        super().__init__(daemon=True)
        self.connect = connect
        self.model = model
        self.backend = backend
        self.clock_offset = clock_offset
        self.stats_every = stats_every
        self.session = session            # TrainingSession or None (survives
        self._reset_req = threading.Event()  # power cycles via PowerManager)
        self._last_enc_t = 0.0            # lazy JPEG re-encode throttle
        self._multi_t = 0.0               # multi-person warning throttle
        self._lock = threading.Lock()
        self._status = PipelineStatus()
        self._latest_jpeg: bytes | None = None
        self._events: deque = deque(maxlen=50)   # recent RepEvents (dicts)
        self._stop_evt = threading.Event()  # NOT _stop: shadows Thread._stop() and breaks join()
        self._record_writer = None
        self._record_path = ""
        self.error: Exception | None = None

    # ---------------------------------------------------------------- api --

    def status(self) -> PipelineStatus:
        with self._lock:
            return self._status

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def recent_events(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def record_to(self, path: str) -> None:
        """Start recording annotated frames to an mp4 (call before start)."""
        self._record_path = path

    def stop(self) -> None:
        self._stop_evt.set()

    def request_reset(self) -> None:
        """Reset counter/smoother/session at the next frame (free mode)."""
        self._reset_req.set()

    def control_endpoint(self) -> str:
        host = self.connect.split("//")[1].rsplit(":", 1)[0]
        port = int(self.connect.rsplit(":", 1)[1]) + 1
        return f"tcp://{host}:{port}"

    def send_control(self, msg: dict, timeout_ms: int = 2000) -> dict | None:
        """One-shot REQ to the edge control channel; None on timeout."""
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.RCVTIMEO, timeout_ms)
        s.setsockopt(zmq.SNDTIMEO, timeout_ms)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(self.control_endpoint())
        try:
            s.send_json(msg)
            return s.recv_json()
        except zmq.ZMQError:
            return None
        finally:
            s.close()

    # ------------------------------------------------------------ the loop --

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:  # surface to frontends instead of dying silently
            self.error = e
            raise
        finally:
            # always release resources, including the exception path (S2):
            # a leaked SUB socket / un-finalized mp4 (no moov atom) otherwise
            w = getattr(self, "_writer", None)
            if w is not None:
                w.release()
                self._writer = None
            sk = getattr(self, "_sock", None)
            if sk is not None:
                sk.close(0)
                self._sock = None
            with self._lock:
                self._status.running = False
            print("[pipeline] stopped", flush=True)

    def _run(self) -> None:
        offset = self.clock_offset
        if offset is None:
            res = estimate_offset(self.control_endpoint())
            if res is None:
                print("[pipeline] WARN: clock-offset estimation failed; e2e n/a",
                      flush=True)
                offset = None
            else:
                offset, rtt = res
                print(f"[pipeline] clock offset A-B = {offset*1000:+.1f} ms "
                      f"(min RTT {rtt:.1f} ms)", flush=True)

        engine = PoseEngine(self.model, backend=self.backend)
        smoother = KeypointSmoother()
        counter = CurlCounter()
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"frame")
        sock.setsockopt(zmq.RCVHWM, 10)
        sock.connect(self.connect)
        self._sock = sock          # released by run()'s finally on ANY exit
        print(f"[pipeline] connecting {self.connect} backend={self.backend}",
              flush=True)

        mlen = 300
        infer_ms: deque = deque(maxlen=mlen)
        e2e_ms: deque = deque(maxlen=mlen)
        capture_iv: deque = deque(maxlen=mlen)
        t_stat = time.time()
        t_begin = time.time()
        n_stat = b_stat = 0
        first_seq = last_seq = None
        last_capture_t = None
        writer = None
        frames_total = 0

        with self._lock:
            self._status.running = True

        while not self._stop_evt.is_set():
            if not sock.poll(500):
                with self._lock:
                    self._status.running = True   # alive, just quiet
                continue
            parts = sock.recv_multipart()
            t_recv = time.time()
            try:
                header, payload = decode_frame(parts)
            except ValueError:
                continue          # one malformed frame must not kill the loop
            frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.perf_counter()
            results = engine.detect(frame)
            infer_ms.append((time.perf_counter() - t0) * 1000)

            annotated = None
            events: list = []
            # real dt for the filters: A-side capture interval, not wall clock
            dt_est = None
            if last_capture_t is not None:
                d = header.ts_capture - last_capture_t
                if 0.001 < d < 1.0:
                    dt_est = d
            if results:
                kps = smoother.update(results[0].keypoints, dt=dt_est)
                events = counter.update(kps)
                annotated = draw(frame, results[0], counter, kps)
            else:
                zeros = np.zeros((17, 3), np.float32)
                smoother.update(zeros, dt=dt_est)
                counter.update(zeros)
                annotated = frame

            # session advances on EVERY frame (wall-clock transitions like
            # countdown->set must run even with nobody in view)
            if self.session is not None:
                self.session.update(counter.stats.reps, events)
            if self._reset_req.is_set():
                self._reset_req.clear()
                smoother = KeypointSmoother()
                counter = CurlCounter()
                events = []
                if self.session is not None:
                    self.session.stop()
            for ev in events:
                with self._lock:
                    self._events.append({
                        "index": ev.index, "side": ev.side, "full": ev.full,
                        "min_angle": round(ev.min_angle, 1),
                        "t_con": round(ev.t_con, 2), "t_ecc": round(ev.t_ecc, 2),
                        "t": round(ev.t - t_begin, 2)})
                print(f"  [rep {ev.index}][{ev.side}] full={ev.full} "
                      f"min_angle={ev.min_angle:.0f}", flush=True)

            # ---- metrics bookkeeping ----
            frames_total += 1
            n_stat += 1
            b_stat += len(payload)
            # consume the A-side NN output (cascade/privacy): a second person
            # in frame breaks the single-person counting assumption
            persons = header.extra.get("persons")
            if persons and len(persons) > 1 and time.time() - self._multi_t > 5.0:
                self._multi_t = time.time()
                print("[pipeline] WARNING: 多人画面，计数可能混淆（单目标假设）",
                      flush=True)
            if first_seq is None:
                first_seq = header.seq
            last_seq = header.seq
            if last_capture_t is not None:
                capture_iv.append(header.ts_capture - last_capture_t)
            last_capture_t = header.ts_capture
            if offset is not None:
                e2e_ms.append((t_recv - header.ts_capture + offset) * 1000)

            # lazy encode: the WS consumer pulls at ~20 Hz, so encoding all
            # ~30 incoming frames/s wastes a third of the encodes
            now_enc = time.time()
            if now_enc - self._last_enc_t >= 0.05:
                self._last_enc_t = now_enc
                ok, buf = cv2.imencode(".jpg", annotated,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()

            if self._record_path:
                if writer is None:
                    est = (1.0 / np.median(capture_iv)
                           if len(capture_iv) >= 5 else 25.0)
                    writer = cv2.VideoWriter(
                        self._record_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        est, (annotated.shape[1], annotated.shape[0]))
                    if not writer.isOpened():
                        print(f"[pipeline] ERROR: cannot open writer for "
                              f"{self._record_path}", flush=True)
                        self._record_path = ""
                        writer = None
                    else:
                        self._writer = writer
                        print(f"[pipeline] recording at {est:.1f} fps "
                              f"-> {self._record_path}", flush=True)
                if writer is not None:
                    writer.write(annotated)

            now = time.time()
            if now - t_stat >= self.stats_every:
                dt = now - t_stat
                got = (last_seq - first_seq + 1
                       if last_seq is not None and first_seq is not None else 0)
                loss = max(0, got - n_stat)
                iv = float(np.mean(capture_iv)) if capture_iv else 0.0
                with self._lock:
                    s = self._status
                    s.mode = header.mode
                    s.reps = counter.stats.reps
                    s.full_reps = counter.stats.full_reps
                    s.half_reps = counter.stats.half_reps
                    s.suppressed = counter.stats.phantom_reps_suppressed
                    s.phase = counter.phase.value
                    s.warnings = list(counter.stats.warnings)[-8:]
                    s.rx_fps = n_stat / dt
                    s.kbps = b_stat / dt / 1024
                    s.loss = loss
                    s.a_eff_fps = 1.0 / iv if iv > 0 else 0.0
                    s.infer_p50_ms = float(np.percentile(infer_ms, 50)) if infer_ms else 0.0
                    s.infer_p95_ms = float(np.percentile(infer_ms, 95)) if infer_ms else 0.0
                    s.e2e_p50_ms = (float(np.percentile(e2e_ms, 50))
                                    if e2e_ms else 0.0)
                    s.offset_ms = (offset or 0.0) * 1000
                    s.frames = frames_total
                    s.jpeg_q = header.jpeg_q
                    s.motion = header.motion
                    s.last_frame_age = time.time() - t_recv
                t_stat, n_stat, b_stat = now, 0, 0
                first_seq = None

            with self._lock:
                self._status.last_frame_age = time.time() - t_recv

        # final refresh so a stopped pipeline reports its true totals (the
        # window tick may have happened before the last reps)
        with self._lock:
            self._status.reps = counter.stats.reps
            self._status.full_reps = counter.stats.full_reps
            self._status.half_reps = counter.stats.half_reps
            self._status.suppressed = counter.stats.phantom_reps_suppressed
            self._status.phase = counter.phase.value
            self._status.warnings = list(counter.stats.warnings)[-8:]

        # normal-path cleanup lives in run()'s finally (S2)

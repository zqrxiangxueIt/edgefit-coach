"""StreamingPipeline integration test: a fake edge (PUB + REP) on localhost
drives the real pipeline thread -- no Pi, no GPU required (CPU backend, plain
gray frames).  Also regression-pins the join() path (round-3 B1: a stop
attribute named _stop used to shadow Thread._stop() and crash every join).
"""

import threading
import time

import cv2
import numpy as np
import pytest
import zmq

from common.protocol import FrameHeader, encode_frame
from server.pipeline import StreamingPipeline

PORT = 5590


def test_pipeline_end_to_end_from_fake_edge():
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{PORT}")
    rep = ctx.socket(zmq.REP)
    rep.bind(f"tcp://127.0.0.1:{PORT + 1}")

    stop = threading.Event()
    seq = 0

    def fake_edge():
        nonlocal seq
        ok, buf = cv2.imencode(".jpg", np.full((480, 640, 3), 60, np.uint8))
        payload = buf.tobytes()
        t0 = time.time()
        while not stop.is_set() and time.time() - t0 < 8:
            if rep.poll(0):                      # serve clock pings / control
                req = rep.recv_json()
                rep.send_json({"t_reply": time.time(), "mode": "raw",
                               "seq": seq, "echo": req})
            seq += 1
            header = FrameHeader(seq=seq, ts_capture=time.time(), mode="raw",
                                 w=640, h=480, jpeg_q=75, a_proc_ms=3.0)
            pub.send_multipart(encode_frame(header, payload))
            time.sleep(1 / 30)

    th = threading.Thread(target=fake_edge, daemon=True)
    th.start()

    pipe = StreamingPipeline(f"tcp://127.0.0.1:{PORT}",
                             model="models/yolo11n-pose.onnx",
                             backend="cpu", stats_every=1.0)
    assert pipe.control_endpoint() == f"tcp://127.0.0.1:{PORT + 1}"
    pipe.start()
    try:
        deadline = time.time() + 6
        while time.time() < deadline:
            s = pipe.status()
            if s.frames > 5 and s.rx_fps > 0:
                break
            time.sleep(0.2)
        s = pipe.status()
        assert s.frames > 5
        assert s.rx_fps > 0
        assert s.last_frame_age < 3.0
        assert pipe.error is None
    finally:
        stop.set()
        pipe.stop()
        pipe.join(timeout=5)          # must not raise (B1 regression pin)
        assert not pipe.is_alive()
        assert pipe.error is None
    th.join(timeout=2)
    pub.close(0)
    rep.close(0)

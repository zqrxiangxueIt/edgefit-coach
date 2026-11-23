"""Task-accuracy vs delivered-bandwidth experiment (batch E, the headline).

For each streaming mode and simulated uplink rate, the video is run through
the REAL A-side preprocessor, a token-bucket decides which frames a link of
that capacity would deliver, and only delivered frames are analysed by the
REAL B-side pose+counter.  The output: counting error per (mode, rate) --
i.e. bandwidth translated into TASK currency ("counting is exact down to
X KB/s"), which is what no filtering-only system report shows.

Fully offline and reproducible (no Pi, no network):

    .venv/Scripts/python scripts/task_accuracy.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.cascade import PersonDetector
from edge.preprocess import (CascadePreprocessor, EventPreprocessor,
                             PrivacyPreprocessor, RawPreprocessor)
from edge.sources import VideoFileSource
from server.analysis.curl_counter import CurlCounter
from server.analysis.filters import KeypointSmoother
from server.inference.pose_engine import PoseEngine

VIDEOS = [  # (path, ground-truth reps: human-verified best-count)
    ("videos/curl_a.mp4", 5),
    ("videos/curl_b.mp4", 7),
]
RATES_KBPS = [0, 2048, 512, 256, 128, 64, 32]   # 0 = unlimited
MODES = ("raw", "event", "cascade", "privacy")


def build_pre(mode: str, det):
    if mode == "raw":
        return RawPreprocessor()
    if mode == "event":
        return EventPreprocessor(keepalive_every=0)
    if mode == "cascade":
        return CascadePreprocessor(det, grace_s=0.8)
    return PrivacyPreprocessor(det)


def run_cell(video: str, gt: int, mode: str, rate: float,
             engine: PoseEngine, det) -> dict:
    src = VideoFileSource(video, realtime=False)   # as fast as possible
    pre = build_pre(mode, det)
    smoother = KeypointSmoother()
    counter = CurlCounter()
    tokens = 0.0 if rate > 0 else float("inf")
    last_ct = 0.0        # CONTENT time of last refill decision: the offline
    # replay runs faster than realtime, so refilling by wall clock would
    # throttle ~3x harder than the same nominal rate on a live link
    n_sent = n_del = 0
    b_del = 0
    i = 0
    fps = src.fps
    zeros = np.zeros((17, 3), np.float32)
    while True:
        item = src.read()
        if item is None:
            break
        frame, _t = item
        i += 1
        t = i / fps
        pf = pre.process(frame, t=t)
        if pf is not None:
            n_sent += 1
            if rate > 0:
                tokens = min(rate * 1024.0,
                             tokens + (t - last_ct) * rate * 1024.0)
            last_ct = t
            if len(pf.jpeg) <= tokens:
                if rate > 0:
                    tokens -= len(pf.jpeg)
                n_del += 1
                b_del += len(pf.jpeg)
                img = cv2.imdecode(np.frombuffer(pf.jpeg, np.uint8),
                                   cv2.IMREAD_COLOR)
                results = engine.detect(img)
                if results:
                    kps = smoother.update(results[0].keypoints, dt=1.0 / fps)
                    counter.update(kps, t=t)
                else:
                    smoother.update(zeros, dt=1.0 / fps)
                    counter.update(zeros, t=t)
    src.close()
    reps = counter.stats.reps
    dur = i / fps
    return {"mode": mode, "rate_kbps": rate, "gt": gt, "reps": reps,
            "error": reps - gt, "abs_err": abs(reps - gt),
            "delivered_fps": round(n_del / dur, 1),
            "delivered_kbps": round(b_del / dur / 1024, 1)}


def main() -> None:
    engine = PoseEngine("models/yolo11n-pose.onnx", backend="dml")
    det = PersonDetector("models/detect.tflite")
    rows = []
    for video, gt in VIDEOS:
        print(f"\n### {Path(video).name} (ground truth {gt} reps)")
        print("| mode | rate KB/s | delivered | reps | error |")
        print("|---|---|---|---|---|")
        for mode in MODES:
            for rate in RATES_KBPS:
                r = run_cell(video, gt, mode, rate, engine, det)
                rows.append({"video": Path(video).name, **r})
                print(f"| {mode} | {rate if rate else '∞'} | "
                      f"{r['delivered_fps']}fps/{r['delivered_kbps']}KB/s | "
                      f"{r['reps']} | {r['error']:+d} |", flush=True)
    out = Path("docs/task_accuracy_data.json")
    out.write_text(__import__("json").dumps(rows, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\ndata -> {out}")


if __name__ == "__main__":
    main()

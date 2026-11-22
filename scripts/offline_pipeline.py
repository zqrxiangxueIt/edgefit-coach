"""M0 acceptance demo: video file -> pose engine -> One-Euro -> curl counter.

Runs entirely on device B (the PC) without the network, proving the analysis
stack before the transmission link exists.  Writes an annotated preview video
so the counting can be eyeballed.

Usage:
    .venv/Scripts/python scripts/offline_pipeline.py videos/curl_a.mp4 \
        --model models/yolo11n-pose.onnx --backend dml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.analysis.curl_counter import CurlCounter
from server.analysis.filters import KeypointSmoother
from server.analysis.visualize import draw
from server.inference.pose_engine import PoseEngine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="models/yolo11n-pose.onnx")
    ap.add_argument("--backend", default="dml", choices=["dml", "openvino", "cpu"])
    ap.add_argument("--out", default="runs/offline_preview.mp4")
    ap.add_argument("--show-every", type=int, default=1, help="annotate every Nth frame")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    show_every = max(1, int(args.show_every))     # guard: N=0 would ZeroDivision
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"cannot create output directory for {args.out}: {e}")
    # only 1/N frames reach the writer -> scale fps so playback speed is true
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / show_every, (int(cap.get(3)), int(cap.get(4))))
    if not writer.isOpened():
        sys.exit(f"cannot open video writer for {args.out}")

    engine = PoseEngine(args.model, backend=args.backend)
    smoother = KeypointSmoother()
    counter = CurlCounter()

    t0 = time.perf_counter()
    infer_ms = []
    events = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        results = engine.detect(frame)
        infer_ms.append((time.perf_counter() - t1) * 1000)

        if results:
            kps = smoother.update(results[0].keypoints)
            for ev in counter.update(kps):
                events.append(ev)
                print(f"  [rep {ev.index}][{ev.side}] full={ev.full} min_angle={ev.min_angle:.0f}")
        else:
            kps = smoother.update(np.zeros((17, 3), np.float32))
            counter.update(kps)

        if i % show_every == 0:
            if results:
                frame = draw(frame, results[0], counter, kps)
            writer.write(frame)
        i += 1

    dt = time.perf_counter() - t0
    print(f"\nframes={i} wall={dt:.1f}s ({i/dt:.1f} fps end-to-end on {args.backend})")
    if infer_ms:
        print(f"inference: mean {np.mean(infer_ms):.1f} ms  p95 {np.percentile(infer_ms, 95):.1f} ms")
    print(f"RESULT: {counter.stats.reps} reps "
          f"({counter.stats.full_reps} full / {counter.stats.half_reps} half)")
    for w in counter.stats.warnings:
        print(f"  warn: {w}")
    print(f"annotated preview -> {args.out}")


if __name__ == "__main__":
    main()

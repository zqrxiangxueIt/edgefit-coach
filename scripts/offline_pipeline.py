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
from server.analysis.geometry import LEFT_ARM, RIGHT_ARM, joint_angle
from server.inference.pose_engine import PoseEngine

SKELETON = [  # COCO-17 limbs
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (5, 6),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw(frame: np.ndarray, res, counter: CurlCounter, smoother_out: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = res.box.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for a, b in SKELETON:
        if res.keypoints[a, 2] > 0.3 and res.keypoints[b, 2] > 0.3:
            p, q = res.keypoints[a, :2].astype(int), res.keypoints[b, :2].astype(int)
            cv2.line(frame, tuple(p), tuple(q), (255, 128, 0), 2)
    for x, y, c in smoother_out:
        if c > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    la = joint_angle(smoother_out, *LEFT_ARM)
    ra = joint_angle(smoother_out, *RIGHT_ARM)
    cv2.putText(frame, f"elbow L {la[0]:.0f}deg({la[1]:.2f})  R {ra[0]:.0f}deg({ra[1]:.2f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"REPS {counter.stats.reps} (half {counter.stats.half_reps})  {counter.phase.value}",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


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
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (int(cap.get(3)), int(cap.get(4))))

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
            ev = counter.update(kps)
            if ev:
                events.append(ev)
                print(f"  [rep {ev.index}] full={ev.full} min_angle={ev.min_angle:.0f}")
        else:
            kps = smoother.update(np.zeros((17, 3), np.float32))
            counter.update(kps)

        if i % args.show_every == 0:
            if results:
                frame = draw(frame, results[0], counter, kps)
            writer.write(frame)
        i += 1

    dt = time.perf_counter() - t0
    print(f"\nframes={i} wall={dt:.1f}s ({i/dt:.1f} fps end-to-end on {args.backend})")
    print(f"inference: mean {np.mean(infer_ms):.1f} ms  p95 {np.percentile(infer_ms, 95):.1f} ms")
    print(f"RESULT: {counter.stats.reps} reps "
          f"({counter.stats.full_reps} full / {counter.stats.half_reps} half)")
    for w in counter.stats.warnings:
        print(f"  warn: {w}")
    print(f"annotated preview -> {args.out}")


if __name__ == "__main__":
    main()

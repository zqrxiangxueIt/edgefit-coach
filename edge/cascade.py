"""On-device person detection for cascade/privacy modes (device A).

Runs the classic SSD-MobileNet-v1 int8 COCO model (300x300) through LiteRT
(`ai-edge_litert`, the official tflite-runtime successor with py3.12
aarch64 wheels).  Class 0 = person.  Typical Pi 5 latency: tens of ms --
measured per frame and shipped in `extra.det_ms`.

Model: Arm-Examples/ML-zoo ssd_mobilenet_v1 tflite_int8 (Apache-2.0); the
original storage.googleapis.com tflite URL is permanently AccessDenied
(verified 2026-11), see scripts notes.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

PERSON_CLASS = 0


class PersonDetector:
    def __init__(self, model_path: str = "models/detect.tflite",
                 score_thr: float = 0.55, num_threads: int = 2) -> None:
        from ai_edge_litert.interpreter import Interpreter  # lazy: Pi-only dep
        self.it = Interpreter(model_path=model_path, num_threads=num_threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = {d["name"]: d["index"] for d in self.it.get_output_details()}
        self.score_thr = score_thr
        self.scale, self.zp = self.inp["quantization"]
        if not self.scale:
            raise ValueError("model input is not quantized; this class expects the int8 export")
        # scale == 1/127.5 -> float input normalized to [-1, 1]
        self._norm = abs(self.scale - 1.0 / 127.5) < 1e-5

    def detect(self, frame_bgr: np.ndarray) -> list[list[float]]:
        """Return person boxes [[x1, y1, x2, y2, score], ...] in source pixels."""
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (300, 300),
                                      interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB).astype(np.float32)
        if self._norm:
            rgb = rgb / 127.5 - 1.0
        q = np.clip(np.round(rgb / self.scale) + self.zp, -128, 127).astype(np.int8)
        self.it.set_tensor(self.inp["index"], q[None])
        self.it.invoke()

        boxes = self.it.get_tensor(self.outs["TFLite_Detection_PostProcess:01"])[0]
        classes = self.it.get_tensor(self.outs["TFLite_Detection_PostProcess:02"])[0]
        scores = self.it.get_tensor(self.outs["TFLite_Detection_PostProcess:03"])[0]
        n = int(self.it.get_tensor(self.outs["TFLite_Detection_PostProcess:04"])[0])

        out = []
        for i in range(n):
            if int(classes[i]) != PERSON_CLASS or scores[i] < self.score_thr:
                continue
            y1, x1, y2, x2 = boxes[i]
            out.append([x1 * w, y1 * h, x2 * w, y2 * h, float(scores[i])])
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out

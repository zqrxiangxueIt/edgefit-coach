"""A-side preliminary computation (the "初步计算" the assignment asks for).

All processors share one interface so the streaming mode is a hot-swappable
strategy:

    process(frame) -> ProcessedFrame | None     (None = do not transmit)

M1 ships RawPreprocessor (sampling + downscale + JPEG).  M2 adds
EventPreprocessor (frame-diff gating + adaptive quality), M4 adds
CascadePreprocessor (on-device NN person filter) and PrivacyPreprocessor
(person-ROI-only masking).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ProcessedFrame:
    jpeg: bytes
    w: int                      # source resolution (before downscale)
    h: int
    jpeg_q: int
    motion: float = 0.0         # normalized 0..1, mode-specific
    a_proc_ms: float = 0.0      # A-side compute time for this frame
    extra: dict = field(default_factory=dict)


class RawPreprocessor:
    """Baseline: uniform frame sampling, downscale, fixed-quality JPEG."""

    def __init__(self, sample_every: int = 1, max_width: int = 960,
                 jpeg_q: int = 75) -> None:
        self.sample_every = max(1, int(sample_every))
        self.max_width = max_width
        self.jpeg_q = jpeg_q
        self._i = 0

    def process(self, frame: np.ndarray) -> ProcessedFrame | None:
        self._i += 1
        if (self._i - 1) % self.sample_every != 0:
            return None
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        if w > self.max_width:
            nh = round(h * self.max_width / w)
            frame = cv2.resize(frame, (self.max_width, nh),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
        if not ok:
            return None
        return ProcessedFrame(jpeg=buf.tobytes(), w=w, h=h, jpeg_q=self.jpeg_q,
                              a_proc_ms=(time.perf_counter() - t0) * 1000)

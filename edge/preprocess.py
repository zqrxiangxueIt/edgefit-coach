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


class EventPreprocessor:
    """Event-driven streaming: frame-diff gating + adaptive JPEG quality.

    The A-side "preliminary computation" that matters for bandwidth: on a
    static scene nothing is transmitted at all; when motion appears, the
    JPEG quality scales with motion intensity (subtle motion -> blurrier
    but small frames is the right trade for pose estimation).

    * motion is measured on a small grayscale downscale (cheap on the Pi)
    * the FIRST frame is always sent (establishes the reference)
    * `keepalive_every` > 0 re-sends a frame every N inputs even when idle,
      so a dashboard can show a live (static) picture instead of freezing
    * header.motion carries the normalized score 0..1; raw score in extra
    """

    def __init__(self, max_width: int = 960,
                 q_min: int = 55, q_max: int = 85,
                 motion_ref: float = 6.0,      # raw diff mean that maps to 1.0
                 idle_thr: float = 3.0,        # raw diff mean below this = idle
                 keepalive_every: int = 0) -> None:
        # idle_thr calibrated on the real Global Shutter Camera @720p:
        # noise floor p50=0.87 p99=1.78 max=2.18 (15s static scene, 451 frames);
        # 3.0 keeps a ~40% margin over worst-case sensor noise.  Synthetic
        # noise-free test fixtures pass their own lower idle_thr.
        self.max_width = max_width
        self.q_min, self.q_max = q_min, q_max
        self.motion_ref = motion_ref
        self.idle_thr = idle_thr
        self.keepalive_every = keepalive_every
        self._prev_small: np.ndarray | None = None
        self._i = 0

    def _motion(self, frame: np.ndarray) -> tuple[float, float, bool]:
        """(raw mean diff, diff p99, had_reference).

        The mean drives gating (calibrated); p99 is recorded in extra --
        a small distant mover barely moves the mean but lights up p99,
        which is the right signal for future per-region gating (N2).
        """
        h, w = frame.shape[:2]
        small_w = 320
        small = cv2.resize(frame, (small_w, max(1, round(h * small_w / w))),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self._prev_small is None:
            self._prev_small = gray
            return 0.0, 0.0, False
        diff = cv2.absdiff(gray, self._prev_small)
        self._prev_small = gray
        return float(diff.mean()), float(np.percentile(diff, 99)), True

    def process(self, frame: np.ndarray) -> ProcessedFrame | None:
        self._i += 1
        t0 = time.perf_counter()
        raw_motion, diff_p99, had_ref = self._motion(frame)
        norm = min(1.0, raw_motion / self.motion_ref) if self.motion_ref > 0 else 0.0

        keepalive = (self.keepalive_every > 0
                     and self._i % self.keepalive_every == 0)
        if had_ref and raw_motion < self.idle_thr and not keepalive:
            return None                                    # idle: do not transmit

        h, w = frame.shape[:2]
        if w > self.max_width:
            nh = round(h * self.max_width / w)
            frame = cv2.resize(frame, (self.max_width, nh),
                               interpolation=cv2.INTER_AREA)
        jpeg_q = round(self.q_min + (self.q_max - self.q_min) * norm)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        if not ok:
            return None
        return ProcessedFrame(jpeg=buf.tobytes(), w=w, h=h, jpeg_q=jpeg_q,
                              motion=norm, a_proc_ms=(time.perf_counter() - t0) * 1000,
                              extra={"motion_raw": round(raw_motion, 2),
                                     "diff_p99": round(diff_p99, 1)})

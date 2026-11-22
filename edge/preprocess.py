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


def _person_boxes_json(persons) -> list[list[float]]:
    """numpy float32 -> plain floats (JSON-serializable for the header)."""
    return [[round(float(v), 1) for v in b[:4]] + [round(float(b[4]), 2)]
            for b in persons]


def _encode(frame: np.ndarray, max_width: int, jpeg_q: int) -> tuple[bytes, int, int] | None:
    """Downscale + JPEG encode; returns (jpeg, w, h) or None on failure."""
    h, w = frame.shape[:2]
    if w > max_width:
        nh = round(h * max_width / w)
        frame = cv2.resize(frame, (max_width, nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return (buf.tobytes(), w, h) if ok else None


class CascadePreprocessor:
    """On-device NN first-stage filter: transmit only frames where the edge
    model sees a person (Reducto/E4-style edge filtering -- the A-side
    "preliminary computation" is itself a neural network).

    `grace_frames` keeps the stream alive briefly after the person leaves
    (hysteresis, avoids flapping on detector dropouts).
    """

    def __init__(self, detector, max_width: int = 960, jpeg_q: int = 75,
                 grace_frames: int = 10, keepalive_every: int = 0) -> None:
        self.det = detector
        self.max_width = max_width
        self.jpeg_q = jpeg_q
        self.grace = grace_frames
        self.keepalive_every = keepalive_every
        self._left = 0
        self._i = 0

    def process(self, frame: np.ndarray) -> ProcessedFrame | None:
        self._i += 1
        t0 = time.perf_counter()
        persons = self.det.detect(frame)
        det_ms = getattr(self.det, "last_ms", 0.0)

        keepalive = (self.keepalive_every > 0
                     and self._i % self.keepalive_every == 0)
        # decide BEFORE decrementing: grace_frames=N sends up to N frames
        # after the person disappears (intuitive semantics)
        send = bool(persons) or self._left > 0 or keepalive or self._i == 1
        if persons:
            self._left = self.grace
        else:
            self._left = max(0, self._left - 1)
        if not send:
            return None                                  # no person: silence

        enc = _encode(frame, self.max_width, self.jpeg_q)
        if enc is None:
            return None
        jpeg, w, h = enc
        return ProcessedFrame(jpeg=jpeg, w=w, h=h, jpeg_q=self.jpeg_q,
                              a_proc_ms=(time.perf_counter() - t0) * 1000,
                              extra={"persons": _person_boxes_json(persons),
                                     "det_ms": round(det_ms, 1)})


class PrivacyPreprocessor:
    """Privacy mode: only the (padded, smoothed) person ROI is ever
    transmitted -- everything outside the person stays on device A.

    Side benefit: the ROI crop acts as a digital zoom, so distant subjects
    fill the pose model's input and keypoint precision improves.
    """

    def __init__(self, detector, max_width: int = 960, jpeg_q: int = 82,
                 pad_frac: float = 0.30, smooth: float = 0.65,
                 lost_reset: int = 15, keepalive_every: int = 0) -> None:
        self.det = detector
        self.max_width = max_width
        self.jpeg_q = jpeg_q
        self.pad_frac = pad_frac
        self.smooth = smooth
        self.lost_reset = lost_reset
        self.keepalive_every = keepalive_every
        self._roi: tuple[int, int, int, int] | None = None   # x, y, w, h
        self._lost = 0
        self._i = 0

    def _track_roi(self, persons, w: int, h: int) -> tuple[int, int, int, int] | None:
        if persons:
            x1, y1, x2, y2, _ = max(persons, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            px = (x2 - x1) * self.pad_frac
            py = (y2 - y1) * self.pad_frac
            x1, y1 = max(0, x1 - px), max(0, y1 - py)
            x2, y2 = min(w, x2 + px), min(h, y2 + py)
            cur = (x1, y1, x2 - x1, y2 - y1)
            if self._roi is None or self._lost >= self.lost_reset:
                self._roi = cur
            else:
                a = np.array(self._roi, dtype=np.float64)
                self._roi = tuple((a * self.smooth + np.array(cur) * (1 - self.smooth)))
            self._lost = 0
            return self._roi
        self._lost += 1
        return None if self._lost > self.lost_reset else self._roi

    def process(self, frame: np.ndarray) -> ProcessedFrame | None:
        self._i += 1
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        persons = self.det.detect(frame)
        det_ms = getattr(self.det, "last_ms", 0.0)
        roi = self._track_roi(persons, w, h)

        keepalive = (self.keepalive_every > 0
                     and self._i % self.keepalive_every == 0)
        if roi is None:
            if not keepalive:
                return None
            # keepalive with an explicit "no person" marker (tiny black frame)
            black = np.zeros((64, 64, 3), np.uint8)
            ok, buf = cv2.imencode(".jpg", black, [cv2.IMWRITE_JPEG_QUALITY, 60])
            return ProcessedFrame(jpeg=buf.tobytes(), w=w, h=h, jpeg_q=60,
                                  a_proc_ms=(time.perf_counter() - t0) * 1000,
                                  extra={"persons": [], "roi": None,
                                         "det_ms": round(det_ms, 1)}) if ok else None

        x, y, rw, rh = [int(v) for v in roi]
        crop = frame[max(0, y):min(h, y + rh), max(0, x):min(w, x + rw)]
        if crop.size == 0:
            return None
        enc = _encode(crop, self.max_width, self.jpeg_q)
        if enc is None:
            return None
        jpeg, _, _ = enc
        return ProcessedFrame(jpeg=jpeg, w=w, h=h, jpeg_q=self.jpeg_q,
                              a_proc_ms=(time.perf_counter() - t0) * 1000,
                              extra={"persons": _person_boxes_json(persons),
                                     "roi": [int(x), int(y), int(rw), int(rh)],
                                     "det_ms": round(det_ms, 1)})

"""Frame sources with a common interface: video file (paced like a camera)
and USB camera (arrives with the hardware, M3).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2


class Source(ABC):
    """Yields (frame_bgr, t_capture); t_capture is this host's clock."""

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @abstractmethod
    def read(self) -> tuple | None:
        """Next frame as (frame, t_capture), or None when exhausted/failed."""

    def close(self) -> None: ...


class VideoFileSource(Source):
    """Replays a file at its native FPS (real-time pacing, like a camera).

    Set realtime=False to drain the file as fast as possible (useful for
    offline benchmarking of the A-side preprocessing only).
    """

    def __init__(self, path: str | Path, realtime: bool = True) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.realtime = realtime
        self._next_t = 0.0
        self._n = 0

    @property
    def fps(self) -> float:
        return self._fps

    def read(self):
        if self.realtime:
            now = time.monotonic()
            if self._next_t == 0.0:
                self._next_t = now
            elif now < self._next_t:
                time.sleep(self._next_t - now)
            elif now > self._next_t + 1.0 / self._fps:
                # stalled behind schedule (GC, overload): a real camera would
                # jump to "now", not fast-forward through the backlog --
                # re-anchor so we never burst (S2)
                self._next_t = now
            self._next_t += 1.0 / self._fps
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._n += 1
        return frame, time.time()

    def close(self) -> None:
        self.cap.release()


class CameraSource(Source):
    """USB webcam via V4L2/OpenCV.

    MJPG is requested explicitly: OpenCV may otherwise negotiate uncompressed
    YUY2, which at 720p30 (~44 MB/s) saturates the USB 2.0 link and drops
    frames.  Compressed MJPG leaves plenty of headroom.
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30) -> None:
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._fps = fps
        self._index = index
        self._warned = False
        # verify what the driver actually negotiated (N5): a silent YUY2
        # fallback would saturate the USB 2.0 link and drop frames
        got_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        got_fmt = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
        if (got_w, got_h) != (width, height) or "MJPG" not in got_fmt:
            print(f"[camera] WARNING: negotiated {got_w}x{got_h} {got_fmt}, "
                  f"requested {width}x{height} MJPG", flush=True)

    @property
    def fps(self) -> float:
        return self._fps

    def read(self):
        """Retry transient V4L2 failures (USB reset), with one full reopen
        before giving up -- a dead handle must not stop the stream for good."""
        for attempt in range(20):
            if attempt == 10:   # half the budget gone: try a fresh handle
                print("[camera] reopening device ...", flush=True)
                self.cap.release()
                self.cap.open(self._index)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            ok, frame = self.cap.read()
            if ok:
                if self._warned:
                    print("[camera] recovered", flush=True)
                    self._warned = False
                return frame, time.time()
            if not self._warned:
                print("[camera] WARNING: read failed, retrying ...", flush=True)
                self._warned = True
            time.sleep(0.05)
        return None

    def close(self) -> None:
        self.cap.release()

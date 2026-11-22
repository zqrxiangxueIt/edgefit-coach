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
            self._next_t += 1.0 / self._fps
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._n += 1
        return frame, time.time()

    def close(self) -> None:
        self.cap.release()


class CameraSource(Source):
    """USB webcam via V4L2/OpenCV; falls back to camera index 0."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30) -> None:
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._fps = fps

    @property
    def fps(self) -> float:
        return self._fps

    def read(self):
        ok, frame = self.cap.read()
        return (frame, time.time()) if ok else None

    def close(self) -> None:
        self.cap.release()

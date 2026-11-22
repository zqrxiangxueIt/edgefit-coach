"""Frame annotation rendering shared by offline pipeline, console receiver
and the web dashboard."""

from __future__ import annotations

import cv2
import numpy as np

from .curl_counter import CurlCounter
from .geometry import LEFT_ARM, RIGHT_ARM, joint_angle

# COCO-17 limb pairs
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (5, 6),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw(frame: np.ndarray, res, counter: CurlCounter,
         smoother_out: np.ndarray) -> np.ndarray:
    """In-place annotation: box, skeleton, smoothed keypoints, HUD."""
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

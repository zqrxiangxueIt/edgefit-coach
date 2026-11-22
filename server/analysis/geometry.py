"""Joint geometry helpers over COCO-17 keypoints.

Keypoint arrays are (17, 3) float: x, y in pixel coords, conf in [0, 1].
"""

from __future__ import annotations

import math

import numpy as np

# COCO-17 keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

# Convenience tuples for the two arms: (shoulder, elbow, wrist)
LEFT_ARM = (L_SHOULDER, L_ELBOW, L_WRIST)
RIGHT_ARM = (R_SHOULDER, R_ELBOW, R_WRIST)


def joint_angle(kps: np.ndarray, a: int, b: int, c: int) -> tuple[float, float]:
    """Interior angle ABC in degrees at joint b; returns (angle, confidence).

    Confidence is the product of the three keypoint confidences, so a single
    shaky point drags the measurement down.
    """
    pa, pb, pc = kps[a], kps[b], kps[c]
    conf = float(pa[2] * pb[2] * pc[2])
    v1 = pa[:2] - pb[:2]
    v2 = pc[:2] - pb[:2]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0, 0.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosang)), conf

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

    Confidence is the min of the three keypoint confidences: the angle is
    only as trustworthy as its weakest point.
    """
    pa, pb, pc = kps[a], kps[b], kps[c]
    conf = float(min(pa[2], pb[2], pc[2]))
    v1 = pa[:2] - pb[:2]
    v2 = pc[:2] - pb[:2]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-3 or n2 < 1e-3:   # below float32 pixel resolution: degenerate
        return 180.0, 0.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosang)), conf


# ------------------------------------------------------------------- view --

def estimate_yaw_scale(kps: np.ndarray, rho_front: float = 0.85,
                       rho_side: float = 0.30) -> tuple[float, bool]:
    """Estimate cos(body-yaw) from shoulder-width / torso-length ratio.

    Standing curl: the forearm sweeps in the sagittal plane, whose normal is
    the shoulder axis -- so shoulder apparent width collapses as the body
    turns toward the camera. rho_front/rho_side (ratio facing / sideways the
    camera) come from anthropometrics or the calibration flow. Returns
    (cos_psi, valid); valid requires all four torso keypoints conf >= 0.3.
    """
    ls, rs, lh, rh = kps[L_SHOULDER], kps[R_SHOULDER], kps[L_HIP], kps[R_HIP]
    if min(ls[2], rs[2], lh[2], rh[2]) < 0.3:
        return 0.0, False
    torso = float(np.median([np.hypot(*(ls[:2] - lh[:2])),
                             np.hypot(*(rs[:2] - rh[:2]))]))
    if torso < 10.0:
        return 0.0, False
    rho = float(np.hypot(*(ls[:2] - rs[:2]))) / torso
    cos_psi = (rho - rho_side) / max(rho_front - rho_side, 1e-6)
    return float(np.clip(cos_psi, 0.0, 1.0)), True


def correct_projected_angle(theta_proj: float, cos_psi: float,
                            min_scale: float = 0.5,
                            max_correction: float = 25.0) -> float:
    """De-project a sagittal-plane joint angle seen at body-yaw psi.

    Closed form (orthographic): tan(theta_true/2) = cos(phi)*tan(theta_proj/2)
    with cos(phi) = sin(psi). Projected angles are ALWAYS >= true, so this
    only shrinks; correction is clamped and skipped when nearly frontal
    (sin(psi) < min_scale: the angle is unobservable edge-on) or outside
    the trustworthy band.
    """
    s2 = 1.0 - cos_psi * cos_psi
    if s2 < min_scale * min_scale:
        return theta_proj            # frontal: unobservable, don't fake it
    if not 30.0 <= theta_proj <= 160.0:
        return theta_proj            # near the extremes the formula is a no-op
    s = math.sqrt(s2)
    theta = 2.0 * math.degrees(math.atan(s * math.tan(math.radians(theta_proj / 2.0))))
    return theta_proj + max(-max_correction, min(max_correction, theta - theta_proj))

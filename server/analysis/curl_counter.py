"""Bicep curl repetition counter: hysteresis state machine over the elbow angle.

Pipeline position: raw keypoints -> One-Euro smoothing (see filters.py) ->
CurlCounter.update() per frame.

Rep logic (v1, dumbbell bicep curl):
    * phase EXTENDED: elbow angle >= up_angle  (arm hanging straight)
    * phase CURLING : elbow angle <= down_angle (dumbbell raised)
    * a rep completes on CURLING -> EXTENDED transition confirmed for
      `debounce` consecutive frames;
    * full reps require the elbow to have closed below full_rep_angle at
      least once during the rep, otherwise the rep is flagged "half" and
      surfaces a form warning.

Side handling: the tracked arm is chosen automatically by cumulative
keypoint confidence (left vs right), locked in after the first few frames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .geometry import LEFT_ARM, RIGHT_ARM, joint_angle, L_SHOULDER, R_SHOULDER


class Phase(Enum):
    IDLE = "idle"          # no confident person/arm
    EXTENDED = "extended"  # arm straight, bottom of the movement
    CURLING = "curling"    # arm flexed, top of the movement


@dataclass
class RepEvent:
    index: int          # 1-based rep number
    full: bool          # False = half rep (insufficient range)
    min_angle: float    # best flexion reached during the rep
    t: float            # completion time (seconds, server clock)


@dataclass
class SessionStats:
    reps: int = 0
    full_reps: int = 0
    half_reps: int = 0
    last_rep: Optional[RepEvent] = None
    warnings: list = field(default_factory=list)


class CurlCounter:
    def __init__(self,
                 up_angle: float = 150.0,
                 down_angle: float = 65.0,
                 full_rep_angle: float = 75.0,
                 debounce: int = 3,
                 min_conf: float = 0.30,
                 side: str = "auto") -> None:
        if not down_angle < full_rep_angle < up_angle:
            raise ValueError("require down_angle < full_rep_angle < up_angle")
        self.up_angle = up_angle
        self.down_angle = down_angle
        self.full_rep_angle = full_rep_angle
        self.debounce = debounce
        self.min_conf = min_conf
        self.side = side

        self.phase = Phase.IDLE
        self.stats = SessionStats()
        self._debounce_left = 0
        self._min_angle_in_rep = 180.0
        self._arm: Optional[tuple[int, int, int]] = None
        self._conf_left = 0.0
        self._conf_right = 0.0

    # ---------------------------------------------------------------- api --

    def update(self, kps: np.ndarray, t: Optional[float] = None) -> Optional[RepEvent]:
        """Feed one (17, 3) keypoint frame; returns a RepEvent on rep completion."""
        t = time.time() if t is None else t
        self._pick_side(kps)
        if self._arm is None:
            self._set_phase(Phase.IDLE)
            return None

        angle, conf = joint_angle(kps, *self._arm)
        if conf < self.min_conf:
            self._set_phase(Phase.IDLE)
            return None

        self._min_angle_in_rep = min(self._min_angle_in_rep, angle)
        return self._step(angle, t)

    # ------------------------------------------------------------ internals --

    def _pick_side(self, kps: np.ndarray) -> None:
        if self.side in ("left", "right"):
            self._arm = LEFT_ARM if self.side == "left" else RIGHT_ARM
            return
        if self._arm is not None:
            return  # locked after first acquisition
        la = min(kps[i, 2] for i in LEFT_ARM)
        ra = min(kps[i, 2] for i in RIGHT_ARM)
        self._conf_left += la
        self._conf_right += ra
        if self._conf_left + self._conf_right > 2.0:  # a few frames of evidence
            self._arm = LEFT_ARM if self._conf_left >= self._conf_right else RIGHT_ARM

    def _step(self, angle: float, t: float) -> Optional[RepEvent]:
        target: Optional[Phase] = None
        if angle <= self.down_angle:
            target = Phase.CURLING
        elif angle >= self.up_angle:
            target = Phase.EXTENDED

        if target is not None and self._debounce_left > 0 and target != self.phase:
            self._debounce_left -= 1          # require N consecutive frames
            if self._debounce_left > 0:
                return None

        if target is None:
            self._debounce_left = self.debounce  # mid-range: reset debounce
            return None

        if target == self.phase:
            return None

        event: Optional[RepEvent] = None
        if target is Phase.EXTENDED and self.phase is Phase.CURLING:
            full = self._min_angle_in_rep <= self.full_rep_angle
            self.stats.reps += 1
            self.stats.full_reps += int(full)
            self.stats.half_reps += int(not full)
            event = RepEvent(self.stats.reps, full, self._min_angle_in_rep, t)
            self.stats.last_rep = event
            if not full:
                self.stats.warnings.append(
                    f"Rep {event.index}: 幅度不足 (最低 {self._min_angle_in_rep:.0f}°, "
                    f"需 < {self.full_rep_angle:.0f}°)")

        if target is Phase.EXTENDED:
            self._min_angle_in_rep = 180.0   # start measuring the next rep
        self._set_phase(target)
        self._debounce_left = self.debounce
        return event

    def _set_phase(self, phase: Phase) -> None:
        if phase is Phase.IDLE and self.phase is not Phase.IDLE:
            self._min_angle_in_rep = 180.0
        self.phase = phase


class TorsoSwingChecker:
    """Flags 'cheating by swinging': horizontal shoulder travel during a rep.

    Shoulder x-displacement is compared against `max_shift_frac` of the
    current shoulder-to-hip torso length, making the check scale-free.
    """

    def __init__(self, max_shift_frac: float = 0.35, window: int = 45) -> None:
        self.max_shift_frac = max_shift_frac
        self.window = window
        self._xs: list[float] = []
        self._active = False

    def start_rep(self) -> None:
        self._xs = []
        self._active = True

    def update(self, kps: np.ndarray) -> Optional[str]:
        if not self._active:
            return None
        s = kps[L_SHOULDER] if kps[L_SHOULDER, 2] >= kps[R_SHOULDER, 2] else kps[R_SHOULDER]
        self._xs.append(float(s[0]))
        if len(self._xs) > self.window:
            self._xs.pop(0)
            self._active = False
            span = max(self._xs) - min(self._xs)
            # torso length as scale reference
            hip = kps[11] if kps[L_SHOULDER, 2] >= kps[R_SHOULDER, 2] else kps[12]
            torso = float(np.hypot(*(s[:2] - hip[:2]))) or 1.0
            if span > self.max_shift_frac * torso:
                return f"躯干摆动过大（借力），span={span:.0f}px / torso={torso:.0f}px"
        return None

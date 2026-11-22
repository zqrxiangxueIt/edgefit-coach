"""Bicep curl repetition counter: hysteresis state machines over elbow angles.

Pipeline position: raw keypoints -> One-Euro smoothing (see filters.py) ->
CurlCounter.update() per frame.

Both arms are tracked independently (alternating dumbbell curls are a real
training pattern); a rep from either arm counts.  A resting arm holds a
roughly constant angle and never oscillates through the hysteresis band,
so it cannot produce phantom reps.

Per-arm rep logic (v1, dumbbell bicep curl):
    * phase EXTENDED: elbow angle >= up_angle   (arm straight, rep boundary;
      140deg by default - gym form rarely fully locks the elbow)
    * phase CURLING : elbow angle <= flex_angle (rep underway)
    * a rep completes on CURLING -> EXTENDED confirmed for `debounce`
      consecutive frames;
    * a completed rep is "full" when the elbow closed below full_rep_angle
      at least once during the rep (flex_angle > full_rep_angle, so a rep
      may legally complete as a half rep -> form warning).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .geometry import LEFT_ARM, RIGHT_ARM, joint_angle, L_SHOULDER, R_SHOULDER


class Phase(Enum):
    IDLE = "idle"          # no confident keypoints for this arm
    EXTENDED = "extended"  # arm straight, bottom of the movement
    CURLING = "curling"    # rep underway


@dataclass
class RepEvent:
    index: int          # 1-based rep number (session-wide)
    side: str           # "left" | "right"
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


class _ArmTracker:
    """Per-arm angle hysteresis machine."""

    def __init__(self, side: str, up_angle: float, flex_angle: float,
                 full_rep_angle: float, debounce: int) -> None:
        self.side = side
        self.up_angle = up_angle
        self.flex_angle = flex_angle
        self.full_rep_angle = full_rep_angle
        self.debounce = debounce
        self.phase = Phase.IDLE
        self._debounce_left = debounce
        self._min_angle_in_rep = 180.0

    def lost(self) -> None:
        self.phase = Phase.IDLE
        self._min_angle_in_rep = 180.0
        self._debounce_left = self.debounce

    def step(self, angle: float, t: float) -> Optional[float]:
        """Advance one frame.

        Returns the rep's minimum elbow angle exactly once per completed
        CURLING -> EXTENDED rep, else None.
        """
        target: Optional[Phase] = None
        if angle <= self.flex_angle:
            target = Phase.CURLING
        elif angle >= self.up_angle:
            target = Phase.EXTENDED
        else:
            self._debounce_left = self.debounce   # mid-range resets debounce
            return None

        if target != self.phase and self._debounce_left > 0:
            self._debounce_left -= 1
            if self._debounce_left > 0:
                return None

        if target == self.phase:
            return None

        completed_min: Optional[float] = None
        if target is Phase.EXTENDED and self.phase is Phase.CURLING:
            completed_min = self._min_angle_in_rep
        self.phase = target
        self._debounce_left = self.debounce
        if target is Phase.EXTENDED:
            self._min_angle_in_rep = 180.0        # start measuring the next rep
        return completed_min


class CurlCounter:
    def __init__(self,
                 up_angle: float = 140.0,
                 flex_angle: float = 110.0,
                 full_rep_angle: float = 75.0,
                 debounce: int = 3,
                 min_conf: float = 0.35,
                 side: str = "both") -> None:
        if not full_rep_angle < flex_angle < up_angle:
            raise ValueError("require full_rep_angle < flex_angle < up_angle")
        if side not in ("both", "left", "right"):
            raise ValueError("side must be one of both|left|right")
        self.min_conf = min_conf
        self.up_angle, self.flex_angle = up_angle, flex_angle
        self.full_rep_angle, self.debounce = full_rep_angle, debounce
        self.stats = SessionStats()
        self.trackers: dict[str, _ArmTracker] = {}
        if side in ("both", "left"):
            self.trackers["left"] = _ArmTracker("left", up_angle, flex_angle, full_rep_angle, debounce)
        if side in ("both", "right"):
            self.trackers["right"] = _ArmTracker("right", up_angle, flex_angle, full_rep_angle, debounce)

    _PHASE_RANK = {Phase.IDLE: 0, Phase.CURLING: 1, Phase.EXTENDED: 2}

    @property
    def phase(self) -> Phase:
        """Most active tracker's phase (UI convenience)."""
        if not self.trackers:
            return Phase.IDLE
        return max(self.trackers.values(),
                   key=lambda tr: self._PHASE_RANK[tr.phase]).phase

    def arm_angle(self, kps: np.ndarray, side: str) -> tuple[float, float]:
        arm = LEFT_ARM if side == "left" else RIGHT_ARM
        return joint_angle(kps, *arm)

    def update(self, kps: np.ndarray, t: Optional[float] = None) -> Optional[RepEvent]:
        """Feed one (17, 3) keypoint frame; returns a RepEvent on rep completion."""
        t = time.time() if t is None else t
        event: Optional[RepEvent] = None
        for side, tracker in self.trackers.items():
            angle, conf = self.arm_angle(kps, side)
            if conf < self.min_conf:
                tracker.lost()
                continue
            # track best flexion before step() may reset it on rep completion
            tracker._min_angle_in_rep = min(tracker._min_angle_in_rep, angle)
            rep_min = tracker.step(angle, t)
            if rep_min is not None:
                full = rep_min <= self.full_rep_angle
                self.stats.reps += 1
                self.stats.full_reps += int(full)
                self.stats.half_reps += int(not full)
                event = RepEvent(self.stats.reps, side, full, rep_min, t)
                self.stats.last_rep = event
                if not full:
                    self.stats.warnings.append(
                        f"[{side}] rep {event.index}: 幅度不足 (最低 {event.min_angle:.0f}°, "
                        f"需 < {self.full_rep_angle:.0f}°)")
        return event


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
        use_left = kps[L_SHOULDER, 2] >= kps[R_SHOULDER, 2]
        s = kps[L_SHOULDER] if use_left else kps[R_SHOULDER]
        hip = kps[11] if use_left else kps[12]
        self._xs.append(float(s[0]))
        if len(self._xs) > self.window:
            self._xs.pop(0)
            self._active = False
            span = max(self._xs) - min(self._xs)
            torso = float(np.hypot(*(s[:2] - hip[:2]))) or 1.0
            if span > self.max_shift_frac * torso:
                return f"躯干摆动过大（借力），span={span:.0f}px / torso={torso:.0f}px"
        return None

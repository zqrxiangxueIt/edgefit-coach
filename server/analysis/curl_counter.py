"""Bicep curl repetition counter: hysteresis state machines over elbow angles.

Pipeline position: raw keypoints -> One-Euro smoothing (see filters.py) ->
CurlCounter.update() per frame.

Both arms are tracked independently (alternating / simultaneous dumbbell and
barbell curls are real training patterns); reps from both arms are reported.
A resting arm is guarded by an activity gate: occlusion can "teleport" its
wrist keypoint to a fake curled pose, so a rep only counts if the wrist
actually travelled a plausible arc path during the CURLING phase
(>= min_path_frac x upper-arm length, per-frame movement capped so a single
teleport cannot fake a journey; the gate re-arms when CURLING is confirmed,
so rest-period drift cannot pre-charge it).  Suppressed phantoms are tallied
in ``SessionStats.phantom_reps_suppressed``.

Known limitation (accepted for v1): a misdetection that *flaps* every frame
between two spots can accumulate capped path and pass the gate; the debounce
consecutive-frames semantics still require it to look like a real phase
trajectory, which continuous flapping does not.

Per-arm rep logic (v1, dumbbell bicep curl):
    * phase EXTENDED: elbow angle >= up_angle   (arm straight, rep boundary;
      140deg by default - gym form rarely fully locks the elbow)
    * phase CURLING : elbow angle <= flex_angle (rep underway)
    * a phase change must be confirmed by `debounce` CONSECUTIVE frames on
      the target side of the band; any frame back inside the band, or back
      on the current phase, re-arms the debounce;
    * a rep completes on CURLING -> EXTENDED and reports the minimum elbow
      angle reached while CURLING was confirmed (glitches while EXTENDED
      cannot pollute it);
    * "full" means the elbow closed below full_rep_angle at least once.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .geometry import LEFT_ARM, RIGHT_ARM, joint_angle, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP


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
    phantom_reps_suppressed: int = 0
    last_rep: Optional[RepEvent] = None
    warnings: list = field(default_factory=list)


class _ActivityGate:
    """Rejects phantom reps from a resting arm whose wrist keypoint was
    teleported by occlusion/misdetection.

    A genuine curl sweeps the wrist along a long arc over many frames; a
    misdetection jumps the wrist once to a wrong spot where it then sits
    still.  We accumulate the wrist path with a per-frame cap so a single
    teleport contributes little, and require a minimum total path (relative
    to the upper-arm length, which is stable while the wrist is not).
    """

    def __init__(self, cap_frac: float = 0.4, min_path_frac: float = 1.0) -> None:
        self.cap_frac = cap_frac
        self.min_path_frac = min_path_frac
        self.path = 0.0
        self._prev: Optional[tuple[float, float]] = None

    def reset(self) -> None:
        self.path = 0.0
        self._prev = None

    def add(self, wrist_xy: tuple[float, float], upper_arm_len: float) -> None:
        if self._prev is not None:
            delta = float(np.hypot(wrist_xy[0] - self._prev[0], wrist_xy[1] - self._prev[1]))
            self.path += min(delta, self.cap_frac * upper_arm_len)
        self._prev = wrist_xy

    def passed(self, upper_arm_len: float) -> bool:
        return self.path >= self.min_path_frac * upper_arm_len


class _ArmTracker:
    """Per-arm angle hysteresis machine with consecutive-frame debouncing."""

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
        self.gate = _ActivityGate()
        self.rep_started_t: Optional[float] = None

    def lost(self) -> None:
        self.phase = Phase.IDLE
        self._min_angle_in_rep = 180.0
        self._debounce_left = self.debounce
        self.gate.reset()
        self.rep_started_t = None

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
            self._debounce_left = self.debounce   # mid-band frame re-arms debounce
            return None

        if target == self.phase:
            self._debounce_left = self.debounce   # confirmed in-phase: re-arm
            return None

        if self._debounce_left > 0:               # not yet N consecutive frames
            self._debounce_left -= 1
            if self._debounce_left > 0:
                return None

        completed_min: Optional[float] = None
        if target is Phase.EXTENDED and self.phase is Phase.CURLING:
            completed_min = self._min_angle_in_rep
        self.phase = target
        self._debounce_left = self.debounce
        if target is Phase.CURLING:
            # seed with the confirming frame's angle: for a fast rep this is
            # often the deepest point, and pre-confirmation frames were
            # measured while EXTENDED (untrusted, see update())
            self._min_angle_in_rep = angle
        else:
            self._min_angle_in_rep = 180.0        # start measuring the next rep
        self.gate.reset()                         # re-arm on every confirmed phase
        return completed_min


class TorsoSwingChecker:
    """Flags 'cheating by swinging': horizontal shoulder travel during a rep.

    Shoulder x-displacement between rep start and completion is compared
    against `max_shift_frac` of the shoulder-to-hip torso length, making the
    check scale-free.  Each side's shoulder is tracked separately and locked
    for the whole rep (no mid-rep side switching).
    """

    _SH_HIP = {"left": (L_SHOULDER, L_HIP), "right": (R_SHOULDER, R_HIP)}

    def __init__(self, max_shift_frac: float = 0.35, history: int = 600) -> None:
        self.max_shift_frac = max_shift_frac
        self._hist: dict[str, deque] = {
            "left": deque(maxlen=history), "right": deque(maxlen=history)}

    def update(self, kps: np.ndarray, t: float) -> None:
        """Record confident shoulder positions; call every frame."""
        for side, (si, hi) in self._SH_HIP.items():
            if kps[si, 2] > 0.3 and kps[hi, 2] > 0.3:
                torso = float(np.hypot(*(kps[si, :2] - kps[hi, :2])))
                self._hist[side].append((t, float(kps[si, 0]), torso))

    def verdict(self, side: str, t_start: float, t_end: float) -> Optional[str]:
        """Swing verdict over one completed rep's [t_start, t_end]."""
        rows = [(x, L) for (t, x, L) in self._hist[side] if t_start <= t <= t_end]
        if len(rows) < 3:
            return None
        xs = [x for x, _ in rows]
        span = max(xs) - min(xs)
        torso = float(np.median([L for _, L in rows])) or 1.0
        if span > self.max_shift_frac * torso:
            return f"躯干摆动过大（疑似借力），span={span:.0f}px / torso={torso:.0f}px"
        return None


class CurlCounter:
    _PHASE_RANK = {Phase.IDLE: 0, Phase.EXTENDED: 1, Phase.CURLING: 2}

    def __init__(self,
                 up_angle: float = 140.0,
                 flex_angle: float = 110.0,
                 full_rep_angle: float = 75.0,
                 debounce: int = 3,
                 min_conf: float = 0.35,
                 side: str = "both",
                 check_torso_swing: bool = True,
                 full_rep_tol: float = 0.5) -> None:
        if not full_rep_angle < flex_angle < up_angle:
            raise ValueError("require full_rep_angle < flex_angle < up_angle")
        if side not in ("both", "left", "right"):
            raise ValueError("side must be one of both|left|right")
        if int(debounce) < 1:
            raise ValueError("debounce must be >= 1")
        if not 0.0 <= min_conf <= 1.0:
            raise ValueError("min_conf must be in [0, 1]")
        if full_rep_tol < 0:
            raise ValueError("full_rep_tol must be >= 0")
        self.min_conf = min_conf
        self.flex_angle = flex_angle
        self.full_rep_angle = full_rep_angle
        self.full_rep_tol = full_rep_tol   # angles are estimates; tolerate ~0.5deg
        self.stats = SessionStats()
        self.torso = TorsoSwingChecker() if check_torso_swing else None
        self.trackers: dict[str, _ArmTracker] = {}
        arms = {"left": LEFT_ARM, "right": RIGHT_ARM}
        for s in (("left", "right") if side == "both" else (side,)):
            self.trackers[s] = _ArmTracker(s, up_angle, flex_angle,
                                           full_rep_angle, int(debounce))

    @property
    def phase(self) -> Phase:
        """Most active tracker's phase (UI convenience: curling wins)."""
        if not self.trackers:
            return Phase.IDLE
        return max(self.trackers.values(),
                   key=lambda tr: self._PHASE_RANK[tr.phase]).phase

    def arm_angle(self, kps: np.ndarray, side: str) -> tuple[float, float]:
        return joint_angle(kps, *LEFT_ARM if side == "left" else RIGHT_ARM)

    def update(self, kps: np.ndarray, t: Optional[float] = None) -> list[RepEvent]:
        """Feed one (17, 3) keypoint frame.

        Returns the (possibly empty) list of reps completed this frame --
        both arms can complete a rep in the same frame (barbell curls).
        """
        t = time.time() if t is None else t
        if self.torso is not None:
            self.torso.update(kps, t)
        events: list[RepEvent] = []
        for side, tracker in self.trackers.items():
            angle, conf = self.arm_angle(kps, side)
            if conf < self.min_conf:
                tracker.lost()
                continue
            if tracker.phase is Phase.CURLING:
                # min only accumulates on confirmed CURLING: an unconfirmed
                # deep-flex glitch while EXTENDED cannot pollute the next rep
                tracker._min_angle_in_rep = min(tracker._min_angle_in_rep, angle)
            prev_phase = tracker.phase
            sh, el, wr = LEFT_ARM if side == "left" else RIGHT_ARM
            upper_arm = float(np.hypot(*(kps[sh, :2] - kps[el, :2])))
            # feed the gate BEFORE step(): step() re-arms it on completion,
            # and we need this rep's verdict from the pre-reset state.
            # NOTE: the gate was reset when CURLING was *confirmed*, so the
            # path measures the actual curl arc, not rest-period drift.
            tracker.gate.add((float(kps[wr, 0]), float(kps[wr, 1])),
                             max(upper_arm, 1.0))
            gate_ok = True
            if upper_arm >= 5.0:
                gate_ok = tracker.gate.passed(max(upper_arm, 1.0))
            rep_min = tracker.step(angle, t)
            if prev_phase is not Phase.CURLING and tracker.phase is Phase.CURLING:
                tracker.rep_started_t = t
            if rep_min is None:
                continue
            if not gate_ok:
                self.stats.phantom_reps_suppressed += 1
                continue
            full = rep_min <= self.full_rep_angle + self.full_rep_tol
            self.stats.reps += 1
            self.stats.full_reps += int(full)
            self.stats.half_reps += int(not full)
            ev = RepEvent(self.stats.reps, side, full, rep_min, t)
            events.append(ev)
            self.stats.last_rep = ev
            if not full:
                self.stats.warnings.append(
                    f"[{side}] rep {ev.index}: 幅度不足 (最低 {ev.min_angle:.0f}°, "
                    f"需 < {self.full_rep_angle:.0f}°)")
            if self.torso is not None and tracker.rep_started_t is not None:
                swing = self.torso.verdict(side, tracker.rep_started_t, t)
                if swing:
                    self.stats.warnings.append(f"[{side}] rep {ev.index}: {swing}")
            tracker.rep_started_t = None
        return events

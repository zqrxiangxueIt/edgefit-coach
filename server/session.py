"""Training session state machine: sets x reps with countdowns and rests.

Pure logic, wall-clock driven, unit-testable -- the pipeline feeds it rep
events and total counts, the dashboard renders its snapshot.

    idle → countdown(3s) → set(counting reps) → rest(Ns) → set ... → done

Free-form counting (no session started) keeps the old behaviour: the
counter just accumulates until the user hits the reset button.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum


class SessionPhase(Enum):
    idle = "idle"
    countdown = "countdown"   # 3-2-1 before a set starts
    set = "set"               # counting reps of the current set
    rest = "rest"             # between sets
    done = "done"             # all sets completed


@dataclass
class SessionConfig:
    sets: int = 3
    reps_per_set: int = 8
    rest_sec: float = 60.0
    start_countdown: float = 3.0

    def validate(self) -> None:
        if not 1 <= self.sets <= 20:
            raise ValueError("sets must be in 1..20")
        if not 1 <= self.reps_per_set <= 100:
            raise ValueError("reps_per_set must be in 1..100")
        if not 5 <= self.rest_sec <= 600:
            raise ValueError("rest_sec must be in 5..600")


class TrainingSession:
    """Thread-safe; all time-dependent transitions computed in update()."""

    def __init__(self, config: SessionConfig | None = None) -> None:
        self.cfg = config or SessionConfig()
        self._lock = threading.Lock()
        self._phase = SessionPhase.idle
        self._t_deadline = 0.0        # countdown / rest end (time.time base)
        self._t_started = 0.0         # session start (for elapsed)
        self._set_index = 0           # 0-based; set_index+1 shown to user
        self._set_base = 0            # counter total when this set began
        self._last_total = 0          # last counter total seen
        self._set_reps: list[int] = []  # reps completed per set (history)

    # ------------------------------------------------------------ controls --

    def start(self, total_reps: int, now: float | None = None,
              sets: int = 3, reps_per_set: int = 8, rest_sec: float = 60.0) -> None:
        now = time.time() if now is None else now
        with self._lock:
            self.cfg = SessionConfig(sets=sets, reps_per_set=reps_per_set,
                                     rest_sec=rest_sec)
            self.cfg.validate()
            self._phase = SessionPhase.countdown
            self._t_deadline = now + self.cfg.start_countdown
            self._t_started = now
            self._set_index = 0
            self._set_base = total_reps
            self._last_total = total_reps
            self._set_reps = []

    def stop(self) -> None:
        with self._lock:
            self._phase = SessionPhase.idle

    def skip_rest(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self._lock:
            if self._phase is SessionPhase.rest:
                self._set_index += 1
                self._begin_set(now)

    # ------------------------------------------------------------- engine --

    def _begin_set(self, now: float) -> None:   # caller holds the lock
        self._phase = SessionPhase.set
        self._set_base = self._last_total
        self._t_deadline = 0.0

    def update(self, total_reps: int, events: list, now: float | None = None) -> None:
        """Feed every frame: total counter value + rep events since last."""
        now = time.time() if now is None else now
        with self._lock:
            self._last_total = total_reps
            # wall-clock transitions
            if (self._phase is SessionPhase.countdown
                    and now >= self._t_deadline):
                self._begin_set(now)
            elif (self._phase is SessionPhase.rest
                    and now >= self._t_deadline):
                self._set_index += 1
                self._begin_set(now)
            # rep-driven transitions (only meaningful inside a set)
            if self._phase is not SessionPhase.set or not events:
                return
            reps_in_set = total_reps - self._set_base
            if reps_in_set >= self.cfg.reps_per_set:
                self._set_reps.append(reps_in_set)
                if self._set_index + 1 >= self.cfg.sets:
                    self._phase = SessionPhase.done
                else:
                    self._phase = SessionPhase.rest
                    self._t_deadline = now + self.cfg.rest_sec

    # ---------------------------------------------------------------- view --

    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            phase = self._phase.value
            remaining = 0.0
            if self._phase is SessionPhase.countdown:
                remaining = max(0.0, self._t_deadline - now)
            elif self._phase is SessionPhase.rest:
                remaining = max(0.0, self._t_deadline - now)
            return {
                "phase": phase,
                "sets_total": self.cfg.sets,
                "set_index": self._set_index + (1 if phase in ("set", "rest", "done") else 0),
                "reps_per_set": self.cfg.reps_per_set,
                "reps_in_set": max(0, self._last_total - self._set_base)
                               if phase in ("set", "rest", "done") else 0,
                "remaining_sec": round(remaining, 1),
                "elapsed_sec": round(now - self._t_started, 1)
                               if self._t_started else 0.0,
                "set_reps": list(self._set_reps),
                "rest_sec": self.cfg.rest_sec,
            }

"""Training history persistence (SQLite): the B-side is the system's memory.

Design (research round: Hevy/Strong-style long-term progress, minimal):
  * sessions  -- one row per training session (plan + outcome + aggregates)
  * reps      -- one row per counted rep (angle, tempo, quality), the raw
                 material for ROM/tempo trend charts
The store is thread-safe (one lock; check_same_thread=False) because the
FastAPI event loop and endpoints call it concurrently.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_start TEXT NOT NULL,
  ts_end TEXT,
  exercise TEXT NOT NULL DEFAULT 'curl',
  sets_planned INT, reps_per_set INT, rest_sec REAL,
  sets_done INT, reps_total INT,
  full_reps INT, half_reps INT, warnings INT,
  duration_sec REAL, min_angle_med REAL
);
CREATE TABLE IF NOT EXISTS reps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INT NOT NULL REFERENCES sessions(id),
  idx INT, side TEXT, full INT,
  min_angle REAL, t_rel REAL,
  t_con REAL, t_ecc REAL
);
CREATE INDEX IF NOT EXISTS ix_reps_session ON reps(session_id);
"""


class HistoryStore:
    def __init__(self, db_path: str = "data/edgefit.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.commit()
        self._current: int | None = None     # open session id
        self._stored_idx = 0                 # last rep index written

    def close(self) -> None:
        """Close the DB handle (Windows cannot delete an open db file)."""
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------ session --

    def begin(self, exercise: str = "curl", sets_planned: int = 0,
              reps_per_set: int = 0, rest_sec: float = 0.0) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO sessions (ts_start, exercise, sets_planned, "
                "reps_per_set, rest_sec) VALUES (?,?,?,?,?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"), exercise,
                 sets_planned, reps_per_set, rest_sec))
            self._db.commit()
            self._current = int(cur.lastrowid)
            self._stored_idx = 0
            return self._current

    def feed(self, events: list[dict]) -> None:
        """Record new rep events of the open session (index-deduped)."""
        if self._current is None:
            return
        with self._lock:
            for e in sorted(events, key=lambda x: x["index"]):
                if e["index"] <= self._stored_idx or self._current is None:
                    continue
                self._db.execute(
                    "INSERT INTO reps (session_id, idx, side, full, min_angle,"
                    " t_rel, t_con, t_ecc) VALUES (?,?,?,?,?,?,?,?)",
                    (self._current, e["index"], e["side"], int(e["full"]),
                     e.get("min_angle"), e.get("t"), e.get("t_con", 0.0),
                     e.get("t_ecc", 0.0)))
                self._stored_idx = e["index"]
            self._db.commit()

    def finish(self, stats: dict, session_snapshot: dict | None = None) -> None:
        """Close the open session with outcome aggregates."""
        if self._current is None:
            return
        snap = session_snapshot or {}
        reps_rows = self._db.execute(
            "SELECT min_angle FROM reps WHERE session_id=? "
            "AND min_angle IS NOT NULL ORDER BY min_angle",
            (self._current,)).fetchall()
        med = (float(reps_rows[len(reps_rows) // 2]["min_angle"])
               if reps_rows else None)
        with self._lock:
            self._db.execute(
                "UPDATE sessions SET ts_end=?, sets_done=?, reps_total=?, "
                "full_reps=?, half_reps=?, warnings=?, duration_sec=?, "
                "min_angle_med=? WHERE id=?",
                (time.strftime("%Y-%m-%d %H:%M:%S"),
                 snap.get("set_index", 0), stats.get("reps", 0),
                 stats.get("full_reps", 0), stats.get("half_reps", 0),
                 stats.get("suppressed", 0),
                 snap.get("elapsed_sec", 0.0), med, self._current))
            self._db.commit()
            self._current = None

    @property
    def current_id(self) -> int | None:
        return self._current

    # ------------------------------------------------------------- queries --

    def recent_sessions(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, ts_start, exercise, sets_planned, reps_per_set, "
                "sets_done, reps_total, full_reps, half_reps, duration_sec, "
                "min_angle_med, "
                "1.0*full_reps/MAX(reps_total,1) AS full_rate "
                "FROM sessions WHERE ts_end IS NOT NULL "
                "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def recent_reps(self, limit: int = 300) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT session_id, idx, side, full, min_angle, t_con, t_ecc "
                "FROM (SELECT * FROM reps ORDER BY id DESC LIMIT ?) "
                "ORDER BY id", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def last_session(self, exercise: str = "curl") -> dict | None:
        rows = self.recent_sessions(limit=50)
        for r in rows:
            if r["exercise"] == exercise:
                return r
        return None

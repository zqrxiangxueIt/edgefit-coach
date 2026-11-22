"""Wire protocol shared between edge (A) and server (B).

ZMQ multipart message layout (PUB/SUB):

    part 0: topic, UTF-8 bytes, currently always "frame"
    part 1: header, compact JSON, UTF-8 bytes
    part 2: payload, encoded image bytes (may be empty for keep-alives)

Header fields
-------------
seq         int     monotonic frame id assigned by A
ts_capture  float   A-side capture timestamp, seconds on A's clock
mode        str     raw | event | cascade | privacy
w, h        int     source resolution in pixels
jpeg_q      int     encoder quality actually used for this frame
motion      float   normalized motion score of the frame, 0..1
a_proc_ms   float   milliseconds A spent on this frame (preprocess + encode)
extra       dict    mode-specific payload, e.g. cascade: {"persons": [[x1,y1,x2,y2,conf], ...]}

Design notes
------------
* Timestamps from A are used for *intervals* (effective FPS, jitter), never
  compared against B's clock directly: the two hosts are not assumed to be
  NTP-synced. End-to-end latency is measured per-frame on B as
  now(B) - ts_receive(B) plus the transport hop, and calibrated once via
  the ROUNDTRIP probe on the control channel.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

TOPIC_FRAME = "frame"
TOPIC_CONTROL = "ctl"  # B -> A commands (mode switch, throttle), JSON only

VALID_MODES = ("raw", "event", "cascade", "privacy")
VALID_CMDS = ("set_mode", "set_rate_limit", "ping")


@dataclass
class FrameHeader:
    seq: int
    ts_capture: float
    mode: str
    w: int
    h: int
    jpeg_q: int = 75
    motion: float = 0.0
    a_proc_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"unknown mode {self.mode!r}, expected one of {VALID_MODES}")
        if not isinstance(self.extra, dict):
            raise ValueError(f"extra must be a dict, got {type(self.extra).__name__}")


def encode_frame(header: FrameHeader, payload: bytes = b"") -> list[bytes]:
    """Build the 3-part ZMQ message to publish on TOPIC_FRAME."""
    return [
        TOPIC_FRAME.encode("utf-8"),
        json.dumps(asdict(header), separators=(",", ":")).encode("utf-8"),
        payload,
    ]


def decode_frame(parts: list[bytes]) -> tuple[FrameHeader, bytes]:
    """Parse a received multipart message. Raises on malformed input."""
    if len(parts) != 3:
        raise ValueError(f"expected 3 parts, got {len(parts)}")
    topic, raw_header, payload = parts
    if topic.decode("utf-8", "replace") != TOPIC_FRAME:
        raise ValueError(f"unexpected topic {topic!r}")
    header = FrameHeader(**json.loads(raw_header.decode("utf-8")))
    return header, payload


# ---------------------------------------------------------------- control --

@dataclass
class ControlMessage:
    """B -> A commands carried on TOPIC_CONTROL (small JSON, no payload part)."""
    cmd: str                       # "set_mode" | "set_rate_limit" | "ping"
    arg: Optional[dict] = None
    ts: float = field(default_factory=time.time)


def encode_control(msg: ControlMessage) -> bytes:
    return json.dumps(asdict(msg), separators=(",", ":")).encode("utf-8")


def decode_control(raw: bytes) -> ControlMessage:
    msg = ControlMessage(**json.loads(raw.decode("utf-8")))
    if msg.cmd not in VALID_CMDS:
        raise ValueError(f"unknown cmd {msg.cmd!r}, expected one of {VALID_CMDS}")
    return msg

import pytest

from common.protocol import (ControlMessage, FrameHeader, decode_control,
                             decode_frame, encode_control, encode_frame)


def test_frame_roundtrip():
    h = FrameHeader(seq=7, ts_capture=123.456, mode="event", w=1280, h=720,
                    jpeg_q=62, motion=0.31, a_proc_ms=9.5,
                    extra={"persons": [[10, 20, 300, 400, 0.9]]})
    parts = encode_frame(h, b"\xff\xd8fakejpg")
    assert parts[0] == b"frame"
    h2, payload = decode_frame(parts)
    assert h2.seq == 7 and h2.mode == "event" and h2.w == 1280
    assert h2.extra["persons"][0][4] == pytest.approx(0.9)
    assert payload == b"\xff\xd8fakejpg"


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        FrameHeader(seq=1, ts_capture=0.0, mode="turbo", w=1, h=1)


def test_decode_bad_part_count():
    with pytest.raises(ValueError):
        decode_frame([b"frame", b"{}"])


def test_control_roundtrip():
    raw = encode_control(ControlMessage(cmd="set_mode", arg={"mode": "cascade"}))
    msg = decode_control(raw)
    assert msg.cmd == "set_mode" and msg.arg["mode"] == "cascade"

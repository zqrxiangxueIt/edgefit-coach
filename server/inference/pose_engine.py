"""YOLO-pose ONNX engine: letterbox preprocess, decode + NMS postprocess.

Expects models exported from Ultralytics with a static 640x640 input and the
standard pose head output [1, 56, N] (4 box + 1 score + 17*3 keypoints).
Verified compatible with both yolo11n-pose and yolo26n-pose exports
(cross-checked against ultralytics' own predict output, review B); the
shape guard in detect() rejects any future export that changes the layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .backends import create_session

NUM_KPS = 17
STRIDE = 56  # 4 box + 1 score + 51 kpts


@dataclass
class PoseResult:
    box: np.ndarray            # xyxy in source-image pixels
    score: float
    keypoints: np.ndarray      # (17, 3): x, y in source pixels, conf


def letterbox(img: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect, pad to (size, size). Returns (img, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, (float(pad_x), float(pad_y))


class PoseEngine:
    def __init__(self, model_path: str, backend: str = "dml",
                 conf_thres: float = 0.35, iou_thres: float = 0.65) -> None:
        self.sess = create_session(model_path, backend)
        self.input_name = self.sess.get_inputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, img_bgr: np.ndarray) -> list[PoseResult]:
        x, scale, (pad_x, pad_y) = letterbox(img_bgr)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]                       # (1, 3, 640, 640)
        out = self.sess.run(None, {self.input_name: x})[0]   # (1, 56, N)
        if out.ndim != 3 or out.shape[1] != STRIDE:
            raise ValueError(
                f"unexpected pose head layout {tuple(out.shape)}, "
                f"expected (1, {STRIDE}, N) -- wrong or incompatible export?")
        return self._postprocess(out[0], scale, pad_x, pad_y)

    # ------------------------------------------------------------ internals --

    def _postprocess(self, pred: np.ndarray, scale: float,
                     pad_x: float, pad_y: float) -> list[PoseResult]:
        # pred arrives as (56, N): 4 box + 1 score + 51 kpts per anchor.
        pred = pred[:, pred[4] > self.conf_thres].T          # (M, 56)
        if pred.shape[0] == 0:
            return []
        boxes = self._cxcywh_to_xyxy(pred[:, :4])
        keep = self._nms(boxes, pred[:, 4], self.iou_thres)
        results = []
        for i in keep:
            kps = pred[i, 5:].reshape(NUM_KPS, 3).copy()
            kps[:, 0] = (kps[:, 0] - pad_x) / scale          # back to source pixels
            kps[:, 1] = (kps[:, 1] - pad_y) / scale
            box = boxes[i].copy()
            box[[0, 2]] = (box[[0, 2]] - pad_x) / scale
            box[[1, 3]] = (box[[1, 3]] - pad_y) / scale
            results.append(PoseResult(box=box, score=float(pred[i, 4]), keypoints=kps))
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
        out = np.empty_like(b)
        out[:, 0] = b[:, 0] - b[:, 2] / 2
        out[:, 1] = b[:, 1] - b[:, 3] / 2
        out[:, 2] = b[:, 0] + b[:, 2] / 2
        out[:, 3] = b[:, 1] + b[:, 3] / 2
        return out

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size:
            i = order[0]
            keep.append(int(i))
            rest = order[1:]
            if rest.size == 0:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                     + (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
                     - inter)
            order = rest[inter / np.maximum(union, 1e-9) <= iou_thres]
        return keep

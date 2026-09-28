"""ONNX pose models run through the OpenCV 5 DNN engine (`cv2.dnn`).

RTMPoseEstimator = RTMDet-nano person detector + RTMPose-s (SimCC) top-down landmarks,
both from OpenMMLab (Apache-2.0), COCO-17 keypoints. Preprocessing constants come from the
models' mmdeploy `pipeline.json`. The detector is the NMS-free cut of the official export
(see scripts/download_models.py); NMS runs in `cv2.dnn.NMSBoxes`.

Inference stays inside OpenCV: only the native engines (new / classic) are allowed, not the
ONNX Runtime engine, so the competition's "OpenCV 5 performs the analysis" rule holds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from kinevra.pose.base import DnnPoseEstimator, Keypoints, PoseResult
from kinevra.vision.types import Image

Engine = Literal["new", "classic", "auto"]
ENGINES: dict[str, int] = {
    "new": cv2.dnn.ENGINE_NEW,
    "classic": cv2.dnn.ENGINE_CLASSIC,
    "auto": cv2.dnn.ENGINE_AUTO,
}

RTMDET_FILE = "rtmdet_nano_person_raw.onnx"
RTMPOSE_FILE = "rtmpose_s_body7_256x192.onnx"

Boxes = NDArray[np.float32]  # (N, 5): x0, y0, x1, y1, score in image pixels


def load_net(path: Path, engine: Engine = "new") -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"model not found: {path}. Run `uv run python scripts/download_models.py`."
        )
    return cv2.dnn.readNetFromONNX(str(path), ENGINES[engine])


def forward(net: Any, blob: NDArray[np.float32]) -> dict[str, NDArray[np.float32]]:
    """Run the net and return outputs keyed by name (engines order outputs differently)."""
    names = list(net.getUnconnectedOutLayersNames())
    net.setInput(blob)
    outs = net.forward(names)
    return {n: np.asarray(o) for n, o in zip(names, outs, strict=True)}


def make_blob(
    image: Image, mean: tuple[float, float, float], std: tuple[float, float, float], swap_rb: bool
) -> NDArray[np.float32]:
    """NCHW float blob: (image[swapped] - mean) / std, using OpenCV's blob API."""
    params = cv2.dnn.Image2BlobParams()
    params.scalefactor = (1 / std[0], 1 / std[1], 1 / std[2])
    params.mean = mean
    params.swapRB = swap_rb
    params.ddepth = cv2.CV_32F
    params.size = (image.shape[1], image.shape[0])
    return np.asarray(cv2.dnn.blobFromImageWithParams(image, params), dtype=np.float32)


# --- RTMDet (person detector) --------------------------------------------------------------


def decode_rtmdet(
    raw_boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    ratio: float,
    image_size: tuple[int, int],
    score_threshold: float,
    nms_threshold: float,
) -> Boxes:
    """Raw RTMDet output → person boxes in image px after threshold + OpenCV NMS, by score.

    raw_boxes: (A, 4) decoded x0, y0, x1, y1 in model-input px; scores: (A,) person scores.
    """
    w, h = image_size
    keep = scores.reshape(-1) >= score_threshold
    if not keep.any():
        return np.zeros((0, 5), dtype=np.float32)
    boxes = np.column_stack([raw_boxes[keep], scores.reshape(-1)[keep]]).astype(np.float32)
    boxes[:, :4] /= ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    kept = cv2.dnn.NMSBoxes(xywh, boxes[:, 4].tolist(), score_threshold, nms_threshold)
    out: Boxes = boxes[np.asarray(kept, dtype=np.int64).reshape(-1)]
    return out[np.argsort(-out[:, 4])]


class RTMDetPerson:
    INPUT = 320
    MEAN = (103.53, 116.28, 123.675)  # BGR
    STD = (57.375, 57.12, 58.395)

    def __init__(
        self,
        path: Path,
        engine: Engine = "new",
        score_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ) -> None:
        self.net = load_net(path, engine)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def detect(self, image: Image) -> Boxes:
        h, w = image.shape[:2]
        ratio = min(self.INPUT / h, self.INPUT / w)
        rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
        resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            0,
            self.INPUT - rh,
            0,
            self.INPUT - rw,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        blob = make_blob(np.asarray(padded, dtype=np.uint8), self.MEAN, self.STD, swap_rb=False)
        out = forward(self.net, blob)
        return decode_rtmdet(
            out["1442"][0],
            out["1415"][0, :, 0],
            ratio,
            (w, h),
            self.score_threshold,
            self.nms_threshold,
        )


# --- RTMPose (top-down SimCC) --------------------------------------------------------------


def box_to_center_size(
    box: tuple[float, float, float, float], padding: float, aspect: float
) -> tuple[float, float, float, float]:
    """(cx, cy, w, h) of the padded box, widened or heightened to `aspect` = w / h."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(x1 - x0, 1.0) * padding, max(y1 - y0, 1.0) * padding
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    return cx, cy, w, h


def crop_affine(
    cx: float, cy: float, w: float, h: float, out_w: int, out_h: int
) -> NDArray[np.float64]:
    """2x3 affine mapping the box (cx, cy, w, h) in the image onto an out_w x out_h input."""
    sx, sy = out_w / w, out_h / h
    return np.array([[sx, 0.0, -(cx - w / 2) * sx], [0.0, sy, -(cy - h / 2) * sy]])


def decode_simcc(
    simcc_x: NDArray[np.float32], simcc_y: NDArray[np.float32], split_ratio: float
) -> Keypoints:
    """(K, Wx), (K, Wy) SimCC logits → (K, 3): x, y in model-input px and score.

    Score follows mmpose `get_simcc_maximum`: the smaller of the two axis maxima.
    """
    x = simcc_x.argmax(axis=1).astype(np.float32) / split_ratio
    y = simcc_y.argmax(axis=1).astype(np.float32) / split_ratio
    score = np.minimum(simcc_x.max(axis=1), simcc_y.max(axis=1)).astype(np.float32)
    return np.stack([x, y, score], axis=1)


def apply_inverse_affine(points: Keypoints, affine: NDArray[np.float64]) -> Keypoints:
    """Map (K, 3) points from model-input px back to image px (score unchanged)."""
    inv = cv2.invertAffineTransform(affine)
    xy = points[:, :2].astype(np.float64) @ inv[:, :2].T + inv[:, 2]
    return np.column_stack([xy, points[:, 2]]).astype(np.float32)


class RTMPoseEstimator(DnnPoseEstimator):
    name = "rtmpose-s"
    INPUT_W, INPUT_H = 192, 256
    SPLIT_RATIO = 2.0
    MEAN = (123.675, 116.28, 103.53)  # RGB
    STD = (58.395, 57.12, 57.375)

    def __init__(
        self,
        model_dir: Path,
        *,
        engine: Engine = "new",
        det_score_threshold: float = 0.5,
        det_nms_threshold: float = 0.45,
        bbox_padding: float = 1.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.detector = RTMDetPerson(
            model_dir / RTMDET_FILE, engine, det_score_threshold, det_nms_threshold
        )
        self.net = load_net(model_dir / RTMPOSE_FILE, engine)
        self.bbox_padding = bbox_padding

    def _landmarks(
        self, image: Image, box: tuple[float, float, float, float], padding: float
    ) -> Keypoints:
        cx, cy, w, h = box_to_center_size(box, padding, self.INPUT_W / self.INPUT_H)
        affine = crop_affine(cx, cy, w, h, self.INPUT_W, self.INPUT_H)
        warped = cv2.warpAffine(
            image,
            affine,
            (self.INPUT_W, self.INPUT_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        out = forward(self.net, make_blob(np.asarray(warped), self.MEAN, self.STD, swap_rb=True))
        kps = decode_simcc(out["simcc_x"][0], out["simcc_y"][0], self.SPLIT_RATIO)
        return apply_inverse_affine(kps, affine)

    def infer(self, image: Image) -> PoseResult:
        boxes = self.detector.detect(image)
        if len(boxes) == 0:
            return PoseResult(None, 0)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x0, y0, x1, y1 = (float(v) for v in boxes[int(areas.argmax()), :4])
        box = (x0, y0, x1, y1)
        return PoseResult(self._landmarks(image, box, self.bbox_padding), len(boxes), box)

    def infer_roi(self, crop: Image) -> PoseResult:
        # Top-down model: the ROI itself is the box, so skip the detector and use every
        # input pixel for the region of interest (this is the "higher resolution" mode).
        h, w = crop.shape[:2]
        box = (0.0, 0.0, float(w), float(h))
        return PoseResult(self._landmarks(crop, box, 1.0), 1, box)

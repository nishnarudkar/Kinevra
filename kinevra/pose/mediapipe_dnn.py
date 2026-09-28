"""MediaPipe person detector + BlazePose landmarks (OpenCV Zoo ONNX, Apache-2.0) in cv2.dnn.

Adapted from opencv_zoo `mp_persondet.py` / `mp_pose.py` (Apache-2.0). Differences: anchors
are generated instead of hard-coded, and the rotated crop is a single `cv2.warpAffine`
(rotate + scale about the mid-hip) instead of crop → pad → rotate → resize.
BlazePose's 33 landmarks are mapped to COCO-17 names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from kinevra.pose.base import DnnPoseEstimator, Keypoints, PoseResult
from kinevra.pose.opencv_dnn import Engine, forward, load_net
from kinevra.vision.types import Image

PERSONDET_FILE = "person_detection_mediapipe_2023mar.onnx"
POSE_FILE = "pose_estimation_mediapipe_2023mar.onnx"

# BlazePose index for each COCO-17 landmark, in COCO17 order.
BLAZEPOSE_TO_COCO17 = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def generate_anchors(
    input_size: int = 224, strides: tuple[int, ...] = (8, 16, 32, 32, 32)
) -> NDArray[np.float32]:
    """SSD anchor centres (normalised) for the MediaPipe pose detector: 2 per layer/cell."""
    out: list[NDArray[np.float64]] = []
    i = 0
    while i < len(strides):
        stride, per_cell = strides[i], 0
        while i < len(strides) and strides[i] == stride:
            per_cell += 2
            i += 1
        fm = int(np.ceil(input_size / stride))
        ys, xs = np.meshgrid(np.arange(fm), np.arange(fm), indexing="ij")
        centres = np.stack([(xs + 0.5) / fm, (ys + 0.5) / fm], axis=-1).reshape(-1, 2)
        out.append(np.repeat(centres, per_cell, axis=0))
    return np.concatenate(out).astype(np.float32)


def _sigmoid(x: NDArray[Any]) -> NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -100, 100))))


class MPPersonDetector:
    INPUT = 224

    def __init__(
        self,
        path: Path,
        engine: Engine = "new",
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ) -> None:
        self.net = load_net(path, engine)
        self.anchors = generate_anchors(self.INPUT)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def detect(self, image: Image) -> NDArray[np.float64]:
        """(N, 13): face box x0,y0,x1,y1 · 4 keypoints (mid-hip, full-body, mid-shoulder,
        upper-body) as x,y · score. Image pixels."""
        h, w = image.shape[:2]
        ratio = min(self.INPUT / h, self.INPUT / w)
        rw, rh = max(1, int(w * ratio)), max(1, int(h * ratio))
        rgb = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.float32) / 127.5 - 1.0
        rgb = np.asarray(cv2.resize(rgb, (rw, rh)), dtype=np.float32)
        left, top = (self.INPUT - rw) // 2, (self.INPUT - rh) // 2
        padded = cv2.copyMakeBorder(
            rgb,
            top,
            self.INPUT - rh - top,
            left,
            self.INPUT - rw - left,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)
        out = forward(self.net, blob)
        regressors = next(v for v in out.values() if v.shape[-1] == 12)[0]
        scores = _sigmoid(next(v for v in out.values() if v.shape[-1] == 1)[0, :, 0])
        pad = np.array([left, top], dtype=np.float64) / ratio
        return self.decode(regressors, scores, max(w, h), pad)

    def decode(
        self,
        regressors: NDArray[np.float32],
        scores: NDArray[np.float64],
        scale: float,
        pad: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        anchors = self.anchors.astype(np.float64)
        reg = regressors.astype(np.float64) / self.INPUT
        cxy, wh = reg[:, :2] + anchors, reg[:, 2:4]
        boxes = np.concatenate([cxy - wh / 2, cxy + wh / 2], axis=1) * scale
        boxes -= np.tile(pad, 2)
        xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes.tolist()]
        keep = np.asarray(
            cv2.dnn.NMSBoxes(xywh, scores.tolist(), self.score_threshold, self.nms_threshold),
            dtype=np.int64,
        ).reshape(-1)
        if keep.size == 0:
            return np.zeros((0, 13))
        kps = (reg[keep, 4:12].reshape(-1, 4, 2) + anchors[keep, None, :]) * scale - pad
        return np.column_stack([boxes[keep], kps.reshape(-1, 8), scores[keep]])


def person_affine(
    mid_hip: NDArray[np.float64], full_body: NDArray[np.float64], out_size: int
) -> NDArray[np.float64]:
    """Rotate so mid-hip → full-body points up, scale a 2·dist square around mid-hip."""
    dist = float(np.linalg.norm(full_body - mid_hip))
    radians = np.pi / 2 - np.arctan2(-(full_body[1] - mid_hip[1]), full_body[0] - mid_hip[0])
    radians -= 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
    centre = (float(mid_hip[0]), float(mid_hip[1]))
    rot = cv2.getRotationMatrix2D(centre, float(np.degrees(radians)), out_size / (2 * dist))
    affine = np.asarray(rot, dtype=np.float64)
    affine[:, 2] += np.array([out_size / 2, out_size / 2]) - mid_hip
    return affine


class MediaPipePoseEstimator(DnnPoseEstimator):
    name = "mediapipe-blazepose"
    INPUT = 256

    def __init__(
        self,
        model_dir: Path,
        *,
        engine: Engine = "new",
        det_score_threshold: float = 0.5,
        det_nms_threshold: float = 0.3,
        pose_conf_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.detector = MPPersonDetector(
            model_dir / PERSONDET_FILE, engine, det_score_threshold, det_nms_threshold
        )
        self.net = load_net(model_dir / POSE_FILE, engine)
        self.pose_conf_threshold = pose_conf_threshold

    def _landmarks(self, image: Image, det: NDArray[np.float64]) -> Keypoints | None:
        kps = det[4:12].reshape(4, 2)
        if np.linalg.norm(kps[1] - kps[0]) < 1.0:
            return None
        affine = person_affine(kps[0], kps[1], self.INPUT)
        warped = cv2.warpAffine(image, affine, (self.INPUT, self.INPUT), flags=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = forward(self.net, np.ascontiguousarray(rgb[np.newaxis]))  # NHWC
        landmarks = next(v for v in out.values() if v.shape == (1, 195))[0].reshape(39, 5)
        conf = float(next(v for v in out.values() if v.shape == (1, 1))[0, 0])
        if conf < self.pose_conf_threshold:
            return None
        inv = cv2.invertAffineTransform(affine)
        xy = landmarks[:, :2].astype(np.float64) @ inv[:, :2].T + inv[:, 2]
        visibility = _sigmoid(landmarks[:, 3])
        coco = list(BLAZEPOSE_TO_COCO17)
        return np.column_stack([xy[coco], visibility[coco]]).astype(np.float32)

    def infer(self, image: Image) -> PoseResult:
        dets = self.detector.detect(image)
        if len(dets) == 0:
            return PoseResult(None, 0)
        # Largest person = longest mid-hip → full-body distance (the box is a face box).
        sizes = np.linalg.norm(dets[:, 6:8] - dets[:, 4:6], axis=1)
        det = dets[int(sizes.argmax())]
        kps = self._landmarks(image, det)
        box = None
        if kps is not None:
            box = (
                float(kps[:, 0].min()),
                float(kps[:, 1].min()),
                float(kps[:, 0].max()),
                float(kps[:, 1].max()),
            )
        return PoseResult(kps, len(dets), box)

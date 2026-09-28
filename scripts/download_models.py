"""Download the pose models into models/ and verify SHA-256 checksums.

Used locally, in CI (optional model tests) and in the Lambda Docker build.

    uv run python scripts/download_models.py              # all models
    uv run python scripts/download_models.py --only rtmpose

RTMDet's official end2end export ends in TopK + NonMaxSuppression with a data-dependent
output size; OpenCV 5's DNN engine keeps the first run's shapes, so later frames fail with an
out-of-range Gather. We therefore derive `rtmdet_nano_person_raw.onnx`: the same graph cut
just before that step (decoded boxes + scores, fixed shape), and do NMS with
`cv2.dnn.NMSBoxes`. Needs the `onnx` package (dev dependency).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelFile:
    group: str
    filename: str
    url: str
    sha256: str
    zip_member: str | None = None  # file inside the zip to extract


@dataclass(frozen=True)
class DerivedModel:
    source: str  # filename in models/
    filename: str
    outputs: tuple[str, ...]  # graph tensors that become the new outputs
    sha256: str


OPENMMLAB = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS = (
    ModelFile(
        "rtmpose",
        "rtmdet_nano_person.onnx",
        f"{OPENMMLAB}/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.zip",
        "8297e829ccc5590c8e2d32d5a211f322a0585fb7467eec85eb12c9525b0b95d6",
        "end2end.onnx",
    ),
    ModelFile(
        "rtmpose",
        "rtmpose_s_body7_256x192.onnx",
        f"{OPENMMLAB}/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
        "9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d",
        "end2end.onnx",
    ),
    ModelFile(
        "mediapipe",
        "person_detection_mediapipe_2023mar.onnx",
        f"{ZOO}/person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx",
        "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f",
    ),
    ModelFile(
        "mediapipe",
        "pose_estimation_mediapipe_2023mar.onnx",
        f"{ZOO}/pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx",
        "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f",
    ),
)


DERIVED = (
    DerivedModel(
        "rtmdet_nano_person.onnx",
        "rtmdet_nano_person_raw.onnx",
        ("1442", "1415"),  # decoded boxes (1, 2100, 4) xyxy, scores (1, 2100, 1)
        "d770bb2255b5c0072cd6d3f8e45658b84f6969f3d1b11f556fbc92284a94e0f8",
    ),
)


def derive(model: DerivedModel, dest: Path) -> bool:
    import onnx
    import onnx.utils

    source, target = dest / model.source, dest / model.filename
    if target.is_file() and sha256(target.read_bytes()) == model.sha256:
        print(f"ok       {model.filename}")
        return True
    if not source.is_file():
        print(f"skip     {model.filename} (source {model.source} missing)", file=sys.stderr)
        return False
    onnx.utils.extract_model(str(source), str(target), ["input"], list(model.outputs))
    onnx.checker.check_model(str(target))
    digest = sha256(target.read_bytes())
    if digest != model.sha256:
        target.unlink()
        print(f"CHECKSUM MISMATCH {model.filename}: {digest}", file=sys.stderr)
        return False
    print(f"derived  {target}")
    return True


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(model: ModelFile) -> bytes:
    with urllib.request.urlopen(model.url, timeout=120) as resp:  # fixed https URLs only
        payload: bytes = resp.read()
    if model.zip_member is None:
        return payload
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        member = next(n for n in zf.namelist() if n.endswith("/" + model.zip_member))
        return zf.read(member)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", type=Path, default=ROOT / "models")
    p.add_argument("--only", choices=sorted({m.group for m in MODELS}))
    args = p.parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)

    failed = False
    for model in MODELS:
        if args.only and model.group != args.only:
            continue
        target = args.dest / model.filename
        if target.is_file() and sha256(target.read_bytes()) == model.sha256:
            print(f"ok       {model.filename}")
            continue
        print(f"download {model.filename} ...", flush=True)
        data = fetch(model)
        digest = sha256(data)
        if digest != model.sha256:
            print(f"CHECKSUM MISMATCH {model.filename}: {digest}", file=sys.stderr)
            failed = True
            continue
        target.write_bytes(data)
        print(f"saved    {target} ({len(data) / 1e6:.1f} MB)")
    if args.only in (None, "rtmpose"):
        for derived in DERIVED:
            failed |= not derive(derived, args.dest)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

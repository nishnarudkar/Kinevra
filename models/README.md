# Models

ONNX weights are not committed (`*.onnx` is git-ignored). Fetch and verify them with:

```bash
uv run python scripts/download_models.py            # all models (~45 MB)
uv run python scripts/download_models.py --only rtmpose
```

The script checks every file's SHA-256 and derives the NMS-free detector (see ADR 0002).

| File | Used by | Source | Licence | SHA-256 |
|---|---|---|---|---|
| `rtmdet_nano_person.onnx` | source for the derived file below | [OpenMMLab RTMPose ONNX SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.zip) (`end2end.onnx`) | Apache-2.0 | `8297e829…b95d6` |
| `rtmdet_nano_person_raw.onnx` | **default detector** | derived: graph cut before TopK/NMS (outputs `1442` boxes, `1415` scores) | Apache-2.0 | `d770bb22…4e0f8` |
| `rtmpose_s_body7_256x192.onnx` | **default pose model** | [OpenMMLab RTMPose ONNX SDK](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip) (`end2end.onnx`) | Apache-2.0 (trained on body7 datasets; see ADR 0002) | `9aeb635b…aef47d` |
| `person_detection_mediapipe_2023mar.onnx` | benchmark baseline | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/person_detection_mediapipe) | Apache-2.0 | `47fd5599…a701f` |
| `pose_estimation_mediapipe_2023mar.onnx` | benchmark baseline | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/pose_estimation_mediapipe) | Apache-2.0 | `9d89c599…2f3f` |

Full checksums are in `scripts/download_models.py`.

# ADR 0002 — Pose model

- **Status:** proposed (preliminary numbers; finalise after benchmarking the recorded clips)
- **Date:** 2026-09-28

## Context

Pose inference must run inside OpenCV 5's DNN engine (`cv2.dnn`) to keep OpenCV at the core of
the analysis (PROJECT.md §0.2, §4), on an x86 laptop (live mode) and on arm64 Lambda (clip
mode). We need COCO-style landmarks with a per-landmark confidence, a person count (for the
`multiple_people` flag), a permissive licence, and support for the agent's ROI re-analysis
tool (Phase 5).

## Candidates tested

| | A: RTMDet-nano + RTMPose-s | B: MediaPipe person detector + BlazePose |
|---|---|---|
| Source | OpenMMLab mmdeploy ONNX SDK | OpenCV Zoo (2023mar) |
| Licence | Apache-2.0 (weights trained on "body7": COCO, AI Challenger, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18 — some of those datasets have research-oriented terms; fine for this non-commercial prototype, revisit before any commercial use) | Apache-2.0 |
| Size | 4.0 MB + 21.9 MB | 12.0 MB + 5.6 MB |
| Output | COCO-17, SimCC (x/y logits) | 33 BlazePose landmarks (+ visibility) → mapped to COCO-17 |
| Loads in `cv2.dnn` | ✅ new engine (detector ❌ classic engine) | ✅ new + classic |

## Findings (preliminary, x86 Windows laptop, OpenCV 5.0.0, `ENGINE_NEW`)

Test image: OpenCV sample `messi5.jpg` (frontal, dynamic pose), repeated 40x, plus synthetic
"far subject" (image scaled to 25–35 % inside a 640x480 frame) and "dark" (18 % brightness)
variants. Numbers from `python -m kinevra.pose.benchmark` and ad-hoc checks.

| Metric | A: RTMPose-s | B: MediaPipe |
|---|---|---|
| Latency per frame (detector + pose), p50 / p95 | ~26–30 / 34 ms | 47–175 / 165–230 ms |
| Live pipeline FPS (640x400 clip, capture + quality + pose + HUD) | **33.4 FPS** | not run |
| Person found on the full-size test image | ✅ score 0.83 | ❌ detector max score 0.496 (< 0.5) |
| Landmarks correct on test image | ✅ incl. left/right | ❌ even with threshold 0.3: body circle too small for a wide pose, arms cut off; also wrong with the unmodified OpenCV Zoo code, so a model limitation, not our decoding |
| Far subject (35 %/25 % scale) full-frame | ❌ detector misses the person | – |
| **Far subject, ROI re-analysis seeded from previous landmarks** | ✅ conf **0.00 → 0.85 / 0.82**, error ≈ 1 px | – |
| Well-framed subject, ROI vs full frame | ROI conf 0.82–0.87 vs 0.96 (no gain) | – |

Engineering issues found and fixed:

1. **RTMDet end2end export fails in OpenCV 5's new engine after the first frame.** Its graph ends in
   TopK + NonMaxSuppression, whose output size depends on the data; the new engine keeps the first
   run's shapes, so the next different frame throws an out-of-range Gather. Fix:
   `scripts/download_models.py` derives `rtmdet_nano_person_raw.onnx` (graph cut before NMS: fixed-shape
   boxes + scores, checksum-verified) and NMS runs in `cv2.dnn.NMSBoxes`. Regression test:
   `tests/integration/test_pose_models.py::test_rtmdet_handles_changing_frames`.
2. **Engines order outputs differently** (the OpenCV Zoo MediaPipe demo breaks on `ENGINE_NEW` for this
   reason). We always read outputs by name.
3. `ENGINE_ORT` (ONNX Runtime) exists in OpenCV 5 but is not allowed in config, so inference is
   always OpenCV's own DNN.

## Decision

Use **A: RTMDet-nano (NMS-free cut) + RTMPose-s** via `cv2.dnn` `ENGINE_NEW` as the default
(`pose.model: rtmpose`). Keep B implemented as a benchmark baseline only.

ROI re-analysis for a top-down model means: crop a padded box (padding 0.6) around the
exercising side's hip/shoulder/elbow/wrist, seeded from the last frame where they were visible,
and run RTMPose directly on it (no detector). Its value is **recovering the subject when the
full-frame pass fails** (small/far subject, detector miss) and giving a second, independent
measurement to confirm or dismiss a deviation, not raising confidence on well-framed frames.
Upsample factor barely matters (RTMPose resizes the crop to 192x256 anyway); CLAHE has a small
effect (±0.01).

Visibility = SimCC score (min of the x/y maxima) clipped to 0..1, and 0 for points outside the
frame. The default `visibility_threshold: 0.5` works on the test image; confirm on real clips.

## To finalise (Phase 2 acceptance, needs the webcam and recorded clips)

- [ ] `uv run python -m kinevra.pose.benchmark --roi data/clips/*.mp4` on your clips (still pose,
      normal session, 3.5 m distance); paste the table here.
- [ ] Left/right check: raise only the right arm, `right_elbow y` changes in the HUD.
- [ ] Record the arm64 Lambda numbers in Phase 7.

## Consequences

- Needs `onnx` (dev dependency) at model-preparation time, including in the Lambda image build.
- Two-stage top-down pipeline, ~26 ms per frame on the laptop; the detector could run every N
  frames with pose-derived boxes in between if Graviton latency requires it.
- Weights' training-data terms must be revisited before any commercial use.

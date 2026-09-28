# ADR 0001 — Overall architecture

- **Status:** accepted
- **Date:** 2026-09-28

## Context

Kinevra must (a) use OpenCV 5 for substantive video analysis, (b) run a meaningful component
on AWS, and (c) qualify for the Agentic Vision path: an agent whose later decisions depend on
new OpenCV 5 measurements it chose to request (PROJECT.md §0). The build window is 29 days
with one developer.

## Decision

1. **One Python package (`kinevra`) shared by two modes.** Live mode (local webcam) and cloud
   clip mode (AWS Lambda) call the same vision, movement, rules and agent code, so what is
   tested locally is what runs in the cloud.
2. **Deterministic measurement, LLM reasoning.** OpenCV 5 + NumPy/SciPy compute every number
   (angles, ROM, reps, quality). The LLM (Amazon Bedrock via LangGraph) only chooses actions
   and tools and writes short feedback; it never computes metrics.
3. **Pose via `cv2.dnn` on an ONNX model**, keeping OpenCV 5 in the core inference path.
   Model choice is ADR 0002.
4. **Agent loop with OpenCV tools.** Deterministic triage skips the LLM for normal reps; for
   unusual reps the LLM may call OpenCV 5 tools (ROI re-analysis, optical flow, camera-setup
   check, evidence snapshots). Guardrails enforce allowed actions, uncertainty handling,
   forced human-review escalation and non-diagnostic language. Every step is traced.
5. **Pydantic schemas (`kinevra/schemas.py`) are the only cross-module contract.**
6. **AWS serverless deployment:** S3 upload → Analysis Lambda (container, arm64/Graviton,
   `opencv-python-headless` 5.x) → DynamoDB; API Gateway + FastAPI/Mangum; React on
   S3 + CloudFront; CloudWatch; all defined in AWS CDK (Python). Fallback to ECS Fargate
   arm64 if Lambda limits are hit (ADR 0003).
7. **Reproducibility:** Python 3.12, `uv` with `uv.lock`; OpenCV pinned to `5.0.0.93`, with
   `gui` / `headless` extras declared as conflicting so each environment installs exactly
   one OpenCV wheel.

## Consequences

- Business logic is testable without a camera, AWS or an LLM (`FakeLLM`, synthetic fixtures).
- Lambda cold start and 15-minute limit bound clip length (max ~90 s, downsampled FPS).
- Two OpenCV wheel variants must be kept in sync; the lockfile pins both.

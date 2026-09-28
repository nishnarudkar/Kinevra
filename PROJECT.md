# PROJECT.md — Kinevra

> **Kinevra** — Agentic Computer Vision for Rehabilitation Movement Monitoring
> *It watches the movement, so you can focus on recovery.*
>
> Team: Anonymous · Developer: Nishant Narudkar
> Competition: **OpenCV AI Competition 2026, powered by AWS** · Featured path: **Agentic Vision**
> Stack: Python, OpenCV 5, pose estimation (ONNX via OpenCV 5 DNN), LangGraph, Amazon Bedrock, AWS (Lambda on Graviton, S3, DynamoDB, API Gateway, CloudWatch), React + TypeScript

This file is the **single source of truth for building Kinevra with Claude**. It defines what to build, in what order, how each piece is verified, what the competition requires, and the rules Claude must follow. Keep it in the repository root and update the Progress Log (Section 20) at the end of every working session.

---

## 0. Competition constraints (read first — these override everything else)

### 0.1 Key dates

| Date (IST unless noted) | Event |
|---|---|
| Mon 28 Sep 2026 | Plan finalised (today) |
| Sep 21 – Oct 2 | Grant check-in window (only if you received the AWS compute grant — 30-min Zoom, progress update, unlocks the other 50% of the grant) |
| **Sat 24 Oct** | **Internal code freeze** — only fixes, docs and video after this |
| **Sun 25 Oct, evening** | **Target submission** (one day of buffer) |
| Oct 26, 11:59 PM Pacific = **Tue 27 Oct, ~12:15–12:30 PM IST** | **Hard deadline** (Devpost shows 27 Oct 12:15 PM IST — treat that as final) |
| Oct 27 – Nov 9 | Judging — the web endpoint must stay up and working the whole time |
| Nov 10 | Winners announced |

> ⚠️ **Eligibility check:** the rules say the build phase is for teams that submitted the required proposal. If you have not submitted a proposal, confirm on Devpost / the official Slack today that you can still enter.

### 0.2 Mandatory requirements (entry is invalid without these)

- [ ] **OpenCV 5 performs substantive image/video analysis** — not just opening the webcam. In Kinevra: capture, preprocessing, quality analysis, pose inference through the OpenCV 5 DNN engine, ROI re-analysis, optical flow, and annotated evidence rendering.
- [ ] **A meaningful component runs on AWS.** In Kinevra, the **OpenCV 5 analysis pipeline itself runs in the cloud** (Lambda container on Graviton/arm64) together with the agent (Bedrock), storage and monitoring. It is not only a database behind a local app.

### 0.3 Agentic Vision path requirements (for the $1,000 award, and they strengthen the main-prize entry too)

- [ ] An agent uses **OpenCV 5 tools** inside a **multi-step perception → decision → action loop**.
- [ ] **Image/video results change what the system does next**: a later plan, tool call, action, or request for human approval. A chatbot explaining a fixed result is **not enough**.
- [ ] Evidence to submit:
  - an **agent workflow diagram** (perception, decision/orchestration, action);
  - a **trace or demo** showing OpenCV 5 output changing a later decision, tool call or action;
  - an **evaluation** of task success, failure handling, observability and appropriate human control.
- [ ] Note: *using Claude to write the code does not count as the agentic workflow.* The agent is the runtime LangGraph + Bedrock system.

### 0.4 Final submission checklist (all required)

- [ ] **Technical report**: problem, users, architecture, OpenCV 5 implementation, AWS deployment, evaluation, limitations, responsible-use considerations (`docs/REPORT.md`, exported to PDF).
- [ ] **Judge-accessible code repository** (private is fine if judges get access; it does not have to be open source).
- [ ] **Pinned dependencies** plus clear **build, deploy and test instructions**.
- [ ] **Architecture diagram** showing OpenCV 5 and AWS components and the agent components.
- [ ] **A working web endpoint** judges can use (primary), plus a live screen-share if requested.
- [ ] **A video of 5 minutes or less** (public or unlisted) showing **the team**, the app working, its architecture and its principal results.
- [ ] **Evaluation evidence**, including failure cases and limitations.
- [ ] Agentic Vision extras from 0.3.

### 0.5 Judging weights → where effort goes

| Criterion | Weight | What wins it in Kinevra |
|---|---|---|
| Technical execution (correctness and depth of OpenCV 5) | **30%** | OpenCV 5 DNN pose inference, correct angle math, rep state machine, ROI re-analysis, optical flow, quality checks, measured accuracy |
| Innovation | 20% | Agent that actively re-perceives (re-analyses segments, asks for camera adjustments) instead of a one-shot classifier |
| Real-world impact | 20% | Home rehab between physio visits; evidence (ROM error, rep accuracy); human-review workflow |
| User experience | 10% | Clean dashboard, clear feedback, accessible design, sample clips for judges |
| Documentation and presentation | 10% | Report, README, diagrams, 5-minute video |
| Cloud delivery, reproducibility, responsible operation | 10% | IaC (CDK), pinned deps, CloudWatch observability, least-privilege IAM, privacy and safety guardrails |

### 0.6 Optional: Best Use of COOL award (stretch goal only)

COOL (Cloud-Optimized OpenCV Library, on AWS Marketplace, optimized for Graviton) has a separate $1,000 award. To claim it, COOL must execute the core workload on Graviton, with a reproducible baseline comparison (latency, throughput, cost), the COOL version, and the instance/deployment configuration. **Only attempt this after the main entry is complete** (Phase 9). First check the Marketplace listing for which operations COOL accelerates and whether it fits a Lambda/container deployment. Do not claim COOL usage unless it is verified.

---

## 1. How to use this file with Claude

### 1.1 Working loop

1. Start every session: *"Read PROJECT.md and CLAUDE.md, then tell me the current phase and today's goal from the schedule."*
2. Work on **one phase at a time**. Each phase has a goal, tasks, acceptance criteria and a ready-to-paste prompt.
3. A phase is done only when every acceptance criterion passes. With the deadline, if a phase runs over by more than a day, cut scope using Section 18 rather than skipping tests.
4. End every session by asking Claude to update the Progress Log.
5. Commit after every completed task.

### 1.2 Rules for Claude (copy into `CLAUDE.md`)

```
- Read PROJECT.md first. Follow the current phase and the schedule in Section 17.
- Competition rules in Section 0 override everything. Never remove OpenCV 5 from the core
  analysis path or move the cloud OpenCV workload off AWS.
- Computer vision measures, the LLM reasons. Never ask the LLM to compute angles, ROM,
  counts or thresholds. All numbers come from deterministic Python/OpenCV code.
- Agent tools that "look again" must actually call OpenCV 5 (re-analysis, optical flow,
  quality checks, snapshot rendering). Every tool call is recorded in the decision trace.
- All data crossing module boundaries uses the Pydantic schemas in kinevra/schemas.py.
  Change a schema deliberately and update every consumer and test in the same change.
- Every new module gets unit tests. Run `make test` and `make lint` before calling a task done.
- No medical diagnosis language anywhere (code, prompts, UI, docs). Say "movement deviation".
- No real patient data. Only self-recorded or consenting-volunteer clips; never commit video.
- Secrets never go in code or git. Use env vars / AWS SSO profiles / GitHub OIDC.
- Pin every dependency (uv.lock, package-lock.json, Docker base image digest).
- OpenCV 5 changed APIs from 4.x. Check the official 5.x docs / migration guide instead of guessing.
- Prefer small, reviewable changes and short explanations of design decisions.
```

### 1.3 Core philosophy

- **Computer vision measures.**
- **The agent decides when to look again, what to say, and when a human must review.**
- **AWS runs the vision and the agent at scale.**
- **Humans remain responsible for healthcare decisions.**

---

## 2. Product definition

### 2.1 One-line description

**Kinevra is an agentic computer-vision system that uses OpenCV 5 to measure rehabilitation movements, turns them into structured movement evidence, and uses an AI agent that decides whether to keep monitoring, re-examine the video with OpenCV tools, ask for a better camera setup, give feedback, or request human review.**

### 2.2 Users

- **Primary:** people doing prescribed home exercises between physiotherapy sessions.
- **Secondary:** physiotherapists or caregivers who review flagged sessions (human-review queue).
- **Judges:** must be able to try it in a browser with sample clips (Section 11).

### 2.3 Scope

- **One exercise:** shoulder abduction (lateral arm raise), frontal camera view.
- **Two modes, one shared `kinevra` Python package:**
  - **Live mode (local edge client):** webcam → OpenCV 5 → metrics → agent feedback in near real time. Used for the demo video and a live screen-share.
  - **Cloud clip mode (the judge web endpoint):** upload a clip or pick a sample in the browser → **OpenCV 5 pipeline runs on AWS Lambda (arm64)** → agent (Bedrock) processes the session rep by rep, calling OpenCV tools → results, trace and review events shown in the dashboard.
- **Non-goals:** diagnosis, many exercises, training a pose model, multi-person tracking (v1 picks the largest person and flags `multiple_people`).

### 2.4 Measurement definitions (fixed project decision)

| Metric | Landmarks (vertex in the middle) | Purpose |
|---|---|---|
| `shoulder_abduction_deg` | hip → shoulder → elbow | **Primary ROM signal** |
| `elbow_flexion_deg` | shoulder → elbow → wrist | Form check (arm should stay nearly straight) |
| `trunk_lean_deg` | mid-hip → mid-shoulder vs. image vertical | Compensation check (leaning sideways) |
| `shoulder_elevation` | shoulder y vs. baseline ÷ torso length | Compensation check (shrugging) |

Shoulder → elbow → wrist is the *elbow* angle, not abduction. A 2D frontal view is valid because abduction happens mainly in the frontal plane; the effect of camera rotation is measured in Phase 8.

---

## 3. Architecture

### 3.1 System overview

```
                         ┌────────────────────────── AWS ──────────────────────────┐
 Browser (React, S3+CloudFront)                                                    │
   │  upload clip / pick sample, view results, review queue                        │
   ▼                                                                               │
 API Gateway (HTTP API) ──► API Lambda (FastAPI + Mangum)                          │
   │                          │  presigned upload URL, sessions, traces, reviews   │
   │                          ▼                                                    │
   │                    S3 (clips, annotated evidence frames)                      │
   │                          │ S3 event                                           │
   │                          ▼                                                    │
   │        Analysis Lambda (container, arm64 / Graviton)                          │
   │        ┌───────────────────────────────────────────────┐                      │
   │        │ OpenCV 5: decode → quality → pose (cv2.dnn)   │                      │
   │        │ → features → reps → rules                     │                      │
   │        │ → LangGraph agent per rep ◄──► Amazon Bedrock │                      │
   │        │   agent tools call OpenCV 5 again             │                      │
   │        │   (ROI re-analysis, optical flow, quality,    │                      │
   │        │    evidence snapshots)                        │                      │
   │        └───────────────────────────────────────────────┘                      │
   │                          │                                                    │
   │                          ▼                                                    │
   │                    DynamoDB (sessions, reps, decisions, trace, review events) │
   │                    CloudWatch (logs, metrics, alarms, dashboard)              │
   └───────────────────────────────────────────────────────────────────────────────┘

 Local edge client (live mode): webcam → same kinevra package → agent → overlay window
                                → optionally syncs session results to the API
```

### 3.2 Agent perception → decision → action loop (for the Agentic Vision evidence)

```
 PERCEIVE (OpenCV 5)            DECIDE (LangGraph + Bedrock)             ACT
 ─────────────────────          ────────────────────────────           ─────────────────────
 frames → quality →      ──►    triage (deterministic)          ──►    continue monitoring
 pose → features →              │ unusual?                             give feedback
 rep metrics → evidence         ▼                                      request camera adjustment
        ▲                       reason (LLM) → choose tool             log event
        │                       │                                      request human review
        │  OpenCV tool results  ▼                                      (with annotated evidence)
        └────────────────────── tools: reanalyze_segment_roi,
                                 verify_motion_optical_flow,
                                 check_camera_setup,
                                 render_evidence_snapshot, ...
                                 │
                                 ▼
                                guardrails → final decision → trace
```

**What makes this qualify:** the agent's decision after a suspicious rep depends on a **new OpenCV 5 measurement it chose to request**. For example, a low-confidence deviation is re-analysed at higher resolution on an arm ROI. The deviation is either confirmed (→ feedback or escalation) or dismissed as tracking noise (→ continue monitoring). Every such step is stored in the trace and shown in the UI.

---

## 4. Technology decisions

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.12 | Pinned |
| Env / deps | `uv` with `uv.lock` | Reproducible installs (required) |
| Computer vision | OpenCV 5 (`opencv-python==5.0.0.x` locally, `opencv-python-headless` in Lambda) | Install only **one** OpenCV wheel variant per environment. aarch64 Linux wheels exist, so Graviton works |
| Pose | ONNX pose model via **`cv2.dnn`** (OpenCV 5 DNN engine) | Keeps OpenCV at the core (30% criterion). MediaPipe only as an emergency fallback, and then OpenCV must still do everything else |
| Numerics | NumPy, SciPy | Filters, peaks, statistics |
| Agent | LangGraph + `langchain-aws` (`ChatBedrockConverse`) | LLM behind an interface; `FakeLLM` in tests |
| LLM | Amazon Bedrock (Converse API) | Model ID + region in config; choose a model enabled in your region |
| Backend | FastAPI + Mangum on Lambda | Same app runs locally with Uvicorn |
| Frontend | React + TypeScript + Vite, Recharts | Hosted on S3 + CloudFront |
| IaC | AWS CDK (Python) | `cdk deploy` / `cdk destroy` reproducibility |
| Quality | pytest, ruff, mypy, pre-commit | |
| CI/CD | GitHub Actions (lint + tests; deploy via OIDC) | |
| Optional | scikit-learn/XGBoost baseline, COOL | Only if time allows (Section 18) |

**Pose model licence:** check the weights' licence before committing to a model (many RTMPose/MMPose weights are Apache-2.0; some YOLO pose weights are AGPL-3.0). Record the decision in `docs/decisions/0002-pose-model.md`.

---

## 5. Repository structure

```
kinevra/
├── PROJECT.md
├── CLAUDE.md
├── README.md                  # quickstart, judge guide, results summary
├── Makefile                   # setup | test | lint | live | api | web | deploy | destroy | eval
├── pyproject.toml / uv.lock
├── .env.example
├── .github/workflows/{ci.yml,deploy.yml}
├── configs/
│   ├── default.yaml
│   └── exercises/shoulder_abduction.yaml
├── kinevra/
│   ├── schemas.py             # ALL shared Pydantic models (Section 6)
│   ├── config.py
│   ├── vision/
│   │   ├── capture.py         # webcam + video file source, timestamps
│   │   ├── buffer.py          # ring buffer of recent frames (for re-analysis)
│   │   ├── preprocess.py      # resize, colour, CLAHE, ROI crop/upsample
│   │   ├── quality.py         # brightness, blur, framing, person count
│   │   ├── flow.py            # optical flow in arm ROI (cv2.calcOpticalFlowFarneback)
│   │   └── overlay.py         # skeleton, angle arc, HUD, evidence snapshots
│   ├── pose/
│   │   ├── base.py            # PoseEstimator protocol
│   │   ├── opencv_dnn.py      # ONNX via cv2.dnn
│   │   └── benchmark.py
│   ├── movement/
│   │   ├── geometry.py
│   │   ├── smoothing.py       # One Euro (live), Savitzky–Golay (offline)
│   │   ├── features.py
│   │   ├── reps.py            # repetition state machine
│   │   └── session.py         # baseline, trend, consistency
│   ├── evaluation/rules.py    # GOOD / DEVIATION / UNCERTAIN + reasons
│   ├── evidence/builder.py
│   ├── agent/
│   │   ├── state.py
│   │   ├── tools.py           # incl. OpenCV 5 tools
│   │   ├── graph.py
│   │   ├── llm.py             # BedrockLLM + FakeLLM
│   │   ├── guardrails.py
│   │   ├── trace.py           # structured decision trace
│   │   └── prompts/system.md
│   ├── pipeline/
│   │   ├── live.py            # live mode orchestration
│   │   └── clip.py            # clip mode orchestration (used by Lambda)
│   ├── storage/{base.py,local.py,aws.py}
│   └── api/{main.py,routes/}
├── lambda/
│   ├── analysis/Dockerfile    # arm64, opencv-python-headless, model baked in
│   └── api/Dockerfile
├── scripts/                   # run_live.py, record_clip.py, label_reps.py, replay.py
├── tests/{unit,integration,fixtures,agent_scenarios}/
├── web/                       # React + TS
├── infra/                     # CDK app
├── eval/                      # scripts, results, plots
├── samples/                   # README with links to sample clips (clips stored in S3, not git)
└── docs/
    ├── REPORT.md              # technical report (required)
    ├── architecture.md        # system + agent diagrams (required)
    ├── decisions/             # ADRs
    ├── data_protocol.md
    ├── safety.md
    └── failure_cases.md
```

---

## 6. Data contracts (build first, keep stable)

```python
from enum import Enum
from typing import Literal, Any
from pydantic import BaseModel, Field

Side = Literal["left", "right"]

class Landmark(BaseModel):
    name: str
    x: float                      # normalised 0..1
    y: float
    visibility: float = Field(ge=0, le=1)

class PoseFrame(BaseModel):
    session_id: str
    frame_idx: int
    t: float                      # seconds since session start
    landmarks: dict[str, Landmark]
    person_count: int
    frame_quality: float          # 0..1
    quality_flags: list[str]      # low_light, blurry, out_of_frame, multiple_people

class FrameFeatures(BaseModel):
    frame_idx: int
    t: float
    side: Side
    shoulder_abduction_deg: float | None
    elbow_flexion_deg: float | None
    trunk_lean_deg: float | None
    angular_velocity_dps: float | None
    confidence: float

class RepQuality(str, Enum):
    GOOD = "GOOD"
    DEVIATION = "DEVIATION"
    UNCERTAIN = "UNCERTAIN"

class RepMetrics(BaseModel):
    rep_index: int
    side: Side
    t_start: float
    t_peak: float
    t_end: float
    min_angle: float
    max_angle: float
    rom: float
    duration_s: float
    mean_velocity_dps: float
    peak_velocity_dps: float
    smoothness: float
    max_elbow_flexion: float
    max_trunk_lean: float
    confidence: float
    rule_quality: RepQuality
    rule_reasons: list[str]

class SessionEvidence(BaseModel):
    session_id: str
    mode: Literal["live", "clip"]
    exercise: Literal["shoulder_abduction"]
    side: Side
    reps_completed: int
    reps_target: int | None
    latest_rep: RepMetrics | None
    baseline_rom: float | None
    recent_roms: list[float]
    rom_trend: Literal["increasing", "stable", "decreasing", "insufficient_data"]
    consecutive_deviations: int
    movement_consistency: float
    tracking_confidence: float
    data_quality_flags: list[str]

class AgentAction(str, Enum):
    CONTINUE = "CONTINUE_MONITORING"
    FEEDBACK = "PROVIDE_FEEDBACK"
    CAMERA_ADJUST = "REQUEST_CAMERA_ADJUSTMENT"
    LOG_EVENT = "LOG_EVENT"
    HUMAN_REVIEW = "REQUEST_HUMAN_REVIEW"

class TraceStep(BaseModel):          # one entry per node / tool call
    step: int
    node: str                        # triage | reason | tool | guard | act
    tool: str | None
    tool_input: dict[str, Any] | None
    tool_output_summary: dict[str, Any] | None
    changed_assessment: bool         # did this OpenCV result change the working assessment?
    latency_ms: int

class AgentDecision(BaseModel):
    session_id: str
    rep_index: int | None
    action: AgentAction
    rationale: str                   # short, cites evidence numbers, no diagnosis
    feedback_text: str | None
    evidence_refs: dict[str, Any]
    trace: list[TraceStep]
    guardrail_overrides: list[str] = []
    llm_used: bool
    total_latency_ms: int

class ReviewEvent(BaseModel):
    event_id: str
    session_id: str
    reason: str
    evidence: dict[str, Any]
    snapshot_keys: list[str]         # S3 keys of annotated start/peak/end frames
    status: Literal["PENDING", "ACKNOWLEDGED", "DISMISSED"] = "PENDING"
    reviewer_note: str | None = None
    created_at: str
```

---

## 7. Phases overview

| Phase | Name | Target dates | Output |
|---|---|---|---|
| 0 | Setup | Sep 28 | Repo, tooling, schemas, CI |
| 1 | OpenCV capture, buffer, quality | Sep 28–29 | Frames, quality flags, live window |
| 2 | Pose via OpenCV 5 DNN | Sep 29–Oct 1 | Landmarks + benchmark ADR |
| 3 | Movement features | Oct 1–2 | Correct angles, smoothing |
| 4 | Reps + rule baseline | Oct 2–4 | Rep counting + GOOD/DEVIATION/UNCERTAIN |
| 5 | Agent + OpenCV tools (FakeLLM) | Oct 5–8 | LangGraph loop, guardrails, trace, scenarios |
| 6 | Bedrock + live MVP | Oct 8–11 | Real reasoning; local end-to-end demo |
| 7 | AWS cloud clip mode + dashboard | Oct 12–18 | Judge web endpoint on AWS |
| 8 | Evaluation + robustness | Oct 19–22 | Numbers, failure cases |
| 9 | Report, diagrams, video, submission | Oct 22–25 | All submission items |
| 10 | Stretch (only if ahead) | – | ML baseline, COOL benchmark |

---

## Phase 0 — Setup (Sep 28)

**Tasks**
1. Folder structure (Section 5), `pyproject.toml`, `uv.lock`, `.gitignore` (ignore `data/`, `*.onnx`, `*.mp4`, `.env`).
2. Core deps: `opencv-python==5.0.*`, `numpy`, `scipy`, `pydantic>=2`, `pyyaml`, `pytest`, `ruff`, `mypy`, `pre-commit`.
3. `kinevra/schemas.py` exactly as Section 6; `config.py`; YAML configs (Section 15).
4. `Makefile`, GitHub Actions CI, `CLAUDE.md`, `docs/decisions/0001-architecture.md`.

**Acceptance criteria**
- [ ] `make setup && make test && make lint` pass on a fresh clone; CI green.
- [ ] `python -c "import cv2; print(cv2.__version__)"` prints `5.x`.
- [ ] Schemas round-trip to/from JSON in tests.

**Prompt:** *Read PROJECT.md. Do Phase 0 only: repository skeleton for the `kinevra` package, schemas, config loader, Makefile, CI and CLAUDE.md. Show the tree and test output.*

---

## Phase 1 — OpenCV capture, frame buffer, quality (Sep 28–29)

**Tasks**
1. `vision/capture.py`: `FrameSource` for webcam index or video file; monotonic timestamps; target FPS/resolution.
2. `vision/buffer.py`: ring buffer of the last N seconds of frames (downscaled) with timestamps, so the agent can re-analyse a segment.
3. `vision/preprocess.py`: resize, BGR→RGB, CLAHE (L channel) for low light, ROI crop + upsample helper. Mirroring only for display, never for left/right semantics.
4. `vision/quality.py`: brightness, blur (variance of Laplacian), and later framing/person checks → `frame_quality` + `quality_flags`.
5. `vision/overlay.py` HUD; `scripts/run_live.py` (`--video`, `--record`); `scripts/record_clip.py` with metadata JSON.

**Acceptance criteria**
- [ ] Live window ≥ 20 FPS at 640×480 on your laptop (record the number).
- [ ] Same path works on video files.
- [ ] Quality flags react when you dim lights or cover the lens.
- [ ] Unit tests for quality and buffer using synthetic arrays.

**Prompt:** *Do Phase 1 of PROJECT.md using OpenCV 5 APIs. Include the frame ring buffer and ROI helper. Tests use synthetic numpy images.*

---

## Phase 2 — Pose estimation via OpenCV 5 DNN (Sep 29–Oct 1)

**Tasks**
1. `pose/base.py`: `PoseEstimator.estimate(frame, t, frame_idx, roi=None) -> PoseFrame`. COCO-17 landmark names.
2. `pose/opencv_dnn.py`: load an ONNX pose model with `cv2.dnn`; preprocessing and output decoding (heatmap, SimCC or keypoint regression depending on the model); person selection (largest box); support an optional **ROI + higher input resolution** mode (used later by the agent's re-analysis tool).
3. `pose/benchmark.py`: FPS, latency p50/p95, still-pose jitter, % frames with required landmarks visible, on your recorded clips; measure on x86 laptop and later on arm64 Lambda.
4. ADR `0002-pose-model.md` with numbers and licence.
5. Skeleton overlay coloured by visibility; `models/README.md` with download + checksum.

**Acceptance criteria**
- [ ] Skeleton tracks your arm smoothly.
- [ ] Left/right correct (raise only the right arm → `right_elbow` moves).
- [ ] ROI mode gives higher landmark confidence on a small/far subject (show numbers).
- [ ] Decoding unit test using a stored model output fixture (no camera needed).

**Prompt:** *Do Phase 2 of PROJECT.md. Help me choose a permissively licensed ONNX pose model that runs in OpenCV 5's DNN engine, implement decoding, the ROI high-resolution mode and the benchmark. Draft ADR 0002 after I run it.*

---

## Phase 3 — Movement features (Oct 1–2)

**Tasks**
1. `geometry.py`: `angle_deg(a, b, c)` (vertex b, `atan2`, clamped); correct for aspect ratio (convert normalised coords to pixels before angles); torso-length normalisation.
2. `smoothing.py`: One Euro (live) and Savitzky–Golay (offline); parameters in config.
3. `features.py`: `FrameFeatures` for the configured side; velocity from real Δt; `None` + low confidence below visibility threshold.
4. Angle arc at the shoulder in the overlay; CSV dump for plotting.

**Acceptance criteria**
- [ ] Tests: known triangles → 0°, 90°, 180°; aspect correction; low-visibility handling.
- [ ] Sanity: arm at side ≈ 0–20°, horizontal ≈ 85–95°, overhead ≈ 160–180° vs. a phone inclinometer (`eval/sanity_angles.md`).
- [ ] Smoothed vs. raw plot saved in `eval/`.

**Prompt:** *Do Phase 3 of PROJECT.md. The primary angle is hip→shoulder→elbow with aspect-ratio correction. Write geometry tests first.*

---

## Phase 4 — Reps + rule baseline (Oct 2–4)

**Tasks**
1. `reps.py`: hysteresis state machine `REST → RAISING → PEAK → LOWERING → REST`; minimum ROM and duration; partial raises logged as `partial_rep`, not counted; emits `RepMetrics`.
2. `session.py`: baseline ROM (median of first N GOOD reps), recent ROMs, trend (slope with dead-band), consistency, consecutive deviations, rolling confidence.
3. `rules.py` (thresholds from YAML, labelled as *prototype defaults, not clinical values*): UNCERTAIN on low confidence/quality flags; DEVIATION on ROM < X% of baseline, trunk lean > Y°, elbow flexion > Z°, abnormal duration; always attach reasons.
4. Overlay: rep counter, last ROM, quality badge.
5. Record your **evaluation clips** now (Section 13.2) so Phase 8 isn't blocked.

**Acceptance criteria**
- [ ] Exact rep count on ≥ 90% of test clips (hand-counted labels in `data/labels/`).
- [ ] Partial raises not counted.
- [ ] Synthetic-sequence tests (clean, noisy, partial, pause at top).
- [ ] Deliberately bad reps (lean, bent elbow, half range) flagged with the right reason.

**Prompt:** *Do Phase 4 of PROJECT.md test-first with synthetic angle sequences in tests/fixtures, then session state and rules. Thresholds come from configs/exercises/shoulder_abduction.yaml.*

---

## Phase 5 — Agent with OpenCV tools (FakeLLM) (Oct 5–8)

**Goal:** a LangGraph agent whose decisions depend on OpenCV 5 measurements it chooses to request. Fully testable without AWS.

### 5.1 Tools (`agent/tools.py`) — deterministic, recorded in the trace

**OpenCV 5 perception tools (the heart of the Agentic Vision path)**

| Tool | What OpenCV does | How it changes the next step |
|---|---|---|
| `reanalyze_segment_roi(rep_index, scale)` | Pulls the rep's frames from the buffer/clip, crops an ROI around shoulder–elbow–wrist, upsamples, optional CLAHE, re-runs `cv2.dnn` pose, recomputes the angle curve | Confirms a deviation (→ feedback or escalation) or dismisses it as tracking noise (→ continue) |
| `verify_motion_optical_flow(rep_index)` | Dense optical flow (Farneback) in the arm ROI; compares flow-derived motion with the landmark trajectory | Separates real movement irregularity from landmark jitter; drives confirm/dismiss |
| `check_camera_setup(window_s)` | Brightness, blur, framing (is the full arm/torso in frame?), person count, subject size | → `REQUEST_CAMERA_ADJUSTMENT` with a specific instruction; after adjustment the agent re-checks (closed loop) |
| `render_evidence_snapshot(rep_index)` | Renders annotated start/peak/end frames (skeleton, angle arc, values); saves locally or to S3 | Attached to human-review events so the reviewer sees visual evidence |

**Analysis tools**

`analyze_recent_repetitions(k)`, `check_movement_consistency()`, `calculate_movement_metrics(rep_index)`, `compare_with_session_baseline()`, `generate_feedback(category)` (from an approved template library), `log_event(type, payload)`, `request_human_review(reason, evidence, snapshot_keys)`.

### 5.2 Graph (`agent/graph.py`)

```
observe → triage ─(GOOD, high confidence, no trend change)─► decide → guard → act
              │
              └─(deviation / uncertain / quality issue)─► reason (LLM) ⇄ tools (max 4 calls)
                                                              │
                                                              └─► decide → guard → act
```

- **triage** is deterministic: normal reps never call the LLM (`llm_used=false`). This saves cost and latency.
- **reason:** the LLM picks tools based on evidence; each call appends a `TraceStep` with `changed_assessment`.
- **guard** (`guardrails.py`):
  - action must be in `AgentAction`;
  - tracking confidence below threshold → `REQUEST_CAMERA_ADJUSTMENT`; never give movement feedback on bad data;
  - deterministic escalation rule (e.g. ≥ 3 consecutive confirmed deviations with ROM < 80% of baseline) **forces** `REQUEST_HUMAN_REVIEW`; the agent may escalate earlier only with evidence refs;
  - feedback must come from or closely paraphrase approved templates; max length; blocklist for diagnostic terms and "push through pain" language;
  - rate-limit feedback (at most once every 2 reps);
  - every override is logged.
- **act:** executes the action and persists the decision + trace.

### 5.3 Scenario tests (`tests/agent_scenarios/*.yaml`)

Each scenario is a scripted sequence of evidence and tool outputs with expected actions **and expected tool calls**:
1. All good reps → CONTINUE only, LLM never called.
2. Low-confidence "deviation" → `reanalyze_segment_roi` → dismissed → CONTINUE. **(OpenCV output changes the decision.)**
3. Same pattern but re-analysis confirms reduced ROM → FEEDBACK. **(OpenCV output changes the decision.)**
4. Persistent confirmed deviation → `render_evidence_snapshot` → HUMAN_REVIEW with snapshots.
5. Poor lighting / out of frame → `check_camera_setup` → CAMERA_ADJUST → re-check passes → monitoring resumes.
6. Trunk-lean compensation → FEEDBACK about posture.
7. Irregular trajectory → `verify_motion_optical_flow` shows landmark jitter only → CONTINUE.
8. LLM proposes diagnosis-style text → guardrail replaces it.
9. LLM proposes invalid action or exceeds tool budget → guardrail override.

**Acceptance criteria**
- [ ] All scenarios pass with `FakeLLM`; guardrail unit tests cover every rule.
- [ ] `scripts/replay.py clip.mp4` runs the full pipeline locally and prints each decision with its trace.
- [ ] At least one real recorded clip shows a trace where an OpenCV tool result changed the decision.

**Prompt:** *Do Phase 5 of PROJECT.md. Implement the OpenCV 5 perception tools first (with tests on fixture frames), then the analysis tools, the LangGraph graph with deterministic triage and guardrails, the trace model and FakeLLM. Write the 9 scenario tests before the graph.*

---

## Phase 6 — Bedrock + live MVP (Oct 8–11)

**Tasks**
1. AWS account: IAM Identity Center user + CLI profile; **AWS Budgets alarms** (e.g. $10 / $50 / $100); enable Bedrock access to your chosen model in your region; apply grant / free-tier credits.
2. `BedrockLLM` behind `LLMClient` using Converse tool calling; tool schemas generated from `tools.py`; low temperature; strict parsing into `AgentDecision` (one retry, then deterministic rule fallback).
3. `agent/prompts/system.md`: assistive (not clinical) role; inputs are evidence JSON; allowed actions; prefer calling an OpenCV tool when confidence is borderline; cite numbers; say "uncertain" when evidence is weak; never diagnose; feedback is short, encouraging, one instruction.
4. Timeout (e.g. 4 s) with rule-based fallback so a session never blocks; log tokens and latency.
5. `pipeline/live.py`: live mode end-to-end with the overlay showing feedback, camera-adjustment prompts and a decision ticker. The agent runs asynchronously so video never freezes.
6. End-of-session summary generated from stored numbers.

**Acceptance criteria (MVP checkpoint)**
- [ ] Live demo: camera → skeleton → reps → bad-rep sequence → agent re-analyses → feedback → escalation event.
- [ ] Scenario suite against Bedrock (`@pytest.mark.bedrock`, not in CI): ≥ 90% expected actions; disagreements documented.
- [ ] Guardrails catch 100% of injected unsafe outputs.
- [ ] Fallback works with network disabled.
- [ ] p95 agent latency and tokens per session recorded.

**Prompt:** *Do Phase 6 of PROJECT.md. Implement BedrockLLM with Converse tool calling, strict parsing, timeouts and fallback, write the system prompt, then wire live mode end to end. Use my AWS profile from the environment.*

---

## Phase 7 — AWS cloud clip mode + dashboard (Oct 12–18)

**Goal:** the judge-accessible web endpoint, with OpenCV 5 and the agent running on AWS.

### 7.1 AWS resources (CDK stacks in `infra/`)

| Stack | Resources |
|---|---|
| storage | S3 bucket (SSE, block public access, CORS for presigned uploads, lifecycle: user uploads deleted after 1 day, samples kept), DynamoDB table (on-demand) |
| analysis | **Analysis Lambda, container image, arm64 (Graviton)**, `opencv-python-headless` 5.x + ONNX model baked in, memory sized by benchmark (start at 3–4 GB), timeout ≤ 15 min, triggered by S3 upload; reserved concurrency cap |
| api | HTTP API Gateway + API Lambda (FastAPI + Mangum), throttling limits |
| web | S3 + CloudFront for the React build |
| monitoring | CloudWatch dashboard, alarms (errors, duration, Bedrock throttling), log retention |

**Clip size limits:** max ~60–90 s and ~50 MB per upload; the Analysis Lambda downsamples to the processing FPS. If measured processing time is too long for Lambda, move the analysis container to ECS Fargate (arm64) and record the reason in `docs/decisions/0003-cloud-compute.md`.

### 7.2 DynamoDB (single table `kinevra`)

| PK | SK | Item |
|---|---|---|
| `SESSION#<sid>` | `META` | status (QUEUED / PROCESSING / DONE / FAILED), mode, created_at, summary |
| `SESSION#<sid>` | `REP#0003` | RepMetrics |
| `SESSION#<sid>` | `DECISION#0003` | AgentDecision incl. trace |
| `SESSION#<sid>` | `EVENT#<ts>` | ReviewEvent |
| GSI1: `status` + `created_at` | | human-review queue |

### 7.3 API

```
POST /sessions                    → {session_id, upload_url}  (presigned S3 PUT)
POST /sessions/sample/{name}      → start analysis on a sample clip
GET  /sessions/{id}               → status, metrics, decisions, summary
GET  /sessions/{id}/trace         → full agent trace (JSON, downloadable)
GET  /reviews?status=PENDING      → review queue
POST /reviews/{event_id}          → acknowledge / dismiss + note (human control)
GET  /health
```

The frontend polls `GET /sessions/{id}` every 2 s while processing (no WebSockets needed).

### 7.4 Dashboard (React + TS)

- **Start screen:** "Try a sample clip" (good session / fatigue decline / trunk lean / poor lighting) or "Upload your own", plus consent checkbox and privacy note.
- **Session view:** annotated video frames or keyframes, angle-over-time chart, ROM-per-rep chart with baseline line, rep table with quality badges.
- **Agent timeline (key for judges):** each decision shows action, rationale, evidence numbers, and the **tool calls with a "changed assessment" marker**. For example: *"Rep 4 flagged (confidence 0.58) → ROI re-analysis → ROM 121° confirmed → feedback."*
- **Review queue:** annotated snapshots, evidence, Acknowledge / Dismiss buttons with a note.
- **Accessibility:** keyboard navigation, sufficient contrast, text alternatives for charts (the rep table), clear non-diagnostic language.
- Persistent disclaimer: *"Assistive prototype. Not a medical device. Stop if you feel pain."*

### 7.5 Security and operations

- Least-privilege IAM: Analysis Lambda may read/write only its bucket prefix and table, and invoke only the chosen Bedrock model; API Lambda has no Bedrock access unless needed.
- API throttling, upload size limits, reserved concurrency, budget alarms → protects the endpoint during judging.
- Optional demo access code (given to judges in the submission) if abuse is a worry.
- Structured JSON logs with `session_id`; custom metrics: frames processed, pose FPS on Graviton, agent latency, LLM calls per session, tool calls by type, guardrail overrides, fallbacks, review events.
- GitHub Actions deploy with OIDC (no long-lived keys).

**Acceptance criteria**
- [ ] `make deploy` (cdk deploy) builds everything from scratch; `make destroy` removes it.
- [ ] A sample clip processed on AWS end to end from the CloudFront URL, on a phone and a laptop.
- [ ] Trace for the "fatigue decline" sample visibly shows an OpenCV tool changing a decision.
- [ ] CloudWatch dashboard shows the custom metrics.
- [ ] Measured cost per session and Graviton processing time recorded.

**Prompt:** *Do Phase 7 of PROJECT.md. First the CDK stacks and the arm64 Analysis Lambda container running kinevra.pipeline.clip; verify it on a sample clip. Then the API, then the React dashboard with the agent timeline and review queue. Explain every IAM permission you add.*

---

## Phase 8 — Evaluation + robustness (Oct 19–22)

### 8.1 Metrics

| Layer | Metrics | Method |
|---|---|---|
| Pose / OpenCV | % usable frames, landmark jitter, pose FPS (laptop x86 vs. Lambda arm64) | Still poses, benchmark script |
| Angles / ROM | mean absolute error vs. reference | Held poses at known angles (inclinometer) |
| Reps | count accuracy, partial-rep false positives | Hand-counted clips |
| Rule baseline | precision/recall/F1 for DEVIATION per rep | Labelled reps |
| **Agent** | decision accuracy vs. labels; **rate at which OpenCV tool calls changed the assessment, and whether that change was correct**; unnecessary feedback per session; escalation precision/recall; camera-adjustment success (issue fixed after the prompt) | Scenario suite + recorded clips |
| Rules vs. agent | same sessions, compare false alarms and missed deviations | Side by side |
| Failure handling | behaviour under Bedrock timeout, bad clips, no person, two people | Fault injection |
| Human control | every escalation has evidence + snapshots; reviewer actions logged | Audit of stored events |
| System | end-to-end processing time per clip, cold start, cost per session | CloudWatch + billing |

### 8.2 Robustness matrix (`docs/failure_cases.md`)

Lighting (bright / dim / backlit), distance (1 m / 2 m / 3.5 m), camera angle (0° / 20° / 40° / 60°), speed (slow / normal / fast), partial occlusion, loose clothing, second person entering, incomplete reps, stopping mid-session.

For each: what happened, why, whether the system correctly went UNCERTAIN / asked for camera adjustment instead of giving wrong feedback, and mitigation.

**Acceptance criteria**
- [ ] `eval/results.md` with tables and plots; every matrix row tested at least once.
- [ ] At least 3 exported agent traces in `eval/traces/` demonstrating OpenCV output changing decisions.

**Prompt:** *Do Phase 8 of PROJECT.md. Write evaluation scripts that produce the metric tables, plots and exported traces from my labelled clips and AWS runs, then help me fill docs/failure_cases.md.*

---

## Phase 9 — Report, diagrams, video, submission (Oct 22–25)

### 9.1 Technical report (`docs/REPORT.md` → PDF), sections required by the rules

1. Problem and real-world impact
2. Users and beneficiaries
3. Architecture (system diagram + agent workflow diagram)
4. OpenCV 5 implementation (DNN pose, preprocessing, quality, ROI re-analysis, optical flow, evidence rendering)
5. Agentic workflow (loop, tools, guardrails, trace examples)
6. AWS deployment (services, Graviton Lambda, IaC, security, observability, cost)
7. Evaluation (all Phase 8 results)
8. Limitations and failure cases
9. Responsible use (no diagnosis, human review, privacy, consent, data retention)
10. Reproducibility (versions, how to build, deploy, test)

### 9.2 Diagrams (`docs/architecture.md`, exported PNG/SVG)

- System architecture: OpenCV 5 components + AWS components + agent.
- Agent workflow: perception → decision/orchestration → action, with the OpenCV tool feedback loop.

### 9.3 README (judge-first)

Top section: **"For judges"** — endpoint URL, access code (if any), which sample clip to try first, what to look for in the agent timeline, link to the video and report. Then quickstart (local), deploy, test commands, results summary, safety statement.

### 9.4 Video (≤ 5:00, unlisted YouTube)

| Time | Content |
|---|---|
| 0:00–0:20 | Team intro (show yourself — required) + problem |
| 0:20–1:40 | Live demo: exercise, skeleton, reps, feedback, camera-adjustment prompt |
| 1:40–2:40 | Agent timeline: OpenCV re-analysis changing a decision; escalation with snapshots; reviewer acknowledges |
| 2:40–3:30 | Architecture: OpenCV 5 pipeline on Graviton Lambda, Bedrock, DynamoDB, S3, CloudWatch |
| 3:30–4:30 | Results: ROM error, rep accuracy, agent vs. rules, failure cases |
| 4:30–5:00 | Limitations, responsible use, what's next |

### 9.5 Submission

- [ ] Devpost form complete; all links tested in a private browser window.
- [ ] Judges have repository access.
- [ ] Endpoint healthy; CloudWatch alarm emails go to you; **do not `cdk destroy` until after Nov 10**.
- [ ] Tag release `v1.0`.

**Prompt:** *Do Phase 9 of PROJECT.md. Draft docs/REPORT.md from the repository, eval results and ADRs; generate the architecture and agent-workflow diagrams; write the judge-first README and a video script matching 9.4.*

---

## Phase 10 — Stretch goals (only if everything above is done)

1. **ML baseline:** RF/XGBoost on per-rep features with grouped CV by person; compare with rules and agent.
2. **COOL on Graviton:** swap in COOL for the claimed workload in the arm64 container; benchmark against standard OpenCV 5 (same inputs; latency, throughput, cost); document version and configuration. Claim the COOL award only if verified.
3. **Longitudinal view:** compare with a previous session's baseline.

---

## 11. Judge experience (design for this from day one)

- Works in a browser with **no install**; sample clips are preloaded so judges don't need to film themselves.
- First result in under ~2 minutes for a sample clip (measure cold start; pre-warm or keep samples cached if needed).
- The agent timeline makes the perception → decision → action loop obvious without reading code.
- A downloadable trace JSON for each session.
- Clear error messages (clip too long, no person detected, etc.).

---

## 12. Safety and responsible AI (applies to every phase)

- **No diagnosis:** outputs describe movement only; enforced in prompts, guardrails and UI copy.
- **Human in the loop:** deterministic escalation rule; reviewer acknowledge/dismiss with notes; the agent can never suppress a required escalation.
- **Uncertainty first:** poor data → camera adjustment or UNCERTAIN, never confident feedback.
- **Explainability:** every decision stores evidence numbers, tool calls, the trace and snapshots.
- **Pain/safety messaging:** never encourage pushing through pain; persistent "stop if you feel pain" notice.
- **Privacy:** consent checkbox on upload; uploads auto-deleted after 1 day; only derived metrics and annotated evidence frames kept; no names; sample clips are of you/consenting volunteers.
- **Security:** encryption at rest, HTTPS, least-privilege IAM, throttling, no secrets in git.
- Documented in `docs/safety.md` and the report.

---

## 13. Data

### 13.1 Rules
Self-recorded or consenting-volunteer clips only; no faces needed (frame from neck down if preferred); never committed to git; stored in S3 with metadata.

### 13.2 Clips to record (by Oct 4)
- 4 **sample clips for judges**: good session; fatigue decline (ROM drops over reps); trunk-lean compensation; poor lighting / partly out of frame.
- ~20–30 **evaluation clips** across the robustness matrix, with hand labels per rep (count, quality, deviation type).
- Held poses at 0°, 45°, 90°, 135°, 170° measured with an inclinometer for ROM error.

---

## 14. Working with Claude — practical tips

- Paste the phase section, errors and test output; Claude can't see your camera, so share screenshots, plots or CSV dumps.
- Test-first for geometry, reps, rules, guardrails and agent scenarios.
- Use plan-first for Phases 5 and 7: ask for a short plan and approve it before code.
- Keep fixtures small (landmark JSON, a few frames) so CI runs without video.
- Review every change to `schemas.py`, `guardrails.py`, IAM and thresholds.

---

## 15. Configuration reference (`configs/exercises/shoulder_abduction.yaml`)

```yaml
# PROTOTYPE DEFAULTS — not clinical thresholds. Tune from your own data.
exercise: shoulder_abduction
side: right
landmarks:
  primary_angle: [hip, shoulder, elbow]
  elbow_angle: [shoulder, elbow, wrist]
visibility_threshold: 0.5
processing_fps: 15
smoothing:
  one_euro: {min_cutoff: 1.0, beta: 0.05, d_cutoff: 1.0}
reps:
  rest_angle_max: 30
  raise_margin: 15
  min_rom: 40
  min_duration_s: 0.8
  max_duration_s: 10
baseline:
  first_n_good_reps: 3
  recent_k: 5
rules:
  rom_ratio_deviation: 0.80
  max_trunk_lean_deg: 10
  max_elbow_flexion_deg: 30
  min_confidence: 0.6
tools:
  reanalyze_roi_scale: 2.0
  reanalyze_trigger_confidence: [0.45, 0.75]   # borderline band → agent should look again
  flow_roi_padding: 0.15
agent:
  max_tool_calls: 4
  feedback_min_reps_between: 2
  escalation:
    consecutive_confirmed_deviations: 3
    rom_ratio_below: 0.80
llm:
  provider: bedrock
  model_id: "<model enabled in your region>"
  region: "<your region>"
  temperature: 0.2
  timeout_s: 4
limits:
  max_clip_seconds: 90
  max_upload_mb: 50
```

---

## 16. Key proposition (use consistently in README, report, video, Devpost)

> **Kinevra doesn't just classify a movement — it decides when to look again.** OpenCV 5 measures every repetition; an AI agent reads that evidence over time, calls OpenCV tools to re-examine uncertain moments or asks for a better camera view, then chooses to keep monitoring, coach the user, or escalate to a human reviewer with annotated visual evidence.

---

## 17. Schedule (29 days)

| Week | Dates | Goal | Must be true by the end |
|---|---|---|---|
| 1 | Sep 28 – Oct 4 | Phases 0–4 | Reps + rules working live; sample and eval clips recorded |
| 2 | Oct 5 – Oct 11 | Phases 5–6 | Agent with OpenCV tools + Bedrock; **live MVP demo recorded as backup** |
| 3 | Oct 12 – Oct 18 | Phase 7 | Public endpoint on AWS processing sample clips with agent timeline |
| 4 | Oct 19 – Oct 25 | Phases 8–9 | Evaluation, report, diagrams, video; **submit Oct 25** |
| – | Oct 26 – 27 | Buffer | Fix-only; confirm submission; endpoint monitored |

Daily rhythm: ~30 min review of yesterday's Progress Log → build → 15 min to test, commit and update the log.

---

## 18. Scope cuts (apply in this order if behind schedule)

1. Drop Phase 10 entirely (ML, COOL, longitudinal).
2. Drop `verify_motion_optical_flow` (keep ROI re-analysis and camera-setup tools; together they still satisfy the agentic requirement).
3. Simplify the dashboard to one page (upload/sample → timeline → review queue).
4. Live mode stays local only (no sync to cloud).
5. **Never cut:** OpenCV 5 in the core pipeline, the OpenCV workload on AWS, the agent trace showing OpenCV changing decisions, guardrails, evaluation with failure cases, report, diagrams, video.

---

## 19. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Pose model won't run in `cv2.dnn` | Test 2–3 ONNX models on day 1 of Phase 2; ADR records fallback |
| Wrong angle definition | Fixed in §2.4; inclinometer sanity check |
| Lambda too slow / cold start | Downsample FPS, cap clip length, size memory by benchmark, pre-warm before demo; ECS Fargate fallback |
| Bedrock latency or throttling | Deterministic triage, timeouts, rule fallback, throttling alarm |
| LLM hallucination / diagnosis language | Numbers only from tools, strict schema, guardrails |
| Cost overrun or abuse during judging | Budgets, throttling, reserved concurrency, upload limits, optional access code |
| Not meeting agentic criteria | Scenario tests 2, 3, 5 plus exported traces are the proof; built in Phase 5, shown in UI |
| Running out of time | Scope cuts §18; code freeze Oct 24 |

---

## 20. Progress log

| Date | Phase | Done | Issues / decisions | Next step |
|---|---|---|---|---|
| 2026-09-28 | – | PROJECT.md aligned with competition rules; name chosen: Kinevra | Confirm proposal/eligibility status | Phase 0 |
| 2026-09-28 | 0 | Repo skeleton, schemas (§6), typed config loader + env overrides, YAML configs, Makefile, CI, pre-commit, CLAUDE.md, ADR 0001. 21 tests pass; ruff + mypy (strict) clean. OpenCV 5.0.0.93 verified (gui + headless) | Python 3.12 pinned; `gui`/`headless` extras declared conflicting in uv so each env has one OpenCV wheel. `make` not installed on Windows dev box; CI not yet run (repo not pushed) | Push, confirm CI green; Phase 1 |
| 2026-09-28 | 1 | `vision/`: capture (webcam + file, monotonic timestamps, file downsampling), ring buffer, preprocess (resize, RGB, CLAHE, ROI crop/upsample + coord mapping), quality (brightness/contrast/Laplacian blur → score + flags), overlay HUD, FPS meter, ClipWriter; `scripts/run_live.py`, `scripts/record_clip.py` (+ `ClipMetadata` schema). 62 tests pass on gui and headless OpenCV; ruff + mypy clean | OpenCV 5 stubs: use `cv2.VideoWriter.fourcc`. Quality thresholds in `default.yaml` are first guesses. **Pending manual checks:** live FPS at 640×480, flags react to dimming / covering lens | Run live checks and record FPS; Phase 2 |
| 2026-09-28 | 2 | `pose/`: base (PoseEstimator, ROI mode, visibility, framing/person flags), RTMDet-nano + RTMPose-s via `cv2.dnn` ENGINE_NEW (default), MediaPipe/BlazePose from OpenCV Zoo (baseline), factory, benchmark CLI; skeleton overlay; `download_models.py` (checksums + NMS-free RTMDet); pose in `run_live.py` (`--pose`, `--roi`). 90 tests pass (incl. real-model integration + stored RTMPose output fixture); ruff + mypy clean. Live pipeline 33.4 FPS on a 640×400 clip | Chose RTMPose (ADR 0002, proposed): MediaPipe detector missed/cropped a clear subject and was 3–5× slower. OpenCV 5 new engine keeps first-run shapes → RTMDet end2end NMS broke on frame 2; fixed by cutting the graph before NMS (NMS in `cv2.dnn.NMSBoxes`). ROI re-analysis recovers a far subject the detector misses (conf 0 → 0.85) but gives no gain on well-framed frames. Added `onnx` dev dep | Webcam checks (left/right, smooth tracking), benchmark on own clips → finalise ADR 0002; Phase 3 |

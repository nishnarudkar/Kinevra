# Kinevra

> **Agentic Computer Vision for Rehabilitation Movement Monitoring**  
> *It watches the movement, so you can focus on recovery.*

[![OpenCV 5](https://img.shields.io/badge/OpenCV-5.0.0-blue.svg)](https://opencv.org/)
[![AWS Graviton](https://img.shields.io/badge/AWS-Graviton%2FARM64-orange.svg)](https://aws.amazon.com/ec2/graviton/)
[![Amazon Bedrock](https://img.shields.io/badge/Amazon-Bedrock-FF9900.svg)](https://aws.amazon.com/bedrock/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)

**Kinevra doesn't just classify a movement — it decides when to look again.**  
OpenCV 5 DNN measures every repetition; an AI agent reads that evidence over time, calls OpenCV tools to re-examine uncertain moments or ask for a better camera view, then chooses whether to keep monitoring, coach the user, or escalate to a human reviewer with annotated visual evidence.

Built for the **OpenCV AI Competition 2026, powered by AWS** (Featured Path: **Agentic Vision**).  
Developer: **Nishant Narudkar**

---

> ⚠️ **Safety Disclaimer:** Kinevra is an assistive prototype for home exercise monitoring between physiotherapy visits. It is **not** a medical device and does not perform medical diagnosis. Users are instructed to stop immediately if they experience pain.

---

## 🏛️ Core Philosophy

- 📐 **Computer vision measures.** All angles, ROM (Range of Motion), velocities, and frame quality scores are calculated deterministically using OpenCV 5. Never by an LLM.
- 🧠 **The agent decides.** The agent (LangGraph + Amazon Bedrock) analyzes session trajectories, triggers OpenCV re-perception tools when confidence is low or anomalies occur, and selects appropriate actions.
- ⚡ **AWS runs vision and agent at scale.** The OpenCV 5 analysis pipeline runs in cloud containers on AWS Lambda (Graviton/arm64) alongside Amazon Bedrock, DynamoDB, S3, and CloudWatch.
- 👩‍⚕️ **Humans remain in control.** Escalate suspicious sessions to a therapist review queue with visual start/peak/end evidence snapshots.

---

## 🔍 Agentic Vision: Perception → Decision → Action Loop

Unlike traditional computer vision pipelines that output static predictions, Kinevra implements an active, closed-loop agent:

```
  PERCEIVE (OpenCV 5)            DECIDE (LangGraph + Bedrock)             ACT
 ─────────────────────          ────────────────────────────           ─────────────────────
 frames → quality →      ──►    triage (deterministic)          ──►    continue monitoring
 pose → features →              │ unusual / uncertain?                 provide feedback
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

### Perception Tools (OpenCV 5 inside the Agent Loop)

1. **`reanalyze_segment_roi(rep_index, scale)`**: Pulls rep frames from the ring buffer, crops the arm region-of-interest (ROI), upsamples, applies CLAHE contrast enhancement, and re-runs ONNX pose inference at high resolution to resolve low-confidence tracking.
2. **`verify_motion_optical_flow(rep_index)`**: Computes dense optical flow (`cv2.calcOpticalFlowFarneback`) inside the arm ROI to distinguish physical movement irregularities from landmark estimation jitter.
3. **`check_camera_setup(window_s)`**: Evaluates scene illumination, Laplacian variance blur, subject framing, and person count to generate actionable user setup feedback (e.g., *"Step back into frame"*).
4. **`render_evidence_snapshot(rep_index)`**: Draws skeleton overlays, angle arcs, and metric HUDs on start, peak, and end rep frames for the clinician review queue.

---

## 📌 For Judges

- **Target Exercise:** Shoulder Abduction (lateral arm raise in frontal camera view).
- **Judge Web Endpoint:** Upload custom clips or run pre-loaded sample test cases (Good Form, Fatigue Decline, Trunk Lean Compensation, Poor Lighting).
- **Agent Decision Trace:** Inspect every decision step in real-time. Look for the `changed_assessment: true` tag where OpenCV perception tool re-analysis actively modified the agent's coaching decision.
- **Technical Report:** Available in [`docs/REPORT.md`](docs/REPORT.md).
- **Architecture & Workflow Diagrams:** Available in [`docs/architecture.md`](docs/architecture.md).

---

## 📦 Repository Layout

```
kinevra/
├── PROJECT.md                 # Single source of truth & phase schedule
├── CLAUDE.md                  # Development rules & guidelines
├── README.md                  # Quickstart, architecture & judge guide
├── Makefile                   # Setup, test, lint & deploy targets
├── pyproject.toml / uv.lock   # Pinned dependencies & build config
├── configs/
│   ├── default.yaml           # Global system & pipeline settings
│   └── exercises/             # Exercise definitions & thresholds
│       └── shoulder_abduction.yaml
├── docs/
│   └── decisions/             # Architecture Decision Records (ADRs)
├── kinevra/                   # Core Python package
│   ├── schemas.py             # Shared Pydantic v2 data contracts
│   ├── config.py              # YAML config loader with env overrides
│   ├── vision/                # OpenCV 5 frame buffer, quality & flow
│   ├── pose/                  # cv2.dnn ONNX pose estimation
│   ├── movement/              # Geometry, smoothing & rep state machine
│   ├── evaluation/            # Rule-based baseline evaluation
│   ├── evidence/              # Evidence package builder
│   ├── agent/                 # LangGraph agent, tools, guardrails & trace
│   ├── pipeline/              # Live & batch clip orchestration
│   └── api/                   # FastAPI routes & Mangum handler
├── lambda/                    # AWS Lambda container configurations (arm64)
├── scripts/                   # Local execution & benchmarking tools
├── tests/                     # Unit, integration & scenario test suite
│   ├── unit/                  # Fast pytest suite
│   ├── fixtures/              # Synthetic test fixtures
│   └── agent_scenarios/       # Scripted agent perception scenarios
├── web/                       # React + TypeScript judge dashboard
└── infra/                     # AWS CDK infrastructure stack (Python)
```

---

## 📊 Shared Data Contracts (`kinevra/schemas.py`)

All module communications are enforced via strict Pydantic v2 schemas:

| Schema | Purpose |
|---|---|
| `PoseFrame` | Raw keypoints, visibility scores, quality flags, person count |
| `FrameFeatures` | Calculated joint angles (`shoulder_abduction`, `elbow_flexion`, `trunk_lean`) & velocities |
| `RepMetrics` | Start/peak/end timings, max ROM, movement smoothness, rule classification |
| `SessionEvidence` | Baseline ROM comparison, trend analysis, consecutive deviation counts |
| `AgentDecision` | Selected `AgentAction`, non-diagnostic rationale, evidence refs, full trace |
| `TraceStep` | Detailed log of every graph node execution & OpenCV tool output |
| `ReviewEvent` | Escalated session packet containing annotated visual snapshots for therapist review |

---

## 🚀 Quickstart & Setup

### Prerequisites

- **Python 3.12**
- **uv** (recommended package installer) or `pip` / `venv`

### Installation

```bash
# Clone the repository
git clone https://github.com/nishnarudkar/Kinevra.git
cd Kinevra

# Install dependencies with GUI support (for local webcam preview)
make setup

# Or using uv directly:
uv sync --extra gui
```

> **Note on OpenCV 5:** Kinevra uses `opencv-python==5.0.0.93` locally (GUI enabled) and `opencv-python-headless==5.0.0.93` in server/Lambda environments.

### Verification & Testing

```bash
# Verify OpenCV 5 installation
python -c "import cv2; print(cv2.__version__)"
# Output: 5.0.0.93

# Run unit test suite
make test
# Or: .\.venv\Scripts\pytest.exe -m "not bedrock and not camera"

# Run linters and type checker
make lint
# Or: .\.venv\Scripts\ruff.exe check . && .\.venv\Scripts\mypy.exe kinevra
```

---

## 🛡️ Safety & Responsible AI

1. **Non-Diagnostic Language Policy:** Code, prompts, and UI strictly use terms like *"movement deviation"*, *"reduced range of motion"*, or *"trunk compensation"*. Medical terms (e.g., *"impingement"*, *"pathology"*, *"diagnosis"*) are prohibited.
2. **Deterministic Guardrails:** High-level actions pass through `guardrails.py`. Bad image quality automatically overrides LLM outputs to request camera adjustment. Persistent deviations trigger mandatory human escalation.
3. **Data Privacy:** User uploaded video clips are processed in temporary memory on AWS Lambda, stored with auto-deletion lifecycle policies (24 hours), and only derived metric data is retained.

---

## 📄 License & Credits

- **Developer:** Nishant Narudkar
- **Competition:** OpenCV AI Competition 2026, powered by AWS
- **License:** MIT License

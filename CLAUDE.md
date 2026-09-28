# CLAUDE.md — rules for working on Kinevra

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

## Git

- Never run `git commit` / `git push` or open PRs. The developer commits personally.
- Never add Claude as author, co-author or contributor anywhere.
- After each task, give a brief summary of what changed (files + one-line commit message)
  so the developer can commit it.

## Environment notes

- One OpenCV wheel per environment: `make setup` (gui, local) or
  `make setup CV_EXTRA=headless` (CI / Lambda).
- On Windows without `make`, run the underlying commands: `uv sync --extra gui`,
  `uv run pytest -m "not bedrock and not camera"`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy`.

# Kinevra developer commands. Requires uv (https://docs.astral.sh/uv/).
# CV_EXTRA selects the single OpenCV 5 wheel for this environment:
#   gui      -> opencv-python (local dev, live window)   [default]
#   headless -> opencv-python-headless (CI, Lambda, servers)
CV_EXTRA ?= gui
UV ?= uv

.PHONY: help setup test lint format typecheck live api web deploy destroy eval clean

help:
	@echo "setup | test | lint | format | live | api | web | deploy | destroy | eval | clean"

setup:
	$(UV) sync --locked --extra $(CV_EXTRA)
	$(UV) run pre-commit install || true

test:
	$(UV) run pytest -m "not bedrock and not camera"

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy

# --- later phases (targets reserved so the interface is stable) ----------------------------
live:
	$(UV) run python scripts/run_live.py $(ARGS)

api:
	@echo "Phase 7: FastAPI app not implemented yet" && exit 1

web:
	@echo "Phase 7: React dashboard not implemented yet" && exit 1

deploy:
	@echo "Phase 7: CDK stacks not implemented yet" && exit 1

destroy:
	@echo "Phase 7: CDK stacks not implemented yet" && exit 1

eval:
	@echo "Phase 8: evaluation scripts not implemented yet" && exit 1

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache

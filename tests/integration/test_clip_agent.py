"""Clip pipeline + agent with the real pose model and real OpenCV tools (PolicyLLM)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from kinevra.agent.llm import PolicyLLM
from kinevra.config import load_config
from kinevra.pipeline.clip import analyze_clip
from kinevra.pose.factory import model_dir
from kinevra.pose.opencv_dnn import RTMDET_FILE, RTMPOSE_FILE
from kinevra.schemas import AgentAction
from kinevra.storage.local import LocalStore


@pytest.mark.models
def test_empty_room_asks_person_to_step_in(tmp_path: Path) -> None:
    cfg = load_config(env={})
    if not all((model_dir(cfg) / f).is_file() for f in (RTMDET_FILE, RTMPOSE_FILE)):
        pytest.skip("rtmpose files missing")
    clip = tmp_path / "empty.avi"
    rng = np.random.default_rng(0)
    room = (
        0.5 * rng.integers(0, 255, (240, 320, 3))
        + 0.5 * np.tile(np.linspace(0, 255, 320), (240, 1))[..., None]
    ).astype(np.uint8)
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter.fourcc(*"MJPG"), 15, (320, 240))
    for _ in range(40):
        writer.write(room)
    writer.release()

    store = LocalStore(tmp_path / "sessions")
    result = analyze_clip(clip, cfg, llm=PolicyLLM(), store=store, session_id="empty1")
    assert result.reps == []
    assert [d.action for d in result.decisions] == [AgentAction.CAMERA_ADJUST]
    d = result.decisions[0]
    tool = next(s for s in d.trace if s.node == "tool")
    assert tool.tool == "check_camera_setup" and tool.changed_assessment
    assert d.feedback_text is not None and d.feedback_text.startswith("Step into")
    assert (tmp_path / "sessions" / "empty1" / "decisions.json").is_file()
    assert len(result.artifact_keys) == 4

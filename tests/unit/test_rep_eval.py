import pytest

from kinevra.evaluation.rep_eval import markdown_report, score_clip, summarise
from kinevra.schemas import ClipLabel, RepLabel, RepMetrics, RepQuality


def rep(i: int, quality: RepQuality, reasons: list[str] | None = None) -> RepMetrics:
    return RepMetrics(
        rep_index=i,
        side="right",
        t_start=0,
        t_peak=1,
        t_end=2,
        min_angle=15,
        max_angle=150,
        rom=135,
        duration_s=2,
        mean_velocity_dps=100,
        peak_velocity_dps=180,
        smoothness=1,
        max_elbow_flexion=5,
        max_trunk_lean=0,
        confidence=0.9,
        rule_quality=quality,
        rule_reasons=reasons or [],
    )


G, D, U = RepQuality.GOOD, RepQuality.DEVIATION, RepQuality.UNCERTAIN


def test_count_only_label() -> None:
    s = score_clip(ClipLabel(clip="a.mp4", side="right", reps=3), [rep(1, G)] * 3, 0)
    assert s.exact and s.tp == s.fp == s.fn == s.tn == 0
    s = score_clip(ClipLabel(clip="a.mp4", side="right", reps=4, partial_reps=1), [rep(1, G)], 1)
    assert not s.exact and s.logged_partials == 1


def test_per_rep_agreement() -> None:
    label = ClipLabel(
        clip="b.mp4",
        side="right",
        reps=5,
        rep_labels=[
            RepLabel(quality="GOOD"),
            RepLabel(quality="DEVIATION", deviations=["trunk_lean"]),
            RepLabel(quality="DEVIATION", deviations=["reduced_rom"]),
            RepLabel(quality="GOOD"),
            RepLabel(quality="DEVIATION", deviations=["elbow_flexion"]),
        ],
    )
    reps = [
        rep(1, G),
        rep(2, D, ["trunk_lean: 14°"]),
        rep(3, G),  # missed reduced ROM
        rep(4, D, ["elbow_flexion: 35°"]),  # false alarm
        rep(5, U, ["low_tracking_confidence: 0.4"]),
    ]
    s = score_clip(label, reps, 0)
    assert (s.tp, s.fp, s.fn, s.tn, s.uncertain) == (1, 1, 1, 1, 1)
    assert s.reason_hits == {"trunk_lean": 1, "reduced_rom": 0}
    summary = summarise([s])
    assert summary["deviation_precision"] == pytest.approx(0.5)
    assert summary["deviation_recall"] == pytest.approx(0.5)
    assert summary["exact_count_pct"] == 100.0
    md = markdown_report([s])
    assert "| b.mp4 | 5 | 5 | yes |" in md and "1/1/1/1" in md


def test_empty_summary() -> None:
    assert summarise([])["exact_count_pct"] is None
    assert "n/a" in markdown_report([])

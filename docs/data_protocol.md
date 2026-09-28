# Data protocol

## Rules (PROJECT.md §13.1)

- Only self-recorded or consenting-volunteer clips. No real patient data.
- Faces are not needed: frame from the neck down if you prefer (the nose/eyes landmarks are
  not used by any measurement).
- Clips (`data/clips/`) are **never committed**; sample clips for judges go to S3 (Phase 7).
- Hand labels (`data/labels/*.json`) **are committed**: small JSON, no personal data, and they
  are the ground truth behind the evaluation numbers.

## Recording (`scripts/record_clip.py`)

Frontal view, whole upper body plus hips in frame, exercising arm = `side` in
`configs/exercises/shoulder_abduction.yaml` (override with `KINEVRA_SIDE`).

```bash
uv run python scripts/record_clip.py --name good_01 --consent --lighting normal \
    --distance-m 2.0 --camera-angle-deg 0 --seconds 60 --notes "10 reps, good form"
```

Each clip gets a metadata sidecar (`ClipMetadata` in `kinevra/schemas.py`).

### Clips to record (§13.2, by Oct 4)

**4 sample clips for judges** (60–90 s each):

| Name | Content |
|---|---|
| `sample_good` | 8–10 full, controlled raises; good posture |
| `sample_fatigue` | 3–4 full raises, then gradually lower peaks (last reps ≈ 60–70 % of the first) |
| `sample_trunk_lean` | normal range but lean the trunk sideways (away from the arm) on some reps |
| `sample_poor_lighting` | dim room or backlit window, and step partly out of frame once |

**~20–30 evaluation clips** covering the robustness matrix (PROJECT.md §8.2): lighting
(bright / dim / backlit), distance (1 / 2 / 3.5 m), camera angle (0 / 20 / 40 / 60°),
speed (slow / normal / fast), partial occlusion, loose clothing, a second person entering,
incomplete reps, stopping mid-session. Mix in deliberate deviations: half-range raises,
bent elbow (> 30°), trunk lean (> 10°), shoulder shrug, and a few partial raises (< 40°) that
should NOT count.

**Held poses** for angle error: 0°, 45°, 90°, 135°, 170° with a phone inclinometer
(`scripts/sanity_angles.py`).

## Labelling (`scripts/label_reps.py`)

The tool plays the clip **without** the system's rep counter (labels must stay independent).
Press a key when each repetition finishes:

| Key | Meaning |
|---|---|
| `g` | good rep |
| `r` / `l` / `e` / `s` | rep with reduced ROM / trunk lean / bent elbow / shoulder shrug |
| `p` | partial raise — **not** a rep |
| `u` | undo last key; `space` pause; `,` `.` step frames while paused; `q` save |

```bash
uv run python scripts/label_reps.py data/clips/sample_fatigue.mp4 --labeller NN
```

### Label format (`ClipLabel` in `kinevra/schemas.py`)

```json
{
  "clip": "sample_fatigue.mp4",
  "side": "right",
  "reps": 8,
  "partial_reps": 1,
  "rep_labels": [
    {"t": 4.2, "quality": "GOOD", "deviations": []},
    {"t": 21.8, "quality": "DEVIATION", "deviations": ["reduced_rom"]}
  ],
  "labeller": "NN",
  "notes": "optional"
}
```

What counts as a rep: the arm rises by at least ~40° from rest and returns down. A raise that
stays below ~40° is a partial. Label what you **see**, not what you think the system does.

## Evaluation

```bash
uv run python scripts/eval_reps.py --verbose     # → eval/rep_counting.md
```

Target (Phase 4): exact rep count on ≥ 90 % of labelled clips. The same labels feed the
DEVIATION precision/recall of the rule baseline (and, in Phase 8, the agent).

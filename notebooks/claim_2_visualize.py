"""claim_2 figure: post-occlusion tracking quality as the memory bank is poisoned with clean distractors."""

import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper"))

import style

RESULTS = "data/claim_2/results.pkl"
FIGURES = Path("data/claim_2/paper")


ARMS = (("sam", "SAM 2 baseline"), ("memory", "memory oracle"), ("mask", "mask oracle"))
COVERAGE_IOU = 0.5


def hygiene(clip_record, box_ious, commits):
    """Visible frames held at box IoU >= COVERAGE_IOU, plus occluded frames the arm left uncommitted,
    over every annotated post-occlusion frame."""

    box_ious = np.asarray(box_ious, dtype=float)
    visible_frames, occluded_frames = clip_record["has_box"] & ~clip_record["occluded"], np.asarray(clip_record["occluded"])
    scored_frames = int(visible_frames.sum() + occluded_frames.sum())
    if not scored_frames:
        return np.nan

    held_frames = float((box_ious[visible_frames] >= COVERAGE_IOU).sum())
    clean_refusals = float((~np.asarray(commits, dtype=bool)[occluded_frames]).sum())
    return (held_frames + clean_refusals) / scored_frames


results = pickle.load(open(RESULTS, "rb"))
scored_clips, probabilities = results["clips"], results["corruption_ps"]
threshold = results["commit_threshold"]

style.use_paper_style()
figure_handle, axis_handle = plt.subplots(figsize=(style.COLUMN, 3.1))

highest = 0.0
for position, (arm_name, label) in enumerate(ARMS):
    plotted_values = np.array([np.nanmean([hygiene(c, c[(arm_name, corruption_probability)], c[(arm_name, corruption_probability, "commit")]) for c in scored_clips]) for corruption_probability in probabilities])
    axis_handle.plot(probabilities, plotted_values, label=label, zorder=3, **style.series_style(position))
    highest = max(highest, float(np.nanmax(plotted_values)))

axis_handle.set_xticks(probabilities)
axis_handle.set_xticklabels([f"{corruption_probability:g}" for corruption_probability in probabilities])
scored_span = probabilities[-1] - probabilities[0]
axis_handle.set_xlim(probabilities[0] - 0.04 * scored_span, probabilities[-1] + 0.04 * scored_span)
style.gridlines(axis_handle, 0.05)
style.headroom(axis_handle, highest)
style.style_axes(axis_handle, "corruption probability (per frame)", f"coverage @ {COVERAGE_IOU:g}")
axis_handle.legend(loc="best")

style.save(figure_handle, FIGURES / "fig_corruption")

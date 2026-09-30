"""claim_2 figure: VOT-LT F as the memory bank is poisoned with clean distractors.

VOT-LT is the primary metric. The coverage@0.5 and memory-hygiene scores this file used to compute are
gone: coverage credits an arm only for the frames it holds, and cannot express the thing corruption
actually breaks -- whether the tracker still knows which frames to speak on once its bank holds the wrong
person.

Three arms, one line each, against corruption probability. The oracles read ground truth, so their margin
over SAM 2 is the headroom a deployable commit gate could recover.

    python notebooks/claim_2_visualize.py [results.pkl] [out_dir]
    WINDOW=1 python notebooks/claim_2_visualize.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper"))

import style
from src.metrics.vot_lt import Predictions

STEPS = 61
WINDOW = os.environ.get("WINDOW", "0") != "0"
ARMS = (("sam", "SAM 2 baseline"), ("memory", "memory oracle"), ("mask", "mask oracle"))
CONFIDENCES = ["iou_token", "object_score"]


def frame_masks(clip_record):
    """`(occluded, evaluated)` for one clip, with the memory window applied when asked for."""

    occluded = np.asarray(clip_record["occluded"], dtype=bool)
    kept = np.asarray(clip_record["has_box"], dtype=bool).copy()
    if WINDOW and occluded.any():
        kept[: int(np.where(occluded)[0][-1]) + 1] = False
    return occluded, kept


def confidence_vector(clip_record, arm_name, probability, signal, n_frames):
    """One arm's confidence at one corruption level, or None when the run stored none.

    `claim_2.py` stores these under `(arm, p, signal)` -- one entry per rate, because poisoning the bank
    changes what the arm believes and therefore what it reports."""

    stored = clip_record.get((arm_name, probability, signal))
    if stored is None:
        return None
    value = np.nan_to_num(np.asarray(stored, dtype=float), nan=-np.inf)
    if len(value) >= n_frames:
        return value[-n_frames:]
    return np.pad(value, (0, n_frames - len(value)), constant_values=-np.inf)


def vot_lt(scored_clips, arm_name, probability):
    """Best VOT-LT F over the arm's confidences at one corruption level. Returns `(F, degenerate)`."""

    best, degenerate = 0.0, True
    for signal in CONFIDENCES:
        vectors, usable = [], True
        for clip_record in scored_clips:
            overlap = clip_record[(arm_name, probability)]
            vector = confidence_vector(clip_record, arm_name, probability, signal, len(np.asarray(overlap)))
            if vector is None:
                usable = False
                break
            vectors.append(vector)
        if not usable:
            continue

        flat = np.concatenate(vectors)
        flat = flat[np.isfinite(flat)]
        if not len(flat):
            continue
        thresholds = np.unique(np.quantile(flat, np.linspace(0, 1, STEPS)))

        predictions = Predictions()
        for clip_record, vector in zip(scored_clips, vectors):
            occluded, kept = frame_masks(clip_record)
            predictions.add(clip_record[(arm_name, probability)], vector, occluded, kept)
        best = max(best, predictions.macro_scores(thresholds)[3])
        degenerate = False

    if degenerate:
        # No stored confidence: report every frame. Still VOT-LT -- it is what the protocol gives a tracker
        # that never abstains -- but precision equals recall and both equal mean overlap.
        predictions = Predictions()
        for clip_record in scored_clips:
            overlap = np.asarray(clip_record[(arm_name, probability)])
            occluded, kept = frame_masks(clip_record)
            predictions.add(overlap, np.ones(len(overlap)), occluded, kept)
        best = predictions.macro_scores([0.5])[3]
    return best, degenerate


def main():
    results_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/claim_2/results.pkl")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/sam-2/claim_2")

    results = pickle.load(open(results_path, "rb"))
    scored_clips, probabilities = results["clips"], results["corruption_ps"]

    style.use_paper_style()
    figure_handle, axis_handle = plt.subplots(figsize=(style.COLUMN, 3.1))

    highest, degenerate_arms, rows = 0.0, [], {}
    for position, (arm_name, label) in enumerate(ARMS):
        scores, degenerate = [], False
        for probability in probabilities:
            value, is_degenerate = vot_lt(scored_clips, arm_name, probability)
            scores.append(value)
            degenerate |= is_degenerate
        scores = np.asarray(scores)
        rows[label] = scores
        if degenerate:
            degenerate_arms.append(label)

        axis_handle.plot(probabilities, scores, label=label, zorder=3, **style.series_style(position))
        highest = max(highest, float(np.nanmax(scores)))
        print(f"{label:16s} " + "  ".join(f"p{p:g}={v:.4f}" for p, v in zip(probabilities, scores)))

    axis_handle.set_xticks(probabilities)
    axis_handle.set_xticklabels([f"{probability:g}" for probability in probabilities])
    span = probabilities[-1] - probabilities[0]
    axis_handle.set_xlim(probabilities[0] - 0.04 * span, probabilities[-1] + 0.04 * span)
    style.gridlines(axis_handle, 0.05)
    style.headroom(axis_handle, highest)
    style.style_axes(axis_handle, "corruption probability (per frame)", "VOT-LT F")
    axis_handle.legend(loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    style.save(figure_handle, out_dir / "fig_corruption")

    with open(out_dir / "table_votlt_by_corruption.csv", "w") as handle:
        handle.write("arm," + ",".join(f"p{p:g}" for p in probabilities) + "\n")
        for label, scores in rows.items():
            handle.write(label + "," + ",".join(f"{value:.4f}" for value in scores) + "\n")

    if degenerate_arms:
        print(f"\nNOTE: no confidence for {', '.join(degenerate_arms)} -- every frame reported, "
              f"so precision = recall = mean overlap.")


if __name__ == "__main__":
    main()

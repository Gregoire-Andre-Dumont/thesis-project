"""claim_4, second figure: can a score tell a confident TARGET mask from a confident DISTRACTOR mask?

The usable-proposal AUC asks whether a score knows a good mask from a bad one, and both scores do that well
on clean rollouts. This asks the harder question the memory bank actually fails on: SAM has produced a
clean, confident mask -- of the wrong person. Nothing about the mask's quality distinguishes the two cases.
Only appearance does.

The frames are those holding BOTH kinds of proposal at once: one whose true mask IoU clears `CONFIDENT`, so
it is on the target, and another whose overlap with some distractor clears the same bar, so it is on
somebody else. Restricting to frames that contain both is what makes the comparison fair -- the score is
asked to separate two masks it saw in the same image, not a good frame from a bad one.

Label 1 for the target proposal, 0 for the distractor proposal; the metric is the AUC of separating them.
Chance is 0.50, and 0.50 here means a score that cannot tell the target from a distractor at all.

    python notebooks/claim_4_distractor.py [confident=0.5]
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper"))

import style
from src.offline_training.main_dataset import MainDataset

PREDICTIONS = Path("data/claim_4/predictions.pkl")
DATASET = Path("data/mask_oracle")
FIGURES = Path("data/claim_4/paper")

CONFIDENT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5   # a proposal is "on" something at this IoU
SCORES = [("SAM IoU token", "sam", 0), ("SAMARA gate", "cnn", 1)]


def distractor_overlap(level, stems):
    """Each proposal's best overlap with any distractor, stacked over `stems` exactly as claim_4 stacks its
    scores: same folder, same frame filter, same order, trajectories with nothing scorable skipped."""

    dataset = MainDataset(dataset_path=str(DATASET), probabilities=[level])
    folder = dataset.folders()[0]

    columns = []
    for stem in stems:
        experiment = pickle.load(open(folder / f"{stem}.pkl", "rb"))
        keep = dataset.scorable(experiment)
        if not keep.any():
            continue
        overlaps = np.asarray(experiment.distractor_iou, dtype=np.float32)[keep]   # (frames, proposals, d)
        columns.append(overlaps.max(axis=2))                                       # best distractor per proposal
    return np.concatenate(columns)


def separation(score, target_iou, distractor_iou):
    """AUC separating confident target proposals from confident distractor ones, on frames holding both."""

    on_target = target_iou > CONFIDENT
    on_distractor = (distractor_iou > CONFIDENT) & ~on_target      # a proposal cannot be both
    both_present = on_target.any(axis=1) & on_distractor.any(axis=1)

    labels = np.concatenate([np.ones(int(on_target[both_present].sum())),
                             np.zeros(int(on_distractor[both_present].sum()))])
    values = np.concatenate([score[both_present][on_target[both_present]],
                             score[both_present][on_distractor[both_present]]])

    keep = np.isfinite(values)
    labels, values = labels[keep], values[keep]
    if labels.min() == labels.max():
        return np.nan, int(both_present.sum())
    return float(roc_auc_score(labels, values)), int(both_present.sum())


def main():
    predictions = pickle.load(open(PREDICTIONS, "rb"))
    style.use_paper_style()

    curves = {key: {} for _, key, _ in SCORES}
    frames_per_level = {}

    for level, folds in sorted(predictions.items()):
        if not folds:
            continue
        for fold in folds:
            distractor_iou = distractor_overlap(level, fold["trajectories"])
            target_iou = np.asarray(fold["truth"], dtype=float)
            if distractor_iou.shape != target_iou.shape:
                raise SystemExit(f"p{level}: distractor {distractor_iou.shape} != truth {target_iou.shape}")

            for _, key, _ in SCORES:
                area, frames = separation(np.asarray(fold[key], float), target_iou, distractor_iou)
                curves[key].setdefault(level, []).append(area)
                frames_per_level[level] = frames_per_level.get(level, 0) + frames

        report = "   ".join(f"{name} {np.nanmean(curves[key][level]):.3f}" for name, key, _ in SCORES)
        print(f"  p{level:.2f}: {report}   frames={frames_per_level[level]}", flush=True)

    figure, axis = plt.subplots(figsize=(style.COLUMN, 3.1))
    highest = 0.0
    for name, key, slot in SCORES:
        levels = sorted(curves[key])
        folds = np.array([curves[key][level] for level in levels])
        appearance = style.series_style(slot)
        axis.fill_between(levels, np.nanmin(folds, 1), np.nanmax(folds, 1), color=appearance["color"],
                          alpha=0.13, linewidth=0, zorder=2)
        axis.plot(levels, np.nanmean(folds, 1), label=name, zorder=3, **appearance)
        highest = max(highest, float(np.nanmax(np.nanmean(folds, 1))))

    axis.axhline(0.5, color=style.INK2, linestyle=(0, (1, 2)), linewidth=0.8, zorder=2)
    levels = sorted(curves[SCORES[0][1]])
    axis.set_xticks(levels)
    axis.set_xticklabels([f"{level:g}" for level in levels])
    span = levels[-1] - levels[0]
    axis.set_xlim(levels[0] - 0.04 * span, levels[-1] + 0.04 * span)
    style.gridlines(axis, 0.05)
    style.headroom(axis, highest)
    style.style_axes(axis, "corruption probability (per frame)", "target-vs-distractor AUC")
    axis.legend(loc="best")

    style.save(figure, FIGURES / f"fig_target_vs_distractor_iou{round(CONFIDENT * 100):02d}")
    print(f"   caption: Separating a confident TARGET mask from a confident DISTRACTOR mask, on frames "
          f"holding both (target IoU > {CONFIDENT:g} and distractor IoU > {CONFIDENT:g}). Mean over "
          f"trajectory-grouped folds, band is the fold spread. The dotted line is chance (0.50).")


if __name__ == "__main__":
    main()

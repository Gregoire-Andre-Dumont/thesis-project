"""claim_5 table: VOT-LT F for SAMARA and the published baselines, split by occlusion length.

VOT-LT is the primary metric. It scores a tracker on the frames it CHOOSES to report, so a controller that
stays silent through an occlusion is credited for the silence rather than only for what it says afterwards
-- which coverage@0.5 cannot express.

THE OPERATING POINT IS CHOSEN ONCE, ON HELD-OUT FOLDS.

  * ONE threshold for the whole table. Re-sweeping inside each occlusion bin would report what an arm
    could achieve in that regime if it knew which regime it was in; a deployed controller ships one
    threshold and meets every occlusion length with it.
  * That threshold, and the confidence it applies to, are picked on the TRAINING folds and applied to the
    held-out fold. Picking them on the same clips the number is reported on is selection on the evaluation
    set and inflates every arm -- most the ones with the most signals to choose between.

Each clip is therefore scored at a threshold chosen without having seen it. The bins then just group
clips; they never re-tune.

WINDOW=1 restricts scoring to the memory window -- every occluded frame plus only the visible frames AFTER
the last occlusion.

    python notebooks/claim_5_visualize.py [arms_dir] [out_dir]
    WINDOW=1 python notebooks/claim_5_visualize.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.vot_lt import Predictions

STEPS = 61
N_BINS = 4
N_FOLDS = 5
WINDOW = os.environ.get("WINDOW", "0") != "0"

# ONE confidence per arm, fixed in advance -- no sweep over signals.
#
# Every baseline is scored on SAM's OBJECT SCORE: the presence logit, which is the signal a long-term
# tracker is supposed to report on. Sweeping each arm over both of its scores and keeping the better one
# would quietly reward whichever arm happens to own two usable heads. SAMURAI's run predates the split
# columns and keeps the object logit in `memory_scores` column 1.
#
# SAMARA is scored on its GATE PROBABILITY, because that is its contribution: the commit decision is the
# thing under test, and the gate probability is what a deployed SAMARA would report on.
#
# Two arms are deliberately absent. DAM4SAM runs `tm/sam2.1_hiera_large.pt` where every other arm here --
# and the oracles in claim_1 -- runs `tm/sam_large.pt`, because the 2.0 weights lack keys its architecture
# needs; its row would confound a better BACKBONE with a better memory. SAM2Long's run stalled at 57 clips
# and the table scores the intersection, so including it would cut every other arm to 57.
BASELINES = [
    ("SAM 2",   "data/baselines_votlt/sam2/results.pkl",    "object_score"),
    ("SAMITE",  "data/baselines_votlt/samite/results.pkl",  "object_score"),
    ("SAMURAI", "data/baselines/samurai/results.pkl",       "memory_scores:1"),
]
SAMARA_SIGNAL = "commit_probability"


def keyed(results_path):
    clips = pickle.load(open(results_path, "rb"))["clips"]
    return {(record["video"], record["person"]): record for record in clips}




def values(record, signal):
    """One confidence as a per-frame vector; `field:column` selects a column."""

    field, _, column = signal.partition(":")
    value = np.asarray(record[field], dtype=float)
    value = value[:, int(column)] if column else (np.nanmax(value, axis=1) if value.ndim == 2 else value)
    return np.nan_to_num(value, nan=-np.inf)


def curves(scored_clips, signal, clip_keys):
    """`(precision, recall, grid)` -- per-clip curves over a shared threshold grid.

    Rows follow `clip_keys`, so a fold can index them directly."""

    vectors = []
    for key in clip_keys:
        record = scored_clips[key]
        n_frames = len(record["occluded"])
        vector = values(record, signal)
        vectors.append(vector[-n_frames:] if len(vector) >= n_frames
                       else np.pad(vector, (0, n_frames - len(vector)), constant_values=-np.inf))

    pool = np.concatenate(vectors)
    pool = pool[np.isfinite(pool)]
    if not len(pool):
        return None
    grid = np.unique(np.quantile(pool, np.linspace(0, 1, STEPS)))

    # `sequence_curves` DROPS a clip with no present frame, which the memory window creates whenever a
    # target never reappears after its last occlusion. The returned rows would then no longer line up with
    # `clip_keys`, so the surviving clips are tracked here and every caller indexes through `scored`.
    predictions = Predictions()
    scored = []
    for position, (key, vector) in enumerate(zip(clip_keys, vectors)):
        record = scored_clips[key]
        occluded = np.asarray(record["occluded"], dtype=bool)
        kept = np.asarray(record["has_box"], dtype=bool).copy()
        if WINDOW and occluded.any():
            kept[: int(np.where(occluded)[0][-1]) + 1] = False
        present = kept & ~occluded
        if not present.any():
            continue
        predictions.add(record["samara"], vector, occluded, kept)
        scored.append(position)

    if not scored:
        return None
    precision, recall = predictions.sequence_curves(grid)
    return precision, recall, grid, np.asarray(scored)


def f_from(precision, recall):
    """Harmonic mean of the AVERAGED precision and recall -- the macro statistic."""

    denominator = precision + recall
    return 0.0 if denominator <= 0 else float(2 * precision * recall / denominator)


def cross_validated(scored_clips, signal, clip_keys, rng):
    """Per-clip (precision, recall) at a threshold chosen WITHOUT that clip.

    For each fold: pick the threshold maximising macro F on the other folds, then read this fold's clips
    off that operating point. The returned arrays therefore contain no information the threshold was
    fitted on, so binning them afterwards cannot re-tune anything. The signal is fixed in advance, so the
    only thing cross-validation protects is the threshold."""

    result = curves(scored_clips, signal, clip_keys)
    if result is None:
        return None
    precision, recall, grid, scored = result

    # NaN for clips the window left with nothing to score; the binning means are nan-aware.
    out_precision = np.full(len(clip_keys), np.nan)
    out_recall = np.full(len(clip_keys), np.nan)

    order = rng.permutation(len(scored))
    for fold in np.array_split(order, N_FOLDS):
        train = np.setdiff1d(order, fold)
        if not len(train) or not len(fold):
            continue
        f_scores = [f_from(precision[train, i].mean(), recall[train, i].mean()) for i in range(len(grid))]
        index = int(np.argmax(f_scores))
        out_precision[scored[fold]] = precision[fold, index]
        out_recall[scored[fold]] = recall[fold, index]

    return out_precision, out_recall


def occlusion_bins(reference, clip_keys):
    """Four QUANTILE bins on the number of occluded frames, plus a pooled column."""

    counts = np.array([reference[key]["occ_count"] for key in clip_keys], dtype=float)
    edges = np.quantile(counts, np.linspace(0, 1, N_BINS + 1))
    bins, labels = [], []
    for index in range(N_BINS):
        low, high = edges[index], (np.inf if index == N_BINS - 1 else edges[index + 1])
        bins.append((low, high))
        labels.append(f"{int(low)}-{int(counts.max()) if np.isinf(high) else int(high)}")
    return bins + [(0.0, np.inf)], labels + ["all"]


def main():
    arms_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/claim_5")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/sam-2/claim_5")

    loaded = {}
    for label, path, signal in BASELINES:
        if Path(path).exists():
            loaded[label] = (keyed(path), signal)
    for path in sorted(arms_dir.glob("*/results.pkl")):
        loaded[f"SAMARA {path.parent.name}"] = (keyed(path), SAMARA_SIGNAL)
    if not loaded:
        raise SystemExit(f"no arms found under {arms_dir} and no baselines present")

    clip_keys = sorted(set.intersection(*(set(scored) for scored, _ in loaded.values())))
    reference = next(iter(loaded.values()))[0]
    bins, labels = occlusion_bins(reference, clip_keys)
    counts = np.array([reference[key]["occ_count"] for key in clip_keys], dtype=float)

    print(f"VOT-LT F  ·  {len(clip_keys)} clips  ·  window={WINDOW}  ·  "
          f"{N_FOLDS}-fold threshold, one operating point per arm")
    print()
    header = f"{'arm':16s}" + "".join(f"{label:>12s}" for label in labels)
    print(header)
    print("-" * len(header))

    rows = {}
    for label, (scored_clips, signal) in loaded.items():
        result = cross_validated(scored_clips, signal, clip_keys, np.random.default_rng(0))
        if result is None:
            continue
        precision, recall = result

        cells = []
        for low, high in bins:
            inside = (counts >= low) & (counts < high)
            cells.append(f_from(np.nanmean(precision[inside]), np.nanmean(recall[inside]))
                         if inside.any() else float("nan"))
        rows[label] = cells
        print(f"{label:16s}" + "".join(f"{value:12.4f}" for value in cells))

    per_bin = [int(((counts >= low) & (counts < high)).sum()) for low, high in bins]
    print("-" * len(header))
    print(f"{'n clips':16s}" + "".join(f"{count:12d}" for count in per_bin))


    out_dir.mkdir(parents=True, exist_ok=True)
    name = "table_votlt_by_occlusion" + ("_window" if WINDOW else "") + ".csv"
    with open(out_dir / name, "w") as handle:
        handle.write("arm," + ",".join(labels) + "\n")
        for label, cells in rows.items():
            handle.write(label + "," + ",".join(f"{value:.4f}" for value in cells) + "\n")
        handle.write("n clips," + ",".join(str(count) for count in per_bin) + "\n")


if __name__ == "__main__":
    main()

"""claim_1 table: VOT-LT F for the published baselines AND the oracles, split by occlusion length.

VOT-LT is the primary metric. The coverage@0.5, robustness and memory-hygiene scores this file used to
compute are gone, as is the anchor-area split -- occlusion length is the covariate the claim is about.

The oracles read ground truth, so their margin over the baselines is the HEADROOM a deployable commit gate
could recover. Putting them in one table is the point of claim_1: the baselines say where the field is, the
oracles say how much is left on the table.

The baselines cannot be rolled inside claim_1 -- each fork vendors its own top-level `sam2` package and
only one can be on the path per interpreter -- so they are JOINED from their own runs on the clips every
arm scored.

FAIRNESS: the oracles' stored records predate confidence recording. VOT-LT scores a tracker on the frames
it chooses to report, so an arm with a confidence can earn precision by abstaining and an arm without one
cannot. Mixing the two in one table would hand the baselines ~0.06 F for free. So when ANY arm lacks a
confidence, every arm is scored always-report and the table says so. Re-run claim_1.py -- which now records
confidences -- to get the swept version.

    python notebooks/claim_1_visualize.py [results.pkl] [out_dir]
    WINDOW=1 python notebooks/claim_1_visualize.py
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
WINDOW = os.environ.get("WINDOW", "0") != "0"
THRESHOLD = float(os.environ.get("THRESHOLD", "0.2"))


BASELINES = [
    ("SAM 2",   "data/baselines_votlt/sam2/results.pkl",    ["object_score", "iou_token"]),
    ("SAMITE",  "data/baselines_votlt/samite/results.pkl",  ["object_score", "iou_token"]),
    ("SAMURAI", "data/baselines/samurai/results.pkl",       ["memory_scores:0", "memory_scores:1"]),
]

ORACLES = [("memory oracle", "memory"), ("mask oracle", "mask")]
ORACLE_CONFIDENCES = ["__true_iou__"]


def at_threshold(value):
    """The swept oracles store {threshold: array}; everything else stores the array itself."""

    if isinstance(value, dict):
        if THRESHOLD in value:
            return value[THRESHOLD]
        return value[min(value, key=lambda key: abs(float(key) - THRESHOLD))]
    return value


def values(record, signal):
    """One confidence as a per-frame vector; `field:column` selects a column."""

    field, _, column = signal.partition(":")
    value = np.asarray(record[field], dtype=float)
    value = value[:, int(column)] if column else (np.nanmax(value, axis=1) if value.ndim == 2 else value)
    return np.nan_to_num(value, nan=-np.inf)


def frame_masks(record):
    """`(occluded, evaluated)` for one clip, with the memory window applied when asked for."""

    occluded = np.asarray(record["occluded"], dtype=bool)
    kept = np.asarray(record["has_box"], dtype=bool).copy()
    if WINDOW and occluded.any():
        kept[: int(np.where(occluded)[0][-1]) + 1] = False
    return occluded, kept


def always_report(arm, clip_keys):
    """VOT-LT with every frame reported: precision = recall = mean overlap over the scored frames."""

    predictions = Predictions()
    for key in clip_keys:
        record, overlap = arm.record(key), np.asarray(arm.overlap(key), dtype=float)
        occluded, kept = frame_masks(record)
        predictions.add(overlap, np.ones(len(overlap)), occluded, kept)
    return predictions.macro_scores([0.5])[3]


def swept(arm, clip_keys):
    """Best VOT-LT F over the arm's confidences, each on its own quantile grid."""

    best = 0.0
    for signal in arm.signals:
        vectors = [arm.confidence(key, signal) for key in clip_keys]
        if any(vector is None for vector in vectors):
            continue
        pool = np.concatenate(vectors)
        pool = pool[np.isfinite(pool)]
        if not len(pool):
            continue
        thresholds = np.unique(np.quantile(pool, np.linspace(0, 1, STEPS)))

        predictions = Predictions()
        for key, vector in zip(clip_keys, vectors):
            record = arm.record(key)
            occluded, kept = frame_masks(record)
            predictions.add(arm.overlap(key), vector, occluded, kept)
        best = max(best, predictions.macro_scores(thresholds)[3])
    return best


class Arm:
    """One row of the table: where its overlap lives, where its confidence lives, and whether it has one."""

    def __init__(self, label, clips, overlap_key, signals, from_claim_1):
        self.label, self.clips, self.overlap_key = label, clips, overlap_key
        self.signals, self.from_claim_1 = signals, from_claim_1

    def record(self, key):
        return self.clips[key]

    def overlap(self, key):
        record = self.clips[key]
        return at_threshold(record[self.overlap_key]) if self.from_claim_1 else record["samara"]

    def confidence(self, key, signal):
        record = self.clips[key]
        n_frames = len(record["occluded"])
        if signal == "__true_iou__":
            # An oracle reports on what it can verify: the true IoU of the mask it just produced.
            value = np.nan_to_num(np.asarray(self.overlap(key), dtype=float), nan=0.0)
        elif self.from_claim_1:
            stored = record.get(f"{self.overlap_key}_confidence")
            if stored is None:
                return None
            stored = at_threshold(stored)
            if signal not in stored:
                return None
            value = np.nan_to_num(np.asarray(stored[signal], dtype=float), nan=-np.inf)
        else:
            value = values(record, signal)
        return value[-n_frames:] if len(value) >= n_frames else np.pad(
            value, (0, n_frames - len(value)), constant_values=-np.inf)

    def has_confidence(self, clip_keys):
        return any(all(self.confidence(key, signal) is not None for key in clip_keys)
                   for signal in self.signals)


def occlusion_bins(reference, clip_keys):
    """Four QUANTILE bins on the number of occluded frames, plus a pooled column.

    Quantiles rather than fixed cut-points so every bin rests on roughly the same number of clips; a fixed
    grid puts most of the draw in one bin and leaves the long tail too thin to read."""

    counts = np.array([reference[key]["occ_count"] for key in clip_keys], dtype=float)
    edges = np.quantile(counts, np.linspace(0, 1, N_BINS + 1))
    bins, labels = [], []
    for index in range(N_BINS):
        low, high = edges[index], (np.inf if index == N_BINS - 1 else edges[index + 1])
        bins.append((low, high))
        labels.append(f"{int(low)}-{int(counts.max()) if np.isinf(high) else int(high)}")
    return bins + [(0.0, np.inf)], labels + ["all"]


def main():
    results_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/claim_1/results.pkl")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/sam-2/claim_1")

    oracle_clips = {(r["video"], r["person"]): r
                    for r in pickle.load(open(results_path, "rb"))["clips"]}

    arms = []
    for label, path, signals in BASELINES:
        if not Path(path).exists():
            continue
        clips = {(r["video"], r["person"]): r for r in pickle.load(open(path, "rb"))["clips"]}
        arms.append(Arm(label, clips, None, signals, from_claim_1=False))

    for label, arm_key in ORACLES:
        arms.append(Arm(label, oracle_clips, arm_key, ORACLE_CONFIDENCES, from_claim_1=True))

    clip_keys = sorted(set.intersection(*(set(arm.clips) for arm in arms)))
    clip_keys = [key for key in clip_keys
                 if all(not arm.from_claim_1 or at_threshold(arm.clips[key].get(arm.overlap_key)) is not None
                        for arm in arms)]
    if not clip_keys:
        raise SystemExit("no clip is scored by every baseline and both oracles")

    missing = [arm.label for arm in arms if not arm.has_confidence(clip_keys)]
    score = always_report if missing else swept

    reference = arms[0].clips
    bins, labels = occlusion_bins(reference, clip_keys)

    rows = {}
    for arm in arms:
        cells = []
        for low, high in bins:
            keys = [key for key in clip_keys if low <= reference[key]["occ_count"] < high]
            cells.append(score(arm, keys) if keys else float("nan"))
        rows[arm.label] = cells
        print(f"{arm.label:16s}" + "".join(f"{value:12.4f}" for value in cells))


    out_dir.mkdir(parents=True, exist_ok=True)
    name = "table_votlt_by_occlusion" + ("_window" if WINDOW else "") + ".csv"
    with open(out_dir / name, "w") as handle:
        handle.write("arm," + ",".join(labels) + "\n")
        for label, cells in rows.items():
            handle.write(label + "," + ",".join(f"{value:.4f}" for value in cells) + "\n")
        handle.write("n clips," + ",".join(str(count) for count in counts) + "\n")


if __name__ == "__main__":
    main()

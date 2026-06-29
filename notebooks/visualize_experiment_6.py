"""Compare experiment_6 (down-weight / upsample easy positives) against the
unmodified baseline `samara_fixed_memory_20` from experiment_4 and SAM baseline
from experiment_1.

Trajectories are MEDIAN-SPLIT into EARLY vs LATE first occlusion (median computed
over the trajectories in the joint pool that have at least one occlusion).
Counting starts at the chosen anchor (index 0).

Joint pool: only trajectories where ALL trackers produced records."""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.metrics import coverage


OUTPUT_PATH = Path("data/visualize_experiment_6.png")
METRIC_THRESHOLD = 0.5
THRESHOLD_SWEEP = np.linspace(0.0, 1.0, 21)

TRACKERS = [
    ("sam_baseline",                            "tab:red",     "SAM baseline (experiment_1)",                            1),
    ("samara_fixed_memory_20",                  "tab:blue",    "Memory @ pad 0.25, baseline (experiment_4)",             4),
    ("samara_fixed_memory_20_downweight_easy",  "tab:orange",  "Memory @ pad 0.25, downweight easy",                     6),
    ("samara_fixed_memory_20_upsample_easy",    "tab:green",   "Memory @ pad 0.25, upsample easy",                       6),
    ("samara_fixed_memory_20_upsample_hard",    "tab:purple",  "Memory @ pad 0.25, upsample HARD trajectories (3×)",     6),
]
OCCLUSION_BUCKETS = ("early", "late")


def load(name, experiment_id):
    path = Path(f"data/experiment_{experiment_id}") / f"{name}.pkl"
    if not path.exists():
        return None
    return pickle.load(open(path, "rb"))["records"]


def first_occlusion_frame(record):
    occluded = np.where(np.asarray(record["occlusions"]) > 0.5)[0]
    return int(occluded[0]) if len(occluded) else 10**9


def coverage_for_keys(records_for_tracker, keys, threshold):
    per_trajectory = [coverage(records_for_tracker[key]["iou_scores"],
                                          records_for_tracker[key]["occlusions"], threshold)
                              for key in keys if key in records_for_tracker]
    return [value for value in per_trajectory if not np.isnan(value)]


def main():
    records_by_key_pair = {}
    for name, _, _, experiment_id in TRACKERS:
        records = load(name, experiment_id)
        track_key = (experiment_id, name)
        if records is None:
            print(f"MISSING data/experiment_{experiment_id}/{name}.pkl")
            continue
        records_by_key_pair[track_key] = records
    if not records_by_key_pair:
        print("no pickles found — nothing to visualize")
        return

    shared = set.intersection(*[
        {(record["video_name"], record["person_id"]) for record in records}
        for records in records_by_key_pair.values()])

    records_by_key = {track_key: {(r["video_name"], r["person_id"]): r for r in records}
                              for track_key, records in records_by_key_pair.items()}

    reference_track_key = next(iter(records_by_key_pair))
    first_occlusions = {key: first_occlusion_frame(records_by_key[reference_track_key][key])
                                  for key in shared}
    finite_occlusions = np.array([f for f in first_occlusions.values() if f < 10**9])
    if len(finite_occlusions) == 0:
        print("no trajectories with any occlusion in the shared pool")
        return
    split_value = float(np.median(finite_occlusions))
    print(f"first-occlusion median split at frame {split_value:.0f}  "
              f"(n={len(finite_occlusions)} of {len(shared)} shared trajectories have an occlusion)")

    early_keys = {key for key, frame in first_occlusions.items()
                          if frame < 10**9 and frame < split_value}
    late_keys  = {key for key, frame in first_occlusions.items()
                          if frame < 10**9 and frame >= split_value}
    bucket_keys = {"early": early_keys, "late": late_keys}
    print(f"  EARLY (first occlusion < {split_value:.0f}):  n={len(early_keys)}")
    print(f"  LATE  (first occlusion >= {split_value:.0f}): n={len(late_keys)}")

    figure, axes_grid = plt.subplots(2, 2, figsize=(14, 11))

    for row_index, bucket in enumerate(OCCLUSION_BUCKETS):
        keys_in_bucket = bucket_keys[bucket]
        axis_curves = axes_grid[row_index, 0]
        axis_bars = axes_grid[row_index, 1]

        bar_values, bar_counts, bar_colors, bar_labels = [], [], [], []
        for name, color, label, experiment_id in TRACKERS:
            track_key = (experiment_id, name)
            if track_key not in records_by_key:
                continue
            sweep_values = []
            for threshold in THRESHOLD_SWEEP:
                values = coverage_for_keys(records_by_key[track_key], keys_in_bucket, float(threshold))
                sweep_values.append(np.mean(values) if values else np.nan)
            axis_curves.plot(THRESHOLD_SWEEP, sweep_values, linewidth=2, marker="o",
                                       markersize=6, markeredgecolor="black",
                                       markeredgewidth=0.5, color=color, label=label)

            values_at_threshold = coverage_for_keys(records_by_key[track_key], keys_in_bucket,
                                                                          METRIC_THRESHOLD)
            mean_value = float(np.mean(values_at_threshold)) if values_at_threshold else float("nan")
            bar_values.append(mean_value)
            bar_counts.append(len(values_at_threshold))
            bar_colors.append(color)
            bar_labels.append(label)

        axis_curves.axvline(METRIC_THRESHOLD, color="gray", linestyle="--",
                                    linewidth=1, alpha=0.6)
        axis_curves.set_xlabel("IoU threshold")
        axis_curves.set_ylabel("Post-first-occlusion coverage")
        axis_curves.set_title(f"{bucket.upper()} occlusion  (n={len(keys_in_bucket)})",
                                      fontsize=11)
        axis_curves.set_xlim(0, 1); axis_curves.set_ylim(0, 1)
        axis_curves.grid(True, alpha=0.3)
        axis_curves.legend(loc="upper right", fontsize=8)

        bar_positions = np.arange(len(bar_labels))
        axis_bars.bar(bar_positions, bar_values, color=bar_colors,
                            edgecolor="black", linewidth=0.6)
        for position, value, count in zip(bar_positions, bar_values, bar_counts):
            if not np.isnan(value):
                axis_bars.text(position, value + 0.015, f"{value:.3f}", ha="center", fontsize=11)
            axis_bars.text(position, 0.02, f"n={count}", ha="center", fontsize=9, color="gray")
        axis_bars.set_xticks(bar_positions)
        axis_bars.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=8)
        axis_bars.set_ylabel(f"Coverage at IoU > {METRIC_THRESHOLD}")
        axis_bars.set_title(f"{bucket.upper()} occlusion  (n={len(keys_in_bucket)})",
                                  fontsize=11)
        axis_bars.set_ylim(0, 1)
        axis_bars.grid(True, alpha=0.3, axis="y")

    figure.suptitle(
        f"Experiment 6 (downweight / upsample easy positives) vs baseline  "
        f"(median split at frame {split_value:.0f})",
        fontsize=12)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.close(figure)
    print(f"saved {OUTPUT_PATH}")

    print()
    for bucket in OCCLUSION_BUCKETS:
        keys_in_bucket = bucket_keys[bucket]
        print(f"  === {bucket.upper()} (n={len(keys_in_bucket)}) ===")
        for name, _, label, experiment_id in TRACKERS:
            track_key = (experiment_id, name)
            if track_key not in records_by_key:
                continue
            values = coverage_for_keys(records_by_key[track_key], keys_in_bucket, METRIC_THRESHOLD)
            mean_value = float(np.mean(values)) if values else float("nan")
            print(f"    {label:<55} coverage@IoU>{METRIC_THRESHOLD} = {mean_value:.3f}  (n={len(values)})")


if __name__ == "__main__":
    main()

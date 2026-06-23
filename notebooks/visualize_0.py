"""Coverage at IoU > 0.5 for three samara_fixed_memory_foreground calibrators trained on
different anchor-size partitions of the training pool:

  - small  → trained only on trajectories whose visible anchor area < median
  - large  → trained only on trajectories whose visible anchor area >= median
  - both   → trained on all training trajectories (no size filter)

All three are deployed on the same test set. Compares overall coverage, and also
breaks it down by the test-trajectory's anchor-size bin to detect specialization
benefits."""

import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.metrics import coverage


INPUT_DIRECTORY = Path("data")
OUTPUT_DIRECTORY = Path("data")
VISIBLE_DIRECTORY = Path("data/person_path/visible")
DATASET_DIRECTORY = Path("data/datasets/memory_anchor/clean")
METRIC_THRESHOLD = 0.5
SAM_RESIZE = 1024
EXPERIMENT_NUMBER = 0

VARIANTS = [
    ("small",        "tab:orange",  "small-anchor training"),
    ("large",        "tab:purple",  "large-anchor training"),
    ("both",         "tab:blue",    "all-anchor training"),
    ("sam_baseline", "tab:red",     "sam_baseline (no calibrator)"),
]


def coverage_per_record(record):
    return coverage(record["iou_scores"], record["occlusions"], METRIC_THRESHOLD)


def load(experiment, name):
    path = INPUT_DIRECTORY / f"experiment_{experiment}" / f"{name}.pkl"
    if not path.exists():
        return None
    return pickle.load(open(path, "rb"))["records"]


def _trajectory_frame_indices(video_name, person_id):
    path = DATASET_DIRECTORY / f"{video_name}_{person_id}.pkl"
    if not path.exists():
        return None
    return pickle.load(open(path, "rb")).frame_indices


def _visible_area_at_chosen_anchor(video_name, person_id):
    frame_indices = _trajectory_frame_indices(video_name, person_id)
    if frame_indices is None or len(frame_indices) == 0:
        return None
    anchor_video_frame = int(frame_indices[0])
    json_path = VISIBLE_DIRECTORY / f"{video_name}.json"
    if not json_path.exists():
        return None
    data = json.load(open(json_path))
    image_width = int(data["metadata"]["resolution"]["width"])
    image_height = int(data["metadata"]["resolution"]["height"])
    scale = SAM_RESIZE / max(image_width, image_height)
    best_bbox, best_diff = None, None
    for entity in data["entities"]:
        if entity["id"] != person_id:
            continue
        diff = abs(int(entity["blob"]["frame_idx"]) - anchor_video_frame)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_bbox = entity["bb"]
    if best_bbox is None:
        return None
    _, _, w, h = best_bbox
    return max(0.0, w) * max(0.0, h) * scale * scale


def main():
    records_by_variant = {}
    for variant, _, _ in VARIANTS:
        if variant == "sam_baseline":
            name = "sam_baseline_20"
        else:
            name = f"samara_fixed_memory_foreground_20_{variant}"
        records = load(EXPERIMENT_NUMBER, name)
        if records is None:
            print(f"MISSING data/experiment_{EXPERIMENT_NUMBER}/{name}.pkl")
            continue
        records_by_variant[variant] = records

    if not records_by_variant:
        print("no pickles found")
        return

    shared = set.intersection(*[
        {(r["video_name"], r["person_id"]) for r in records}
        for records in records_by_variant.values()])
    print(f"shared intersection: n={len(shared)}")

    records_by_key = {variant: {(r["video_name"], r["person_id"]): r for r in records}
                       for variant, records in records_by_variant.items()}

    test_anchor_areas = {k: _visible_area_at_chosen_anchor(k[0], k[1]) for k in shared}
    valid_areas = np.array([a for a in test_anchor_areas.values() if a is not None])
    test_median = float(np.median(valid_areas)) if len(valid_areas) else 0.0
    print(f"test-pool median anchor area: {test_median:.0f} px² (1024-resize)")

    test_bins = {"small": set(), "large": set(), "all": set(shared)}
    for key, area in test_anchor_areas.items():
        if area is None:
            continue
        test_bins["small" if area < test_median else "large"].add(key)
    for bin_name, keys in test_bins.items():
        print(f"  test bin '{bin_name}': n={len(keys)}")

    matrix = {}                           # matrix[train_variant][test_bin] = mean coverage
    for variant, _, _ in VARIANTS:
        if variant not in records_by_variant:
            continue
        matrix[variant] = {}
        for bin_name, keys in test_bins.items():
            records = [records_by_key[variant][k] for k in keys if k in records_by_key[variant]]
            values = [coverage_per_record(r) for r in records]
            values = [v for v in values if v is not None and not np.isnan(v)]
            matrix[variant][bin_name] = (np.mean(values) if values else np.nan, len(values))

    print(f"\n{'train':<25} {'test=small':<20} {'test=large':<20} {'test=all':<20}")
    print("-" * 90)
    for variant, _, label in VARIANTS:
        if variant not in matrix:
            continue
        row = matrix[variant]
        cells = []
        for bin_name in ("small", "large", "all"):
            mean, n = row[bin_name]
            cells.append(f"{mean:.4f} (n={n})")
        print(f"  {label:<23} " + "   ".join(f"{c:<18}" for c in cells))

    figure, axes_one = plt.subplots(figsize=(10, 6))
    bin_order = ["small", "large", "all"]
    n_groups = len(bin_order)
    n_variants_present = sum(1 for v, _, _ in VARIANTS if v in matrix)
    bar_width = 0.8 / n_variants_present
    xs_base = np.arange(n_groups)

    variant_index = 0
    for variant, color, label in VARIANTS:
        if variant not in matrix:
            continue
        ys = [matrix[variant][bin_name][0] for bin_name in bin_order]
        ns = [matrix[variant][bin_name][1] for bin_name in bin_order]
        offset = (variant_index - (n_variants_present - 1) / 2) * bar_width
        bars = axes_one.bar(xs_base + offset, ys, width=bar_width,
                                 color=color, edgecolor="black", linewidth=0.8,
                                 label=label)
        for bar, y, n in zip(bars, ys, ns):
            if not np.isnan(y):
                axes_one.annotate(f"{y:.3f}\n(n={n})", xy=(bar.get_x() + bar.get_width()/2, y),
                                       xytext=(0, 3), textcoords="offset points",
                                       ha="center", fontsize=8)
        variant_index += 1

    axes_one.set_xticks(xs_base)
    axes_one.set_xticklabels(["test = small\nanchors",
                                   "test = large\nanchors",
                                   "test = all\nanchors"], fontsize=10)
    axes_one.set_ylabel(f"Coverage (IoU > {METRIC_THRESHOLD})")
    axes_one.set_title(f"Train on small / large / both anchors (split at training-pool median); "
                          f"test on the same {len(shared)} trajectories", fontsize=11)
    axes_one.grid(True, axis="y", alpha=0.3)
    axes_one.legend(loc="best", fontsize=10, title="Training data")
    figure.tight_layout()
    output_path = OUTPUT_DIRECTORY / "visualize_0.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()

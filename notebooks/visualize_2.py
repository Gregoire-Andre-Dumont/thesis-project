"""Coverage at IoU > 0.5 for the size-dispatch tracker vs sam_baseline at threshold 0.2,
restricted to trajectories with EARLY first occlusion (`first_occlusion < 15`) and
binned by anchor visible-bbox area at the chosen anchor into quartiles.

samara_dispatch_size_20 uses the samara calibrator on SMALL anchors (visible area
< 48² at 1024-resize) and SAM 2's pred_iou gate on LARGE anchors — so the goal is to
see whether it picks the right side per quartile and beats sam_baseline on the small
end while staying competitive on the large end."""

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
DATASET_DIRECTORY = Path("data/datasets/hiera_anchor/clean")
METRIC_THRESHOLD = 0.5
SAM_RESIZE = 1024
MAX_FIRST_OCCLUSION = 20

# (experiment_num, tracker_name, line_color, label)
FAMILIES = [
    (5,  "sam_baseline_20",                    "tab:red",   "sam_baseline_20"),
    (6,  "samara_fixed_memory_foreground_20",  "tab:green", "samara_fixed_memory_foreground_20"),
    (10, "samara_dispatch_size_20",            "tab:blue",  "samara_dispatch_size_20"),
]
ANCHOR_SIZE_LABELS = ["Q1", "Q2", "Q3", "Q4"]


def coverage_per_record(record):
    return coverage(record["iou_scores"], record["occlusions"], METRIC_THRESHOLD)


def load(exp, name):
    path = INPUT_DIRECTORY / f"experiment_{exp}" / f"{name}.pkl"
    if not path.exists():
        return None
    return pickle.load(open(path, "rb"))["records"]


def first_occlusion_frame(record):
    occluded = np.where(np.asarray(record["occlusions"]) > 0.5)[0]
    return int(occluded[0]) if len(occluded) else 10**9


def _trajectory_frame_indices(video_name, person_id):
    path = DATASET_DIRECTORY / f"{video_name}_{person_id}.pkl"
    if not path.exists():
        return None
    return pickle.load(open(path, "rb")).frame_indices


def _bbox_area_at_video_frame(json_directory, video_name, person_id, video_frame_idx):
    json_path = Path(json_directory) / f"{video_name}.json"
    if not json_path.exists():
        return None
    data = json.load(open(json_path))
    best_bbox, best_diff = None, None
    for entity in data["entities"]:
        if entity["id"] != person_id:
            continue
        diff = abs(int(entity["blob"]["frame_idx"]) - video_frame_idx)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_bbox = entity["bb"]
    if best_bbox is None:
        return None
    _, _, w, h = best_bbox
    return max(0.0, w) * max(0.0, h)


def visible_area_at_chosen_anchor(video_name, person_id):
    frame_indices = _trajectory_frame_indices(video_name, person_id)
    if frame_indices is None or len(frame_indices) == 0:
        return None
    anchor_video_frame = int(frame_indices[0])
    return _bbox_area_at_video_frame(VISIBLE_DIRECTORY, video_name, person_id, anchor_video_frame)


def main():
    records_by_tracker = {}
    for exp, name, _, _ in FAMILIES:
        records = load(exp, name)
        if records is not None:
            records_by_tracker[name] = records
        else:
            print(f"MISSING data/experiment_{exp}/{name}.pkl")
    if len(records_by_tracker) < 2:
        print("need both pickles to compare")
        return

    shared = set.intersection(*[
        {(r["video_name"], r["person_id"]) for r in records}
        for records in records_by_tracker.values()])
    print(f"shared intersection: n={len(shared)}")

    reference_by_key = {(r["video_name"], r["person_id"]): r
                          for r in next(iter(records_by_tracker.values()))
                          if (r["video_name"], r["person_id"]) in shared}
    early_keys = {k for k in shared
                       if first_occlusion_frame(reference_by_key[k]) < MAX_FIRST_OCCLUSION}
    print(f"early-occlusion filter: kept {len(early_keys)}/{len(shared)} trajectories "
            f"with first_occlusion < {MAX_FIRST_OCCLUSION}")

    anchor_areas_by_key = {k: visible_area_at_chosen_anchor(k[0], k[1]) for k in early_keys}
    valid_areas = np.array([a for a in anchor_areas_by_key.values() if a is not None])
    size_edges = (np.quantile(valid_areas, [0.25, 0.50, 0.75]) if len(valid_areas)
                       else np.array([0.0, 0.0, 0.0]))
    print(f"anchor-area quartile edges: "
            f"Q1 < {size_edges[0]:.0f} px²  "
            f"<= Q2 < {size_edges[1]:.0f} px²  "
            f"<= Q3 < {size_edges[2]:.0f} px²  "
            f"<= Q4  (visible bbox at chosen anchor, NATIVE pixels)")

    def size_bin(key):
        area = anchor_areas_by_key.get(key)
        if area is None: return None
        if area < size_edges[0]: return 0
        if area < size_edges[1]: return 1
        if area < size_edges[2]: return 2
        return 3

    bins_by_size = {si: set() for si in range(len(ANCHOR_SIZE_LABELS))}
    for key in early_keys:
        si = size_bin(key)
        if si is None:
            continue
        bins_by_size[si].add(key)

    records_by_key = {name: {(r["video_name"], r["person_id"]): r for r in records_by_tracker[name]}
                       for name in records_by_tracker}

    figure, axes_one = plt.subplots(figsize=(9, 6))
    xs = list(range(len(ANCHOR_SIZE_LABELS)))

    for _, name, color, label in FAMILIES:
        if name not in records_by_key:
            continue
        ys, ns = [], []
        for si in range(len(ANCHOR_SIZE_LABELS)):
            keys_in_bin = bins_by_size[si]
            records = [records_by_key[name][k] for k in keys_in_bin if k in records_by_key[name]]
            values = [coverage_per_record(r) for r in records]
            values = [v for v in values if v is not None and not np.isnan(v)]
            ns.append(len(values))
            ys.append(np.mean(values) if values else np.nan)
        print(f"  {label:<35} " +
                "  ".join(f"{size_label}: {y:.3f} (n={n})"
                            for size_label, y, n in zip(ANCHOR_SIZE_LABELS, ys, ns)))
        axes_one.plot(xs, ys, marker="o", color=color, linewidth=2,
                          markersize=10, markeredgecolor="black", markeredgewidth=0.8,
                          label=label)

    axes_one.set_xticks(xs)
    axes_one.set_xticklabels([str(si + 1) for si in xs], fontsize=11)
    axes_one.set_xlabel("anchor-size quartile (1 = smallest, 4 = largest)")
    axes_one.set_ylabel(f"Coverage (IoU > {METRIC_THRESHOLD})")
    axes_one.set_title(f"samara_dispatch_size_20 vs sam_baseline_20 by anchor-size quartile  "
                          f"(first_occlusion < {MAX_FIRST_OCCLUSION}, n={len(early_keys)})",
                          fontsize=11)
    axes_one.grid(True, alpha=0.3)
    axes_one.legend(loc="best", fontsize=10)
    figure.tight_layout()
    output_path = OUTPUT_DIRECTORY / "visualize_2.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()

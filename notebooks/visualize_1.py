"""Coverage at IoU > 0.5 for experiments 3 / 4 / 5, split into 6 bins:
  rows: VISIBLE bbox area at the CHOSEN ANCHOR (frame_indices[0]) at 1024-resize,
        TERCILE-SPLIT — SMALL / MEDIUM / LARGE (equal n per row).
  cols: first-occlusion frame index. MEDIAN-SPLIT — EARLY = first_occlusion < median;
        LATE = first_occlusion >= median  (so cols have roughly equal n).
Each subplot shows the families plotted against the threshold suffix `{0.2, 0.3, 0.4}`
on the joint-pool intersection of that bin."""

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
AMODAL_DIRECTORY = Path("data/person_path/amodal")
DATASET_DIRECTORY = Path("data/datasets/hiera_anchor/clean")
METRIC_THRESHOLD = 0.5
SAM_RESIZE = 1024

FAMILIES = [
    (3, "samara_fixed_hiera_{}",   "tab:green", "samara_fixed_hiera"),
    (4, "samara_fixed_memory_{}",  "tab:cyan",  "samara_fixed_memory"),
    (5, "sam_baseline_{}",         "tab:red",   "sam_baseline"),
]
THRESHOLDS_PCT = [20, 30, 40]
ANCHOR_SIZE_LABELS = ["small", "medium", "large"]
FIRST_OCCLUSION_LABELS = ["early", "late"]


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
    image_width = int(data["metadata"]["resolution"]["width"])
    image_height = int(data["metadata"]["resolution"]["height"])
    scale = SAM_RESIZE / max(image_width, image_height)
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
    return max(0.0, w) * max(0.0, h) * scale * scale


def visible_area_at_chosen_anchor(video_name, person_id):
    frame_indices = _trajectory_frame_indices(video_name, person_id)
    if frame_indices is None or len(frame_indices) == 0:
        return None
    anchor_video_frame = int(frame_indices[0])
    return _bbox_area_at_video_frame(VISIBLE_DIRECTORY, video_name, person_id, anchor_video_frame)


def main():
    records_by_tracker = {}
    for exp, template, _, _ in FAMILIES:
        for pct in THRESHOLDS_PCT:
            name = template.format(pct)
            records = load(exp, name)
            if records is not None:
                records_by_tracker[name] = records
            else:
                print(f"MISSING data/experiment_{exp}/{name}.pkl")
    if not records_by_tracker:
        print("no pickles found")
        return

    shared = set.intersection(*[
        {(r["video_name"], r["person_id"]) for r in records}
        for records in records_by_tracker.values()])
    print(f"shared intersection: n={len(shared)}")

    reference_by_key = {(r["video_name"], r["person_id"]): r
                          for r in next(iter(records_by_tracker.values()))
                          if (r["video_name"], r["person_id"]) in shared}

    anchor_areas_by_key = {k: visible_area_at_chosen_anchor(k[0], k[1]) for k in shared}
    first_occlusions = np.array([first_occlusion_frame(reference_by_key[k]) for k in shared
                                       if first_occlusion_frame(reference_by_key[k]) < 10**9])
    first_edge = float(np.median(first_occlusions)) if len(first_occlusions) else 0
    valid_areas = np.array([a for a in anchor_areas_by_key.values() if a is not None])
    size_edges = (np.quantile(valid_areas, [1/3, 2/3]) if len(valid_areas)
                       else np.array([0.0, 0.0]))
    print(f"first-occlusion median split: EARLY = first occlusion < {first_edge:.0f} frames")
    print(f"anchor-area tercile edges:    SMALL < {size_edges[0]:.0f} px²  "
            f"<= MEDIUM < {size_edges[1]:.0f} px²  <= LARGE  "
            f"(visible bbox at chosen anchor, 1024-resize)")

    def first_bin(record):
        first = first_occlusion_frame(record)
        if first >= 10**9: return None
        return 0 if first < first_edge else 1

    def size_bin(key):
        area = anchor_areas_by_key.get(key)
        if area is None: return None
        if area < size_edges[0]: return 0
        if area < size_edges[1]: return 1
        return 2

    bins = {(si, fi): set() for si in range(3) for fi in range(2)}
    for key in shared:
        record = reference_by_key[key]
        si = size_bin(key)
        fi = first_bin(record)
        if si is None or fi is None:
            continue
        bins[(si, fi)].add(key)

    records_by_key = {name: {(r["video_name"], r["person_id"]): r for r in records_by_tracker[name]}
                       for name in records_by_tracker}

    figure, axes_grid = plt.subplots(3, 2, figsize=(11, 13), sharey=False)
    xs = [t / 100 for t in THRESHOLDS_PCT]

    for si, size_label in enumerate(ANCHOR_SIZE_LABELS):
        for fi, first_label in enumerate(FIRST_OCCLUSION_LABELS):
            axes_one = axes_grid[si, fi]
            keys_in_bin = bins[(si, fi)]
            print(f"\n=== {size_label} anchor · {first_label} first occlusion  (n={len(keys_in_bin)}) ===")
            all_ys = []
            for family_index, (_, template, color, family_label) in enumerate(FAMILIES):
                ys, ns = [], []
                for pct in THRESHOLDS_PCT:
                    name = template.format(pct)
                    records = [records_by_key[name][k] for k in keys_in_bin
                                  if k in records_by_key.get(name, {})]
                    values = [coverage_per_record(r) for r in records]
                    values = [v for v in values if v is not None and not np.isnan(v)]
                    ns.append(len(values))
                    ys.append(np.mean(values) if values else np.nan)
                print(f"  {family_label:<35} " +
                        " ".join(f"@{p/100:.1f}: {y:.3f} (n={n})"
                                    for p, y, n in zip(THRESHOLDS_PCT, ys, ns)))
                all_ys.extend(y for y in ys if not np.isnan(y))
                dodge = (family_index - 1) * 0.008
                xs_dodged = [x + dodge for x in xs]
                axes_one.plot(xs_dodged, ys, marker="o", color=color, linewidth=2,
                                 markersize=10, markeredgecolor="black",
                                 markeredgewidth=0.8, label=family_label)
            axes_one.set_xticks(xs)
            axes_one.set_title(f"{size_label} anchor · {first_label} first occlusion\nn={len(keys_in_bin)}",
                                  fontsize=10)
            axes_one.grid(True, alpha=0.3)
            if all_ys:
                lo, hi = min(all_ys), max(all_ys)
                span = max(0.02, hi - lo)
                axes_one.set_ylim(lo - 0.4 * span, hi + 0.4 * span)
            if si == len(ANCHOR_SIZE_LABELS) - 1:
                axes_one.set_xlabel("threshold")
            if fi == 0:
                axes_one.set_ylabel(f"Coverage (IoU > {METRIC_THRESHOLD})")

    handles, labels = axes_grid[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(FAMILIES),
                     fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))
    figure.suptitle(f"Coverage by anchor-size × first-occlusion bin "
                       f"(visible bbox at chosen anchor, 1024-resize; joint pool n={len(shared)})",
                       fontsize=11)
    figure.tight_layout(rect=[0, 0.05, 1, 0.97])
    output_path = OUTPUT_DIRECTORY / "visualize_1.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()

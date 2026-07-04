"""visualize_4: coverage (IoU > 0.5) of the experiment-4 memory-variant trackers.

Auto-discovers every `data/experiment_4/*.pkl` tracker, computes coverage on the shared
trajectory intersection, and reports it OVERALL and split by first-occlusion timing
(EARLY vs LATE, median split) — the axis that matters for an occlusion-recovery method.
`sam_baseline_20` (exp 5) and `memory_oracle_20` (exp 2) are added as references when present.

Output: prints a per-tracker table and saves data/visualize_4.png (grouped bar chart).
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.metrics import coverage, coverage_auc


INPUT_DIRECTORY = Path("data")
OUTPUT_DIRECTORY = Path("data")
METRIC_THRESHOLD = 0.5

# (experiment_num, tracker_name) reference trackers to plot alongside experiment 4.
REFERENCES = [(5, "sam_baseline_20"), (2, "memory_oracle_20")]


def cov05(iou_scores, occlusions):
    return coverage(iou_scores, occlusions, METRIC_THRESHOLD)


# metrics reported side by side (each takes iou_scores, occlusions)
METRICS = [(f"Coverage @ IoU > {METRIC_THRESHOLD}", cov05),
           ("Coverage-AUC (mean over IoU thresholds 0..1)", coverage_auc)]


def first_occlusion_frame(record):
    occluded = np.where(np.asarray(record["occlusions"]) > 0.5)[0]
    return int(occluded[0]) if len(occluded) else 10**9


def load(exp, name):
    path = INPUT_DIRECTORY / f"experiment_{exp}" / f"{name}.pkl"
    if not path.exists():
        return None
    return pickle.load(open(path, "rb"))["records"]


def mean_metric(records_for_keys, metric_fn):
    values = [metric_fn(r["iou_scores"], r["occlusions"]) for r in records_for_keys]
    values = [v for v in values if v is not None and not np.isnan(v)]
    return (np.mean(values) if values else np.nan), len(values)


def main():
    # ---- collect experiment-4 trackers (auto-discovered) + references ----
    trackers = {}
    exp4_dir = INPUT_DIRECTORY / "experiment_4"
    if exp4_dir.exists():
        for path in sorted(exp4_dir.glob("*.pkl")):
            trackers[path.stem] = pickle.load(open(path, "rb"))["records"]
    experiment4_names = list(trackers)
    if not experiment4_names:
        print(f"no experiment-4 pickles found under {exp4_dir} — run run_experiment.py experiment=4 first")
        return

    reference_names = []
    for exp, name in REFERENCES:
        records = load(exp, name)
        if records is not None:
            trackers[name] = records
            reference_names.append(name)
        else:
            print(f"(reference missing) data/experiment_{exp}/{name}.pkl")

    # ---- shared trajectory intersection ----
    shared = set.intersection(*[
        {(r["video_name"], r["person_id"]) for r in records} for records in trackers.values()])
    print(f"trackers: {len(trackers)}  |  shared trajectory intersection: n={len(shared)}")
    if not shared:
        print("no shared trajectories across the loaded trackers")
        return

    records_by_key = {name: {(r["video_name"], r["person_id"]): r for r in records}
                      for name, records in trackers.items()}

    # ---- first-occlusion median split (GT occlusions are shared across trackers) ----
    reference = next(iter(records_by_key.values()))
    first_occlusions = [first_occlusion_frame(reference[k]) for k in shared
                        if first_occlusion_frame(reference[k]) < 10**9]
    edge = float(np.median(first_occlusions)) if first_occlusions else 0.0
    early = {k for k in shared if first_occlusion_frame(reference[k]) < edge}
    late = {k for k in shared if edge <= first_occlusion_frame(reference[k]) < 10**9}
    print(f"first-occlusion median split: EARLY = first_occlusion < {edge:.0f} frames "
          f"(early n={len(early)}, late n={len(late)})\n")

    # ---- per-tracker tables: coverage @ IoU>0.5 AND coverage-AUC (overall / early / late) ----
    ordered = experiment4_names + reference_names
    tables = {}
    for title, metric_fn in METRICS:
        print(f"\n=== {title} ===")
        print(f"{'tracker':40} {'overall':>16} {'early':>16} {'late':>16}")
        print("-" * 92)
        tbl = {}
        for name in ordered:
            by_key = records_by_key[name]
            overall = mean_metric([by_key[k] for k in shared if k in by_key], metric_fn)
            early_c = mean_metric([by_key[k] for k in early if k in by_key], metric_fn)
            late_c = mean_metric([by_key[k] for k in late if k in by_key], metric_fn)
            tbl[name] = (overall, early_c, late_c)
            print(f"{name:40} {overall[0]:8.3f} (n={overall[1]:4}) "
                  f"{early_c[0]:8.3f} (n={early_c[1]:4}) {late_c[0]:8.3f} (n={late_c[1]:4})")
        tables[title] = tbl
    table = tables[f"Coverage @ IoU > {METRIC_THRESHOLD}"]   # bar chart below uses the IoU>0.5 table

    # ---- grouped bar chart (sorted by overall coverage) ----
    ordered.sort(key=lambda n: np.nan_to_num(table[n][0][0]), reverse=True)
    overall_vals = [table[n][0][0] for n in ordered]
    early_vals = [table[n][1][0] for n in ordered]
    late_vals = [table[n][2][0] for n in ordered]

    x = np.arange(len(ordered))
    width = 0.27
    figure, axes = plt.subplots(figsize=(max(9, 1.1 * len(ordered)), 6))
    axes.bar(x - width, overall_vals, width, label="overall", color="tab:blue")
    axes.bar(x, early_vals, width, label="early first occlusion", color="tab:orange")
    axes.bar(x + width, late_vals, width, label="late first occlusion", color="tab:green")

    # mark references with a hatched edge so they're distinguishable from exp-4 trackers
    for i, name in enumerate(ordered):
        if name in reference_names:
            for dx in (-width, 0, width):
                axes.bar(x[i] + dx, [overall_vals, early_vals, late_vals][[-width, 0, width].index(dx)][i],
                         width, fill=False, edgecolor="black", linewidth=1.2, hatch="//")

    axes.set_xticks(x)
    axes.set_xticklabels(ordered, rotation=40, ha="right", fontsize=9)
    axes.set_ylabel(f"Coverage (IoU > {METRIC_THRESHOLD})")
    axes.set_title(f"Experiment 4 (memory variant) coverage by first-occlusion timing  "
                   f"(shared n={len(shared)}; hatched = reference)", fontsize=11)
    axes.grid(True, axis="y", alpha=0.3)
    axes.legend(loc="best", fontsize=10)
    figure.tight_layout()

    output_path = OUTPUT_DIRECTORY / "visualize_4.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()

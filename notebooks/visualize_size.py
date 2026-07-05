"""visualize_size: post-occlusion coverage (IoU > 0.5) vs backbone CAPACITY, one line per variant.

Auto-discovers every data/experiment_*/*.pkl (named <variant>_<size>), computes coverage on the
shared trajectory intersection (so all variants are compared on the same trajectories), and plots
one line per variant with the x-axis = SAM backbone parameter count (tiny -> large).

Output: prints a variant x size table and saves data/visualize_size.png.
Run: python notebooks/visualize_size.py
"""

import sys
import glob
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.metrics import coverage, coverage_auc

METRIC_THRESHOLD = 0.5
SIZES = ["tiny", "small", "base_plus", "large"]
CAPACITY = {"tiny": 38.9, "small": 46.1, "base_plus": 80.8, "large": 224.4}   # M params (SAM2 model)


def parse(stem):
    """'<variant>_<size>' -> (variant, size). base_plus checked first (it contains '_')."""
    for size in ("base_plus", "tiny", "small", "large"):
        if stem.endswith("_" + size):
            return stem[: -len(size) - 1], size
    return None, None


def cov05(record):
    return coverage(record["iou_scores"], record["occlusions"], METRIC_THRESHOLD)


def covauc(record):
    return coverage_auc(record["iou_scores"], record["occlusions"])


METRICS = [(f"Coverage (IoU > {METRIC_THRESHOLD})", cov05),
           ("Coverage-AUC (mean over IoU thresholds)", covauc)]


def main():
    trackers = {}   # (variant, size) -> {(video, person): record}
    for path in sorted(glob.glob("data/experiment_*/*.pkl")):
        variant, size = parse(Path(path).stem)
        if variant is None:
            continue
        records = pickle.load(open(path, "rb"))["records"]
        trackers[(variant, size)] = {(r["video_name"], r["person_id"]): r for r in records}

    if not trackers:
        print("no data/experiment_*/*.pkl found -- run run_experiment.py first")
        return

    variants = sorted({v for v, _ in trackers})

    # Intersection is PER-SIZE: at each backbone size, ALL trackers are scored on the same trajectories.
    size_shared = {}
    for s in SIZES:
        ds = [set(trackers[(v, s)]) for v in variants if (v, s) in trackers]
        size_shared[s] = set.intersection(*ds) if ds else set()
    print(f"loaded {len(trackers)} (variant,size) trackers")
    print("per-size shared trajectories (across all trackers at that size):")
    for s in SIZES:
        print(f"  {s:10} n={len(size_shared[s])}")

    # Global first-occlusion median split (GT occlusions are identical across trackers).
    def first_occ(key):
        for d in trackers.values():
            if key in d:
                w = np.where(np.asarray(d[key]["occlusions"]) > 0.5)[0]
                return int(w[0]) if len(w) else None
        return None
    allkeys = set().union(*[set(d) for d in trackers.values()])
    focc = {k: f for k, f in ((k, first_occ(k)) for k in allkeys) if f is not None}
    edge = float(np.median(list(focc.values()))) if focc else 0.0
    early = {k for k, f in focc.items() if f < edge}
    late = {k for k, f in focc.items() if f >= edge}
    print(f"\nfirst-occlusion median split at {edge:.0f} frames (global threshold, applied within each variant)")

    def compute(keyset):
        results = {}
        for v in variants:
            results[v] = {}
            for s in SIZES:
                d = trackers.get((v, s))
                if d is None:
                    continue
                keys = size_shared[s] & keyset          # same trajectory set for every tracker at size s
                vals = [c for c in (covauc(d[k]) for k in keys) if c is not None and not np.isnan(c)]
                if vals:
                    results[v][s] = (float(np.mean(vals)), len(vals))
        return results

    panels = [(f"Coverage-AUC | EARLY occlusion (< {edge:.0f} fr)", early),
              (f"Coverage-AUC | LATE occlusion (>= {edge:.0f} fr)", late)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for ax, (title, keyset) in zip(axes, panels):
        results = compute(keyset)

        print(f"\n=== {title} ===")
        print(f"{'variant':16} " + " ".join(f"{s:>13}" for s in SIZES))
        print("-" * 74)
        for v in variants:
            cells = [f"{results[v][s][0]:.3f} (n{results[v][s][1]})".rjust(13) if s in results[v] else " " * 13
                     for s in SIZES]
            print(f"{v:16} " + " ".join(cells))

        for v in variants:
            pts = [(CAPACITY[s], results[v][s][0]) for s in SIZES if s in results[v]]
            if not pts:
                continue
            xs, ys = zip(*pts)
            style = {"marker": "o", "linewidth": 2.5, "markersize": 7}
            if v == "samara_fixed":                   # the method: bold + squares, on top
                style.update(linewidth=3.5, marker="s", markersize=9, zorder=5, color="black")
            if v == "memory_oracle":                  # oracle upper bound: dashed
                style.update(linestyle="--", alpha=0.75)
            ax.plot(xs, ys, label=v, **style)

        ax.set_xscale("log")
        ax.set_xticks([CAPACITY[s] for s in SIZES])
        ax.set_xticklabels([f"{s}\n{CAPACITY[s]:.0f}M" for s in SIZES])
        ax.set_xlabel("SAM backbone capacity (parameters)")
        ax.set_ylabel("Coverage-AUC (post-occlusion)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Coverage-AUC vs backbone capacity, by first-occlusion timing  (same trajectories per size)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path("data/visualize_size.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

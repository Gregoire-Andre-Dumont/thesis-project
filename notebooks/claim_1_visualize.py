"""claim_1 table: coverage@0.5 for the baselines AND the oracles, split by occlusion length.

The oracles read ground truth, so their margin over the baselines is the HEADROOM a deployable commit gate
could recover. That is the point of claim_1: the baselines say where the field is, the oracles say how much
is left on the table.

Coverage is the primary metric here, and it is the right one for an oracle comparison specifically: it asks
only whether each VISIBLE frame's mask reached IoU >= 0.5, so no confidence signal is involved. An oracle's
"confidence" would be the true IoU it already used to select the mask, which would make any reporting-based
comparison circular.

Two spans are printed. The FULL SPAN is everything after the first occlusion. The MEMORY WINDOW is every
occluded frame plus only the visible frames AFTER the last occlusion.

Every arm here runs `tm/sam_large.pt`. DAM4SAM is excluded because it requires the 2.1 checkpoint: on the
shared clips it beat the MASK ORACLE, which is impossible on equal footing and is the signature of a better
backbone rather than a better memory.

    python notebooks/claim_1_visualize.py [results.pkl] [out_dir]
    N_BINS=4 python notebooks/claim_1_visualize.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_BINS = int(os.environ.get("N_BINS", "3"))
LOCALISATION = 0.5
THRESHOLD = float(os.environ.get("THRESHOLD", "0.2"))   # which swept oracle to report

BASELINES = [
    ("SAM 2",    "data/baselines_votlt/sam2/results.pkl"),
    ("SAMITE",   "data/baselines_votlt/samite/results.pkl"),
    ("SAMURAI",  "data/baselines/samurai/results.pkl"),
    ("SAM2Long", "data/baselines_votlt/sam2long/results.pkl"),
]
ORACLES = [("memory oracle", "memory"), ("mask oracle", "mask")]


def keyed(results_path):
    clips = pickle.load(open(results_path, "rb"))["clips"]
    return {(record["video"], record["person"]): record for record in clips}


def at_threshold(value):
    """The swept oracles store {commit threshold: array}; everything else stores the array itself."""

    if isinstance(value, dict):
        if THRESHOLD in value:
            return value[THRESHOLD]
        return value[min(value, key=lambda key: abs(float(key) - THRESHOLD))]
    return value


def window_coverage(overlap, occluded, has_box, window):
    """Fraction of VISIBLE frames reaching IoU >= 0.5, over the full span or the memory window.

    Returns NaN when the window leaves no visible frame -- the target never reappeared after its last
    occlusion -- so those clips are dropped rather than scored as zero."""

    overlap = np.nan_to_num(np.asarray(overlap, dtype=float), nan=0.0)
    occluded = np.asarray(occluded, dtype=bool)
    kept = np.asarray(has_box, dtype=bool).copy()
    if window and occluded.any():
        kept[: int(np.where(occluded)[0][-1]) + 1] = False

    visible = kept & ~occluded
    length = min(len(overlap), len(visible))
    overlap, visible = overlap[:length], visible[:length]
    return float((overlap[visible] >= LOCALISATION).mean()) if visible.any() else np.nan


def occlusion_bins(counts):
    """`N_BINS` QUANTILE bins on the number of occluded frames, plus a pooled column."""

    edges = np.quantile(counts, np.linspace(0, 1, N_BINS + 1))
    bins, labels = [], []
    for index in range(N_BINS):
        low, high = edges[index], (np.inf if index == N_BINS - 1 else edges[index + 1])
        bins.append((low, high))
        labels.append(f"{int(low)}-{int(counts.max()) if np.isinf(high) else int(high)}")
    return bins + [(0.0, np.inf)], labels + ["all"]


def table(rows, counts, bins, labels, title, handle=None):
    header = f"{'arm':16s}" + "".join(f"{label:>12s}" for label in labels)
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for label, values in rows.items():
        cells = [np.nanmean(values[(counts >= low) & (counts < high)]) for low, high in bins]
        print(f"{label:16s}" + "".join(f"{value:12.4f}" for value in cells))
        if handle:
            handle.write(label + "," + ",".join(f"{value:.4f}" for value in cells) + "\n")

    # Report what was AVERAGED, not bin membership: the window drops clips with no visible frame, so
    # printing bin membership makes the bins fail to recombine into the pooled column. The window depends
    # only on `occluded`/`has_box`, which every arm shares, so these counts hold for every row.
    finite = np.isfinite(next(iter(rows.values())))
    averaged = [int(((counts >= low) & (counts < high) & finite).sum()) for low, high in bins]
    dropped = [int(((counts >= low) & (counts < high) & ~finite).sum()) for low, high in bins]
    print("-" * len(header))
    print(f"{'n averaged':16s}" + "".join(f"{count:12d}" for count in averaged))
    print(f"{'n no-visible':16s}" + "".join(f"{count:12d}" for count in dropped))
    if handle:
        handle.write("n averaged," + ",".join(str(count) for count in averaged) + "\n")
        handle.write("n no-visible," + ",".join(str(count) for count in dropped) + "\n")


def main():
    results_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/claim_1/results.pkl")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/sam-2/claim_1")

    oracle_clips = keyed(results_path)
    baselines = {label: keyed(path) for label, path in BASELINES if Path(path).exists()}

    clip_keys = sorted(set(oracle_clips).intersection(*(set(records) for records in baselines.values())))
    clip_keys = [key for key in clip_keys
                 if all(at_threshold(oracle_clips[key].get(oracle_key)) is not None
                        for _, oracle_key in ORACLES)]
    if not clip_keys:
        raise SystemExit("no clip is scored by every baseline and both oracles")

    counts = np.array([oracle_clips[key]["occ_count"] for key in clip_keys], dtype=float)
    bins, labels = occlusion_bins(counts)

    out_dir.mkdir(parents=True, exist_ok=True)
    for window, span in ((False, "full span"), (True, "memory window")):
        rows = {}
        for label, records in baselines.items():
            rows[label] = np.array([window_coverage(records[key]["samara"], records[key]["occluded"],
                                                    records[key]["has_box"], window) for key in clip_keys])
        for label, oracle_key in ORACLES:
            rows[label] = np.array([window_coverage(at_threshold(oracle_clips[key][oracle_key]),
                                                    oracle_clips[key]["occluded"],
                                                    oracle_clips[key]["has_box"], window)
                                    for key in clip_keys])

        name = "table_coverage_by_occlusion" + ("_window" if window else "") + ".csv"
        with open(out_dir / name, "w") as handle:
            handle.write("arm," + ",".join(labels) + "\n")
            table(rows, counts, bins, labels,
                  f"coverage@0.5 · {span}  ·  {len(clip_keys)} clips  ·  {N_BINS} quantile bins", handle)
        print(f"  -> {out_dir / name}")


if __name__ == "__main__":
    main()

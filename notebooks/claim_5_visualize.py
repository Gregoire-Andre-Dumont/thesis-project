"""claim_5 table: coverage@0.5 for SAMARA and the published baselines, split by occlusion length.

Coverage is the primary metric: for each VISIBLE frame, did the emitted mask reach IoU >= 0.5. It needs no
confidence signal, so every arm is compared on its masks alone and nothing depends on which score an arm
happens to expose or how well that score ranks frames.

Two spans are printed. The FULL SPAN is everything after the first occlusion. The MEMORY WINDOW is every
occluded frame plus only the visible frames AFTER the last occlusion -- the regime the commit gate exists
for, where the bank has had time to drift and the target has to be re-found.

Clips are grouped into N_BINS quantile bins on `occ_count`, so each bin holds a comparable number of clips
and the bin edges follow the data rather than being imposed.

DAM4SAM is absent: it runs `tm/sam2.1_hiera_large.pt` where every other arm here -- and the oracles in
claim_1 -- runs `tm/sam_large.pt`, so its row would confound a better BACKBONE with a better memory.

    python notebooks/claim_5_visualize.py [arms_dir] [out_dir]
    N_BINS=4 python notebooks/claim_5_visualize.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_BINS = int(os.environ.get("N_BINS", "3"))
LOCALISATION = 0.5

BASELINES = [
    ("SAM 2",    "data/baselines_votlt/sam2/results.pkl"),
    ("SAMITE",   "data/baselines_votlt/samite/results.pkl"),
    ("SAMURAI",  "data/baselines/samurai/results.pkl"),
    ("SAM2Long", "data/baselines_votlt/sam2long/results.pkl"),
]


def keyed(results_path):
    clips = pickle.load(open(results_path, "rb"))["clips"]
    return {(record["video"], record["person"]): record for record in clips}


def window_coverage(record, window):
    """Fraction of VISIBLE frames reaching IoU >= 0.5, over the full span or the memory window.

    The window clears `has_box` up to and including the last occluded frame, leaving the occluded frames
    plus the visible tail after the target last disappeared. A clip the window leaves with no visible frame
    returns NaN and drops out of the nan-aware means."""

    overlap = np.nan_to_num(np.asarray(record["samara"], dtype=float), nan=0.0)
    occluded = np.asarray(record["occluded"], dtype=bool)
    kept = np.asarray(record["has_box"], dtype=bool).copy()
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
    # Report what was AVERAGED, not bin membership. The memory window leaves some clips with no visible
    # frame at all -- the target never reappears after its last occlusion -- and `np.nanmean` drops those.
    # Printing bin membership instead makes the bins fail to recombine into the pooled column, which looks
    # like an arithmetic error in the table. The window depends only on `occluded`/`has_box`, which every
    # arm shares, so these counts are the same for every row.
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
    arms_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/claim_5")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/sam-2/claim_5")

    loaded = {}
    for label, path in BASELINES:
        if Path(path).exists():
            loaded[label] = keyed(path)
    for path in sorted(arms_dir.glob("*/results.pkl")):
        loaded[f"SAMARA {path.parent.name}"] = keyed(path)
    if not loaded:
        raise SystemExit(f"no arms found under {arms_dir} and no baselines present")

    clip_keys = sorted(set.intersection(*(set(records) for records in loaded.values())))
    reference = next(iter(loaded.values()))
    counts = np.array([reference[key]["occ_count"] for key in clip_keys], dtype=float)
    bins, labels = occlusion_bins(counts)

    out_dir.mkdir(parents=True, exist_ok=True)
    for window, span in ((False, "full span"), (True, "memory window")):
        rows = {label: np.array([window_coverage(records[key], window) for key in clip_keys])
                for label, records in loaded.items()}
        name = "table_coverage_by_occlusion" + ("_window" if window else "") + ".csv"
        with open(out_dir / name, "w") as handle:
            handle.write("arm," + ",".join(labels) + "\n")
            table(rows, counts, bins, labels,
                  f"coverage@0.5 · {span}  ·  {len(clip_keys)} clips  ·  {N_BINS} quantile bins", handle)
        print(f"  -> {out_dir / name}")


if __name__ == "__main__":
    main()

"""Regenerate every claim's table as MEAN IoU over the MEMORY WINDOW -- the one configuration reported.

Mean IoU, not coverage@0.5: a threshold throws away how badly a frame was missed, which is why the window
figure moves so much when a handful of clips cross 0.5. The memory window is the occluded frames plus the
visible tail AFTER the target last disappeared -- the regime the commit gate exists for, where the bank has
had time to drift and the target has to be re-found.

Every table is restricted to ONE clip set: the 600-trajectory draw, intersected across the claim_5 arms.
claim_1 is rolled on a wider set, so it is filtered down rather than reported on its own denominator; a
table whose rows come from different clips cannot be read across.

    python notebooks/mean_iou_window.py            # all four claims
    python notebooks/mean_iou_window.py claim_2    # just one

Writes data/sam-2/claim_N/mean_iou_window.csv, and for claim_6 the pool-size figure as well.
"""

import csv
import glob
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks" / "paper"))

DATA = Path("data/sam-2")
CLAIM_5 = DATA / "claim_5"                       # the IoU-token gate run: results_00/05/10/15.pkl
OCCLUSION_EDGES = [(2, 8), (8, 18), (18, 50), (0, 10 ** 6)]
OCCLUSION_LABELS = ["2-8", "8-18", "18-49", "All"]

# claim_1 stores its oracles as a sweep over commit thresholds. 0.2 is the threshold claim_2 deploys, so
# picking it here keeps the two tables describing the same controller rather than two different ones.
ORACLE_THRESHOLD = 0.2

BASELINES = [
    ("SAM 2", DATA / "baselines_votlt/sam2/results.pkl"),
    ("SAMITE", DATA / "baselines_votlt/samite/results.pkl"),
    ("SAMURAI", DATA / "baselines/samurai/results.pkl"),
    ("SAM2Long", DATA / "baselines_votlt/sam2long/results.pkl"),
]
ORACLE_ARMS = [("sam", "SAM baseline"), ("memory", "memory oracle"), ("mask", "mask oracle")]


def clips_of(path):
    """The clip records in a results pickle, keyed or raw."""

    loaded = pickle.load(open(path, "rb"))
    return loaded["clips"] if isinstance(loaded, dict) and "clips" in loaded else loaded


def key_of(record):
    return str(record["video"]), int(record["person"])


def window_mean_iou(record, series):
    """Mean IoU over the memory window: visible, annotated frames AFTER the last occluded one.

    A clip the window leaves with no visible frame returns NaN and drops out of the nan-aware means."""

    overlap = np.nan_to_num(np.asarray(series, dtype=float), nan=0.0)
    occluded = np.asarray(record["occluded"], dtype=bool)
    kept = np.asarray(record["has_box"], dtype=bool).copy()
    if occluded.any():
        kept[: int(np.where(occluded)[0][-1]) + 1] = False

    visible = kept & ~occluded
    length = min(len(overlap), len(visible))
    overlap, visible = overlap[:length], visible[:length]
    return float(overlap[visible].mean()) if visible.any() else np.nan


def reference_draw():
    """The clip set every table is reported on: the 600-draw, as the claim_5 arms scored it."""

    paths = sorted(CLAIM_5.glob("results_*.pkl"))
    if not paths:
        raise SystemExit(f"no claim_5 arms under {CLAIM_5}; nothing defines the draw")
    return set.intersection(*[{key_of(record) for record in clips_of(path)} for path in paths])


def selection_signal(record):
    """The reference gate's mean p(commit) over the WHOLE sequence -- no onset, no truncation.

    `reference_probability_full` is the BLOCK's p0.10 gate scoring this arm's masks, so the four arms are
    ranked by one judge that never saw the clip it is scoring. Nothing here reads an annotation."""

    probability = np.asarray(record["reference_probability_full"], dtype=float)
    return float(np.nanmean(probability)) if np.isfinite(probability).any() else np.nan


def binned(values, occlusion_counts, total):
    cells = []
    for low, high in OCCLUSION_EDGES:
        inside = (occlusion_counts >= low) & (occlusion_counts < high) if high < 10 ** 5 else slice(None)
        cells.append(np.nanmean(values[inside]))
    return [f"{cell:.4f}" for cell in cells]


def write_csv(path, header, rows, counts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
        writer.writerow(["n_clips"] + counts)
    print(f"wrote {path}\n{path.read_text()}")


def bin_counts(occlusion_counts, total):
    return [str(int(((occlusion_counts >= low) & (occlusion_counts < high)).sum())) if high < 10 ** 5
            else str(total) for low, high in OCCLUSION_EDGES]


# --------------------------------------------------------------------------------------- the four claims

def claim_1(draw):
    """SAM against the two oracles, by occlusion length."""

    records = [r for r in clips_of(DATA / "claim_1/results.pkl") if key_of(r) in draw]
    occlusion_counts = np.array([int(r["occ_count"]) for r in records])
    rows = []
    for arm, label in ORACLE_ARMS:
        series = [r[arm] if arm == "sam" else r[arm][ORACLE_THRESHOLD] for r in records]
        values = np.array([window_mean_iou(r, s) for r, s in zip(records, series)], dtype=float)
        rows.append([label] + binned(values, occlusion_counts, len(records)))
    write_csv(DATA / "claim_1/mean_iou_window.csv", ["method"] + OCCLUSION_LABELS, rows,
              bin_counts(occlusion_counts, len(records)))


def claim_2(draw):
    """The same three arms under memory corruption, by corruption probability."""

    loaded = pickle.load(open(DATA / "claim_2/results.pkl", "rb"))
    records = [r for r in loaded["clips"] if key_of(r) in draw]
    probabilities = list(loaded["corruption_ps"])
    rows = []
    for arm, label in ORACLE_ARMS:
        row = [label]
        for probability in probabilities:
            values = np.array([window_mean_iou(r, r[(arm, probability)]) for r in records
                               if (arm, probability) in r], dtype=float)
            row.append(f"{np.nanmean(values):.4f}" if len(values) else "")
        rows.append(row)
    write_csv(DATA / "claim_2/mean_iou_window.csv",
              ["method"] + [f"p={p:g}" for p in probabilities], rows,
              [str(len(records))] * len(probabilities))


def load_arms(draw):
    """Each arm's per-clip window IoU and selection signal, aligned on the draw."""

    paths = sorted(CLAIM_5.glob("results_*.pkl"))
    names = [f"p0.{path.stem.split('_')[-1]}" for path in paths]
    by_arm = [{key_of(r): r for r in clips_of(path)} for path in paths]
    keys = sorted(draw.intersection(*[set(arm) for arm in by_arm]))
    metric = np.array([[window_mean_iou(arm[k], arm[k]["samara"]) for k in keys] for arm in by_arm])
    signal = np.array([[selection_signal(arm[k]) for k in keys] for arm in by_arm])
    return names, keys, by_arm, metric, signal


def select(metric, signal):
    """Per clip, the chosen arm's score. Ties and all-NaN signals fall back to the pool's strongest arm."""

    fallback = int(np.nanargmax(np.nanmean(metric, axis=1)))
    return np.array([metric[fallback if not np.isfinite(signal[:, j]).any()
                            else int(np.nanargmax(signal[:, j])), j]
                     for j in range(metric.shape[1])], dtype=float)


def claim_5(draw):
    """Every arm and the label-free selection against the published baselines, by occlusion length."""

    names, keys, by_arm, metric, signal = load_arms(draw)
    occlusion_counts = np.array([int(by_arm[0][k]["occ_count"]) for k in keys])
    rows = []
    for label, path in BASELINES:
        baseline = {key_of(r): r for r in clips_of(path)}
        values = np.array([window_mean_iou(baseline[k], baseline[k]["samara"]) for k in keys], dtype=float)
        rows.append([label] + binned(values, occlusion_counts, len(keys)))
    for index, name in enumerate(names):
        rows.append([f"SAMARA {name}"] + binned(metric[index], occlusion_counts, len(keys)))
    rows.append(["SAMARA selection"] + binned(select(metric, signal), occlusion_counts, len(keys)))
    rows.append(["per-clip oracle"] + binned(np.nanmax(metric, axis=0), occlusion_counts, len(keys)))
    write_csv(CLAIM_5 / "mean_iou_window.csv", ["method"] + OCCLUSION_LABELS, rows,
              bin_counts(occlusion_counts, len(keys)))


def claim_6(draw):
    """Selection against one representative gate, plus what it adds as the pool of gates grows."""

    names, keys, by_arm, metric, signal = load_arms(draw)
    occlusion_counts = np.array([int(by_arm[0][k]["occ_count"]) for k in keys])

    rows = []
    for label, path in BASELINES:
        baseline = {key_of(r): r for r in clips_of(path)}
        values = np.array([window_mean_iou(baseline[k], baseline[k]["samara"]) for k in keys], dtype=float)
        rows.append([label] + binned(values, occlusion_counts, len(keys)))
    representative = names.index("p0.05")
    rows.append([f"SAMARA {names[representative]}"] + binned(metric[representative], occlusion_counts, len(keys)))
    rows.append(["SAMARA selection"] + binned(select(metric, signal), occlusion_counts, len(keys)))
    rows.append(["per-clip oracle"] + binned(np.nanmax(metric, axis=0), occlusion_counts, len(keys)))
    write_csv(DATA / "claim_6/mean_iou_window.csv", ["method"] + OCCLUSION_LABELS, rows,
              bin_counts(occlusion_counts, len(keys)))

    # The figure: what selection ADDS over the best gate of the pool it is given. The y-axis is a
    # difference, so a pool of one is exactly zero -- with one gate the selector has nothing to choose.
    import style
    import matplotlib.pyplot as plt

    style.use_paper_style()
    sizes = list(range(1, len(names) + 1))
    gains, ceilings = {}, {}
    for size in sizes:
        gain, ceiling = [], []
        for combo in combinations(range(len(names)), size):
            pool, pool_signal = metric[list(combo)], signal[list(combo)]
            best = np.nanmax(np.nanmean(pool, axis=1))        # best gate of THIS pool, chosen with labels
            gain.append(float(np.nanmean(select(pool, pool_signal)) - best))
            ceiling.append(float(np.nanmean(np.nanmax(pool, axis=0)) - best))
        gains[size], ceilings[size] = np.array(gain), np.array(ceiling)

    figure, axis = plt.subplots(figsize=(style.COLUMN, 3.1))
    appearance = style.series_style(1, slots=style.SLOTS_6)
    axis.fill_between(sizes, [gains[s].min() for s in sizes], [gains[s].max() for s in sizes],
                      color=appearance["color"], alpha=0.13, linewidth=0, zorder=2)
    axis.plot(sizes, [ceilings[s].mean() for s in sizes], color=style.INK2, linewidth=0.8,
              linestyle=(0, (1, 2)), zorder=3, label="per-clip oracle")
    axis.plot(sizes, [gains[s].mean() for s in sizes], label="selection", markersize=3.6,
              linewidth=1.3, zorder=4, **appearance)
    axis.axhline(0.0, color=style.INK2, linewidth=0.8, linestyle=(0, (1, 2)), zorder=2)
    axis.set_xticks(sizes)
    axis.set_xlim(sizes[0] - 0.12, sizes[-1] + 0.12)
    axis.set_ylim(-0.003, 0.06)
    style.gridlines(axis, 0.01)
    style.style_axes(axis, "gates available to the selector", "mean IoU gained over the best gate")
    axis.legend(loc="upper left", ncol=2)
    figure.tight_layout()
    style.save(figure, DATA / "claim_6/paper/fig_pool_size")


CLAIMS = {"claim_1": claim_1, "claim_2": claim_2, "claim_5": claim_5, "claim_6": claim_6}


def main():
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or list(CLAIMS)
    unknown = [a for a in wanted if a not in CLAIMS]
    if unknown:
        raise SystemExit(f"unknown claim(s) {unknown}; choose from {list(CLAIMS)}")

    draw = reference_draw()
    print(f"reference draw: {len(draw)} clips\n")
    for name in wanted:
        CLAIMS[name](draw)


if __name__ == "__main__":
    main()

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VISIBLE_DIR = Path("data/person_path/visible")

COVERAGE_IOU = 0.5        # a visible frame counts as held at this box IoU
FAILURE_IOU = 0.1         # below this the target is considered lost (VOT's threshold)
BINS = 4
BOOTSTRAP = 10000
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
ARMS = {"sam baseline": "#cc4444", "samara gate": "#22aa77"}
SWEEP_COLOURS = {"coverage": "#cc4444", "hygiene": "#22aa77", "robustness": "#4477cc"}


# ---------------------------------------------------------------------------------------
# metrics, all read from the same per-frame arrays
# ---------------------------------------------------------------------------------------

def visible(clip, ious):
    """The scored frames: annotated and not occluded, so a missing box never counts as a miss."""

    return np.asarray(ious, dtype=float)[clip["has_box"] & ~clip["occluded"]]


def coverage(clip, ious, commits=None):
    """Fraction of visible post-occlusion frames held at box IoU >= COVERAGE_IOU."""

    scored = visible(clip, ious)
    return float((scored >= COVERAGE_IOU).mean()) if len(scored) else np.nan


def hygiene(clip, ious, commits):
    """Coverage over ALL annotated post-occlusion frames: a visible frame counts when it is held, an
    occluded frame when the arm did NOT write it to memory -- committing while the target is hidden is
    exactly how a bank gets poisoned."""

    seen, hidden = clip["has_box"] & ~clip["occluded"], np.asarray(clip["occluded"])
    scored = int(seen.sum() + hidden.sum())
    if not scored:
        return np.nan

    ious = np.asarray(ious, dtype=float)
    held = float((ious[seen] >= COVERAGE_IOU).sum())
    clean = float((~np.asarray(commits, dtype=bool)[hidden]).sum())
    return (held + clean) / scored


def robustness(clip, ious, commits=None):
    """Fraction of the visible frames preceding the first failure; 1.0 when the arm never fails."""

    scored = visible(clip, ious)
    if not len(scored):
        return np.nan
    failed = scored < FAILURE_IOU
    return float(int(np.argmax(failed)) / len(scored)) if failed.any() else 1.0


METRICS = [("coverage", coverage, f"coverage\n(visible frames at IoU >= {COVERAGE_IOU:g})"),
           ("hygiene", hygiene, "hygiene coverage\n(+ occluded frames not committed)"),
           ("robustness", robustness, f"robustness\n(frames before first IoU < {FAILURE_IOU:g})")]


# ---------------------------------------------------------------------------------------
# covariates
# ---------------------------------------------------------------------------------------

_boxes = {}


def anchor_area(clip):
    """The target's VISIBLE box area at its anchor, px^2 at the 1024 working resolution; NaN if missing."""

    video = clip["video"]
    if video not in _boxes:
        _boxes[video] = json.load(open(VISIBLE_DIR / f"{video}.json"))

    data = _boxes[video]
    resolution = data["metadata"]["resolution"]
    scale = 1024 / max(float(resolution["width"]), float(resolution["height"]))
    for entity in data["entities"]:
        if entity["id"] == clip["person"] and int(entity["blob"]["frame_idx"]) == int(clip["anchor"]):
            box = entity["bb"]
            return float(box[2]) * float(box[3]) * scale ** 2 if float(box[2]) > 0 else np.nan
    return np.nan


def occluded_frames(clip):
    """Number of frames of the scored span on which the target is hidden -- the stored occlusion flags."""

    hidden = np.asarray(clip["occluded"], dtype=bool)
    return float(hidden.sum()) if len(hidden) else np.nan


# ---------------------------------------------------------------------------------------
# pairing and drawing
# ---------------------------------------------------------------------------------------

def keyed(path):
    """Clip records indexed by (video, person)."""

    return {(clip["video"], clip["person"]): clip for clip in pickle.load(open(path, "rb"))["clips"]}


def paired(samara_clips, baseline_clips):
    """[(samara clip, baseline clip)] for the clips both experiments scored."""

    baseline = {(clip["video"], clip["person"]): clip for clip in baseline_clips}
    return [(clip, baseline[(clip["video"], clip["person"])]) for clip in samara_clips
            if (clip["video"], clip["person"]) in baseline]


def arm_scores(pairs, metric):
    """(sam, samara) per-clip arrays for one metric, in a shared clip order."""

    sam = np.array([metric(twin, twin["sam"], twin["sam_commit"]) for _, twin in pairs], dtype=float)
    samara = np.array([metric(clip, clip["samara"], clip["samara_commit"]) for clip, _ in pairs], dtype=float)
    return sam, samara


def interval(values, rng):
    """95% bootstrap interval for the mean, resampled over clips."""

    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    draws = rng.choice(values, (BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return np.percentile(draws, [2.5, 97.5])


def style(axis, title, xlabel, ylabel):
    """The shared chart furniture: recessive grid and axes, no top/right spines."""

    axis.set_xlabel(xlabel, fontsize=10, color=INK2)
    axis.set_ylabel(ylabel, fontsize=10, color=INK2)
    axis.set_title(title, fontsize=12, color=INK, pad=12, loc="left")
    axis.grid(axis="y", color=INK2, alpha=0.13, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK2)
        axis.spines[side].set_alpha(0.35)
    axis.tick_params(colors=INK2, labelsize=9)


def finish(figure, subtitle, filename):
    """Caption, tighten and write."""

    figure.text(0.008, 0.955, subtitle, fontsize=9, color=INK2, ha="left")
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(filename, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"saved {filename}")


def grouped_bars(groups, labels, title, subtitle, ylabel, filename):
    """One group of two bars per label; `groups` is [{arm: per-clip array}] aligned with `labels`."""

    rng = np.random.default_rng(0)
    figure, axis = plt.subplots(figsize=(9.0, 5.4), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    positions = np.arange(len(labels))
    width = 0.36

    for offset, arm in zip((-width / 2, width / 2), ARMS):
        means, lows, highs = [], [], []
        for group in groups:
            values = group[arm][np.isfinite(group[arm])]
            low, high = interval(values, rng)
            means.append(values.mean() if len(values) else np.nan)
            lows.append(means[-1] - low)
            highs.append(high - means[-1])

        bars = axis.bar(positions + offset, means, width, label=arm, color=ARMS[arm],
                        edgecolor=SURFACE, linewidth=2, zorder=3)
        axis.errorbar(positions + offset, means, yerr=[lows, highs], fmt="none",
                      ecolor=INK2, elinewidth=1.4, capsize=4, zorder=4)
        for bar, mean in zip(bars, means):
            axis.text(bar.get_x() + bar.get_width() / 2, mean + 0.02, f"{mean:.3f}",
                      ha="center", fontsize=9, color=INK2)

    axis.set_xticks(positions)
    axis.set_xticklabels(labels, fontsize=9, color=INK2)
    axis.set_ylim(0, 1.05)
    style(axis, title, "", ylabel)
    axis.legend(frameon=False, fontsize=10, loc="upper right")
    finish(figure, subtitle, filename)


def binned(pairs, covariate, metric):
    """(centres, groups, labels) for `metric` over equal-count bins of `covariate`.

    Equal-count rather than equal-width, so every point rests on the same number of clips."""

    values = np.array([covariate(clip) for clip, _ in pairs], dtype=float)
    keep = [pair for pair, value in zip(pairs, values) if np.isfinite(value)]
    finite = values[np.isfinite(values)]
    if len(finite) < BINS:
        return [], [], []          # early in a run there are not yet enough clips to fill the bins

    ranking = np.argsort(finite)
    centres, groups, labels = [], [], []
    for index, block in enumerate(np.array_split(ranking, BINS), start=1):
        block_pairs = [keep[i] for i in block]
        sam, samara = arm_scores(block_pairs, metric)
        centres.append(float(np.median(finite[block])))
        groups.append({"sam baseline": sam, "samara gate": samara})
        labels.append(f"Q{index} {finite[block].min():.3g}-{finite[block].max():.3g} n={len(block)}")
    return centres, groups, labels


def line_plot(pairs, covariate, metric, title, subtitle, xlabel, ylabel, filename, tick="{:.0f}"):
    """Both arms across quantile bins of `covariate`: categorical bins on the x axis, each tick carrying
    its range and count, and the arms labelled directly at the right edge."""

    values = np.array([covariate(clip) for clip, _ in pairs], dtype=float)
    keep = [pair for pair, value in zip(pairs, values) if np.isfinite(value)]
    finite = values[np.isfinite(values)]
    if len(finite) < BINS:
        print(f"skipped {filename}  (n={len(finite)})")
        return

    edges = np.unique(np.quantile(finite, np.linspace(0, 1, BINS + 1)))
    index = np.clip(np.digitize(finite, edges[1:-1]), 0, len(edges) - 2)
    x = np.arange(len(edges) - 1)

    figure, axis = plt.subplots(figsize=(8.6, 5.4), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    ends = []
    for arm, colour in ARMS.items():
        scored = np.array([metric(clip, clip["samara"], clip["samara_commit"]) if arm == "samara gate"
                           else metric(twin, twin["sam"], twin["sam_commit"]) for clip, twin in keep])
        per_bin = np.array([np.nanmean(scored[index == k]) for k in x])
        axis.plot(x, per_bin, color=colour, linewidth=2, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, label=arm, zorder=3)
        ends.append((per_bin[-1], colour, arm.split()[0]))

    # Direct labels at the right edge, nudged apart when the two arms finish at nearly the same value.
    gap = max(max(end[0] for end in ends) - min(end[0] for end in ends), 0.05) * 0.16
    placed = []
    for value, colour, short in sorted(ends):
        y = value if not placed else max(value, placed[-1] + gap)
        placed.append(y)
        axis.annotate(short, (x[-1], value), xytext=(x[-1] + 0.12, y), textcoords="data",
                      color=colour, fontsize=10, va="center")

    axis.set_xticks(x)
    axis.set_xticklabels([f"{tick.format(edges[k])}-{tick.format(edges[k + 1])}"
                          f"\nn={int((index == k).sum())}" for k in x], fontsize=9, color=INK2)
    axis.set_xlim(-0.35, len(x) - 1 + 0.55)
    style(axis, title, xlabel, ylabel)
    axis.legend(frameon=False, fontsize=10, loc="best")
    finish(figure, subtitle, filename)


def report(title, groups, labels):
    """The same numbers as the figure, with the paired difference and win count per bin."""

    rng = np.random.default_rng(0)
    print(f"\n{title}")
    print(f"{'bin':22}{'n':>4}{'sam':>9}{'samara':>9}{'diff':>9}{'95% CI of diff':>22}{'wins':>10}")
    for group, label in zip(groups, labels):
        sam, samara = group["sam baseline"], group["samara gate"]
        keep = np.isfinite(sam) & np.isfinite(samara)
        difference = samara[keep] - sam[keep]
        low, high = interval(difference, rng)
        print(f"{label.replace(chr(10), ' '):22}{int(keep.sum()):>4}{sam[keep].mean():>9.3f}"
              f"{samara[keep].mean():>9.3f}{difference.mean():>+9.3f}"
              f"{f'[{low:+.3f}, {high:+.3f}]':>22}{f'{int((difference > 0).sum())}/{int(keep.sum())}':>10}")


HYGIENE_LABEL = f"coverage  (visible: box IoU ≥ {COVERAGE_IOU:g}   ·   occluded: did not commit)"
SPLITS = ((anchor_area, "anchor size", "anchor visible box area  (px² @1024)", "fig_by_area", "{:.0f}"),
          (occluded_frames, "number of occlusion frames", "number of occlusion frames",
           "fig_by_occlusion", "{:.0f}"))


def draw_arm(results_path, baseline_clips, out_dir):
    """Every figure for one arm, written beside its results."""

    pairs = paired(pickle.load(open(results_path, "rb"))["clips"], baseline_clips)
    if not pairs:
        print(f"{results_path}: no clip is present in both experiments")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    caption = (f"claim_5  ·  {len(pairs)} clips scored by both experiments  ·  SAM picks the mask in both "
               f"arms, the commit gate is the only difference  ·  95% bootstrap intervals over clips")

    headline = [{"sam baseline": arm_scores(pairs, metric)[0], "samara gate": arm_scores(pairs, metric)[1]}
                for _, metric, _ in METRICS]
    grouped_bars(headline, [label for _, _, label in METRICS],
                 "SAMARA's commit gate against SAM's own", caption,
                 "mean over clips", out_dir / "fig_gate_vs_baseline.png")
    report("overall", headline, [name for name, _, _ in METRICS])

    for covariate, name, xlabel, filename, tick in SPLITS:
        line_plot(pairs, covariate, hygiene, f"Coverage + memory hygiene by {name}",
                  f"n={len(pairs)} clips  ·  SAM picks the mask in both arms, the commit gate is the only "
                  f"difference", xlabel, HYGIENE_LABEL, out_dir / f"{filename}.png", tick)
        report(f"by {name}", *binned(pairs, covariate, hygiene)[1:])

    # The occlusion split again with the LARGEST anchors dropped. The two arms converge on the top quartile
    # -- a big target is easy and an appearance check has nothing to add -- so those clips only dilute it.
    areas = np.array([anchor_area(clip) for clip, _ in pairs], dtype=float)
    cut = np.nanquantile(areas, 0.75)
    small = [pair for pair, area in zip(pairs, areas) if np.isfinite(area) and area < cut]

    line_plot(small, occluded_frames, hygiene, "Coverage + memory hygiene by number of occlusion frames",
              f"n={len(small)} clips, largest-anchor quartile dropped (area < {cut:.0f} px²)  ·  "
              f"SAM picks the mask in both arms, the commit gate is the only difference",
              "number of occlusion frames", HYGIENE_LABEL,
              out_dir / "fig_by_occlusion_small.png", "{:.0f}")
    report(f"by number of occlusion frames  (anchor area < {cut:.0f} px², top quartile dropped)",
           *binned(small, occluded_frames, hygiene)[1:])


def draw_sweep(arms, baseline, out_dir):
    """Each metric's paired margin over SAM against the arm's corruption level."""

    shared = sorted(set.intersection(*(set(clips) for clips in arms.values())) & set(baseline))
    print(f"\nsweep: {len(arms)} arms, {len(shared)} clips scored by all of them\n")
    if len(shared) < 2:
        print("not enough shared clips for the sweep figure yet")
        return

    rng = np.random.default_rng(0)
    figure, axis = plt.subplots(figsize=(8.6, 5.4), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    x = np.arange(len(arms))

    print(f"{'metric':12}" + "".join(f"{name:>12}" for name in arms))
    for label, metric, _ in METRICS:
        sam = np.array([metric(baseline[k], baseline[k]["sam"], baseline[k]["sam_commit"]) for k in shared])
        means, lows, highs = [], [], []
        for clips in arms.values():
            arm = np.array([metric(clips[k], clips[k]["samara"], clips[k]["samara_commit"]) for k in shared])
            keep = np.isfinite(arm) & np.isfinite(sam)
            difference = arm[keep] - sam[keep]
            low, high = interval(difference, rng)
            means.append(difference.mean()); lows.append(low); highs.append(high)

        colour = SWEEP_COLOURS[label]
        axis.fill_between(x, lows, highs, color=colour, alpha=0.12, linewidth=0)
        axis.plot(x, means, color=colour, linewidth=2, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        axis.annotate(label, (x[-1], means[-1]), xytext=(x[-1] + 0.10, means[-1]), textcoords="data",
                      color=colour, fontsize=10, va="center")
        print(f"{label:12}" + "".join(f"{value:>+12.3f}" for value in means))

    axis.axhline(0.0, color=INK2, linewidth=1.2, alpha=0.5, zorder=2)
    axis.set_xticks(x)
    axis.set_xticklabels([f"{name}\n{float(name.lstrip('p')):g}" for name in arms], fontsize=9, color=INK2)
    axis.set_xlim(-0.35, len(x) - 1 + 0.55)
    style(axis, "How much memory corruption should the commit gate see?",
          "corruption level the gate trains on  (clean rollouts kept in every arm)",
          "margin over SAM baseline  (paired, per clip)")
    axis.legend(frameon=False, fontsize=10, loc="best")
    finish(figure, f"claim_5 sweep  ·  {len(shared)} clips scored by every arm  ·  selection off, the "
                   f"commit gate is the only intervention  ·  95% bootstrap bands over clips",
           out_dir / "fig_sweep.png")


def main():
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/claim_5")
    baseline_path = Path(sys.argv[2] if len(sys.argv) > 2 else "data/claim_1/results.pkl")
    baseline_clips = pickle.load(open(baseline_path, "rb"))["clips"]

    if target.is_file():
        draw_arm(target, baseline_clips, target.parent)
        return

    arm_paths = sorted(target.glob("*/results.pkl"))
    if not arm_paths:
        raise SystemExit(f"no arm results under {target} (expected <arm>/results.pkl)")

    for path in arm_paths:
        print(f"\n{'=' * 20} {path.parent.name} {'=' * 20}")
        draw_arm(path, baseline_clips, path.parent)

    if len(arm_paths) > 1:
        draw_sweep({path.parent.name: keyed(path) for path in arm_paths},
                   {(clip["video"], clip["person"]): clip for clip in baseline_clips}, target)


if __name__ == "__main__":
    main()

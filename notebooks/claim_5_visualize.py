"""claim_5 figures: every gate arm against SAM's own gate, on the covariates where the gate should matter.

claim_5 stores per-frame IoUs over claim_1's span, so the two pkls join on (video, person) and every metric
is computed the same way for every arm from the same frames. Only clips EVERY arm and the baseline scored
are used, so a slow arm cannot shift the comparison by contributing fewer clips. SAM's IoU token picks the
mask in all of them -- the commit gate is the only difference.

Three figures, each carrying the baseline and all arms:

    fig_by_area           hygiene coverage against the target's visible box area at the anchor. A small
                          target is where SAM's own confidence is least reliable, so it is where an
                          appearance check has the most to add.

    fig_by_occlusion      hygiene coverage against how many frames of the scored span the target is hidden
                          for. This is the covariate the claim is really about: the gate exists to keep the
                          bank clean through an occlusion. Counted in frames rather than as a share of the
                          span, so a long clip with a long occlusion is not filed beside a short clip with
                          a brief one.

    fig_by_occlusion_small  the same split with the largest-anchor quartile dropped. The arms converge on
                          big targets, so those clips only dilute the occlusion trend.

HYGIENE is the metric throughout: a visible frame counts when it is held, an occluded frame when the arm
refrained from committing -- the decision the gate actually makes while there is nothing to track. One
caveat travels with the occlusion figures: hygiene credits an occluded frame for NOT committing, which
every arm mostly gets right, so a heavily occluded bin draws more of its score from the easier half. That
shifts every arm together and so does not bias the DIFFERENCES between them, but it does mean the absolute
level is not comparable across bins.

Printed tables carry 95% bootstrap intervals on each arm's paired margin over SAM, resampled over CLIPS --
the independent unit. Frames within a clip are not.

    python notebooks/claim_5_visualize.py [data/claim_5] [claim_1/results.pkl]
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent / "paper"))

import style

VISIBLE_DIR = Path("data/person_path/visible")

COVERAGE_IOU = 0.5        # a visible frame counts as held at this box IoU
BINS = 4
SMALL_ANCHOR_QUANTILE = 0.50   # the small-anchor figure keeps clips below this quantile of anchor area

# A derived arm that picks a gate by anchor size: big targets take the first, small ones the second.
# The heatmaps say the clean gate is fine on big anchors and the corrupted gate wins on small ones, so
# this asks what taking both would be worth. It is NOT a deployable result -- the split point and the
# pairing are chosen on the same clips they are scored on, which makes it an upper bound.
COMBINED = ("p0.00", "p0.15")
BOOTSTRAP = 10000
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"

# SAM is the reference and gets the one warm hue; the arms are an ordered quantity -- corruption level --
# so they take a single-hue sequential ramp, light for clean through dark for the most corrupted.
BASELINE_COLOUR = "#cc4444"
ARM_RAMP = ["#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"]

# Worse-than-SAM and better-than-SAM are opposite states, not two ends of one magnitude, so the heatmap
# takes a diverging scale: two hues meeting at a NEUTRAL GRAY midpoint, never a hue at zero.
#
# The poles are red and BLUE, not red and green. A red-green map is the classic accessibility failure and
# it fails here by measurement, not by reputation: the two poles separate by only 5.5 in OKLab under
# deuteranopia -- below the floor of 6 -- so the sign of a cell, which is the whole point of the figure,
# would be invisible to a deuteranope. Red against blue separates by 26.8 under the same simulation.
DIVERGING = LinearSegmentedColormap.from_list(
    "gate_vs_sam", ["#7f2222", "#cc4444", "#e6b3b3", "#f0efec", "#a9c8ea", "#2a78d6", "#154272"])


# ---------------------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------------------

def hygiene(clip, ious, commits):
    """Coverage over ALL annotated post-occlusion frames: a visible frame counts when it is held, an
    occluded frame when the arm did NOT write it to memory -- committing while the target is hidden is
    exactly how a bank gets poisoned."""

    seen = clip["has_box"] & ~clip["occluded"]
    hidden = np.asarray(clip["occluded"])
    scored = int(seen.sum() + hidden.sum())
    if not scored:
        return np.nan

    ious = np.asarray(ious, dtype=float)
    held = float((ious[seen] >= COVERAGE_IOU).sum())
    clean = float((~np.asarray(commits, dtype=bool)[hidden]).sum())
    return (held + clean) / scored


def arm_hygiene(clips, keys):
    """One arm's hygiene per clip, in the given clip order."""

    return np.array([hygiene(clips[k], clips[k]["samara"], clips[k]["samara_commit"]) for k in keys])


def baseline_hygiene(baseline, keys):
    """SAM's hygiene per clip, in the given clip order."""

    return np.array([hygiene(baseline[k], baseline[k]["sam"], baseline[k]["sam_commit"]) for k in keys])


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
        matches_person = entity["id"] == clip["person"]
        matches_anchor = int(entity["blob"]["frame_idx"]) == int(clip["anchor"])
        if matches_person and matches_anchor:
            box = entity["bb"]
            if float(box[2]) <= 0:
                return np.nan
            return float(box[2]) * float(box[3]) * scale ** 2
    return np.nan


def occluded_frames(clip):
    """Number of frames of the scored span on which the target is hidden -- the stored occlusion flags."""

    hidden = np.asarray(clip["occluded"], dtype=bool)
    return float(hidden.sum()) if len(hidden) else np.nan


# ---------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------

def keyed(results_path):
    """Clip records indexed by (video, person)."""

    clips = pickle.load(open(results_path, "rb"))["clips"]
    return {(clip["video"], clip["person"]): clip for clip in clips}


def combined_arm(arms, keys, quantile=SMALL_ANCHOR_QUANTILE):
    """A synthetic arm taking `COMBINED[0]` on large anchors and `COMBINED[1]` on small ones."""

    large, small = arms[COMBINED[0]], arms[COMBINED[1]]
    areas = np.array([anchor_area(large[k]) for k in keys], dtype=float)
    cut = np.nanquantile(areas, quantile)
    return {k: (small[k] if np.isfinite(a) and a < cut else large[k]) for k, a in zip(keys, areas)}


def shared_clips(arms, baseline):
    """The clips every arm AND the baseline scored, so all the lines rest on the same set."""

    scored_by_all = set.intersection(*(set(clips) for clips in arms.values()))
    return sorted(scored_by_all & set(baseline))


def interval(values, rng):
    """95% bootstrap interval for the mean, resampled over clips."""

    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan

    draws = rng.choice(values, (BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return np.percentile(draws, [2.5, 97.5])


# ---------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------

def quantile_bins(values):
    """(bin index per clip, edges) for equal-count bins, so every point rests on the same many clips."""

    edges = np.unique(np.quantile(values, np.linspace(0, 1, BINS + 1)))
    index = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    return index, edges


def heatmap(arms, baseline, keys, covariate, title, subtitle, xlabel, filename, tick="{:.0f}"):
    """Relative improvement over SAM per (corruption level, covariate bin).

    A cell is `(arm - sam) / sam` on that bin -- a margin of 0.05 where SAM scores 0.5 reads as +10%.
    Dividing by SAM's own level matters here because the bins are not equally hard: the same absolute
    margin is worth far more in a bin where SAM scores 0.51 than one where it scores 0.77.

    Diverging colour, neutral at zero, symmetric limits -- so worse-than-SAM is visibly a different thing
    from better-than-SAM rather than just a smaller number."""

    reference = next(iter(arms.values()))
    values = np.array([covariate(reference[k]) for k in keys], dtype=float)
    finite = np.isfinite(values)
    keys = [key for key, ok in zip(keys, finite) if ok]
    values = values[finite]

    if len(values) < BINS:
        print(f"skipped {filename}  (n={len(values)})")
        return

    index, edges = quantile_bins(values)
    sam = baseline_hygiene(baseline, keys)
    n_bins = len(edges) - 1

    relative = np.full((len(arms), n_bins), np.nan)
    for row, clips in enumerate(arms.values()):
        scored = arm_hygiene(clips, keys)
        for column in range(n_bins):
            in_bin = index == column
            sam_level = np.nanmean(sam[in_bin])
            if sam_level > 0:
                relative[row, column] = np.nanmean(scored[in_bin]) / sam_level - 1.0

    limit = float(np.nanmax(np.abs(relative))) or 1.0
    # Height follows the row count so cells stay close to square whatever the number of arms.
    figure, axis = plt.subplots(figsize=(style.COLUMN, 0.42 * len(arms) + 1.2))
    image = axis.imshow(relative, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")

    for row in range(len(arms)):
        for column in range(n_bins):
            value = relative[row, column]
            if not np.isfinite(value):
                continue
            # Ink flips to the surface colour only where the cell is dark enough to swallow black text.
            shade = style.INK if abs(value) < 0.6 * limit else SURFACE
            axis.text(column, row, f"{value:+.1%}", ha="center", va="center", fontsize=7, color=shade)

    # A thin white rule between cells: the grid a heatmap needs is the gap, not a drawn line.
    axis.set_xticks(np.arange(n_bins + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(arms) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=1.2)
    axis.tick_params(which="minor", length=0)

    axis.set_xticks(np.arange(n_bins))
    axis.set_xticklabels([f"{tick.format(edges[k])}-{tick.format(edges[k + 1])}" for k in range(n_bins)])
    axis.set_yticks(np.arange(len(arms)))
    axis.set_yticklabels(list(arms))
    axis.set_xlabel(xlabel)
    axis.set_ylabel("corruption level the gate trains on")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)

    bar = figure.colorbar(image, ax=axis, fraction=0.030, pad=0.02)
    bar.set_label("memory hygiene, relative to SAM")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2)

    counts = ", ".join(str(int((index == k).sum())) for k in range(n_bins))
    style.save(figure, filename)
    print(f"   caption: {subtitle} Bins hold {counts} clips.")


def line_plot(arms, baseline, keys, covariate, title, subtitle, xlabel, filename, tick="{:.0f}"):
    """Every arm's MARGIN over SAM across quantile bins of `covariate`.

    Plotting the margin rather than the absolute hygiene puts SAM on the zero line, so the quantity the
    experiment is about -- does the gate help in this bin -- is read off the y axis instead of measured as
    the vertical gap between two curves. It also drops one line and spreads the arms apart, since their
    margins differ by much more than their absolute levels do."""

    reference = next(iter(arms.values()))
    values = np.array([covariate(reference[k]) for k in keys], dtype=float)
    finite = np.isfinite(values)
    keys = [key for key, ok in zip(keys, finite) if ok]
    values = values[finite]

    if len(values) < BINS:
        print(f"skipped {filename}  (n={len(values)})")
        return

    index, edges = quantile_bins(values)
    x = np.arange(len(edges) - 1)
    sam = baseline_hygiene(baseline, keys)

    figure, axis = plt.subplots(figsize=(9.0, 5.4), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    axis.axhline(0.0, color=BASELINE_COLOUR, linewidth=2, zorder=2)
    axis.annotate("sam baseline", (x[-1] + 0.12, 0.0), color=BASELINE_COLOUR, fontsize=9, va="center")

    for position, (arm, clips) in enumerate(arms.items()):
        margin = arm_hygiene(clips, keys) - sam
        per_bin = np.array([np.nanmean(margin[index == k]) for k in x])
        colour = ARM_RAMP[position % len(ARM_RAMP)]
        axis.plot(x, per_bin, color=colour, linewidth=2, marker="o", markersize=8,
                  markeredgecolor=SURFACE, markeredgewidth=2, label=arm, zorder=3)

    axis.set_xticks(x)
    axis.set_xticklabels([f"{tick.format(edges[k])}-{tick.format(edges[k + 1])}"
                          f"\nn={int((index == k).sum())}" for k in x], fontsize=9, color=INK2)
    axis.set_xlim(-0.35, len(x) - 1 + 0.75)
    axis.set_xlabel(xlabel, fontsize=10, color=INK2)
    axis.set_ylabel("memory hygiene, margin over SAM  (paired, per clip)", fontsize=10, color=INK2)
    axis.set_title(title, fontsize=12, color=INK, pad=12, loc="left")
    axis.grid(axis="y", color=INK2, alpha=0.13, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK2)
        axis.spines[side].set_alpha(0.35)
    axis.tick_params(colors=INK2, labelsize=9)
    axis.legend(frameon=False, fontsize=9, loc="best", title="gate trained on", ncol=2)

    figure.text(0.008, 0.955, subtitle, fontsize=9, color=INK2, ha="left")
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(filename, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def report(arms, baseline, keys, covariate, title):
    """Each arm's hygiene per bin, with its paired margin over SAM."""

    reference = next(iter(arms.values()))
    values = np.array([covariate(reference[k]) for k in keys], dtype=float)
    finite = np.isfinite(values)
    keys = [key for key, ok in zip(keys, finite) if ok]
    values = values[finite]
    if len(values) < BINS:
        return

    index, edges = quantile_bins(values)
    rng = np.random.default_rng(0)
    sam = baseline_hygiene(baseline, keys)

    print(f"\n{title}")
    header = f"{'bin':22}{'n':>5}{'sam':>8}"
    for arm in arms:
        header += f"{arm:>17}"
    print(header)

    for k in range(len(edges) - 1):
        in_bin = index == k
        row = f"{f'{edges[k]:.3g}-{edges[k + 1]:.3g}':22}{int(in_bin.sum()):>5}{np.nanmean(sam[in_bin]):>8.3f}"
        for arm, clips in arms.items():
            scored = arm_hygiene(clips, keys)
            difference = scored[in_bin] - sam[in_bin]
            keep = np.isfinite(difference)
            low, high = interval(difference[keep], rng)
            row += f"  {np.nanmean(scored[in_bin]):.3f} {difference[keep].mean():+.3f}"
            row += "*" if low > 0 or high < 0 else " "
        print(row)

    print("  each arm: hygiene, then paired margin over sam; * marks a 95% bootstrap CI clear of zero")


# ---------------------------------------------------------------------------------------
# the figures
# ---------------------------------------------------------------------------------------

def main():
    style.use_paper_style()
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/claim_5")
    baseline_path = Path(sys.argv[2] if len(sys.argv) > 2 else "data/claim_1/results.pkl")

    arm_paths = sorted(target.glob("*/results.pkl"))
    if not arm_paths:
        raise SystemExit(f"no arm results under {target} (expected <arm>/results.pkl)")

    arms = {path.parent.name: keyed(path) for path in arm_paths}
    baseline = keyed(baseline_path)
    keys = shared_clips(arms, baseline)

    if all(name in arms for name in COMBINED):
        label = f"{COMBINED[0]}/{COMBINED[1]}"
        arms[label] = combined_arm(arms, keys)
        print(f"derived arm {label}: {COMBINED[1]} on the smallest "
              f"{SMALL_ANCHOR_QUANTILE:.0%} of anchors, {COMBINED[0]} on the rest (chosen post-hoc)")

    print(f"{len(arms)} arms ({', '.join(arms)}), {len(keys)} clips scored by every arm")
    if len(keys) < BINS:
        raise SystemExit("not enough shared clips to bin yet")

    caption = (f"claim_5  ·  {len(keys)} clips scored by every arm  ·  SAM picks the mask in all of them, "
               f"the commit gate is the only difference")

    heatmap(arms, baseline, keys, anchor_area,
              "Coverage + memory hygiene by anchor size", caption,
              "anchor visible box area (px$^2$ @1024)", target / "fig_by_area", "{:.0f}")
    report(arms, baseline, keys, anchor_area, "by anchor size")

    heatmap(arms, baseline, keys, occluded_frames,
              "Coverage + memory hygiene by number of occlusion frames", caption,
              "number of occlusion frames", target / "fig_by_occlusion", "{:.0f}")
    report(arms, baseline, keys, occluded_frames, "by number of occlusion frames")

    # The occlusion split again restricted to the SMALLEST anchors. Every arm converges with SAM on big
    # targets -- the memory decision has little left to change when the target is easy to segment -- so
    # those clips only dilute the occlusion trend. SMALL_ANCHOR_QUANTILE sets how much is kept: 0.75 drops
    # the top quartile, 0.5 keeps only the bottom half.
    reference = next(iter(arms.values()))
    areas = np.array([anchor_area(reference[k]) for k in keys], dtype=float)
    cut = np.nanquantile(areas, SMALL_ANCHOR_QUANTILE)
    small = [key for key, area in zip(keys, areas) if np.isfinite(area) and area < cut]

    small_caption = (f"claim_5  ·  {len(small)} clips, smallest {SMALL_ANCHOR_QUANTILE:.0%} of anchors "
                     f"(area < {cut:.0f} px²)  ·  the commit gate is the only difference")
    heatmap(arms, baseline, small, occluded_frames,
              "Coverage + memory hygiene by number of occlusion frames", small_caption,
              "number of occlusion frames", target / "fig_by_occlusion_small", "{:.0f}")
    report(arms, baseline, small, occluded_frames,
           f"by number of occlusion frames  (anchor area < {cut:.0f} px², smallest "
           f"{SMALL_ANCHOR_QUANTILE:.0%} of anchors)")


if __name__ == "__main__":
    main()

"""claim_5 figures: every gate arm against SAM's own gate, on the covariates where the gate should matter."""

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

COVERAGE_IOU = 0.5
BINS = 4
SMALL_ANCHOR_QUANTILE = 0.50


COMBINED = ("p0.00", "p0.15")
BOOTSTRAP = 10000
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


BASELINE_COLOUR = "#cc4444"
ARM_RAMP = ["#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"]


DIVERGING = LinearSegmentedColormap.from_list("gate_vs_sam", ["#7f2222", "#cc4444", "#e6b3b3", "#f0efec", "#a9c8ea", "#2a78d6", "#154272"])


def hygiene(clip_record, box_ious, commits):
    """Coverage over ALL annotated post-occlusion frames: a visible frame counts when it is held, an occluded frame
    when the arm did NOT write it to memory -- committing while the target is hidden is exactly how a bank gets"""

    seen = clip_record["has_box"] & ~clip_record["occluded"]
    occluded_frames = np.asarray(clip_record["occluded"])
    scored_values = int(seen.sum() + occluded_frames.sum())
    if not scored_values:
        return np.nan

    box_ious = np.asarray(box_ious, dtype=float)
    held = float((box_ious[seen] >= COVERAGE_IOU).sum())
    clean = float((~np.asarray(commits, dtype=bool)[occluded_frames]).sum())
    return (held + clean) / scored_values


def arm_hygiene(scored_clips, clip_keys):
    """One arm's hygiene per clip, in the given clip order."""

    return np.array(
        [
            hygiene(scored_clips[clip_key], scored_clips[clip_key]["samara"], scored_clips[clip_key]["samara_commit"])
            for clip_key in clip_keys
        ]
    )


def baseline_hygiene(baseline, clip_keys):
    """SAM's hygiene per clip, in the given clip order."""

    return np.array(
        [
            hygiene(baseline[clip_key], baseline[clip_key]["sam"], baseline[clip_key]["sam_commit"])
            for clip_key in clip_keys
        ]
    )


_boxes = {}


def anchor_area(clip_record):
    """The target's VISIBLE box area at its anchor, px^2 at the 1024 working resolution; NaN if missing."""

    video = clip_record["video"]
    if video not in _boxes:
        _boxes[video] = json.load(open(VISIBLE_DIR / f"{video }.json"))

    data = _boxes[video]
    resolution = data["metadata"]["resolution"]
    scale = 1024 / max(float(resolution["width"]), float(resolution["height"]))

    for entity in data["entities"]:
        matches_person = entity["id"] == clip_record["person"]
        matches_anchor = int(entity["blob"]["frame_idx"]) == int(clip_record["anchor"])
        if matches_person and matches_anchor:
            box = entity["bb"]
            if float(box[2]) <= 0:
                return np.nan
            return float(box[2]) * float(box[3]) * scale**2
    return np.nan


def occluded_frames(clip_record):
    """Number of frames of the scored span on which the target is hidden -- the stored occlusion flags."""

    occluded_frames = np.asarray(clip_record["occluded"], dtype=bool)
    return float(occluded_frames.sum()) if len(occluded_frames) else np.nan


def keyed(results_path):
    """Clip records indexed by (video, person)."""

    scored_clips = pickle.load(open(results_path, "rb"))["clips"]
    return {(clip_record["video"], clip_record["person"]): clip_record for clip_record in scored_clips}


def combined_arm(arm_names, clip_keys, quantile=SMALL_ANCHOR_QUANTILE):
    """A synthetic arm taking `COMBINED[0]` on large anchors and `COMBINED[1]` on small ones."""

    large, small_anchor_mask = arm_names[COMBINED[0]], arm_names[COMBINED[1]]
    anchor_areas = np.array([anchor_area(large[clip_key]) for clip_key in clip_keys], dtype=float)
    bin_cut = np.nanquantile(anchor_areas, quantile)
    return {
        clip_key: (small_anchor_mask[clip_key] if np.isfinite(a) and a < bin_cut else large[clip_key])
        for clip_key, a in zip(clip_keys, anchor_areas)
    }


def shared_clips(arm_names, baseline):
    """The clips every arm AND the baseline scored, so all the lines rest on the same set."""

    scored_by_all = set.intersection(*(set(scored_clips) for scored_clips in arm_names.values()))
    return sorted(scored_by_all & set(baseline))


def interval(metric_values, random_generator):
    """95% bootstrap interval for the mean, resampled over clips."""

    metric_values = metric_values[np.isfinite(metric_values)]
    if not len(metric_values):
        return np.nan, np.nan

    draws = random_generator.choice(metric_values, (BOOTSTRAP, len(metric_values)), replace=True).mean(axis=1)
    return np.percentile(draws, [2.5, 97.5])


def quantile_bins(metric_values):
    """(bin index per clip, edges) for equal-count bins, so every point rests on the same many clips."""

    bin_edges = np.unique(np.quantile(metric_values, np.linspace(0, 1, BINS + 1)))
    bin_index = np.clip(np.digitize(metric_values, bin_edges[1:-1]), 0, len(bin_edges) - 2)
    return bin_index, bin_edges


def heatmap(
    arm_names, baseline, clip_keys, covariate, table_title, subtitle, x_axis_label, filename, tick_format="{:.0f}"
):
    """Relative improvement over SAM per (corruption level, covariate bin)."""

    reference = next(iter(arm_names.values()))
    metric_values = np.array([covariate(reference[clip_key]) for clip_key in clip_keys], dtype=float)
    finite_values = np.isfinite(metric_values)
    clip_keys = [record_key for record_key, usable_mask in zip(clip_keys, finite_values) if usable_mask]
    metric_values = metric_values[finite_values]

    if len(metric_values) < BINS:
        return

    bin_index, bin_edges = quantile_bins(metric_values)
    baseline_scores = baseline_hygiene(baseline, clip_keys)
    n_bins = len(bin_edges) - 1

    relative = np.full((len(arm_names), n_bins), np.nan)
    for table_row, scored_clips in enumerate(arm_names.values()):
        scored_values = arm_hygiene(scored_clips, clip_keys)
        for column_values in range(n_bins):
            in_bin = bin_index == column_values
            sam_level = np.nanmean(baseline_scores[in_bin])
            if sam_level > 0:
                relative[table_row, column_values] = np.nanmean(scored_values[in_bin]) / sam_level - 1.0

    limit = float(np.nanmax(np.abs(relative))) or 1.0

    figure_handle, axis = plt.subplots(figsize=(style.COLUMN, 0.42 * len(arm_names) + 1.2))
    image = axis.imshow(relative, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")

    for table_row in range(len(arm_names)):
        for column_values in range(n_bins):
            value = relative[table_row, column_values]
            if not np.isfinite(value):
                continue

            shade = style.INK if abs(value) < 0.6 * limit else SURFACE
            axis.text(column_values, table_row, f"{value:+.1%}", ha="center", va="center", fontsize=7, color=shade)

    axis.set_xticks(np.arange(n_bins + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(arm_names) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=1.2)
    axis.tick_params(which="minor", length=0)

    axis.set_xticks(np.arange(n_bins))
    axis.set_xticklabels(
        [
            f"{tick_format.format(bin_edges[clip_key])}-{tick_format.format(bin_edges[clip_key +1])}"
            for clip_key in range(n_bins)
        ]
    )
    axis.set_yticks(np.arange(len(arm_names)))
    axis.set_yticklabels(list(arm_names))
    axis.set_xlabel(x_axis_label)
    axis.set_ylabel("corruption level the gate trains on")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)

    bar = figure_handle.colorbar(image, ax=axis, fraction=0.030, pad=0.02)
    bar.set_label("memory hygiene, relative to SAM")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2)

    counts = ", ".join(str(int((bin_index == clip_key).sum())) for clip_key in range(n_bins))
    style.save(figure_handle, filename)


def report(arm_names, baseline, clip_keys, covariate, table_title):
    """Each arm's hygiene per bin, with its paired margin over SAM."""

    reference = next(iter(arm_names.values()))
    metric_values = np.array([covariate(reference[clip_key]) for clip_key in clip_keys], dtype=float)
    finite_values = np.isfinite(metric_values)
    clip_keys = [record_key for record_key, usable_mask in zip(clip_keys, finite_values) if usable_mask]
    metric_values = metric_values[finite_values]
    if len(metric_values) < BINS:
        return

    bin_index, bin_edges = quantile_bins(metric_values)
    random_generator = np.random.default_rng(0)
    baseline_scores = baseline_hygiene(baseline, clip_keys)

    table_header = f"{'bin':22}{'n':>5}{'sam':>8}"
    for arm_label in arm_names:
        table_header += f"{arm_label:>17}"

    for clip_key in range(len(bin_edges) - 1):
        in_bin = bin_index == clip_key
        table_row = f"{f'{bin_edges[clip_key]:.3g}-{bin_edges[clip_key +1]:.3g}':22}{int(in_bin.sum()):>5}{np.nanmean(baseline_scores[in_bin]):>8.3f}"
        for arm_label, scored_clips in arm_names.items():
            scored_values = arm_hygiene(scored_clips, clip_keys)
            difference = scored_values[in_bin] - baseline_scores[in_bin]
            keep = np.isfinite(difference)
            low, high = interval(difference[keep], random_generator)
            table_row += f"  {np.nanmean(scored_values[in_bin]):.3f} {difference[keep].mean():+.3f}"
            table_row += "*" if low > 0 or high < 0 else " "



def main():
    style.use_paper_style()
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/claim_5")
    baseline_path = Path(sys.argv[2] if len(sys.argv) > 2 else "data/claim_1/results.pkl")

    arm_paths = sorted(target.glob("*/results.pkl"))
    if not arm_paths:
        raise SystemExit(f"no arm results under {target } (expected <arm>/results.pkl)")

    arm_names = {path.parent.name: keyed(path) for path in arm_paths}
    baseline = keyed(baseline_path)
    clip_keys = shared_clips(arm_names, baseline)

    if all(name in arm_names for name in COMBINED):
        label = f"{COMBINED[0]}/{COMBINED[1]}"
        arm_names[label] = combined_arm(arm_names, clip_keys)

    if len(clip_keys) < BINS:
        raise SystemExit("not enough shared clips to bin yet")

    caption = (f"claim_5 · {len(clip_keys)} clips scored by every arm · SAM picks the mask in all of them, " f"the commit gate is the only difference")

    heatmap(
        arm_names,
        baseline,
        clip_keys,
        anchor_area,
        "Coverage + memory hygiene by anchor size",
        caption,
        "anchor visible box area (px$^2$ @1024)",
        target / "fig_by_area",
        "{:.0f}")
    report(arm_names, baseline, clip_keys, anchor_area, "by anchor size")

    heatmap(
        arm_names,
        baseline,
        clip_keys,
        occluded_frames,
        "Coverage + memory hygiene by number of occlusion frames",
        caption,
        "number of occlusion frames",
        target / "fig_by_occlusion",
        "{:.0f}")
    report(arm_names, baseline, clip_keys, occluded_frames, "by number of occlusion frames")

    reference = next(iter(arm_names.values()))
    anchor_areas = np.array([anchor_area(reference[clip_key]) for clip_key in clip_keys], dtype=float)
    bin_cut = np.nanquantile(anchor_areas, SMALL_ANCHOR_QUANTILE)
    small_anchor_mask = [record_key for record_key, area in zip(clip_keys, anchor_areas) if np.isfinite(area) and area < bin_cut]

    small_caption = (f"claim_5 · {len(small_anchor_mask)} clips, smallest {SMALL_ANCHOR_QUANTILE:.0%} of anchors " f"(area < {bin_cut:.0f} px²) · the commit gate is the only difference")
    heatmap(
        arm_names,
        baseline,
        small_anchor_mask,
        occluded_frames,
        "Coverage + memory hygiene by number of occlusion frames",
        small_caption,
        "number of occlusion frames",
        target / "fig_by_occlusion_small",
        "{:.0f}")
    report(
        arm_names,
        baseline,
        small_anchor_mask,
        occluded_frames,
        f"by number of occlusion frames  (anchor area < {bin_cut:.0f} px², smallest "
        f"{SMALL_ANCHOR_QUANTILE:.0%} of anchors)")


if __name__ == "__main__":
    main()

"""claim_1 figures: post-occlusion COVERAGE for all three arms (sam, memory oracle, mask oracle), binned by
occlusion length and by how far the target travels from its anchor.

Two figures, one per covariate, both reading the same per-clip score: the fraction of visible post-occlusion
frames held at box IoU >= COVERAGE_IOU. Absolute values, not deltas -- so the baseline's own difficulty is
visible and the oracle gaps are read against it. sam has no commit gate, so it is a single line; each oracle
is drawn at its own best commit threshold, chosen post-hoc on this same data, which makes the gaps an upper
bound rather than an unbiased estimate."""


import sys
import json
import pickle
from pathlib import Path

import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper"))

import style

def anchor_area(video, person_id, anchor):
    """Visible box area on the ANCHOR frame, px² at the 1024 working resolution.

    The size of the target at the moment the memory bank is seeded, and the quantity `min_visible_area`
    gates on -- so this bins clips by exactly what the selection controls. Anchor-frame only, deliberately:
    it describes the reference encoding, not the trajectory. A target can clear the floor on its anchor and
    be a quarter of that size for the remaining two hundred frames, so read it as a property of the seed."""

    if video not in _visible_cache:
        _visible_cache[video] = json.load(open(f"data/person_path/visible/{video}.json"))
    visible = _visible_cache[video]
    scale = 1024 / max(float(visible["metadata"]["resolution"]["width"]),
                       float(visible["metadata"]["resolution"]["height"]))
    for entity in visible["entities"]:
        if entity["id"] == person_id and int(entity["blob"]["frame_idx"]) == int(anchor):
            box = entity["bb"]
            return float(box[2]) * float(box[3]) * scale ** 2 if float(box[2]) > 0 else np.nan
    return np.nan


COVERAGE_IOU = 0.5                           # a visible frame counts as held at this box IoU
FAILURE_IOU = 0.1                            # below this the target is considered lost (VOT's threshold)
N_BINS = 4
SMALLEST_N = 400                             # clips kept in the fixed-count small-anchor figure
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MEMORY, MASK, SAM = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3, fixed order, entity-stable

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "data/claim_1/results.pkl"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""


METRIC = sys.argv[4] if len(sys.argv) > 4 else "coverage"     
CONFIG = sys.argv[3] if len(sys.argv) > 3 else "conf/experiments/claim_1.yaml"
person_path = hydra.utils.instantiate(OmegaConf.load(CONFIG).person_path)
anchor_of = {(v, int(p)): int(a) for v, p, a in zip(
    person_path.selected_video_names.tolist(), person_path.selected_person_ids.tolist(),
    person_path.selected_anchor_video_frames.tolist())}


# ---------------------------------------------------------------------------------------
# per-clip score and covariate
# ---------------------------------------------------------------------------------------

def visible(clip, ious):
    """The clip's per-frame IoUs on the VISIBLE annotated post-occlusion frames.

    Occluded frames are excluded, which is the standard single-object-tracking convention -- and here it is
    also the only defensible one for these figures. A pooled score that credits occluded frames (1 when the
    arm declines to write them to memory) cannot be binned by occlusion length without circularity: the
    x-axis then sets how much of the y-axis comes from the occluded half, and since those frames score
    higher than visible ones for every arm -- 1.0 by construction for the oracles, 0.81 for the baseline --
    longer occlusions inflate the score exactly as tracking gets worse. The two effects cancel and the curve
    goes flat for reasons that have nothing to do with tracking.

    Commit behaviour on occluded frames is still worth reporting; it just belongs in its own number rather
    than mixed into these. The flags are in the pkl (`*_commit`) whenever that is wanted."""

    ious = np.asarray(ious, dtype=float)
    # New-format pkls store the whole post-occlusion span with NaN where there is no GT box; older archives
    # store the visible frames only, already filtered.
    return ious[clip["has_box"] & ~clip["occluded"]] if "occluded" in clip else ious


def coverage(clip, ious, commits=None):
    """Fraction of visible post-occlusion frames held at box IoU >= COVERAGE_IOU. Credits a tracker that
    loses the target and re-finds it, and is indifferent to WHEN in the clip the held frames fall."""

    ious = visible(clip, ious)
    return float((ious >= COVERAGE_IOU).mean()) if len(ious) else np.nan


def coverage_with_occlusion(clip, ious, commits):
    """Coverage over ALL annotated post-occlusion frames, scoring the two kinds of frame by what the arm can
    actually get right on each: a VISIBLE frame counts when box IoU >= COVERAGE_IOU, an OCCLUDED frame counts
    when the arm did NOT write it to memory.

    While the target is hidden there is nothing to track, so the only decision an arm makes is whether to
    commit -- and committing then is exactly how a bank gets poisoned. Scoring it folds memory hygiene into
    the same number as tracking quality, which is the thing the oracles are actually intervening on.

    This must NOT be binned by occlusion length. The occluded half scores higher than the visible half for
    every arm (1.0 by construction for the oracles, ~0.81 for the baseline), so a longer-occlusion bin gets
    more of its score from the easy half and the curve flattens for reasons unrelated to tracking. Against
    covariates that are not occlusion length -- anchor size, target motion -- that circularity does not
    arise, and the metric is strictly more informative than visible-only coverage.

    NaN for archives predating the commit flags."""

    if commits is None or "occluded" not in clip:
        return np.nan

    ious = np.asarray(ious, dtype=float)
    seen, hidden = clip["has_box"] & ~clip["occluded"], np.asarray(clip["occluded"])
    scored = int(seen.sum() + hidden.sum())
    if not scored:
        return np.nan

    held = float((ious[seen] >= COVERAGE_IOU).sum())
    clean = float((~np.asarray(commits, dtype=bool)[hidden]).sum())
    return (held + clean) / scored


def robustness(clip, ious):
    """Fraction of the visible post-occlusion frames that precede the FIRST failure, a failure being the
    first visible frame at box IoU < FAILURE_IOU. 1.0 when the arm never fails.

    The VOT-style companion to coverage: where coverage asks how many frames were held, this asks how long
    the arm survives before the first collapse, and gives no credit for anything after it. The two come
    apart exactly where the interesting behaviour is -- an arm that drops the target and recovers scores
    well on coverage and near zero here, so reading them together separates "never lost it" from "lost it
    and got it back", which is the distinction the memory bank is supposed to make.

    Single-pass, so there is no VOT reset protocol: this is the fraction of the sequence tracked before the
    first failure, not a failure count."""

    ious = visible(clip, ious)
    if not len(ious):
        return np.nan
    failed = ious < FAILURE_IOU
    return float(int(np.argmax(failed)) / len(ious)) if failed.any() else 1.0


_visible_cache = {}


def mean_displacement(video, person_id, anchor, n_frames):
    """Mean distance from the ANCHOR box centre to the box centre on every visible frame of the clip, px at
    the 1024 working resolution.

    A motion covariate that needs no label: a target that stays put scores near zero however it is annotated,
    and one that walks across the scene scores large. Being continuous, it quantile-bins like the occlusion
    count rather than forcing a two-group split on a rare flag such as `sitting_person`.

    Displacement from the anchor, not path length: a target that leaves and returns reads as low motion,
    which is the right reading for a memory bank seeded at the anchor."""

    if video not in _visible_cache:
        _visible_cache[video] = json.load(open(f"data/person_path/visible/{video}.json"))
    visible = _visible_cache[video]
    scale = 1024 / max(float(visible["metadata"]["resolution"]["width"]),
                       float(visible["metadata"]["resolution"]["height"]))

    boxes = {e["blob"]["frame_idx"]: e["bb"] for e in visible["entities"] if e["id"] == person_id}
    centre = lambda b: (float(b[0]) + float(b[2]) / 2.0, float(b[1]) + float(b[3]) / 2.0)
    frames = sorted(f for f in boxes if f >= anchor and float(boxes[f][2]) > 0)[:n_frames]
    if not frames:
        return np.nan

    anchor_x, anchor_y = centre(boxes[frames[0]])
    offsets = [np.hypot(centre(boxes[f])[0] - anchor_x, centre(boxes[f])[1] - anchor_y) for f in frames]
    return float(np.mean(offsets) * scale)


# ---------------------------------------------------------------------------------------
# collect every clip into the covariate and score arrays the two figures share
# ---------------------------------------------------------------------------------------

score, YLABEL = {
    "coverage": (coverage, f"coverage @ {COVERAGE_IOU:g}"),
    "robustness": (robustness, f"robustness @ {FAILURE_IOU:g}"),
}[METRIC]

# The occlusion figure keeps the visible-only metric (see `coverage_with_occlusion` for why it must); the
# covariates that are not occlusion length get the occlusion-aware one.
HYGIENE_YLABEL = f"coverage @ {COVERAGE_IOU:g}"

results = pickle.load(open(RESULTS, "rb"))
thresholds = list(results["thresholds"])


def arm_scores(clip, metric):
    """(sam, [memory per threshold], [mask per threshold]) under one metric, with the matching commit flags."""

    return (metric(clip, clip["sam"], clip.get("sam_commit")),
            [metric(clip, clip["memory"][t], clip.get("memory_commit", {}).get(t)) for t in thresholds],
            [metric(clip, clip["mask"][t], clip.get("mask_commit", {}).get(t)) for t in thresholds])


occlusions, motion, areas, scored, scored_hygiene = [], [], [], [], []
for clip in results["clips"]:
    anchor = anchor_of.get((clip["video"], int(clip["person"])))
    if not len(clip["sam"]) or anchor is None:
        continue
    displacement = mean_displacement(clip["video"], int(clip["person"]), anchor, int(clip["n_frames"]))
    area = anchor_area(clip["video"], int(clip["person"]), anchor)
    if not (np.isfinite(displacement) and np.isfinite(area)):
        continue

    occlusions.append(int(clip["occ_count"]))
    motion.append(displacement)                  # mean centre displacement from the anchor, px @1024
    areas.append(area)                           # anchor visible box area, px² @1024
    scored.append(arm_scores(clip, score))
    scored_hygiene.append(arm_scores(clip, coverage_with_occlusion))

occlusions = np.array(occlusions)
motion = np.array(motion)
areas = np.array(areas)


def arm_arrays(rows):
    """Stack the per-clip tuples into (sam (n,), memory (n, thresholds), mask (n, thresholds))."""
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]))


ARMS = arm_arrays(scored)                        # visible-only coverage
HYGIENE = arm_arrays(scored_hygiene)             # + occluded frames scored on commit behaviour
sam = ARMS[0]


# ---------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------

def draw(values, xlabel, caption, filename, arms=None, ylabel=None, tick="{:.0f}", keep=None):
    r"""One paper figure: the three arms' score across quantile bins of `values`, over every clip.

    `arms` selects which metric's score arrays to plot -- ARMS (visible only) or HYGIENE. No title is drawn:
    `caption` is printed for pasting into the LaTeX \caption, where it is numbered and set in the paper's
    own type rather than duplicated inside the axes at the wrong size."""

    baseline_scores, memory, mask = ARMS if arms is None else arms
    if keep is not None:
        values = values[keep]
        baseline_scores, memory, mask = baseline_scores[keep], memory[keep], mask[keep]
    if len(baseline_scores) < 2:                 # early in a run there is nothing to bin yet
        print(f"skipped {filename}  (n={len(baseline_scores)})")
        return

    # Each oracle at the threshold maximising its own pooled score UNDER THIS METRIC -- picked on this data,
    # so the gap it shows is an upper bound rather than unbiased.
    memory_best = int(np.nanmean(memory, axis=0).argmax())
    mask_best = int(np.nanmean(mask, axis=0).argmax())

    edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)))
    index = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    x = np.arange(len(edges) - 1)

    figure, axis = plt.subplots(figsize=(style.COLUMN, 3.1))

    # Slot order is fixed and entity-stable: sam always slot 0, memory 1, mask 2, in every claim_1 figure.
    series = [(np.array([baseline_scores[index == k].mean() for k in x]), "SAM 2 baseline")]
    for values_by_threshold, label, featured in ((memory, "memory oracle", memory_best),
                                                 (mask, "mask oracle", mask_best)):
        per_bin = np.array([values_by_threshold[index == k, featured].mean() for k in x])
        series.append((per_bin, rf"{label} ($\tau$={thresholds[featured]:g})"))

    for position, (per_bin, label) in enumerate(series):
        axis.plot(x, per_bin, label=label, zorder=3, **style.series_style(position))

    axis.set_xticks(x)
    axis.set_xticklabels([f"{tick.format(edges[k])}-{tick.format(edges[k + 1])}" for k in x])
    axis.set_xlim(-0.18, len(x) - 1 + 0.18)
    style.headroom(axis, max(per_bin.max() for per_bin, _ in series))
    style.style_axes(axis, xlabel, YLABEL if ylabel is None else ylabel)
    axis.legend(loc="best", ncol=1)

    style.save(figure, filename)
    counts = ", ".join(str(int((index == k).sum())) for k in x)
    print(f"   caption: {caption} n={len(baseline_scores)} clips ({counts} per bin); each oracle "
          f"at its own best commit threshold, chosen post-hoc on this data.")


style.use_paper_style()
FIGURES = Path("data/claim_1/paper")

# Binned by occlusion length, the hygiene metric is circular and the curve has to be read with that in mind:
# the occluded half scores higher than the visible half for every arm (1.0 by construction for the oracles,
# ~0.81 for the baseline), so a longer-occlusion bin draws more of its score from the easier half and
# flattens for reasons unrelated to tracking. `fig_occlusion_visible` is the same binning on visible frames
# only, where that effect cannot arise -- read the pair together.
draw(occlusions, "number of occluded frames",
     "Coverage and memory hygiene by occlusion length.",
     FIGURES / f"fig_occlusion{SUFFIX}", arms=HYGIENE, ylabel=HYGIENE_YLABEL)

draw(occlusions, "number of occluded frames",
     f"{METRIC.capitalize()} by occlusion length, visible frames only.",
     FIGURES / f"fig_occlusion_visible{SUFFIX}")

# The same occlusion binning with the LARGEST anchors dropped. Every arm converges with the baseline on the
# top quartile -- a big target is easy and the memory decision has little left to change -- so those clips
# only dilute the occlusion trend the figure is about.
small = areas < np.nanquantile(areas, 0.75)
draw(occlusions, "number of occluded frames",
     "Coverage and memory hygiene by occlusion length, largest-anchor quartile dropped.",
     FIGURES / f"fig_occlusion_small{SUFFIX}", arms=HYGIENE, ylabel=HYGIENE_YLABEL, keep=small)


smallest = np.zeros(len(areas), dtype=bool)
smallest[np.argsort(np.where(np.isfinite(areas), areas, np.inf))[:SMALLEST_N]] = True
smallest &= np.isfinite(areas)
draw(occlusions, "number of occluded frames",
     f"Coverage and memory hygiene by occlusion length, the {SMALLEST_N} smallest anchors.",
     FIGURES / f"fig_occlusion_smallest{SUFFIX}", arms=HYGIENE, ylabel=HYGIENE_YLABEL, keep=smallest)

# How big the target is where the memory bank is seeded -- the quantity `min_visible_area` gates on.
draw(areas, "anchor visible area",
     "Coverage and memory hygiene by anchor size.",
     FIGURES / f"fig_area{SUFFIX}", arms=HYGIENE, ylabel=HYGIENE_YLABEL)


def pooled(label, arms):
    baseline, memory, mask = arms
    best_memory, best_mask = int(np.nanmean(memory, 0).argmax()), int(np.nanmean(mask, 0).argmax())
    print(f"   pooled {label:<22} sam {np.nanmean(baseline):.4f}   "
          f"memory {np.nanmean(memory[:, best_memory]):.4f} (thr {thresholds[best_memory]:g})   "
          f"mask {np.nanmean(mask[:, best_mask]):.4f} (thr {thresholds[best_mask]:g})")


print(f"\nn={len(sam)} clips total")
pooled(METRIC, ARMS)
pooled("coverage + hygiene", HYGIENE)

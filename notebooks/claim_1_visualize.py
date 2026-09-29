"""claim_1 figures: post-occlusion COVERAGE for all three arms (sam, memory oracle, mask oracle), binned by
occlusion length and by how far the target travels from its anchor."""

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


def anchor_area(video_name, person_id, anchor_frame):
    """Visible box area on the ANCHOR frame, px² at the 1024 working resolution."""

    if video_name not in _visible_cache:
        _visible_cache[video_name] = json.load(open(f"data/person_path/visible/{video_name}.json"))
    visible = _visible_cache[video_name]
    resolution_scale = 1024 / max(float(visible["metadata"]["resolution"]["width"]), float(visible["metadata"]["resolution"]["height"]))
    for annotation_entity in visible["entities"]:
        if annotation_entity["id"] == person_id and int(annotation_entity["blob"]["frame_idx"]) == int(anchor_frame):
            box = annotation_entity["bb"]
            return float(box[2]) * float(box[3]) * resolution_scale**2 if float(box[2]) > 0 else np.nan
    return np.nan


COVERAGE_IOU = 0.5
FAILURE_IOU = 0.1  # below this the target is considered lost (our choice; VOT declares failure at ZERO overlap)
N_BINS = 4
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
MEMORY, MASK, SAM = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, fixed order, entity-stable

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "data/claim_1/results.pkl"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""


METRIC = sys.argv[4] if len(sys.argv) > 4 else "coverage"
CONFIG = sys.argv[3] if len(sys.argv) > 3 else "conf/experiments/claim_1.yaml"
person_path = hydra.utils.instantiate(OmegaConf.load(CONFIG).person_path)
anchor_of = {(v, int(p)): int(a) for v, p, a in zip(person_path.selected_video_names.tolist(), person_path.selected_person_ids.tolist(), person_path.selected_anchor_video_frames.tolist())}


def visible(clip_record, box_ious):
    """The clip's per-frame IoUs on the VISIBLE annotated post-occlusion frames."""

    box_ious = np.asarray(box_ious, dtype=float)
    return box_ious[clip_record["has_box"] & ~clip_record["occluded"]] if "occluded" in clip_record else box_ious


def coverage(clip_record, box_ious, commits=None):
    """Fraction of visible post-occlusion frames held at box IoU >= COVERAGE_IOU. Credits a tracker that
    loses the target and re-finds it, and is indifferent to WHEN in the clip the held frames fall."""

    box_ious = visible(clip_record, box_ious)
    return float((box_ious >= COVERAGE_IOU).mean()) if len(box_ious) else np.nan


def coverage_with_occlusion(clip_record, box_ious, commits):
    """Coverage over ALL annotated post-occlusion frames, scoring the two kinds of frame by what the arm can
    actually get right on each: a VISIBLE frame counts when box IoU >= COVERAGE_IOU, an OCCLUDED frame counts"""

    if commits is None or "occluded" not in clip_record:
        return np.nan

    box_ious = np.asarray(box_ious, dtype=float)
    seen, hidden = clip_record["has_box"] & ~clip_record["occluded"], np.asarray(clip_record["occluded"])
    scored_rows = int(seen.sum() + hidden.sum())
    if not scored_rows:
        return np.nan

    held_frames = float((box_ious[seen] >= COVERAGE_IOU).sum())
    clean_refusals = float((~np.asarray(commits, dtype=bool)[hidden]).sum())
    return (held_frames + clean_refusals) / scored_rows


def robustness(clip_record, box_ious):
    """Fraction of the visible post-occlusion frames that precede the FIRST failure, a failure being the first
    visible frame at box IoU < FAILURE_IOU. 1.0 when the arm never fails."""

    box_ious = visible(clip_record, box_ious)
    if not len(box_ious):
        return np.nan
    failure_mask = box_ious < FAILURE_IOU
    return float(int(np.argmax(failure_mask)) / len(box_ious)) if failure_mask.any() else 1.0


_visible_cache = {}


def mean_displacement(video_name, person_id, anchor_frame, n_frames):
    """Mean distance from the ANCHOR box centre to the box centre on every visible frame of the clip, px at the
    1024 working resolution."""

    if video_name not in _visible_cache:
        _visible_cache[video_name] = json.load(open(f"data/person_path/visible/{video_name}.json"))
    visible = _visible_cache[video_name]
    resolution_scale = 1024 / max(float(visible["metadata"]["resolution"]["width"]), float(visible["metadata"]["resolution"]["height"]))

    entity_boxes = {e["blob"]["frame_idx"]: e["bb"] for e in visible["entities"] if e["id"] == person_id}
    box_centre = lambda b: (float(b[0]) + float(b[2]) / 2.0, float(b[1]) + float(b[3]) / 2.0)
    frame_indices = sorted(frame_index for frame_index in entity_boxes if frame_index >= anchor_frame and float(entity_boxes[frame_index][2]) > 0)[:n_frames]
    if not frame_indices:
        return np.nan

    anchor_x, anchor_y = box_centre(entity_boxes[frame_indices[0]])
    offsets = [np.hypot(box_centre(entity_boxes[frame_index])[0] - anchor_x, box_centre(entity_boxes[frame_index])[1] - anchor_y) for frame_index in frame_indices]
    return float(np.mean(offsets) * resolution_scale)


score_function, YLABEL = {"coverage": (coverage, f"coverage @ {COVERAGE_IOU:g}"), "robustness": (robustness, f"robustness @ {FAILURE_IOU:g}")}[METRIC]
HYGIENE_YLABEL = f"coverage @ {COVERAGE_IOU:g}"

results = pickle.load(open(RESULTS, "rb"))
thresholds = list(results["thresholds"])


def arm_scores(clip_record, metric_name):
    """(sam, [memory per threshold], [mask per threshold]) under one metric, with the matching commit flags."""

    return (
        metric_name(clip_record, clip_record["sam"], clip_record.get("sam_commit")),
        [
            metric_name(
                clip_record,
                clip_record["memory"][commit_threshold],
                clip_record.get("memory_commit", {}).get(commit_threshold))
            for commit_threshold in thresholds
        ],
        [
            metric_name(
                clip_record,
                clip_record["mask"][commit_threshold],
                clip_record.get("mask_commit", {}).get(commit_threshold))
            for commit_threshold in thresholds
        ])


occlusions, displacements, anchor_areas, scored_rows, scored_hygiene = [], [], [], [], []
for clip_record in results["clips"]:
    anchor_frame = anchor_of.get((clip_record["video"], int(clip_record["person"])))
    if not len(clip_record["sam"]) or anchor_frame is None:
        continue
    displacement = mean_displacement(clip_record["video"], int(clip_record["person"]), anchor_frame, int(clip_record["n_frames"]))
    area = anchor_area(clip_record["video"], int(clip_record["person"]), anchor_frame)
    if not (np.isfinite(displacement) and np.isfinite(area)):
        continue

    occlusions.append(int(clip_record["occ_count"]))
    displacements.append(displacement)
    anchor_areas.append(area)
    scored_rows.append(arm_scores(clip_record, score_function))
    scored_hygiene.append(arm_scores(clip_record, coverage_with_occlusion))

occlusions = np.array(occlusions)
displacements = np.array(displacements)
anchor_areas = np.array(anchor_areas)


def arm_arrays(rows):
    """Stack the per-clip tuples into (sam (n), memory (n, thresholds), mask (n, thresholds))."""
    return (
        np.array([score_row[0] for score_row in rows]),
        np.array([score_row[1] for score_row in rows]),
        np.array([score_row[2] for score_row in rows]))


ARMS = arm_arrays(scored_rows)
HYGIENE = arm_arrays(scored_hygiene)
baseline_scores = ARMS[0]


def draw(covariate_values, xlabel, caption, filename, arm_arrays_selected=None, ylabel=None, tick="{:.0f}", keep=None):
    """One paper figure: the three arms' score across quantile bins of `values`, over every clip."""

    baseline_scores, memory_oracle_scores, mask_oracle_scores = (ARMS if arm_arrays_selected is None else arm_arrays_selected)
    if keep is not None:
        covariate_values = covariate_values[keep]
        baseline_scores, memory_oracle_scores, mask_oracle_scores = (baseline_scores[keep], memory_oracle_scores[keep], mask_oracle_scores[keep])
    if len(baseline_scores) < 2:
        return

    memory_best = int(np.nanmean(memory_oracle_scores, axis=0).argmax())
    mask_best = int(np.nanmean(mask_oracle_scores, axis=0).argmax())

    edges = np.unique(np.quantile(covariate_values, np.linspace(0, 1, N_BINS + 1)))
    index = np.clip(np.digitize(covariate_values, edges[1:-1]), 0, len(edges) - 2)
    x = np.arange(len(edges) - 1)

    figure, axis = plt.subplots(figsize=(style.COLUMN, 3.1))
    series = [(np.array([baseline_scores[index == bin_index].mean() for bin_index in x]), "SAM 2 baseline")]

    for values_by_threshold, series_label, featured in (
        (memory_oracle_scores, "memory oracle", memory_best),
        (mask_oracle_scores, "mask oracle", mask_best)):
        
        per_bin = np.array([values_by_threshold[index == bin_index, featured].mean() for bin_index in x])
        series.append((per_bin, rf"{series_label} ($\tau$={thresholds[featured]:g})"))

    for position, (per_bin, series_label) in enumerate(series):
        axis.plot(x, per_bin, label=series_label, zorder=3, **style.series_style(position))

    axis.set_xticks(x)
    axis.set_xticklabels([f"{tick.format(edges[bin_index])}-{tick.format(edges[bin_index + 1])}" for bin_index in x])
    axis.set_xlim(-0.18, len(x) - 1 + 0.18)
    style.headroom(axis, max(per_bin.max() for per_bin, _ in series))
    style.style_axes(axis, xlabel, YLABEL if ylabel is None else ylabel)
    axis.legend(loc="best", ncol=1)

    style.save(figure, filename)
    counts = ",".join(str(int((index == bin_index).sum())) for bin_index in x)


style.use_paper_style()
FIGURES = Path("data/claim_1/paper")

draw(occlusions, "number of occluded frames", "Coverage and memory hygiene by occlusion length.", FIGURES / f"fig_occlusion{SUFFIX}",arm_arrays_selected=HYGIENE,ylabel=HYGIENE_YLABEL)
draw(anchor_areas, "anchor visible area","Coverage and memory hygiene by anchor size.", FIGURES / f"fig_area{SUFFIX}", arm_arrays_selected=HYGIENE, ylabel=HYGIENE_YLABEL)

def pooled(series_label, arm_arrays_selected):
    baseline, memory_oracle_scores, mask_oracle_scores = arm_arrays_selected
    best_memory, best_mask = int(np.nanmean(memory_oracle_scores, 0).argmax()), int(np.nanmean(mask_oracle_scores, 0).argmax())


pooled(METRIC, ARMS)
pooled("coverage + hygiene", HYGIENE)

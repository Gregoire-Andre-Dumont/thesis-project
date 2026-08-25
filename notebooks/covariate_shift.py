"""Covariate shift of a score-gated memory policy, measured on the tracker's OWN predicted masks.

The POLICY is SAM 2 whose memory bank is gated live on the re-ID score: at each frame it commits the chosen mask
iff  alpha * chamfer(pred, anchor) + (1 - alpha) * mean chamfer(pred, recent) >= tau. A wrong commit corrupts the
memory the next commit is judged against -- the covariate shift.

The candidates are the policy's own per-frame predicted masks (which can drift onto a distractor). Each frame is
labelled by mask overlap: positive = on target (target_iou >= label_positive), negative = distractor grab
(target_iou < label_negative_target and distractor_iou >= label_negative_distractor), the rest dropped. We then
ask: can the score separate on-target frames from distractor grabs? scored two ways -- against the policy's own
(corrupted) memory, and against the oracle's true-IoU (clean) memory. The gap (oracle AUC - policy AUC) is the
covariate shift: at low anchor weight the corrupted memory can no longer flag the very grabs it caused (the
grabbed distractor now matches the memory), while the clean memory still can.

Memory size is fixed (config.memory_size); we sweep the anchor weight. tau is calibrated per cell in phase 1 as
the score cut best matching (max F1) the true-IoU "should commit" label. For each encoder we draw one AUC-vs-
anchor-weight figure: the policy and oracle curves with the covariate-shift gap (oracle AUC - policy AUC) shaded
between. (Each candidate's anchor-candidate distance is still recorded, for optional near/far splits.)
"""

import gc
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root on path for `src` and create_anchor_dataset

from pe_reid_longrange import test_trajectories, box_prompt_masks, chamfer_scores, DEVICE, DTYPE
from create_anchor_dataset import anchor_trajectory_index
from src.offline_training.dataset_encoders import crop_around_masks, anchor_size_pixels, load_dataset_encoders
from src.offline_training.dataset_labels import load_clean_boxes_by_frame, _box_center as box_center
from src.utils.load_bboxes import load_bboxes


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
os.environ["HYDRA_FULL_ERROR"] = "1"

def foreground_chamfer(reference_tokens, candidate_tokens):
    """Unidirectional foreground chamfer of the candidate to a reference (candidate -> reference)."""

    unidirectional, _bidirectional = chamfer_scores(reference_tokens, candidate_tokens)
    return unidirectional


@torch.inference_mode()
def mask_foreground(encoders, frames, masks, floor, crop_size, chunk=8):
    """Foreground tokens of each (frame, mask) for every encoder, cropped around the mask at the anchor scale.
    Returns a list of {encoder: foreground_tokens}; a tensor is empty if the mask vanished after cropping.
    Crops are encoded in chunks so peak GPU memory stays bounded no matter how many masks are passed at once."""

    crops, crop_masks = crop_around_masks(frames, masks.astype(np.float32), crop_size, 0.2, floor)
    crop_masks_tensor = torch.from_numpy(crop_masks).unsqueeze(1)

    foreground_per_item = [{} for _ in range(len(masks))]
    for encoder_name, encode in encoders.items():
        for start in range(0, len(crops), chunk):
            span = slice(start, start + chunk)
            tokens = encode(crops[span])

            grid_size = round(tokens.shape[1] ** 0.5)
            resized_masks = F.interpolate(crop_masks_tensor[span].to(tokens.device).float(), size=(grid_size, grid_size), mode="nearest")
            foreground = (resized_masks > 0.5).flatten(1)

            for item, item_tokens, item_foreground in zip(foreground_per_item[span], tokens, foreground):
                item[encoder_name] = item_tokens[item_foreground].clone()

            del tokens
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    return foreground_per_item


def load_window(detection_data, trajectory, max_frames):
    """Decode only the frames the tracker needs -- the warmup frame plus `max_frames` from the anchor, not the
    whole (possibly 100s-of-frames) trajectory. Reads the annotations first (no decode) to find the window,
    then re-loads just those frames. Returns (warmup, anchor_index), or None if the anchor is missing."""

    video, person, anchor_frame = trajectory
    keep_load_frames = detection_data.load_frames
    detection_data.load_frames = False
    detection_data.initialize_target(video, person)                       # annotations only, no video decode
    anchor_index = anchor_trajectory_index(detection_data, anchor_frame)
    if anchor_index is None:
        detection_data.load_frames = keep_load_frames
        return None

    warmup = 1 if anchor_index >= 1 else 0                                 # mirrors slice_detection_data_for_tracker
    start = anchor_index - warmup
    window = detection_data.frame_indices[start:start + warmup + max_frames]

    detection_data.load_frames = keep_load_frames
    detection_data.initialize_target(video, person, frame_indices=window)  # decode only the window
    return warmup, anchor_index


def cells(config):
    """The (encoder, memory size, anchor weight) cells rolled out: memory is fixed (config.memory_size), so this
    is just the anchor-weight sweep per encoder. Distance is a post-hoc binning axis, not a rollout cell."""

    return [(encoder, config.memory_size, alpha) for encoder in config.encoders for alpha in config.alphas]


def oracle_pass(oracle, detection_data, trajectory, max_frames, frame_cache):
    """Run the true-IoU oracle over the trajectory window (decoding only the frames it needs) and return, sliced
    to the post-warmup frames, its predicted masks, per-frame pseudo-GT mask IoU, the shared frames/boxes/
    occlusions/frame_indices, and the anchor index + warmup. None if the anchor is missing. Populates the shared
    image cache so every later policy-cell rollout over the same window skips the SAM backbone."""

    window = load_window(detection_data, trajectory, max_frames)
    if window is None:
        return None
    warmup, anchor_index = window

    oracle.frame_cache = frame_cache
    predicted_masks = oracle.predict_masks(detection_data).numpy()

    keep = slice(warmup, None)
    return {
        "predicted": predicted_masks[keep],
        "mask_iou": oracle.mask_iou_scores.numpy()[keep],
        "frames": detection_data.frames[keep],
        "boxes": detection_data.bboxes_norm[keep],
        "occlusions": detection_data.occlusions[keep],
        "frame_indices": detection_data.frame_indices[keep],
        "anchor_index": anchor_index,
        "warmup": warmup,
    }


def best_f1_threshold(scores, labels):
    """Threshold on `scores` (commit iff score >= threshold) maximizing F1 against boolean `labels`, in one
    sorted pass; NaN if the labels are single-class."""

    positive_count = int(labels.sum())
    if positive_count == 0 or positive_count == len(labels):
        return np.nan

    descending = np.argsort(-scores)
    scores_descending = scores[descending]
    labels_descending = labels[descending].astype(int)

    true_positives = np.cumsum(labels_descending)                # among the top-i highest scores
    predicted_positives = np.arange(1, len(scores) + 1)
    f1_scores = 2 * true_positives / (predicted_positives + positive_count)

    best_cut = int(np.argmax(f1_scores))
    return float(scores_descending[best_cut])


def anchor_reference(sam, encoders, frames, boxes, floor, crop_size):
    """Foreground tokens (per encoder) of the anchor: the frame-0 target, box-prompted then cropped at its scale."""

    anchor_mask = box_prompt_masks(sam, frames[0], [boxes[0]])[0]
    return mask_foreground(encoders, frames[0][None], anchor_mask[None], floor, crop_size)[0]


def crop_floor(config, trajectory, anchor_index, frames, boxes):
    """The pixel crop floor from the anchor's amodal box (its visible box when the amodal one is missing), so
    every crop is taken at the target's true scale regardless of how occluded a later frame is."""

    video, person, _ = trajectory
    visible_json = Path(config.detection_data.visible_directory) / f"{video}.json"
    amodal_json = Path(config.detection_data.amodal_directory) / f"{video}.json"
    anchor_amodal = load_bboxes(str(amodal_json), str(visible_json), person, use_amodal=True)[anchor_index]
    anchor_box = anchor_amodal if float(anchor_amodal[2]) > 0 else boxes[0]
    return anchor_size_pixels(anchor_box, frames[0].shape)


def true_iou_commits(mask_iou, occlusions, label_iou_threshold):
    """The ground-truth 'should commit' label per frame: the pseudo-GT mask IoU clears the threshold and the
    target is visible. This is the oracle's gate and the target phase-1 calibrates the score threshold against."""

    return (mask_iou > label_iou_threshold) & (occlusions < 0.5)


def committed_reference(encoders, positions, frames, masks, floor, crop_size):
    """Foreground tokens (per encoder) of the committed frames' masks, i.e. the memory bank's contents."""

    if len(positions) == 0:
        return positions, []
    return positions, mask_foreground(encoders, frames[positions], masks[positions], floor, crop_size)


def calibration_records(sam, encoders, oracle_data, floor, config):
    """Phase 1: one record per oracle frame -- its predicted-mask foreground scored (anchor chamfer + chamfers to
    the oracle's true-IoU memory strictly before it) and labelled by whether the frame truly should have been
    committed. Pooled across trajectories these calibrate each cell's commit threshold; mixed_score reads the
    records under a single 'oracle' memory."""

    frames, masks = oracle_data["frames"], oracle_data["predicted"]
    should_commit = true_iou_commits(oracle_data["mask_iou"], oracle_data["occlusions"], config.label_iou_threshold)
    committed_positions = np.where(should_commit)[0]

    anchor_tokens = anchor_reference(sam, encoders, frames, oracle_data["boxes"], floor, config.crop_size)
    frame_tokens = mask_foreground(encoders, frames, masks, floor, config.crop_size)
    max_memory = config.memory_size

    records = []
    for frame in range(len(frames)):
        prior_positions = committed_positions[committed_positions < frame][-max_memory:]

        anchor_similarity = {}
        recent_similarity = {}
        for encoder in encoders:
            anchor_similarity[encoder] = foreground_chamfer(anchor_tokens[encoder], frame_tokens[frame][encoder])
            similarities = []
            for position in prior_positions:
                similarities.append(foreground_chamfer(frame_tokens[position][encoder], frame_tokens[frame][encoder]))
            recent_similarity[encoder] = np.array(similarities, np.float32)

        records.append({"label": bool(should_commit[frame]), "anchor": anchor_similarity,
                        "recent": {"oracle": recent_similarity}})
    return records


def commit_thresholds_from_records(records, config):
    """Phase 1: per (encoder, memory, alpha) cell, the score cut maximizing F1 against the should-commit label,
    over all trajectories' calibration records pooled. NaN (single-class) means 'commit everything' downstream."""

    labels = np.array([record["label"] for record in records])

    thresholds = {}
    for encoder, memory_size, alpha in cells(config):
        scores = np.array([mixed_score(record, encoder, "oracle", alpha, memory_size) for record in records])
        valid = ~np.isnan(scores)
        thresholds[(encoder, memory_size, alpha)] = best_f1_threshold(scores[valid], labels[valid])
    return thresholds


def anchor_distance(candidate_box, anchor_box, frame_shape):
    """Pixel distance between the candidate's box centre and the anchor's box centre (the anchor-candidate
    distance the AUC is binned over -- how far this candidate sits from the frame-0 target)."""

    height, width = frame_shape[0], frame_shape[1]
    candidate_x, candidate_y = box_center(candidate_box)
    anchor_x, anchor_y = box_center(anchor_box)
    return float(np.hypot((candidate_x - anchor_x) * width, (candidate_y - anchor_y) * height))


def mask_to_bbox(mask):
    """Normalized xywh bounding box of a predicted mask (all-zeros for an empty mask)."""

    rows, columns = np.where(mask > 0)
    if len(columns) == 0:
        return np.zeros(4, np.float32)
    size = float(mask.shape[0])
    x0, x1, y0, y1 = columns.min(), columns.max(), rows.min(), rows.max()
    return np.array([x0 / size, y0 / size, (x1 - x0 + 1) / size, (y1 - y0 + 1) / size], np.float32)


def box_iou(box_a, box_b):
    """IoU of two normalized xywh boxes."""

    ax0, ay0, aw, ah = box_a
    bx0, by0, bw, bh = box_b
    inter_x = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    inter_y = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    intersection = inter_x * inter_y
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def iou_label(target_iou, distractor_iou, config):
    """Class of a predicted mask by its overlaps: 1 = on target, 0 = grabbed a distractor, -1 = ambiguous (drop)."""

    if target_iou >= config.label_positive:
        return 1
    if target_iou < config.label_negative_target and distractor_iou >= config.label_negative_distractor:
        return 0
    return -1


def recent_chamfer(predicted, positions, tokens, frame, memory_size):
    """Mean chamfer of a predicted foreground to the most-recent committed entries strictly before `frame`; NaN
    while the memory is still empty."""

    recent = [token for position, token in zip(positions, tokens) if position < frame][-memory_size:]
    if not recent:
        return np.nan
    return float(np.mean([foreground_chamfer(entry, predicted) for entry in recent]))


def mixed(anchor_similarity, recent_similarity, alpha):
    """alpha-weighted anchor/recent score, falling back to the anchor while the memory is still empty."""

    memory_similarity = anchor_similarity if np.isnan(recent_similarity) else recent_similarity
    return alpha * anchor_similarity + (1 - alpha) * memory_similarity


def scored_frames(frames, boxes, occlusions, frame_indices, clean_boxes, stride):
    """Yield (frame, distractor_boxes) for each visible, boxed frame that has at least one clean distractor;
    distractor_boxes is every clean person in the frame, so a prediction is checked against all of them."""

    for frame in range(1, len(frames), stride):
        if occlusions[frame] > 0.5 or float(boxes[frame][2]) <= 0:
            continue
        distractor_boxes = clean_boxes.get(int(frame_indices[frame]), [])   # every clean person in the frame
        if distractor_boxes:
            yield frame, distractor_boxes


def release_cache(frame_cache, trackers):
    """Free the per-trajectory image-embedding cache and drop the trackers' references to it."""

    frame_cache.clear()
    for tracker in trackers.values():
        tracker.frame_cache = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def policy_rollout(policy, detection_data, warmup, encode, anchor_tokens, floor, alpha, memory_size, threshold, config):
    """Run one score-gated rollout and return, in post-warmup (sliced) indexing: the predicted masks, the per-frame
    predicted foreground (every frame -- these are the candidates), and the score-gated memory (committed positions
    + CPU tokens)."""

    policy.encode = encode
    policy.anchor_weight = alpha
    policy.memory_size = memory_size
    policy.chamfer_threshold = threshold
    policy.crop_size = config.crop_size
    policy.prepare(anchor_tokens, floor)
    predicted_masks = policy.predict_masks(detection_data).numpy()[warmup:]

    foreground = {position - warmup: token for position, token in policy.foreground_log if position >= warmup}
    committed_positions = [position - warmup for position, _ in policy.committed_log if position >= warmup]
    committed_tokens = [token for position, token in policy.committed_log if position >= warmup]
    return predicted_masks, foreground, committed_positions, committed_tokens


def score_predictions(scored, foreground, predicted_masks, boxes, anchor_box, frame_shape, anchor_cpu,
                      policy_positions, policy_tokens, oracle_positions, oracle_tokens, alpha, memory_size):
    """Score every scored frame's predicted mask: its (target_iou, distractor_iou) label inputs, its policy- and
    oracle-memory scores, and its anchor-candidate distance. Returns five parallel arrays."""

    target_ious, distractor_ious, policy_values, oracle_values, distances = [], [], [], [], []
    for frame, distractor_boxes in scored:
        predicted = foreground.get(frame)
        if predicted is None:
            continue
        predicted_bbox = mask_to_bbox(predicted_masks[frame])

        anchor_similarity = foreground_chamfer(anchor_cpu, predicted)
        policy_recent = recent_chamfer(predicted, policy_positions, policy_tokens, frame, memory_size)
        oracle_recent = recent_chamfer(predicted, oracle_positions, oracle_tokens, frame, memory_size)

        target_ious.append(box_iou(predicted_bbox, boxes[frame]))
        distractor_ious.append(max(box_iou(predicted_bbox, distractor) for distractor in distractor_boxes))
        policy_values.append(mixed(anchor_similarity, policy_recent, alpha))
        oracle_values.append(mixed(anchor_similarity, oracle_recent, alpha))
        distances.append(anchor_distance(predicted_bbox, anchor_box, frame_shape))

    return (np.array(target_ious, np.float32), np.array(distractor_ious, np.float32), np.array(policy_values, np.float32),
            np.array(oracle_values, np.float32), np.array(distances, np.float32))


def collect_trajectory(oracle, policy, sam, encoders, detection_data, trajectory, config, commit_thresholds):
    """Phase 2: one oracle rollout (true-IoU memory) plus one score-gated rollout per cell, sharing the image
    cache. The candidates are the POLICY's own per-frame predicted masks; each is scored against the policy's
    (corrupted) memory and against the oracle's (clean) memory. Returns per-cell (target_iou, distractor_iou,
    policy score, oracle score, distance) arrays, or None if the trajectory is unusable."""

    frame_cache = {} if config.get("share_image_cache", True) else None
    oracle_data = oracle_pass(oracle, detection_data, trajectory, config.max_frames, frame_cache)
    if oracle_data is None or len(oracle_data["frames"]) < 2 or float(oracle_data["boxes"][0][2]) <= 0:
        if frame_cache is not None:
            release_cache(frame_cache, {"oracle": oracle, "policy": policy})
        return None

    frames, boxes = oracle_data["frames"], oracle_data["boxes"]
    occlusions, frame_indices = oracle_data["occlusions"], oracle_data["frame_indices"]
    anchor_box = boxes[0]
    floor = crop_floor(config, trajectory, oracle_data["anchor_index"], frames, boxes)
    anchor_tokens = anchor_reference(sam, encoders, frames, boxes, floor, config.crop_size)

    # Oracle clean memory: the true-IoU committed frames' foreground, moved to CPU for the post-hoc scoring.
    oracle_commits = np.where(true_iou_commits(oracle_data["mask_iou"], occlusions, config.label_iou_threshold))[0]
    oracle_positions, oracle_entry_tokens = committed_reference(encoders, oracle_commits, frames, oracle_data["predicted"], floor, config.crop_size)

    video, person, _ = trajectory
    clean_boxes = load_clean_boxes_by_frame(Path(config.detection_data.visible_directory) / f"{video}.json", person)
    scored = list(scored_frames(frames, boxes, occlusions, frame_indices, clean_boxes, config.stride))

    warmup = oracle_data["warmup"]
    policy.frame_cache = frame_cache                         # every cell's rollout reuses this window's SAM features
    per_cell = {}
    for encoder, memory_size, alpha in cells(config):
        anchor_cpu = anchor_tokens[encoder].detach().float().cpu()
        oracle_tokens = [entry[encoder].detach().float().cpu() for entry in oracle_entry_tokens]

        threshold = commit_thresholds[(encoder, memory_size, alpha)]
        threshold = -np.inf if np.isnan(threshold) else threshold
        predicted_masks, foreground, policy_positions, policy_tokens = policy_rollout(
            policy, detection_data, warmup, encoders[encoder], anchor_tokens[encoder], floor, alpha, memory_size, threshold, config)

        per_cell[(encoder, memory_size, alpha)] = score_predictions(
            scored, foreground, predicted_masks, boxes, anchor_box, frames[0].shape, anchor_cpu,
            policy_positions, policy_tokens, oracle_positions, oracle_tokens, alpha, memory_size)

    if frame_cache is not None:
        release_cache(frame_cache, {"oracle": oracle, "policy": policy})
    return per_cell


def target_auc(labels, scores):
    """Target-vs-distractor ROC-AUC over (label, score) pairs; NaN when a class is absent or there are none."""

    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    return roc_auc_score(labels, scores)


def mixed_score(candidate, encoder, policy, alpha, memory_size):
    """The candidate's alpha-weighted score: its anchor chamfer blended with the mean chamfer to the most
    recent committed memory entries, falling back to the anchor while the memory is still empty."""

    anchor_similarity = candidate["anchor"][encoder]

    recent_similarities = candidate["recent"][policy][encoder][-memory_size:]
    recent_similarities = recent_similarities[~np.isnan(recent_similarities)]
    memory_similarity = recent_similarities.mean() if len(recent_similarities) else anchor_similarity   # anchor only until memory fills

    return alpha * anchor_similarity + (1 - alpha) * memory_similarity


def auc_by_alpha(cells_data, encoder, alphas, memory_size, config):
    """Per anchor weight, the AUC with which the score separates on-target predicted masks from distractor grabs
    (pooled over all distances), under the policy's own (corrupted) memory and under the oracle's clean memory."""

    policy_aucs = []
    oracle_aucs = []
    for alpha in alphas:
        arrays = cells_data.get((encoder, memory_size, alpha))
        if arrays is None:
            policy_aucs.append(np.nan)
            oracle_aucs.append(np.nan)
            continue
        target_iou, distractor_iou, policy_values, oracle_values, _distances = arrays
        labels = np.array([iou_label(target, distractor, config) for target, distractor in zip(target_iou, distractor_iou)])
        keep = labels >= 0
        policy_aucs.append(target_auc(labels[keep], policy_values[keep]))
        oracle_aucs.append(target_auc(labels[keep], oracle_values[keep]))
    return policy_aucs, oracle_aucs


def draw_curves(cells_data, config, n_traj):
    """For each encoder, one AUC-vs-anchor-weight figure: how well the score separates on-target predicted masks
    from distractor grabs, under the policy's own memory and the oracle's clean memory, with the covariate-shift
    gap shaded between. Returns the curves for caching."""

    curves = {}
    for encoder in config.encoders:
        policy_aucs, oracle_aucs = auc_by_alpha(cells_data, encoder, config.alphas, config.memory_size, config)

        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(list(config.alphas), oracle_aucs, marker="o", color="tab:green", label="oracle (clean memory)")
        axis.plot(list(config.alphas), policy_aucs, marker="o", color="tab:red", label="policy (score-gated memory)")
        axis.fill_between(list(config.alphas), policy_aucs, oracle_aucs, color="tab:red", alpha=0.12, label="covariate shift")

        axis.set_xlabel("anchor weight α")
        axis.set_ylabel("on-target vs distractor-grab AUC")
        axis.set_ylim(0.5, 1.0)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        axis.set_title(f"{encoder}  |  {n_traj} trajectories  |  memory {config.memory_size}", fontsize=10)
        figure.tight_layout()
        figure.savefig(f"{config.out_dir}/{encoder}.png", dpi=120)
        plt.close(figure)
        curves[encoder] = {"policy": policy_aucs, "oracle": oracle_aucs}
    return curves


def load_pickle(path, default):
    """The pickled object at `path`, or `default` when the file does not exist yet."""

    return pickle.loads(Path(path).read_bytes()) if Path(path).exists() else default


def save_pickle(path, obj):
    """Atomically write `obj` to `path` so an interrupted run resumes from the last checkpoint, not a torn file."""

    tmp = Path(path).with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(path)


def merge_cells(pooled, trajectory_cells):
    """Concatenate one trajectory's per-cell arrays (target_iou, distractor_iou, policy, oracle, distance) onto the
    pooled totals, in place."""

    for cell, arrays in trajectory_cells.items():
        if cell in pooled:
            pooled[cell] = tuple(np.concatenate([previous, new]) for previous, new in zip(pooled[cell], arrays))
        else:
            pooled[cell] = arrays


def load_trackers(config):
    """Instantiate the true-IoU oracle and the score-gated policy trackers, sharing one SAM 2 backbone."""

    oracle = hydra.utils.instantiate(OmegaConf.load(config.oracle_config).tracker)
    policy = hydra.utils.instantiate(OmegaConf.load(config.baseline_config).tracker)
    policy.model = oracle.model                              # one backbone on the GPU; the memories stay separate
    return {"oracle": oracle, "policy": policy}


def calibrate(oracle, sam, encoders, detection_data, trajectories, config, records_path):
    """Phase 1: pool per-frame calibration records over every trajectory (resuming from the pkl) and return each
    cell's commit threshold -- the score cut whose commits best match the oracle's true-IoU gate."""

    records = load_pickle(records_path, {})
    print(f"phase 1: calibration over {len(trajectories)} trajectories; resuming with {len(records)} cached")

    for completed, trajectory in enumerate(tqdm(trajectories, desc="phase 1: calibrate"), start=1):
        key = f"{trajectory[0]}_{trajectory[1]}_{trajectory[2]}"
        if key in records:
            continue

        frame_cache = {} if config.get("share_image_cache", True) else None
        oracle_data = oracle_pass(oracle, detection_data, trajectory, config.max_frames, frame_cache)
        if frame_cache is not None:
            release_cache(frame_cache, {"oracle": oracle})
        floor = None if oracle_data is None else crop_floor(config, trajectory, oracle_data["anchor_index"], oracle_data["frames"], oracle_data["boxes"])
        records[key] = [] if oracle_data is None else calibration_records(sam, encoders, oracle_data, floor, config)

        if completed % config.checkpoint_every == 0:
            save_pickle(records_path, records)
    save_pickle(records_path, records)

    pooled = [record for trajectory_records in records.values() for record in trajectory_records]
    return commit_thresholds_from_records(pooled, config)


def collect_scores(trackers, encoders, detection_data, trajectories, config, commit_thresholds, scores_path):
    """Phase 2: per trajectory run the oracle plus one score-gated rollout per cell (resuming from the pkl),
    pooling the per-cell prediction arrays and refreshing the heatmaps. Returns (cells_data, finished count)."""

    oracle, policy, sam = trackers["oracle"], trackers["policy"], trackers["oracle"].model
    state = load_pickle(scores_path, {"cells": {}, "done": []})
    cells_data, done = state["cells"], set(state["done"])
    print(f"phase 2: {len(trajectories)} trajectories, {len(cells(config))} cells each; resuming with {len(done)} done")

    for completed, trajectory in enumerate(tqdm(trajectories, desc="phase 2: policies"), start=1):
        key = f"{trajectory[0]}_{trajectory[1]}_{trajectory[2]}"
        if key in done:
            continue

        collected = collect_trajectory(oracle, policy, sam, encoders, detection_data, trajectory, config, commit_thresholds)
        if collected is not None:
            merge_cells(cells_data, collected)
        done.add(key)

        if completed % config.checkpoint_every == 0:
            save_pickle(scores_path, {"cells": cells_data, "done": list(done)})
        if completed % config.plot_every == 0 and cells_data:
            draw_curves(cells_data, config, len(done))

    save_pickle(scores_path, {"cells": cells_data, "done": list(done)})
    return cells_data, len(done)


@hydra.main(config_path="../conf", config_name="covariate_shift", version_base=None)
def run(config: DictConfig):
    """Phase 1: calibrate each cell's commit threshold on the oracle rollouts. Phase 2: run the oracle and the
    per-cell score-gated policies, pool the candidate scores, and draw the covariate-shift heatmaps."""

    os.makedirs(config.out_dir, exist_ok=True)
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    encoders = load_dataset_encoders(list(config.encoders), DEVICE, DTYPE)
    trackers = load_trackers(config)
    trajectories = test_trajectories(person_path, config.n_traj)

    commit_thresholds = calibrate(
        trackers["oracle"], trackers["oracle"].model, encoders, detection_data, trajectories, config,
        f"{config.out_dir}/calibration.pkl")
    cells_data, n_traj = collect_scores(
        trackers, encoders, detection_data, trajectories, config, commit_thresholds, f"{config.out_dir}/scores.pkl")

    curves = draw_curves(cells_data, config, n_traj)
    summary = {"curves": curves, "alphas": list(config.alphas), "memory_size": config.memory_size,
               "commit_thresholds": {str(cell): value for cell, value in commit_thresholds.items()}, "n_traj": n_traj}
    np.save(f"{config.out_dir}/curves.npy", summary, allow_pickle=True)


if __name__ == "__main__":
    run()

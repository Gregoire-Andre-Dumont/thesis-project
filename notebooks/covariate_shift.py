"""Covariate shift of a score-gated memory policy, and how anchor weight + memory size reduce it, per encoder.

The POLICY is SAM 2 whose memory bank is gated live on the re-ID score itself: at each frame it commits the
chosen mask iff  alpha * chamfer(pred, anchor) + (1 - alpha) * mean_e chamfer(pred, recent_entry_e) >= tau.
A wrong commit therefore corrupts the very memory the next commit is judged against -- the covariate shift.
The ORACLE is SAM 2 whose memory is gated on the true IoU, so its memory never drifts; it is the ceiling.

Because the score drives the memory, each (encoder, memory size, alpha) cell needs its OWN policy rollout --
there is no post-hoc sweep for the policy. The full grid is expensive, so we run only two 1-D sweeps (on the
perception encoder): anchor weight at a fixed memory (alpha_sweep_memory), and memory size at a fixed anchor
weight (memory_sweep_alpha). The commit threshold tau is calibrated per cell in phase 1: the score cut, pooled
over the oracle rollouts, that best matches (max F1) the true-IoU "should commit" label (true IoU > threshold
and no occlusion). Phase 2 then runs, per trajectory, one oracle rollout plus one score-gated rollout per cell
(all sharing the SAM backbone image cache), and scores the same candidates -- the target and its nearest
distractors -- against each rollout's resulting memory.

For each encoder we draw two line panels: target-vs-distractor AUC vs anchor weight, and vs memory size, each
with the oracle (true-IoU memory) and policy (score-gated memory) curves and the covariate-shift gap shaded
between. A higher anchor weight (the score leans on the incorruptible anchor) or a larger memory (a single bad
commit is diluted) should shrink the gap, and only when the encoder can re-identify the target at all.
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

def nearest_distractors(clean_boxes, target_box, k):
    """The k clean people whose box centre is closest to the target's centre (the candidate distractors)."""

    target_x, target_y = box_center(target_box)

    def squared_distance_to_target(box):
        box_x, box_y = box_center(box)
        return (box_x - target_x) ** 2 + (box_y - target_y) ** 2

    ranked_boxes = sorted(clean_boxes, key=squared_distance_to_target)
    return ranked_boxes[:k]


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
    """The (encoder, memory size, anchor weight) cells actually rolled out: the union of the two 1-D sweeps --
    anchor weight at a fixed memory, and memory size at a fixed anchor weight -- not the full grid. Stable order,
    de-duplicated (the two sweeps share their crossing point)."""

    combinations = []
    for encoder in config.encoders:
        for alpha in config.alphas:
            combinations.append((encoder, config.alpha_sweep_memory, alpha))
        for memory_size in config.memory_sizes:
            combinations.append((encoder, memory_size, config.memory_sweep_alpha))

    seen = set()
    unique = []
    for cell in combinations:
        if cell not in seen:
            seen.add(cell)
            unique.append(cell)
    return unique


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
    committed. Pooled across trajectories these calibrate each cell's commit threshold. Same record shape as
    candidate_record, under a single 'oracle' memory, so mixed_score reads it unchanged."""

    frames, masks = oracle_data["frames"], oracle_data["predicted"]
    should_commit = true_iou_commits(oracle_data["mask_iou"], oracle_data["occlusions"], config.label_iou_threshold)
    committed_positions = np.where(should_commit)[0]

    anchor_tokens = anchor_reference(sam, encoders, frames, oracle_data["boxes"], floor, config.crop_size)
    frame_tokens = mask_foreground(encoders, frames, masks, floor, config.crop_size)
    max_memory = max(config.memory_sizes)

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


def candidate_record(candidate_tokens, label, anchor_tokens, committed_tokens, frame, max_memory, encoders):
    """One candidate's record: its chamfer to the anchor and to the most-recent committed entries (per policy,
    per encoder). An empty candidate scores NaN, dropped later when the F1 is measured."""

    anchor_similarity = {}
    for encoder_name in encoders:
        anchor_similarity[encoder_name] = foreground_chamfer(anchor_tokens[encoder_name], candidate_tokens[encoder_name])

    recent_similarity = {}
    for policy, (committed_positions, entry_tokens) in committed_tokens.items():
        recent_entries = []
        for entry, position in enumerate(committed_positions):
            if position <= frame:
                recent_entries.append(entry)
        recent_entries = recent_entries[-max_memory:]

        recent_similarity[policy] = {}
        for encoder_name in encoders:
            similarities = []
            for entry in recent_entries:
                similarity = foreground_chamfer(entry_tokens[entry][encoder_name], candidate_tokens[encoder_name])
                similarities.append(similarity)
            recent_similarity[policy][encoder_name] = np.array(similarities, np.float32)

    return {"label": label, "anchor": anchor_similarity, "recent": recent_similarity}


def scored_frames(frames, boxes, occlusions, frame_indices, clean_boxes, stride, k_distractors):
    """Yield (frame, distractor_boxes) for each visible, boxed frame that has at least one clean distractor."""

    for frame in range(1, len(frames), stride):
        if occlusions[frame] > 0.5 or float(boxes[frame][2]) <= 0:
            continue
        distractor_boxes = nearest_distractors(clean_boxes.get(int(frame_indices[frame]), []), boxes[frame], k_distractors)
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


def candidate_frames(sam, encoders, oracle_data, floor, config, trajectory):
    """Cell-independent per scored frame: the candidate foreground tokens (the target and its nearest clean
    distractors) and their labels (1 = target, 0 = distractor). Reused to score against every cell's memory."""

    frames, boxes = oracle_data["frames"], oracle_data["boxes"]
    occlusions, frame_indices = oracle_data["occlusions"], oracle_data["frame_indices"]

    video, person, _ = trajectory
    clean_boxes = load_clean_boxes_by_frame(Path(config.detection_data.visible_directory) / f"{video}.json", person)

    scored = []
    for frame, distractor_boxes in scored_frames(frames, boxes, occlusions, frame_indices, clean_boxes, config.stride, config.k_distractors):
        candidate_boxes = [boxes[frame]] + distractor_boxes           # index 0 = target (label 1), rest distractors (0)
        candidate_labels = [1] + [0] * len(distractor_boxes)
        candidate_masks = box_prompt_masks(sam, frames[frame], candidate_boxes)

        repeated_frame = np.repeat(frames[frame][None], len(candidate_masks), axis=0)
        candidate_tokens = mask_foreground(encoders, repeated_frame, np.stack(candidate_masks), floor, config.crop_size)
        scored.append({"frame": frame, "tokens": candidate_tokens, "labels": candidate_labels})
    return scored


def policy_memory(policy, detection_data, warmup, encode, anchor_tokens, floor, alpha, memory_size, threshold, config):
    """Run one score-gated rollout (the window is already decoded and the image cache already warm) and return
    the committed frames' foreground tokens as a memory bank: (post-warmup positions, [{encoder: tokens}])."""

    policy.encode = encode
    policy.anchor_weight = alpha
    policy.memory_size = memory_size
    policy.chamfer_threshold = threshold
    policy.crop_size = config.crop_size
    policy.label_mask_iou = False                       # the policy's own mask IoU is never read -- skip the decode
    policy.prepare(anchor_tokens, floor)
    policy.predict_masks(detection_data)

    positions = []
    tokens = []
    for position, entry_tokens in policy.committed_log:
        if position >= warmup:
            positions.append(position - warmup)
            tokens.append(entry_tokens)
    return np.array(positions, dtype=int), tokens


def collect_trajectory(oracle, policy, sam, encoders, detection_data, trajectory, config, commit_thresholds):
    """Phase 2: one oracle rollout (true-IoU memory) plus one score-gated rollout per cell, all sharing this
    trajectory's SAM image cache. Returns (oracle_records, policy_scores): the oracle candidate records (swept
    post-hoc over alpha/memory) and, per cell, the (labels, scores) of the same candidates against that cell's
    score-gated memory. None if the trajectory is unusable."""

    frame_cache = {} if config.get("share_image_cache", True) else None
    oracle_data = oracle_pass(oracle, detection_data, trajectory, config.max_frames, frame_cache)
    if oracle_data is None or len(oracle_data["frames"]) < 2 or float(oracle_data["boxes"][0][2]) <= 0:
        if frame_cache is not None:
            release_cache(frame_cache, {"oracle": oracle, "policy": policy})
        return None

    frames, boxes = oracle_data["frames"], oracle_data["boxes"]
    floor = crop_floor(config, trajectory, oracle_data["anchor_index"], frames, boxes)
    anchor_tokens = anchor_reference(sam, encoders, frames, boxes, floor, config.crop_size)
    scored = candidate_frames(sam, encoders, oracle_data, floor, config, trajectory)

    # Oracle: its true-IoU memory is policy-independent, so one rollout feeds the whole alpha/memory sweep.
    oracle_commits = np.where(true_iou_commits(oracle_data["mask_iou"], oracle_data["occlusions"], config.label_iou_threshold))[0]
    oracle_memory = {"oracle": committed_reference(encoders, oracle_commits, frames, oracle_data["predicted"], floor, config.crop_size)}
    max_memory = max(config.memory_sizes)

    oracle_records = []
    for entry in scored:
        for candidate_tokens, label in zip(entry["tokens"], entry["labels"]):
            oracle_records.append(candidate_record(candidate_tokens, label, anchor_tokens, oracle_memory, entry["frame"], max_memory, encoders))

    # Policy: the score gates the memory, so every cell is its own rollout scored against its own memory.
    warmup = oracle_data["warmup"]
    policy.frame_cache = frame_cache                         # every cell's rollout reuses this window's SAM features
    policy_scores = {}
    for encoder, memory_size, alpha in cells(config):
        threshold = commit_thresholds[(encoder, memory_size, alpha)]
        threshold = -np.inf if np.isnan(threshold) else threshold
        positions, entry_tokens = policy_memory(policy, detection_data, warmup, encoders[encoder],
                                                anchor_tokens[encoder], floor, alpha, memory_size, threshold, config)
        memory = {"policy": (positions, [{encoder: token} for token in entry_tokens])}

        labels = []
        scores = []
        for entry in scored:
            for candidate_tokens, label in zip(entry["tokens"], entry["labels"]):
                record = candidate_record(candidate_tokens, label, anchor_tokens, memory, entry["frame"], memory_size, [encoder])
                score = mixed_score(record, encoder, "policy", alpha, memory_size)
                if not np.isnan(score):
                    labels.append(label)
                    scores.append(score)
        policy_scores[(encoder, memory_size, alpha)] = (np.array(labels), np.array(scores, np.float32))

    if frame_cache is not None:
        release_cache(frame_cache, {"oracle": oracle, "policy": policy})
    return oracle_records, policy_scores


def labelled_scores(candidates, encoder, policy, alpha, memory_size):
    """The (labels, scores) arrays for one cell: every candidate's alpha-weighted score under this policy,
    with NaN scores (empty foreground) dropped."""

    labels = np.array([candidate["label"] for candidate in candidates])
    scores = np.array([mixed_score(candidate, encoder, policy, alpha, memory_size) for candidate in candidates])
    valid = ~np.isnan(scores)
    return labels[valid], scores[valid]


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


def sweep_aucs(oracle_records, policy_scores, encoder, memory_alpha_points):
    """Oracle and policy target-vs-distractor AUC at each (memory size, anchor weight) point of a 1-D sweep. The
    oracle AUC is read from its one true-IoU memory swept post-hoc; the policy AUC from that cell's rollout."""

    oracle_aucs = []
    policy_aucs = []
    for memory_size, alpha in memory_alpha_points:
        oracle_labels, oracle_values = labelled_scores(oracle_records, encoder, "oracle", alpha, memory_size)
        policy_labels, policy_values = policy_scores.get((encoder, memory_size, alpha), (np.array([]), np.array([])))
        oracle_aucs.append(target_auc(oracle_labels, oracle_values))
        policy_aucs.append(target_auc(policy_labels, policy_values))
    return oracle_aucs, policy_aucs


def draw_curve(axis, positions, tick_labels, oracle_aucs, policy_aucs, title, xlabel):
    """One AUC-vs-swept-variable panel: the oracle and policy curves with the covariate-shift gap shaded between."""

    axis.plot(positions, oracle_aucs, marker="o", color="tab:green", label="oracle (true-IoU memory)")
    axis.plot(positions, policy_aucs, marker="o", color="tab:red", label="policy (score-gated memory)")
    axis.fill_between(positions, policy_aucs, oracle_aucs, color="tab:red", alpha=0.12, label="covariate shift")

    axis.set_xticks(positions, [str(label) for label in tick_labels])
    axis.set_xlabel(xlabel)
    axis.set_ylabel("target-vs-distractor AUC")
    axis.set_ylim(0.5, 1.0)
    axis.set_title(title, fontsize=10)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)


def draw_sweeps(oracle_records, policy_scores, config, n_traj):
    """For each encoder, two line panels: AUC vs anchor weight (memory fixed) and AUC vs memory size (anchor
    weight fixed). Returns the swept AUCs for caching."""

    curves = {}
    for encoder in config.encoders:
        figure, (alpha_axis, memory_axis) = plt.subplots(1, 2, figsize=(12, 5))

        alpha_points = [(config.alpha_sweep_memory, alpha) for alpha in config.alphas]
        oracle_alpha, policy_alpha = sweep_aucs(oracle_records, policy_scores, encoder, alpha_points)
        draw_curve(alpha_axis, list(config.alphas), list(config.alphas), oracle_alpha, policy_alpha,
                   f"AUC vs anchor weight  (memory {config.alpha_sweep_memory})", "anchor weight α")

        memory_points = [(memory_size, config.memory_sweep_alpha) for memory_size in config.memory_sizes]
        oracle_memory, policy_memory = sweep_aucs(oracle_records, policy_scores, encoder, memory_points)
        draw_curve(memory_axis, range(len(config.memory_sizes)), list(config.memory_sizes), oracle_memory, policy_memory,
                   f"AUC vs memory size  (α {config.memory_sweep_alpha})", "memory size")

        figure.suptitle(f"{encoder}  |  {n_traj} trajectories", fontsize=11)
        figure.tight_layout()
        figure.savefig(f"{config.out_dir}/{encoder}.png", dpi=120)
        plt.close(figure)
        curves[encoder] = {"alpha": (oracle_alpha, policy_alpha), "memory": (oracle_memory, policy_memory)}
    return curves


def load_pickle(path, default):
    """The pickled object at `path`, or `default` when the file does not exist yet."""

    return pickle.loads(Path(path).read_bytes()) if Path(path).exists() else default


def save_pickle(path, obj):
    """Atomically write `obj` to `path` so an interrupted run resumes from the last checkpoint, not a torn file."""

    tmp = Path(path).with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(path)


def merge_policy_scores(pooled, trajectory_scores):
    """Concatenate one trajectory's per-cell (labels, scores) onto the pooled totals, in place."""

    for cell, (labels, scores) in trajectory_scores.items():
        if cell in pooled:
            previous_labels, previous_scores = pooled[cell]
            pooled[cell] = (np.concatenate([previous_labels, labels]), np.concatenate([previous_scores, scores]))
        else:
            pooled[cell] = (labels, scores)


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
    pooling the oracle candidate records and the per-cell policy scores, and refreshing the heatmaps. Returns
    (oracle_records, policy_scores, number of finished trajectories)."""

    oracle, policy, sam = trackers["oracle"], trackers["policy"], trackers["oracle"].model
    state = load_pickle(scores_path, {"oracle_records": [], "policy_scores": {}, "done": []})
    oracle_records, policy_scores, done = state["oracle_records"], state["policy_scores"], set(state["done"])
    print(f"phase 2: {len(trajectories)} trajectories, {len(cells(config))} cells each; resuming with {len(done)} done")

    for completed, trajectory in enumerate(tqdm(trajectories, desc="phase 2: policies"), start=1):
        key = f"{trajectory[0]}_{trajectory[1]}_{trajectory[2]}"
        if key in done:
            continue

        collected = collect_trajectory(oracle, policy, sam, encoders, detection_data, trajectory, config, commit_thresholds)
        if collected is not None:
            trajectory_records, trajectory_scores = collected
            oracle_records.extend(trajectory_records)
            merge_policy_scores(policy_scores, trajectory_scores)
        done.add(key)

        if completed % config.checkpoint_every == 0:
            save_pickle(scores_path, {"oracle_records": oracle_records, "policy_scores": policy_scores, "done": list(done)})
        if completed % config.plot_every == 0 and oracle_records:
            draw_sweeps(oracle_records, policy_scores, config, len(done))

    save_pickle(scores_path, {"oracle_records": oracle_records, "policy_scores": policy_scores, "done": list(done)})
    return oracle_records, policy_scores, len(done)


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
    oracle_records, policy_scores, n_traj = collect_scores(
        trackers, encoders, detection_data, trajectories, config, commit_thresholds, f"{config.out_dir}/scores.pkl")

    curves = draw_sweeps(oracle_records, policy_scores, config, n_traj)
    summary = {"curves": curves, "alphas": list(config.alphas), "memory_sizes": list(config.memory_sizes),
               "alpha_sweep_memory": config.alpha_sweep_memory, "memory_sweep_alpha": config.memory_sweep_alpha,
               "commit_thresholds": {str(cell): value for cell, value in commit_thresholds.items()}, "n_traj": n_traj}
    np.save(f"{config.out_dir}/curves.npy", summary, allow_pickle=True)


if __name__ == "__main__":
    run()

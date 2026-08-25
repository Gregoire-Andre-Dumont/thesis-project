"""Covariate shift of a memory policy, and how anchor weight + memory size reduce it, per image encoder.

Two SAM 2 rollouts per trajectory share the same frames: the ORACLE commits to memory on the true IoU, the
POLICY on SAM's own predicted IoU. Each policy's commit threshold is the one maximizing F1, on the oracle
rollout, against the ground-truth "should commit" label (true IoU > threshold and no occlusion) -- which is
why the oracle runs first. At each step the memory holds the anchor (frame-0 target) and the recent committed
frames' predicted masks; we take foreground tokens of those references and of the candidates (the target and
its nearest distractors) and score each candidate

    score = alpha * chamfer(candidate, anchor) + (1 - alpha) * mean_e chamfer(candidate, recent_entry_e)

then take the target-vs-distractor AUC. alpha (anchor weight) and memory size are swept post-hoc. For each of
two encoders (perception, hiera_sam) we draw two heatmaps over (alpha, memory size): the policy-rollout AUC,
and the oracle-minus-policy AUC gap -- the covariate shift, which a higher anchor weight or larger memory
should shrink, and only when the encoder can actually re-identify the target.
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

POLICIES = ("oracle", "policy")


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


def run_rollouts(trackers, detection_data, trajectory, max_frames, frame_cache):
    """Run both policies' SAM 2 rollouts over the trajectory, sharing the frame cache (target-independent
    image encodings). Returns (rollout, shared, anchor_index), or None if the anchor is missing, where
    rollout[policy] holds the post-warmup predicted masks and per-frame IoU signals (mask_iou = true pseudo-GT
    IoU, pred_iou = SAM's own estimate) and shared holds the frames/boxes/occlusions/frame_indices."""

    rollout = {}
    shared = None
    anchor_index = None

    for policy, tracker in trackers.items():
        window = load_window(detection_data, trajectory, max_frames)
        if window is None:
            return None
        warmup, anchor_index = window

        tracker.frame_cache = frame_cache
        predicted_masks = tracker.predict_masks(detection_data).numpy()

        keep = slice(warmup, None)
        rollout[policy] = {
            "predicted": predicted_masks[keep],
            "mask_iou": tracker.mask_iou_scores.numpy()[keep],
            "pred_iou": tracker.iou_scores.numpy()[keep],
        }
        if shared is None:
            shared = (detection_data.frames[keep], detection_data.bboxes_norm[keep],
                      detection_data.occlusions[keep], detection_data.frame_indices[keep])

    return rollout, shared, anchor_index


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


def oracle_signals(oracle, detection_data, trajectory, max_frames):
    """Phase 1: run only the oracle rollout and return its per-frame IoU signals (mask_iou = true pseudo-GT IoU,
    pred_iou = SAM's estimate, occlusion), or None if the anchor is missing. No chamfers, no policy rollout --
    these are pooled across all trajectories to calibrate the global commit thresholds."""

    window = load_window(detection_data, trajectory, max_frames)
    if window is None:
        return None
    warmup, _ = window

    oracle.frame_cache = None
    oracle.predict_masks(detection_data)
    keep = slice(warmup, None)
    return {"mask_iou": oracle.mask_iou_scores.numpy()[keep],
            "pred_iou": oracle.iou_scores.numpy()[keep],
            "occlusion": np.asarray(detection_data.occlusions)[keep]}


def global_thresholds(signals, label_iou_threshold):
    """The single F1-optimal commit threshold per policy, over every trajectory's oracle signals pooled: on
    true IoU for the oracle and on predicted IoU for the policy, against the "should commit" label (true IoU >
    threshold and no occlusion)."""

    mask_iou = np.concatenate([signal["mask_iou"] for signal in signals.values()])
    pred_iou = np.concatenate([signal["pred_iou"] for signal in signals.values()])
    occlusion = np.concatenate([signal["occlusion"] for signal in signals.values()])

    should_commit = (mask_iou > label_iou_threshold) & (occlusion < 0.5)
    return best_f1_threshold(mask_iou, should_commit), best_f1_threshold(pred_iou, should_commit)


def apply_commits(rollout, occlusions, oracle_threshold, policy_threshold):
    """Per-policy commit masks from the global thresholds: the oracle gates on true IoU (and visibility), the
    policy on SAM's own predicted IoU. Returns {policy: bool array}."""

    is_visible = occlusions < 0.5
    oracle_commits = (rollout["oracle"]["mask_iou"] >= oracle_threshold) & is_visible
    policy_commits = rollout["policy"]["pred_iou"] >= policy_threshold
    return {"oracle": oracle_commits, "policy": policy_commits}


def reference_tokens(sam, encoders, rollout, frames, boxes, floor, commits, crop_size):
    """The anchor reference (frame-0 target foreground tokens) and each policy's committed predicted-mask
    tokens at their own frames. Returns (anchor_tokens, {policy: (positions, [tokens])})."""

    anchor_mask = box_prompt_masks(sam, frames[0], [boxes[0]])[0]
    anchor_tokens = mask_foreground(encoders, frames[0][None], anchor_mask[None], floor, crop_size)[0]

    committed_tokens = {}
    for policy, commit_mask in commits.items():
        committed_positions = np.where(commit_mask)[0]
        if len(committed_positions) == 0:
            committed_tokens[policy] = (committed_positions, [])
            continue

        committed_frames = frames[committed_positions]
        committed_masks = rollout[policy]["predicted"][committed_positions]
        tokens = mask_foreground(encoders, committed_frames, committed_masks, floor, crop_size)
        committed_tokens[policy] = (committed_positions, tokens)

    return anchor_tokens, committed_tokens


def candidate_record(candidate_tokens, label, anchor_tokens, committed_tokens, frame, max_memory, encoders):
    """One candidate's record: its chamfer to the anchor and to the most-recent committed entries (per policy,
    per encoder). An empty candidate scores NaN, dropped later at the AUC."""

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


def collect_trajectory(trackers, sam, encoders, detection_data, trajectory, config, oracle_threshold, policy_threshold):
    """Phase 2: run both rollouts, commit with the global thresholds, and return one record per candidate (see
    candidate_record) so the alpha/memory sweep is pure arithmetic. None if the trajectory is unusable."""

    # Sharing the image cache across the oracle/policy rollouts saves one SAM backbone pass but holds
    # max_frames image encodings in RAM -- only worth it when reused across many rollouts, so default off.
    frame_cache = {} if config.get("share_image_cache", False) else None
    result = run_rollouts(trackers, detection_data, trajectory, config.max_frames, frame_cache)
    if frame_cache is not None:
        release_cache(frame_cache, trackers)         # rollouts done; the token extraction below re-encodes fresh
    if result is None:
        return None
    rollout, (frames, boxes, occlusions, frame_indices), anchor_index = result
    if len(frames) < 2 or float(boxes[0][2]) <= 0:
        return None

    video, person, _ = trajectory
    visible_json = Path(config.detection_data.visible_directory) / f"{video}.json"
    amodal_json = Path(config.detection_data.amodal_directory) / f"{video}.json"
    anchor_amodal = load_bboxes(str(amodal_json), str(visible_json), person, use_amodal=True)[anchor_index]
    anchor_box = anchor_amodal if float(anchor_amodal[2]) > 0 else boxes[0]
    floor = anchor_size_pixels(anchor_box, frames[0].shape)

    commits = apply_commits(rollout, occlusions, oracle_threshold, policy_threshold)
    anchor_tokens, committed_tokens = reference_tokens(sam, encoders, rollout, frames, boxes, floor, commits, config.crop_size)

    clean_boxes = load_clean_boxes_by_frame(visible_json, person)
    max_memory = max(config.memory_sizes)

    candidates = []
    for frame, distractor_boxes in scored_frames(frames, boxes, occlusions, frame_indices, clean_boxes, config.stride, config.k_distractors):
        candidate_boxes = [boxes[frame]] + distractor_boxes           # index 0 = target (label 1), rest distractors (0)
        candidate_labels = [1] + [0] * len(distractor_boxes)
        candidate_masks = box_prompt_masks(sam, frames[frame], candidate_boxes)

        repeated_frame = np.repeat(frames[frame][None], len(candidate_masks), axis=0)
        candidate_tokens = mask_foreground(encoders, repeated_frame, np.stack(candidate_masks), floor, config.crop_size)

        for tokens, label in zip(candidate_tokens, candidate_labels):
            candidates.append(candidate_record(tokens, label, anchor_tokens, committed_tokens, frame, max_memory, encoders))

    return candidates


def auc(labelled_scores):
    """Target-vs-distractor AUC over (label, score) pairs; NaN scores dropped; NaN if one class remains."""

    labels = np.array([label for label, _score in labelled_scores])
    scores = np.array([score for _label, score in labelled_scores])

    valid = ~np.isnan(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.sum() == 0 or labels.sum() == len(labels):
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


def score_grid(candidates, encoder, policy, alphas, memory_sizes):
    """AUC grid (rows = memory sizes, columns = anchor weights) for one encoder and policy."""

    grid = np.full((len(memory_sizes), len(alphas)), np.nan)
    for row, memory_size in enumerate(memory_sizes):
        for column, alpha in enumerate(alphas):
            labelled_scores = []
            for candidate in candidates:
                score = mixed_score(candidate, encoder, policy, alpha, memory_size)
                labelled_scores.append((candidate["label"], score))
            grid[row, column] = auc(labelled_scores)
    return grid


def annotate_cells(axis, grid, low, high):
    """Write each finite AUC value into its heatmap cell, in a colour that contrasts with the cell."""

    midpoint = (low + high) / 2
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = grid[row, column]
            if not np.isfinite(value):
                continue
            text_color = "white" if value < midpoint else "black"
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)


def draw_panel(axis, grid, title, cmap, low, high, alphas, memory_sizes):
    """Draw one AUC heatmap (anchor weight on x, memory size on y) with per-cell labels; return the image."""

    image = axis.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=low, vmax=high)
    axis.set_xticks(range(len(alphas)), [f"{alpha:g}" for alpha in alphas])
    axis.set_yticks(range(len(memory_sizes)), [str(memory_size) for memory_size in memory_sizes])
    axis.set_xlabel("anchor weight α")
    axis.set_ylabel("memory size")
    axis.set_title(title, fontsize=10)
    annotate_cells(axis, grid, low, high)
    return image


def plot_heatmaps(policy_grid, gap_grid, encoder, alphas, memory_sizes, n_traj, path):
    """Two heatmaps for one encoder: the policy-rollout AUC, and the oracle-minus-policy AUC gap (covariate shift)."""

    figure, (policy_axis, gap_axis) = plt.subplots(1, 2, figsize=(12, 5))

    policy_image = draw_panel(policy_axis, policy_grid, "policy-rollout AUC", "viridis", 0.5, 1.0, alphas, memory_sizes)
    figure.colorbar(policy_image, ax=policy_axis, fraction=0.046)

    gap_top = np.nanmax(gap_grid)
    gap_high = gap_top if np.isfinite(gap_top) else 0.1
    gap_image = draw_panel(gap_axis, gap_grid, "covariate shift (oracle - policy AUC)", "magma", 0.0, gap_high, alphas, memory_sizes)
    figure.colorbar(gap_image, ax=gap_axis, fraction=0.046)

    figure.suptitle(f"{encoder}  |  {n_traj} trajectories", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def draw_heatmaps(candidates, config, n_traj):
    """Score every encoder over both policies and draw its two heatmaps; return the grids for caching."""

    grids = {}
    for encoder in config.encoders:
        policy_grid = score_grid(candidates, encoder, "policy", config.alphas, config.memory_sizes)
        oracle_grid = score_grid(candidates, encoder, "oracle", config.alphas, config.memory_sizes)
        gap_grid = oracle_grid - policy_grid

        figure_path = f"{config.out_dir}/{encoder}.png"
        plot_heatmaps(policy_grid, gap_grid, encoder, list(config.alphas), list(config.memory_sizes), n_traj, figure_path)
        grids[encoder] = {"policy": policy_grid, "oracle": oracle_grid, "gap": gap_grid}
    return grids


def load_checkpoint(path):
    """Load (candidates, done) from the checkpoint, or empties when there is none."""

    if not Path(path).exists():
        return [], set()
    state = pickle.loads(Path(path).read_bytes())
    return state["candidates"], set(state["done"])


def save_checkpoint(path, candidates, done):
    """Atomically write the collected candidate chamfers and finished trajectory keys, so the run resumes here."""

    tmp = Path(path).with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps({"candidates": candidates, "done": list(done)}))
    tmp.replace(path)


def load_signals(path):
    """Load the phase-1 cache {trajectory_key: per-frame oracle signals}, or empty when there is none."""

    if not Path(path).exists():
        return {}
    return pickle.loads(Path(path).read_bytes())


def save_signals(path, signals):
    """Atomically write the phase-1 oracle signals cache."""

    tmp = Path(path).with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps(signals))
    tmp.replace(path)


def load_trackers(config):
    """Instantiate the oracle and policy SAM 2 trackers from their configs."""

    policy_config = {"oracle": config.oracle_config, "policy": config.baseline_config}
    trackers = {}
    for policy in POLICIES:
        trackers[policy] = hydra.utils.instantiate(OmegaConf.load(policy_config[policy]).tracker)
    return trackers


def calibrate_thresholds(oracle, detection_data, trajectories, config, signals_path):
    """Phase 1: cache every trajectory's oracle IoU signals (resuming from the pkl), then fit and return the
    global (oracle_threshold, policy_threshold)."""

    signals = load_signals(signals_path)
    print(f"phase 1: oracle signals over {len(trajectories)} trajectories; resuming with {len(signals)} cached")

    for completed, trajectory in enumerate(tqdm(trajectories, desc="phase 1: oracle"), start=1):
        key = f"{trajectory[0]}_{trajectory[1]}_{trajectory[2]}"
        if key in signals:
            continue
        signal = oracle_signals(oracle, detection_data, trajectory, config.max_frames)
        if signal is not None:
            signals[key] = signal
        if completed % config.checkpoint_every == 0:
            save_signals(signals_path, signals)
    save_signals(signals_path, signals)

    oracle_threshold, policy_threshold = global_thresholds(signals, config.label_iou_threshold)
    print(f"global thresholds -> oracle(true IoU)={oracle_threshold:.3f}  policy(pred IoU)={policy_threshold:.3f}")
    return oracle_threshold, policy_threshold


def collect_chamfers(trackers, encoders, detection_data, trajectories, config, oracle_threshold, policy_threshold, chamfers_path):
    """Phase 2: run the policies with the global thresholds, caching candidate chamfers (resuming from the pkl)
    and refreshing the heatmaps. Returns (candidates, number of finished trajectories)."""

    sam = trackers["oracle"].model                           # box-prompts the anchor / candidate masks
    candidates, done = load_checkpoint(chamfers_path)

    for completed, trajectory in enumerate(tqdm(trajectories, desc="phase 2: policies"), start=1):
        key = f"{trajectory[0]}_{trajectory[1]}_{trajectory[2]}"
        if key in done:
            continue

        collected = collect_trajectory(trackers, sam, encoders, detection_data,
                                       trajectory, config, oracle_threshold, policy_threshold)
        if collected is not None:
            candidates.extend(collected)
        done.add(key)

        if completed % config.checkpoint_every == 0:
            save_checkpoint(chamfers_path, candidates, done)
        if completed % config.plot_every == 0 and candidates:
            draw_heatmaps(candidates, config, len(done))

    save_checkpoint(chamfers_path, candidates, done)
    return candidates, len(done)




@hydra.main(config_path="../conf", config_name="covariate_shift", version_base=None)
def run(config: DictConfig):
    """Phase 1: cache oracle IoU signals and fit the global thresholds. Phase 2: run the policies, cache the
    chamfer similarities, and draw the covariate-shift heatmaps."""

    os.makedirs(config.out_dir, exist_ok=True)
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    encoders = load_dataset_encoders(list(config.encoders), DEVICE, DTYPE)
    trackers = load_trackers(config)
    trajectories = test_trajectories(person_path, config.n_traj)

    oracle_threshold, policy_threshold = calibrate_thresholds(
        trackers["oracle"], detection_data, trajectories, config, f"{config.out_dir}/signals.pkl")
    candidates, n_traj = collect_chamfers(
        trackers, encoders, detection_data, trajectories, config,
        oracle_threshold, policy_threshold, f"{config.out_dir}/chamfers.pkl")

    grids = draw_heatmaps(candidates, config, n_traj)
    summary = {"grids": grids, "alphas": list(config.alphas), "memory_sizes": list(config.memory_sizes),
               "oracle_threshold": oracle_threshold, "policy_threshold": policy_threshold, "n_traj": n_traj}
    np.save(f"{config.out_dir}/grids.npy", summary, allow_pickle=True)


if __name__ == "__main__":
    run()

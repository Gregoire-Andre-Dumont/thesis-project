"""Covariate shift of a memory policy, and how anchor weight + memory size reduce it, per image encoder.

Two SAM 2 rollouts per trajectory share the same frames: the ORACLE commits to memory on the true IoU, the
POLICY on SAM's own predicted IoU (both gated at the same oracle-derived threshold). At each step the memory
holds the anchor (frame-0 target) and the recent committed frames' predicted masks. We take foreground tokens
of those references and of the candidates (the target and its nearest distractors), and score each candidate

    score = alpha * chamfer(candidate, anchor) + (1 - alpha) * mean_e chamfer(candidate, recent_entry_e)

then take the target-vs-distractor AUC. alpha (anchor weight) and memory size are swept post-hoc. For each of
two encoders (perception, hiera_sam) we draw two heatmaps over (alpha, memory size): the policy-rollout AUC,
and the oracle-minus-policy AUC gap -- the covariate shift, which a higher anchor weight or larger memory
should shrink, and only when the encoder can actually re-identify the target.
"""
import logging
import os
import pickle
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root on path for `src` and create_anchor_dataset

from pe_reid_longrange import test_trajectories, box_prompt_masks, chamfer_scores, DEVICE, DTYPE
from create_anchor_dataset import anchor_trajectory_index, slice_detection_data_for_tracker
from src.offline_training.dataset_encoders import crop_around_masks, anchor_size_pixels, load_dataset_encoders
from src.offline_training.dataset_labels import load_clean_boxes_by_frame, _box_center as box_center
from src.utils.load_bboxes import load_bboxes

logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
os.environ["HYDRA_FULL_ERROR"] = "1"

POLICIES = ("oracle", "policy")


def nearest_distractors(clean_boxes, target_box, k):
    """The k clean people whose box centre is closest to the target's centre (the candidate distractors)."""
    target_center = box_center(target_box)
    ranked = sorted(clean_boxes, key=lambda b: (box_center(b)[0] - target_center[0]) ** 2
                                               + (box_center(b)[1] - target_center[1]) ** 2)
    return ranked[:k]


@torch.inference_mode()
def mask_foreground(encoders, frames, masks, floor, crop_size):
    """Foreground tokens of each (frame, mask) for every encoder: crop around the mask (floored at the anchor
    scale) and keep the tokens the mask covers. `frames`/`masks` are aligned arrays; returns a list of
    {encoder: fg_tokens} (a tensor may be empty if the mask vanished after cropping)."""
    crops, crop_masks = crop_around_masks(frames, masks.astype(np.float32), crop_size, 0.2, floor)
    crop_masks_t = torch.from_numpy(crop_masks).unsqueeze(1)
    per_item = [{} for _ in range(len(masks))]
    for name, encode in encoders.items():
        tokens = encode(crops)
        grid = round(tokens.shape[1] ** 0.5)
        fg = (F.interpolate(crop_masks_t.to(tokens.device).float(), size=(grid, grid), mode="nearest") > 0.5).flatten(1)
        for item, token_map, mask in zip(per_item, tokens, fg):
            item[name] = token_map[mask].clone()
        del tokens
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return per_item


def sim(reference_tokens, candidate_tokens):
    """Unidirectional foreground chamfer of the candidate to a reference (candidate -> reference)."""
    return chamfer_scores(reference_tokens, candidate_tokens)[0]


def run_rollouts(trackers, detection_data, trajectory, max_frames, frame_cache):
    """Run both policies' SAM 2 rollouts over the trajectory. Returns per policy a dict of the post-warmup
    predicted masks and per-frame IoU signals (mask_iou = true pseudo-GT IoU, pred_iou = SAM's own estimate),
    plus the shared frames/boxes/occlusions/frame_indices and the anchor index, or None if the anchor is
    missing. The frame cache (target-independent image encodings) is shared across both."""
    video, person, anchor_frame = trajectory
    rollout, shared, anchor_index = {}, None, None
    for policy, tracker in trackers.items():
        detection_data.initialize_target(video, person)
        anchor_index = anchor_trajectory_index(detection_data, anchor_frame)
        if anchor_index is None:
            return None
        warmup = slice_detection_data_for_tracker(detection_data, anchor_index)
        end = warmup + max_frames
        for attribute in ("frames", "bboxes_norm", "occlusions", "frame_indices"):
            setattr(detection_data, attribute, getattr(detection_data, attribute)[:end])
        tracker.frame_cache = frame_cache
        predicted = tracker.predict_masks(detection_data).numpy()
        keep = slice(warmup, None)
        rollout[policy] = {"predicted": predicted[keep],
                           "mask_iou": tracker.mask_iou_scores.numpy()[keep],
                           "pred_iou": tracker.iou_scores.numpy()[keep]}
        if shared is None:
            shared = (detection_data.frames[keep], detection_data.bboxes_norm[keep],
                      detection_data.occlusions[keep], detection_data.frame_indices[keep])
    return rollout, shared, anchor_index


def best_f1_threshold(scores, labels):
    """Threshold on `scores` (commit iff score >= τ) that maximizes F1 against boolean `labels`; NaN if the
    labels are single-class. Computed by sweeping every candidate cut in one sorted pass."""
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return np.nan
    order = np.argsort(-scores)
    sorted_scores, sorted_labels = scores[order], labels[order].astype(int)
    true_positive = np.cumsum(sorted_labels)                     # among the top-i highest scores
    predicted_positive = np.arange(1, len(scores) + 1)
    f1 = 2 * true_positive / (predicted_positive + positives)
    return float(sorted_scores[int(np.argmax(f1))])


def collect_trajectory(trackers, sam, encoders, detection_data, trajectory, config, frame_cache):
    """Score-free collection: run both rollouts and return one record per candidate holding its chamfer to the
    anchor and to the most-recent committed entries (per policy, per encoder), so the alpha/memory sweep is pure
    arithmetic afterwards. Returns a list of candidate dicts, or None."""
    result = run_rollouts(trackers, detection_data, trajectory, config.max_frames, frame_cache)
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

    # Anchor reference: the frame-0 target's foreground tokens (policy-independent, uncorruptible).
    anchor_mask = box_prompt_masks(sam, frames[0], [boxes[0]])[0]
    if (anchor_mask > 0).sum() == 0:
        return None
    anchor_tokens = mask_foreground(encoders, frames[0][None], anchor_mask[None], floor, config.crop_size)[0]

    # Commit gate per policy: the threshold maximizing F1 (on the ORACLE rollout) against the ground-truth
    # "should commit" label -- true IoU > label threshold and no occlusion. The oracle gates on the true IoU,
    # the policy on SAM's own predicted IoU; each is calibrated to that same oracle label, then applied to its
    # own rollout's signal. (This is why the oracle rollout is needed first.)
    visible = occlusions < 0.5
    label = (rollout["oracle"]["mask_iou"] > config.label_iou_threshold) & visible
    tau = {"oracle": best_f1_threshold(rollout["oracle"]["mask_iou"], label),
           "policy": best_f1_threshold(rollout["oracle"]["pred_iou"], label)}
    if np.isnan(tau["oracle"]) or np.isnan(tau["policy"]):
        return None
    commits = {"oracle": (rollout["oracle"]["mask_iou"] >= tau["oracle"]) & visible,
               "policy": rollout["policy"]["pred_iou"] >= tau["policy"]}

    # Recent-memory references: each policy's committed predicted masks, at their own frames.
    max_memory = max(config.memory_sizes)
    entries = {}
    for policy, committed in commits.items():
        positions = np.where(committed)[0]
        tokens = (mask_foreground(encoders, frames[positions], rollout[policy]["predicted"][positions], floor, config.crop_size)
                  if len(positions) else [])
        entries[policy] = (positions, tokens)

    clean_boxes = load_clean_boxes_by_frame(visible_json, person)
    candidates = []
    for t in range(1, len(frames), config.stride):
        if occlusions[t] > 0.5 or float(boxes[t][2]) <= 0:
            continue
        distractors = nearest_distractors(clean_boxes.get(int(frame_indices[t]), []), boxes[t], config.k_distractors)
        if not distractors:
            continue

        candidate_boxes = [boxes[t]] + distractors            # index 0 = target (label 1), rest distractors (0)
        labels = [1] + [0] * len(distractors)
        masks = box_prompt_masks(sam, frames[t], candidate_boxes)
        frame_tiles = np.repeat(frames[t][None], len(masks), axis=0)
        candidate_tokens = mask_foreground(encoders, frame_tiles, np.stack(masks), floor, config.crop_size)

        for candidate, label in zip(candidate_tokens, labels):
            anchor_sim = {name: sim(anchor_tokens[name], candidate[name]) for name in encoders}
            if any(np.isnan(v) for v in anchor_sim.values()):
                continue
            recent = {}
            for policy, (positions, tokens) in entries.items():
                usable = [i for i, p in enumerate(positions) if p <= t][-max_memory:]   # most-recent committed by t
                recent[policy] = {name: np.array([sim(tokens[i][name], candidate[name]) for i in usable], np.float32)
                                  for name in encoders}
            candidates.append({"label": label, "anchor": anchor_sim, "recent": recent})
    return candidates


def auc(pairs):
    """Target-vs-distractor AUC over (label, score) pairs; NaN scores dropped; NaN if one class remains."""
    labels = np.array([label for label, _ in pairs])
    scores = np.array([score for _, score in pairs])
    keep = ~np.isnan(scores)
    labels, scores = labels[keep], scores[keep]
    return roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else np.nan


def score_grid(candidates, encoder, policy, alphas, memory_sizes):
    """AUC grid (rows = memory sizes, cols = alphas) for one encoder/policy over all collected candidates."""
    grid = np.full((len(memory_sizes), len(alphas)), np.nan)
    for r, memory_size in enumerate(memory_sizes):
        for c, alpha in enumerate(alphas):
            pairs = []
            for candidate in candidates:
                anchor_sim = candidate["anchor"][encoder]
                recent = candidate["recent"][policy][encoder][-memory_size:]
                recent = recent[~np.isnan(recent)]
                recent_sim = recent.mean() if len(recent) else anchor_sim     # no memory yet -> anchor only
                pairs.append((candidate["label"], alpha * anchor_sim + (1 - alpha) * recent_sim))
            grid[r, c] = auc(pairs)
    return grid


def plot_heatmaps(policy_grid, diff_grid, encoder, alphas, memory_sizes, n_traj, path):
    """Two heatmaps for one encoder: the policy-rollout AUC, and the oracle-minus-policy AUC gap (covariate shift)."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, grid, title, cmap, vlim in (
            (axes[0], policy_grid, "policy-rollout AUC", "viridis", (0.5, 1.0)),
            (axes[1], diff_grid, "covariate shift (oracle - policy AUC)", "magma", (0.0, np.nanmax(diff_grid) if np.isfinite(np.nanmax(diff_grid)) else 0.1))):
        image = axis.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        axis.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas])
        axis.set_yticks(range(len(memory_sizes)), [str(m) for m in memory_sizes])
        axis.set_xlabel("anchor weight α")
        axis.set_ylabel("memory size")
        axis.set_title(title, fontsize=10)
        for r in range(len(memory_sizes)):
            for c in range(len(alphas)):
                if np.isfinite(grid[r, c]):
                    axis.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center", fontsize=7,
                              color="white" if grid[r, c] < (vlim[0] + vlim[1]) / 2 else "black")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(f"{encoder}  |  {n_traj} trajectories", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


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


def draw(candidates, config, out_dir, n_traj):
    """Score every encoder over both policies and draw its two heatmaps; returns the grids for caching."""
    grids = {}
    for encoder in config.encoders:
        policy_grid = score_grid(candidates, encoder, "policy", config.alphas, config.memory_sizes)
        oracle_grid = score_grid(candidates, encoder, "oracle", config.alphas, config.memory_sizes)
        diff_grid = oracle_grid - policy_grid
        plot_heatmaps(policy_grid, diff_grid, encoder, list(config.alphas), list(config.memory_sizes),
                      n_traj, f"{out_dir}/{encoder}.png")
        grids[encoder] = {"policy": policy_grid, "oracle": oracle_grid, "diff": diff_grid}
    return grids


@hydra.main(config_path="../conf", config_name="covariate_shift", version_base=None)
def run(config: DictConfig):
    os.makedirs(config.out_dir, exist_ok=True)
    checkpoint = f"{config.out_dir}/checkpoint.pkl"
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    encoders = load_dataset_encoders(list(config.encoders), DEVICE, DTYPE)

    policy_config = {"oracle": config.oracle_config, "policy": config.baseline_config}
    trackers = {p: hydra.utils.instantiate(OmegaConf.load(policy_config[p]).tracker) for p in POLICIES}
    sam = trackers["oracle"].model                           # box-prompts the anchor / candidate masks

    trajectories = test_trajectories(person_path, config.n_traj)
    candidates, done = load_checkpoint(checkpoint)
    print(f"covariate shift over {len(trajectories)} trajectories; resuming with {len(done)} done, {len(candidates)} candidates")

    for completed, trajectory in enumerate(tqdm(trajectories, desc="covariate-shift"), start=1):
        key = f"{trajectory[0]}_{trajectory[1]}_{trajectory[2]}"
        if key in done:
            continue
        frame_cache = {}
        collected = collect_trajectory(trackers, sam, encoders, detection_data, trajectory, config, frame_cache)
        if collected:
            candidates.extend(collected)
        done.add(key)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if completed % config.checkpoint_every == 0:
            save_checkpoint(checkpoint, candidates, done)
        if completed % config.plot_every == 0 and candidates:
            draw(candidates, config, config.out_dir, len(done))

    save_checkpoint(checkpoint, candidates, done)
    grids = draw(candidates, config, config.out_dir, len(done))
    np.save(f"{config.out_dir}/grids.npy",
            {"grids": grids, "alphas": list(config.alphas), "memory_sizes": list(config.memory_sizes),
             "n_traj": len(done)}, allow_pickle=True)
    print("\ncovariate shift (max oracle-policy AUC gap) per encoder:")
    for encoder, g in grids.items():
        print(f"  {encoder:11s} max_gap={np.nanmax(g['diff']):.3f}  policy_auc[min..max]={np.nanmin(g['policy']):.3f}..{np.nanmax(g['policy']):.3f}")


if __name__ == "__main__":
    run()

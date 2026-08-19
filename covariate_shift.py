"""Covariate shift of SAM 2's IoU estimator (vs a PE foreground score) across two memory-gating policies.

The IoU head is a fixed predictor. Under the ORACLE policy it commits frames to memory on the true IoU;
under the BASELINE policy it commits on its own predicted IoU. Each policy induces a different state
distribution, so a score's ability to tell "on-target" from "captured-by-distractor" can differ.

Per frame we track the policy, box-prompt SAM for the proposal's mask-IoU against the target and its
nearest distractors, and also score the proposal two ways:
  * predicted_iou  -- SAM 2's own IoU-head estimate (mask quality, identity-blind)
  * pe_fg_chamfer  -- unidirectional foreground chamfer of the proposal's PE tokens to the anchor target's

Frame labels: target (target_iou > target_threshold -> 1), distractor (distractor_iou > t and not on-target
-> 0), else ignored. For each distractor threshold t we take each score's AUC separating the two, and draw
one figure per policy (a line per score). A big oracle->baseline drop is the covariate-shift signature.
"""
import logging
import os
import warnings
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from pe_reid_longrange import test_trajectories, chamfer_scores, DEVICE, DTYPE
from create_anchor_dataset import anchor_trajectory_index, slice_detection_data_for_tracker
from src.offline_training.dataset_labels import load_clean_boxes_by_frame, pseudo_iou_labels
from src.offline_training.dataset_encoders import (
    crop_around_masks, anchor_size_pixels, _patch_masks, load_dataset_encoders)

logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

POLICIES = ("oracle", "baseline")
SCORES = ("predicted_iou", "pe_fg_chamfer")
SCORE_COLORS = {"predicted_iou": "#3377cc", "pe_fg_chamfer": "#22aa77"}


@torch.inference_mode()
def pe_foreground_tokens(pe_encode, frames, masks, floor, crop_size, chunk=16):
    """Per-frame PE foreground tokens of each proposal mask, cropped at the anchor scale. Returns a list
    aligned to frames; a frame with an empty mask gets an empty token tensor."""

    crops, crop_masks = crop_around_masks(frames, masks.astype(np.float32), crop_size, 0.2, floor)
    tokens = torch.cat([pe_encode(crops[s:s + chunk]) for s in range(0, len(crops), chunk)], dim=0)
    foreground = _patch_masks(crop_masks, tokens.device)
    return [tokens[i][foreground[i]].clone() for i in range(len(crops))]


def track_trajectory(tracker, pe_encode, detection_data, trajectory, visible_directory, crop_size, frame_cache,
                     max_frames, target_threshold, t_min):
    """Run one policy over a trajectory and return per-frame (target_iou, distractor_iou, predicted_iou,
    pe_fg_chamfer), or None if the anchor is missing. distractor_iou is the max over nearest distractors;
    pe_fg_chamfer is the proposal's PE foreground chamfer to the frame-0 target (NaN on frames the AUC
    ignores). `frame_cache` is a shared {frame_index: image features} dict -- filled by the first policy
    on this trajectory, reused by the rest."""

    video, person, anchor_frame = trajectory
    detection_data.initialize_target(video, person)
    anchor_index = anchor_trajectory_index(detection_data, anchor_frame)
    if anchor_index is None:
        return None

    warmup = slice_detection_data_for_tracker(detection_data, anchor_index)
    end = warmup + max_frames                            # cap the trajectory length before tracking
    detection_data.frames = detection_data.frames[:end]
    detection_data.bboxes_norm = detection_data.bboxes_norm[:end]
    detection_data.occlusions = detection_data.occlusions[:end]
    detection_data.frame_indices = detection_data.frame_indices[:end]

    tracker.frame_cache = frame_cache                   # reuse SAM image encodings across policies + label prompts
    predicted_masks = tracker.predict_masks(detection_data).numpy()
    predicted_iou = tracker.iou_scores.numpy()

    keep = slice(warmup, None)                          # drop the warmup frames so everything starts at the anchor
    predicted_masks = predicted_masks[keep]
    frames = detection_data.frames[keep]
    occlusions = detection_data.occlusions[keep]
    boxes = detection_data.bboxes_norm[keep]
    frame_indices = detection_data.frame_indices[keep]
    predicted_iou = predicted_iou[keep]
    if len(frames) < 2 or float(boxes[0][2]) <= 0:
        return None

    # The pseudo-GT labels reuse the cached per-frame encodings (cache is keyed on the warmup-inclusive clip).
    clean_boxes = load_clean_boxes_by_frame(Path(visible_directory) / f"{video}.json", person)
    precomputed = {t: frame_cache[warmup + t] for t in range(len(frames)) if warmup + t in frame_cache}
    target_iou, distractor_iou = pseudo_iou_labels(
        tracker.model, frames, predicted_masks, boxes, clean_boxes, frame_indices, occlusions,
        precomputed_features=precomputed if len(precomputed) == len(frames) else None, include_occluded=True)
    distractor_iou = distractor_iou.max(axis=1)

    # PE foreground chamfer, but only on the frames the AUC will use (target, or distractor at the lowest t),
    # plus the anchor reference (frame 0). Ignored frames stay NaN and are dropped from the AUC anyway.
    selected = (target_iou > target_threshold) | (distractor_iou > t_min)
    selected[0] = True                                   # the anchor is the chamfer reference
    indices = np.where(selected)[0]
    floor = anchor_size_pixels(boxes[0], frames[0].shape)
    fg_tokens = pe_foreground_tokens(pe_encode, frames[indices], predicted_masks[indices], floor, crop_size)
    anchor_fg = fg_tokens[0]                              # frame 0 is the first selected index
    pe_chamfer = np.full(len(frames), np.nan, np.float32)
    for local, global_index in enumerate(indices):
        pe_chamfer[global_index] = chamfer_scores(anchor_fg, fg_tokens[local])[0]

    return target_iou, distractor_iou, predicted_iou, pe_chamfer


def auc_vs_threshold(target_iou, distractor_iou, score, target_threshold, t_values):
    """AUC of `score` separating target frames (target_iou > target_threshold, label 1) from distractor
    frames (distractor_iou > t and not on target, label 0), for each t. NaN scores/empty classes -> NaN."""

    is_target = target_iou > target_threshold
    curve = []
    for t in t_values:
        is_distractor = (distractor_iou > t) & ~is_target
        keep = (is_target | is_distractor) & ~np.isnan(score)
        labels = is_target[keep].astype(int)
        values = score[keep]
        curve.append(roc_auc_score(labels, values) if 0 < labels.sum() < len(labels) else np.nan)
    return curve


def plot_policy(policy, curves, t_values, target_threshold, n_traj, plot_path):
    """One figure for a policy: a line per score, AUC versus the distractor threshold t."""

    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    for score, curve in curves.items():
        axis.plot(t_values, curve, marker="o", markersize=5, color=SCORE_COLORS.get(score), label=score)
    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (0.50)")

    axis.set_xlabel("distractor threshold t  (frame is a distractor when distractor IoU > t)")
    axis.set_ylabel("target-vs-distractor AUC")
    axis.set_title(f"{policy} policy  |  SAM IoU estimator vs PE foreground\n"
                   f"target = target IoU > {target_threshold}  |  {n_traj} trajectories", fontsize=10)
    axis.set_ylim(0.35, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right", fontsize=9)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=120)
    plt.close(figure)


@hydra.main(config_path="conf", config_name="covariate_shift", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    trajectories = test_trajectories(person_path, config.n_traj)
    t_values = np.linspace(config.t_min, config.t_max, config.t_steps)
    pe_encode = load_dataset_encoders(["perception"], DEVICE, DTYPE)["perception"]
    print(f"evaluating {len(trajectories)} trajectories over {len(t_values)} distractor thresholds")

    def curves_from(columns):
        target_iou, distractor_iou, predicted_iou, pe_chamfer = [np.concatenate(c) for c in columns]
        return {"predicted_iou": auc_vs_threshold(target_iou, distractor_iou, predicted_iou, config.target_threshold, t_values),
                "pe_fg_chamfer": auc_vs_threshold(target_iou, distractor_iou, pe_chamfer, config.target_threshold, t_values)}

    policy_config = {"oracle": config.oracle_config, "baseline": config.baseline_config}
    trackers = {policy: hydra.utils.instantiate(OmegaConf.load(policy_config[policy]).tracker) for policy in POLICIES}
    columns = {policy: [[], [], [], []] for policy in POLICIES}   # per policy: target_iou, distractor_iou, predicted_iou, pe_fg_chamfer

    # Interleaved: run both policies on each trajectory so both figures build together. Both policies now use
    # the SAME SAM model, so one per-trajectory cache is shared -- the first policy fills the SAM image
    # encodings, the second (and both label passes) reuse them.
    for completed, trajectory in enumerate(tqdm(trajectories, desc="covariate-shift"), start=1):
        frame_cache = {}
        for policy in POLICIES:
            result = track_trajectory(trackers[policy], pe_encode, detection_data, trajectory,
                                      config.detection_data.visible_directory, config.crop_size, frame_cache,
                                      config.max_frames, config.target_threshold, config.t_min)
            if result is not None:
                for column, values in zip(columns[policy], result):
                    column.append(values)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        if completed % config.plot_every == 0:                    # refresh both figures as they build
            for policy in POLICIES:
                if columns[policy][0]:
                    target_iou = np.concatenate(columns[policy][0])
                    distractor_iou = np.concatenate(columns[policy][1])
                    n_target = int((target_iou > config.target_threshold).sum())
                    n_distractor = int(((distractor_iou > config.t_min) & (target_iou <= config.target_threshold)).sum())
                    print(f"[{completed}] {policy:9s} frames: target={n_target}  distractor(>{config.t_min})={n_distractor}"
                          f"  | target_iou max={target_iou.max():.2f} mean={target_iou.mean():.2f}"
                          f"  distractor_iou max={distractor_iou.max():.2f}", flush=True)
                    plot_policy(policy, curves_from(columns[policy]), t_values, config.target_threshold, completed, f"{config.plot_prefix}_{policy}.png")

    all_curves = {}
    for policy in POLICIES:
        all_curves[policy] = curves_from(columns[policy])
        target_iou, distractor_iou = np.concatenate(columns[policy][0]), np.concatenate(columns[policy][1])
        n_target = int((target_iou > config.target_threshold).sum())
        n_distractor = int(((distractor_iou > config.t_min) & (target_iou <= config.target_threshold)).sum())
        print(f"{policy:9s} frames: target={n_target}  distractor(>{config.t_min})={n_distractor}")
        plot_policy(policy, all_curves[policy], t_values, config.target_threshold, len(trajectories), f"{config.plot_prefix}_{policy}.png")

    np.save(f"{config.plot_prefix}.npy", {"t_values": t_values, "curves": all_curves, "n_traj": len(trajectories)}, allow_pickle=True)
    print("\nAUC vs distractor threshold t  (predicted_iou | pe_fg_chamfer):")
    for policy in POLICIES:
        print(f"\n{policy}")
        for i, t in enumerate(t_values):
            print(f"  t={t:.2f}  iou={all_curves[policy]['predicted_iou'][i]:.3f}  pe={all_curves[policy]['pe_fg_chamfer'][i]:.3f}")


if __name__ == "__main__":
    run()

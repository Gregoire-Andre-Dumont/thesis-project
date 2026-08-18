"""Covariate shift of SAM 2's IoU estimator across two memory-gating policies.

The IoU head is a fixed predictor. Under the ORACLE policy it commits frames to memory on the true IoU;
under the BASELINE policy it commits on its own predicted IoU. Each policy induces a different state
distribution, so the head's ability to tell "on-target" from "captured-by-distractor" can differ.

Per frame we track the policy, then box-prompt SAM to get the proposal's mask-IoU against the target and
its nearest distractors:
  * target frame     -- target_iou > target_threshold      (label 1)
  * distractor frame -- distractor_iou > t  and not on-target (label 0)
  * everything else  -- ignored
For each distractor threshold t we take the AUC of the predicted-IoU estimator separating the two, and
plot AUC vs t, one line per policy. A big oracle->baseline drop is the covariate-shift signature.
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
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from pe_reid_longrange import test_trajectories
from create_anchor_dataset import anchor_trajectory_index, slice_detection_data_for_tracker
from src.offline_training.dataset_labels import load_clean_boxes_by_frame, pseudo_iou_labels

logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

POLICIES = ("oracle", "baseline")
COLORS = {"oracle": "#22aa77", "baseline": "#cc4444"}


def track_trajectory(tracker, detection_data, trajectory, visible_directory):
    """Run one policy over a trajectory and return per-frame (target_iou, distractor_iou, predicted_iou),
    or None if the anchor is missing. distractor_iou is the max over the nearest clean distractors."""

    video, person, anchor_frame = trajectory
    detection_data.initialize_target(video, person)
    anchor_index = anchor_trajectory_index(detection_data, anchor_frame)
    if anchor_index is None:
        return None

    warmup = slice_detection_data_for_tracker(detection_data, anchor_index)
    predicted_masks = tracker.predict_masks(detection_data).numpy()
    predicted_iou = tracker.iou_scores.numpy()

    keep = slice(warmup, None)                          # drop the warmup frames so everything starts at the anchor
    predicted_masks = predicted_masks[keep]
    frames = detection_data.frames[keep]
    occlusions = detection_data.occlusions[keep]
    boxes = detection_data.bboxes_norm[keep]
    frame_indices = detection_data.frame_indices[keep]
    predicted_iou = predicted_iou[keep]
    if len(frames) == 0:
        return None

    clean_boxes = load_clean_boxes_by_frame(Path(visible_directory) / f"{video}.json", person)
    target_iou, distractor_iou = pseudo_iou_labels(
        tracker.model, frames, predicted_masks, boxes, clean_boxes, frame_indices, occlusions)
    return target_iou, distractor_iou.max(axis=1), predicted_iou


def collect_policy(tracker, detection_data, trajectories, visible_directory):
    """Concatenate per-frame (target_iou, distractor_iou, predicted_iou) over all trajectories for one policy."""

    target_iou, distractor_iou, predicted_iou = [], [], []
    for trajectory in tqdm(trajectories, desc="tracking"):
        result = track_trajectory(tracker, detection_data, trajectory, visible_directory)
        if result is None:
            continue
        target_iou.append(result[0])
        distractor_iou.append(result[1])
        predicted_iou.append(result[2])
    return np.concatenate(target_iou), np.concatenate(distractor_iou), np.concatenate(predicted_iou)


def auc_vs_threshold(target_iou, distractor_iou, predicted_iou, target_threshold, t_values):
    """AUC of the predicted-IoU estimator separating target frames (target_iou > target_threshold, label 1)
    from distractor frames (distractor_iou > t and not on target, label 0), for each distractor threshold t.
    NaN where a class is empty."""

    is_target = target_iou > target_threshold
    curve = []
    for t in t_values:
        is_distractor = (distractor_iou > t) & ~is_target
        keep = is_target | is_distractor
        labels = is_target[keep].astype(int)
        scores = predicted_iou[keep]
        curve.append(roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else np.nan)
    return curve


def plot_curves(curves, t_values, target_threshold, n_traj, plot_path, npy_path):
    """One figure, one line per policy: estimator AUC versus the distractor threshold t."""

    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    for policy, curve in curves.items():
        axis.plot(t_values, curve, marker="o", markersize=5, color=COLORS.get(policy), label=policy)
    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (0.50)")

    axis.set_xlabel("distractor threshold t  (frame is a distractor when distractor IoU > t)")
    axis.set_ylabel("predicted-IoU AUC (target vs distractor frames)")
    axis.set_title(f"SAM 2 IoU estimator: covariate shift across policies\n"
                   f"target = target IoU > {target_threshold}  |  {n_traj} trajectories", fontsize=10)
    axis.set_ylim(0.35, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right", fontsize=9)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=120)
    plt.close(figure)
    np.save(npy_path, {"t_values": t_values, "curves": curves, "n_traj": n_traj}, allow_pickle=True)


@hydra.main(config_path="conf", config_name="covariate_shift", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    trajectories = test_trajectories(person_path, config.n_traj)
    t_values = np.linspace(config.t_min, config.t_max, config.t_steps)
    print(f"evaluating {len(trajectories)} trajectories over {len(t_values)} distractor thresholds")

    policy_config = {"oracle": config.oracle_config, "baseline": config.baseline_config}
    curves = {}
    for policy in POLICIES:
        tracker = hydra.utils.instantiate(OmegaConf.load(policy_config[policy]).tracker)
        target_iou, distractor_iou, predicted_iou = collect_policy(
            tracker, detection_data, trajectories, config.detection_data.visible_directory)
        curves[policy] = auc_vs_threshold(target_iou, distractor_iou, predicted_iou, config.target_threshold, t_values)
        n_target = int((target_iou > config.target_threshold).sum())
        n_distractor = int(((distractor_iou > config.t_min) & (target_iou <= config.target_threshold)).sum())
        print(f"{policy:9s} frames: target={n_target}  distractor(>{config.t_min})={n_distractor}")
        plot_curves(curves, t_values, config.target_threshold, len(trajectories), config.plot, config.npy)

    print("\nAUC vs distractor threshold t:")
    print("  t     " + "  ".join(f"{p:>9s}" for p in curves))
    for i, t in enumerate(t_values):
        print(f"  {t:.2f}  " + "  ".join(f"{curves[p][i]:9.3f}" for p in curves))


if __name__ == "__main__":
    run()

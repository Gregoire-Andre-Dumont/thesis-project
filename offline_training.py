import os
import json
import logging
import warnings
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.offline_training.main_dataset import collate_fn
from src.utils.compute_iou import compute_iou
from src.metrics import coverage_auc


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"


def predict_on_dataset(model, dataset, batch_size=32):
    """Run the trained calibrator on every sample of `dataset` and return predictions in order."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn=collate_fn)
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in loader:
            predictions.append(model(features.to(device)).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def build_trajectory_split(dataset_path, test_size):
    """List the clean trajectories under `dataset_path` and split BY VIDEO following
    PersonPath22's OFFICIAL split (`data/person_path/splits.json` — 138 train / 98
    test video names). This guarantees no within-video leakage and matches the
    standard benchmark protocol."""

    splits = json.load(open("data/person_path/splits.json", "r"))
    train_videos_set = set(name.replace(".mp4", "") for name in splits["train"])
    test_videos_set  = set(name.replace(".mp4", "") for name in splits["test"])

    trajectory_paths = np.array([str(Path(dataset_path) / filename) for filename in sorted(os.listdir(dataset_path))])
    video_names = np.array([Path(path).name.split(".mp4_")[0] for path in trajectory_paths])

    train_indices = np.array([i for i, v in enumerate(video_names) if v in train_videos_set])
    test_indices  = np.array([i for i, v in enumerate(video_names) if v in test_videos_set])
    unmapped = sum(1 for v in video_names if v not in train_videos_set and v not in test_videos_set)
    if unmapped:
        print(f"[build_trajectory_split] WARNING: {unmapped} trajectories whose video is "
                  f"in neither PersonPath22 train nor test split — they will be excluded")
    return trajectory_paths, train_indices, test_indices


def evaluate_calibrator(trainer, test_indices):
    """Run the trained calibrator on the held-out validation split"""

    val_dataset = deepcopy(trainer.dataset)
    val_dataset.initialize(test_indices)

    predictions = predict_on_dataset(trainer.model, val_dataset)
    stride = val_dataset.epoch_size_divisor
    true_iou = np.asarray([val_dataset._frame_map[i * stride][2] for i in range(len(predictions))], dtype=np.float32)
    true_labels = (true_iou > val_dataset.iou_threshold).astype(np.int32)
    probabilities = 1.0 / (1.0 + np.exp(-predictions[:, 0]))
    predicted_labels = (probabilities > 0.5).astype(np.int32)

    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    auc = roc_auc_score(true_labels, probabilities) if len(set(true_labels)) > 1 else float("nan")
    print(f"[val] F1={f1:.4f}  AUC={auc:.4f}")


def stream_metrics(tracker, trajectory_paths, test_indices, detection_data):
    """Log per-trajectory coverage-AUC to the terminal (AUC over the IoU threshold)."""

    coverages = []
    test_trajectories = [trajectory_paths[i] for i in test_indices]

    pbar = tqdm(test_trajectories, desc="Coverage")
    for trajectory_path in pbar:
        video_name, person_id = Path(trajectory_path).stem.rsplit("_", 1)
        detection_data.initialize_target(video_name, int(person_id))
        predicted_masks = tracker.predict_masks(detection_data).numpy()

        iou_scores = compute_iou(detection_data.bboxes_norm, predicted_masks)
        iou_scores[detection_data.occlusions > 0.5] = 0.0

        coverage = coverage_auc(iou_scores, detection_data.occlusions)
        if not np.isnan(coverage):
            coverages.append(coverage)

        pbar.set_postfix(avg_coverage_auc=(np.mean(coverages) if coverages else float("nan")))

@hydra.main(config_path="conf", config_name="offline_training", version_base=None)
def train_models(config: DictConfig):
    """Train and evaluate the calibrator, then optionally stream tracker coverage."""

    main_trainer = hydra.utils.instantiate(config.offline_trainers.main_trainer)

    trajectory_paths, train_indices, test_indices = build_trajectory_split(
        dataset_path=main_trainer.dataset.dataset_path, test_size=config.train_test_split)

    main_trainer.custom_train(
        x=trajectory_paths,
        y=trajectory_paths,
        train_indices=train_indices,
        validation_indices=test_indices)
    evaluate_calibrator(main_trainer, test_indices)

    if config.deploy_controller:
        detection_data = hydra.utils.instantiate(config.detection_data)
        tracker = hydra.utils.instantiate(config.tracker.tracker)

        if (hasattr(tracker, "model") and tracker.model is not None and hasattr(tracker.model, "controller")):
            tracker.model.controller = main_trainer.model
            tracker.model.eval()
        stream_metrics(tracker, trajectory_paths, test_indices, detection_data)


if __name__ == "__main__":
    train_models()

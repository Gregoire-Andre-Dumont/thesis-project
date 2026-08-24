import logging
import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from src.experiments.dataset_experiment import DatasetExperiment
from src.utils.compute_iou import compute_iou
from src.offline_training.dataset_encoders import (
    load_dataset_encoders, crop_around_masks, encode_trajectory, anchor_size_pixels)
from src.offline_training.dataset_labels import load_clean_boxes_by_frame, pseudo_iou_labels


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

# The four LARGE (~212-304M) backbones the calibrator dataset is built for (one token set each, no variants).
ENCODER_NAMES = ("perception", "dino", "hiera_sam", "hiera_mae")


def shard_by_video(person_path, shard_index, num_shards):
    """Group the selected trajectories by video and keep the ones owned by this shard.
    Each trajectory carries the anchor frame PersonPath pre-selected for it."""

    by_video = defaultdict(list)
    for video_name, person_id, anchor_video_frame in zip(
            person_path.selected_video_names.tolist(),
            person_path.selected_person_ids.tolist(),
            person_path.selected_anchor_video_frames.tolist()):
        by_video[video_name].append((video_name, int(person_id), int(anchor_video_frame)))
    videos = sorted(by_video)[shard_index::num_shards]
    return videos, [pair for video in videos for pair in by_video[video]]


def anchor_trajectory_index(detection_data, anchor_video_frame):
    """Find where the anchor's video frame sits inside the trajectory's frame list.
    Returns its position, or None when the anchor is not present."""

    positions = np.where(detection_data.frame_indices == anchor_video_frame)[0]
    return int(positions[0]) if len(positions) else None


def slice_detection_data_for_tracker(detection_data, anchor_index):
    """Trim the leading frames so the anchor lands where the tracker starts reading.
    Returns how many warmup frames were dropped, either zero or one."""

    warmup_count = 1 if anchor_index >= 1 else 0
    start = anchor_index - warmup_count
    detection_data.frames = detection_data.frames[start:]
    detection_data.bboxes_norm = detection_data.bboxes_norm[start:]
    detection_data.occlusions = detection_data.occlusions[start:]
    detection_data.frame_indices = detection_data.frame_indices[start:]
    return warmup_count


def encoder_path(dataset_path, encoder_name, stem):
    """Output file path for one encoder's features of a trajectory: <dataset_path>/<encoder>/<stem>.pkl."""

    return Path(dataset_path) / encoder_name / f"{stem}.pkl"


def trajectory_is_complete(dataset_path, stem):
    """True when every encoder file for this trajectory already exists (skip on resume)."""

    return all(encoder_path(dataset_path, name, stem).exists() for name in ENCODER_NAMES)


def process_trajectory(tracker, label_model, encoders, detection_data, video_name,
                       person_id, anchor_video_frame, visible_directory, config):
    """Track the trajectory once, then per frame build the four encoders' anchor-similarity maps and
    the target + 3-nearest-distractor pseudo-IoU labels. Returns (metadata, features_by_encoder) or
    None when the anchor is missing."""

    detection_data.initialize_target(video_name, person_id)
    anchor_index = anchor_trajectory_index(detection_data, anchor_video_frame)
    if anchor_index is None:
        return None
    warmup = slice_detection_data_for_tracker(detection_data, anchor_index)

    predicted_masks = tracker.predict_masks(detection_data).numpy()
    box_iou = compute_iou(detection_data.bboxes_norm, predicted_masks)
    box_iou[detection_data.occlusions > 0.5] = 0.0
    predicted_iou = tracker.iou_scores.numpy()

    keep = slice(warmup, None)
    predicted_masks = predicted_masks[keep]
    frames = detection_data.frames[keep]
    occlusions = detection_data.occlusions[keep]
    bboxes = detection_data.bboxes_norm[keep]
    frame_indices = detection_data.frame_indices[keep]
    box_iou = box_iou[keep]
    predicted_iou = predicted_iou[keep]

    # Labels: proposal-mask pseudo-IoU vs the target and its 3 nearest clean distractors (box-prompted).
    clean_boxes = load_clean_boxes_by_frame(Path(visible_directory) / f"{video_name}.json", person_id)
    target_iou, distractor_iou = pseudo_iou_labels(
        label_model, frames, predicted_masks, bboxes, clean_boxes, frame_indices, occlusions)

    # Features: crop once around each proposal (floored at the anchor box size, matching deployment),
    # then re-encode with every backbone at the shared 32x32 grid. frames[0]/bboxes[0] are the anchor.
    size_floor = anchor_size_pixels(bboxes[0], frames[0].shape)
    crop_frames, crop_masks = crop_around_masks(
        frames, predicted_masks, config.crop_resize, config.pad_ratio, size_floor)
    features_by_encoder = encode_trajectory(encoders, crop_frames, crop_masks)

    metadata = {
        "frame_indices": frame_indices.astype(np.int64),
        "iou_scores":    target_iou.astype(np.float32),        # target pseudo-GT mask IoU (main label)
        "distractor_iou": distractor_iou.astype(np.float32),   # (n, 3) nearest-distractor pseudo IoU
        "box_iou":       box_iou.astype(np.float32),
        "occlusions":    occlusions.astype(np.float32),
        "predicted_iou": predicted_iou.astype(np.float32),
        "true_bboxes":   bboxes.astype(np.float32),
    }
    return metadata, features_by_encoder


def save_encoder_features(dataset_path, stem, video_name, person_id, metadata, features_by_encoder):
    """Write one DatasetExperiment per encoder; existing files are left untouched."""

    for encoder_name, features in features_by_encoder.items():
        output_path = encoder_path(dataset_path, encoder_name, stem)
        if output_path.exists():
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = DatasetExperiment(video_name=video_name, person_id=person_id, features=features, **metadata)
        output_path.write_bytes(pickle.dumps(dataset))


@hydra.main(config_path="conf", config_name="create_anchor_dataset", version_base=None)
def create_anchor_dataset(config: DictConfig):
    """Build one calibrator dataset per backbone from a single tracking pass (oracle or baseline).
    Each encoder re-encodes the shared proposal masks at the fixed 32x32 grid; labels are the target
    and 3-nearest-distractor pseudo-IoU, box-prompted with the tracker's SAM model."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    tracker = hydra.utils.instantiate(config.tracker.tracker)
    encoders = load_dataset_encoders()

    dataset_path = config.dataset_path
    visible_directory = config.detection_data.visible_directory
    _, pairs = shard_by_video(person_path, config.shard_index, config.num_shards)

    for video_name, person_id, anchor_video_frame in tqdm(pairs, desc=f"shard {config.shard_index}"):
        stem = f"{video_name}_{person_id}"
        if trajectory_is_complete(dataset_path, stem):
            continue
        result = process_trajectory(tracker, tracker.model, encoders, detection_data,
                                    video_name, person_id, anchor_video_frame, visible_directory, config)
        if result is not None:
            metadata, features_by_encoder = result
            save_encoder_features(dataset_path, stem, video_name, person_id, metadata, features_by_encoder)


if __name__ == "__main__":
    create_anchor_dataset()

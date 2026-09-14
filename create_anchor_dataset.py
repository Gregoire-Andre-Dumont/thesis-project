import logging
import os
import pickle
import warnings
import zlib
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from src.experiments.dataset_experiment import DatasetExperiment
from src.utils.compute_iou import compute_iou
from src.offline_training.dataset_encoders import (
    load_dataset_encoders, crop_around_masks, anchor_size_pixels,
    encode_tokens, anchor_foreground, similarity_feature_map)
from src.offline_training.dataset_labels import load_clean_boxes_by_frame, pseudo_iou_labels
from src.utils.corruption import nearest_distractor_boxes


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

ENCODER_NAME = "perception"


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


def trajectory_path(dataset_path, corruption_p, stem):
    """Output file for one trajectory at one corruption probability:
    <dataset_path>/p<probability>/<stem>.pkl. Its existence is what makes the run resumable.

    One folder per probability rather than one file per trajectory: the corruption changes the tracking
    rollout itself, so every probability produces different proposal masks, different features and
    different labels. They are separate datasets, not separate columns of one."""

    return Path(dataset_path) / f"p{float(corruption_p):.2f}" / f"{stem}.pkl"


def rollout_features(tracker, label_model, token_fn, detection_data, clean_boxes, warmup, config):
    """One tracking pass at the tracker's current corruption setting -> (metadata, features).

    Everything downstream of the rollout depends on the proposal masks it produced, so labels and features
    are both recomputed per pass -- that is the whole point of sweeping the corruption probability."""

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
    target_iou, distractor_iou = pseudo_iou_labels(
        label_model, frames, predicted_masks, bboxes, clean_boxes, frame_indices, occlusions)

    # Features: crop once around each proposal (floored at the anchor box size, matching deployment), then
    size_floor = anchor_size_pixels(bboxes[0], frames[0].shape)
    crop_frames, crop_masks = crop_around_masks(frames, predicted_masks, config.crop_resize, config.pad_ratio, size_floor)
    tokens = encode_tokens(token_fn, crop_frames)
    features = similarity_feature_map(tokens, crop_masks, anchor_foreground(tokens, crop_masks))

    metadata = {
        "frame_indices": frame_indices.astype(np.int64),
        "iou_scores":    target_iou.astype(np.float32),        # target pseudo-GT mask IoU (main label)
        "distractor_iou": distractor_iou.astype(np.float32),   # (n, 3) nearest-distractor pseudo IoU
        "box_iou":       box_iou.astype(np.float32),
        "occlusions":    occlusions.astype(np.float32),
        "predicted_iou": predicted_iou.astype(np.float32),
        "true_bboxes":   bboxes.astype(np.float32),
    }
    return metadata, features


def process_trajectory(tracker, label_model, token_fn, detection_data, video_name, person_id,
                       anchor_video_frame, visible_directory, config, probabilities):
    """Roll the trajectory out once per corruption probability. Returns {probability: (metadata, features)},
    or None when the anchor is missing or the clip has no distractor to inject.

    The frame cache is shared across the sweep, so SAM's image embeddings are computed once for the clip
    however many probabilities are swept -- only the memory bank and what follows from it differ. The
    corruption seed is derived from the clip id so the injected identities are identical across launches
    and across probabilities, which makes the sweep a nested series rather than independent draws."""

    detection_data.initialize_target(video_name, person_id)
    anchor_index = anchor_trajectory_index(detection_data, anchor_video_frame)
    if anchor_index is None:
        return None
    warmup = slice_detection_data_for_tracker(detection_data, anchor_index)

    clean_boxes = load_clean_boxes_by_frame(Path(visible_directory) / f"{video_name}.json", person_id)
    corruption_boxes = nearest_distractor_boxes(detection_data, clean_boxes)
    if not any(box is not None for box in corruption_boxes):
        return None                                    # no distractor anywhere: nothing to corrupt

    tracker.frame_cache = {}
    tracker.corruption_boxes = corruption_boxes
    tracker.corruption_seed = zlib.crc32(f"{video_name}:{person_id}".encode())
    try:
        by_probability = {}
        for probability in probabilities:
            tracker.corruption_p = float(probability)
            by_probability[probability] = rollout_features(
                tracker, label_model, token_fn, detection_data, clean_boxes, warmup, config)
        return by_probability
    finally:
        tracker.frame_cache = None
        tracker.corruption_boxes = None
        tracker.corruption_p = 0.0


@hydra.main(config_path="conf", config_name="create_anchor_dataset", version_base=None)
def create_anchor_dataset(config: DictConfig):
    """Build one calibrator dataset per memory-corruption probability, from the mask oracle.

    The mask oracle picks the best available proposal every frame, so without corruption its memory bank is
    close to clean -- which is exactly the regime where a calibrator sees no negatives to learn from.
    Sweeping `corruption_probabilities` injects a controlled rate of clean nearby distractors into the bank
    (identity errors, not quality errors), giving one dataset per rate. Every proposal mask is cropped and
    encoded to its anchor-similarity map; labels are the target and 3-nearest-distractor pseudo-IoU,
    box-prompted with the tracker's SAM model."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    tracker = hydra.utils.instantiate(config.tracker.tracker)
    token_fn = load_dataset_encoders([ENCODER_NAME])[ENCODER_NAME]
    probabilities = [float(p) for p in config.corruption_probabilities]

    dataset_path = config.dataset_path
    visible_directory = config.detection_data.visible_directory
    _, pairs = shard_by_video(person_path, config.shard_index, config.num_shards)

    for video_name, person_id, anchor_video_frame in tqdm(pairs, desc=f"shard {config.shard_index}"):
        stem = f"{video_name}_{person_id}"
        missing = [p for p in probabilities if not trajectory_path(dataset_path, p, stem).exists()]
        if not missing:
            continue

        # Swept together even when only some probabilities are missing: the rollouts share one frame cache,
        results = process_trajectory(tracker, tracker.model, token_fn, detection_data, video_name,
                    person_id, anchor_video_frame, visible_directory, config, missing)
        
        if results is None:
            continue

        for probability, (metadata, features) in results.items():
            output_path = trajectory_path(dataset_path, probability, stem)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataset = DatasetExperiment(video_name=video_name, person_id=person_id, features=features, **metadata)
            output_path.write_bytes(pickle.dumps(dataset))


if __name__ == "__main__":
    create_anchor_dataset()

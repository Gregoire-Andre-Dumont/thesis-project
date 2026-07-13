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
from src.offline_training.references import (
    VARIANTS,
    encode_with_encoders,
    load_encoder_models,
    predict_and_filter_trajectory,
)


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"


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


def variant_path(dataset_path, encoder_name, variant_name, stem):
    """Build the output file path for one encoder and variant of a trajectory.
    The folder name is the dataset name with the encoder appended."""

    return Path(f"{dataset_path}_{encoder_name}") / variant_name / f"{stem}.pkl"


def trajectory_is_complete(dataset_path, encoders, stem):
    """Check whether every encoder and variant file for this trajectory already exists.
    Used to skip trajectories that were fully written on a previous run."""

    return all(variant_path(dataset_path, encoder, variant, stem).exists()
               for encoder in encoders for variant in VARIANTS)


def track_trajectory(tracker, detection_data, video_name, person_id, anchor_video_frame, config):
    """Initialize the target, find its anchor, and run the oracle tracker once.
    Returns the tracking result, or None when the trajectory has no anchor or fails the filter."""

    detection_data.initialize_target(video_name, person_id)
    anchor_index = anchor_trajectory_index(detection_data, anchor_video_frame)
    if anchor_index is None:
        return None

    warmup_count = slice_detection_data_for_tracker(detection_data, anchor_index)
    return predict_and_filter_trajectory(
        tracker, detection_data, warmup_count,
        coverage_threshold=config.coverage_threshold,
        commit_threshold=config.commit_threshold)


def save_encoder_features(dataset_path, stem, video_name, person_id, metadata_kwargs, features_by_encoder):
    """Write each encoder and variant feature set to disk.
    Files that already exist are left untouched."""

    for encoder_name, variants in features_by_encoder.items():
        for variant_name, features in variants.items():
            output_path = variant_path(dataset_path, encoder_name, variant_name, stem)
            if output_path.exists():
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataset = DatasetExperiment(
                video_name=video_name, 
                person_id=person_id,
                features=features, 
                **metadata_kwargs)
            output_path.write_bytes(pickle.dumps(dataset))


@hydra.main(config_path="conf", config_name="create_anchor_dataset", version_base=None)
def create_anchor_dataset(config: DictConfig):
    """Build one calibrator dataset per backbone encoder from a single oracle tracking pass.
    Each encoder re-encodes the shared masks at the fixed crop size and padding."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    tracker = hydra.utils.instantiate(config.oracle.tracker)
    encoder_models = load_encoder_models(config.encoders, config.crop_resize, config.pad_ratio)

    dataset_path = config.dataset_path
    _, pairs = shard_by_video(person_path, config.shard_index, config.num_shards)

    for video_name, person_id, anchor_video_frame in tqdm(pairs, desc=f"shard {config.shard_index}"):
        stem = f"{video_name}_{person_id}"
        if trajectory_is_complete(dataset_path, encoder_models, stem):
            continue

        # Track ONCE with the oracle, then re-encode the shared masks with every backbone.
        result = track_trajectory(tracker, detection_data, video_name, person_id, anchor_video_frame, config)
        if result is not None:
            metadata_kwargs, frames, predicted_masks = result
            features_by_encoder = encode_with_encoders(encoder_models, frames, predicted_masks)
            save_encoder_features(dataset_path, stem, video_name, person_id, metadata_kwargs, features_by_encoder)


if __name__ == "__main__":
    create_anchor_dataset()

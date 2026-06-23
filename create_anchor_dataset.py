import os
import logging
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from src.experiments.dataset_experiment import DatasetExperiment
from src.offline_training.references import build_reference_features
from src.typing.setup_wandb import setup_wandb
from src.utils.compute_iou import compute_iou


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"


def shard_by_video(person_path, shard_index, num_shards):
    """Group selected `(video, person_id, anchor_video_frame)` triples by video, keep
    videos owned by this shard. The anchor frame is the one PersonPath pre-selected via
    its visible/nearest-amodal ratio test."""

    by_video = defaultdict(list)
    for video_name, person_id, anchor_video_frame in zip(
            person_path.selected_video_names.tolist(),
            person_path.selected_person_ids.tolist(),
            person_path.selected_anchor_video_frames.tolist()):
        by_video[video_name].append((video_name, int(person_id), int(anchor_video_frame)))
    videos = sorted(by_video)[shard_index::num_shards]
    return videos, [pair for video in videos for pair in by_video[video]]


def post_occlusion_coverage(iou_scores, occlusions, threshold):
    """Fraction of visible post-first-occlusion frames with IoU > `threshold`."""

    occluded = np.where(occlusions > 0.5)[0]
    visible = np.where(occlusions < 0.5)[0]
    if len(occluded) == 0:
        return 0.0
    post = visible[visible > occluded[0]]
    return (iou_scores[post] > threshold).mean() if len(post) else 0.0


def anchor_trajectory_index(detection_data, anchor_video_frame):
    """Map the anchor's *video* frame index (selected by `PersonPath`) to its index in
    the trajectory's `frame_indices` array. Returns `None` only if `anchor_video_frame`
    is not present in `frame_indices` at all (shouldn't happen since PersonPath selects
    from annotated frames)."""

    positions = np.where(detection_data.frame_indices == anchor_video_frame)[0]
    return int(positions[0]) if len(positions) else None


def slice_detection_data_for_tracker(detection_data, anchor_index):
    """Slice every per-frame array so the anchor lands at sliced index 1 — where
    `MainMemory.initialize_references` reads from. When `anchor_index == 0` (anchor is
    the first annotated frame), there's nothing to slice back to, so the sliced data is
    unchanged; the tracker then initializes from `frames[1]` = the next annotated frame
    rather than the anchor itself. Returns the number of pre-anchor warmup frames that
    must be dropped from every per-frame array after tracking (0 or 1) so that the
    SAVED trajectory's frame 0 is always the anchor."""

    warmup_count = 1 if anchor_index >= 1 else 0
    start = anchor_index - warmup_count
    detection_data.frames = detection_data.frames[start:]
    detection_data.bboxes_norm = detection_data.bboxes_norm[start:]
    detection_data.amodal_norm = detection_data.amodal_norm[start:]
    detection_data.occlusions = detection_data.occlusions[start:]
    detection_data.frame_indices = detection_data.frame_indices[start:]
    return warmup_count


@hydra.main(config_path="conf", config_name="create_anchor_dataset", version_base=None)
def create_anchor_dataset(config: DictConfig):
    """Build two parallel per-trajectory datasets of precomputed patch-similarity
    features, anchored on the first fully-visible frame and cropped at a fixed amodal-
    floor square. Both datasets share the same anchor, tracker prediction, and crops;
    they differ only in which encoder produces the patch tokens:
    `SamaraHieraModel.extract_raw_patch_tokens` (Hiera image tokens) → `hiera_dataset_path`."""

    
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    tracker = hydra.utils.instantiate(config.oracle.tracker)
    model = tracker.model

    output_directory = Path(config.hiera_dataset_path) / "clean"
    output_directory.mkdir(parents=True, exist_ok=True)

    videos, pairs = shard_by_video(person_path, config.shard_index, config.num_shards)
    print(f"[create_anchor_dataset] shard {config.shard_index}/{config.num_shards} → "
          f"{len(videos)} videos, {len(pairs)} trajectories")

    skipped_already_exists = 0
    skipped_no_anchor = 0
    skipped_coverage = 0
    saved = 0

    for video_name, person_id, anchor_video_frame in tqdm(pairs, desc=f"shard {config.shard_index}"):
        stem = f"{video_name}_{person_id}"
        output_path = output_directory / f"{stem}.pkl"
        if output_path.exists():
            skipped_already_exists += 1
            continue

        detection_data.initialize_target(video_name, person_id)
        anchor_index = anchor_trajectory_index(detection_data, anchor_video_frame)
        if anchor_index is None:
            skipped_no_anchor += 1
            continue

        warmup_count = slice_detection_data_for_tracker(detection_data, anchor_index)

        # Configure the model's fixed crop side from the ANCHOR's amodal bbox. The
        # anchor sits at `warmup_count` in the sliced detection_data (index 1 when we
        # have a warmup, index 0 when the anchor is the first annotated frame).
        anchor_slot = warmup_count
        anchor_frame_height, anchor_frame_width = detection_data.frames[anchor_slot].shape[:2]
        model.set_amodal_anchor(detection_data.amodal_norm[anchor_slot],
                                  frame_height=anchor_frame_height,
                                  frame_width=anchor_frame_width)

        predicted_masks = tracker.predict_masks(detection_data).numpy()
        iou_scores = compute_iou(detection_data.bboxes_norm, predicted_masks)
        iou_scores[detection_data.occlusions > 0.5] = 0.0

        # Drop the pre-anchor warmup frame(s) so saved frame 0 IS the anchor.
        predicted_masks = predicted_masks[warmup_count:]
        iou_scores = iou_scores[warmup_count:]
        occlusions = detection_data.occlusions[warmup_count:]
        bboxes_norm = detection_data.bboxes_norm[warmup_count:]
        frames = detection_data.frames[warmup_count:]
        frame_indices = detection_data.frame_indices[warmup_count:]
        tracker_iou_scores = tracker.iou_scores.numpy()[warmup_count:]

        coverage = post_occlusion_coverage(iou_scores, occlusions, config.commit_threshold)
        if coverage < config.coverage_threshold:
            skipped_coverage += 1
            continue

        cropped_frames, cropped_masks = model.extract_crops(frames, predicted_masks)
        hiera_tokens, hiera_patch_masks = model.extract_raw_patch_tokens(cropped_frames, cropped_masks)
        foreground, background = model.split_foreground_background(hiera_tokens, hiera_patch_masks)

        features, fifo_ious = build_reference_features(
            model=model,
            foreground=foreground,
            background=background,
            iou_scores=iou_scores,
            occlusions=occlusions,
            n_references=config.n_references,
            commit_threshold=config.commit_threshold)

        metadata = DatasetExperiment(
            video_name=video_name,
            person_id=person_id,
            frame_indices=frame_indices.astype(np.int64),
            iou_scores=iou_scores.astype(np.float32),
            occlusions=occlusions.astype(np.float32),
            predicted_iou=tracker_iou_scores.astype(np.float32),
            true_bboxes=bboxes_norm.astype(np.float32),
            features=features,
            fifo_ious=fifo_ious)
        with open(output_path, "wb") as handle:
            pickle.dump(metadata, handle)
        saved += 1

    total = len(pairs)
    print(f"[create_anchor_dataset] total={total}  saved={saved}  "
          f"already_exists={skipped_already_exists}  "
          f"no_anchor={skipped_no_anchor}  "
          f"coverage_filter={skipped_coverage}")


if __name__ == "__main__":
    create_anchor_dataset()

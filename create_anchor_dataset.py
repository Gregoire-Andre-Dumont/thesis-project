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
from src.offline_training.dataset_encoders import load_dataset_encoders, anchor_size_pixels, proposal_feature_maps
from src.offline_training.dataset_labels import load_clean_boxes_by_frame, distractor_boxes_per_frame, pseudo_iou_labels
from src.utils.corruption import nearest_distractor_boxes


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

ENCODER_NAME = "perception"


def shard_by_video(person_path, shard_index, num_shards):
    """Group the selected trajectories by video and keep the ones owned by this shard.
    Each trajectory carries the anchor frame PersonPath pre-selected for it."""

    video_names = person_path.selected_video_names.tolist()
    person_ids = person_path.selected_person_ids.tolist()
    anchor_frames = person_path.selected_anchor_video_frames.tolist()

    by_video = defaultdict(list)
    for video_name, person_id, anchor_video_frame in zip(video_names, person_ids, anchor_frames):
        by_video[video_name].append((video_name, int(person_id), int(anchor_video_frame)))
    videos = sorted(by_video)[shard_index::num_shards]
    return videos, [pair for video in videos for pair in by_video[video]]


def anchor_trajectory_index(detection_data, anchor_video_frame):
    """Find where the anchor's video frame sits inside the trajectory's frame list.
    Returns its position, or None when the anchor is not present."""

    positions = np.where(detection_data.frame_indices == anchor_video_frame)[0]
    return int(positions[0]) if len(positions) else None


def load_window(detection_data, video_name, person_id, anchor_video_frame, max_frames):
    """Load only the frames the tracker needs -- `max_frames` starting AT the anchor, plus one warmup frame
    before it. Returns the warmup count, or None when the anchor is not in the trajectory.

    The annotations are read first with `load_frames` off so the window can be chosen without decoding
    anything; only then are those frames decoded. Trimming after a full load would not help: one trajectory
    holds its decoded frames plus the SAM encodings cached across all five corruption levels, and on a long
    clip that alone exceeds the container's memory."""

    detection_data.load_frames = False
    detection_data.initialize_target(video_name, person_id)
    anchor_index = anchor_trajectory_index(detection_data, anchor_video_frame)
    if anchor_index is None:
        return None

    warmup_count = 1 if anchor_index >= 1 else 0
    start = anchor_index - warmup_count
    window = detection_data.frame_indices[start:start + warmup_count + max_frames]

    detection_data.load_frames = True
    detection_data.initialize_target(video_name, person_id, frame_indices=window)
    return warmup_count


def trajectory_path(dataset_path, corruption_p, stem):
    """Output file for one trajectory at one corruption probability:
    <dataset_path>/p<probability>/<stem>.pkl. Its existence is what makes the run resumable.

    One folder per probability rather than one file per trajectory: the corruption changes the tracking
    rollout itself, so every probability produces different proposal masks, different features and
    different labels. They are separate datasets, not separate columns of one."""

    return Path(dataset_path) / f"p{float(corruption_p):.2f}" / f"{stem}.pkl"


def rollout_features(tracker, label_model, token_fn, detection_data, clean_boxes, warmup, config, truth_cache=None):
    """One tracking pass at the tracker's current corruption setting -> (metadata, features).

    Everything downstream of the rollout depends on the proposal masks it produced, so labels and features
    are both recomputed per pass -- that is the whole point of sweeping the corruption probability.

    All THREE of SAM's proposals are labelled and encoded, not just the one the arm tracked with. Selecting
    among them is a decision the tracker makes every frame and the thing a calibrator is meant to improve,
    so the alternatives have to be in the dataset -- and they only exist during the rollout. Labelling all
    three is free (a box's pseudo-GT does not depend on which proposal scores against it, so each box is
    prompted once and IoU'd three times); encoding all three costs three passes, since each is cropped
    around its own mask exactly as a candidate would be at deployment."""

    predicted_masks = tracker.predict_masks(detection_data).numpy()
    box_iou = compute_iou(detection_data.bboxes_norm, predicted_masks)
    box_iou[detection_data.occlusions > 0.5] = 0.0
    predicted_iou = tracker.iou_scores.numpy()          # chosen proposal's IoU token
    object_score = tracker.object_scores.numpy()        # raw pre-sigmoid presence logit
    proposal_iou_scores = tracker.proposal_iou_scores.numpy()   # (n, 3) token per proposal
    proposal_true_iou = tracker.proposal_true_iou.numpy()       # (n, 3) box IoU per proposal
    proposal_masks = tracker.proposal_masks.float().numpy()     # (n, 3, 256, 256) mask logits
    chosen_index = tracker.chosen_index.numpy()                 # (n,) which proposal the arm tracked with

    keep = slice(warmup, None)
    frames = detection_data.frames[keep]
    occlusions = detection_data.occlusions[keep]
    bboxes = detection_data.bboxes_norm[keep]
    frame_indices = detection_data.frame_indices[keep]
    box_iou = box_iou[keep]
    predicted_iou = predicted_iou[keep]
    object_score = object_score[keep]
    proposal_iou_scores = proposal_iou_scores[keep]
    proposal_true_iou = proposal_true_iou[keep]
    proposal_masks = proposal_masks[keep]
    chosen_index = chosen_index[keep]

    # Labels: per-proposal pseudo-IoU vs the target and its 3 nearest clean distractors (box-prompted).
    cache = getattr(tracker, "frame_cache", None)
    precomputed = None
    if cache is not None and all(t + warmup in cache for t in range(len(frames))):
        precomputed = {t: cache[t + warmup] for t in range(len(frames))}

    distractor_boxes = distractor_boxes_per_frame(clean_boxes, frame_indices)
    label_arguments = (label_model, frames, proposal_masks, bboxes, distractor_boxes, occlusions)
    target_iou, distractor_iou = pseudo_iou_labels(*label_arguments, precomputed_features=precomputed, truth_cache=truth_cache)

    # Features: each proposal cropped around its OWN mask, floored at the anchor box size as at deployment.
    # The reference is the proposal the arm tracked with on frame 0 -- what the memory bank was seeded from.
    size_floor = anchor_size_pixels(bboxes[0], frames[0].shape)
    anchor_proposal = int(chosen_index[0])
    features = proposal_feature_maps(token_fn, frames, proposal_masks, config.crop_resize, config.pad_ratio, size_floor, anchor_proposal)

    metadata = {
        "frame_indices": frame_indices.astype(np.int64),
        "iou_scores":    target_iou.astype(np.float32),        # (n, 3) target pseudo-GT mask IoU -- the label
        "distractor_iou": distractor_iou.astype(np.float32),   # (n, 3, 3) proposal x nearest-distractor IoU
        "box_iou":       box_iou.astype(np.float32),           # chosen proposal, vs the GT box
        "occlusions":    occlusions.astype(np.float32),
        "predicted_iou": predicted_iou.astype(np.float32),     # SAM's IoU token for the CHOSEN proposal
        "object_score":  object_score.astype(np.float32),      # SAM's raw presence logit (signed)
        "proposal_iou_scores": proposal_iou_scores.astype(np.float32),   # (n, 3) token per proposal
        "proposal_true_iou":   proposal_true_iou.astype(np.float32),     # (n, 3) box IoU per proposal
        "chosen_index":  chosen_index.astype(np.int64),        # (n,) proposal the arm tracked with, 0-2
        "true_bboxes":   bboxes.astype(np.float32),
    }
    return metadata, features


def process_trajectory(tracker, label_model, token_fn, detection_data, video_name, person_id, anchor_video_frame, visible_directory, config, probabilities):
    """Roll the trajectory out once per corruption probability. Returns {probability: (metadata, features)},
    or None when the anchor is missing.

    The frame cache is shared across the sweep, so SAM's image embeddings are computed once for the clip
    however many probabilities are swept -- only the memory bank and what follows from it differ. The
    corruption seed is derived from the clip id so the corrupted frames are identical across launches and
    nested across probabilities, which makes the sweep a series rather than independent draws."""

    warmup = load_window(detection_data, video_name, person_id, anchor_video_frame, config.max_frames)
    if warmup is None:
        return None

    # The same clean-person pool serves both roles: the corruption injects the nearest of them into the bank,
    visible_path = Path(visible_directory) / f"{video_name}.json"
    non_targets = tuple(config.person_path.non_targets)
    clean_boxes = load_clean_boxes_by_frame(visible_path, person_id, non_targets)
    corruption_boxes = nearest_distractor_boxes(detection_data, clean_boxes)
    if not any(box is not None for box in corruption_boxes):
        return None                                    # no distractor anywhere: nothing to corrupt

    tracker.frame_cache = {}
    tracker.corruption_boxes = corruption_boxes
    tracker.corruption_seed = zlib.crc32(f"{video_name}:{person_id}".encode())
    # A box's pseudo-GT depends on the frame and the box, neither of which the corruption changes, so the
    # sweep decodes each one once instead of once per probability.
    truth_cache = {}
    try:
        by_probability = {}
        for probability in probabilities:
            tracker.corruption_p = float(probability)
            rollout = rollout_features(tracker, label_model, token_fn, detection_data, clean_boxes, warmup, config, truth_cache)
            by_probability[probability] = rollout
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
    (identity errors, not quality errors), giving one dataset per rate.

    Each frame contributes all THREE of SAM's proposals, not just the one the arm tracked with: every
    proposal is cropped and encoded to its anchor-similarity map, and labelled with the target and
    3-nearest-distractor pseudo-IoU, box-prompted with the tracker's SAM model. That makes the dataset
    support two questions rather than one -- how good is this mask (calibration, across frames) and which
    of these three is best (selection, within a frame)."""

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

        # Swept together even when only some probabilities are missing: the rollouts share one frame cache.
        trajectory = (video_name, person_id, anchor_video_frame)
        results = process_trajectory(tracker, tracker.model, token_fn, detection_data, *trajectory, visible_directory, config, missing)
        if results is None:
            continue

        for probability, (metadata, features) in results.items():
            output_path = trajectory_path(dataset_path, probability, stem)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataset = DatasetExperiment(video_name=video_name, person_id=person_id, features=features, **metadata)
            output_path.write_bytes(pickle.dumps(dataset))


if __name__ == "__main__":
    create_anchor_dataset()

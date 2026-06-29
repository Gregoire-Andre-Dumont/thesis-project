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
    compute_features,
    extract_tokens_at_pad_ratio,
    predict_and_filter_trajectory,
)
from src.typing.setup_wandb import setup_wandb


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"


# Token-source variants. Patch similarities use MAX reduction over the anchor's FG
# patches; the per-pad_ratio sweep is configured in `create_anchor_dataset.yaml`.
VARIANTS = [
    ("hiera",  "hiera"),
    ("memory", "memory"),
]


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


def anchor_trajectory_index(detection_data, anchor_video_frame):
    """Map the anchor's video-frame index to its index in the trajectory's
    `frame_indices` array. Returns `None` only if the anchor isn't in the array."""

    positions = np.where(detection_data.frame_indices == anchor_video_frame)[0]
    return int(positions[0]) if len(positions) else None


def slice_detection_data_for_tracker(detection_data, anchor_index):
    """Slice every per-frame array so the anchor lands at sliced index 1 — where
    `MainMemory.initialize_references` reads from. Returns the number of pre-anchor
    warmup frames that must be dropped post-tracking (0 or 1)."""

    warmup_count = 1 if anchor_index >= 1 else 0
    start = anchor_index - warmup_count
    detection_data.frames = detection_data.frames[start:]
    detection_data.bboxes_norm = detection_data.bboxes_norm[start:]
    detection_data.amodal_norm = detection_data.amodal_norm[start:]
    detection_data.occlusions = detection_data.occlusions[start:]
    detection_data.frame_indices = detection_data.frame_indices[start:]
    return warmup_count


def pad_ratio_label(pad_ratio):
    """Folder-friendly label for a pad ratio (e.g. `0.5` → `padding_0.50`)."""

    return f"padding_{float(pad_ratio):.2f}"


def variant_path(base_directory, pad_ratio, variant_name, stem):
    return base_directory / variant_name / pad_ratio_label(pad_ratio) / f"{stem}.pkl"


def save_variants_for_pad_ratio(model, base_directory, pad_ratio, stem, video_name, person_id,
                                            metadata_kwargs, hiera_pair, memory_pair):
    """Compute features for each token-source variant at this pad_ratio and pickle into
    `<base>/<variant>/padding_<X.XX>/<stem>.pkl`. Returns the number of variants
    actually written (existing files are skipped). Each variant's anchor FG and BG
    patches are sliced from the first frame of the (foreground, background) pair —
    the BG slice powers the new diff channels in `compute_features`."""

    hiera_anchor_fg = hiera_pair[0][0:1]
    hiera_anchor_bg = hiera_pair[1][0:1]
    memory_anchor_fg = memory_pair[0][0:1]
    memory_anchor_bg = memory_pair[1][0:1]
    sources = {
        "hiera":  (hiera_pair,  hiera_anchor_fg,  hiera_anchor_bg),
        "memory": (memory_pair, memory_anchor_fg, memory_anchor_bg),
    }
    saved_here = 0
    for variant_name, token_source in VARIANTS:
        output_path = variant_path(base_directory, pad_ratio, variant_name, stem)
        if output_path.exists():
            continue
        (foreground, background), anchor_fg, anchor_bg = sources[token_source]
        features = compute_features(model, foreground, background, anchor_fg, anchor_bg)
        with open(output_path, "wb") as handle:
            pickle.dump(DatasetExperiment(video_name=video_name, person_id=person_id,
                                                      features=features, **metadata_kwargs), handle)
        saved_here += 1
    return saved_here


@hydra.main(config_path="conf", config_name="create_anchor_dataset", version_base=None)
def create_anchor_dataset(config: DictConfig):
    """Build per-trajectory datasets of precomputed patch-similarity features at MULTIPLE
    pad_ratios in a single tracker pass:

        <dataset_path>/<variant>/padding_<X.XX>/<video>_<person_id>.pkl

    where `variant ∈ {hiera, memory}` and `pad_ratio ∈ config.pad_ratios`.

    Efficiency: the expensive SAM 2 tracker forward pass runs ONCE per trajectory
    (via `predict_and_filter_trajectory`); only the crop + encoder pass is repeated
    per pad_ratio."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    tracker = hydra.utils.instantiate(config.oracle.tracker)
    model = tracker.model

    pad_ratios = [float(p) for p in config.pad_ratios]

    base_directory = Path(config.dataset_path)
    for variant_name, _ in VARIANTS:
        for pad_ratio in pad_ratios:
            (base_directory / variant_name / pad_ratio_label(pad_ratio)
             ).mkdir(parents=True, exist_ok=True)

    videos, pairs = shard_by_video(person_path, config.shard_index, config.num_shards)
    print(f"[create_anchor_dataset] shard {config.shard_index}/{config.num_shards} → "
          f"{len(videos)} videos, {len(pairs)} trajectories, pad_ratios={pad_ratios}")

    skipped = defaultdict(int)
    saved = 0

    for video_name, person_id, anchor_video_frame in tqdm(pairs, desc=f"shard {config.shard_index}"):
        stem = f"{video_name}_{person_id}"

        all_done = all(variant_path(base_directory, pad_ratio, variant, stem).exists()
                            for pad_ratio in pad_ratios for variant, _ in VARIANTS)
        if all_done:
            skipped["already_exists"] += 1
            continue

        detection_data.initialize_target(video_name, person_id)
        anchor_index = anchor_trajectory_index(detection_data, anchor_video_frame)
        if anchor_index is None:
            skipped["no_anchor"] += 1
            continue

        warmup_count = slice_detection_data_for_tracker(detection_data, anchor_index)

        # Run the tracker ONCE and reuse its outputs across all pad_ratios.
        trajectory_result = predict_and_filter_trajectory(
            tracker, detection_data, warmup_count,
            coverage_threshold=config.coverage_threshold,
            commit_threshold=config.commit_threshold)
        if trajectory_result is None:
            skipped["coverage_filter"] += 1
            continue
        metadata_kwargs, frames, predicted_masks = trajectory_result

        for pad_ratio in pad_ratios:
            # Skip pad_ratios whose variants are already all on disk for this trajectory.
            pad_done = all(variant_path(base_directory, pad_ratio, variant, stem).exists() for variant, _ in VARIANTS)
            if pad_done:
                continue

            hiera_pair, memory_pair = extract_tokens_at_pad_ratio(
                model, frames, predicted_masks, pad_ratio)
            saved += save_variants_for_pad_ratio(
                model, base_directory, pad_ratio, stem, video_name, person_id,
                metadata_kwargs, hiera_pair, memory_pair)

    total = len(pairs)
    print(f"[create_anchor_dataset] total={total}  saved_files={saved}  "
          f"already_exists={skipped['already_exists']}  "
          f"no_anchor={skipped['no_anchor']}  "
          f"coverage_filter={skipped['coverage_filter']}")


if __name__ == "__main__":
    create_anchor_dataset()

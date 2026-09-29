"""claim_2: how well does each arm survive a CORRUPTED memory bank?

Scored with VOT-LT (`src.metrics.vot_lt`), so each arm is judged on the box AND the confidence it
reports and is swept to its own operating point. Under corruption that matters more than usual: a bank
poisoned with a distractor makes an arm confidently wrong, which a fixed-threshold coverage metric
cannot distinguish from being confidently right."""

import sys
import zlib
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claim_1 import (
    test_trajectories,
    load_window,
    first_occlusion_frame,
    visible_frames,
    frame_record,
    commit_flags,
    confidence_signals,
    load_results)
from src.utils.corruption import nearest_distractor_boxes
from src.offline_training.dataset_labels import load_clean_boxes_by_frame

import pickle

SAM_CONFIG = "conf/trackers/baselines/sam_baseline.yaml"
SUCCESS_IOU = 0.5

BASELINE_ARM, MEMORY_ORACLE_ARM, MASK_ORACLE_ARM = "sam", "memory", "mask"


def save_results(results_path, processed_trajectories, scored_clips, corruption_probabilities, commit_threshold):
    """Checkpoint after every clip, so a killed run resumes instead of restarting."""

    saved_state = {"processed": processed_trajectories, "clips": scored_clips, "corruption_ps": corruption_probabilities, "commit_threshold": float(commit_threshold)}
    pickle.dump(saved_state, open(results_path, "wb"))


def clip_metadata(video_name, person_id, occlusion_series, ground_truth_boxes, first_occlusion, corruption_boxes):
    """The clip's identity and occlusion structure, before any arm has been rolled over it."""

    scored_span = slice(first_occlusion, len(occlusion_series))
    number_of_corruptible_frames = sum(box is not None for box in corruption_boxes)

    return {
        "video": video_name,
        "person": person_id,
        "n_frames": int(len(occlusion_series)),
        "first_occlusion": int(first_occlusion),
        "occ_count": int((occlusion_series >= 0.5).sum()),
        "occlusions": np.asarray(occlusion_series, dtype=np.float32),
        "occluded": np.asarray(occlusion_series[scored_span] >= 0.5, dtype=bool),
        "has_box": np.asarray(ground_truth_boxes[scored_span][:, 2] > 0, dtype=bool),
        "corruptible": int(number_of_corruptible_frames)}


def report_clip(record, arm_names, corruption_probabilities):
    """One line per clip: each arm's held fraction clean, then at the heaviest corruption rate."""

    visible_and_annotated = record["has_box"] & ~record["occluded"]
    lowest_probability = corruption_probabilities[0]
    highest_probability = corruption_probabilities[-1]

    def held_fraction(arm_ious):
        return (
            float((arm_ious[visible_and_annotated] >= SUCCESS_IOU).mean())
            if visible_and_annotated.any()
            else float("nan")
        )

    arm_summaries = []
    for arm_name in arm_names:
        clean_held = held_fraction(record[(arm_name, lowest_probability)])
        corrupted_held = held_fraction(record[(arm_name, highest_probability)])
        corrupted_writes = record[(arm_name, highest_probability, "corrupted")]
        arm_summaries.append(f"{arm_name} {clean_held:.3f}->{corrupted_held:.3f} ({corrupted_writes:2d} bad)")



@hydra.main(config_path="../conf", config_name="experiments/claim_2", version_base=None)
def run(config: DictConfig):
    """Roll every arm over every clip once per corruption rate, sharing one image-embedding cache."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)

    memory_oracle_tracker = hydra.utils.instantiate(OmegaConf.load(config.oracle_config).tracker)
    mask_oracle_tracker = hydra.utils.instantiate(OmegaConf.load(config.mask_oracle_config).tracker)
    baseline_tracker = hydra.utils.instantiate(OmegaConf.load(SAM_CONFIG).tracker)
    baseline_tracker.label_mask_iou = False

    commit_threshold = float(config.commit_threshold)
    memory_oracle_tracker.iou_threshold = commit_threshold
    mask_oracle_tracker.iou_threshold = commit_threshold

    trackers_by_arm = {BASELINE_ARM: baseline_tracker, MEMORY_ORACLE_ARM: memory_oracle_tracker, MASK_ORACLE_ARM: mask_oracle_tracker}
    corruption_probabilities = [float(probability) for probability in config.corruption_ps]

    output_directory = Path(config.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = output_directory / "results.pkl"
    processed_trajectories, scored_clips = load_results(results_path)

    for trajectory in test_trajectories(person_path, config.n_traj):
        video_name, person_id = trajectory[0], trajectory[1]
        if (video_name, person_id) in processed_trajectories:
            continue
        processed_trajectories.add((video_name, person_id))

        loaded_window = load_window(detection_data, trajectory[:3], config.max_frames)
        if loaded_window is None:
            continue

        occlusion_series = detection_data.occlusions
        ground_truth_boxes = detection_data.bboxes_norm
        first_occlusion = first_occlusion_frame(occlusion_series)
        if not visible_frames(occlusion_series, ground_truth_boxes, first_occlusion):
            continue

        visible_annotations_path = f"{config.detection_data.visible_directory}/{video_name}.json"
        clean_boxes_by_frame = load_clean_boxes_by_frame(visible_annotations_path, int(person_id))
        corruption_boxes = nearest_distractor_boxes(detection_data, clean_boxes_by_frame)
        if not any(box is not None for box in corruption_boxes):
            continue

        shared_frame_cache = {}
        clip_seed = zlib.crc32(f"{video_name}:{int(person_id)}".encode())
        for tracker in trackers_by_arm.values():
            tracker.frame_cache = shared_frame_cache
            tracker.corruption_boxes = corruption_boxes
            tracker.corruption_seed = clip_seed

        record = clip_metadata(video_name, person_id, occlusion_series, ground_truth_boxes, first_occlusion, corruption_boxes)

        for arm_name, tracker in trackers_by_arm.items():
            for corruption_probability in corruption_probabilities:
                tracker.corruption_p = corruption_probability

                predicted_masks = tracker.predict_masks(detection_data).numpy()
                box_ious = frame_record(predicted_masks, occlusion_series, ground_truth_boxes, first_occlusion)
                per_frame_commits = commit_flags(tracker, predicted_masks.shape[0])[first_occlusion:]
                corrupted_frames = getattr(tracker, "corrupted_frames", [])

                confidence = confidence_signals(tracker, predicted_masks.shape[0])

                record[(arm_name, corruption_probability)] = box_ious
                record[(arm_name, corruption_probability, "commit")] = per_frame_commits
                record[(arm_name, corruption_probability, "corrupted")] = len(corrupted_frames)
                # The confidence VOT-LT sweeps, per arm AND per corruption rate: poisoning the bank changes
                # what the arm believes, so its confidence has to be recorded at every rate, not once.
                record[(arm_name, corruption_probability, "iou_token")] = confidence["iou_token"][first_occlusion:]
                record[(arm_name, corruption_probability, "object_score")] = confidence["object_score"][first_occlusion:]

        for tracker in trackers_by_arm.values():
            tracker.frame_cache = None
            tracker.corruption_boxes = None
            tracker.corruption_p = 0.0

        scored_clips.append(record)
        save_results(results_path, processed_trajectories, scored_clips, corruption_probabilities, commit_threshold)
        report_clip(record, list(trackers_by_arm), corruption_probabilities)


if __name__ == "__main__":
    run()

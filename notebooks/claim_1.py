import sys
import pickle
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from create_anchor_dataset import anchor_trajectory_index
from src.utils.compute_iou import compute_iou

SUCCESS_IOU = 0.5
SAM_CONFIG = "conf/trackers/baselines/sam2.yaml"
BASE_DRAW = 400


def test_trajectories(person_path, n_traj, random_seed=42, base=BASE_DRAW):
    """Draw `n_traj` trajectories as a seeded held-out test set, in shuffled order.
    Nested in `n_traj`: the first `base` are the original split, a larger request appends a disjoint tail."""

    video_names = person_path.selected_video_names.tolist()
    person_ids = person_path.selected_person_ids.tolist()
    anchor_frames = person_path.selected_anchor_video_frames.tolist()
    overlaps = person_path.selected_anchor_overlaps.tolist()

    trajectories = [(video, int(person), int(anchor), float(overlap)) for video, person, anchor, overlap in zip(video_names, person_ids, anchor_frames, overlaps)]
    requested = min(n_traj, len(trajectories))
    base_size = min(base, len(trajectories))

    _, base_indices = train_test_split(np.arange(len(trajectories)), test_size=base_size, random_state=random_seed, shuffle=True)
    if requested <= len(base_indices):
        return [trajectories[index] for index in base_indices[:requested]]

    remaining = np.setdiff1d(np.arange(len(trajectories)), base_indices)
    np.random.default_rng(random_seed).shuffle(remaining)
    drawn = np.concatenate([base_indices, remaining[:requested - len(base_indices)]])
    return [trajectories[index] for index in drawn]


def load_window(detection_data, trajectory, max_frames):
    """Load `max_frames` starting AT the anchor into `detection_data`, or None when the anchor is missing.
    The clip begins at the anchor, so there is no warmup and nothing before it is scored."""

    video_name, person_id, anchor_video_frame = trajectory
    detection_data.load_frames = False
    detection_data.initialize_target(video_name, person_id)

    anchor_index = anchor_trajectory_index(detection_data, anchor_video_frame)
    if anchor_index is None:
        return None

    window = detection_data.frame_indices[anchor_index:anchor_index + max_frames]
    detection_data.load_frames = True
    detection_data.initialize_target(video_name, person_id, frame_indices=window)
    return 0, anchor_index


def first_occlusion_frame(occlusions):
    """Index of the first occluded frame, or the clip length when the target is never occluded."""

    return int(np.argmax(occlusions > 0)) if float(occlusions.max()) > 0 else len(occlusions)


def visible_frames(occlusions, boxes, start):
    """Frames at or after `start` that are scorable: not occluded, and carrying a real annotated box.
    Requiring a box excludes annotation gaps, where there is nothing to score the tracker against."""

    return [frame for frame in range(start, len(occlusions)) if occlusions[frame] < 0.5 and boxes[frame][2] > 0]


def frame_record(predicted_masks, occlusions, boxes, first_occlusion):
    """Per-frame box IoU from the first occlusion onward, keeping occluded frames in place.
    A frame without a box scores NaN rather than a meaningless 0, and each metric decides what that means."""

    span = slice(first_occlusion, len(occlusions))
    masks, span_boxes = predicted_masks[span], boxes[span]
    has_box = np.asarray(span_boxes[:, 2] > 0)

    ious = np.full(len(masks), np.nan, dtype=np.float32)
    if has_box.any():
        ious[has_box] = compute_iou(span_boxes[has_box], masks[has_box])
    return ious


def as_array(values, n_frames):
    """A torch buffer, a numpy array or None, as an `(n_frames,)` float32 array of NaN when absent."""

    if values is None:
        return np.full(n_frames, np.nan, dtype=np.float32)
    values = values.numpy() if hasattr(values, "numpy") else values
    return np.asarray(values, dtype=np.float32)


def confidence_signals(tracker, n_frames):
    """Per-frame confidence, normalised across arms to `iou_token` and `object_score`.

    VOT-LT scores a box AND a confidence, sweeping the confidence -- so every arm has to expose one, and
    they expose it differently. SAMURAI, SAMITE and SAM2Long replay the predictor's stored scores into
    `memory_scores`, whose columns are the mask-affinity (IoU) score, the object logit and, for SAMURAI,
    the Kalman motion term. SAMBaseline and SAM 2 instead record `iou_scores`/`object_scores` as they roll.
    SAM 3 keeps one score per frame and no IoU token.

    SCALES DIFFER between arms -- SAMBaseline's object score has already been through a sigmoid
    (`sam_v2_model.step_video_masking`), the others are raw logits -- so a threshold sweep has to run over
    each arm's OWN range. Sweeping a shared grid would silently hand some arms an operating point they
    cannot reach; that is the mistake `min_obj_score_logits: -1` already encodes for SAMURAI."""

    scores = getattr(tracker, "memory_scores", None)
    if scores is not None:
        scores = as_array(scores, n_frames)
        if scores.ndim == 1:                      # SAM 3: a single per-frame score, no token
            return {"iou_token": np.full(n_frames, np.nan, dtype=np.float32), "object_score": scores}
        return {"iou_token": scores[:, 0], "object_score": scores[:, 1]}

    # SAMBaseline/SAM 2 fill these directly; SAMURAI and SAMITE allocate them but never write to them,
    # which is why `memory_scores` is preferred above rather than merged with them.
    token = as_array(getattr(tracker, "iou_scores", None), n_frames)
    if token.ndim > 1:                            # (frames, 1 + proposals): keep SAM's chosen proposal
        token = token[:, 1:].max(axis=1)
    return {"iou_token": token, "object_score": as_array(getattr(tracker, "object_scores", None), n_frames)}


def commit_flags(tracker, n_frames):
    """Per-frame memory-commit decisions, however the arm records them.
    The oracles keep committed frame indices, the trackers a per-frame flag array."""

    if not hasattr(tracker, "committed_frames"):
        return np.asarray(tracker.update_memory.numpy(), dtype=bool)

    flags = np.zeros(n_frames, dtype=bool)
    committed = np.asarray(tracker.committed_frames, dtype=int)
    if committed.size:
        flags[committed] = True
    return flags


def load_results(results_path):
    """Resume the per-trajectory records, or start empty when there is no pickle yet.
    `processed` holds every trajectory already seen, unscorable ones included, so none is retried."""

    if not results_path.exists():
        return set(), []

    state = pickle.load(open(results_path, "rb"))
    return set(state["processed"]), list(state["clips"])


def save_results(results_path, processed, clips, thresholds, n_bins):
    """Checkpoint after every clip, so a killed run resumes instead of restarting."""

    state = {"processed": processed, "clips": clips, "thresholds": thresholds, "n_bins": int(n_bins)}
    results_path.write_bytes(pickle.dumps(state))


def roll_arm(tracker, detection_data, occlusions, boxes, first_occlusion):
    """One arm over the loaded clip: per-frame IoUs across the scored span, and its commit decisions."""

    predicted_masks = tracker.predict_masks(detection_data).numpy()
    ious = frame_record(predicted_masks, occlusions, boxes, first_occlusion)
    commits = commit_flags(tracker, predicted_masks.shape[0])[first_occlusion:]
    return ious, commits


def sweep_arm(oracle, detection_data, occlusions, boxes, first_occlusion, thresholds):
    """The same oracle at every commit-gate threshold, keyed by threshold.
    The shared frame cache means SAM's image encoder runs once for the clip however long the sweep is."""

    ious, commits = {}, {}
    for threshold in thresholds:
        oracle.iou_threshold = threshold
        ious[threshold], commits[threshold] = roll_arm(oracle, detection_data, occlusions, boxes, first_occlusion)
    return ious, commits


def clip_record(trajectory, occlusions, boxes, first_occlusion, ious_per_arm, commits_per_arm):
    """One trajectory's record: identity, occlusion structure, and every arm's raw IoUs.
    The scored span starts at the first occlusion, so `occluded` and `has_box` align with each arm's array."""

    video_name, person_id, _, overlap = trajectory
    span = slice(first_occlusion, len(occlusions))

    record = {"video": video_name, "person": person_id, "distractor_overlap": float(overlap)}
    record["n_frames"] = int(len(occlusions))
    record["first_occlusion"] = int(first_occlusion)
    record["occ_count"] = int((occlusions >= 0.5).sum())
    record["occlusions"] = occlusions.astype(np.float32)
    record["occluded"] = np.asarray(occlusions[span] >= 0.5, dtype=bool)
    record["has_box"] = np.asarray(boxes[span][:, 2] > 0, dtype=bool)

    for arm_name, ious in ious_per_arm.items():
        record[arm_name] = ious
        record[f"{arm_name}_commit"] = commits_per_arm[arm_name]
    return record


def process_trajectory(trackers, detection_data, trajectory, max_frames, thresholds):
    """Roll every arm over one clip, sharing a frame cache, or None when the clip cannot be scored."""

    if load_window(detection_data, trajectory[:3], max_frames) is None:
        return None

    occlusions, boxes = detection_data.occlusions, detection_data.bboxes_norm
    first_occlusion = first_occlusion_frame(occlusions)
    if not visible_frames(occlusions, boxes, first_occlusion):
        return None

    baseline, memory_oracle, mask_oracle = trackers
    frame_cache = {}
    for tracker in trackers:
        tracker.frame_cache = frame_cache

    try:
        ious_per_arm, commits_per_arm = {}, {}
        ious_per_arm["sam"], commits_per_arm["sam"] = roll_arm(baseline, detection_data, occlusions, boxes, first_occlusion)
        ious_per_arm["memory"], commits_per_arm["memory"] = sweep_arm(memory_oracle, detection_data, occlusions, boxes, first_occlusion, thresholds)
        ious_per_arm["mask"], commits_per_arm["mask"] = sweep_arm(mask_oracle, detection_data, occlusions, boxes, first_occlusion, thresholds)
    finally:
        for tracker in trackers:
            tracker.frame_cache = None

    return clip_record(trajectory, occlusions, boxes, first_occlusion, ious_per_arm, commits_per_arm)


@hydra.main(config_path="../conf", config_name="experiments/claim_1", version_base=None)
def claim_1(config: DictConfig):
    """Standard SAM 2 against the memory and mask oracles, over a seeded draw of held-out trajectories.
    The oracles read ground truth, so their margin is the headroom a deployable commit gate could recover."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    memory_oracle = hydra.utils.instantiate(OmegaConf.load(config.oracle_config).tracker)
    mask_oracle = hydra.utils.instantiate(OmegaConf.load(config.mask_oracle_config).tracker)
    baseline = hydra.utils.instantiate(OmegaConf.load(SAM_CONFIG).tracker)
    baseline.label_mask_iou = False

    trackers = (baseline, memory_oracle, mask_oracle)
    thresholds = [float(threshold) for threshold in config.thresholds]

    results_path = Path(config.out_dir) / "results.pkl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    processed, clips = load_results(results_path)

    for trajectory in tqdm(test_trajectories(person_path, config.n_traj), desc="claim_1"):
        video_name, person_id = trajectory[0], trajectory[1]
        if (video_name, person_id) in processed:
            continue
        processed.add((video_name, person_id))

        record = process_trajectory(trackers, detection_data, trajectory, config.max_frames, thresholds)
        if record is None:
            continue

        clips.append(record)
        save_results(results_path, processed, clips, thresholds, config.n_bins)


if __name__ == "__main__":
    claim_1()

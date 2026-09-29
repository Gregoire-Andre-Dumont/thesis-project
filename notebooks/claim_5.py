"""claim_5: SAMARA deployed, scored with VOT-LT, and split by how long the target is occluded.

Two confidences are trained per fold on the same features and recorded on the SAME trajectories, so
they can be compared without confounding the score with the masks it was computed on:
  * the GATE, a classifier on 'is this mask worth committing' (`BCEIouLoss`), whose probability also
    drives the memory write;
  * a MAP-ONLY gate (`cnn_gate_map_samara`): identical loss and label, but SAM's IoU token and object
    score withheld, so the pair isolates what those two scalars are worth.
Only the scalars gate drives memory; the map-only one is reported and never acted on.

The corruption-prior sweep was dropped: F varied by 0.021 across p in [0, 0.2] with no monotone trend,
so the prior is not a hyperparameter worth tuning, and picking its argmax on the evaluation set would
be selection on test. p0.00 -- no corruption, the simplest arm -- is the one reported."""

import hashlib
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from claim_1 import test_trajectories
from claim_1 import load_window
from claim_1 import first_occlusion_frame
from claim_1 import visible_frames
from claim_1 import frame_record
from claim_1 import commit_flags

logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"


# PURE corruption priors, one level each. Mixing p0.00 with a corrupted level was tried and changed
# nothing: every mixed arm landed at or just below p0.00 in both the short and long occlusion splits, with
# every interval straddling zero. Mixing also doubles the training set, so a mixed arm sees twice the
# gradient steps at the same epoch budget -- the prior and the optimisation are confounded, which a pure
# sweep avoids.
ARMS = {
    "p0.00": [0.00],
    "p0.05": [0.05],
    "p0.10": [0.10],
    "p0.15": [0.15],
}

def load_results(results_path):
    """Resume an arm's records, or start empty. Returns (processed (video, person) set, clip records)."""

    if not results_path.exists():
        return set(), []

    arm_state = pickle.load(open(results_path, "rb"))
    processed = set(arm_state["processed"])
    clips_scored = list(arm_state["clips"])
    return processed, clips_scored


def save_results(results_path, processed, clips_scored):
    """Atomically write an arm's records, so a killed run resumes from the last completed clip."""

    arm_state = {"processed": list(processed), "clips": clips_scored}
    temporary_path = results_path.with_suffix(".pkl.tmp")
    temporary_path.write_bytes(pickle.dumps(arm_state))
    temporary_path.replace(results_path)


def scorable(detection_data, clip_triple, max_frames):
    """Load this clip's window and say whether it can be scored, without a separate enumeration pass."""

    if load_window(detection_data, clip_triple, max_frames) is None:
        return False

    occlusions = detection_data.occlusions
    ground_truth_boxes = detection_data.bboxes_norm
    first_occlusion = first_occlusion_frame(occlusions)
    return bool(visible_frames(occlusions, ground_truth_boxes, first_occlusion))


def checkpoint_name(arm_name, block_stems, all_stems):
    """A model name that identifies the arm AND the split."""

    held_out = " ".join(sorted(block_stems))
    split_digest = hashlib.sha1(held_out.encode()).hexdigest()[:10]
    return f"cnn_gate_scalars_{arm_name }_{len(block_stems)}of{len(all_stems)}_{split_digest }"


def train_arm(trainer_config, probabilities, all_stems, block_stems, fold_index, arm_name):
    """One arm's gate for one block, trained on every dataset trajectory the block does NOT deploy on."""

    config = OmegaConf.create(OmegaConf.to_container(trainer_config, resolve=True))
    config.dataset.probabilities = list(probabilities)
    config.model_name = checkpoint_name(arm_name, block_stems, all_stems)
    trainer = hydra.utils.instantiate(config)

    is_deployed = np.array([trajectory_stem in block_stems for trajectory_stem in all_stems])
    train_indices = np.flatnonzero(~is_deployed)
    validation_indices = np.flatnonzero(is_deployed)
    if len(validation_indices) == 0:
        raise SystemExit("no deployed trajectory is in the calibrator dataset: nothing is held out")

    loss_name = type(trainer.criterion).__name__

    trainer.custom_train(
        x=all_stems,
        y=all_stems,
        train_indices=list(train_indices),
        validation_indices=list(validation_indices),
        fold=fold_index)
    return trainer.model


def clip_record(tracker, detection_data, clip_triple, fold_index, arm_name):
    """One arm's record for one clip, in claim_1's field names so the pkls join on (video, person)."""

    video_name, person_id, anchor_frame = clip_triple
    occlusions = detection_data.occlusions
    ground_truth_boxes = detection_data.bboxes_norm

    first_occlusion = first_occlusion_frame(occlusions)
    scored_span = slice(first_occlusion, len(occlusions))
    predicted_masks = tracker.predict_masks(detection_data).numpy()

    return {
        "video": video_name,
        "person": person_id,
        "anchor": anchor_frame,
        "block": fold_index,
        "arm": arm_name,
        "n_frames": int(len(occlusions)),
        "first_occlusion": int(first_occlusion),
        "occ_count": int((occlusions >= 0.5).sum()),
        "occlusions": np.asarray(occlusions, dtype=np.float32),
        "occluded": np.asarray(occlusions[scored_span] >= 0.5, dtype=bool),
        "has_box": np.asarray(ground_truth_boxes[scored_span][:, 2] > 0, dtype=bool),
        "samara": frame_record(predicted_masks, occlusions, ground_truth_boxes, first_occlusion),
        "samara_commit": commit_flags(tracker, predicted_masks.shape[0])[scored_span],
        "commit_all": np.asarray(tracker.committed.numpy(), dtype=bool),
        # Three confidences on ONE set of trajectories: the gate's calibrated p(commit), the regressor's
        # predicted IoU, and SAM's own token and object score. VOT-LT sweeps whichever is under test.
        "commit_probability": tracker.commit_probabilities[scored_span].numpy().astype(np.float32),
        "map_only_score": tracker.calibrator_scores[scored_span].numpy().astype(np.float32),
        "iou_token": tracker.token_scores[scored_span].numpy().astype(np.float32),
        "object_score": tracker.object_scores[scored_span].numpy().astype(np.float32),
        "mask_area": tracker.mask_area[scored_span].numpy().astype(np.int32)}


def roll_arms(tracker, detection_data, clip_triple, gate_per_arm, map_gate_per_arm, fold_index, state_per_arm, arms_to_roll):
    """Roll one clip through the arms that still need it, sharing SAM's image features across them."""

    frame_cache = {}
    tracker.frame_cache = frame_cache

    for arm_name in arms_to_roll:
        tracker.model.gate_controller = gate_per_arm[arm_name]
        tracker.model.controller = map_gate_per_arm[arm_name]
        tracker.model.eval()
        state_per_arm[arm_name]["clips"].append(clip_record(tracker, detection_data, clip_triple, fold_index, arm_name))

    tracker.frame_cache = None


@hydra.main(config_path="../conf", config_name="experiments/claim_5", version_base=None)
def run(config: DictConfig):
    """Roll every arm over every block, sharing each clip's image features across the arms."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    tracker = hydra.utils.instantiate(config.tracker.tracker)
    arms = {name: ARMS[name] for name in config.get("arms", list(ARMS))}
    max_frames = int(config.max_frames)
    out_dir = Path(config.out_dir)

    state_per_arm = {}
    for arm_name in arms:
        arm_dir = out_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        processed, clips_scored = load_results(arm_dir / "results.pkl")
        state_per_arm[arm_name] = {"processed": processed, "clips": clips_scored}

    drawn_trajectories = test_trajectories(person_path, int(config.n_traj))
    scored_clips = [(video_name, int(person_id), int(anchor_frame)) for video_name, person_id, anchor_frame, *_ in drawn_trajectories]
    video_names = {video_name for video_name, _, _ in scored_clips}
    cross_validation_blocks = np.array_split(np.arange(len(scored_clips)), int(config.folds))

    template = hydra.utils.instantiate(config.gate_trainers.main_trainer)
    all_stems = np.array(template.dataset.trajectories())

    arm_summary = ", ".join(f"{arm_name }={corruption_levels }" for arm_name, corruption_levels in arms.items())
    for arm_name in arms:
        print(f"  {arm_name }: resuming with {len(state_per_arm[arm_name]['processed'])} clips already scored")

    for fold_index, block_indices in enumerate(cross_validation_blocks):
        block_clips = [scored_clips[clip_index] for clip_index in block_indices]

        remaining = [clip_triple for clip_triple in block_clips if any((clip_triple[0], clip_triple[1]) not in state_per_arm[arm_name]["processed"] for arm_name in arms)]

        if not remaining:
            continue

        block_stems = {f"{video_name }_{person_id }" for video_name, person_id, _ in block_clips}
        gate_per_arm, map_gate_per_arm = {}, {}
        for arm_name, probabilities in arms.items():
            gate_per_arm[arm_name] = train_arm(config.gate_trainers.main_trainer, probabilities, all_stems, block_stems, fold_index, arm_name)
            map_gate_per_arm[arm_name] = train_arm(config.offline_trainers.main_trainer, probabilities, all_stems, block_stems, fold_index, f"map_{arm_name}")

        for clip_triple in tqdm(remaining, desc=f"block {fold_index +1 }"):
            video_name, person_id, _ = clip_triple
            arms_to_roll = [arm_name for arm_name in arms if (video_name, person_id) not in state_per_arm[arm_name]["processed"]]
            for arm_name in arms:
                state_per_arm[arm_name]["processed"].add((video_name, person_id))

            if not scorable(detection_data, clip_triple, max_frames):
                continue

            roll_arms(tracker, detection_data, clip_triple, gate_per_arm, map_gate_per_arm, fold_index, state_per_arm, arms_to_roll)
            for arm_name in arms_to_roll:
                save_results(
                    out_dir / arm_name / "results.pkl",
                    state_per_arm[arm_name]["processed"],
                    state_per_arm[arm_name]["clips"])

    for arm_name in arms:
        print(f"{arm_name }: {len(state_per_arm[arm_name]['clips'])} clips -> {out_dir /arm_name /'results.pkl'}")


if __name__ == "__main__":
    run()

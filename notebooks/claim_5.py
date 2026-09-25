"""claim_5: how much memory corruption should SAMARA's commit gate be trained on?

Four gate arms over claim_1's 400 clips, differing in ONE thing -- the corruption level the calibrator's
training rollouts were recorded at. `p0.00` is the clean control. One level per arm means every arm trains
on the same NUMBER of samples, so the only thing that varies is how corrupted the memory bank was.

SELECTION IS OFF. Seven configurations with the calibrator picking the mask were measured and none beat
gate-only on any metric, so SAM's IoU token picks and the commit gate is the only intervention. The gate
needs one proposal scored rather than three, so `SamaraController` crops and encodes only the mask SAM
kept -- the Perception Encoder is 19% of a rollout instead of ~40%.

THE GATE READS SAM'S OWN SCORES. Alongside the anchor similarity map it takes the per-proposal IoU token
and the per-frame object-presence logit (`cnn_gate_scalars_samara.yaml`). They cost nothing -- SAM computes
them every frame anyway -- and they answer the question the map cannot: not 'is this the target' but 'is
this mask any good'.

BLOCKED CROSS-VALIDATION. The clips are cut into `folds` blocks; for each block every arm's calibrator
trains on every dataset trajectory EXCEPT that block's, then is deployed on it. So every clip is scored by
a calibrator that never saw it. Checkpoint identity includes WHAT WAS HELD OUT: epochalyst hashes the
trainer config, which does not mention the split, so a run over a different `n_traj` once produced an
identical hash and silently deployed a gate trained on 92 of the 100 clips it was scored on.

ONE PASS OVER THE PIXELS. The arms are rolled back-to-back on the same clip sharing a frame cache: SAM's
image encoder depends only on the frame, so it runs once and the other arms reuse its output. Profiled at
39% of a rollout, so sharing it takes the sweep to ~71% of four sequential runs. The cache holds ~8.4 MB
per frame -- ~2.5 GB for a 300-frame clip -- and is dropped with the clip.

NO BASELINE ARM. claim_1 already rolled these clips with this same `SAMBaseline`, so its `sam` records are
the comparison. Records carry claim_1's field names and its scored span, so the pkls join on
(video, person). Each arm writes its own results.pkl under `out_dir/<arm>/`.

    python notebooks/claim_5.py [n_traj=400] [folds=4]
"""
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

# Arm name -> the corruption levels its gate trains on. One level each, so sample count is held constant.
ARMS = {
    "p0.00": [0.00],
    "p0.05": [0.05],
    "p0.10": [0.10],
    "p0.15": [0.15],
    "p0.20": [0.20],
}


# ---------------------------------------------------------------------------------------
# resumable per-arm records
# ---------------------------------------------------------------------------------------

def load_results(results_path):
    """Resume an arm's records, or start empty. Returns (processed (video, person) set, clip records)."""

    if not results_path.exists():
        return set(), []

    state = pickle.load(open(results_path, "rb"))
    processed = set(state["processed"])
    clips_scored = list(state["clips"])
    return processed, clips_scored


def save_results(results_path, processed, clips_scored):
    """Atomically write an arm's records, so a killed run resumes from the last completed clip."""

    state = {"processed": list(processed), "clips": clips_scored}
    temporary_path = results_path.with_suffix(".pkl.tmp")
    temporary_path.write_bytes(pickle.dumps(state))
    temporary_path.replace(results_path)


def scorable(detection_data, clip, max_frames):
    """Load this clip's window and say whether it can be scored, without a separate enumeration pass.

    claim_1's two filters -- the window must load, and the clip must hold at least one visible annotated
    frame after its first occlusion -- applied here rather than up front, because deciding them needs the
    decoded window. Pre-filtering would decode every clip once to throw the pixels away and again to track
    them. Unscorable clips simply leave their block one clip short."""

    if load_window(detection_data, clip, max_frames) is None:
        return False

    occlusions = detection_data.occlusions
    boxes = detection_data.bboxes_norm
    first_occlusion = first_occlusion_frame(occlusions)
    return bool(visible_frames(occlusions, boxes, first_occlusion))


# ---------------------------------------------------------------------------------------
# training one arm's gate for one block
# ---------------------------------------------------------------------------------------

def checkpoint_name(arm, block_stems, all_stems):
    """A model name that identifies the arm AND the split.

    epochalyst's checkpoint hash covers the trainer config, which says nothing about which trajectories
    were held out. Without the digest, a run over a different number of trajectories reloads weights
    trained on a different -- possibly overlapping -- set. The `scalars` prefix separates these gates from
    the earlier map-only sweep, which shares the arm, the split and the corruption level but reads
    fewer features."""

    held_out = " ".join(sorted(block_stems))
    digest = hashlib.sha1(held_out.encode()).hexdigest()[:10]
    return f"cnn_gate_scalars_{arm}_{len(block_stems)}of{len(all_stems)}_{digest}"


def train_arm(trainer_config, probabilities, all_stems, block_stems, fold, arm):
    """One arm's gate for one block, trained on every dataset trajectory the block does NOT deploy on.

    The held-out stems double as the trainer's validation set, so early stopping reads the block being
    scored -- the usual cost of not carving out a third split."""

    config = OmegaConf.create(OmegaConf.to_container(trainer_config, resolve=True))
    config.dataset.probabilities = list(probabilities)
    config.model_name = checkpoint_name(arm, block_stems, all_stems)
    trainer = hydra.utils.instantiate(config)

    is_deployed = np.array([stem in block_stems for stem in all_stems])
    train_indices = np.flatnonzero(~is_deployed)
    validation_indices = np.flatnonzero(is_deployed)
    if len(validation_indices) == 0:
        raise SystemExit("no deployed trajectory is in the calibrator dataset: nothing is held out")

    loss_name = type(trainer.criterion).__name__
    print(f"  gate {arm}: {len(train_indices)} training trajectories, {len(validation_indices)} held out, "
          f"{loss_name} on p={list(trainer.dataset.probabilities)} [{config.model_name}]", flush=True)

    trainer.custom_train(x=all_stems, y=all_stems, train_indices=list(train_indices),
                         validation_indices=list(validation_indices), fold=fold)
    return trainer.model


# ---------------------------------------------------------------------------------------
# rolling one clip through every arm
# ---------------------------------------------------------------------------------------

def clip_record(tracker, detection_data, clip, fold, arm):
    """One arm's record for one clip, in claim_1's field names so the pkls join on (video, person)."""

    video, person, anchor = clip
    occlusions = detection_data.occlusions
    boxes = detection_data.bboxes_norm

    first_occlusion = first_occlusion_frame(occlusions)
    span = slice(first_occlusion, len(occlusions))
    predicted_masks = tracker.predict_masks(detection_data).numpy()

    return {
        "video": video,
        "person": person,
        "anchor": anchor,
        "block": fold,
        "arm": arm,
        "n_frames": int(len(occlusions)),
        "first_occlusion": int(first_occlusion),
        "occ_count": int((occlusions >= 0.5).sum()),
        "occlusions": np.asarray(occlusions, dtype=np.float32),
        "occluded": np.asarray(occlusions[span] >= 0.5, dtype=bool),
        "has_box": np.asarray(boxes[span][:, 2] > 0, dtype=bool),
        "samara": frame_record(predicted_masks, occlusions, boxes, first_occlusion),
        "samara_commit": commit_flags(tracker, predicted_masks.shape[0])[span],
        # Full-length commits, not the span slice: the gate acts from frame 0, so diagnosing when two arms
        # diverge needs the frames before the scored span too.
        "commit_all": np.asarray(tracker.committed.numpy(), dtype=bool),
        "commit_probability": tracker.commit_probabilities[span].numpy().astype(np.float32),
    }


def roll_arms(tracker, detection_data, clip, gates, fold, states, arms_to_roll):
    """Roll one clip through the arms that still need it, sharing SAM's image features across them.

    The cache is filled by the first arm and read by the rest, then dropped with the clip. Verified
    transparent: a cache-fed rollout is bit-identical to a cold one with the same gate.

    Only `arms_to_roll` are recorded. Rolling every arm unconditionally would append a second copy of the
    clip to arms that already had it -- which is what happens when an arm is ADDED to a finished sweep and
    its clips are re-rolled to catch it up."""

    frame_cache = {}
    tracker.frame_cache = frame_cache

    for arm in arms_to_roll:
        tracker.model.gate_controller = gates[arm]
        tracker.model.eval()
        states[arm]["clips"].append(clip_record(tracker, detection_data, clip, fold, arm))

    tracker.frame_cache = None


# ---------------------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="experiments/claim_5", version_base=None)
def run(config: DictConfig):
    """Roll every arm over every block, sharing each clip's image features across the arms."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    tracker = hydra.utils.instantiate(config.tracker.tracker)
    max_frames = int(config.max_frames)
    out_dir = Path(config.out_dir)


    states = {}
    for arm in ARMS:
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        processed, clips_scored = load_results(arm_dir / "results.pkl")
        states[arm] = {"processed": processed, "clips": clips_scored}

    drawn = test_trajectories(person_path, int(config.n_traj))
    clips = [(video, int(person), int(anchor)) for video, person, anchor, *_ in drawn]
    videos = {video for video, _, _ in clips}
    blocks = np.array_split(np.arange(len(clips)), int(config.folds))

    template = hydra.utils.instantiate(config.gate_trainers.main_trainer)
    all_stems = np.array(template.dataset.trajectories())

    arm_summary = ", ".join(f"{arm}={levels}" for arm, levels in ARMS.items())
    print(f"{len(clips)} clips over {len(videos)} videos, {len(blocks)} blocks of ~{len(blocks[0])}")
    print(f"arms: {arm_summary}")
    print(f"tracker: gate={tracker.gate} threshold={tracker.commit_threshold}")
    for arm in ARMS:
        print(f"  {arm}: resuming with {len(states[arm]['processed'])} clips already scored")
    print(flush=True)

    for fold, block in enumerate(blocks):
        block_clips = [clips[index] for index in block]
        # A clip is re-rolled only while some arm still needs it, so a killed run resumes per arm.
        remaining = [clip for clip in block_clips
                     if any((clip[0], clip[1]) not in states[arm]["processed"] for arm in ARMS)]

        if not remaining:
            print(f"block {fold + 1}/{len(blocks)}: already complete", flush=True)
            continue

        print(f"block {fold + 1}/{len(blocks)}: {len(block_clips)} clips, {len(remaining)} to score",
              flush=True)
        block_stems = {f"{video}_{person}" for video, person, _ in block_clips}
        gates = {}
        for arm, probabilities in ARMS.items():
            gates[arm] = train_arm(config.gate_trainers.main_trainer, probabilities,
                                   all_stems, block_stems, fold, arm)

        for clip in tqdm(remaining, desc=f"block {fold + 1}"):
            video, person, _ = clip
            arms_to_roll = [arm for arm in ARMS if (video, person) not in states[arm]["processed"]]
            for arm in ARMS:
                states[arm]["processed"].add((video, person))

            if not scorable(detection_data, clip, max_frames):
                continue

            roll_arms(tracker, detection_data, clip, gates, fold, states, arms_to_roll)
            for arm in arms_to_roll:
                save_results(out_dir / arm / "results.pkl",
                             states[arm]["processed"], states[arm]["clips"])

    for arm in ARMS:
        print(f"{arm}: {len(states[arm]['clips'])} clips -> {out_dir / arm / 'results.pkl'}")
    print("compare each against claim_1's `sam` records, which join on (video, person)")


if __name__ == "__main__":
    run()

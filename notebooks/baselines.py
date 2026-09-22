"""Published SAM 2 tracking baselines on the same clips, the same anchors and the same metric as SAMARA.

Every arm here is an EXISTING method that modifies SAM 2's memory or its mask choice, so they answer the
question a reviewer asks first: is the commit gate doing something these already do?

    samite      SAMITE -- prototypical token propagation, resisting drift towards distractors
    samurai     SAMURAI -- motion-aware memory selection, a Kalman filter scoring the proposals
    sam2long    SAM2Long -- a tree search over `num_pathway` candidate memory states
    sentry      SENTRY -- consistency-validated memory writes, the authors' released package

The first three run SAM 2.1 through `muggled_sam` with their own video predictors, and `sentry` drives
the authors' released package, which builds the official `sam2` predictor. All four own their memory state
and their own initialisation. That is why there is no shared frame cache here as there is in claim_5: each
predictor encodes the clip itself, so the arms are independent rollouts and the script costs one full pass
per arm.

CHECKPOINT PARITY. The stock configs under `conf/trackers/baselines/` point at `sam_base_plus`, but claim_1's
SAM baseline and SAMARA both run `sam_large`. A comparison across a backbone size measures the backbone, so
this script overrides each arm onto the large config and checkpoint. `OVERRIDES` is the one place that
happens; set `--stock` to run a config exactly as it sits on disk instead.

THE DRAW IS claim_1'S. Same `person_path` block, same `n_traj`, same anchors, so every record joins on
(video, person) against claim_1's `sam` / oracle records and claim_5's gate arms. Each arm writes its own
resumable `results.pkl` under `out_dir/<arm>/`, so a killed run resumes per arm and an arm can be added
later without re-rolling the others.

COMMIT FLAGS. Each arm reports a per-frame memory decision, so hygiene is computable and these sit on
claim_5's heatmaps beside the gate arms. The decision is each method's OWN, not one invented here:

    samurai/samite   memory conditioning keeps a stored frame only when its mask affinity, object score
                     and Kalman motion score clear the model's thresholds, so that predicate is replayed
                     over the scores the predictor already stored. These arms refuse frames.
    sam2long         no per-frame refusal -- every frame enters each candidate pathway and the tree search
                     chooses among pathways, so the flag is all-True.
    sentry           refines WHICH mask is written rather than whether to write, so the flag is all-True;
                     its decision shows up in `sources`, the branch of its three-tier selection.

ONE ARM PER PROCESS. Each method vendors its own `sam2` fork and they are mutually incompatible --
SAMURAI's and SAMITE's add a `samurai_mode` argument to `SAM2Base` that the others' do not have -- so two
arms cannot share an interpreter. The arm is therefore a required argument rather than a loop, and the
caller binds the matching fork with PYTHONPATH.

    python notebooks/baselines.py <arm> [--stock]
"""
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
from claim_5 import load_results
from claim_5 import save_results


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

# Arm name -> the tracker config that builds it.
ARMS = {
    "samite": "conf/trackers/baselines/samite.yaml",
    "samurai": "conf/trackers/baselines/samurai.yaml",
    "sam2long": "conf/trackers/baselines/sam2long.yaml",
    "sentry": "conf/trackers/baselines/sentry.yaml",
}

# Backbone parity with claim_1's SAM baseline and SAMARA, both of which run sam_large. `sentry` is absent
# because its own config already names sam_large, through the official sam2 builder rather than muggled_sam.
OVERRIDES = {
    "samite": {"checkpoint": "tm/sam_large.pt", "model_config": "SAM2/samite_hiera_large.yaml"},
    "samurai": {"checkpoint": "tm/sam_large.pt", "model_config": "SAM2/samurai_hiera_large.yaml"},
    "sam2long": {"checkpoint": "tm/sam_large.pt", "model_config": "SAM2/sam2long_hiera_large.yaml"},
}


# ---------------------------------------------------------------------------------------
# building one arm
# ---------------------------------------------------------------------------------------

def build_tracker(arm, stock):
    """Instantiate one arm's tracker, on the large backbone unless `stock` keeps the config as written."""

    config_path = Path(ARMS[arm])
    if not config_path.exists():
        return None, f"missing config {config_path}"

    config = OmegaConf.load(config_path).tracker
    if not stock and arm in OVERRIDES:
        for key, value in OVERRIDES[arm].items():
            config[key] = value

    try:
        return hydra.utils.instantiate(config), None
    except Exception as error:                     # a missing checkpoint or vendored module, not a bug here
        return None, f"{type(error).__name__}: {error}"


def scorable(detection_data, clip, max_frames):
    """claim_1's two filters -- the window loads, and at least one visible annotated frame follows the first
    occlusion -- applied once per clip and shared by every arm, so all arms score the same set."""

    if load_window(detection_data, clip, max_frames) is None:
        return False

    occlusions = detection_data.occlusions
    boxes = detection_data.bboxes_norm
    first_occlusion = first_occlusion_frame(occlusions)
    return bool(visible_frames(occlusions, boxes, first_occlusion))


# ---------------------------------------------------------------------------------------
# rolling one clip through one arm
# ---------------------------------------------------------------------------------------

def clip_record(tracker, detection_data, clip, arm):
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
        "arm": arm,
        "n_frames": int(len(occlusions)),
        "first_occlusion": int(first_occlusion),
        "occ_count": int((occlusions >= 0.5).sum()),
        "occlusions": np.asarray(occlusions, dtype=np.float32),
        "occluded": np.asarray(occlusions[span] >= 0.5, dtype=bool),
        "has_box": np.asarray(boxes[span][:, 2] > 0, dtype=bool),
        "samara": frame_record(predicted_masks, occlusions, boxes, first_occlusion),
        "samara_commit": commit_flags(tracker, predicted_masks.shape[0])[span],
        "commit_all": np.asarray(tracker.update_memory.numpy(), dtype=bool),
    }


def coverage(record, iou_threshold=0.5):
    """Fraction of visible annotated post-occlusion frames held at `iou_threshold` -- claim_1's metric."""

    visible = record["has_box"] & ~record["occluded"]
    if not visible.any():
        return float("nan")
    return float((np.asarray(record["samara"], float)[visible] >= iou_threshold).mean())


# ---------------------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="experiments/baselines", version_base=None)
def run(config: DictConfig):
    """Roll every requested arm over claim_1's draw, one arm at a time."""

    arm, stock = SELECTION
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    max_frames = int(config.max_frames)
    out_dir = Path(config.out_dir)

    drawn = test_trajectories(person_path, int(config.n_traj))
    clips = [(video, int(person), int(anchor)) for video, person, anchor, *_ in drawn]
    videos = {video for video, _, _ in clips}

    print(f"{len(clips)} clips over {len(videos)} videos")
    print(f"arm: {arm}   backbone: {'per config' if stock else 'sam_large (parity override)'}")
    print(flush=True)

    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    results_path = arm_dir / "results.pkl"
    processed, clips_scored = load_results(results_path)

    remaining = [clip for clip in clips if (clip[0], clip[1]) not in processed]
    if not remaining:
        print(f"{arm}: already complete ({len(clips_scored)} clips)", flush=True)
        return

    tracker, failure = build_tracker(arm, stock)
    if tracker is None:
        # A missing checkout, checkpoint or dependency is a setup problem, not a bug here -- report what
        # it was rather than a stack trace from deep inside hydra.
        raise SystemExit(f"{arm}: cannot build -- {failure}")

    print(f"{arm}: {len(processed)} clips already scored, {len(remaining)} to go", flush=True)
    progress = tqdm(remaining, desc=arm)
    for clip in progress:
        video, person, _ = clip
        processed.add((video, person))

        if not scorable(detection_data, clip, max_frames):
            continue

        clips_scored.append(clip_record(tracker, detection_data, clip, arm))
        save_results(results_path, processed, clips_scored)
        progress.set_postfix(coverage=float(np.nanmean([coverage(r) for r in clips_scored])))

    held = float(np.nanmean([coverage(record) for record in clips_scored]))
    print(f"{arm}: {len(clips_scored)} clips -> {results_path}   coverage@0.5 {held:.4f}", flush=True)

    print("\ncompare against claim_1's `sam` records and claim_5's gate arms, which join on (video, person)")


def select_arm():
    """Read THE arm and `--stock` off the command line, and hide them from hydra.

    hydra.main parses everything left in `sys.argv` as a config override, so a bare word like `samurai`
    would be rejected before `run` is ever called. These are consumed here instead.

    Exactly one arm is required: the vendored `sam2` forks cannot coexist in one interpreter, so a loop
    over arms would silently run the second and third against whichever fork the first imported."""

    requested = [word for word in sys.argv[1:] if not word.startswith("-") and "=" not in word]
    stock = "--stock" in sys.argv

    unknown = [word for word in requested if word not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {list(ARMS)}")
    if len(requested) != 1:
        raise SystemExit(f"name exactly one arm ({', '.join(ARMS)}); each needs its own process")

    sys.argv = [word for word in sys.argv if word not in requested and word != "--stock"]
    return requested[0], stock


if __name__ == "__main__":
    SELECTION = select_arm()
    run()

"""Train the SAMARA calibrator once, then put it inside the tracker and measure what it is worth.

claim_4 scores the calibrator offline, on proposals a rollout already produced. That cannot tell you whether
a better ranking survives contact with the tracker: a better mask changes the next memory entry, which
changes the next frame's proposals, and errors compound either way. This runs the loop.

Two arms:

    sam       the real `SAMBaseline` from conf/trackers/baselines/sam_baseline.yaml -- SAM's IoU token picks
              the mask, and SAM's OWN confidence gates the commit (object score > 0.5 and predicted
              IoU > its `iou_threshold`). Using the class claim_1 and claim_2 use keeps the numbers
              comparable across claims, and it is the honest incumbent: a "commit every frame" arm would
              be a weaker tracker than SAM actually is, and would flatter the calibrator.
    samara    `SamaraController` -- the calibrator ranks the three proposals AND replaces SAM's commit gate.

Both pick the mask by the same rule when SAM decides (`1 + argmax(iou_scores[:, 1:])`), so what separates
the arms is the calibrator's two decisions, not plumbing.

Both arms roll EVERY clip before the next clip is loaded, sharing that clip's image-embedding cache. So the
two are always scored on identical clips -- the comparison is paired and valid at any point during the run,
not only at the end -- and SAM's image encoder runs once per frame instead of twice.

Clips and metric follow claim_1: the rollout starts AT the anchor (`load_window`), and coverage is the
hygiene metric over every annotated post-occlusion frame -- a visible frame counts when box IoU clears
`coverage_threshold`, an occluded frame counts when the arm did NOT commit it. Visible-only coverage is
reported alongside it, since the two answer different questions and the gate only moves the first.

The calibrator is held out by TRAJECTORY: the deployment clips' own (video, person) pairs are removed from
its training stems, but other people from the same videos remain. Scenes are therefore shared between
training and deployment, which flatters the calibrator relative to a new-scene deployment; the alternative
(holding out whole videos) costs about half the training set on this dataset.

    python notebooks/samara_deploy.py [n_traj=40] [max_frames=300]
"""
import logging
import os
import pickle
import warnings
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root on path for `src`
sys.path.insert(0, str(Path(__file__).resolve().parent))          # and `notebooks` for claim_1's helpers

from claim_1 import test_trajectories, load_window, first_occlusion_frame, commit_flags
from src.utils.compute_iou import compute_iou


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

ARM_NAMES = ("sam", "samara")


def deployment_clips(person_path, detection_data, n_traj, max_frames):
    """The trajectories to deploy on, as (video, person, anchor), keeping only ones that load and occlude."""

    clips = []
    for trajectory in tqdm(test_trajectories(person_path, n_traj), desc="clips", leave=False):
        video, person, anchor = trajectory[0], int(trajectory[1]), int(trajectory[2])
        if load_window(detection_data, (video, person, anchor), max_frames) is None:
            continue
        if first_occlusion_frame(detection_data.occlusions) >= len(detection_data.occlusions):
            continue                                   # never occluded: the memory bank is never tested
        clips.append((video, person, anchor))
    return clips


def clip_coverage(tracker, detection_data, threshold):
    """Coverage WITH OCCLUSION over every annotated post-occlusion frame, claim_1's hygiene metric.

    Each kind of frame is scored on what the arm can actually get right on it: a VISIBLE frame counts when
    box IoU >= `threshold`, an OCCLUDED frame counts when the arm did NOT write it to memory. While the
    target is hidden there is nothing to track, so the only decision left is whether to commit -- and
    committing then is exactly how a bank gets poisoned. This is the number the commit gate is supposed to
    move, and visible-only coverage cannot see it.

    Returns (coverage, visible-only coverage) so the two are comparable on the same rollout; NaN when the
    clip has no annotated post-occlusion frame."""

    predicted = tracker.predict_masks(detection_data).numpy()
    occlusions, boxes = detection_data.occlusions, detection_data.bboxes_norm
    span = slice(first_occlusion_frame(occlusions), len(occlusions))

    hidden = np.asarray(occlusions[span] >= 0.5, dtype=bool)
    seen = np.asarray(boxes[span][:, 2] > 0, dtype=bool) & ~hidden
    scored = int(seen.sum() + hidden.sum())
    if not scored:
        return float("nan"), float("nan")

    ious = np.full(len(hidden), np.nan, dtype=float)
    if seen.any():
        frames = np.flatnonzero(seen) + span.start
        ious[seen] = compute_iou(boxes[frames], predicted[frames])

    commits = commit_flags(tracker, predicted.shape[0])[span]
    held = float((ious[seen] >= threshold).sum())
    clean = float((~commits[hidden]).sum())
    visible_only = float((ious[seen] >= threshold).mean()) if seen.any() else float("nan")
    return (held + clean) / scored, visible_only


def report(scores, out_dir):
    """Print the paired table and write the per-clip scores, so a partial run is still analysable."""

    names = list(ARM_NAMES)
    paired = [row for row in scores if all(np.isfinite(row[name]) for name in names)]
    if not paired:
        return

    means = {name: float(np.mean([row[name] for row in paired])) for name in names}
    visible = {name: float(np.nanmean([row.get(f"{name}_visible", np.nan) for row in paired])) for name in names}
    difference = np.array([row["samara"] - row["sam"] for row in paired])

    print(f"\n{len(paired)} clips scored by both arms")
    print(f"{'arm':14}{'coverage':>12}{'visible only':>15}")
    for name in names:
        print(f"{name:14}{means[name]:>12.4f}{visible[name]:>15.4f}")
    print(f"{'difference':14}{means['samara'] - means['sam']:>+12.4f}{visible['samara'] - visible['sam']:>+15.4f}   "
          f"samara better on {int((difference > 0).sum())}, worse on {int((difference < 0).sum())}, "
          f"tied on {int((difference == 0).sum())}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.pkl").write_bytes(pickle.dumps(scores))


@hydra.main(config_path="../conf", config_name="experiments/samara_deploy", version_base=None)
def deploy(config: DictConfig):
    """Train the calibrator on the trajectories not deployed on, then roll both arms over every clip."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    max_frames, threshold = int(config.max_frames), float(config.coverage_threshold)
    out_dir = Path(config.out_dir)

    clips = deployment_clips(person_path, detection_data, int(config.n_traj), max_frames)
    deployed_stems = {f"{video}_{person}" for video, person, _ in clips}

    trainer = hydra.utils.instantiate(config.offline_trainers.main_trainer)
    stems = np.array(trainer.dataset.trajectories())
    is_deployed = np.array([stem in deployed_stems for stem in stems])
    train_index, test_index = np.flatnonzero(~is_deployed), np.flatnonzero(is_deployed)

    print(f"{len(clips)} deployment clips over {len({video for video, _, _ in clips})} videos")
    print(f"calibrator: {len(train_index)} training trajectories, {len(test_index)} held out "
          f"(the deployed trajectories themselves)")
    print(f"training on p={list(trainer.dataset.probabilities)}\n", flush=True)
    if len(test_index) == 0:
        raise SystemExit("no deployed trajectory is in the calibrator dataset: nothing is held out")

    trainer.custom_train(x=stems, y=stems, train_indices=list(train_index),
                         validation_indices=list(test_index), fold=0)

    samara = hydra.utils.instantiate(config.tracker.tracker)
    samara.model.controller = trainer.model

    # With `separate_gate`, the commit gate gets its OWN model trained only on the binary target, on the same
    # trajectory split so it never sees a deployed clip. Otherwise the gate reads the selector's gate column.
    if config.get("separate_gate", False):
        gate_trainer = hydra.utils.instantiate(config.gate_trainers.main_trainer)
        print(f"\ntraining the gate separately on p={list(gate_trainer.dataset.probabilities)}", flush=True)
        gate_trainer.custom_train(x=stems, y=stems, train_indices=list(train_index),
                                  validation_indices=list(test_index), fold=0)
        samara.model.gate_controller = gate_trainer.model

    samara.model.eval()
    baseline = hydra.utils.instantiate(OmegaConf.load(config.baseline_config).tracker)
    arms = [("sam", baseline), ("samara", samara)]

    print(f"\ndeploying both arms on {len(clips)} clips", flush=True)
    print(f"  sam    : {type(baseline).__name__}, commit gate object>0.5 and predicted IoU>{baseline.iou_threshold}")
    gate_source = "separate model" if samara.model.gate_controller is not None else "selector head"
    print(f"  samara : {type(samara).__name__}, select={samara.select} gate={samara.gate} "
          f"threshold={samara.commit_threshold}  (gate from {gate_source})", flush=True)

    scores = []
    progress = tqdm(clips, desc="rollout")
    for video, person, anchor in progress:
        if load_window(detection_data, (video, person, anchor), max_frames) is None:
            continue

        cache = {}                                  # shared by both arms, dropped when the clip is done
        row = {"video": video, "person": person, "anchor": anchor}
        for name, tracker in arms:
            tracker.frame_cache = cache
            row[name], row[f"{name}_visible"] = clip_coverage(tracker, detection_data, threshold)
        scores.append(row)

        finished = [r for r in scores if all(np.isfinite(r[name]) for name in ARM_NAMES)]
        if finished:
            progress.set_postfix(sam=f"{np.mean([r['sam'] for r in finished]):.3f}",
                                 samara=f"{np.mean([r['samara'] for r in finished]):.3f}", n=len(finished))
        if len(scores) % 5 == 0:
            report(scores, out_dir)

    report(scores, out_dir)
    print(f"\ncoverage = every annotated post-occlusion frame (visible: box IoU >= {threshold:g}  ·  "
          f"occluded: did not commit)  ·  per-clip scores in {out_dir / 'results.pkl'}")


if __name__ == "__main__":
    deploy()

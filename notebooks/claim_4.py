"""claim_4: train the calibrator with cross-validation and judge it on the decision it exists to make --
ranking the three proposals SAM emits each frame.

The objective is regression on a proposal's true mask IoU. A classifier answers "is this mask above
threshold", which is the commit-gate question; ranking needs an ordering, and on the frames where the
choice matters all three proposals usually sit on the same side of any fixed threshold.

Folds are split by TRAJECTORY. A trajectory appears once per corruption folder as a near-identical rollout
of the same clip, and `MainDataset` pulls those together from one index, so the copies cannot land on
opposite sides of a fold. Trajectories from the same video still can: people in one scene share lighting,
camera and background, and often appear as each other's distractors, so these scores are optimistic
relative to a new-scene deployment. Each fold reports how many of its test videos were also trained on.

The held-out fold doubles as epochalyst's validation set, so early stopping reads the trajectories the fold
is scored on. Across the folds every trajectory is held out exactly once, so nothing is permanently unseen
-- the cost is that the stopping epoch is chosen on the fold being reported.

Training uses every LABELLED frame (visible, annotated), pre-occlusion ones included. Evaluation stays on
the narrower post-occlusion set, excluding the anchor frame the similarity maps are measured against.
Trained on the corruption levels in `dataset.probabilities`; every level on disk is evaluated, so the ones
absent from that list are pure extrapolation.
"""
import logging
import os
import pickle
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root on path for `src`

from src.utils.compute_iou import compute_iou
from src.metrics import coverage_auc


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

USABLE_IOU = 0.2           # a proposal is usable when its true mask IoU clears this
METRIC_NAMES = ("AUC", "usable")


# ---------------------------------------------------------------------------------------
# scoring the held-out trajectories
# ---------------------------------------------------------------------------------------

def calibrator_scores(model, similarity_maps, batch_size=256):
    """The calibrator's score for every proposal: (frames, proposals) from the frame's similarity maps."""

    n_proposals = similarity_maps.shape[1]
    per_proposal = similarity_maps.reshape(-1, 1, *similarity_maps.shape[2:])
    features = torch.from_numpy(per_proposal)
    device = next(model.parameters()).device

    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start:start + batch_size].to(device)
            chunks.append(model(batch).reshape(-1))

    scores = torch.cat(chunks).cpu().numpy()
    return scores.reshape(-1, n_proposals)


def held_out_scores(model, dataset, stems):
    """(calibrator, token, truth), each (frames, proposals), pooled over the given trajectories."""

    model.eval()
    folder = dataset.folders()[0]

    calibrator_columns = []
    token_columns = []
    truth_columns = []

    for stem in stems:
        experiment = pickle.load(open(folder / f"{stem}.pkl", "rb"))
        keep = dataset.scorable(experiment)
        if not keep.any():
            continue

        similarity_maps = np.asarray(experiment.features, np.float32)[keep]
        calibrator_columns.append(calibrator_scores(model, similarity_maps))
        token_columns.append(np.asarray(experiment.proposal_iou_scores, np.float32)[keep])
        truth_columns.append(np.asarray(experiment.iou_scores, np.float32)[keep])

    calibrator = np.concatenate(calibrator_columns)
    token = np.concatenate(token_columns)
    truth = np.concatenate(truth_columns)
    return calibrator, token, truth


# ---------------------------------------------------------------------------------------
# ranking: does the calibrator order the three proposals better than SAM's own IoU token?
# ---------------------------------------------------------------------------------------

def ranking(predicted, truth):
    """How well one score separates USABLE proposals from unusable ones.

    A proposal is usable when its true mask IoU clears `USABLE_IOU`, and the metric is the probability the
    score ranks a usable proposal above an unusable one -- an AUC over every proposal, pooled. Chance is
    0.50.

    This asks a different question from ranking the three proposals against each other. Selection needs an
    ordering WITHIN a frame; the gate needs to know whether the mask in hand is worth committing at all,
    which is a decision about one proposal against a threshold. On the frames where all three proposals are
    bad, the ordering is irrelevant and this is the only question left."""

    truth = np.asarray(truth, float).reshape(-1)
    predicted = np.asarray(predicted, float).reshape(-1)

    usable = truth > USABLE_IOU
    keep = np.isfinite(truth) & np.isfinite(predicted)
    usable, predicted = usable[keep], predicted[keep]

    # AUC is undefined when every proposal falls on the same side of the threshold.
    single_class = usable.all() or not usable.any()
    area = np.nan if single_class else float(roc_auc_score(usable, predicted))

    return {
        "AUC": area,
        "usable": float(usable.mean()) if len(usable) else np.nan,
        "n": int(len(usable)),
    }


def fold_mean(rows):
    """Mean of each metric across folds."""

    return {metric: float(np.mean([row[metric] for row in rows])) for metric in rows[0]}


def format_metrics(metrics):
    """One score's half of a summary row."""

    return f"{metrics['AUC']:>12.3f}{metrics['usable']:>12.1%}"


def summarise(rows, levels, trained_on, folds):
    """Fold-mean table, calibrator against SAM's IoU token, one row per corruption level."""

    heading = f"{'AUC':>12}{'usable':>12}"
    print()
    print(f"{'':7}{'calibrator (MSE, cross-validated)':>24}{'SAM IoU token':>24}")
    print(f"{'p':7}{heading}{heading}   trained")

    for level in levels:
        if not rows[level]:
            continue

        calibrator = fold_mean([calibrator for calibrator, _ in rows[level]])
        token = fold_mean([token for _, token in rows[level]])
        seen = "yes" if level in trained_on else "NO (extrapolation)"
        print(f"p{level:<6.2f}{format_metrics(calibrator)}{format_metrics(token)}   {seen}")

    print()
    print(f"fold means over {folds} trajectory-grouped folds; AUC = P(the score ranks a usable "
          f"proposal above an unusable one), usable = true mask IoU > {USABLE_IOU:g}, chance 0.500")


def report_fold(level, calibrator, token):
    """One line per corruption level inside a fold."""

    print(f"      p{level:.2f}: cnn AUC {calibrator['AUC']:.3f}   sam AUC {token['AUC']:.3f}   "
          f"usable {calibrator['usable']:.1%}   n={calibrator['n']}", flush=True)


# ---------------------------------------------------------------------------------------
# deployment: the calibrator inside the tracker
# ---------------------------------------------------------------------------------------

def stream_metrics(tracker, trajectories, detection_data):
    """Run the tracker on each held-out trajectory and average its post-occlusion coverage."""

    coverages = []
    progress = tqdm(trajectories, desc="Coverage")

    for stem in progress:
        video_name, person_id = stem.rsplit("_", 1)
        detection_data.initialize_target(video_name, int(person_id))
        predicted_masks = tracker.predict_masks(detection_data).numpy()

        iou_scores = compute_iou(detection_data.bboxes_norm, predicted_masks)
        iou_scores[detection_data.occlusions > 0.5] = 0.0
        coverage = coverage_auc(iou_scores, detection_data.occlusions)
        if not np.isnan(coverage):
            coverages.append(coverage)

        running_mean = np.mean(coverages) if coverages else float("nan")
        progress.set_postfix(avg_coverage_auc=running_mean)

    return float(np.mean(coverages)) if coverages else float("nan")


def deployed_coverage(config, model, held_out):
    """Coverage AUC with this fold's calibrator installed as the tracker's controller."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    tracker = hydra.utils.instantiate(config.tracker.tracker)

    host = getattr(tracker, "model", None)
    if host is not None and hasattr(host, "controller"):
        host.controller = model
        host.eval()

    return stream_metrics(tracker, held_out, detection_data)


# ---------------------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------------------

def corruption_levels(dataset_path):
    """Every corruption level on disk, read off the `pXX` folder names."""

    folders = [folder for folder in Path(dataset_path).iterdir() if folder.is_dir()]
    return sorted(float(folder.name[1:]) for folder in folders)


def score_level(trainer, level, held_out):
    """The calibrator and SAM's token, both ranked against the truth, at one corruption level."""

    dataset = deepcopy(trainer.dataset)
    dataset.probabilities = [level]

    calibrator_score, token_score, truth = held_out_scores(trainer.model, dataset, held_out)
    calibrator = ranking(calibrator_score, truth)
    token = ranking(token_score, truth)
    return calibrator, token, calibrator_score, token_score, truth


@hydra.main(config_path="../conf", config_name="experiments/claim_4", version_base=None)
def train_models(config: DictConfig):
    """Cross-validate the calibrator and report ranking quality at every corruption level."""

    folds = int(config.get("folds", 5))
    template = hydra.utils.instantiate(config.offline_trainers.main_trainer)
    stems = np.array(template.dataset.trajectories())
    videos = [stem.rsplit("_", 1)[0] for stem in stems]

    levels = corruption_levels(template.dataset.dataset_path)
    trained_on = [float(probability) for probability in template.dataset.probabilities]

    print(f"{len(stems)} trajectories over {len(set(videos))} videos")
    print(f"training on p={trained_on}, evaluating on p={levels}\n", flush=True)

    rows = {level: [] for level in levels}
    predictions = {level: [] for level in levels}
    splits = GroupKFold(n_splits=folds).split(stems, groups=stems)

    for fold, (train_indices, test_indices) in enumerate(splits):
        test_videos = {videos[index] for index in test_indices}
        seen_videos = {videos[index] for index in train_indices}
        overlap = f"{len(test_videos & seen_videos)}/{len(test_videos)} test videos also trained on"
        print(f"fold {fold + 1}/{folds}: {len(train_indices)} train / {len(test_indices)} held-out "
              f"({overlap})", flush=True)

        trainer = hydra.utils.instantiate(config.offline_trainers.main_trainer)
        trainer.custom_train(x=stems, y=stems, train_indices=list(train_indices), validation_indices=list(test_indices), fold=fold)
        held_out = stems[test_indices]

        for level in levels:
            calibrator, token, calibrator_score, token_score, truth = score_level(trainer, level, held_out)
            predictions[level].append({"fold": fold, "trajectories": list(held_out), "truth": truth, "cnn": calibrator_score, "sam": token_score})

            rows[level].append((calibrator, token))
            report_fold(level, calibrator, token)

        if config.deploy_controller:
            coverage = deployed_coverage(config, trainer.model, held_out)
            print(f"      deployed coverage AUC: {coverage:.4f}", flush=True)

    destination = Path(config.predictions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pickle.dumps(predictions))

    summarise(rows, levels, trained_on, folds)
    print(f"held-out predictions saved to {destination}")


if __name__ == "__main__":
    train_models()

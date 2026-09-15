"""Train the calibrator with 5-fold cross-validation and judge it on the decision it exists to make:
ranking the three proposals SAM emits each frame.

The objective is regression on a proposal's true mask IoU. A classifier answers "is this mask above
threshold", which is the commit-gate question; ranking needs an ordering, and on the frames where the choice
matters all three proposals usually sit on the same side of any fixed threshold.

Folds are split by TRAJECTORY. A trajectory appears once per corruption folder as a near-identical rollout
of the same clip, and `MainDataset` pulls those together from one index, so the copies cannot land on
opposite sides of a fold. Trajectories from the same video still can: people in one scene share lighting,
camera and background, and often appear as each other's distractors, so these scores are optimistic
relative to a new-scene deployment. Each fold reports how many of its test videos were also trained on.

The held-out fold doubles as epochalyst's validation set, so early stopping reads the trajectories the fold
is scored on. Across five folds every trajectory is held out exactly once, so nothing is permanently
unseen -- the cost is that the stopping epoch is chosen on the fold being reported.

Training uses every LABELLED frame (visible, annotated), pre-occlusion ones included. Evaluation stays on
the narrower post-occlusion set, excluding the anchor frame the similarity maps are measured against.
Trained on the corruption levels in `dataset.probabilities`; every level on disk is evaluated, so the ones
absent from that list are pure extrapolation.
"""
import logging
import os
import pickle
import warnings
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.utils.compute_iou import compute_iou
from src.metrics import coverage_auc


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

MARGIN = 0.2               # a frame counts only when the best and second-best proposal differ by this much
METRICS = ("agree", "regret", "picked", "R2", "R2 in")
PREDICTIONS = Path("data/calibrator_cv_predictions.pkl")


# ---------------------------------------------------------------------------------------
# scoring the held-out trajectories
# ---------------------------------------------------------------------------------------

def calibrator_scores(model, similarity_maps, batch=256):
    """The calibrator's score for every proposal: (frames, proposals) from the frame's similarity maps."""

    proposals = similarity_maps.shape[1]
    features = torch.from_numpy(similarity_maps.reshape(-1, 1, *similarity_maps.shape[2:]))
    device = next(model.parameters()).device
    starts = range(0, len(features), batch)

    with torch.no_grad():
        chunks = [model(features[start:start + batch].to(device)).reshape(-1) for start in starts]
    return torch.cat(chunks).cpu().numpy().reshape(-1, proposals)


def held_out_scores(model, dataset, stems):
    """(calibrator, token, truth), each (frames, proposals), pooled over the given trajectories."""

    model.eval()
    calibrator, token, truth = [], [], []
    for stem in stems:
        experiment = pickle.load(open(dataset.folders()[0] / f"{stem}.pkl", "rb"))
        keep = dataset.scorable(experiment)
        if not keep.any():
            continue
        similarity_maps = np.asarray(experiment.features, np.float32)[keep]
        calibrator.append(calibrator_scores(model, similarity_maps))
        token.append(np.asarray(experiment.proposal_iou_scores, np.float32)[keep])
        truth.append(np.asarray(experiment.iou_scores, np.float32)[keep])
    return np.concatenate(calibrator), np.concatenate(token), np.concatenate(truth)


# ---------------------------------------------------------------------------------------
# ranking: does the calibrator order the three proposals better than SAM's own IoU token?
# ---------------------------------------------------------------------------------------

def r_squared(truth, predicted):
    """Pooled and WITHIN-FRAME R^2 against the true IoU.

    Pooled R^2 is a calibration measure, and most of its variance sits BETWEEN frames -- easy frames where
    every proposal is good, hard ones where none is. Ranking never asks that question. Centring both columns
    on each frame's own mean strips the between-frame part and leaves exactly what a selector uses, so the
    two can disagree sharply: a score can track frame difficulty beautifully and still be useless at
    ordering the three masks inside a frame."""

    def fit(observed, estimated):
        residual = float(((observed - estimated) ** 2).sum())
        total = float(((observed - observed.mean()) ** 2).sum())
        return 1.0 - residual / total if total > 0 else np.nan

    centred_truth = (truth - truth.mean(1, keepdims=True)).reshape(-1)
    centred_predicted = (predicted - predicted.mean(1, keepdims=True)).reshape(-1)
    return fit(truth.reshape(-1), predicted.reshape(-1)), fit(centred_truth, centred_predicted)


def ranking(predicted, truth):
    """How well one score orders the proposals, over the frames where the choice actually matters.

    The R^2 pair covers ALL frames rather than the high-margin ones: it measures calibration, and
    restricting it to frames picked out by the label would condition it on the label."""

    pooled, within_frame = r_squared(truth, predicted)
    ordered = np.sort(truth, axis=1)[:, ::-1]
    matters = (ordered[:, 0] - ordered[:, 1]) >= MARGIN

    picked = predicted.argmax(1)
    got = truth[np.arange(len(truth)), picked]
    return {"agree": float((picked[matters] == truth.argmax(1)[matters]).mean()),
            "regret": float((truth.max(1) - got)[matters].mean()),
            "picked": float(got[matters].mean()),
            "R2": pooled,
            "R2 in": within_frame,
            "n": int(matters.sum())}


def fold_mean(rows):
    """Mean of each metric across folds."""

    return {metric: float(np.mean([row[metric] for row in rows])) for metric in rows[0]}


def format_metrics(metrics):
    """One score's half of a summary row."""

    return f"{metrics['agree']:>9.1%}{metrics['regret']:>9.3f}{metrics['picked']:>9.3f}{metrics['R2']:>8.3f}{metrics['R2 in']:>8.3f}"


def summarise(rows, levels, trained_on, folds):
    """Fold-mean table, calibrator against SAM's IoU token, one row per corruption level."""

    heading = f"{'agree':>9}{'regret':>9}{'picked':>9}{'R2':>8}{'R2 in':>8}"
    print(f"\n{'':7}{'calibrator (MSE, 5-fold)':>42}{'SAM IoU token':>42}")
    print(f"{'p':7}{heading}{heading}   trained")

    for level in levels:
        if not rows[level]:
            continue
        calibrator = fold_mean([calibrator for calibrator, _ in rows[level]])
        token = fold_mean([token for _, token in rows[level]])
        seen = "yes" if level in trained_on else "NO (extrapolation)"
        print(f"p{level:<6.2f}{format_metrics(calibrator)}{format_metrics(token)}   {seen}")

    print(f"\nfold means over {folds} trajectory-grouped folds; agree/regret/picked on frames with margin >= {MARGIN:g} (chance 33.3%)")
    print("R2 = pooled, 'R2 in' = within-frame, both over all held-out frames")


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
        progress.set_postfix(avg_coverage_auc=(np.mean(coverages) if coverages else float("nan")))
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

@hydra.main(config_path="conf", config_name="offline_training", version_base=None)
def train_models(config: DictConfig):
    """Cross-validate the calibrator and report ranking quality at every corruption level."""

    folds = int(config.get("folds", 5))
    template = hydra.utils.instantiate(config.offline_trainers.main_trainer)
    stems = np.array(template.dataset.trajectories())
    videos = [stem.rsplit("_", 1)[0] for stem in stems]

    folders = [folder for folder in Path(template.dataset.dataset_path).iterdir() if folder.is_dir()]
    levels = sorted(float(folder.name[1:]) for folder in folders)
    trained_on = [float(probability) for probability in template.dataset.probabilities]

    print(f"{len(stems)} trajectories over {len(set(videos))} videos")
    print(f"training on p={trained_on}, evaluating on p={levels}\n", flush=True)

    rows = {level: [] for level in levels}
    predictions = {level: [] for level in levels}
    splits = GroupKFold(n_splits=folds).split(stems, groups=stems)

    for fold, (train_index, test_index) in enumerate(splits):
        test_videos = {videos[index] for index in test_index}
        seen_videos = {videos[index] for index in train_index}
        overlap = f"{len(test_videos & seen_videos)}/{len(test_videos)} test videos also trained on"
        print(f"fold {fold + 1}/{folds}: {len(train_index)} train / {len(test_index)} held-out ({overlap})", flush=True)


        trainer = hydra.utils.instantiate(config.offline_trainers.main_trainer)
        trainer.custom_train(x=stems, y=stems, train_indices=list(train_index), validation_indices=list(test_index), fold=fold)
        held_out = stems[test_index]

        for level in levels:
            dataset = deepcopy(trainer.dataset)
            dataset.probabilities = [level]
            calibrator_score, token_score, truth = held_out_scores(trainer.model, dataset, held_out)
            calibrator, token = ranking(calibrator_score, truth), ranking(token_score, truth)

            rows[level].append((calibrator, token))
            fold_predictions = {"fold": fold, "trajectories": list(held_out), "truth": truth,
                                "cnn": calibrator_score, "sam": token_score}
            predictions[level].append(fold_predictions)

            report = f"cnn {calibrator['agree']:.1%} regret {calibrator['regret']:.3f} R2 {calibrator['R2']:.3f}/{calibrator['R2 in']:.3f}"
            against = f"sam {token['agree']:.1%} regret {token['regret']:.3f} R2 {token['R2']:.3f}/{token['R2 in']:.3f}"
            print(f"      p{level:.2f}: {report}   {against}   n={calibrator['n']}", flush=True)

        if config.deploy_controller:
            coverage = deployed_coverage(config, trainer.model, held_out)
            print(f"      deployed coverage AUC: {coverage:.4f}", flush=True)

    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.write_bytes(pickle.dumps(predictions))
    summarise(rows, levels, trained_on, folds)
    print(f"held-out predictions saved to {PREDICTIONS}")


if __name__ == "__main__":
    train_models()

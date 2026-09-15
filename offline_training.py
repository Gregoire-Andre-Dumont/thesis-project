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
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.offline_training.main_dataset import collate_fn
from src.utils.compute_iou import compute_iou
from src.metrics import coverage_auc


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

MARGIN = 0.2               # a frame counts only when the best and second-best proposal differ by this much
PREDICTIONS = Path("data/calibrator_cv_predictions.pkl")


def predict_on_dataset(model, dataset, batch_size=256):
    """Run the trained calibrator over every sample of the dataset in order.
    Returns the stacked raw predictions, one per proposal."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in loader:
            predictions.append(model(features.to(device)).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def evaluate_calibrator(trainer, test_indices):
    """Regression quality on the held-out trajectories: error against the true mask IoU."""

    val_dataset = deepcopy(trainer.dataset)
    val_dataset.initialize(test_indices)

    predictions = predict_on_dataset(trainer.model, val_dataset).reshape(-1)
    truth = val_dataset._labels.cpu().numpy().reshape(-1)

    mse = float(np.mean((predictions - truth) ** 2))
    mae = float(np.mean(np.abs(predictions - truth)))
    variance = float(np.mean((truth - truth.mean()) ** 2))
    print(f"      calibration: MSE={mse:.4f}  MAE={mae:.4f}  "
          f"R2={1.0 - mse / variance if variance > 0 else float('nan'):.4f}", flush=True)


# ---------------------------------------------------------------------------------------
# ranking: does the calibrator order the three proposals better than SAM's own IoU token?
# ---------------------------------------------------------------------------------------

def clip_scores(model, dataset, stems, batch=256):
    """Per-frame (n, 3) arrays of calibrator score, SAM token and true IoU for each held-out trajectory."""

    device = next(model.parameters()).device
    model.eval()
    scored = []
    for stem in stems:
        experiment = pickle.load(open(dataset.folders()[0] / f"{stem}.pkl", "rb"))
        truth = np.asarray(experiment.iou_scores, dtype=np.float32)
        if truth.ndim != 2:                            # single-proposal pickle from an older schema
            continue
        keep = dataset.scorable(experiment)
        if not keep.any():
            continue

        maps = np.asarray(experiment.features, dtype=np.float32)[keep]
        features = torch.from_numpy(maps.reshape(-1, 1, *maps.shape[2:]))
        predictions = []
        with torch.no_grad():
            for start in range(0, len(features), batch):
                predictions.append(model(features[start:start + batch].to(device)).reshape(-1).cpu().numpy())
        scored.append({
            "cnn": np.concatenate(predictions).reshape(-1, 3),
            "sam": np.asarray(experiment.proposal_iou_scores, dtype=np.float32)[keep],
            "truth": truth[keep],
        })
    return scored


def r_squared(truth, predicted):
    """Pooled and WITHIN-FRAME R^2 against the true IoU.

    Pooled R^2 is a calibration measure, and most of its variance sits BETWEEN frames -- easy frames where
    every proposal is good, hard ones where none is. Ranking never asks that question. Centring both columns
    on each frame's own mean strips the between-frame part and leaves exactly what a selector uses, so the
    two can disagree sharply: a score can track frame difficulty beautifully and still be useless at
    ordering the three masks inside a frame."""

    def fit(y, yhat):
        residual = float(((y - yhat) ** 2).sum())
        total = float(((y - y.mean()) ** 2).sum())
        return 1.0 - residual / total if total > 0 else np.nan

    centred_truth = (truth - truth.mean(1, keepdims=True)).reshape(-1)
    centred_predicted = (predicted - predicted.mean(1, keepdims=True)).reshape(-1)
    return fit(truth.reshape(-1), predicted.reshape(-1)), fit(centred_truth, centred_predicted)


def ranking(scored, key):
    """(agreement, regret, picked IoU, pooled R^2, within-frame R^2, n) for one score column."""

    truth = np.concatenate([s["truth"] for s in scored])
    predicted = np.concatenate([s[key] for s in scored])
    ordered = np.sort(truth, axis=1)[:, ::-1]
    matters = (ordered[:, 0] - ordered[:, 1]) >= MARGIN
    if not matters.any():
        return (np.nan,) * 5 + (0,)

    picked = predicted.argmax(1)
    got = truth[np.arange(len(truth)), picked]
    # R^2 covers ALL held-out frames: it measures calibration, and restricting it to high-margin frames
    # would condition it on the label.
    pooled, centred = r_squared(truth, predicted)

    agreement = float((picked[matters] == truth.argmax(1)[matters]).mean())
    regret = float((truth.max(1) - got)[matters].mean())
    picked_iou = float(got[matters].mean())
    return agreement, regret, picked_iou, pooled, centred, int(matters.sum())


def summarise(rows, levels, trained_on, folds):
    """Fold-mean table, calibrator against SAM's IoU token, one row per corruption level."""

    print(f"\n{'':7}{'calibrator (MSE, 5-fold)':>42}{'SAM IoU token':>42}")
    print(f"{'p':7}{'agree':>9}{'regret':>9}{'picked':>9}{'R2':>8}{'R2 in':>8}"
          f"{'agree':>9}{'regret':>9}{'picked':>9}{'R2':>8}{'R2 in':>8}   trained")
    for level in levels:
        values = np.array([r for r in rows[level] if np.isfinite(r[0])], dtype=float)
        if not len(values):
            continue
        m = values.mean(axis=0)
        seen = "yes" if level in trained_on else "NO (extrapolation)"
        print(f"p{level:<6.2f}{m[0]:>9.1%}{m[1]:>9.3f}{m[2]:>9.3f}{m[3]:>8.3f}{m[4]:>8.3f}"
              f"{m[5]:>9.1%}{m[6]:>9.3f}{m[7]:>9.3f}{m[8]:>8.3f}{m[9]:>8.3f}   {seen}")
    print(f"\nfold means over {folds} trajectory-grouped folds; agreement/regret on frames with "
          f"margin >= {MARGIN:g} (chance 33.3%); R2 = pooled, 'R2 in' = within-frame, all held-out frames")


def stream_metrics(tracker, trajectories, detection_data):
    """Run the tracker on each held-out trajectory and measure its post-occlusion coverage."""

    coverages = []
    pbar = tqdm(trajectories, desc="Coverage")
    for stem in pbar:
        video_name, person_id = stem.rsplit("_", 1)
        detection_data.initialize_target(video_name, int(person_id))
        predicted_masks = tracker.predict_masks(detection_data).numpy()

        iou_scores = compute_iou(detection_data.bboxes_norm, predicted_masks)
        iou_scores[detection_data.occlusions > 0.5] = 0.0

        coverage = coverage_auc(iou_scores, detection_data.occlusions)
        if not np.isnan(coverage):
            coverages.append(coverage)
        pbar.set_postfix(avg_coverage_auc=(np.mean(coverages) if coverages else float("nan")))
    return float(np.mean(coverages)) if coverages else float("nan")


@hydra.main(config_path="conf", config_name="offline_training", version_base=None)
def train_models(config: DictConfig):
    """Cross-validate the calibrator and report ranking quality at every corruption level."""

    folds = int(config.get("folds", 5))
    template = hydra.utils.instantiate(config.offline_trainers.main_trainer)
    stems = template.dataset.trajectories()
    videos = [stem.rsplit("_", 1)[0] for stem in stems]
    folders = [folder for folder in Path(template.dataset.dataset_path).iterdir() if folder.is_dir()]
    levels = sorted(float(folder.name[1:]) for folder in folders)
    trained_on = [float(probability) for probability in template.dataset.probabilities]

    print(f"{len(stems)} trajectories over {len(set(videos))} videos")
    print(f"training on p={trained_on}, evaluating on p={levels}\n", flush=True)

    rows = {level: [] for level in levels}
    predictions = {level: [] for level in levels}

    for fold, (train_index, test_index) in enumerate(GroupKFold(n_splits=folds).split(stems, groups=stems)):
        test_videos = {videos[i] for i in test_index}
        seen_videos = {videos[i] for i in train_index}
        print(f"fold {fold + 1}/{folds}: {len(train_index)} train / {len(test_index)} held-out trajectories "
              f"({len(test_videos & seen_videos)}/{len(test_videos)} test videos also trained on)", flush=True)

        trainer = hydra.utils.instantiate(config.offline_trainers.main_trainer)
        trainer._fold = fold
        all_stems = np.array(stems)
        trainer.custom_train(x=all_stems, y=all_stems, train_indices=list(train_index), validation_indices=list(test_index))
        evaluate_calibrator(trainer, list(test_index))

        held_out = [stems[i] for i in test_index]
        for level in levels:
            dataset = deepcopy(trainer.dataset)
            dataset.probabilities = [level]
            scored = clip_scores(trainer.model, dataset, held_out)
            if not scored:
                continue
            cnn, sam = ranking(scored, "cnn"), ranking(scored, "sam")
            rows[level].append(cnn[:5] + sam[:5] + (cnn[5],))
            predictions[level].append({"fold": fold, "trajectories": held_out, "scored": scored})
            print(f"      p{level:.2f}: cnn {cnn[0]:.1%} (regret {cnn[1]:.3f}, R2 {cnn[3]:.3f}/{cnn[4]:.3f})"
                  f"   sam {sam[0]:.1%} (regret {sam[1]:.3f}, R2 {sam[3]:.3f}/{sam[4]:.3f})"
                  f"   n={cnn[5]}", flush=True)

        if config.deploy_controller:
            detection_data = hydra.utils.instantiate(config.detection_data)
            tracker = hydra.utils.instantiate(config.tracker.tracker)
            if hasattr(tracker, "model") and tracker.model is not None and hasattr(tracker.model, "controller"):
                tracker.model.controller = trainer.model
                tracker.model.eval()
            print(f"      deployed coverage AUC: {stream_metrics(tracker, held_out, detection_data):.4f}",
                  flush=True)

    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.write_bytes(pickle.dumps(predictions))
    summarise(rows, levels, trained_on, folds)
    print(f"held-out predictions saved to {PREDICTIONS}")


if __name__ == "__main__":
    train_models()

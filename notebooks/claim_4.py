"""claim_4: train the calibrator with cross-validation and judge it on the decision it exists to make --
ranking the three proposals SAM emits each frame."""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.compute_iou import compute_iou
from src.metrics import coverage_auc

logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

USABLE_IOU = 0.2
METRIC_NAMES = ("AUC", "usable")


def calibrator_scores(trained_model, similarity_maps, batch_size=256):
    """The calibrator's score for every proposal: (frames, proposals) from the frame's similarity maps."""

    n_proposals = similarity_maps.shape[1]
    per_proposal = similarity_maps.reshape(-1, 1, *similarity_maps.shape[2:])
    features = torch.from_numpy(per_proposal)
    device = next(trained_model.parameters()).device

    chunks = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(device)
            chunks.append(trained_model(batch).reshape(-1))

    scores = torch.cat(chunks).cpu().numpy()
    return scores.reshape(-1, n_proposals)


def held_out_scores(trained_model, dataset, trajectory_stems):
    """(calibrator, token, object score, truth), each (frames, proposals), pooled over the trajectories.

    SAM's OBJECT SCORE joins its IoU token as a second untrained reference. The two are not interchangeable:
    the token estimates how good a proposal's mask is, the object score whether the target is in the frame
    at all -- and the object score turned out to be the stronger absence signal at deployment, so it belongs
    in any comparison that claims the learned score beats SAM's own. It is one value per FRAME, broadcast
    across that frame's proposals, so it can only separate frames and never rank within one."""

    trained_model.eval()
    level_folder = dataset.folders()[0]

    calibrator_columns = []
    token_columns = []
    object_columns = []
    truth_columns = []

    for trajectory_stem in trajectory_stems:
        experiment = pickle.load(open(level_folder / f"{trajectory_stem }.pkl", "rb"))
        kept_indices = dataset.scorable(experiment)
        if not kept_indices.any():
            continue

        similarity_maps = np.asarray(experiment.features, np.float32)[kept_indices]
        calibrator_columns.append(calibrator_scores(trained_model, similarity_maps))
        token_columns.append(np.asarray(experiment.proposal_iou_scores, np.float32)[kept_indices])
        truth_columns.append(np.asarray(experiment.iou_scores, np.float32)[kept_indices])

        per_frame_object = np.asarray(experiment.object_score, np.float32).reshape(-1)[kept_indices]
        object_columns.append(np.repeat(per_frame_object[:, None], similarity_maps.shape[1], axis=1))

    calibrator = np.concatenate(calibrator_columns)
    iou_token = np.concatenate(token_columns)
    object_score = np.concatenate(object_columns)
    ground_truth_iou = np.concatenate(truth_columns)
    return calibrator, iou_token, object_score, ground_truth_iou


def ranking(predicted, ground_truth_iou):
    """How well one score separates USABLE proposals from unusable ones."""

    ground_truth_iou = np.asarray(ground_truth_iou, float).reshape(-1)
    predicted = np.asarray(predicted, float).reshape(-1)

    usable_samples = ground_truth_iou > USABLE_IOU
    kept_indices = np.isfinite(ground_truth_iou) & np.isfinite(predicted)
    usable_samples, predicted = usable_samples[kept_indices], predicted[kept_indices]

    single_class = usable_samples.all() or not usable_samples.any()
    area = np.nan if single_class else float(roc_auc_score(usable_samples, predicted))

    return {
        "AUC": area,
        "usable": float(usable_samples.mean()) if len(usable_samples) else np.nan,
        "n": int(len(usable_samples))}


def fold_mean(sample_rows):
    """Mean of each metric across folds."""

    return {metric: float(np.mean([row[metric] for row in sample_rows])) for metric in sample_rows[0]}


def summarise(sample_rows, corruption_levels, trained_on, number_of_folds):
    """Fold-mean table, calibrator against SAM's IoU token, one row per corruption level."""

    print()
    print(f"{number_of_folds}-fold means, calibrator vs SAM's own IoU token and object score:")

    for corruption_level in corruption_levels:
        if not sample_rows[corruption_level]:
            continue

        calibrator = fold_mean([row[0] for row in sample_rows[corruption_level]])
        iou_token = fold_mean([row[1] for row in sample_rows[corruption_level]])
        object_score = fold_mean([row[2] for row in sample_rows[corruption_level]])
        seen = "yes" if corruption_level in trained_on else "NO (extrapolation)"

        print(f'p{corruption_level:<5.2f} trained_on={seen:<20} '
              f'calibrator AUC {calibrator["AUC"]:.3f}  token {iou_token["AUC"]:.3f}  '
              f'object {object_score["AUC"]:.3f}  usable {calibrator["usable"]:.3f}')



def report_fold(corruption_level, calibrator, iou_token, object_score):
    """One line per corruption level inside a fold."""

    print(f'  p{corruption_level:<5.2f} AUC  calibrator {calibrator["AUC"]:.3f}  '
          f'token {iou_token["AUC"]:.3f}  object {object_score["AUC"]:.3f}  '
          f'(usable {calibrator["usable"]:.3f}, n={calibrator["n"]})')



def stream_metrics(tracker, trajectories, detection_data):
    """Run the tracker on each held-out trajectory and average its post-occlusion coverage."""

    coverages = []
    progress = tqdm(trajectories, desc="Coverage")

    for trajectory_stem in progress:
        video_name, person_id = trajectory_stem.rsplit("_", 1)
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


def deployed_coverage(config, trained_model, held_out):
    """Coverage AUC with this fold's calibrator installed as the tracker's controller."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    tracker = hydra.utils.instantiate(config.tracker.tracker)

    host = getattr(tracker, "model", None)
    if host is not None and hasattr(host, "controller"):
        host.controller = trained_model
        host.eval()

    return stream_metrics(tracker, held_out, detection_data)


def corruption_levels(dataset_path):
    """Every corruption level on disk, read off the `pXX` folder names."""

    folders = [level_folder for level_folder in Path(dataset_path).iterdir() if level_folder.is_dir()]
    return sorted(float(level_folder.name[1:]) for level_folder in folders)


def score_level(trainer, corruption_level, held_out):
    """The calibrator against BOTH of SAM's own scores, ranked on the truth, at one corruption level."""

    dataset = deepcopy(trainer.dataset)
    dataset.probabilities = [corruption_level]

    calibrator_score, token_score, object_score, ground_truth_iou = held_out_scores(
        trainer.model, dataset, held_out)
    return {
        "calibrator": ranking(calibrator_score, ground_truth_iou),
        "iou_token": ranking(token_score, ground_truth_iou),
        "object_score": ranking(object_score, ground_truth_iou),
        "scores": {"cnn": calibrator_score, "sam": token_score, "object": object_score},
        "truth": ground_truth_iou}


@hydra.main(config_path="../conf", config_name="experiments/claim_4", version_base=None)
def train_models(config: DictConfig):
    """Cross-validate the calibrator and report ranking quality at every corruption level."""

    number_of_folds = int(config.get("folds", 5))
    template = hydra.utils.instantiate(config.offline_trainers.main_trainer)
    trajectory_stems = np.array(template.dataset.trajectories())
    videos = [trajectory_stem.rsplit("_", 1)[0] for trajectory_stem in trajectory_stems]

    corruption_levels = corruption_levels(template.dataset.dataset_path)
    trained_on = [float(probability) for probability in template.dataset.probabilities]


    sample_rows = {corruption_level: [] for corruption_level in corruption_levels}
    predictions = {corruption_level: [] for corruption_level in corruption_levels}
    splits = GroupKFold(n_splits=number_of_folds).split(trajectory_stems, groups=trajectory_stems)

    for fold, (train_indices, test_indices) in enumerate(splits):
        test_videos = {videos[sample_index] for sample_index in test_indices}
        seen_videos = {videos[sample_index] for sample_index in train_indices}
        overlap = f"{len(test_videos &seen_videos)}/{len(test_videos)} test videos also trained on"

        trainer = hydra.utils.instantiate(config.offline_trainers.main_trainer)
        trainer.custom_train(
            x=trajectory_stems,
            y=trajectory_stems,
            train_indices=list(train_indices),
            validation_indices=list(test_indices),
            fold=fold)
        held_out = trajectory_stems[test_indices]

        for corruption_level in corruption_levels:
            level = score_level(trainer, corruption_level, held_out)
            predictions[corruption_level].append(
                {
                    "fold": fold,
                    "trajectories": list(held_out),
                    "truth": level["truth"],
                    "cnn": level["scores"]["cnn"],
                    "sam": level["scores"]["sam"],
                    "object": level["scores"]["object"]}
            )

            sample_rows[corruption_level].append(
                (level["calibrator"], level["iou_token"], level["object_score"]))
            report_fold(corruption_level, level["calibrator"], level["iou_token"], level["object_score"])

        if config.deploy_controller:
            coverage = deployed_coverage(config, trainer.model, held_out)

    destination = Path(config.predictions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pickle.dumps(predictions))

    summarise(sample_rows, corruption_levels, trained_on, number_of_folds)


if __name__ == "__main__":
    train_models()

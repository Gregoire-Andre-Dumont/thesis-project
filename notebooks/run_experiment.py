"""Run every tracker in an experiment over its test trajectories and save per-frame IoU.

An "experiment" is a directory `conf/trackers/experiment_<N>/` of tracker configs (one per
variant x backbone size). For each tracker this script:
  1. builds the train/test trajectory split (from the tracker's own trainer dataset if it
     declares one, else the global `dataset_path`),
  2. optionally trains the calibrator (trackers that point at a `trainer_config_path`),
  3. tracks every test trajectory and writes `data/experiment_<N>/<tracker>.pkl`.

CLI selectors (Hydra overrides on top of conf/offline_training.yaml):
  experiment=5                 which conf/trackers/experiment_<N>/ to run
  size=tiny|small|base_plus|large   only that backbone size's tracker(s)
  tracker_name=samite_tiny     a single named tracker (skips the glob)
  max_test_trajectories=20     cap the number of test trajectories (quick runs)
"""

import os
import sys
import gc
import logging
import pickle
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils.compute_iou import compute_iou
from offline_training import build_trajectory_split


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"


def select_tracker_yamls(config, trackers_directory):
    """Resolve which tracker configs to run: a single `tracker_name`, else every yaml in the
    experiment directory, optionally narrowed to one backbone `size`."""
    tracker_name = config.get("tracker_name", None)
    if tracker_name:
        return [trackers_directory / f"{tracker_name}.yaml"]

    tracker_yamls = sorted(trackers_directory.glob("*.yaml"))

    size = config.get("size", None)
    if size:
        tracker_yamls = [y for y in tracker_yamls if y.stem.endswith(f"_{size}")]
        if not tracker_yamls:
            available = sorted(p.stem for p in trackers_directory.glob("*.yaml"))
            raise ValueError(f"no tracker matching size={size!r} in {trackers_directory}. available: {available}")
    return tracker_yamls


def resolve_split(tracker_config, config):
    """Build the trajectory split for one tracker. If the tracker declares a
    `trainer_config_path`, the split is taken from that trainer's dataset (so training and
    evaluation share one split); otherwise the global `dataset_path` is used.

    Returns `(trainer_config, trajectory_paths, train_indices, test_indices)`; `trainer_config`
    is None for trackers that need no calibrator training."""
    trainer_config_path = tracker_config.tracker.get("trainer_config_path", None)
    if trainer_config_path is not None and Path(trainer_config_path).exists():
        trainer_config = OmegaConf.load(trainer_config_path)
        split_dataset_path = trainer_config.main_trainer.dataset.dataset_path
    else:
        trainer_config = None
        split_dataset_path = config.dataset_path

    trajectory_paths, train_indices, test_indices = build_trajectory_split(
        dataset_path=split_dataset_path, test_size=config.train_test_split)
    print(f"  split source: {split_dataset_path}  -> {len(test_indices)}/{len(trajectory_paths)} test trajectories")
    return trainer_config, trajectory_paths, train_indices, test_indices


def train_calibrator(trainer_config, config, trajectory_paths, train_indices, test_indices):
    """Train the tracker's calibrator. An optional `train_fraction` sub-samples the available
    train trajectories (fixed seed) so training-set size can be swept independently of the split."""
    train_fraction = trainer_config.get("train_fraction", None)
    if train_fraction is not None:
        available_train = len(train_indices)
        n_train = int(round(float(train_fraction) * len(trajectory_paths)))
        if n_train > available_train:
            raise ValueError(
                f"train_fraction={train_fraction} requires {n_train} train trajectories but only "
                f"{available_train} are available with train_test_split={config.train_test_split}")
        shuffled = np.random.RandomState(seed=43).permutation(train_indices)
        train_indices = np.sort(shuffled[:n_train])
        print(f"  train_fraction={train_fraction}: using {n_train}/{available_train} train trajectories")

    trainer = hydra.utils.instantiate(trainer_config.main_trainer)
    trainer.custom_train(x=trajectory_paths, y=trajectory_paths,
                         train_indices=train_indices, validation_indices=test_indices)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


def run_tracker(tracker, trajectory_paths, test_indices, detection_data, output_path):
    """Track every test trajectory and append its per-frame IoU (occluded frames zeroed) to
    `output_path`. Resumes by skipping trajectories already saved; writes atomically."""
    label = output_path.stem
    if output_path.exists():
        with open(output_path, "rb") as handle:
            records = pickle.load(handle)["records"]
        done = {(record["video_name"], record["person_id"]) for record in records}
        print(f"resuming {label}: {len(done)} trajectories already saved")
    else:
        records, done = [], set()

    for index in tqdm(test_indices, desc="Trajectories"):
        trajectory_path = trajectory_paths[index]
        video_name, person_id_str = Path(trajectory_path).stem.rsplit("_", 1)
        person_id = int(person_id_str)
        if (video_name, person_id) in done:
            continue

        with open(trajectory_path, "rb") as handle:
            trajectory_metadata = pickle.load(handle)
        detection_data.initialize_target(video_name, person_id, frame_indices=trajectory_metadata.frame_indices)
        predicted_masks = tracker.predict_masks(detection_data).numpy()

        iou_scores = compute_iou(detection_data.bboxes_norm, predicted_masks)
        iou_scores[detection_data.occlusions > 0.5] = 0.0

        records.append({"video_name": video_name,
                        "person_id": person_id,
                        "iou_scores": iou_scores.astype(np.float32),
                        "occlusions": np.asarray(detection_data.occlusions, dtype=np.float32)})
        done.add((video_name, person_id))

        tmp_path = output_path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as handle:
            pickle.dump({"tracker": label, "records": records}, handle)
        tmp_path.replace(output_path)

        gc.collect()
        torch.cuda.empty_cache()

    return records


@hydra.main(config_path="../conf", config_name="offline_training", version_base=None)
def main(config: DictConfig):
    experiment_number = int(config.experiment)
    trackers_directory = Path(f"conf/trackers/experiment_{experiment_number}")
    output_directory = Path(f"data/experiment_{experiment_number}")
    output_directory.mkdir(parents=True, exist_ok=True)

    detection_data = hydra.utils.instantiate(config.detection_data)
    tracker_yamls = select_tracker_yamls(config, trackers_directory)

    for tracker_yaml in tqdm(tracker_yamls, desc="Trackers"):
        tracker_name = tracker_yaml.stem
        print(f"\n=== {tracker_name} ===")
        tracker_config = OmegaConf.load(tracker_yaml)

        trainer_config, trajectory_paths, train_indices, test_indices = resolve_split(tracker_config, config)

        max_test_trajectories = config.get("max_test_trajectories", None)
        if max_test_trajectories is not None:
            test_indices = test_indices[: int(max_test_trajectories)]

        if trainer_config is not None:
            train_calibrator(trainer_config, config, trajectory_paths, train_indices, test_indices)

        tracker = hydra.utils.instantiate(tracker_config.tracker)
        output_path = output_directory / f"{tracker_name}.pkl"
        records = run_tracker(tracker, trajectory_paths, test_indices, detection_data, output_path)
        print(f"saved {len(records)} -> {output_path}")

        del tracker
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

import os
import sys
import gc
import json
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


SAM_RESIZE = 1024
VISIBLE_DIRECTORY = Path("data/person_path/visible")


def _anchor_area_at_chosen_anchor(trajectory_path):
    """Visible-bbox area at the chosen anchor frame for one trajectory pickle, scaled to
    1024-resize. Returns `None` if missing."""

    with open(trajectory_path, "rb") as handle:
        trajectory = pickle.load(handle)
    frame_indices = trajectory.frame_indices
    if frame_indices is None or len(frame_indices) == 0:
        return None
    stem = Path(trajectory_path).stem
    video_name, person_id_str = stem.rsplit("_", 1)
    person_id = int(person_id_str)
    anchor_video_frame = int(frame_indices[0])
    visible_path = VISIBLE_DIRECTORY / f"{video_name}.json"
    if not visible_path.exists():
        return None
    data = json.load(open(visible_path))
    image_width = int(data["metadata"]["resolution"]["width"])
    image_height = int(data["metadata"]["resolution"]["height"])
    scale = SAM_RESIZE / max(image_width, image_height)
    best_bbox, best_diff = None, None
    for entity in data["entities"]:
        if entity["id"] != person_id:
            continue
        diff = abs(int(entity["blob"]["frame_idx"]) - anchor_video_frame)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_bbox = entity["bb"]
    if best_bbox is None:
        return None
    _, _, w, h = best_bbox
    return max(0.0, w) * max(0.0, h) * scale * scale


def run_tracker(tracker, trajectory_paths, test_indices, detection_data, output_path):
    label = output_path.stem
    if output_path.exists():
        with open(output_path, "rb") as handle:
            payload = pickle.load(handle)
        records = payload["records"]
        done = {(record["video_name"], record["person_id"]) for record in records}
        print(f"resuming {label}: {len(done)} trajectories already saved")
    else:
        records = []
        done = set()

    test_trajectories = [trajectory_paths[i] for i in test_indices]
    for trajectory_path in tqdm(test_trajectories, desc="Trajectories"):
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

    trajectory_paths, train_indices, test_indices = build_trajectory_split(
        dataset_path=config.dataset_path, test_size=config.train_test_split)
    print(f"running on {len(test_indices)}/{len(trajectory_paths)} test trajectories")

    max_test_trajectories = config.get("max_test_trajectories", None)
    if max_test_trajectories is not None:
        test_indices = test_indices[: int(max_test_trajectories)]

    detection_data = hydra.utils.instantiate(config.detection_data)

    tracker_name_override = config.get("tracker_name", None)
    if tracker_name_override:
        tracker_yamls = [trackers_directory / f"{tracker_name_override}.yaml"]
    else:
        tracker_yamls = sorted(trackers_directory.glob("*.yaml"))

    for tracker_yaml in tqdm(tracker_yamls, desc="Trackers"):
        tracker_name = tracker_yaml.stem
        print(f"\n=== {tracker_name} ===")
        tracker_config = OmegaConf.load(tracker_yaml)

        trainer_config_path = tracker_config.tracker.get("trainer_config_path", None)
        if trainer_config_path is not None and Path(trainer_config_path).exists():
            trainer_config = OmegaConf.load(trainer_config_path)
            train_fraction = trainer_config.get("train_fraction", None)
            if train_fraction is not None:
                n_train = int(round(float(train_fraction) * len(trajectory_paths)))
                if n_train > len(train_indices):
                    raise ValueError(
                        f"train_fraction={train_fraction} requires {n_train} train trajectories "
                        f"but only {len(train_indices)} are available with "
                        f"train_test_split={config.train_test_split}")
                rng = np.random.RandomState(seed=43)
                shuffled_train_indices = rng.permutation(train_indices)
                effective_train_indices = np.sort(shuffled_train_indices[:n_train])
                print(f"  train_fraction={train_fraction}: using {n_train}/{len(train_indices)} train trajectories")
            else:
                effective_train_indices = train_indices

            anchor_size_filter = trainer_config.get("anchor_size_filter", None)
            if anchor_size_filter in ("small", "large"):
                anchor_areas = {int(idx): _anchor_area_at_chosen_anchor(trajectory_paths[idx])
                                    for idx in effective_train_indices}
                valid_areas = np.array([a for a in anchor_areas.values() if a is not None])
                if len(valid_areas) == 0:
                    raise ValueError("no valid anchor areas computed for the training pool")
                size_threshold = float(np.median(valid_areas))
                if anchor_size_filter == "small":
                    keep = [idx for idx, area in anchor_areas.items() if area is not None and area < size_threshold]
                else:
                    keep = [idx for idx, area in anchor_areas.items() if area is not None and area >= size_threshold]
                effective_train_indices = np.sort(np.array(keep, dtype=np.int64))
                print(f"  anchor_size_filter={anchor_size_filter}: "
                        f"split at {size_threshold:.0f} px² (1024-resize), "
                        f"kept {len(effective_train_indices)}/{len(anchor_areas)} train trajectories")
            elif anchor_size_filter not in (None, "both"):
                raise ValueError(f"anchor_size_filter must be 'small'/'large'/'both'/None, got {anchor_size_filter!r}")

            trainer = hydra.utils.instantiate(trainer_config.main_trainer)
            trainer.custom_train(x=trajectory_paths, y=trajectory_paths,
                                   train_indices=effective_train_indices, validation_indices=test_indices)
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

        tracker = hydra.utils.instantiate(tracker_config.tracker)
        output_path = output_directory / f"{tracker_name}.pkl"
        records = run_tracker(tracker, trajectory_paths, test_indices, detection_data, output_path)
        print(f"saved {len(records)} -> {output_path}")

        del tracker
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

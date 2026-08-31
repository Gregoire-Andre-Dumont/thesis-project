"""Print the occlusion rate (fraction of tracked post-warmup frames with occlusion >= 0.5) for the first N
trajectories in processing order. Annotations only -- no frame decode, no model."""

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pe_reid_longrange import test_trajectories
from create_anchor_dataset import anchor_trajectory_index


@hydra.main(config_path="../conf", config_name="robust_scoring", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    for i, (video, person, anchor_frame) in enumerate(test_trajectories(person_path, 20), 1):
        detection_data.load_frames = False
        detection_data.initialize_target(video, person)
        anchor_index = anchor_trajectory_index(detection_data, anchor_frame)
        if anchor_index is None:
            print(f"{i:2d}  {video} p{person}: no anchor")
            continue
        warmup = 1 if anchor_index >= 1 else 0
        window = detection_data.frame_indices[anchor_index - warmup:anchor_index - warmup + warmup + config.max_frames]
        detection_data.initialize_target(video, person, frame_indices=window)
        occ = detection_data.occlusions[warmup:]
        n = len(occ)
        n_occ = int((occ >= 0.5).sum())
        onset = int(np.argmax(occ > 0)) if float(occ.max()) > 0 else -1
        print(f"{i:2d}  {video} p{person:<4d} frames={n:3d}  occluded={n_occ:3d}  rate={n_occ / n:.2f}  first_occ@{onset}")


if __name__ == "__main__":
    run()

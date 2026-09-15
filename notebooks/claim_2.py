"""claim_2 (experiment): how robust is each arm to a CORRUPTED memory bank?

Every trajectory is rolled out once per arm per corruption probability. Corruption is an unconditional per-frame
write: on EVERY frame from the first occlusion onward -- occluded or visible -- the bank is pushed a clean,
well-formed mask of the nearest OTHER person with probability `p`, so the injected error is one of IDENTITY, not
mask quality. It does NOT depend on the arm's own commit gate, so all three arms face the same corruption events
and differ only in how they cope; each arm's own commits proceed unchanged alongside it.

`corruption_ps` includes 0.0, which IS the clean rollout -- no separate clean pass. All arms share one
image-embedding cache, the same per-frame distractor boxes, and a per-clip seed derived with crc32 (not `hash`,
which Python salts per interpreter run), so the corruption pattern is reproducible across launches.

`out_dir/results.pkl` holds one record per trajectory with per-frame box IoUs over the whole post-occlusion
span (NaN where there is no GT box) plus each rollout's per-frame memory-commit flags, so metrics that score
the occluded half on commit behaviour are available post-hoc alongside plain coverage. Coverage at any IoU
threshold and any slice is likewise a post-hoc computation.
"""
import sys
import pickle
import zlib
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claim_1 import (test_trajectories, load_window, first_occlusion_frame, visible_frames,
                     frame_record, commit_flags, load_results)
from src.utils.corruption import nearest_distractor_boxes
from src.offline_training.dataset_labels import load_clean_boxes_by_frame

SAM_CONFIG = "conf/trackers/baselines/sam_baseline.yaml"


@hydra.main(config_path="../conf", config_name="experiments/claim_2", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    memory_oracle = hydra.utils.instantiate(OmegaConf.load(config.oracle_config).tracker)
    mask_oracle = hydra.utils.instantiate(OmegaConf.load(config.mask_oracle_config).tracker)
    sam = hydra.utils.instantiate(OmegaConf.load(SAM_CONFIG).tracker)
    sam.label_mask_iou = False
    memory_oracle.iou_threshold = mask_oracle.iou_threshold = float(config.commit_threshold)
    arms = (("sam", sam), ("memory", memory_oracle), ("mask", mask_oracle))
    probabilities = [float(p) for p in config.corruption_ps]

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.pkl"
    processed, clips = load_results(results_path)

    for trajectory in test_trajectories(person_path, config.n_traj):
        video, person = trajectory[0], trajectory[1]
        if (video, person) in processed:
            continue
        processed.add((video, person))

        window = load_window(detection_data, trajectory[:3], config.max_frames)
        if window is None:
            continue
        warmup, _ = window
        occlusions = detection_data.occlusions[warmup:]
        boxes = detection_data.bboxes_norm[warmup:]
        first_occlusion = first_occlusion_frame(occlusions)
        if not visible_frames(occlusions, boxes, first_occlusion):
            continue

        # One distractor-box series per clip, shared by every arm so they face identical injected errors.
        clean_boxes = load_clean_boxes_by_frame(
            f"{config.detection_data.visible_directory}/{video}.json", int(person))
        corruption_boxes = nearest_distractor_boxes(detection_data, clean_boxes)
        if not any(box is not None for box in corruption_boxes):
            continue                                            # no distractor anywhere: nothing to corrupt

        cache = {}
        seed = zlib.crc32(f"{video}:{int(person)}".encode())     # stable across launches, unlike hash()
        for _, tracker in arms:
            tracker.frame_cache = cache
            tracker.corruption_boxes = corruption_boxes
            tracker.corruption_seed = seed

        # The scored span starts at the first occlusion; `occluded` and `has_box` line up index for index
        # with every arm's per-frame array, so a metric can score the occluded half on commit behaviour
        # rather than dropping it.
        span = slice(first_occlusion, len(occlusions))
        record = {
            "video": video, "person": person,
            "n_frames": int(len(occlusions)),
            "first_occlusion": int(first_occlusion),
            "occ_count": int((occlusions >= 0.5).sum()),
            "occlusions": np.asarray(occlusions, dtype=np.float32),
            "occluded": np.asarray(occlusions[span] >= 0.5, dtype=bool),
            "has_box": np.asarray(boxes[span][:, 2] > 0, dtype=bool),
            "corruptible": int(sum(box is not None for box in corruption_boxes)),
        }
        for name, tracker in arms:
            for probability in probabilities:
                tracker.corruption_p = probability
                predicted = tracker.predict_masks(detection_data).numpy()
                flags = commit_flags(tracker, predicted.shape[0])[warmup + first_occlusion:]
                record[(name, probability)] = frame_record(predicted[warmup:], occlusions, boxes, first_occlusion)
                record[(name, probability, "commit")] = flags
                record[(name, probability, "corrupted")] = len(getattr(tracker, "corrupted_frames", []))

        for _, tracker in arms:
            tracker.frame_cache = None
            tracker.corruption_boxes = None
            tracker.corruption_p = 0.0

        clips.append(record)
        pickle.dump({"processed": processed, "corruption_ps": probabilities,
                     "commit_threshold": float(config.commit_threshold), "clips": clips},
                    open(results_path, "wb"))
        low, high = probabilities[0], probabilities[-1]
        visible = record["has_box"] & ~record["occluded"]
        held = lambda ious: float((ious[visible] >= 0.5).mean()) if visible.any() else float("nan")
        print(f"{len(clips):3d}  occ={record['occ_count']:3d}  " +
              "  ".join(f"{name} {held(record[(name, low)]):.3f}->{held(record[(name, high)]):.3f}"
                        f" ({record[(name, high, 'corrupted')]:2d} bad)" for name, _ in arms), flush=True)


if __name__ == "__main__":
    run()

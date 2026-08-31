"""Post-occlusion tracking coverage -- over VISIBLE frames after the first occlusion, the fraction where the
predicted mask's BOX IoU with the GT box >= 0.5 (the tracker is on the target) -- for three rollouts:
  oracle clean    : true-IoU oracle, no corruption
  oracle corrupt  : true-IoU oracle + non-sticky distractor injection (corruption_p)
  sam_baseline    : the realistic SAM 2 baseline (commits on its own confidence), no corruption
Box IoU (mask bbox vs GT box) is model-independent, so all three trackers are graded against the same reference."""

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pe_reid_longrange import test_trajectories
from robust_scoring import load_window, build_corruption_boxes
from src.offline_training.dataset_labels import load_clean_boxes_by_frame
from src.utils.compute_iou import compute_iou


def coverage(predicted, occlusions, boxes, first_occlusion):
    """Fraction of visible post-first-occlusion frames where the predicted mask's box IoU with the GT box >= 0.5."""

    visible = [f for f in range(first_occlusion, len(occlusions)) if occlusions[f] < 0.5 and float(boxes[f][2]) > 0]
    if not visible:
        return np.nan
    ious = compute_iou(boxes[visible], predicted[visible])
    return float(np.mean(ious >= 0.5))


@hydra.main(config_path="../conf", config_name="robust_scoring", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    oracle = hydra.utils.instantiate(OmegaConf.load(config.oracle_config).tracker)
    sam = hydra.utils.instantiate(OmegaConf.load("conf/trackers/baselines/sam_baseline.yaml").tracker)
    sam.label_mask_iou = False                                 # box IoU is computed post-hoc; skip the pseudo-GT decode

    clean_cov, corrupt_cov, sam_cov = [], [], []
    for trajectory in test_trajectories(person_path, config.n_traj):
        window = load_window(detection_data, trajectory, config.max_frames)
        if window is None:
            continue
        warmup, _ = window
        video, person, _ = trajectory
        clean_boxes = load_clean_boxes_by_frame(Path(config.detection_data.visible_directory) / f"{video}.json", person)

        oracle.corruption_boxes = build_corruption_boxes(detection_data, clean_boxes)
        oracle.frame_cache = {}
        oracle.corruption_p = 0.0
        clean_pred = oracle.predict_masks(detection_data).numpy()[warmup:].copy()
        occlusions = detection_data.occlusions[warmup:]
        boxes = detection_data.bboxes_norm[warmup:]

        oracle.corruption_p = config.corruption_p
        corrupt_pred = oracle.predict_masks(detection_data).numpy()[warmup:].copy()
        oracle.frame_cache = None
        oracle.corruption_boxes = None

        sam_pred = sam.predict_masks(detection_data).numpy()[warmup:].copy()   # realistic baseline, its own gate, no corruption

        first_occlusion = int(np.argmax(occlusions > 0)) if float(occlusions.max()) > 0 else len(occlusions)
        c = coverage(clean_pred, occlusions, boxes, first_occlusion)
        o = coverage(corrupt_pred, occlusions, boxes, first_occlusion)
        s = coverage(sam_pred, occlusions, boxes, first_occlusion)
        if np.isnan(c):
            continue
        clean_cov.append(c); corrupt_cov.append(o); sam_cov.append(s)
        print(f"{len(clean_cov):3d}  clean={np.nanmean(clean_cov):.3f}  corrupt={np.nanmean(corrupt_cov):.3f}  sam_baseline={np.nanmean(sam_cov):.3f}", flush=True)

    print(f"\nPOST-OCCLUSION VISIBLE COVERAGE ({len(clean_cov)} traj):"
          f"  oracle-clean={np.nanmean(clean_cov):.3f}  oracle-corrupt={np.nanmean(corrupt_cov):.3f}  sam_baseline={np.nanmean(sam_cov):.3f}")


if __name__ == "__main__":
    run()

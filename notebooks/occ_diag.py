"""Per-trajectory oracle-clean vs sam_baseline coverage (and mean box IoU) over visible post-occlusion frames,
with the occlusion rate -- to see whether sam_baseline genuinely matches the oracle everywhere or only on easy clips."""

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pe_reid_longrange import test_trajectories
from robust_scoring import load_window
from src.utils.compute_iou import compute_iou


@hydra.main(config_path="../conf", config_name="robust_scoring", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    oracle = hydra.utils.instantiate(OmegaConf.load(config.oracle_config).tracker)
    sam = hydra.utils.instantiate(OmegaConf.load("conf/trackers/baselines/sam_baseline.yaml").tracker)
    sam.label_mask_iou = False

    for i, trajectory in enumerate(test_trajectories(person_path, 12), 1):
        window = load_window(detection_data, trajectory, config.max_frames)
        if window is None:
            continue
        warmup, _ = window
        oracle.frame_cache = {}
        oracle.corruption_p = 0.0
        clean = oracle.predict_masks(detection_data).numpy()[warmup:].copy()
        oracle.frame_cache = None
        sam_pred = sam.predict_masks(detection_data).numpy()[warmup:].copy()

        occ = detection_data.occlusions[warmup:]
        boxes = detection_data.bboxes_norm[warmup:]
        first_occ = int(np.argmax(occ > 0)) if float(occ.max()) > 0 else len(occ)
        visible = [f for f in range(first_occ, len(occ)) if occ[f] < 0.5 and float(boxes[f][2]) > 0]
        if not visible:
            continue
        iou_clean = compute_iou(boxes[visible], clean[visible])
        iou_sam = compute_iou(boxes[visible], sam_pred[visible])
        rate = float((occ >= 0.5).mean())
        print(f"{i:2d} occ={rate:.2f} vis={len(visible):3d} | oracle cov={np.mean(iou_clean >= 0.5):.3f} meanIoU={iou_clean.mean():.3f}"
              f" | sam cov={np.mean(iou_sam >= 0.5):.3f} meanIoU={iou_sam.mean():.3f}", flush=True)


if __name__ == "__main__":
    run()

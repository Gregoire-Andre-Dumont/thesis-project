"""coverage_sweep: post-occlusion coverage AUC (mean box IoU over visible post-first-occlusion frames) as a
function of the NUMBER OF OCCLUDED FRAMES (config.n_bins quantile bins), sweeping the oracle commit-gate
box-IoU threshold over config.thresholds for both:
  memory oracle : commit gated by true box IoU; mask picked by SAM 2's IoU token.
  mask oracle   : same gate; mask picked as the proposed mask with the best true box IoU vs GT.
sam_baseline is a single reference (its own self-confidence gate). One shared image-embedding cache per
trajectory lets all rollouts (1 sam + 2 * len(thresholds) oracle) reuse a single image encode per frame.
Writes <out_dir>/coverage_sweep_memory.png and coverage_sweep_mask.png."""

import sys
import pickle
from pathlib import Path

import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pe_reid_longrange import test_trajectories
from robust_scoring import load_window
from src.utils.compute_iou import compute_iou


def coverage(predicted, occlusions, boxes, first_occlusion):
    """Over visible post-first-occlusion frames, return (AUC, success@0.5, success@0.7):
      AUC         = mean box IoU (= threshold-free success-plot AUC),
      success@t   = fraction of frames with box IoU >= t.
    (nan, nan, nan) if there are no such frames."""

    visible = [f for f in range(first_occlusion, len(occlusions)) if occlusions[f] < 0.5 and float(boxes[f][2]) > 0]
    if not visible:
        return np.nan, np.nan, np.nan
    ious = compute_iou(boxes[visible], predicted[visible])
    return float(np.mean(ious)), float(np.mean(ious >= 0.5)), float(np.mean(ious >= 0.7))


@hydra.main(config_path="../conf", config_name="coverage_sweep", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)

    memory_oracle = hydra.utils.instantiate(OmegaConf.load(config.oracle_config).tracker)
    mask_oracle = hydra.utils.instantiate(OmegaConf.load(config.mask_oracle_config).tracker)

    for tracker in (memory_oracle, mask_oracle):
        tracker.corruption_boxes = None
        tracker.corruption_p = -1.0

    sam = hydra.utils.instantiate(OmegaConf.load("conf/trackers/baselines/sam_baseline.yaml").tracker)
    sam.label_mask_iou = False
    thresholds = list(config.thresholds)

    # Resume from a checkpoint on /workspace (survives pod restarts, unlike /tmp)
    out_dir = Path(config.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "coverage_sweep_ckpt.pkl"

    if checkpoint.exists():
        state = pickle.load(open(checkpoint, "rb"))
        processed, occ_counts = state["processed"], state["occ_counts"]
        sam_cov, sam_s05, sam_s07 = state["sam_cov"], state["sam_s05"], state["sam_s07"]
        memory_cov, memory_s05, memory_s07 = state["memory_cov"], state["memory_s05"], state["memory_s07"]
        mask_cov, mask_s05, mask_s07 = state["mask_cov"], state["mask_s05"], state["mask_s07"]
        print(f"resumed: {len(occ_counts)} scored, {len(processed)} trajectories already seen", flush=True)
    else:
        processed, occ_counts, sam_cov, sam_s05, sam_s07 = set(), [], [], [], []
        memory_cov = {t: [] for t in thresholds}; memory_s05 = {t: [] for t in thresholds}; memory_s07 = {t: [] for t in thresholds}
        mask_cov = {t: [] for t in thresholds}; mask_s05 = {t: [] for t in thresholds}; mask_s07 = {t: [] for t in thresholds}

    for trajectory in test_trajectories(person_path, config.n_traj):
        video, person, _ = trajectory
        if (video, person) in processed:
            continue
        window = load_window(detection_data, trajectory, config.max_frames)
        if window is None:
            processed.add((video, person))
            continue
        warmup, _ = window
        occlusions = detection_data.occlusions[warmup:]
        boxes = detection_data.bboxes_norm[warmup:]
        first_occlusion = int(np.argmax(occlusions > 0)) if float(occlusions.max()) > 0 else len(occlusions)
        if first_occlusion <= config.first_occ_min:                  # require a clean run-in
            processed.add((video, person))
            continue
        if not [f for f in range(first_occlusion, len(occlusions)) if occlusions[f] < 0.5 and float(boxes[f][2]) > 0]:
            processed.add((video, person))
            continue

        cache = {}                                                   # shared image encode for every rollout of this clip
        sam.frame_cache = cache
        s_auc, s_s05, s_s07 = coverage(sam.predict_masks(detection_data).numpy()[warmup:], occlusions, boxes, first_occlusion)
        memory_oracle.frame_cache = mask_oracle.frame_cache = cache
        row_m, row_m05, row_m07, row_k, row_k05, row_k07 = {}, {}, {}, {}, {}, {}
        for threshold in thresholds:
            memory_oracle.iou_threshold = threshold
            row_m[threshold], row_m05[threshold], row_m07[threshold] = coverage(memory_oracle.predict_masks(detection_data).numpy()[warmup:], occlusions, boxes, first_occlusion)
            mask_oracle.iou_threshold = threshold
            row_k[threshold], row_k05[threshold], row_k07[threshold] = coverage(mask_oracle.predict_masks(detection_data).numpy()[warmup:], occlusions, boxes, first_occlusion)
        sam.frame_cache = memory_oracle.frame_cache = mask_oracle.frame_cache = None

        occ_counts.append(int((occlusions >= 0.5).sum()))
        sam_cov.append(s_auc); sam_s05.append(s_s05); sam_s07.append(s_s07)
        for threshold in thresholds:
            memory_cov[threshold].append(row_m[threshold]); memory_s05[threshold].append(row_m05[threshold]); memory_s07[threshold].append(row_m07[threshold])
            mask_cov[threshold].append(row_k[threshold]); mask_s05[threshold].append(row_k05[threshold]); mask_s07[threshold].append(row_k07[threshold])
        processed.add((video, person))
        pickle.dump({"processed": processed, "occ_counts": occ_counts,
                     "sam_cov": sam_cov, "sam_s05": sam_s05, "sam_s07": sam_s07,
                     "memory_cov": memory_cov, "memory_s05": memory_s05, "memory_s07": memory_s07,
                     "mask_cov": mask_cov, "mask_s05": mask_s05, "mask_s07": mask_s07}, open(checkpoint, "wb"))
        best_memory = max(np.nanmean(memory_s05[t]) for t in thresholds)
        best_mask = max(np.nanmean(mask_s05[t]) for t in thresholds)
        print(f"{len(occ_counts):3d}  occ_frames={occ_counts[-1]:3d}  sam@0.5={s_s05:.3f}  "
              f"best_memory@0.5={best_memory:.3f}  best_mask@0.5={best_mask:.3f}", flush=True)

    occ_counts = np.array(occ_counts)
    edges = np.quantile(occ_counts, np.linspace(0, 1, config.n_bins + 1)); edges[-1] += 1e-6
    bin_of = np.clip(np.digitize(occ_counts, edges) - 1, 0, config.n_bins - 1)
    present = [b for b in range(config.n_bins) if (bin_of == b).any()]
    centers = [float(occ_counts[bin_of == b].mean()) for b in present]

    def binned(values):
        arr = np.array(values, dtype=float)
        return [float(np.nanmean(arr[bin_of == b])) for b in present]

    print(f"\nOCC-FRAME BIN EDGES: {[round(e) for e in edges]}   n={len(occ_counts)}   bins {[round(c) for c in centers]}")
    for metric, sam_m, mem_m, mask_m, ylabel in [
            ("s05", sam_s05, memory_s05, mask_s05, "post-occlusion coverage (success @ IoU >= 0.5)"),
            ("s07", sam_s07, memory_s07, mask_s07, "post-occlusion coverage (success @ IoU >= 0.7)"),
            ("auc", sam_cov, memory_cov, mask_cov, "post-occlusion coverage AUC (mean IoU)")]:
        for name, cov in [("memory", mem_m), ("mask", mask_m)]:
            plt.figure(figsize=(7.5, 5.5))
            print(f"{name} oracle [{metric}]:")
            for threshold in thresholds:
                y = binned(cov[threshold])
                plt.plot(centers, y, "o-", label=f"thr={threshold}")
                print(f"  thr={threshold}: " + "  ".join(f"{v:.3f}" for v in y))
            plt.plot(centers, binned(sam_m), "k--", marker="^", label="sam baseline")
            plt.xlabel("number of occluded frames"); plt.ylabel(ylabel)
            plt.title(f"{name} oracle ({metric}): coverage vs occlusion frames, threshold sweep (n={len(occ_counts)})")
            plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(Path(config.out_dir) / f"coverage_sweep_{name}_{metric}.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    run()

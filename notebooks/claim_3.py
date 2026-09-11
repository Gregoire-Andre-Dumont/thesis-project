"""claim_3 (experiment): can Perception-Encoder similarity replace SAM's own mask selection and memory control?

Two arms over the identical trajectories, anchor and frames:

  sam   -- the baseline: SAM 2's IoU token picks the mask, SAM's own confidence gates the commit.
  pe    -- PE picks the mask AND gates the commit, both from the same FG-BG margin against the
           anchor's foreground patches. No ground truth anywhere, so this is a deployable tracker
           and not an upper bound (contrast memory_oracle / mask_oracle in claim_1).

Both arms share one image-embedding cache per clip, so the SAM encoder runs once per frame no matter
how many arms or thresholds are rolled -- the comparison is paired frame for frame.

`out_dir/results.pkl` holds one record per trajectory with the raw per-frame box IoUs, so coverage at
any IoU threshold and any slice is a post-hoc computation. Checkpointed per clip (resumable).
"""
import sys
import pickle
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claim_1 import (test_trajectories, load_window, first_occlusion_frame, visible_frames,
                     frame_ious, coverage, load_results)
from src.offline_training.dataset_encoders import load_dataset_encoders, anchor_size_pixels
from src.utils.load_bboxes import load_bboxes
from src.utils.load_frame_ids import load_frame_ids

SAM_CONFIG = "conf/trackers/baselines/sam_baseline.yaml"


def amodal_anchor_floor(config, video, person, detection_data):
    """Pixel crop floor from the anchor's AMODAL box, so every crop is taken at the target's true
    scale even when the anchor is partly occluded. Mirrors robust_scoring.anchor_floor; falls back to
    the visible box when the amodal annotation is missing."""

    amodal_json = str(Path(config.detection_data.amodal_directory) / f"{video}.json")
    visible_json = str(Path(config.detection_data.visible_directory) / f"{video}.json")
    amodal_ids, _, _ = load_frame_ids(amodal_json, person)
    visible_ids, _, _ = load_frame_ids(visible_json, person)
    union = np.unique(np.concatenate([amodal_ids, visible_ids]))       # the axis load_bboxes returns on
    position = int(np.searchsorted(union, int(detection_data.frame_indices[0])))

    box = load_bboxes(amodal_json, visible_json, person, use_amodal=True)[position]
    if float(box[2]) <= 0:
        box = detection_data.bboxes_norm[0]
    return anchor_size_pixels(box, detection_data.frames[0].shape)


@hydra.main(config_path="../conf", config_name="experiments/claim_3", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    sam = hydra.utils.instantiate(OmegaConf.load(SAM_CONFIG).tracker)
    sam.label_mask_iou = False
    pe = hydra.utils.instantiate(OmegaConf.load(config.pe_config).tracker)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pe.encode = load_dataset_encoders([config.encoder], dev=device)[config.encoder]
    pe.pe_select = bool(config.pe_select)      # False isolates the gate: SAM still picks the mask
    pe.gate = str(config.gate)                 # "sam" = object score + predicted IoU; "pe" = FG similarity
    thresholds = [float(t) for t in config.gate_thresholds]

    def set_threshold(value):
        """The swept value feeds whichever gate is active."""
        if pe.gate == "sam":
            pe.iou_threshold = value
        else:
            pe.pe_threshold = value

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

        cache = {}                                        # SAM image embeddings, shared by every rollout
        sam.frame_cache = pe.frame_cache = cache
        pe.anchor_floor = amodal_anchor_floor(config, video, int(person), detection_data)

        def rollout(tracker):
            predicted = tracker.predict_masks(detection_data).numpy()[warmup:]
            return frame_ious(predicted, occlusions, boxes, first_occlusion)

        record = {
            "video": video, "person": person,
            "distractor_overlap": float(trajectory[3]),
            "n_frames": int(len(occlusions)),
            "first_occlusion": int(first_occlusion),
            "occ_count": int((occlusions >= 0.5).sum()),
            "occlusions": np.asarray(occlusions, dtype=np.float32),
        }
        record["sam"] = rollout(sam)
        for threshold in thresholds:
            set_threshold(threshold)
            record[("pe", threshold)] = rollout(pe)
            record[("pe", threshold, "committed")] = len(pe.committed_frames)
            record[("pe", threshold, "scores")] = pe.pe_scores.numpy().astype(np.float32)[warmup:]

        sam.frame_cache = pe.frame_cache = None

        clips.append(record)
        pickle.dump({"processed": processed, "pe_thresholds": thresholds,
                     "gate": pe.gate, "clips": clips}, open(results_path, "wb"))
        print(f"{len(clips):3d}  occ={record['occ_count']:3d}  sam {coverage(record['sam']):.3f}  " +
              "  ".join(f"pe@{t:g} {coverage(record[('pe', t)]):.3f}"
                        f" ({record[('pe', t, 'committed')]:3d} commits)" for t in thresholds), flush=True)


if __name__ == "__main__":
    run()

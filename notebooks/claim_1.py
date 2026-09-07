"""claim_1 (experiment): after an occlusion the memory oracle and mask oracle recover the target far better
than the sam baseline, and their advantage widens the longer the occlusion lasts.

For every trajectory we roll out three trackers over the same clip, sharing one image-embedding cache: sam, the
memory oracle, and the mask oracle -- each at a fixed commit-gate threshold (0.2). config.out_dir/results.pkl
holds one record PER TRAJECTORY -- its (video, person) id, occluded-frame count, the full occlusion series, and
the raw per-frame box IoUs (predicted mask vs GT box) on the visible post-first-occlusion frames for every arm.
Storing raw IoUs + per-clip metadata means coverage at any IoU threshold / AUC and occlusion/video/first-N/last-N
slices are all post-hoc in claim_1_visualize.ipynb. The run is checkpointed per clip (resumable).
"""
import sys
import pickle
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robust_scoring import load_window
from src.utils.compute_iou import compute_iou

SUCCESS_IOU = 0.5                                                   # a frame is covered when box IoU >= this
SAM_CONFIG = "conf/trackers/baselines/sam_baseline.yaml"


def test_trajectories(person_path, n_traj, random_seed=42):
    """Draw `n_traj` trajectories as a single seeded held-out test set, returned in shuffled order.
    Taking one random subset (never back-filling from the train split) and not grouping by video
    means no video is over-represented -- the sample is representative of the selection pool."""

    triples = [(v, int(p), int(a), float(o), float(r)) for v, p, a, o, r in zip(
        person_path.selected_video_names.tolist(),
        person_path.selected_person_ids.tolist(),
        person_path.selected_anchor_video_frames.tolist(),
        person_path.selected_anchor_overlaps.tolist(),
        person_path.selected_anchor_visible_ratios.tolist())]

    n_test = min(n_traj, len(triples))
    _, test_indices = train_test_split(np.arange(len(triples)), test_size=n_test, random_state=random_seed, shuffle=True)
    return [triples[i] for i in test_indices]


# ---------------------------------------------------------------------------------------
# coverage metric
# ---------------------------------------------------------------------------------------

def first_occlusion_frame(occlusions):
    """Index of the first occluded frame, or len(occlusions) if the target is never occluded."""
    return int(np.argmax(occlusions > 0)) if float(occlusions.max()) > 0 else len(occlusions)


def visible_frames(occlusions, boxes, start):
    """Frames at or after `start` where the target is visible"""
    return [f for f in range(start, len(occlusions)) if occlusions[f] < 0.5]

def frame_ious(predicted, occlusions, boxes, first_occlusion):
    """Box IoU (predicted mask vs GT box) on each visible post-first-occlusion frame; empty if none.
    We store the raw IoUs (not a thresholded fraction) so coverage at any IoU threshold -- @0.25, @0.5,
    @0.7 -- is a post-hoc computation over the same pkl."""
    frames = visible_frames(occlusions, boxes, first_occlusion)
    if not frames:
        return np.array([], dtype=np.float32)
    return compute_iou(boxes[frames], predicted[frames]).astype(np.float32)


def coverage(ious, threshold=SUCCESS_IOU):
    """Fraction of a clip's frame IoUs at or above `threshold`; NaN when the clip has no scorable frame."""
    return float(np.mean(np.asarray(ious) >= threshold)) if len(ious) else np.nan


# ---------------------------------------------------------------------------------------
# per-clip rollout loop (resumable)
# ---------------------------------------------------------------------------------------

def load_results(path):
    """Resume the per-trajectory records, or start empty when there is no pkl yet.
    Returns (processed set of every (video, person) seen, list of scored-clip records)."""
    if not path.exists():
        return set(), []
    state = pickle.load(open(path, "rb"))
    return set(state["processed"]), list(state["clips"])


@hydra.main(config_path="../conf", config_name="experiments/claim_1", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    memory_oracle = hydra.utils.instantiate(OmegaConf.load(config.oracle_config).tracker)
    mask_oracle = hydra.utils.instantiate(OmegaConf.load(config.mask_oracle_config).tracker)
    sam = hydra.utils.instantiate(OmegaConf.load(SAM_CONFIG).tracker)
    sam.label_mask_iou = False
    thresholds = [float(t) for t in config.thresholds]             # commit-gate sweep for the two oracles

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.pkl"
    processed, clips = load_results(results_path)

    def rollout_ious(tracker):
        predicted = tracker.predict_masks(detection_data).numpy()[warmup:]
        return frame_ious(predicted, occlusions, boxes, first_occlusion)

    def sweep_ious(tracker):
        """Roll the oracle out once per commit-gate threshold; the shared frame cache means the image
        encodings are computed once and reused across every threshold. Keyed by threshold."""
        out = {}
        for thr in thresholds:
            tracker.iou_threshold = thr
            out[thr] = rollout_ious(tracker)
        return out

    for trajectory in test_trajectories(person_path, config.n_traj):
        video, person, _, overlap, visible_ratio = trajectory
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

        cache = {}
        sam.frame_cache = memory_oracle.frame_cache = mask_oracle.frame_cache = cache

        record = {
            "video": video, "person": person,
            "distractor_overlap": float(overlap),
            "anchor_visible_ratio": float(visible_ratio),
            "n_frames": int(len(occlusions)),
            "first_occlusion": int(first_occlusion),
            "occ_count": int((occlusions >= 0.5).sum()),
            "occlusions": np.asarray(occlusions, dtype=np.float32),
            "sam": rollout_ious(sam),                              # sam's gate is its own confidence (not swept)
            "memory": sweep_ious(memory_oracle),                  # {threshold: per-frame IoUs}
            "mask": sweep_ious(mask_oracle),                      # {threshold: per-frame IoUs}
        }
        sam.frame_cache = memory_oracle.frame_cache = mask_oracle.frame_cache = None
        clips.append(record)
        t0 = thresholds[0]
        print(f"{len(clips):3d}  occ_frames={record['occ_count']:3d}  "
              f"cov@0.5 (thr={t0}) sam={coverage(record['sam']):.3f} "
              f"mem={coverage(record['memory'][t0]):.3f} "
              f"mask={coverage(record['mask'][t0]):.3f}", flush=True)

        pickle.dump({"processed": processed, "n_bins": int(config.n_bins),
                     "thresholds": thresholds, "clips": clips}, open(results_path, "wb"))



if __name__ == "__main__":
    run()

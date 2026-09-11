"""Connected-component oracles vs the normal setting, on the same clips.

For each trajectory both oracles are rolled out twice -- `use_components` off and on -- over one shared image
embedding cache, so the only difference is whether a proposal may be narrowed to a component subset:

  memory oracle : components change only what is COMMITTED (reported mask is SAM's unfiltered pick),
                  so its delta isolates the memory-poisoning effect.
  mask oracle   : components widen the SELECTION candidate set; the chosen subset is reported and committed.

Prints per-clip coverage and a pooled summary, plus how many frames were actually narrowed (if that count is
~0 the two settings are identical by construction and any delta is noise).
"""
import sys
import pickle
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robust_scoring import load_window
from claim_1 import test_trajectories, first_occlusion_frame, visible_frames, frame_ious, coverage

N_CLIPS = 60
THRESHOLD = 0.4                                                    # commit gate, held fixed for this comparison
SAM_CONFIG = "conf/trackers/baselines/sam_baseline.yaml"
RESULTS_PATH = "data/claim_1/component_compare.pkl"


@hydra.main(config_path="../conf", config_name="experiments/claim_1", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    memory_oracle = hydra.utils.instantiate(OmegaConf.load(config.oracle_config).tracker)
    mask_oracle = hydra.utils.instantiate(OmegaConf.load(config.mask_oracle_config).tracker)
    sam = hydra.utils.instantiate(OmegaConf.load(SAM_CONFIG).tracker)
    sam.label_mask_iou = False
    memory_oracle.iou_threshold = mask_oracle.iou_threshold = THRESHOLD

    # Checkpointed per clip: re-running with a larger N_CLIPS continues instead of recomputing.
    rows = pickle.load(open(RESULTS_PATH, "rb")) if Path(RESULTS_PATH).exists() else []
    processed = {(r[7], r[8]) for r in rows}
    print(f"resuming with {len(rows)} clips already done", flush=True)

    for trajectory in test_trajectories(person_path, config.n_traj):
        if len(rows) >= N_CLIPS:
            break
        if (trajectory[0], int(trajectory[1])) in processed:
            continue
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

        def roll(tracker, components):
            tracker.use_components = components
            predicted = tracker.predict_masks(detection_data).numpy()[warmup:]
            narrowed = len([f for f in tracker.filtered_frames if f >= warmup])
            return coverage(frame_ious(predicted, occlusions, boxes, first_occlusion)), narrowed

        sam_cov = coverage(frame_ious(sam.predict_masks(detection_data).numpy()[warmup:],
                                      occlusions, boxes, first_occlusion))
        mem_plain, _ = roll(memory_oracle, False)
        mem_comp, mem_narrowed = roll(memory_oracle, True)
        msk_plain, _ = roll(mask_oracle, False)
        msk_comp, msk_narrowed = roll(mask_oracle, True)
        sam.frame_cache = memory_oracle.frame_cache = mask_oracle.frame_cache = None

        rows.append((sam_cov, mem_plain, mem_comp, msk_plain, msk_comp, mem_narrowed, msk_narrowed,
                     trajectory[0], int(trajectory[1])))
        pickle.dump(rows, open(RESULTS_PATH, "wb"))
        print(f"{len(rows):3d} {trajectory[0]:>24} p{trajectory[1]:<4} sam={sam_cov:.3f} | "
              f"mem {mem_plain:.3f}->{mem_comp:.3f} ({mem_comp-mem_plain:+.3f}, {mem_narrowed:3d} narrowed) | "
              f"mask {msk_plain:.3f}->{msk_comp:.3f} ({msk_comp-msk_plain:+.3f}, {msk_narrowed:3d} narrowed)",
              flush=True)

    data = np.array([r[:5] for r in rows], dtype=float)
    narrowed = np.array([r[5:7] for r in rows], dtype=float)
    sam_c, mem_p, mem_c, msk_p, msk_c = data.mean(axis=0)
    print(f"\n=== pooled over n={len(rows)} clips (commit threshold {THRESHOLD}) ===")
    print(f"  sam                       {sam_c:.3f}")
    print(f"  memory  plain             {mem_p:.3f}  ({mem_p-sam_c:+.3f} vs sam)")
    print(f"  memory  components        {mem_c:.3f}  ({mem_c-sam_c:+.3f} vs sam)   [components {mem_c-mem_p:+.3f}]")
    print(f"  mask    plain             {msk_p:.3f}  ({msk_p-sam_c:+.3f} vs sam)")
    print(f"  mask    components        {msk_c:.3f}  ({msk_c-sam_c:+.3f} vs sam)   [components {msk_c-msk_p:+.3f}]")
    print(f"  frames narrowed / clip:   memory {narrowed[:, 0].mean():.1f}   mask {narrowed[:, 1].mean():.1f}")


if __name__ == "__main__":
    run()

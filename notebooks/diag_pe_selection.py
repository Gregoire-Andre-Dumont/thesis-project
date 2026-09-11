"""Diagnostic: WHY does the PE tracker underperform sam on large targets?

Hypothesis: FG-only chamfer is a MEAN over the candidate's foreground patches, so it is maximised by a
small, highly-typical mask -- a torso-only proposal whose every patch matches the anchor beats the
correct whole-person mask, which adds head/limb/boundary patches that match less well. If true, PE
should systematically pick SMALLER-area proposals than SAM's IoU token, and lose box IoU by
under-segmenting rather than by tracking the wrong person.

Per frame this logs, for all three proposals: PE foreground-similarity score, mask area, and box IoU vs
the GT box -- plus which proposal PE picked and which SAM's IoU token would have picked. Prints the
comparison split by anchor size. Read-only: it never commits anything, so it does not perturb tracking.
"""
import sys
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from collections import deque
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claim_1 import test_trajectories, load_window, first_occlusion_frame, visible_frames
from claim_3 import amodal_anchor_floor
from src.offline_training.dataset_encoders import load_dataset_encoders
from src.utils.compute_iou import compute_iou
from src.typing.person_path import _nearest_index

_cache = {}


def anchor_area(video, person_id, anchor):
    if video not in _cache:
        _cache[video] = json.load(open(f"data/person_path/visible/{video}.json"))
    visible = _cache[video]
    scale = 1024 / max(float(visible["metadata"]["resolution"]["width"]),
                       float(visible["metadata"]["resolution"]["height"]))
    entities = sorted([e for e in visible["entities"] if e["id"] == person_id],
                      key=lambda e: e["blob"]["frame_idx"])
    if not entities:
        return np.nan
    frames = np.array([e["blob"]["frame_idx"] for e in entities])
    wh = np.array([e["bb"][2:4] for e in entities], dtype=np.float64)
    return float(np.prod(wh[_nearest_index(frames, anchor)]) * scale ** 2)


@hydra.main(config_path="../conf", config_name="experiments/claim_3", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    pe = hydra.utils.instantiate(OmegaConf.load(config.pe_config).tracker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pe.encode = load_dataset_encoders([config.encoder], dev=device)[config.encoder]

    rows = []
    for trajectory in test_trajectories(person_path, config.n_traj):
        video, person, anchor = trajectory[0], int(trajectory[1]), int(trajectory[2])
        window = load_window(detection_data, trajectory[:3], config.max_frames)
        if window is None:
            continue
        warmup, _ = window
        occlusions, boxes = detection_data.occlusions[warmup:], detection_data.bboxes_norm[warmup:]
        first_occlusion = first_occlusion_frame(occlusions)
        if not visible_frames(occlusions, boxes, first_occlusion):
            continue

        pe.anchor_floor = amodal_anchor_floor(config, video, person, detection_data)
        pe.main_memory.reset_memory()
        init_mask = pe.main_memory.initialize_references(pe.model, detection_data, anchor_index=0)
        pe.seed_anchor(detection_data, init_mask)
        pe.reference_tokens = deque(maxlen=pe.reference_history)   # normally built in predict_masks

        for idx, frame in enumerate(detection_data.frames):
            if idx < first_occlusion or occlusions[idx] > 0.5 or float(boxes[idx][2]) <= 0:
                continue                                   # score only GT-verifiable post-occlusion frames
            mask_preds, iou_scores, pointers, object_score, lowres, _ = pe.model.propose_masks(
                main_memory=pe.main_memory, current_frame=frame)
            candidates = mask_preds[0, 1:].to(torch.float64).cpu().numpy()
            tokens, foreground = pe.encode_proposals(frame, candidates)
            scores = pe.proposal_scores(tokens, foreground)
            if np.all(np.isnan(scores)):
                continue

            binary = candidates > 0.0
            areas = binary.reshape(3, -1).sum(1).astype(float)
            ious = compute_iou(np.repeat(boxes[idx][None, :], 3, axis=0), binary)
            pe_pick = int(np.nanargmax(scores))
            sam_pick = int(torch.argmax(iou_scores[0, 1:]))
            rows.append((anchor_area(video, person, anchor), pe_pick, sam_pick,
                         areas[pe_pick], areas[sam_pick], ious[pe_pick], ious[sam_pick]))

            # Track on SAM's pick so the rollout stays on the baseline trajectory for every clip.
            chosen, pointer, encoding = pe.model.commit_candidate(
                mask_preds, 1 + sam_pick, pointers, object_score, lowres)
            pe.main_memory.update_memory(pointer, encoding)

        if len({r[0] for r in rows}) >= int(config.get("diag_clips", 8)):
            break

    data = np.array(rows, dtype=float)
    median_area = np.median(data[:, 0])
    for label, subset in (("SMALL", data[:, 0] <= median_area), ("LARGE", data[:, 0] > median_area)):
        d = data[subset]
        print(f"\n{label}  ({len(d)} frames, anchor area {'<=' if label == 'SMALL' else '>'} {median_area:.0f} px²)")
        print(f"  PE and SAM pick the same proposal: {(d[:, 1] == d[:, 2]).mean():.2f}")
        print(f"  mean mask area   PE {d[:, 3].mean():8.0f}   SAM {d[:, 4].mean():8.0f}"
              f"   (PE/SAM {d[:, 3].mean() / max(d[:, 4].mean(), 1):.2f})")
        print(f"  PE picks the SMALLER mask on {(d[:, 3] < d[:, 4]).mean():.2f} of frames")
        print(f"  mean box IoU     PE {d[:, 5].mean():.3f}   SAM {d[:, 6].mean():.3f}")
        disagree = d[d[:, 1] != d[:, 2]]
        if len(disagree):
            print(f"  on the {len(disagree)} DISAGREEMENT frames: PE IoU {disagree[:, 5].mean():.3f}"
                  f"   SAM IoU {disagree[:, 6].mean():.3f}"
                  f"   PE smaller on {(disagree[:, 3] < disagree[:, 4]).mean():.2f}")


if __name__ == "__main__":
    run()

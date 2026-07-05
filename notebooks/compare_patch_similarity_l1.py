"""compare_patch_similarity_l1: how different is the calibrator's INPUT (the cosine-similarity
maps) across backbone sizes?

Unlike raw memory tokens (unaligned 64-dim bases -> cosine ~0 across sizes), the anchor-fg
similarity map is a within-encoder cosine map in [-1, 1], so it lives in the SAME space for every
encoder and element-wise L1 between maps IS meaningful. The query crop uses a fixed box mask so
all sizes share one 32x32 grid -> maps are spatially aligned.

For N trajectories, per size, computes the (32x32) foreground-to-anchor-fg similarity map at a
mid-trajectory visible frame, then reports vs the large-encoder map:
  L1     = mean|map_size - map_large|          (0 = identical map; scale is cosine units)
  corr   = Pearson r of the two flattened maps (1 = identical structure)
  mean   = mean map value per size

Run: python notebooks/compare_patch_similarity_l1.py [N]
"""

import sys
import glob
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from src.typing.detection_data import DetectionData
from src.utils.load_bboxes import convert_bbox
from src.modules.samara_hiera_model import SamaraHieraModel

SIZES = ["tiny", "small", "base_plus", "large"]
REF = "large"
CROP_RESIZE, PAD_RATIO = 512, 0.25


def box_mask(bbox_norm, height, width):
    x, y, w, h = bbox_norm
    x0, y0 = int(round(x * width)), int(round(y * height))
    x1, y1 = int(round((x + w) * width)), int(round((y + h) * height))
    m = np.zeros((height, width), dtype=np.float64)
    m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1.0
    return m


def visible_frames(occ):
    return np.where(np.asarray(occ) < 0.5)[0]


def sim_map(model, anchor_rgb, a_bbox, a_amodal, query_rgb, q_bbox):
    """(side, side) best-cosine-sim map of every query patch to this encoder's own anchor fg.
    Anchor fg uses the encoder's SAM init mask (real pipeline); query crop uses a fixed box mask
    so the grid aligns across sizes."""
    model.set_anchor_amodal_from_normalized(a_amodal, anchor_rgb.shape[:2])
    init_encoded, _, _ = model.encode_image(cv2.cvtColor(anchor_rgb, cv2.COLOR_RGB2BGR))
    init_mask, _, _ = model.initialize_video_masking(init_encoded, convert_bbox(a_bbox))
    mask_np = (init_mask > 0.0).to(torch.float64).cpu().numpy().squeeze()
    a_crop, a_maskcrop = model.extract_crops(anchor_rgb[None], mask_np[None])
    fg, _ = model.extract_patch_tokens(a_crop, a_maskcrop)

    qmask = box_mask(q_bbox, *query_rgb.shape[:2])
    q_crop, q_maskcrop = model.extract_crops(query_rgb[None], qmask[None])
    tokens, _ = model.extract_raw_patch_tokens(q_crop, q_maskcrop)
    side = int(round(tokens.shape[1] ** 0.5))
    return model.compute_patch_similarities(fg, tokens).reshape(side, side)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    metas = [pickle.load(open(p, "rb"))
             for p in sorted(glob.glob("data/memory_oracle/memory/padding_0.25/*.pkl"))[:n]]

    detection_data = DetectionData(
        amodal_directory="data/person_path/amodal", visible_directory="data/person_path/visible",
        video_directory="data/person_path/videos", load_frames=True, num_threads=32, resize_resolution=1024)

    trajectories = []
    for m in metas:
        detection_data.initialize_target(m.video_name, m.person_id, frame_indices=m.frame_indices)
        vis = visible_frames(detection_data.occlusions)
        if len(vis) < 2:
            continue
        a = int(vis[0])
        later = vis[vis > a]
        if len(later) == 0:
            continue
        q = int(later[np.argmin(np.abs(later - 0.6 * len(detection_data.occlusions)))])
        trajectories.append(dict(a_frame=detection_data.frames[a].astype(np.uint8),
                                 a_bbox=detection_data.bboxes_norm[a].copy(),
                                 a_amodal=detection_data.amodal_norm[a].copy(),
                                 q_frame=detection_data.frames[q].astype(np.uint8),
                                 q_bbox=detection_data.bboxes_norm[q].copy()))
    print(f"{len(trajectories)} trajectories\n")

    maps = {}
    for size in SIZES:
        model = SamaraHieraModel(sam_model_path=f"tm/sam_{size}.pt", crop_resize=CROP_RESIZE, pad_ratio=PAD_RATIO)
        maps[size] = [sim_map(model, t["a_frame"], t["a_bbox"], t["a_amodal"], t["q_frame"], t["q_bbox"])
                      for t in trajectories]
        print(f"  extracted {size:10} mean map value={np.mean([m.mean() for m in maps[size]]):.3f}")
        del model
        torch.cuda.empty_cache()

    print(f"\nvs {REF}-encoder similarity map (per-trajectory, then averaged):")
    print(f"{'size':10} {'L1':>8} {'corr':>8} {'mean':>8}")
    print("-" * 38)
    for size in SIZES:
        l1, corr, mean = [], [], []
        for ms, mr in zip(maps[size], maps[REF]):
            l1.append(np.abs(ms - mr).mean())
            corr.append(np.corrcoef(ms.ravel(), mr.ravel())[0, 1])
            mean.append(ms.mean())
        print(f"{size:10} {np.mean(l1):8.3f} {np.mean(corr):8.3f} {np.mean(mean):8.3f}")

    print("\npairwise L1 (mean|map_row - map_col|):")
    print(" " * 11 + "".join(f"{s:>11}" for s in SIZES))
    for a in SIZES:
        row = "".join(f"{np.mean([np.abs(xa - xb).mean() for xa, xb in zip(maps[a], maps[b])]):11.3f}"
                      for b in SIZES)
        print(f"{a:10} " + row)

    # reference points: how big is L1 vs the map's own spread, and vs a shuffled-baseline
    ref_std = np.mean([m.std() for m in maps[REF]])
    print(f"\nfor scale: large map std = {ref_std:.3f}  (L1 << this => maps nearly identical)")


if __name__ == "__main__":
    main()

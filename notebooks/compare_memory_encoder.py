"""compare_memory_encoder: are the memory-encoder outputs ~the same across backbone sizes?

For N trajectories, at the anchor frame, run the FULL memory path (image encoder -> memory
encoder) for tiny/small/base_plus/large on the SAME crop + SAME box mask (so the only variable
is the encoder weights). Memory tokens are (1, 32*32, 64) for every size, so they're directly
comparable. Reports, vs the large-encoder reference:

  rel-L1  = mean|x_size - x_large| / mean|x_large|      (0 = identical)
  cos     = mean per-token cosine similarity            (1 = identical direction)
  |x|     = mean abs token value (raw magnitude per size)

Run: python notebooks/compare_memory_encoder.py [N]
"""

import sys
import glob
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.typing.detection_data import DetectionData
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


def memory_tokens(model, frame_rgb, bbox_norm, amodal_norm):
    """(H*W, 64) memory-encoder tokens for the anchor crop, with a fixed box mask so the crop
    window is identical across sizes (anchor_amodal + pad_ratio drive the window, not the size)."""
    model.set_anchor_amodal_from_normalized(amodal_norm, frame_rgb.shape[:2])
    mask = box_mask(bbox_norm, *frame_rgb.shape[:2])
    crop, crop_mask = model.extract_crops(frame_rgb[None], mask[None])
    tokens, _ = model.extract_memory_patch_tokens(crop, crop_mask)   # (1, H*W, 64)
    return tokens[0].float().cpu()


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
        if len(vis) == 0:
            continue
        a = int(vis[0])
        trajectories.append(dict(frame=detection_data.frames[a].astype(np.uint8),
                                 bbox=detection_data.bboxes_norm[a].copy(),
                                 amodal=detection_data.amodal_norm[a].copy()))
    print(f"{len(trajectories)} trajectories\n")

    # tokens[size] = list over trajectories of (H*W, 64) tensors
    tokens = {}
    for size in SIZES:
        model = SamaraHieraModel(sam_model_path=f"tm/sam_{size}.pt",
                                 crop_resize=CROP_RESIZE, pad_ratio=PAD_RATIO, token_source="memory")
        tokens[size] = [memory_tokens(model, t["frame"], t["bbox"], t["amodal"]) for t in trajectories]
        mag = np.mean([x.abs().mean().item() for x in tokens[size]])
        print(f"  extracted {size:10} mean|x|={mag:.4f}")
        del model
        torch.cuda.empty_cache()

    print(f"\nvs {REF}-encoder reference (per-trajectory, then averaged):")
    print(f"{'size':10} {'rel-L1':>8} {'cos':>8} {'|x|':>8}")
    print("-" * 38)
    for size in SIZES:
        rel_l1, cos, mag = [], [], []
        for xs, xr in zip(tokens[size], tokens[REF]):
            rel_l1.append((xs - xr).abs().mean().item() / (xr.abs().mean().item() + 1e-8))
            cos.append(F.cosine_similarity(xs, xr, dim=-1).mean().item())   # per-token, then mean
            mag.append(xs.abs().mean().item())
        print(f"{size:10} {np.mean(rel_l1):8.3f} {np.mean(cos):8.3f} {np.mean(mag):8.4f}")

    # full pairwise rel-L1 matrix
    print("\npairwise rel-L1 (row vs col):")
    print(" " * 11 + "".join(f"{s:>11}" for s in SIZES))
    for a in SIZES:
        row = []
        for b in SIZES:
            v = np.mean([(xa - xb).abs().mean().item() / (xb.abs().mean().item() + 1e-8)
                         for xa, xb in zip(tokens[a], tokens[b])])
            row.append(f"{v:11.3f}")
        print(f"{a:10} " + "".join(row))


if __name__ == "__main__":
    main()

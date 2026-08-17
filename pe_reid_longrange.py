"""Long-range re-identification benchmark for frozen image backbones.

Question: as time passes after an anchor frame, how well does each backbone still recognise the
target person against nearby distractors -- using only foreground patch similarity, no tracker?

Protocol (tracker-free, ground-truth boxes):
  * Query      -- the target at its anchor frame: SAM box-prompt -> mask -> crop -> foreground tokens.
  * Gallery    -- at each later visible frame: the target + its N nearest clean distractors, encoded
                  the same way.
  * Score      -- foreground chamfer similarity of each candidate's tokens to the query's.
  * Metric     -- ROC-AUC of separating the target (label 1) from the distractors (label 0) by that
                  score, computed over the whole dataset and per temporal bin.
  * Long range -- AUC is binned by dt = frames elapsed since the anchor.

The result plot is target-vs-distractor AUC (y) versus temporal bin (x), one line per backbone.

Backbones are all LARGE scale (~212-454M params) so capacity is comparable, and each emits a 32x32
token grid (PE-L at 448, Hiera-L stage-2 at 512, SAM3 at 448):
  pe_core, pe_spatial (PE-L ~310M), hiera_mae, hiera_sam (Hiera-L ~212M), pe_sam3 (SAM3 ~454M)

Usage (prepend PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on a small GPU):
  python pe_reid_longrange.py                 # full held-out test split
  python pe_reid_longrange.py +n_traj=60 +stride=2
"""
import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, NamedTuple

import cv2
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.offline_training.dataset_encoders import (
    crop_around_masks, anchor_size_pixels, _hiera_stage2, _norm, HALF)
from src.offline_training.dataset_labels import load_clean_boxes_by_frame
from src.utils.load_bboxes import convert_bbox, load_bboxes
from create_anchor_dataset import anchor_trajectory_index, slice_detection_data_for_tracker

logging.getLogger("timm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

SAM_BASELINE_CONFIG = "conf/trackers/baselines/sam_baseline.yaml"  # SAM 2 model used as the box->mask prompter
SAM3_VISION_CONFIG = "conf/sam3_vision_config.json"                # bundled so we avoid the gated facebook/sam3 repo
CROP_SIZE = 512                     # crop resolution fed to every backbone (-> ~32x32 tokens)
SAM3_INPUT = 448                    # SAM 3's ViT input (patch-14 aligned to a 32x32 grid)
N_DISTRACTORS = 2                   # gallery = target + this many nearest distractors
MAX_FRAMES = 250                    # cap each trajectory to this many frames from the anchor
MIN_BIN_SAMPLES = 20                # a temporal bin needs this many candidate scores to be plotted

DT_EDGES = np.arange(0, 301, 60)    # temporal-bin edges, in frames since the anchor (5 bins of 60)
SPATIAL_EDGES = np.round(np.arange(0, 0.51, 0.1), 2)   # spatial bins: |centroid_t - centroid_anchor|, normalized
PLOT_PATH = "data/pe_reid_longrange.png"
NPY_PATH = "data/pe_reid_longrange.npy"
SPATIAL_PLOT_PATH = "data/pe_reid_longrange_spatial.png"
SPATIAL_NPY_PATH = "data/pe_reid_longrange_spatial.npy"
COLORS = {"pe_core": "#cc4444", "pe_spatial": "#22aa77", "pe_sam3": "#3377cc",
          "hiera_mae": "#dd9933", "hiera_sam": "#9944bb"}

TokenFn = Callable[[np.ndarray], torch.Tensor]  # crops (n, H, W, 3) -> patch tokens (n, num_patches, dim)


class Trajectory(NamedTuple):
    video: str
    person: int
    anchor_frame: int


# --------------------------------------------------------------------------------------------------
# Trajectory selection
# --------------------------------------------------------------------------------------------------
def test_trajectories(person_path, test_size: float = 0.20, seed: int = 42) -> list[Trajectory]:
    """The held-out test split of the create_anchor selection, reproduced exactly from
    build_trajectory_split (sorted trajectory stems, sklearn split)."""
    triples = [Trajectory(v, int(p), int(a)) for v, p, a in zip(
        person_path.selected_video_names.tolist(),
        person_path.selected_person_ids.tolist(),
        person_path.selected_anchor_video_frames.tolist())]
    order = sorted(range(len(triples)), key=lambda i: f"{triples[i].video}_{triples[i].person}")
    _, test_idx = train_test_split(np.array(order), test_size=test_size, random_state=seed, shuffle=True)
    return [triples[i] for i in sorted(test_idx)]


# --------------------------------------------------------------------------------------------------
# Backbones -- each is a TokenFn
# --------------------------------------------------------------------------------------------------
def load_sam3_encoder() -> TokenFn | None:
    """SAM 3's frozen vision encoder as a TokenFn, or None if it can't be fetched.

    Built from the ungated facebook/sam3 config plus the ungated 1038lab/sam3 mirror weights via
    transformers' Sam3VisionModel, run at SAM3_INPUT (pos-embed / RoPE interpolated to that grid)."""
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import Sam3VisionModel, Sam3VisionConfig

        config = json.load(open(SAM3_VISION_CONFIG))
        model = Sam3VisionModel(Sam3VisionConfig(**{**config, "image_size": SAM3_INPUT}))
        weights = load_file(hf_hub_download("1038lab/sam3", "sam3.safetensors"))  # ungated mirror
        prefix = "detector_model.vision_encoder."
        model.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})
        model = model.eval().to(DEVICE).to(DTYPE)
    except Exception as error:
        print(f"pe_sam3 unavailable ({str(error)[:80]}) -- running without it")
        return None

    @torch.inference_mode()
    def tokens(crops: np.ndarray) -> torch.Tensor:
        output = model(_norm(crops, SAM3_INPUT, HALF, HALF, DEVICE, DTYPE))
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        return hidden.float()

    return tokens


def _load_hiera_mae_large() -> torch.nn.Module:
    """Hiera-L MAE backbone at 512 (features_only). Bicubic-resizes the pretrained 56x56 stage-0
    pos_embed to the 128x128 grid (512/4), mirroring dataset_encoders._load_hiera_mae for the base."""
    src = timm.create_model("hiera_large_224.mae", pretrained=True).state_dict()
    pe = src["pos_embed"]
    side, dim = int(pe.shape[1] ** 0.5), pe.shape[2]
    src["pos_embed"] = F.interpolate(pe.reshape(1, side, side, dim).permute(0, 3, 1, 2).float(),
                                     size=(128, 128), mode="bicubic", align_corners=False
                                     ).permute(0, 2, 3, 1).reshape(1, 128 * 128, dim)
    model = timm.create_model("hiera_large_224.mae", pretrained=False, img_size=512, features_only=True)
    model.load_state_dict({"model." + k: v for k, v in src.items()}, strict=False)
    return model.eval().to(DEVICE).to(DTYPE)


def load_backbones() -> dict[str, TokenFn]:
    """The five backbones keyed by name, all at LARGE scale (~212-454M params) so capacity is comparable.
    Every one emits a 32x32 token grid: PE-L (patch14) at 448, Hiera-L (stage-2) at 512, SAM3 at 448."""
    pe_core = timm.create_model("vit_pe_core_large_patch14_336.fb", pretrained=True,
                                num_classes=0, img_size=448).eval().to(DEVICE).to(DTYPE)
    pe_spatial = timm.create_model("vit_pe_spatial_large_patch14_448.fb", pretrained=True,
                                   num_classes=0).eval().to(DEVICE).to(DTYPE)
    hiera_sam = timm.create_model("sam2_hiera_large.fb_r1024_2pt1", pretrained=True,
                                  features_only=True).eval().to(DEVICE).to(DTYPE)
    hiera_mae = _load_hiera_mae_large()

    @torch.inference_mode()
    def pe_core_tokens(crops: np.ndarray) -> torch.Tensor:
        return pe_core.forward_features(_norm(crops, 448, HALF, HALF, DEVICE, DTYPE))[:, pe_core.num_prefix_tokens:].float()

    @torch.inference_mode()
    def pe_spatial_tokens(crops: np.ndarray) -> torch.Tensor:
        return pe_spatial.forward_features(_norm(crops, 448, HALF, HALF, DEVICE, DTYPE))[:, pe_spatial.num_prefix_tokens:].float()

    backbones = {
        "pe_core": pe_core_tokens,
        "pe_spatial": pe_spatial_tokens,
        "hiera_mae": lambda crops: _hiera_stage2(hiera_mae, crops, DEVICE, DTYPE),
        "hiera_sam": lambda crops: _hiera_stage2(hiera_sam, crops, DEVICE, DTYPE),
    }
    sam3 = load_sam3_encoder()
    if sam3 is not None:
        backbones["pe_sam3"] = sam3
    return backbones


# --------------------------------------------------------------------------------------------------
# Foreground extraction and similarity
# --------------------------------------------------------------------------------------------------
@torch.inference_mode()
def box_prompt_masks(sam, frame: np.ndarray, boxes: list) -> list[np.ndarray]:
    """One SAM mask LOGIT map per box, prompting SAM with each box after a single frame encode.
    Logits (not a binary mask) are returned so crop_around_masks thresholds after resizing, exactly
    as create_anchor_dataset does. bf16 is cast to float32 for numpy."""
    image = sam.image_encoder.prepare_image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 1024, True)
    features = sam.image_encoder(image)
    masks = []
    for box in boxes:
        mask, _, _ = sam.initialize_video_masking(features, convert_bbox(np.asarray(box, np.float32)))
        masks.append(mask.squeeze().float().cpu().numpy())
    return masks


def _grid_foreground(crop_mask: np.ndarray, grid_side: int) -> torch.Tensor:
    """Downsample a crop's binary mask to the token grid and flatten it to a per-patch foreground mask."""
    mask = torch.from_numpy(crop_mask[None, None].astype(np.float32))
    return (F.interpolate(mask, size=(grid_side, grid_side), mode="nearest") > 0.5).flatten()


def foreground_tokens(sam, backbones: dict[str, TokenFn], frame: np.ndarray, boxes: list, floor: int):
    """Foreground tokens for several entities in one frame.

    Each box becomes a SAM mask (single frame encode), is cropped around that mask, and is encoded by
    every backbone in one batched pass. `floor` is the crop-size floor in pixels -- always the ANCHOR
    box size (as in create_anchor_dataset), so every entity at every frame is cropped at the anchor's
    scale rather than zoomed to fill its own box. Returns a list aligned to `boxes`, each entry a dict
    {backbone: (num_foreground_patches, dim)} -- or None if any box yields an empty mask."""
    masks = box_prompt_masks(sam, frame, boxes)
    if any((mask > 0).sum() == 0 for mask in masks):   # empty mask (logits: foreground is logit > 0)
        return None

    crops, crop_masks = [], []
    for mask in masks:
        crop, crop_mask = crop_around_masks(frame[None], mask[None].astype(np.float32), CROP_SIZE, 0.25, floor)
        crops.append(crop[0])
        crop_masks.append(crop_mask[0])
    crops = np.stack(crops)

    entities = [{} for _ in boxes]
    for name, tokens in backbones.items():
        patches = tokens(crops)                         # (num_entities, num_patches, dim), one batched pass
        grid_side = round(patches.shape[1] ** 0.5)
        for entity, patch_tokens, crop_mask in zip(entities, patches, crop_masks):
            foreground = _grid_foreground(crop_mask, grid_side).to(patch_tokens.device)
            entity[name] = patch_tokens[foreground]
    return entities


def chamfer_similarity(query_fg: torch.Tensor, candidate_fg: torch.Tensor) -> float:
    """Symmetric chamfer between anchor (query) and candidate foreground tokens: the mean of
      * each candidate patch's best cosine to any anchor patch  (candidate -> anchor), and
      * each anchor patch's best cosine to any candidate patch  (anchor -> candidate).
    The anchor->candidate direction penalizes candidates that lack the anchor's distinctive parts,
    which the one-directional version misses."""
    if query_fg.shape[0] == 0 or candidate_fg.shape[0] == 0:
        return float("nan")
    query = F.normalize(query_fg, dim=-1)
    candidate = F.normalize(candidate_fg, dim=-1)
    sim = candidate @ query.T                          # (num_candidate_patches, num_anchor_patches)
    return float(0.5 * (sim.max(dim=1).values.mean() + sim.max(dim=0).values.mean()))


# --------------------------------------------------------------------------------------------------
# Per-trajectory evaluation
# --------------------------------------------------------------------------------------------------
def _box_center(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def nearest_distractors(target_box, clean_boxes: list, k: int = N_DISTRACTORS) -> list:
    """The k clean-person boxes whose centre is closest to the target box centre."""
    cx, cy = _box_center(target_box)
    return sorted(clean_boxes, key=lambda b: (_box_center(b)[0] - cx) ** 2 + (_box_center(b)[1] - cy) ** 2)[:k]


def evaluate_trajectory(sam, backbones, detection_data, trajectory: Trajectory, visible_dir, amodal_dir, stride):
    """Yield (backbone, dt, label, score) for every candidate at every visible frame of one trajectory.
    label is 1 for the target (candidates[0]) and 0 for a distractor; score is the chamfer similarity
    to the anchor query."""
    detection_data.initialize_target(trajectory.video, trajectory.person)
    anchor_index = anchor_trajectory_index(detection_data, trajectory.anchor_frame)
    if anchor_index is None:
        return

    # Amodal box of the anchor (full extent, incl. occluded parts) -> crop-size floor, as in the old
    # pipeline. load_bboxes is aligned with the full frame union, i.e. the same axis as anchor_index.
    amodal_boxes = load_bboxes(str(Path(amodal_dir) / f"{trajectory.video}.json"),
                               str(Path(visible_dir) / f"{trajectory.video}.json"),
                               trajectory.person, use_amodal=True)
    anchor_amodal = amodal_boxes[anchor_index]

    warmup = slice_detection_data_for_tracker(detection_data, anchor_index)
    kept = slice(warmup, warmup + MAX_FRAMES)  # anchor is now index 0; cap the trajectory length
    frames = detection_data.frames[kept]
    occlusions = detection_data.occlusions[kept]
    boxes = detection_data.bboxes_norm[kept]
    frame_indices = detection_data.frame_indices[kept]
    if len(frames) < 2 or float(boxes[0][2]) <= 0:
        return

    anchor_box = anchor_amodal if float(anchor_amodal[2]) > 0 else boxes[0]
    floor = anchor_size_pixels(anchor_box, frames[0].shape)   # anchor amodal box size -> crop scale for the trajectory
    anchor = foreground_tokens(sam, backbones, frames[0], [boxes[0]], floor)
    if anchor is None:
        return
    query = anchor[0]
    anchor_center = _box_center(boxes[0])                  # for the spatial-distance axis
    clean_boxes = load_clean_boxes_by_frame(Path(visible_dir) / f"{trajectory.video}.json", trajectory.person)

    for t in range(1, len(frames), stride):
        if occlusions[t] > 0.5 or float(boxes[t][2]) <= 0:
            continue
        distractors = nearest_distractors(boxes[t], clean_boxes.get(int(frame_indices[t]), []))
        if not distractors:
            continue
        candidates = foreground_tokens(sam, backbones, frames[t], [boxes[t], *distractors], floor)  # index 0 = target
        if candidates is None:
            continue

        dt = int(frame_indices[t]) - int(frame_indices[0])
        cx, cy = _box_center(boxes[t])                     # how far the target has moved from the anchor
        dist = float(((cx - anchor_center[0]) ** 2 + (cy - anchor_center[1]) ** 2) ** 0.5)
        for name in backbones:
            for i, candidate in enumerate(candidates):
                score = chamfer_similarity(query[name], candidate[name])
                if not np.isnan(score):
                    yield name, dt, dist, int(i == 0), score   # label 1 = target, 0 = distractor


# --------------------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------------------
def dataset_auc(samples: list) -> float:
    """Overall target-vs-distractor AUC across a backbone's samples (NaN if only one class).
    Sample = (dt, spatial_dist, label, score)."""
    label = np.array([s[2] for s in samples])
    score = np.array([s[3] for s in samples])
    return roc_auc_score(label, score) if 0 < label.sum() < len(label) else float("nan")


def _binned_auc(samples: list, edges: np.ndarray, key: int) -> list[float]:
    """Target-vs-distractor AUC per bin along axis `key` of the sample tuple
    (key=0 -> frames since anchor, key=1 -> spatial distance). NaN for under-filled bins."""
    axis = np.array([s[key] for s in samples])
    curve = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = [s for s, a in zip(samples, axis) if lo <= a < hi]
        curve.append(dataset_auc(in_bin) if len(in_bin) >= MIN_BIN_SAMPLES else np.nan)
    return curve


def plot_auc(results: dict[str, list], edges: np.ndarray, key: int, xlabel: str,
             path: str, npy_path: str, n_traj: int):
    """Target-vs-distractor AUC versus a binning axis, one line per backbone. Saves the figure + curves."""
    centers = (edges[:-1] + edges[1:]) / 2
    curves = {name: _binned_auc(samples, edges, key) for name, samples in results.items()}

    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    for name, curve in curves.items():
        axis.plot(centers, curve, marker="o", markersize=5, color=COLORS.get(name), label=name)
    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (0.50)")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(f"target-vs-distractor AUC (vs {N_DISTRACTORS} distractors)")
    axis.set_title(f"Long-range re-ID via foreground similarity  |  {n_traj} trajectories\n"
                   f"all large-scale (~212-454M params)", fontsize=10)
    axis.set_ylim(0.45, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    np.save(npy_path, {"edges": edges, "curves": curves, "n_traj": n_traj}, allow_pickle=True)


# --------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------
@hydra.main(config_path="conf", config_name="create_anchor_dataset", version_base=None)
def main(config: DictConfig):
    stride = int(config.get("stride", 2))
    visible_dir = config.detection_data.visible_directory
    amodal_dir = config.detection_data.amodal_directory

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    sam = hydra.utils.instantiate(OmegaConf.load(SAM_BASELINE_CONFIG).tracker).model

    backbones = load_backbones()
    print(f"backbones: {list(backbones)}")

    trajectories = test_trajectories(person_path)
    results = defaultdict(list)

    for completed, trajectory in enumerate(tqdm(trajectories, desc="re-id"), start=1):
        for name, dt, dist, label, score in evaluate_trajectory(sam, backbones, detection_data, trajectory, visible_dir, amodal_dir, stride):
            results[name].append((dt, dist, label, score))
        plot_auc(results, DT_EDGES, 0, "frames since anchor", PLOT_PATH, NPY_PATH, completed)
        plot_auc(results, SPATIAL_EDGES, 1, "spatial distance from anchor (normalized)",
                 SPATIAL_PLOT_PATH, SPATIAL_NPY_PATH, completed)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("\noverall target-vs-distractor AUC (whole dataset):")
    for name, samples in results.items():
        print(f"  {name:11s} {dataset_auc(samples):.3f}   (n={len(samples)})")



if __name__ == "__main__":
    main()

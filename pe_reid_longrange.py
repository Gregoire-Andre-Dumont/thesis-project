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

Backbones share a common ~512px input (a 32x32 token grid) so none runs off its native resolution:
  hiera_mae, hiera_sam (SAM 2), pe_core, pe_spatial   -- base tier, as used by the dataset
  pe_sam3                                              -- SAM 3's vision encoder (only exists at ~L)

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
    load_dataset_encoders, crop_around_masks, anchor_size_pixels, _norm, HALF)
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
PLOT_PATH = "data/pe_reid_longrange.png"
NPY_PATH = "data/pe_reid_longrange.npy"
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
    return [triples[i] for i in test_idx]


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


def load_backbones() -> dict[str, TokenFn]:
    """The backbones keyed by name. pe_spatial / hiera_* reuse the dataset encoders (32x32 @ 512),
    pe_core is added here, and pe_sam3 joins when its weights are reachable."""
    encoders = load_dataset_encoders(["perception", "hiera_sam", "hiera_mae"], DEVICE, DTYPE)  # perception == pe_spatial
    pe_core = timm.create_model("vit_pe_core_base_patch16_224.fb", pretrained=True,
                                num_classes=0, img_size=CROP_SIZE).eval().to(DEVICE).to(DTYPE)

    @torch.inference_mode()
    def pe_core_tokens(crops: np.ndarray) -> torch.Tensor:
        features = pe_core.forward_features(_norm(crops, CROP_SIZE, HALF, HALF, DEVICE, DTYPE))
        return features[:, pe_core.num_prefix_tokens:].float()

    backbones = {
        "pe_core": pe_core_tokens,
        "pe_spatial": encoders["perception"],
        "hiera_mae": encoders["hiera_mae"],
        "hiera_sam": encoders["hiera_sam"],
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
    """One binary mask per box, prompting SAM with each box after a single frame encode."""
    image = sam.image_encoder.prepare_image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 1024, True)
    features = sam.image_encoder(image)
    masks = []
    for box in boxes:
        mask, _, _ = sam.initialize_video_masking(features, convert_bbox(np.asarray(box, np.float32)))
        masks.append((mask.squeeze() > 0.0).cpu().numpy())
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
    if any(mask.sum() == 0 for mask in masks):
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
    """Mean best-cosine of each candidate foreground patch to any query foreground patch."""
    if query_fg.shape[0] == 0 or candidate_fg.shape[0] == 0:
        return float("nan")
    query = F.normalize(query_fg, dim=-1)
    candidate = F.normalize(candidate_fg, dim=-1)
    return float((candidate @ query.T).max(dim=1).values.mean())


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
        for name in backbones:
            for i, candidate in enumerate(candidates):
                score = chamfer_similarity(query[name], candidate[name])
                if not np.isnan(score):
                    yield name, dt, int(i == 0), score      # label 1 = target, 0 = distractor


# --------------------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------------------
def dataset_auc(samples: list[tuple[int, int, float]]) -> float:
    """Overall target-vs-distractor AUC across a backbone's samples (NaN if only one class)."""
    label = np.array([s[1] for s in samples])
    score = np.array([s[2] for s in samples])
    return roc_auc_score(label, score) if 0 < label.sum() < len(label) else float("nan")


def _binned_auc(samples: list[tuple[int, int, float]]) -> list[float]:
    """Target-vs-distractor AUC per temporal bin (NaN where a bin has too few samples or one class)."""
    dt = np.array([s[0] for s in samples])
    curve = []
    for lo, hi in zip(DT_EDGES[:-1], DT_EDGES[1:]):
        in_bin = [s for s, d in zip(samples, dt) if lo <= d < hi]
        curve.append(dataset_auc(in_bin) if len(in_bin) >= MIN_BIN_SAMPLES else np.nan)
    return curve


def plot_auc_vs_time(results: dict[str, list], n_traj: int):
    """Target-vs-distractor AUC versus temporal bin, one line per backbone. Saves the figure and curves."""
    bin_centers = (DT_EDGES[:-1] + DT_EDGES[1:]) / 2
    curves = {name: _binned_auc(samples) for name, samples in results.items()}

    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    for name, curve in curves.items():
        axis.plot(bin_centers, curve, marker="o", markersize=5, color=COLORS.get(name), label=name)
    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (0.50)")
    axis.set_xlabel("frames since anchor")
    axis.set_ylabel(f"target-vs-distractor AUC (vs {N_DISTRACTORS} distractors)")
    axis.set_title(f"Long-range re-ID via foreground similarity  |  {n_traj} trajectories\n"
                   f"pe_sam3 is ~L, the others are base tier", fontsize=10)
    axis.set_ylim(0.45, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=120)
    plt.close(figure)
    np.save(NPY_PATH, {"edges": DT_EDGES, "curves": curves, "n_traj": n_traj}, allow_pickle=True)


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
        for name, dt, label, score in evaluate_trajectory(sam, backbones, detection_data, trajectory, visible_dir, amodal_dir, stride):
            results[name].append((dt, label, score))
        plot_auc_vs_time(results, completed)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()



if __name__ == "__main__":
    main()

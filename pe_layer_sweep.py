"""Layer-wise re-ID sweep for the PE variants: pe_core vs pe_spatial vs pe_sam3.

Same tracker-free setup as pe_reid_longrange (SAM box-prompt -> mask -> crop -> foreground tokens), but
instead of the final layer we extract patch tokens at every 4th transformer block and score foreground
chamfer at each. Only candidates whose distance from the anchor exceeds MIN_DIST px (@1024) are used --
this is the long-range regime. The plot: x = layer index, y = target-vs-distractor AUC, one line per model.

Hypothesis: SAM 3's detector fine-tuning erases the re-ID signal from its trunk, so pe_sam3's per-layer
AUC stays low / degrades, while pe_spatial (and pe_core) keep a strong signal in their mid/late layers.
"""
import json
import logging
import os
import warnings
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from pe_reid_longrange import (test_trajectories, box_prompt_masks, chamfer_scores, distance, to_pixel,
                               clean_distractors, DEVICE, DTYPE)
from create_anchor_dataset import anchor_trajectory_index, slice_detection_data_for_tracker
from src.offline_training.dataset_encoders import crop_around_masks, anchor_size_pixels, _norm, HALF
from src.offline_training.dataset_labels import load_clean_boxes_by_frame
from src.utils.load_bboxes import load_bboxes

logging.getLogger("timm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HYDRA_FULL_ERROR"] = "1"

MIN_DIST = 400.0                                  # only score candidates this far (px @1024) from the anchor
LAYER_STEP = 4                                    # sample every this many transformer blocks
COLORS = {"pe_core": "#cc4444", "pe_spatial": "#22aa77", "pe_sam3": "#3377cc"}
PLOT_PATH, NPY_PATH = "data/pe_layer_sweep.png", "data/pe_layer_sweep.npy"


def sampled_layers(depth):
    """Block indices sampled every LAYER_STEP, always including the last block."""
    return sorted(set(list(range(0, depth, LAYER_STEP)) + [depth - 1]))


def load_pe_timm(name, size):
    """A timm PE ViT as a per-layer encoder: crops -> {layer: (n, num_patches, dim)} patch tokens."""
    model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(DEVICE).to(DTYPE)
    layers = sampled_layers(len(model.blocks))

    @torch.inference_mode()
    def encode(crops):
        x = _norm(crops, size, HALF, HALF, DEVICE, DTYPE)
        feats = model.forward_intermediates(x, indices=layers, return_prefix_tokens=False,
                                            norm=False, output_fmt="NLC", intermediates_only=True)
        return {layer: f.float() for layer, f in zip(layers, feats)}
    return encode


def load_pe_sam3(sam3_config, sam3_input):
    """SAM 3's vision encoder as a per-layer encoder via output_hidden_states."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import Sam3VisionModel, Sam3VisionConfig

    config = json.load(open(sam3_config))
    model = Sam3VisionModel(Sam3VisionConfig(**{**config, "image_size": sam3_input}))
    weights = load_file(hf_hub_download("1038lab/sam3", "sam3.safetensors"))
    prefix = "detector_model.vision_encoder."
    model.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})
    model = model.eval().to(DEVICE).to(DTYPE)

    @torch.inference_mode()
    def encode(crops):
        x = _norm(crops, sam3_input, HALF, HALF, DEVICE, DTYPE)
        hidden = model(x, output_hidden_states=True).hidden_states       # tuple, len = depth + 1
        out = {}                                                          # hidden[0] = embedding, hidden[b+1] = after block b
        for layer in sampled_layers(len(hidden) - 1):
            h = hidden[layer + 1]                                        # "after block `layer`", matching timm's indexing
            if h.dim() == 4:                                             # (n, C, H, W) -> (n, HW, C)
                h = h.flatten(2).transpose(1, 2)
            patches = h.shape[1]
            grid = round(patches ** 0.5)
            if grid * grid != patches:                                  # drop leading prefix tokens (e.g. CLS)
                h = h[:, patches - grid * grid:]
            out[layer] = h.float()
        return out
    return encode


@torch.inference_mode()
def layer_foreground(sam, encoders, frame, boxes, floor, crop_size):
    """Per box, per model, per layer: foreground tokens. Returns {model: {layer: [fg per box]}}, or None
    if any box's mask is empty. Grid is read per layer, so pe_core's 24x24 and the others' 32x32 both work."""
    box_masks = box_prompt_masks(sam, frame, boxes)
    if any((mask > 0).sum() == 0 for mask in box_masks):
        return None

    tiled = np.repeat(frame[None], len(box_masks), axis=0)
    crops, crop_masks = crop_around_masks(tiled, np.stack(box_masks).astype(np.float32), crop_size, 0.2, floor)
    crop_masks_t = torch.from_numpy(crop_masks).unsqueeze(1)

    result = {}
    for model_name, encode in encoders.items():
        layers = encode(crops)
        per_layer = {}
        for layer, tokens in layers.items():
            grid = round(tokens.shape[1] ** 0.5)
            fg = (F.interpolate(crop_masks_t.to(tokens.device).float(), size=(grid, grid),
                                mode="nearest") > 0.5).flatten(1)
            per_layer[layer] = [tokens[i][fg[i]].clone() for i in range(len(crops))]
        result[model_name] = per_layer
        del layers
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return result


def evaluate_trajectory(sam, encoders, detection_data, trajectory, config, rng, crop_size):
    """Yield (model, layer, label, score) for every far candidate (distance > MIN_DIST) per frame, where
    score is the bidirectional foreground chamfer of that candidate's layer tokens to the anchor's."""
    video, person, anchor_frame = trajectory
    detection_data.initialize_target(video, person)
    anchor_index = anchor_trajectory_index(detection_data, anchor_frame)
    if anchor_index is None:
        return

    visible_json = Path(config.detection_data.visible_directory) / f"{video}.json"
    amodal_json = Path(config.detection_data.amodal_directory) / f"{video}.json"
    anchor_amodal = load_bboxes(str(amodal_json), str(visible_json), person, use_amodal=True)[anchor_index]

    warmup = slice_detection_data_for_tracker(detection_data, anchor_index)
    window = slice(warmup, warmup + config.max_frames)
    frames = detection_data.frames[window]
    boxes = detection_data.bboxes_norm[window]
    occlusions = detection_data.occlusions[window]
    frame_indices = detection_data.frame_indices[window]
    if len(frames) < 2 or float(boxes[0][2]) <= 0:
        return

    anchor_box = anchor_amodal if float(anchor_amodal[2]) > 0 else boxes[0]
    floor = anchor_size_pixels(anchor_box, frames[0].shape)
    anchor = layer_foreground(sam, encoders, frames[0], [boxes[0]], floor, crop_size)   # {model: {layer: [fg]}}
    if anchor is None:
        return

    height, width = frames[0].shape[:2]
    px_scale = 1024.0 / max(width, height)
    anchor_center = to_pixel(boxes[0], width, height, px_scale)
    min_norm_area = config.person_path.min_visible_area / (width * height * px_scale ** 2)
    clean_boxes = load_clean_boxes_by_frame(visible_json, person)

    for t in range(1, len(frames), config.stride):
        distractors = clean_distractors(clean_boxes.get(int(frame_indices[t]), []), config.n_distractors, min_norm_area, rng)
        if occlusions[t] > 0.5 or float(boxes[t][2]) <= 0 or not distractors:
            continue

        candidate_boxes = [boxes[t]] + distractors                       # index 0 = target
        labels = [1] + [0] * len(distractors)
        dists = [distance(to_pixel(box, width, height, px_scale), anchor_center) for box in candidate_boxes]
        far = [(box, lab) for box, lab, d in zip(candidate_boxes, labels, dists) if d > MIN_DIST]
        if not far:
            continue

        candidates = layer_foreground(sam, encoders, frames[t], [box for box, _ in far], floor, crop_size)
        if candidates is None:
            continue

        for model_name in encoders:
            for layer, anchor_fg_list in anchor[model_name].items():
                anchor_fg = anchor_fg_list[0]
                for i, (_, label) in enumerate(far):
                    score = chamfer_scores(anchor_fg, candidates[model_name][layer][i])[1]   # bidirectional
                    if not np.isnan(score):
                        yield model_name, layer, label, score


def layer_auc(pairs):
    """AUC over (label, score) pairs; NaN if one class."""
    labels = np.array([p[0] for p in pairs])
    scores = np.array([p[1] for p in pairs])
    return roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else np.nan


def plot_sweep(samples, n_traj):
    """AUC vs layer index, one line per model. samples: {(model, layer): [(label, score), ...]}."""
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    curves = {}
    for model_name in sorted({m for m, _ in samples}):
        layers = sorted(layer for m, layer in samples if m == model_name)
        aucs = [layer_auc(samples[(model_name, layer)]) for layer in layers]
        curves[model_name] = (layers, aucs)
        axis.plot(layers, aucs, marker="o", markersize=5, color=COLORS.get(model_name), label=model_name)

    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (0.50)")
    axis.set_xlabel("transformer layer index (sampled every 4 blocks)")
    axis.set_ylabel(f"target-vs-distractor AUC  (candidates > {int(MIN_DIST)} px from anchor)")
    axis.set_title(f"PE layer-wise long-range re-ID  |  {n_traj} trajectories", fontsize=10)
    axis.set_ylim(0.45, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=120)
    plt.close(figure)
    np.save(NPY_PATH, {"curves": curves, "n_traj": n_traj}, allow_pickle=True)


@hydra.main(config_path="conf", config_name="pe_reid", version_base=None)
def run(config: DictConfig):
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    sam = hydra.utils.instantiate(config.tracker.tracker).model
    encoders = {
        "pe_core":    load_pe_timm("vit_pe_core_large_patch14_336.fb", 336),
        "pe_spatial": load_pe_timm("vit_pe_spatial_large_patch14_448.fb", 448),
        "pe_sam3":    load_pe_sam3(config.sam3_vision_config, config.sam3_input),
    }
    print(f"encoders: {list(encoders)}")

    trajectories = test_trajectories(person_path, config.n_traj)
    rng = np.random.RandomState(0)
    samples = defaultdict(list)
    for completed, trajectory in enumerate(tqdm(trajectories, desc="layer-sweep"), start=1):
        for model_name, layer, label, score in evaluate_trajectory(sam, encoders, detection_data, trajectory, config, rng, config.crop_size):
            samples[(model_name, layer)].append((label, score))
        if completed % 5 == 0 and samples:
            plot_sweep(samples, completed)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    plot_sweep(samples, len(trajectories))
    print("\nfinal per-layer AUC (candidates > {:.0f} px):".format(MIN_DIST))
    for model_name in sorted({m for m, _ in samples}):
        layers = sorted(layer for m, layer in samples if m == model_name)
        print(f"  {model_name:11s} " + "  ".join(f"L{layer}={layer_auc(samples[(model_name, layer)]):.3f}" for layer in layers))


if __name__ == "__main__":
    run()

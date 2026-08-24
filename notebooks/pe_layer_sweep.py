"""Layer-wise re-ID sweep for the detector-fine-tuning pairs: clip vs owlvit vs pe_spatial vs pe_sam3.

Same tracker-free setup as pe_reid_longrange (SAM box-prompt -> mask -> crop -> foreground tokens), but
instead of the final layer we read patch tokens every LAYER_STEP transformer blocks (the last block always
included) and score foreground chamfer at each. Only candidates farther than MIN_DISTANCE px (@1024) from the
anchor are used -- the long-range regime. Two figures (uni- and bidirectional chamfer) plot, per model,
target-vs-distractor AUC against the layer index.

Hypothesis: detector fine-tuning erodes re-ID through the trunk, so the fine-tuned encoders (pe_sam3 from
pe_spatial, owlvit from clip) trail their base encoders' per-layer AUC, especially in the late blocks.
"""

import json
import logging
import os
import pickle
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root on path for `src` and create_anchor_dataset

from pe_reid_longrange import (test_trajectories, box_prompt_masks, chamfer_scores, distance, to_pixel,
                               clean_distractors, DEVICE, DTYPE)
from create_anchor_dataset import anchor_trajectory_index, slice_detection_data_for_tracker
from src.offline_training.dataset_encoders import crop_around_masks, anchor_size_pixels, _norm, HALF
from src.offline_training.dataset_labels import load_clean_boxes_by_frame
from src.utils.load_bboxes import load_bboxes


logging.getLogger("timm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HYDRA_FULL_ERROR"] = "1"

MIN_DISTANCE = 400.0                              # only score candidates this far (px @1024) from the anchor
LAYER_STEP = 2                                    # sample every this many transformer blocks (last block always included)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)   # CLIP / OWL-ViT normalization
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
COLORS = {"pe_spatial": "#22aa77", "pe_sam3": "#3377cc", "clip": "#cc4444", "owlvit": "#ff7f0e"}
DIRECTIONS = {"unidirectional": 1, "bidirectional": 2}          # column of a (label, uni, bi) sample per chamfer direction
OUTPUT_DIR = "data/pe_layer_sweep"
CHECKPOINT_PATH = f"{OUTPUT_DIR}/checkpoint.pkl"


def sampled_layers(depth):
    """Block indices sampled every LAYER_STEP, always including the last block."""

    return sorted(set(range(0, depth, LAYER_STEP)) | {depth - 1})


def layers_from_hidden_states(hidden_states):
    """Per-layer patch tokens from a transformers hidden_states tuple: sample every LAYER_STEP block (indexed
    'after block b', like timm), flatten SAM 3's channels-last maps, and drop any leading CLS / prefix token."""

    per_layer = {}
    for layer in sampled_layers(len(hidden_states) - 1):        # hidden_states[0] is the pre-block embedding
        tokens = hidden_states[layer + 1]
        if tokens.dim() == 4:                                   # SAM 3 emits (n, H, W, C) channels-last
            tokens = tokens.flatten(1, 2)                       # -> (n, H*W, C), matching last_hidden_state
        patch_count = tokens.shape[1]
        grid = round(patch_count ** 0.5)
        if grid * grid != patch_count:                         # a prefix token remains -> drop it
            tokens = tokens[:, patch_count - grid * grid:]
        per_layer[layer] = tokens.float()
    return per_layer


def load_pe_timm(model_name, input_size):
    """A timm PE ViT as a per-layer encoder: crops -> {layer: (n, num_patches, dim)} patch tokens."""

    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(DEVICE).to(DTYPE)
    layers = sampled_layers(len(model.blocks))

    @torch.inference_mode()
    def encode(crops):
        normalized = _norm(crops, input_size, HALF, HALF, DEVICE, DTYPE)
        intermediates = model.forward_intermediates(normalized, indices=layers, return_prefix_tokens=False,
                                                    norm=False, output_fmt="NLC", intermediates_only=True)
        return {layer: tokens.float() for layer, tokens in zip(layers, intermediates)}
    return encode


def load_pe_sam3(sam3_config, input_size):
    """SAM 3's vision encoder as a per-layer encoder via output_hidden_states."""

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import Sam3VisionModel, Sam3VisionConfig

    config = json.load(open(sam3_config))
    model = Sam3VisionModel(Sam3VisionConfig(**{**config, "image_size": input_size}))
    weights = load_file(hf_hub_download("1038lab/sam3", "sam3.safetensors"))
    prefix = "detector_model.vision_encoder."
    model.load_state_dict({key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)})
    model = model.eval().to(DEVICE).to(DTYPE)

    @torch.inference_mode()
    def encode(crops):
        normalized = _norm(crops, input_size, HALF, HALF, DEVICE, DTYPE)
        return layers_from_hidden_states(model(normalized, output_hidden_states=True).hidden_states)
    return encode


def load_hf_vision(model_class, model_name, input_size, mean, std):
    """A transformers CLIP-style vision model (CLIP / OWL-ViT) as a per-layer encoder; drops the CLS token.
    interpolate_pos_encoding lets it run at `input_size` with its position embeddings scaled to that grid."""

    model = model_class.from_pretrained(model_name).eval().to(DEVICE).to(DTYPE)

    @torch.inference_mode()
    def encode(crops):
        normalized = _norm(crops, input_size, mean, std, DEVICE, DTYPE)
        return layers_from_hidden_states(model(normalized, interpolate_pos_encoding=True, output_hidden_states=True).hidden_states)
    return encode


def load_encoders(config):
    """The four ViT encoders keyed by name, each a per-layer token function at its own native resolution."""

    from transformers import CLIPVisionModel, OwlViTVisionModel
    return {
        "pe_spatial": load_pe_timm("vit_pe_spatial_large_patch14_448.fb", 448),
        "pe_sam3": load_pe_sam3(config.sam3_vision_config, config.sam3_input),
        "clip": load_hf_vision(CLIPVisionModel, "openai/clip-vit-large-patch14", config.clip_input, CLIP_MEAN, CLIP_STD),
        "owlvit": load_hf_vision(OwlViTVisionModel, "google/owlvit-large-patch14", config.owlvit_input, CLIP_MEAN, CLIP_STD),
    }


@torch.inference_mode()
def layer_foreground(sam, encoders, frame, boxes, floor, crop_size):
    """Per box, per encoder, per layer: the foreground tokens the box's mask covers, as
    {encoder: {layer: [fg per box]}}. A box whose mask is empty yields empty token tensors (scored as
    NaN and dropped downstream). The grid is read per layer, so each encoder's own grid works."""

    box_masks = box_prompt_masks(sam, frame, boxes)
    tiled_frame = np.repeat(frame[None], len(box_masks), axis=0)
    crops, crop_masks = crop_around_masks(tiled_frame, np.stack(box_masks).astype(np.float32), crop_size, 0.2, floor)
    crop_masks_tensor = torch.from_numpy(crop_masks).unsqueeze(1)

    foreground_by_encoder = {}
    for encoder_name, encode in encoders.items():
        layers = encode(crops)
        foreground_by_layer = {}
        for layer, tokens in layers.items():
            grid_size = round(tokens.shape[1] ** 0.5)
            resized_masks = F.interpolate(crop_masks_tensor.to(tokens.device).float(), size=(grid_size, grid_size), mode="nearest")
            foreground = (resized_masks > 0.5).flatten(1)

            tokens_per_box = []
            for box in range(len(crops)):
                tokens_per_box.append(tokens[box][foreground[box]].clone())
            foreground_by_layer[layer] = tokens_per_box
        foreground_by_encoder[encoder_name] = foreground_by_layer
        del layers
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return foreground_by_encoder


def evaluate_trajectory(sam, encoders, detection_data, trajectory, config, rng):
    """Yield (encoder, layer, label, uni, bi) for every far candidate (distance > MIN_DISTANCE) per frame;
    uni/bi are the uni- and bidirectional foreground chamfer of that candidate's layer tokens to the anchor's."""

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
    anchor = layer_foreground(sam, encoders, frames[0], [boxes[0]], floor, config.crop_size)

    height, width = frames[0].shape[:2]
    pixel_scale = 1024.0 / max(width, height)
    anchor_center = to_pixel(boxes[0], width, height, pixel_scale)
    min_norm_area = config.person_path.min_visible_area / (width * height * pixel_scale ** 2)
    clean_boxes = load_clean_boxes_by_frame(visible_json, person)

    for frame in range(1, len(frames), config.stride):
        distractors = clean_distractors(clean_boxes.get(int(frame_indices[frame]), []), config.n_distractors, min_norm_area, rng)
        if occlusions[frame] > 0.5 or float(boxes[frame][2]) <= 0 or not distractors:
            continue

        candidate_boxes = [boxes[frame]] + distractors                                # index 0 = target
        candidate_labels = [1] + [0] * len(distractors)

        far_boxes = []
        far_labels = []
        for box, label in zip(candidate_boxes, candidate_labels):
            candidate_center = to_pixel(box, width, height, pixel_scale)
            if distance(candidate_center, anchor_center) > MIN_DISTANCE:
                far_boxes.append(box)
                far_labels.append(label)
        if not far_boxes:
            continue

        candidate_foreground = layer_foreground(sam, encoders, frames[frame], far_boxes, floor, config.crop_size)

        for encoder_name in encoders:
            for layer, anchor_foreground in anchor[encoder_name].items():
                for candidate_index, label in enumerate(far_labels):
                    uni, bi = chamfer_scores(anchor_foreground[0], candidate_foreground[encoder_name][layer][candidate_index])
                    yield encoder_name, layer, label, uni, bi


def layer_auc(samples, direction_column):
    """Target-vs-distractor AUC over (label, uni, bi) samples using one chamfer direction; NaN if degenerate."""

    valid = [sample for sample in samples if not np.isnan(sample[direction_column])]
    if not valid:
        return np.nan
    labels = np.array([sample[0] for sample in valid])
    scores = np.array([sample[direction_column] for sample in valid])
    return roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else np.nan


def recorded_encoders(samples):
    """The encoder names present in the collected samples, sorted."""

    return sorted({encoder for encoder, _layer in samples})


def recorded_layers(samples, encoder_name):
    """The layer indices recorded for one encoder, sorted."""

    layers = set()
    for encoder, layer in samples:
        if encoder == encoder_name:
            layers.add(layer)
    return sorted(layers)


def plot_sweep(samples, direction, path, n_traj):
    """Plot per-encoder AUC against layer index for one chamfer direction, and return the drawn curves."""

    direction_column = DIRECTIONS[direction]
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    curves = {}
    for encoder_name in recorded_encoders(samples):
        layers = recorded_layers(samples, encoder_name)
        aucs = [layer_auc(samples[(encoder_name, layer)], direction_column) for layer in layers]
        curves[encoder_name] = (layers, aucs)
        axis.plot(layers, aucs, marker="o", markersize=5, color=COLORS.get(encoder_name), label=encoder_name)

    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (0.50)")
    axis.set_xlabel(f"transformer layer index (sampled every {LAYER_STEP} blocks)")
    axis.set_ylabel(f"target-vs-distractor AUC  (candidates > {int(MIN_DISTANCE)} px from anchor)")
    axis.set_title(f"PE layer-wise long-range re-ID ({direction} chamfer)  |  {n_traj} trajectories", fontsize=10)
    axis.set_ylim(0.45, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower left", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return curves


def load_checkpoint(path):
    """Load (samples, done) from the checkpoint, or empties when there is none. samples maps
    (encoder, layer) -> [(label, uni, bi), ...]; done is the set of finished trajectory keys."""

    if not Path(path).exists():
        return defaultdict(list), set()
    state = pickle.loads(Path(path).read_bytes())
    return defaultdict(list, state["samples"]), set(state["done"])


def save_checkpoint(path, samples, done):
    """Atomically write the accumulated per-(encoder, layer) samples and finished trajectory keys."""

    tmp = Path(path).with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps({"samples": dict(samples), "done": list(done)}))
    tmp.replace(path)


@hydra.main(config_path="../conf", config_name="pe_reid", version_base=None)
def run(config: DictConfig):
    """Sweep per-layer foreground chamfer over the test trajectories and draw the uni/bi AUC-vs-layer figures."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    sam = hydra.utils.instantiate(config.tracker.tracker).model
    encoders = load_encoders(config)
    print(f"encoders: {list(encoders)}")

    trajectories = test_trajectories(person_path, config.n_traj)
    rng = np.random.RandomState(0)
    samples, done = load_checkpoint(CHECKPOINT_PATH)         
    print(f"sweeping {len(trajectories)} trajectories; resuming with {len(done)} done")

    for completed, trajectory in enumerate(tqdm(trajectories, desc="layer-sweep"), start=1):
        key = f"{trajectory[0]}_{trajectory[1]}_{trajectory[2]}"
        if key in done:
            continue
        for encoder_name, layer, label, uni, bi in evaluate_trajectory(sam, encoders, detection_data, trajectory, config, rng):
            samples[(encoder_name, layer)].append((label, uni, bi))
        done.add(key)
        if completed % config.checkpoint_every == 0:
            save_checkpoint(CHECKPOINT_PATH, samples, done)
            for direction in DIRECTIONS:
                plot_sweep(samples, direction, f"{OUTPUT_DIR}/{direction}.png", len(done))
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    save_checkpoint(CHECKPOINT_PATH, samples, done)
    curves = {direction: plot_sweep(samples, direction, f"{OUTPUT_DIR}/{direction}.png", len(done)) for direction in DIRECTIONS}
    np.save(f"{OUTPUT_DIR}/curves.npy", {"curves": curves, "n_traj": len(done)}, allow_pickle=True)

    print(f"\nfinal per-layer AUC (candidates > {int(MIN_DISTANCE)} px)  [uni | bi]:")
    for encoder_name in recorded_encoders(samples):
        cells = []
        for layer in recorded_layers(samples, encoder_name):
            uni = layer_auc(samples[(encoder_name, layer)], 1)
            bi = layer_auc(samples[(encoder_name, layer)], 2)
            cells.append(f"L{layer}={uni:.3f}|{bi:.3f}")
        print(f"  {encoder_name:11s} {'  '.join(cells)}")


if __name__ == "__main__":
    run()

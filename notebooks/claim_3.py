import os
import json
import logging
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

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

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.offline_training.dataset_encoders import (
    crop_around_masks,
    anchor_size_pixels,
    load_dataset_encoders,
    _normalise_crops,
    _patch_masks,
    HALF)
from src.offline_training.dataset_labels import load_clean_boxes_by_frame, _box_centre as box_center
from src.utils.load_bboxes import convert_bbox, load_bboxes
from create_anchor_dataset import anchor_trajectory_index

logging.getLogger("timm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HYDRA_FULL_ERROR"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def slice_detection_data_for_tracker(detection_data, anchor_index):
    """Trim the leading frames so the anchor lands where the tracker starts reading.
    Returns how many warmup frames were dropped, either zero or one."""

    warmup_count = 1 if anchor_index >= 1 else 0
    start_index = anchor_index - warmup_count
    detection_data.frames = detection_data.frames[start_index:]
    detection_data.bboxes_norm = detection_data.bboxes_norm[start_index:]
    detection_data.occlusions = detection_data.occlusions[start_index:]
    detection_data.frame_indices = detection_data.frame_indices[start_index:]
    return warmup_count


def test_trajectories(person_path, n_traj, test_size=0.20, random_seed=42):
    """List the selected trajectories, held-out test split first, extended with the rest up to n_traj.
    The split is by trajectory and seeded so it stays the same across runs."""

    triples = [(vector, int(p), int(first_encoder)) for vector, p, first_encoder in zip(person_path.selected_video_names.tolist(), person_path.selected_person_ids.tolist(), person_path.selected_anchor_video_frames.tolist())]

    encoder_order = sorted(range(len(triples)), key=lambda layer_index: f"{triples[layer_index][0]}_{triples[layer_index][1]}")
    train_indices, test_indices = train_test_split(np.array(encoder_order), test_size=test_size, random_state=random_seed, shuffle=True)
    ordered = [triples[layer_index] for layer_index in sorted(test_indices)] + [triples[layer_index] for layer_index in sorted(train_indices)]
    return ordered[:n_traj]


SAM3_REVISION = "f055b060a4"


def load_sam3_encoder(sam3_config, sam3_input):
    """Load SAM 3's frozen vision encoder as a token function.
    Built from the bundled config plus the ungated 1038lab/sam3 mirror weights."""

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import Sam3VisionModel, Sam3VisionConfig

    config = json.load(open(sam3_config))
    encoder_model = Sam3VisionModel(Sam3VisionConfig(**{**config, "image_size": sam3_input}))
    weights = load_file(hf_hub_download("1038lab/sam3", "sam3.safetensors", revision=SAM3_REVISION))
    weight_prefix = "detector_model.vision_encoder."
    encoder_model.load_state_dict(
        {k[len(weight_prefix) :]: vector for k, vector in weights.items() if k.startswith(weight_prefix)}
    )
    encoder_model = encoder_model.eval().to(DEVICE).to(DTYPE)

    @torch.inference_mode()
    def encoded_tokens(anchor_crops):
        output = encoder_model(_normalise_crops(anchor_crops, sam3_input, HALF, HALF, DEVICE, DTYPE))
        return (output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]).float()

    return encoded_tokens


def load_hf_vision(model_cls, encoder_name, size, mean, std):
    """A transformers CLIP-style vision model (CLIP / OWL-ViT) as a final-layer token function; drops the CLS
    token. interpolate_pos_encoding lets it run at `size` (native by default) with its position embeddings scaled."""

    encoder_model = model_cls.from_pretrained(encoder_name).eval().to(DEVICE).to(DTYPE)

    @torch.inference_mode()
    def encoded_tokens(anchor_crops):
        hidden_states = encoder_model(_normalise_crops(anchor_crops, size, mean, std, DEVICE, DTYPE), interpolate_pos_encoding=True).last_hidden_state
        patches = hidden_states.shape[1]
        grid = round(patches**0.5)
        if grid * grid != patches:
            hidden_states = hidden_states[:, patches - grid * grid :]
        return hidden_states.float()

    return encoded_tokens


def load_backbones(config):
    """Load the six LARGE backbones keyed by name; each emits its own token grid (grid-adaptive downstream). Two
    base->fine-tuned pairs are covered: pe_spatial->pe_sam3 and clip->owlvit (detector fine-tuning), plus the"""

    from transformers import CLIPVisionModel, OwlViTVisionModel

    shared = load_dataset_encoders(["perception", "hiera_sam", "hiera_mae"], DEVICE, DTYPE)
    return {
        "pe_spatial": shared["perception"],
        "pe_sam3": load_sam3_encoder(config.sam3_vision_config, config.sam3_input),
        "hiera_mae": shared["hiera_mae"],
        "hiera_sam": shared["hiera_sam"],
        "clip": load_hf_vision(
            CLIPVisionModel, "openai/clip-vit-large-patch14", config.clip_input, CLIP_MEAN, CLIP_STD
        ),
        "owlvit": load_hf_vision(
            OwlViTVisionModel, "google/owlvit-large-patch14", config.owlvit_input, CLIP_MEAN, CLIP_STD
        )}


@torch.inference_mode()
def box_prompt_masks(sam_encoder, anchor_frame, anchor_boxes):
    """Return one SAM mask logit map per box from a single frame encode.
    Logits (not binary masks) are returned so crop_around_masks can threshold after resizing."""

    image = sam_encoder.image_encoder.prepare_image(cv2.cvtColor(anchor_frame, cv2.COLOR_RGB2BGR), 1024, True)
    features = sam_encoder.image_encoder(image)
    prompts = [convert_bbox(np.asarray(anchor_box, np.float32)) for anchor_box in anchor_boxes]
    masks = [sam_encoder.initialize_video_masking(features, prompt)[0] for prompt in prompts]
    return [anchor_mask.squeeze().float().cpu().numpy() for anchor_mask in masks]


@torch.inference_mode()
def entity_tokens(sam_encoder, backbones, anchor_frame, anchor_boxes, area_floor, crop_size):
    """Return each box's foreground tokens for every backbone, cropped around its own mask but floored at the
    anchor scale. Aligned to boxes ({backbone: fg_tokens}); an empty mask yields empty tokens (scored NaN)."""

    box_masks = box_prompt_masks(sam_encoder, anchor_frame, anchor_boxes)

    frame_per_box = np.repeat(anchor_frame[None], len(box_masks), axis=0)
    box_crops, box_crop_masks = crop_around_masks(frame_per_box, np.stack(box_masks).astype(np.float32), crop_size, 0.2, area_floor)

    crop_masks = torch.from_numpy(box_crop_masks).unsqueeze(1)
    tokens_by_box = [{} for _ in anchor_boxes]
    for backbone_name, encode in backbones.items():
        patch_tokens = encode(box_crops)
        grid_size = round(patch_tokens.shape[1] ** 0.5)
        resized_masks = F.interpolate(crop_masks.to(patch_tokens.device).float(), size=(grid_size, grid_size), mode="nearest")
        foreground_masks = (resized_masks > 0.5).flatten(1)

        for box_tokens, box_patch_tokens, anchor_mask in zip(tokens_by_box, patch_tokens, foreground_masks):
            box_tokens[backbone_name] = box_patch_tokens[anchor_mask].clone()
        del patch_tokens
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return tokens_by_box


@torch.inference_mode()
def chamfer_scores(reference_fg, encoded_tokens):
    """Return (unidirectional, bidirectional) chamfer between the anchor foreground and the candidate tokens. uni =
    mean best cosine of each candidate token to any anchor patch; bi = average of that with the reverse"""

    if reference_fg.shape[0] == 0 or encoded_tokens.shape[0] == 0:
        return float("nan"), float("nan")
    reference = F.normalize(reference_fg, dim=-1)
    encoded_tokens = F.normalize(encoded_tokens, dim=-1)
    similarity = encoded_tokens @ reference.T
    tokens_to_anchor = similarity.max(dim=1).values.mean()
    anchor_to_tokens = similarity.max(dim=0).values.mean()
    return float(tokens_to_anchor), float(0.5 * (tokens_to_anchor + anchor_to_tokens))


def distance(first_encoder, b):
    """Euclidean distance between two points."""

    return ((first_encoder[0] - b[0]) ** 2 + (first_encoder[1] - b[1]) ** 2) ** 0.5


def to_pixel(anchor_box, frame_width, frame_height, scale):
    """Box centre in pixels at the common working resolution -- un-normalized, and isotropic in x and y
    (both scaled the same), unlike the per-axis normalized coordinates."""

    cx, cy = box_center(anchor_box)
    return cx * frame_width * scale, cy * frame_height * scale


def clean_distractors(clean_boxes, n_distractors, random_generator):
    """Up to n_distractors clean people, chosen at random rather than by proximity so they are not biased toward
    the target's local background."""

    if len(clean_boxes) <= n_distractors:
        return list(clean_boxes)
    return [
        clean_boxes[layer_index]
        for layer_index in random_generator.choice(len(clean_boxes), size=n_distractors, replace=False)
    ]


def evaluate_trajectory(sam_encoder, backbones, detection_data, trajectory, config, random_generator):
    """Yield (backbone, dists, bi_scores, uni_scores) per visible frame; index 0 is the target, rest distractors.
    bi = bidirectional chamfer of candidate FG vs anchor FG; uni = the one-directional (candidate->anchor) chamfer."""

    video, person, anchor_frame = trajectory
    detection_data.initialize_target(video, person)
    anchor_index = anchor_trajectory_index(detection_data, anchor_frame)
    if anchor_index is None:
        return

    visible_json = Path(config.detection_data.visible_directory) / f"{video }.json"
    amodal_json = Path(config.detection_data.amodal_directory) / f"{video }.json"
    anchor_amodal = load_bboxes(str(amodal_json), str(visible_json), person, use_amodal=True)[anchor_index]

    warmup = slice_detection_data_for_tracker(detection_data, anchor_index)
    window = slice(warmup, warmup + config.max_frames)
    frames = detection_data.frames[window]
    anchor_boxes = detection_data.bboxes_norm[window]
    occlusions = detection_data.occlusions[window]
    frame_indices = detection_data.frame_indices[window]
    if len(frames) < 2 or float(anchor_boxes[0][2]) <= 0:
        return

    anchor_box = anchor_amodal if float(anchor_amodal[2]) > 0 else anchor_boxes[0]
    area_floor = anchor_size_pixels(anchor_box, frames[0].shape)
    anchor = entity_tokens(sam_encoder, backbones, frames[0], [anchor_boxes[0]], area_floor, config.crop_size)
    query = anchor[0]
    clean_boxes = load_clean_boxes_by_frame(visible_json, person)

    frame_height, frame_width = frames[0].shape[:2]
    px_scale = 1024.0 / max(frame_width, frame_height)
    anchor_center = to_pixel(anchor_boxes[0], frame_width, frame_height, px_scale)

    for t in range(1, len(frames), config.stride):
        distractors = clean_distractors(clean_boxes.get(int(frame_indices[t]), []), config.n_distractors, random_generator)
        if occlusions[t] > 0.5 or float(anchor_boxes[t][2]) <= 0 or not distractors:
            continue

        candidate_boxes = [anchor_boxes[t]] + distractors
        candidate_dists = [distance(to_pixel(anchor_box, frame_width, frame_height, px_scale), anchor_center) for anchor_box in candidate_boxes]
        candidates = entity_tokens(sam_encoder, backbones, frames[t], candidate_boxes, area_floor, config.crop_size)

        for encoder_name in backbones:
            pairs = [chamfer_scores(query[encoder_name], candidate[encoder_name]) for candidate in candidates]
            uni_scores = [unidirectional_score for unidirectional_score, _ in pairs]
            bi_scores = [bidirectional_score for _, bidirectional_score in pairs]
            if any(np.isnan(bi_scores)):
                continue
            yield encoder_name, candidate_dists, bi_scores, uni_scores


def auc(samples):
    """AUC separating target (label 1) from distractor (label 0) over samples
    (candidate_distance, label, score, trajectory_key). NaN if empty or single-class."""

    if not samples:
        return float("nan")
    labels = np.array([sample_row[1] for sample_row in samples])
    scores = np.array([sample_row[2] for sample_row in samples])
    return roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else float("nan")


def load_checkpoint(results_path):
    """Load (results_bi, results_uni, per_traj, done) from the checkpoint, or empties when there is none.
    results_bi/uni are the bidirectional/unidirectional chamfer samples; per_traj maps trajectory key ->"""

    if not Path(results_path).exists():
        return defaultdict(list), defaultdict(list), {}, set()
    state = pickle.loads(Path(results_path).read_bytes())
    return (
        defaultdict(list, state["results_bi"]),
        defaultdict(list, state["results_uni"]),
        state.get("per_traj", {}),
        set(state["done"]))


def save_checkpoint(results_path, results_bi, results_uni, per_traj, completed_stems):
    """Atomically write the bidirectional/unidirectional samples, per-trajectory AUC, and finished keys so
    the run resumes here. Also dumps per_traj to a standalone npy for offline hardness analysis."""

    tmp = Path(results_path).with_suffix(".pkl.tmp")
    tmp.write_bytes(
        pickle.dumps(
            {
                "results_bi": dict(results_bi),
                "results_uni": dict(results_uni),
                "per_traj": per_traj,
                "done": list(completed_stems)}
        )
    )
    tmp.replace(results_path)
    np.save("data/claim_3/per_trajectory_auc.npy", per_traj, allow_pickle=True)


@hydra.main(config_path="../conf", config_name="experiments/claim_3", version_base=None)
def run_reid(config: DictConfig):
    """Score every backbone's foreground similarity as re-ID against distractors, on the test trajectories.
    Resumable: the checkpoint is the result, and claim_3_visualize.py draws from it."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    sam_encoder = hydra.utils.instantiate(config.tracker.tracker).model
    backbones = load_backbones(config)

    colors = OmegaConf.to_container(config.colors)
    groups = OmegaConf.to_container(config.groups)
    trajectories = test_trajectories(person_path, config.n_traj)
    results_bi, results_uni, per_traj, completed_stems = load_checkpoint(config.checkpoint)

    random_generator = np.random.RandomState(0)
    for completed, trajectory in enumerate(tqdm(trajectories, desc="re-id"), start=1):
        key = f"{trajectory[0]}_{trajectory[1]}_{trajectory[2]}"
        if key in completed_stems:
            continue

        this_traj = defaultdict(list)
        for encoder_name, dists, bi_scores, uni_scores in evaluate_trajectory(
            sam_encoder, backbones, detection_data, trajectory, config, random_generator
        ):
            for index, (candidate_distance, bidirectional_score, unidirectional_score) in enumerate(
                zip(dists, bi_scores, uni_scores)
            ):
                label = 1 if index == 0 else 0
                this_traj[encoder_name].append((candidate_distance, label, bidirectional_score, key))
                results_uni[encoder_name].append((candidate_distance, label, unidirectional_score, key))
        for encoder_name, samples in this_traj.items():
            results_bi[encoder_name].extend(samples)
        per_traj[key] = {encoder_name: auc(samples) for encoder_name, samples in this_traj.items()}
        completed_stems.add(key)

        if completed % config.checkpoint_every == 0:
            save_checkpoint(config.checkpoint, results_bi, results_uni, per_traj, completed_stems)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    save_checkpoint(config.checkpoint, results_bi, results_uni, per_traj, completed_stems)
    for encoder_name in results_bi:
        print(
            f"  {encoder_name:11s} bi={auc(results_bi[encoder_name]):.3f}   uni={auc(results_uni[encoder_name]):.3f}   (n={len(results_bi[encoder_name])})"
        )


if __name__ == "__main__":
    run_reid()

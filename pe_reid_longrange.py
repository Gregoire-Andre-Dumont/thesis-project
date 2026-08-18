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

from src.offline_training.dataset_encoders import crop_around_masks, anchor_size_pixels, load_dataset_encoders, _norm, _patch_masks, HALF
from src.offline_training.dataset_labels import load_clean_boxes_by_frame, _box_center as box_center
from src.utils.load_bboxes import convert_bbox, load_bboxes
from create_anchor_dataset import anchor_trajectory_index, slice_detection_data_for_tracker


logging.getLogger("timm").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["HYDRA_FULL_ERROR"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def test_trajectories(person_path, n_traj, test_size=0.20, random_seed=42):
    """List the selected trajectories, held-out test split first, extended with the rest up to n_traj.
    The split is by trajectory and seeded so it stays the same across runs."""

    triples = [(v, int(p), int(a)) for v, p, a in zip(
        person_path.selected_video_names.tolist(),
        person_path.selected_person_ids.tolist(),
        person_path.selected_anchor_video_frames.tolist())]

    order = sorted(range(len(triples)), key=lambda i: f"{triples[i][0]}_{triples[i][1]}")
    train_indices, test_indices = train_test_split(np.array(order), test_size=test_size, random_state=random_seed, shuffle=True)
    ordered = [triples[i] for i in sorted(test_indices)] + [triples[i] for i in sorted(train_indices)]
    return ordered[:n_traj]


def load_sam3_encoder(sam3_config, sam3_input):
    """Load SAM 3's frozen vision encoder as a token function.
    Built from the bundled config plus the ungated 1038lab/sam3 mirror weights."""

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
    def tokens(crops):
        output = model(_norm(crops, sam3_input, HALF, HALF, DEVICE, DTYPE))
        return (output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]).float()
    return tokens


def load_backbones(sam3_config, sam3_input):
    """Load the five LARGE backbones (~212-454M params) keyed by name; each emits a 32x32 token grid.
    pe_spatial / hiera_sam / hiera_mae are the shared calibrator encoders; pe_core and pe_sam3 are added here."""

    shared = load_dataset_encoders(["perception", "hiera_sam", "hiera_mae"], DEVICE, DTYPE)
    pe_core = timm.create_model("vit_pe_core_large_patch14_336.fb", pretrained=True, num_classes=0, img_size=448).eval().to(DEVICE).to(DTYPE)

    @torch.inference_mode()
    def pe_core_tokens(crops):
        return pe_core.forward_features(_norm(crops, 448, HALF, HALF, DEVICE, DTYPE))[:, pe_core.num_prefix_tokens:].float()

    return {
        "pe_core": pe_core_tokens,
        "pe_spatial": shared["perception"],
        "hiera_mae": shared["hiera_mae"],
        "hiera_sam": shared["hiera_sam"],
        "pe_sam3": load_sam3_encoder(sam3_config, sam3_input),
    }


@torch.inference_mode()
def box_prompt_masks(sam, frame, boxes):
    """Return one SAM mask logit map per box from a single frame encode.
    Logits (not binary masks) are returned so crop_around_masks can threshold after resizing."""

    image = sam.image_encoder.prepare_image(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 1024, True)
    features = sam.image_encoder(image)
    prompts = [convert_bbox(np.asarray(box, np.float32)) for box in boxes]
    masks = [sam.initialize_video_masking(features, prompt)[0] for prompt in prompts]
    return [mask.squeeze().float().cpu().numpy() for mask in masks]


@torch.inference_mode()
def foreground_tokens(sam, backbones, frame, boxes, floor, crop_size):
    """Return each box's foreground tokens for every backbone, cropped at the anchor scale (floor).
    Returns a list aligned to boxes, or None if any box yields an empty mask."""

    box_masks = box_prompt_masks(sam, frame, boxes)
    if any((box_mask > 0).sum() == 0 for box_mask in box_masks):
        return None

    # Crop every box out of the (repeated) frame in one call, floored at the anchor scale.
    frame_per_box = np.repeat(frame[None], len(box_masks), axis=0)
    box_crops, box_crop_masks = crop_around_masks(frame_per_box, np.stack(box_masks).astype(np.float32), crop_size, 0.25, floor)

    foreground_by_box = [{} for _ in boxes]
    for backbone_name, encode in backbones.items():
        patch_tokens = encode(box_crops)                                        # (num_boxes, 32*32, dim)
        foreground_masks = _patch_masks(box_crop_masks, patch_tokens.device)

        for box_foreground, box_patch_tokens, mask in zip(foreground_by_box, patch_tokens, foreground_masks):
            box_foreground[backbone_name] = box_patch_tokens[mask].clone()
        del patch_tokens 
                        
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return foreground_by_box


@torch.inference_mode()
def chamfer_similarity(query_fg, candidate_fg):
    """Unidirectional chamfer: mean best cosine of each candidate foreground patch to the anchor's."""

    if query_fg.shape[0] == 0 or candidate_fg.shape[0] == 0:
        return float("nan")
    query = F.normalize(query_fg, dim=-1)
    candidate = F.normalize(candidate_fg, dim=-1)
    return float((candidate @ query.T).max(dim=1).values.mean())


def distance(a, b):
    """Euclidean distance between two points."""

    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def uniform_distractors(anchor_center, clean_boxes, rng, n_distractors, dist_range):
    """Pick n_distractors clean-person boxes whose distance from the anchor is spread uniformly over dist_range.
    For each slot, sample a target distance and take the nearest still-available box to it."""

    lo, hi = dist_range
    pool = [(box, distance(box_center(box), anchor_center)) for box in clean_boxes]
    pool = [(box, dist) for box, dist in pool if lo <= dist <= hi]

    chosen = []
    while pool and len(chosen) < n_distractors:
        target = rng.uniform(lo, hi)
        nearest = min(range(len(pool)), key=lambda i: abs(pool[i][1] - target))
        chosen.append(pool.pop(nearest))
    return chosen


def evaluate_trajectory(sam, backbones, detection_data, trajectory, config, rng):
    """Yield (backbone, dists, scores) per visible frame; index 0 is the target, the rest distractors.
    dists[i] is candidate i's distance from the anchor centre, scores[i] its chamfer to the anchor query."""

    video, person, anchor_frame = trajectory
    detection_data.initialize_target(video, person)
    anchor_index = anchor_trajectory_index(detection_data, anchor_frame)
    if anchor_index is None:
        return

    # The anchor's amodal box (full extent, incl. occluded parts) sets the crop scale for the trajectory.
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

    # Anchor query: the target's foreground tokens at frame 0, cropped at the anchor's amodal scale.
    anchor_box = anchor_amodal if float(anchor_amodal[2]) > 0 else boxes[0]
    floor = anchor_size_pixels(anchor_box, frames[0].shape)
    anchor = foreground_tokens(sam, backbones, frames[0], [boxes[0]], floor, config.crop_size)
    if anchor is None:
        return
    query = anchor[0]
    anchor_center = box_center(boxes[0])
    clean_boxes = load_clean_boxes_by_frame(visible_json, person)

    for t in range(1, len(frames), config.stride):
        distractors = uniform_distractors(anchor_center, clean_boxes.get(int(frame_indices[t]), []),
                                          rng, config.n_distractors, config.distractor_dist_range)
        if occlusions[t] > 0.5 or float(boxes[t][2]) <= 0 or not distractors:
            continue

        # Candidates are the target (index 0) then the distractors, each with its distance from the anchor.
        candidate_boxes = [boxes[t]] + [box for box, _ in distractors]
        candidate_dists = [distance(box_center(boxes[t]), anchor_center)] + [dist for _, dist in distractors]
        candidates = foreground_tokens(sam, backbones, frames[t], candidate_boxes, floor, config.crop_size)
        if candidates is None:
            continue

        for name in backbones:
            scores = [chamfer_similarity(query[name], candidate[name]) for candidate in candidates]
            if not any(np.isnan(scores)):
                yield name, candidate_dists, scores


def auc(samples):
    """AUC separating target (label 1) from distractor (label 0) over samples (candidate_distance, label, score).
    NaN if empty or single-class."""

    if not samples:
        return float("nan")
    labels = np.array([s[1] for s in samples])
    scores = np.array([s[2] for s in samples])
    return roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else float("nan")


def binned_auc(samples, edges, min_bin_samples):
    """Target-vs-distractor AUC per bin of the candidate's own distance from the anchor; NaN for thin bins."""

    bins = [[s for s in samples if lo <= s[0] < hi] for lo, hi in zip(edges[:-1], edges[1:])]
    return [auc(b) if len(b) >= min_bin_samples else np.nan for b in bins]


def plot_auc(results, spec, edges, colors, min_bin_samples, n_traj):
    """Plot one figure spec (its backbones' AUC vs candidate distance from anchor) and save the figure and curves."""

    curves = {name: binned_auc(results[name], edges, min_bin_samples) for name in spec.group if name in results}
    centers = (edges[:-1] + edges[1:]) / 2

    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    for name, curve in curves.items():
        axis.plot(centers, curve, marker="o", markersize=5, color=colors.get(name), label=name)
    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (0.50)")

    axis.set_xlabel("candidate distance from anchor (normalized)")
    axis.set_ylabel("target-vs-distractor AUC")
    axis.set_title(f"Re-ID via foreground similarity  |  {n_traj} trajectories", fontsize=10)
    axis.set_ylim(0.45, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower left", fontsize=8)

    figure.tight_layout()
    figure.savefig(spec.plot, dpi=120)
    plt.close(figure)
    np.save(spec.npy, {"edges": edges, "curves": curves, "n_traj": n_traj}, allow_pickle=True)


def load_checkpoint(path):
    """Load (results, done) from the checkpoint file, or an empty pair when there is none."""

    if not Path(path).exists():
        return defaultdict(list), set()
    state = pickle.loads(Path(path).read_bytes())
    return defaultdict(list, state["results"]), set(state["done"])


def save_checkpoint(path, results, done):
    """Atomically write the raw pairs and finished-trajectory keys so the next run resumes here."""

    tmp = Path(path).with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps({"results": dict(results), "done": list(done)}))
    tmp.replace(path)


@hydra.main(config_path="conf", config_name="pe_reid", version_base=None)
def run_reid(config: DictConfig):
    """Score every backbone's foreground similarity as re-ID against distractors, on the test trajectories.
    Resumes from a checkpoint and refreshes the two comparison figures after each trajectory."""

    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    sam = hydra.utils.instantiate(config.tracker.tracker).model
    backbones = load_backbones(config.sam3_vision_config, config.sam3_input)
    print(f"backbones: {list(backbones)}")

    edges = np.array(config.spatial_edges, dtype=float)
    colors = OmegaConf.to_container(config.colors)
    trajectories = test_trajectories(person_path, config.n_traj)
    results, done = load_checkpoint(config.checkpoint)
    print(f"benchmarking {len(trajectories)} trajectories; resuming with {len(done)} done")

    rng = np.random.RandomState(0)
    for completed, trajectory in enumerate(tqdm(trajectories, desc="re-id"), start=1):
        key = f"{trajectory[0]}_{trajectory[1]}_{trajectory[2]}"
        if key in done:
            continue

        for name, dists, scores in evaluate_trajectory(sam, backbones, detection_data, trajectory, config, rng):
            for index, (candidate_distance, score) in enumerate(zip(dists, scores)):
                label = 1 if index == 0 else 0               # index 0 is the target, the rest are distractors
                results[name].append((candidate_distance, label, score))
        done.add(key)

        for spec in config.figures:
            plot_auc(results, spec, edges, colors, config.min_bin_samples, len(done))
        if completed % config.checkpoint_every == 0:
            save_checkpoint(config.checkpoint, results, done)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    save_checkpoint(config.checkpoint, results, done)


if __name__ == "__main__":
    run_reid()

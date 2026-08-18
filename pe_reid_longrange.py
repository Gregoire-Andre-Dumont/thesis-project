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
def entity_tokens(sam, backbones, frame, boxes, floor, crop_size):
    """Return each box's (foreground, background) tokens for every backbone, cropped around its own mask
    but floored at the anchor scale. Returns a list aligned to boxes ({backbone: (fg, bg)}), or None if any mask is empty."""

    box_masks = box_prompt_masks(sam, frame, boxes)
    if any((box_mask > 0).sum() == 0 for box_mask in box_masks):
        return None

    # Crop every box out of the (repeated) frame in one call, floored at the anchor scale.
    frame_per_box = np.repeat(frame[None], len(box_masks), axis=0)
    box_crops, box_crop_masks = crop_around_masks(frame_per_box, np.stack(box_masks).astype(np.float32), crop_size, 0.5, floor)

    tokens_by_box = [{} for _ in boxes]
    for backbone_name, encode in backbones.items():
        patch_tokens = encode(box_crops)                                        # (num_boxes, 32*32, dim)
        foreground_masks = _patch_masks(box_crop_masks, patch_tokens.device)

        for box_tokens, box_patch_tokens, mask in zip(tokens_by_box, patch_tokens, foreground_masks):
            box_tokens[backbone_name] = (box_patch_tokens[mask].clone(), box_patch_tokens[~mask].clone())
        del patch_tokens                                # free this backbone before the next large model runs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return tokens_by_box


@torch.inference_mode()
def chamfer_similarity(reference_fg, tokens):
    """Bidirectional (symmetric) chamfer: average of tokens->anchor and anchor->tokens mean best cosine.
    NaN if either side is empty."""

    if reference_fg.shape[0] == 0 or tokens.shape[0] == 0:
        return float("nan")
    reference = F.normalize(reference_fg, dim=-1)
    tokens = F.normalize(tokens, dim=-1)
    similarity = tokens @ reference.T                        # (n_tokens, n_anchor_fg)
    tokens_to_anchor = similarity.max(dim=1).values.mean()   # each candidate token -> best anchor patch
    anchor_to_tokens = similarity.max(dim=0).values.mean()   # each anchor patch -> best candidate token
    return float(0.5 * (tokens_to_anchor + anchor_to_tokens))


def distance(a, b):
    """Euclidean distance between two points."""

    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def to_pixel(box, width, height, scale):
    """Box centre in pixels at the common working resolution -- un-normalized, and isotropic in x and y
    (both scaled the same), unlike the per-axis normalized coordinates."""

    cx, cy = box_center(box)
    return cx * width * scale, cy * height * scale


def clean_distractors(clean_boxes, n_distractors, min_norm_area, rng):
    """Up to n_distractors clean people whose box is large enough (>= min_norm_area -- the same visible-area
    floor the target satisfies), chosen at random rather than by proximity, so they are not biased toward
    the target's local background."""

    pool = [box for box in clean_boxes if box[2] * box[3] >= min_norm_area]
    if len(pool) <= n_distractors:
        return pool
    return [pool[i] for i in rng.choice(len(pool), size=n_distractors, replace=False)]


def evaluate_trajectory(sam, backbones, detection_data, trajectory, config, rng):
    """Yield (backbone, dists, fg_scores, fgbg_scores) per visible frame; index 0 is the target, rest distractors.
    fg = candidate foreground vs anchor foreground; fgbg = that minus candidate background vs anchor foreground."""

    video, person, anchor_frame = trajectory
    detection_data.initialize_target(video, person)
    anchor_index = anchor_trajectory_index(detection_data, anchor_frame)
    if anchor_index is None:
        return

    # The anchor's amodal box (full extent, incl. occluded parts) sets the crop-size floor for the trajectory.
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
    anchor = entity_tokens(sam, backbones, frames[0], [boxes[0]], floor, config.crop_size)
    if anchor is None:
        return
    query = anchor[0]
    clean_boxes = load_clean_boxes_by_frame(visible_json, person)

    # Distances are measured in pixels at the common 1024 working resolution (un-normalized, isotropic).
    height, width = frames[0].shape[:2]
    px_scale = 1024.0 / max(width, height)
    anchor_center = to_pixel(boxes[0], width, height, px_scale)
    min_norm_area = config.person_path.min_visible_area / (width * height * px_scale ** 2)   # target's visible-area floor, in normalized area

    for t in range(1, len(frames), config.stride):
        distractors = clean_distractors(clean_boxes.get(int(frame_indices[t]), []), config.n_distractors, min_norm_area, rng)
        if occlusions[t] > 0.5 or float(boxes[t][2]) <= 0 or not distractors:
            continue

        # Candidates are the target (index 0) then its clean distractors, each binned by its distance from the anchor.
        candidate_boxes = [boxes[t]] + distractors
        candidate_dists = [distance(to_pixel(box, width, height, px_scale), anchor_center) for box in candidate_boxes]
        candidates = entity_tokens(sam, backbones, frames[t], candidate_boxes, floor, config.crop_size)
        if candidates is None:
            continue

        for name in backbones:
            anchor_fg = query[name][0]
            fg_scores = [chamfer_similarity(anchor_fg, candidate[name][0]) for candidate in candidates]   # candidate FG vs anchor FG
            bg_scores = [chamfer_similarity(anchor_fg, candidate[name][1]) for candidate in candidates]   # candidate BG vs anchor FG
            if any(np.isnan(fg_scores)) or any(np.isnan(bg_scores)):
                continue
            fgbg_scores = [fg - bg for fg, bg in zip(fg_scores, bg_scores)]
            yield name, candidate_dists, fg_scores, fgbg_scores


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


def quantile_edges(results, n_bins):
    """Equal-count distance-bin edges from the pooled candidate distances (identical across backbones),
    so each bin holds the same number of samples. None until there are enough samples to bin."""

    samples = next(iter(results.values()), [])
    if len(samples) < n_bins:
        return None
    distances = np.array([s[0] for s in samples])
    edges = np.quantile(distances, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9                                    # make the top edge inclusive
    return edges


def diagnose(results, edges):
    """Print, per backbone and bin, the target/distractor counts and mean similarity, so the AUC shape
    can be read off directly: a dip is either the gap collapsing (dis_sim rising to tgt_sim) or the bin
    being target-starved (tiny n_tgt)."""

    for name, samples in results.items():
        print(f"\n{name}")
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = [s for s in samples if lo <= s[0] < hi]
            target_sims = [s[2] for s in in_bin if s[1] == 1]
            distractor_sims = [s[2] for s in in_bin if s[1] == 0]
            if target_sims and distractor_sims:
                tgt, dis = np.mean(target_sims), np.mean(distractor_sims)
                print(f"  d[{lo:.1f},{hi:.1f}) n_tgt={len(target_sims):4d} n_dis={len(distractor_sims):5d} "
                      f"tgt_sim={tgt:.3f} dis_sim={dis:.3f} gap={tgt - dis:+.3f}")


def plot_auc(results, spec, edges, colors, min_bin_samples, n_traj, metric="foreground similarity"):
    """Plot one figure spec (its backbones' AUC vs candidate distance from anchor) and save the figure and curves."""

    curves = {name: binned_auc(results[name], edges, min_bin_samples) for name in spec.group if name in results}

    # Place each point at its bin's median distance (bins are equal-count, so unevenly spaced in distance).
    ref = next((results[name] for name in spec.group if name in results), [])
    ref_distances = np.array([s[0] for s in ref])
    centers = [np.median(ref_distances[(ref_distances >= lo) & (ref_distances < hi)])
               if ((ref_distances >= lo) & (ref_distances < hi)).any() else (lo + hi) / 2
               for lo, hi in zip(edges[:-1], edges[1:])]

    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    for name, curve in curves.items():
        axis.plot(centers, curve, marker="o", markersize=5, color=colors.get(name), label=name)
    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance (0.50)")

    axis.set_xlabel("candidate distance from anchor (pixels @1024, equal-count bins)")
    axis.set_ylabel("target-vs-distractor AUC")
    axis.set_title(f"Re-ID via {metric}  |  {n_traj} trajectories", fontsize=10)
    axis.set_ylim(0.45, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower left", fontsize=8)

    figure.tight_layout()
    figure.savefig(spec.plot, dpi=120)
    plt.close(figure)
    np.save(spec.npy, {"edges": edges, "curves": curves, "n_traj": n_traj}, allow_pickle=True)


def load_checkpoint(path):
    """Load (results_fg, results_fgbg, done) from the checkpoint, or empties when there is none."""

    if not Path(path).exists():
        return defaultdict(list), defaultdict(list), set()
    state = pickle.loads(Path(path).read_bytes())
    return defaultdict(list, state["results_fg"]), defaultdict(list, state["results_fgbg"]), set(state["done"])


def save_checkpoint(path, results_fg, results_fgbg, done):
    """Atomically write the FG and FG-BG samples plus finished-trajectory keys so the next run resumes here."""

    tmp = Path(path).with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps({"results_fg": dict(results_fg), "results_fgbg": dict(results_fgbg), "done": list(done)}))
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

    colors = OmegaConf.to_container(config.colors)
    trajectories = test_trajectories(person_path, config.n_traj)
    results_fg, results_fgbg, done = load_checkpoint(config.checkpoint)
    print(f"benchmarking {len(trajectories)} trajectories; resuming with {len(done)} done")

    rng = np.random.RandomState(0)          # reproducible random distractor choice
    for completed, trajectory in enumerate(tqdm(trajectories, desc="re-id"), start=1):
        key = f"{trajectory[0]}_{trajectory[1]}_{trajectory[2]}"
        if key in done:
            continue

        for name, dists, fg_scores, fgbg_scores in evaluate_trajectory(sam, backbones, detection_data, trajectory, config, rng):
            for index, (candidate_distance, fg, fgbg) in enumerate(zip(dists, fg_scores, fgbg_scores)):
                label = 1 if index == 0 else 0               # index 0 is the target, the rest are distractors
                results_fg[name].append((candidate_distance, label, fg))
                results_fgbg[name].append((candidate_distance, label, fgbg))
        done.add(key)

        edges = quantile_edges(results_fg, config.n_bins)
        if edges is None:
            continue
        for spec in config.figures:
            plot_auc(results_fg, spec, edges, colors, config.min_bin_samples, len(done))
        plot_auc(results_fgbg, config.fgbg_figure, edges, colors, config.min_bin_samples, len(done), metric="FG - BG")
        if completed % config.checkpoint_every == 0:
            save_checkpoint(config.checkpoint, results_fg, results_fgbg, done)
            diagnose(results_fg, edges)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    save_checkpoint(config.checkpoint, results_fg, results_fgbg, done)
    print("\noverall target-vs-distractor AUC (FG | FG-BG):")
    for name in results_fg:
        print(f"  {name:11s} FG={auc(results_fg[name]):.3f}   FG-BG={auc(results_fgbg[name]):.3f}   (n={len(results_fg[name])})")


if __name__ == "__main__":
    run_reid()

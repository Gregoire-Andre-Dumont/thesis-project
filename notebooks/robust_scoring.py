"""Robust scoring: covariate shift of a SAM 2 memory tracker's re-ID gate under real memory-bank corruption.

We roll SAM 2 out over each trajectory TWICE. The clean pass is the true-IoU oracle: it commits a crop to SAM 2's
memory bank only when the target is visible and the prediction is good. The corrupted pass is the same tracker but,
at each such commit, with probability `corruption_p` it instead commits a nearby distractor's mask (box-prompted
from one of the k nearest people) into the bank -- a controlled, real poisoning that feeds back into the rollout.

To isolate the memory's effect on the re-ID gate we keep ONE fixed candidate set -- the clean pass's predicted
masks -- and score each candidate against two re-ID memories: the CLEAN pass's committed crops and the CORRUPTED
pass's committed crops (which include the bad occlusion commits). We label every candidate on-target vs
distractor-grab by mask overlap and score its re-ID similarity with the Perception Encoder,

    score(alpha) = alpha * chamfer(candidate, anchor) + (1 - alpha) * chamfer(candidate, memory)

sweeping the anchor weight alpha. Per encoder we draw the clean-memory and corrupted-memory AUC-vs-alpha curves
with the covariate-shift gap shaded. Same candidates and anchor mean the curves coincide at alpha=1; the gap at
alpha<1 is exactly the poisoned memory failing to tell the target from a grab.
"""

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
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root on path for `src` and create_anchor_dataset

from pe_reid_longrange import test_trajectories, box_prompt_masks, chamfer_scores, distance, to_pixel, DEVICE, DTYPE
from create_anchor_dataset import anchor_trajectory_index
from src.offline_training.dataset_encoders import crop_around_masks, anchor_size_pixels, load_dataset_encoders
from src.offline_training.dataset_labels import load_clean_boxes_by_frame
from src.utils.load_bboxes import load_bboxes


logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
os.environ["HYDRA_FULL_ERROR"] = "1"

CLEAN_COLOR = "#22aa77"                           # the clean memory / bank
CORRUPT_COLOR = "#cc4444"                         # the corrupted memory / bank


def mask_iou(mask_a, mask_b):
    """IoU of two binary masks (0 when their union is empty)."""

    union = np.logical_or(mask_a, mask_b).sum()
    return float(np.logical_and(mask_a, mask_b).sum() / union) if union > 0 else 0.0


def mask_to_box(mask):
    """Normalized xywh bounding box of a mask (all zeros for an empty mask)."""

    rows, cols = np.where(mask > 0)
    if len(cols) == 0:
        return np.zeros(4, np.float32)
    size = float(mask.shape[0])
    return np.array([cols.min() / size, rows.min() / size,
                     (cols.max() - cols.min() + 1) / size, (rows.max() - rows.min() + 1) / size], np.float32)


def anchor_distance(predicted_mask, anchor_center, width, height, pixel_scale):
    """Pixel distance (@1024) between a candidate mask's box centre and the anchor's centre; NaN for an empty mask."""

    box = mask_to_box(predicted_mask)
    if box[2] <= 0:
        return np.nan
    return distance(to_pixel(box, width, height, pixel_scale), anchor_center)


def build_corruption_boxes(detection_data, clean_boxes):
    """Per frame from the first occlusion onward, the nearest OTHER person to the target (its last visible position
    carried forward during occlusion). NON-STICKY: the injected identity switches frame to frame. None before the
    first occlusion, or where no person is present. Only the corrupt rollout (corruption_p > 0) uses it."""

    boxes, frame_indices = detection_data.bboxes_norm, detection_data.frame_indices
    occlusions, shape = detection_data.occlusions, detection_data.frames[0].shape
    height, width, n = shape[0], shape[1], len(boxes)
    first_occlusion = int(np.argmax(occlusions > 0)) if float(occlusions.max()) > 0 else n

    corruption_boxes, last_box = [], None
    for index in range(n):
        if float(boxes[index][2]) > 0:
            last_box = boxes[index]
        people = clean_boxes.get(int(frame_indices[index]), [])
        if index < first_occlusion or last_box is None or not people:
            corruption_boxes.append(None)
            continue
        ref_x, ref_y = last_box[0] + last_box[2] / 2.0, last_box[1] + last_box[3] / 2.0
        nearest = min(people, key=lambda b: np.hypot((b[0] + b[2] / 2.0 - ref_x) * width, (b[1] + b[3] / 2.0 - ref_y) * height))
        corruption_boxes.append(np.asarray(nearest, np.float32))
    return corruption_boxes


def box_masks(model, frame, boxes):
    """Binary SAM box-prompt masks for a list of boxes, from a single frame encode (empty list for no boxes)."""

    return [logits > 0 for logits in box_prompt_masks(model, frame, boxes)] if len(boxes) else []


@torch.inference_mode()
def encode_foregrounds(encoders, frames, masks, floor, crop_size, chunk=8):
    """Foreground patch tokens of each (frame, mask) for every encoder, cropped around the mask at the anchor
    scale. Returns a list of {encoder: tokens}; a token tensor is empty if the mask vanished."""

    crops, crop_masks = crop_around_masks(frames, masks.astype(np.float32), crop_size, 0.2, floor)
    crop_masks_tensor = torch.from_numpy(crop_masks).unsqueeze(1)

    foreground_per_item = [{} for _ in range(len(masks))]
    for encoder_name, encode in encoders.items():
        for start in range(0, len(crops), chunk):
            span = slice(start, start + chunk)
            tokens = encode(crops[span])
            grid_size = round(tokens.shape[1] ** 0.5)
            resized_masks = F.interpolate(crop_masks_tensor[span].to(tokens.device).float(), size=(grid_size, grid_size), mode="nearest")
            foreground = (resized_masks > 0.5).flatten(1)
            for item, item_tokens, item_foreground in zip(foreground_per_item[span], tokens, foreground):
                item[encoder_name] = item_tokens[item_foreground].clone()
            del tokens
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    return foreground_per_item


def chamfer(reference, candidate):
    """Unidirectional foreground chamfer (candidate -> reference); NaN if either side is empty."""

    if reference.shape[0] == 0 or candidate.shape[0] == 0:
        return np.nan
    unidirectional, _bidirectional = chamfer_scores(reference, candidate)
    return unidirectional


def mean_chamfer(candidate, memory):
    """Mean chamfer of a candidate to the memory entries, skipping empty (vanished) entries; NaN if none remain."""

    finite = [value for value in (chamfer(entry, candidate) for entry in memory) if value == value]
    return float(np.mean(finite)) if finite else np.nan


def causal_recent(positions, tokens, frame, memory_size):
    """The memory tokens committed strictly before `frame`, keeping only the most recent `memory_size`."""

    return [token for position, token in zip(positions, tokens) if position < frame][-memory_size:]


def load_window(detection_data, trajectory, max_frames):
    """Load only the frames the tracker needs -- the warmup frame plus `max_frames` from the anchor -- into
    `detection_data`. Returns (warmup, anchor_index), or None if the anchor is missing."""

    video, person, anchor_frame = trajectory
    detection_data.load_frames = False
    detection_data.initialize_target(video, person)
    anchor_index = anchor_trajectory_index(detection_data, anchor_frame)
    if anchor_index is None:
        return None
    warmup = 1 if anchor_index >= 1 else 0
    window = detection_data.frame_indices[anchor_index - warmup:anchor_index - warmup + warmup + max_frames]
    detection_data.load_frames = True
    detection_data.initialize_target(video, person, frame_indices=window)
    return warmup, anchor_index


def roll_out(oracle, detection_data, corruption_p, warmup):
    """Run one SAM 2 rollout over the loaded window. `corruption_p` > 0 lets the tracker commit its prediction to
    the memory bank on occluded frames, poisoning it. Returns the post-warmup predicted masks and the post-warmup
    indices of the frames it committed to memory."""

    oracle.corruption_p = corruption_p
    predicted = oracle.predict_masks(detection_data).numpy()[warmup:]
    iou = oracle.iou_scores.numpy()[warmup:]          # SAM 2's own predicted-IoU (its self-confidence) per frame
    committed = [frame - warmup for frame in oracle.committed_frames if frame >= warmup]
    committed_masks = [mask for frame, mask in zip(oracle.committed_frames, oracle.committed_masks) if frame >= warmup]
    return {"predicted": predicted, "iou": iou, "committed": committed, "committed_masks": committed_masks}


def committed_memory(encoders, frames, rollout, floor, crop_size):
    """Foreground tokens (per encoder) of the actual crops a rollout committed to memory (target crops, or
    distractor crops where corrupted), with their frame positions."""

    committed = rollout["committed"]
    if not committed:
        return committed, []
    masks = np.stack(rollout["committed_masks"])
    return committed, encode_foregrounds(encoders, frames[committed], masks, floor, crop_size)


def anchor_floor(config, trajectory, anchor_index, frames, boxes):
    """Pixel crop floor from the anchor's amodal box, so every crop is taken at the target's true scale."""

    video, person, _ = trajectory
    amodal_json = Path(config.detection_data.amodal_directory) / f"{video}.json"
    visible_json = Path(config.detection_data.visible_directory) / f"{video}.json"
    anchor_amodal = load_bboxes(str(amodal_json), str(visible_json), person, use_amodal=True)[anchor_index]
    anchor_box = anchor_amodal if float(anchor_amodal[2]) > 0 else boxes[0]
    return anchor_size_pixels(anchor_box, frames[0].shape)


def prompt_labels(model, frames, boxes, occlusions, frame_indices, clean_boxes, scored):
    """Per scored frame, the box-prompted target mask (None while occluded) and the masks of every other visible
    person. Returns (target_masks, distractor_masks); a candidate is labelled by its overlap with these."""

    target_masks = []
    distractor_masks = []
    for frame in scored:
        visible = occlusions[frame] < 0.5 and float(boxes[frame][2]) > 0
        people = [np.asarray(box, np.float32) for box in clean_boxes[int(frame_indices[frame])]]
        prompted = box_masks(model, frames[frame], ([boxes[frame]] if visible else []) + people)
        target_masks.append(prompted[0] if visible else None)
        distractor_masks.append(prompted[1:] if visible else prompted)
    return target_masks, distractor_masks


def candidate_overlaps(predicted, scored, target_masks, distractor_masks):
    """(target_iou, distractor_iou) of each scored-frame candidate mask vs the box-prompted target and distractors."""

    overlaps = []
    for index, frame in enumerate(scored):
        predicted_mask = predicted[frame] > 0
        target_iou = mask_iou(predicted_mask, target_masks[index]) if target_masks[index] is not None else 0.0
        distractor_iou = max((mask_iou(predicted_mask, mask) for mask in distractor_masks[index]), default=0.0)
        overlaps.append((target_iou, distractor_iou))
    return overlaps


def evaluate_trajectory(oracle, encoders, detection_data, trajectory, config):
    """Roll SAM 2 out clean and corrupted over the trajectory and yield two families of tagged candidate samples:

    ("pe", encoder): the PE re-ID gate. Candidates POOLED from both rollouts (clean + corrupt predictions), each
        scored against the clean vs the corrupted memory -> (target_iou, distractor_iou, anchor, clean, corrupt, distance).
    ("iou", pass):   SAM 2's own predicted-IoU gate. Each pass's own predictions scored by SAM's self-confidence
        -> (target_iou, distractor_iou, sam_iou, distance). Its covariate shift is clean-bank vs corrupt-bank AUC."""

    window = load_window(detection_data, trajectory, config.max_frames)
    if window is None:
        return
    warmup, anchor_index = window

    video, person, _ = trajectory
    visible_json = Path(config.detection_data.visible_directory) / f"{video}.json"
    clean_boxes = load_clean_boxes_by_frame(visible_json, person)
    oracle.corruption_boxes = build_corruption_boxes(detection_data, clean_boxes)

    oracle.frame_cache = {}                          # the clean pass fills it; the corrupt pass reuses the encodings
    clean = roll_out(oracle, detection_data, 0.0, warmup)
    corrupt = roll_out(oracle, detection_data, config.corruption_p, warmup)
    oracle.frame_cache = None
    oracle.corruption_boxes = None

    frames = detection_data.frames[warmup:]
    boxes = detection_data.bboxes_norm[warmup:]
    occlusions = detection_data.occlusions[warmup:]
    frame_indices = detection_data.frame_indices[warmup:]
    if len(frames) < 2 or float(boxes[0][2]) <= 0:
        return

    floor = anchor_floor(config, trajectory, anchor_index, frames, boxes)
    # Only score AFTER the first occlusion -- pre-occlusion frames are clean on-target with no corruption in play.
    first_occlusion = int(np.argmax(occlusions > 0)) if float(occlusions.max()) > 0 else len(occlusions)
    scored = [frame for frame in range(max(1, first_occlusion), len(frames)) if clean_boxes.get(int(frame_indices[frame]))]
    if not scored:
        return

    anchor_mask = np.stack(box_prompt_masks(oracle.model, frames[0], [boxes[0]]))
    anchor = encode_foregrounds(encoders, frames[:1], anchor_mask, floor, config.crop_size)[0]
    target_masks, distractor_masks = prompt_labels(oracle.model, frames, boxes, occlusions, frame_indices, clean_boxes, scored)

    height, width = frames[0].shape[:2]
    pixel_scale = 1024.0 / max(width, height)
    anchor_center = to_pixel(boxes[0], width, height, pixel_scale)   # the anchor target's centre, @1024 px

    clean_positions, clean_memory = committed_memory(encoders, frames, clean, floor, config.crop_size)
    corrupt_positions, corrupt_memory = committed_memory(encoders, frames, corrupt, floor, config.crop_size)

    # SAM 2's IoU gate: each pass's own predictions, scored by its self-confidence under that bank.
    for pass_name, rollout in (("clean", clean), ("corrupt", corrupt)):
        overlaps = candidate_overlaps(rollout["predicted"], scored, target_masks, distractor_masks)
        for index, frame in enumerate(scored):
            target_iou, distractor_iou = overlaps[index]
            dist = anchor_distance(rollout["predicted"][frame], anchor_center, width, height, pixel_scale)
            yield (("iou", pass_name), target_iou, distractor_iou, float(rollout["iou"][frame]), dist)

    # PE re-ID gate: candidates POOLED from BOTH rollouts (so on-target instances (clean) and distractor-grabs
    # (corrupt) are present at every frame, including the video end), each scored against the clean AND corrupt
    # memory. Same pooled candidate set for both curves -> alpha=1 coincides; the gap at alpha<1 is the memory.
    for rollout in (clean, corrupt):
        overlaps = candidate_overlaps(rollout["predicted"], scored, target_masks, distractor_masks)
        candidate_foreground = encode_foregrounds(encoders, frames[scored], rollout["predicted"][scored], floor, config.crop_size)
        for encoder_name in config.encoders:
            anchor_tokens = anchor[encoder_name]
            clean_tokens = [entry[encoder_name] for entry in clean_memory]
            corrupt_tokens = [entry[encoder_name] for entry in corrupt_memory]
            for index, frame in enumerate(scored):
                candidate = candidate_foreground[index][encoder_name]
                clean_recent = causal_recent(clean_positions, clean_tokens, frame, config.memory_size)
                corrupt_recent = causal_recent(corrupt_positions, corrupt_tokens, frame, config.memory_size)
                target_iou, distractor_iou = overlaps[index]
                dist = anchor_distance(rollout["predicted"][frame], anchor_center, width, height, pixel_scale)
                yield (("pe", encoder_name), target_iou, distractor_iou, chamfer(anchor_tokens, candidate),
                       mean_chamfer(candidate, clean_recent), mean_chamfer(candidate, corrupt_recent), dist)


def columns_of(candidate_samples, n_columns):
    """Stack a list of same-length sample tuples into `n_columns` float arrays."""

    if not candidate_samples:
        return [np.array([], np.float32) for _ in range(n_columns)]
    return [np.array(column, np.float32) for column in zip(*candidate_samples)]


def candidate_labels(target_iou, distractor_iou, config):
    """Class of each candidate: 1 = on target, 0 = distractor grab, -1 = ambiguous (dropped)."""

    labels = np.full(len(target_iou), -1)
    labels[target_iou >= config.label_positive] = 1
    labels[(target_iou < config.label_negative_target) & (distractor_iou >= config.label_negative_distractor)] = 0
    return labels


def mixed(anchor, memory, alpha):
    """alpha-weighted anchor/memory score, falling back to the anchor wherever the memory is empty (NaN)."""

    return alpha * anchor + (1 - alpha) * np.where(np.isnan(memory), anchor, memory)


def separation_auc(labels, scores):
    """Target-vs-distractor ROC-AUC; NaN scores dropped, NaN returned when a class is absent."""

    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    return roc_auc_score(labels, scores)


def encoder_curves(candidate_samples, alphas, config):
    """For one encoder (candidates pooled from both rollouts), the on-target-vs-distractor AUC at each anchor weight
    under the clean vs corrupted memory, plus the kept-candidate count."""

    target_iou, distractor_iou, anchor, clean, corrupt, _distance = columns_of(candidate_samples, 6)
    labels = candidate_labels(target_iou, distractor_iou, config)
    keep = labels >= 0
    clean_auc = [separation_auc(labels[keep], mixed(anchor, clean, alpha)[keep]) for alpha in alphas]
    corrupt_auc = [separation_auc(labels[keep], mixed(anchor, corrupt, alpha)[keep]) for alpha in alphas]
    return clean_auc, corrupt_auc, int(keep.sum())


def iou_estimator_auc(iou_samples, config):
    """On-target-vs-distractor AUC of SAM 2's own predicted-IoU on one pass's predictions, plus the kept count."""

    target_iou, distractor_iou, sam_iou, _distance = columns_of(iou_samples, 4)
    labels = candidate_labels(target_iou, distractor_iou, config)
    keep = labels >= 0
    return separation_auc(labels[keep], sam_iou[keep]), int(keep.sum())


def binned_auc(distances, labels, scores, edges):
    """AUC of `scores` separating labelled candidates, per anchor-distance bin defined by `edges`."""

    keep = labels >= 0
    aucs = []
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        in_bin = (distances >= low) & (distances <= high if index == len(edges) - 2 else distances < high)
        selected = in_bin & keep
        aucs.append(separation_auc(labels[selected], scores[selected]))
    return aucs


def distance_figure(samples, config, n_traj):
    """AUC vs anchor-distance (5 bins) for three gates under corruption: SAM's IoU estimator (corrupt bank), the PE
    anchor-only score (alpha=1), and the PE corrupt-memory score (alpha=0). Tests whether SAM's floor is anchor
    reliance -- i.e. whether its IoU tracks the anchor-only curve as the target moves away from the anchor."""

    encoder = config.encoders[0]
    target_iou, distractor_iou, anchor, clean, corrupt, dist_pe = columns_of(samples.get(("pe", encoder), []), 6)
    iou_target, iou_distractor, sam_iou, dist_iou = columns_of(samples.get(("iou", "corrupt"), []), 4)
    if len(dist_pe) == 0 or len(dist_iou) == 0:
        return

    finite = np.concatenate([dist_pe[np.isfinite(dist_pe)], dist_iou[np.isfinite(dist_iou)]])
    edges = np.quantile(finite, np.linspace(0, 1, 6))
    centers = 0.5 * (edges[:-1] + edges[1:])

    pe_labels = candidate_labels(target_iou, distractor_iou, config)
    iou_labels = candidate_labels(iou_target, iou_distractor, config)
    anchor_only = binned_auc(dist_pe, pe_labels, mixed(anchor, clean, 1.0), edges)       # alpha=1 -> anchor only
    corrupt_memory = binned_auc(dist_pe, pe_labels, mixed(anchor, corrupt, 0.0), edges)  # alpha=0 -> corrupt memory
    iou_auc = binned_auc(dist_iou, iou_labels, sam_iou, edges)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(centers, iou_auc, marker="s", color="#7733aa", label="SAM iou-estimator (corrupt bank)")
    axis.plot(centers, anchor_only, marker="o", color="#2277cc", label="PE anchor only (α=1)")
    axis.plot(centers, corrupt_memory, marker="o", color=CORRUPT_COLOR, label="PE corrupt memory (α=0)")
    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axis.set_xlabel("anchor distance (px @1024)")
    axis.set_ylabel("on-target vs distractor-grab AUC")
    axis.set_title(f"{encoder}  |  p={config.corruption_p:g}  |  AUC vs anchor distance  |  {n_traj} trajectories", fontsize=9)
    axis.set_ylim(0.4, 1.0)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(f"{config.out_dir}/{encoder}_distance.png", dpi=120)
    plt.close(figure)


def plot_curves(samples, config, n_traj):
    """Per encoder, draw the clean-memory vs corrupted-memory AUC-vs-anchor-weight curves with the covariate-shift
    gap shaded, and return the drawn curves."""

    alphas = list(config.alphas)
    clean_iou_auc, _ = iou_estimator_auc(samples.get(("iou", "clean"), []), config)
    corrupt_iou_auc, _ = iou_estimator_auc(samples.get(("iou", "corrupt"), []), config)
    curves = {"iou": {"clean": clean_iou_auc, "corrupted": corrupt_iou_auc}}
    for encoder_name in config.encoders:
        clean_auc, corrupt_auc, n_candidates = encoder_curves(samples.get(("pe", encoder_name), []), alphas, config)
        curves[encoder_name] = {"alphas": alphas, "clean": clean_auc, "corrupted": corrupt_auc}

        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(alphas, clean_auc, marker="o", color=CLEAN_COLOR, label="PE, clean memory")
        axis.plot(alphas, corrupt_auc, marker="o", color=CORRUPT_COLOR, label=f"PE, {config.corruption_p:g}-corrupted memory")
        axis.fill_between(alphas, corrupt_auc, clean_auc, color=CORRUPT_COLOR, alpha=0.12, label="PE covariate shift")
        axis.axhline(clean_iou_auc, color=CLEAN_COLOR, linestyle=":", linewidth=1.2, label=f"SAM iou, clean bank ({clean_iou_auc:.3f})")
        axis.axhline(corrupt_iou_auc, color=CORRUPT_COLOR, linestyle=":", linewidth=1.2, label=f"SAM iou, corrupt bank ({corrupt_iou_auc:.3f})")
        axis.set_xlabel("anchor weight α")
        axis.set_ylabel("on-target vs distractor-grab AUC")
        axis.set_title(f"{encoder_name}  |  p={config.corruption_p:g} corruption  |  {n_traj} trajectories, {n_candidates} candidates", fontsize=9)
        axis.set_ylim(0.4, 1.0)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(f"{config.out_dir}/{encoder_name}_robust.png", dpi=120)
        plt.close(figure)
    return curves


def load_checkpoint(path):
    """Load (samples, done) from the checkpoint, or empties when there is none. samples maps a tagged key ->
    candidate rows: ("pe", encoder) -> (target_iou, distractor_iou, anchor, clean, corrupt, distance); ("iou", pass)
    -> (target_iou, distractor_iou, sam_iou, distance). done is the set of finished trajectory keys."""

    if not Path(path).exists():
        return defaultdict(list), set()
    state = pickle.loads(Path(path).read_bytes())
    return defaultdict(list, state["samples"]), set(state["done"])


def save_checkpoint(path, samples, done):
    """Atomically write the accumulated per-encoder candidate samples and finished trajectory keys."""

    tmp = Path(path).with_suffix(".pkl.tmp")
    tmp.write_bytes(pickle.dumps({"samples": dict(samples), "done": list(done)}))
    tmp.replace(path)


@hydra.main(config_path="../conf", config_name="robust_scoring", version_base=None)
def run(config: DictConfig):
    """Roll SAM 2 out clean and corrupted over the test trajectories, score the clean predictions against both
    memories with the Perception Encoder, and draw the covariate-shift-vs-anchor-weight figures."""

    os.makedirs(config.out_dir, exist_ok=True)
    checkpoint_path = f"{config.out_dir}/scores.pkl"
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    encoders = load_dataset_encoders(list(config.encoders), DEVICE, DTYPE)
    oracle = hydra.utils.instantiate(OmegaConf.load(config.oracle_config).tracker)
    print(f"encoders: {list(encoders)}")

    trajectories = test_trajectories(person_path, config.n_traj)
    samples, done = load_checkpoint(checkpoint_path)
    print(f"scoring {len(trajectories)} trajectories; resuming with {len(done)} done")

    for trajectory in tqdm(trajectories, desc="robust-scoring"):
        key = f"{trajectory[0]}_{trajectory[1]}_{trajectory[2]}"
        if key in done:
            continue
        for encoder_name, *sample in evaluate_trajectory(oracle, encoders, detection_data, trajectory, config):
            samples[encoder_name].append(tuple(sample))
        done.add(key)
        save_checkpoint(checkpoint_path, samples, done)
        if any(samples.values()):
            plot_curves(samples, config, len(done))
            distance_figure(samples, config, len(done))
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    curves = plot_curves(samples, config, len(done))
    distance_figure(samples, config, len(done))
    np.save(f"{config.out_dir}/curves.npy", {"curves": curves, "n_traj": len(done)}, allow_pickle=True)

if __name__ == "__main__":
    run()

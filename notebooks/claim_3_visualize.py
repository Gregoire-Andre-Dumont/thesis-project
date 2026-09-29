"""claim_3 figure: how far can each backbone re-identify the target from its anchor?"""

import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper"))

import style

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "data/claim_3/results.pkl"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "conf/experiments/claim_3.yaml"
OUT_DIR = Path("data/claim_3/paper")


BACKBONE_SLOT = {"pe_spatial": 0, "pe_sam3": 1, "hiera_sam": 2, "hiera_mae": 3, "clip": 4, "owlvit": 5}


BACKBONE_NAME = {"pe_spatial": "PE spatial", "pe_sam3": "PE (SAM 3)", "hiera_sam": "Hiera (SAM 2)", "hiera_mae": "Hiera (MAE)", "clip": "CLIP", "owlvit": "OWL-ViT"}


def auc(samples):
    """Target-vs-distractor AUC over (distance, label, score) samples; NaN when one class is missing."""

    bin_labels = np.array([sample_scores[1] for sample_scores in samples])
    encoder_scores = np.array([sample_scores[2] for sample_scores in samples])
    return roc_auc_score(bin_labels, encoder_scores) if 0 < bin_labels.sum() < len(bin_labels) else np.nan


def binned_auc(samples, bin_edges, min_bin_samples):
    """AUC within each distance bin, NaN where a bin holds too few samples to be worth reporting."""

    distances = np.array([sample_scores[0] for sample_scores in samples])
    score_curve = []
    for lower_bound, upper_bound in zip(bin_edges[:-1], bin_edges[1:]):
        inside_mask = [sample_scores for sample_scores, distance in zip(samples, distances) if lower_bound <= distance < upper_bound]
        score_curve.append(auc(inside_mask) if len(inside_mask) >= min_bin_samples else np.nan)
    return score_curve


def quantile_edges(results, n_bins):
    """Equal-count distance-bin edges from the pooled candidate distances, identical across backbones."""

    samples = next(iter(results.values()), [])
    if len(samples) < n_bins:
        return None
    distances = np.array([sample_scores[0] for sample_scores in samples])
    bin_edges = np.quantile(distances, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9
    return bin_edges


def bin_centres(samples, bin_edges):
    """Each bin's median distance -- equal-count bins are unevenly spaced, so the midpoint would mislead."""

    distances = np.array([sample_scores[0] for sample_scores in samples])
    centres = []
    for lower_bound, upper_bound in zip(bin_edges[:-1], bin_edges[1:]):
        inside_mask = (distances >= lower_bound) & (distances < upper_bound)
        centres.append(np.median(distances[inside_mask]) if inside_mask.any() else (lower_bound + upper_bound) / 2)
    return centres


def draw(results, encoder_group, bin_edges, series_colors, min_bin_samples, trajectories, filename, metric):
    """One group's backbones' AUC against candidate distance from the anchor."""

    curves = {encoder_name: binned_auc(results[encoder_name], bin_edges, min_bin_samples) for encoder_name in encoder_group if encoder_name in results}
    if not curves:
        return

    reference = next(results[encoder_name] for encoder_name in encoder_group if encoder_name in results)
    centres = bin_centres(reference, bin_edges)

    figure, axis = plt.subplots(figsize=(style.COLUMN, 3.1))
    for encoder_name, score_curve in curves.items():
        axis.plot(
            centres,
            score_curve,
            label=BACKBONE_NAME.get(encoder_name, encoder_name.replace("_", " ")),
            zorder=3,
            **style.series_style(BACKBONE_SLOT.get(encoder_name, 0), slots=style.SLOTS_6))
    axis.axhline(0.5, color=style.INK2, linestyle=(0, (1, 2)), linewidth=0.8, zorder=2)

    axis.set_ylim(0.45, 1.02)
    style.gridlines(axis, 0.05)
    style.style_axes(axis, "candidate distance from the anchor (px @1024)", "target-vs-distractor AUC")
    axis.legend(loc="upper right", ncol=2)

    style.save(figure, filename)


style.use_paper_style()
saved_state = pickle.load(open(RESULTS, "rb"))
config = OmegaConf.load(CONFIG)
series_colors = OmegaConf.to_container(config.colors)
encoder_groups = OmegaConf.to_container(config.groups)
trajectories = len(saved_state["done"])

OUT_DIR.mkdir(parents=True, exist_ok=True)
for direction, results in (
    ("bidirectional", saved_state["results_bi"]),
    ("unidirectional", saved_state["results_uni"])):
    bin_edges = quantile_edges(results, int(config.n_bins))
    if bin_edges is None:
        continue
    for group_name, encoder_group in encoder_groups.items():
        draw(
            results,
            encoder_group,
            bin_edges,
            series_colors,
            int(config.min_bin_samples),
            trajectories,
            OUT_DIR / f"fig_{group_name }_{direction }",
            f"{direction } chamfer")

for encoder_name in sorted(saved_state["results_bi"]):
    print(
        f"{encoder_name:14}{auc(saved_state['results_bi'][encoder_name]):>16.3f}{auc(saved_state['results_uni'][encoder_name]):>16.3f}"
        f"{len(saved_state['results_bi'][encoder_name]):>10}"
    )

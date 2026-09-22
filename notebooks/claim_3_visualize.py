"""claim_3 figure: how far can each backbone re-identify the target from its anchor?

Every candidate on a frame -- the target plus its nearest distractors -- is scored by chamfer similarity
between its foreground tokens and the anchor's. A candidate is labelled 1 if it IS the target, 0 otherwise,
so the AUC is the probability a backbone ranks the target above a distractor. Chance is 0.50.

Binned by how far the candidate sits from the anchor's last known position, in equal-count bins so every
point rests on the same number of samples. The distance axis is the question: a backbone that only works
when the target has barely moved is no use after a long occlusion, which is precisely when the memory bank
needs re-identification rather than proximity.

Two chamfer directions are drawn. Unidirectional (candidate -> anchor) asks whether everything in the
candidate resembles something in the anchor, and so tolerates a candidate that shows only part of the
person. Bidirectional also penalises anchor content missing from the candidate, which is stricter under
partial occlusion. They can disagree, and the pair is the honest report.

    python notebooks/claim_3_visualize.py [results.pkl] [config.yaml]
"""
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

# Slot per BACKBONE, not per position in a figure: pe_spatial and pe_sam3 appear in both groups, and a
# backbone that changed colour between the two figures would make the pair unreadable together.
BACKBONE_SLOT = {"pe_spatial": 0, "pe_sam3": 1, "hiera_sam": 2, "hiera_mae": 3, "clip": 4, "owlvit": 5}

# Display names, not the config keys. Two reasons: a paper names a backbone the way its own authors do, and
# cmr10 carries no underscore glyph, so "hiera_sam" renders with a raised dot where the underscore should be.
BACKBONE_NAME = {
    "pe_spatial": "PE spatial",
    "pe_sam3": "PE (SAM 3)",
    "hiera_sam": "Hiera (SAM 2)",
    "hiera_mae": "Hiera (MAE)",
    "clip": "CLIP",
    "owlvit": "OWL-ViT",
}


def auc(samples):
    """Target-vs-distractor AUC over (distance, label, score) samples; NaN when one class is missing."""

    labels = np.array([sample[1] for sample in samples])
    scores = np.array([sample[2] for sample in samples])
    return roc_auc_score(labels, scores) if 0 < labels.sum() < len(labels) else np.nan


def binned_auc(samples, edges, min_bin_samples):
    """AUC within each distance bin, NaN where a bin holds too few samples to be worth reporting."""

    distances = np.array([sample[0] for sample in samples])
    curve = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = [sample for sample, distance in zip(samples, distances) if low <= distance < high]
        curve.append(auc(inside) if len(inside) >= min_bin_samples else np.nan)
    return curve


def quantile_edges(results, n_bins):
    """Equal-count distance-bin edges from the pooled candidate distances, identical across backbones."""

    samples = next(iter(results.values()), [])
    if len(samples) < n_bins:
        return None
    distances = np.array([sample[0] for sample in samples])
    edges = np.quantile(distances, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9                                    # make the top edge inclusive
    return edges


def bin_centres(samples, edges):
    """Each bin's median distance -- equal-count bins are unevenly spaced, so the midpoint would mislead."""

    distances = np.array([sample[0] for sample in samples])
    centres = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (distances >= low) & (distances < high)
        centres.append(np.median(distances[inside]) if inside.any() else (low + high) / 2)
    return centres


def draw(results, group, edges, colors, min_bin_samples, trajectories, filename, metric):
    """One group's backbones' AUC against candidate distance from the anchor."""

    curves = {name: binned_auc(results[name], edges, min_bin_samples) for name in group if name in results}
    if not curves:
        print(f"skipped {filename}  (no backbones from this group in the results)")
        return

    reference = next(results[name] for name in group if name in results)
    centres = bin_centres(reference, edges)

    figure, axis = plt.subplots(figsize=(style.COLUMN, 3.1))
    for name, curve in curves.items():
        axis.plot(centres, curve, label=BACKBONE_NAME.get(name, name.replace("_", " ")), zorder=3,
                  **style.series_style(BACKBONE_SLOT.get(name, 0), slots=style.SLOTS_6))
    axis.axhline(0.5, color=style.INK2, linestyle=(0, (1, 2)), linewidth=0.8, zorder=2)

    axis.set_ylim(0.45, 1.02)
    style.gridlines(axis, 0.05)
    style.style_axes(axis, "candidate distance from the anchor (px @1024)", "target-vs-distractor AUC")
    axis.legend(loc="upper right", ncol=2)

    style.save(figure, filename)
    print(f"   caption: Re-identification AUC by {metric}, against how far the candidate sits from the "
          f"anchor. n={trajectories} trajectories, {len(reference)} candidate scores; equal-count bins "
          f"holding >= {min_bin_samples} samples each. The dotted line is chance (0.50).")


style.use_paper_style()
state = pickle.load(open(RESULTS, "rb"))
config = OmegaConf.load(CONFIG)
colors = OmegaConf.to_container(config.colors)
groups = OmegaConf.to_container(config.groups)
trajectories = len(state["done"])

OUT_DIR.mkdir(parents=True, exist_ok=True)
for direction, results in (("bidirectional", state["results_bi"]), ("unidirectional", state["results_uni"])):
    edges = quantile_edges(results, int(config.n_bins))
    if edges is None:
        continue
    for group_name, group in groups.items():
        draw(results, group, edges, colors, int(config.min_bin_samples), trajectories,
             OUT_DIR / f"fig_{group_name}_{direction}", f"{direction} chamfer")

print(f"\nn={trajectories} trajectories   pooled target-vs-distractor AUC")
print(f"{'backbone':14}{'bidirectional':>16}{'unidirectional':>16}{'samples':>10}")
for name in sorted(state["results_bi"]):
    print(f"{name:14}{auc(state['results_bi'][name]):>16.3f}{auc(state['results_uni'][name]):>16.3f}"
          f"{len(state['results_bi'][name]):>10}")

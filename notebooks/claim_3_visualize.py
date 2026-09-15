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

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "data/claim_3/results.pkl"
CONFIG = sys.argv[2] if len(sys.argv) > 2 else "conf/experiments/claim_3.yaml"
OUT_DIR = Path("data/claim_3")
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


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

    figure, axis = plt.subplots(figsize=(8.6, 5.4), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    for name, curve in curves.items():
        axis.plot(centres, curve, marker="o", markersize=8, linewidth=2, color=colors.get(name),
                  markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)
    axis.axhline(0.5, color=INK2, linestyle=":", linewidth=1, label="chance (0.50)")

    axis.set_xlabel("candidate distance from the anchor  (px @1024, equal-count bins)", fontsize=10, color=INK2)
    axis.set_ylabel("target-vs-distractor AUC", fontsize=10, color=INK2)
    axis.set_title(f"Re-identification by {metric}", fontsize=12, color=INK, pad=12, loc="left")
    axis.set_ylim(0.45, 1.02)
    axis.grid(axis="y", color=INK2, alpha=0.13, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK2)
        axis.spines[side].set_alpha(0.35)
    axis.tick_params(colors=INK2, labelsize=9)
    axis.legend(frameon=False, fontsize=10, loc="lower left")

    figure.text(0.008, 0.955, f"n={trajectories} trajectories  ·  {len(reference)} candidate scores  ·  "
                              f"bins hold >= {min_bin_samples} samples", fontsize=9, color=INK2, ha="left")
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(filename, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"saved {filename}")


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
             OUT_DIR / f"fig_{group_name}_{direction}.png", f"{direction} chamfer")

print(f"\nn={trajectories} trajectories   pooled target-vs-distractor AUC")
print(f"{'backbone':14}{'bidirectional':>16}{'unidirectional':>16}{'samples':>10}")
for name in sorted(state["results_bi"]):
    print(f"{name:14}{auc(state['results_bi'][name]):>16.3f}{auc(state['results_uni'][name]):>16.3f}"
          f"{len(state['results_bi'][name]):>10}")

"""Post-hoc F1 sweep from the cached covariate_shift scores -- reads scores.pkl, no re-run, no GPU, no torch.

For each swept point (anchor weight, or memory size) it fits the F1-optimal threshold on the ORACLE's candidate
scores and reports F1 of the oracle and of the score-gated policy at that same threshold, so the covariate shift
is measured at the operating point the memory policy would actually commit on. Mirrors the AUC sweep figure but
writes <encoder>_f1.png alongside it. Safe to run while the main experiment is still going (atomic pkl writes)."""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf


def mixed_score(record, encoder, policy, alpha, memory_size):
    """The candidate's alpha-weighted score: anchor chamfer blended with the mean chamfer to the most recent
    committed entries, falling back to the anchor while the memory is still empty."""

    anchor = record["anchor"][encoder]
    recent = record["recent"][policy][encoder][-memory_size:]
    recent = recent[~np.isnan(recent)]
    memory = recent.mean() if len(recent) else anchor
    return alpha * anchor + (1 - alpha) * memory


def oracle_labelled(records, encoder, alpha, memory_size):
    """(labels, scores) for the oracle's post-hoc sweep at this cell, NaN scores dropped."""

    labels = np.array([record["label"] for record in records])
    scores = np.array([mixed_score(record, encoder, "oracle", alpha, memory_size) for record in records])
    valid = ~np.isnan(scores)
    return labels[valid].astype(int), scores[valid]


def best_f1_threshold(scores, labels):
    """Threshold on `scores` (commit iff score >= threshold) maximizing F1 against boolean `labels`, one pass."""

    positive = int(labels.sum())
    if positive == 0 or positive == len(labels):
        return np.nan
    order = np.argsort(-scores)
    true_positives = np.cumsum(labels[order].astype(int))
    predicted_positives = np.arange(1, len(scores) + 1)
    f1 = 2 * true_positives / (predicted_positives + positive)
    return float(scores[order][int(np.argmax(f1))])


def target_f1(labels, scores, threshold):
    """F1 of the target class (label 1), calling a candidate 'target' iff its score >= threshold."""

    predicted_target = scores >= threshold
    is_target = labels == 1
    true_positives = int((predicted_target & is_target).sum())
    denominator = int(predicted_target.sum()) + int(is_target.sum())
    return 2 * true_positives / denominator if denominator else np.nan


def sweep_f1(records, policy_scores, encoder, memory_alpha_points):
    """Oracle and policy F1 at each swept point, both at the F1-optimal threshold fitted on the oracle scores."""

    oracle_f1 = []
    policy_f1 = []
    for memory_size, alpha in memory_alpha_points:
        oracle_labels, oracle_scores = oracle_labelled(records, encoder, alpha, memory_size)
        policy_labels, policy_scores_cell = policy_scores.get((encoder, memory_size, alpha), (np.array([]), np.array([])))

        threshold = best_f1_threshold(oracle_scores, oracle_labels)
        if np.isnan(threshold):
            oracle_f1.append(np.nan)
            policy_f1.append(np.nan)
            continue

        oracle_f1.append(target_f1(oracle_labels, oracle_scores, threshold))
        policy_f1.append(target_f1(np.asarray(policy_labels), np.asarray(policy_scores_cell), threshold))
    return oracle_f1, policy_f1


def draw_curve(axis, positions, tick_labels, oracle_f1, policy_f1, title, xlabel):
    """One F1-vs-swept-variable panel: oracle and policy curves with the covariate-shift gap shaded between."""

    axis.plot(positions, oracle_f1, marker="o", color="tab:green", label="oracle (true-IoU memory)")
    axis.plot(positions, policy_f1, marker="o", color="tab:red", label="policy (score-gated memory)")
    axis.fill_between(positions, policy_f1, oracle_f1, color="tab:red", alpha=0.12, label="covariate shift")

    axis.set_xticks(positions, [str(label) for label in tick_labels])
    axis.set_xlabel(xlabel)
    axis.set_ylabel("target-vs-distractor F1")
    axis.set_ylim(0.5, 1.0)
    axis.set_title(title, fontsize=10)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)


def main():
    config = OmegaConf.load("conf/covariate_shift.yaml")
    state = pickle.loads(Path(f"{config.out_dir}/scores.pkl").read_bytes())
    records, policy_scores, n_traj = state["oracle_records"], state["policy_scores"], len(state["done"])

    for encoder in config.encoders:
        figure, (alpha_axis, memory_axis) = plt.subplots(1, 2, figsize=(12, 5))

        alpha_points = [(config.alpha_sweep_memory, alpha) for alpha in config.alphas]
        oracle_alpha, policy_alpha = sweep_f1(records, policy_scores, encoder, alpha_points)
        draw_curve(alpha_axis, list(config.alphas), list(config.alphas), oracle_alpha, policy_alpha,
                   f"F1 vs anchor weight  (memory {config.alpha_sweep_memory})", "anchor weight α")

        memory_points = [(memory_size, config.memory_sweep_alpha) for memory_size in config.memory_sizes]
        oracle_memory, policy_memory = sweep_f1(records, policy_scores, encoder, memory_points)
        draw_curve(memory_axis, range(len(config.memory_sizes)), list(config.memory_sizes), oracle_memory, policy_memory,
                   f"F1 vs memory size  (α {config.memory_sweep_alpha})", "memory size")

        figure.suptitle(f"{encoder}  |  {n_traj} trajectories  |  F1 (post-hoc)", fontsize=11)
        figure.tight_layout()
        figure.savefig(f"{config.out_dir}/{encoder}_f1.png", dpi=120)
        plt.close(figure)
        print("wrote", f"{config.out_dir}/{encoder}_f1.png", "| n_traj =", n_traj)


if __name__ == "__main__":
    main()

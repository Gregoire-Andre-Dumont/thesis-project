"""Post-hoc: AUC vs anchor weight, split by anchor-candidate distance into two regimes -- near (< split px) and
far (>= split px) -- each written to its own PNG. Reads scores.pkl, no re-run, no GPU, no torch. Safe to run
while the main experiment is still going (atomic pkl writes)."""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

DISTANCE_SPLIT = 300.0        # px; near = distance < this, far = distance >= this


def mixed_score(record, encoder, policy, alpha, memory_size):
    """The candidate's alpha-weighted score: anchor chamfer blended with the mean chamfer to the most recent
    committed entries, falling back to the anchor while the memory is still empty."""

    anchor = record["anchor"][encoder]
    recent = record["recent"][policy][encoder][-memory_size:]
    recent = recent[~np.isnan(recent)]
    memory = recent.mean() if len(recent) else anchor
    return alpha * anchor + (1 - alpha) * memory


def target_auc(labels, scores):
    """Target-vs-distractor ROC-AUC; NaN when a class is absent or there are no samples."""

    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    return roc_auc_score(labels, scores)


def oracle_auc(records, encoder, alpha, memory_size, keep):
    """Oracle target-vs-distractor AUC over the records whose distance passes `keep`, scores swept post-hoc."""

    labels = []
    scores = []
    for record in records:
        if not keep(record["distance"]):
            continue
        score = mixed_score(record, encoder, "oracle", alpha, memory_size)
        if not np.isnan(score):
            labels.append(record["label"])
            scores.append(score)
    return target_auc(np.array(labels), np.array(scores))


def policy_auc(policy_scores, encoder, alpha, memory_size, keep):
    """Policy target-vs-distractor AUC over the cell's candidates whose distance passes `keep`."""

    labels, scores, distances = policy_scores.get((encoder, memory_size, alpha), (np.array([]), np.array([]), np.array([])))
    in_regime = keep(distances)
    return target_auc(labels[in_regime], scores[in_regime])


def draw_regime(config, records, policy_scores, encoder, keep, title, path, n_traj):
    """One AUC-vs-anchor-weight panel for a distance regime: oracle and policy curves, covariate shift shaded."""

    oracle_aucs = [oracle_auc(records, encoder, alpha, config.memory_size, keep) for alpha in config.alphas]
    policy_aucs = [policy_auc(policy_scores, encoder, alpha, config.memory_size, keep) for alpha in config.alphas]

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(list(config.alphas), oracle_aucs, marker="o", color="tab:green", label="oracle (true-IoU memory)")
    axis.plot(list(config.alphas), policy_aucs, marker="o", color="tab:red", label="policy (score-gated memory)")
    axis.fill_between(list(config.alphas), policy_aucs, oracle_aucs, color="tab:red", alpha=0.12, label="covariate shift")

    axis.set_xlabel("anchor weight α")
    axis.set_ylabel("target-vs-distractor AUC")
    axis.set_ylim(0.5, 1.0)
    axis.set_title(f"{encoder}  |  {n_traj} trajectories  |  {title}", fontsize=10)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main():
    config = OmegaConf.load("conf/covariate_shift.yaml")
    state = pickle.loads(Path(f"{config.out_dir}/scores.pkl").read_bytes())
    records, policy_scores, n_traj = state["oracle_records"], state["policy_scores"], len(state["done"])

    for encoder in config.encoders:
        draw_regime(config, records, policy_scores, encoder, lambda distance: distance < DISTANCE_SPLIT,
                    f"anchor-candidate distance < {int(DISTANCE_SPLIT)} px", f"{config.out_dir}/{encoder}_near.png", n_traj)
        draw_regime(config, records, policy_scores, encoder, lambda distance: distance >= DISTANCE_SPLIT,
                    f"anchor-candidate distance >= {int(DISTANCE_SPLIT)} px", f"{config.out_dir}/{encoder}_far.png", n_traj)
        print("wrote", f"{config.out_dir}/{encoder}_near.png", "and", f"{config.out_dir}/{encoder}_far.png", "| n_traj =", n_traj)


if __name__ == "__main__":
    main()

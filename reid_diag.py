"""Diagnostic for pe_reid_longrange: per-backbone target-vs-distractor AUC and chamfer gap on a
handful of test trajectories. Confirms whether each backbone actually separates the target from its
distractors (and sanity-checks the anchor-floor crop), without waiting for the full run.

  python reid_diag.py             # 5 trajectories
  python reid_diag.py +n_diag=10  # more
"""
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

from pe_reid_longrange import load_backbones, evaluate_trajectory, test_trajectories, SAM_BASELINE_CONFIG


@hydra.main(config_path="conf", config_name="create_anchor_dataset", version_base=None)
def main(config: DictConfig):
    n_diag = int(config.get("n_diag", 5))
    stride = int(config.get("stride", 3))
    detection_data = hydra.utils.instantiate(config.detection_data)
    person_path = hydra.utils.instantiate(config.person_path)
    sam = hydra.utils.instantiate(OmegaConf.load(SAM_BASELINE_CONFIG).tracker).model
    backbones = load_backbones()
    visible_dir = config.detection_data.visible_directory
    print(f"backbones: {list(backbones)}")

    samples = {name: [] for name in backbones}   # backbone -> [(label, score)]
    for i, trajectory in enumerate(test_trajectories(person_path)[:n_diag], start=1):
        for name, dt, label, score in evaluate_trajectory(sam, backbones, detection_data, trajectory, visible_dir, stride):
            samples[name].append((label, score))
        print(f"traj {i}/{n_diag}", flush=True)

    print(f"\n{'backbone':11s} {'AUC':>6s} {'tgt_sim':>8s} {'dist_sim':>9s} {'gap':>7s}   n")
    for name, rows in samples.items():
        if not rows:
            print(f"{name:11s}  no samples")
            continue
        label = np.array([r[0] for r in rows])
        score = np.array([r[1] for r in rows])
        tgt, dist = score[label == 1].mean(), score[label == 0].mean()
        auc = roc_auc_score(label, score) if 0 < label.sum() < len(label) else float("nan")
        print(f"{name:11s} {auc:6.3f} {tgt:8.3f} {dist:9.3f} {tgt - dist:7.3f}   {len(label)}")


if __name__ == "__main__":
    main()

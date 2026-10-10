"""claim_2 figure: coverage@0.5 on the MEMORY WINDOW as the memory bank is poisoned with clean distractors.

Coverage is the primary metric here, scored on the memory window -- every occluded frame plus only the
visible frames AFTER the last occlusion. That span is the one corruption actually damages: a poisoned bank
shows up when the target reappears, not before it was ever lost.

Coverage asks one question per visible frame -- did the emitted mask reach IoU >= 0.5 -- and needs no
confidence signal, so the three arms are compared on their masks alone. That matters for the oracles, whose
"confidence" would be the true IoU and would make a reporting comparison meaningless.

Three arms, one line each. The oracles read ground truth, so their margin over SAM 2 is the headroom a
deployable commit gate could recover, and the way that margin WIDENS with corruption is the claim: the
dirtier the bank, the more a good commit decision is worth.

    python notebooks/claim_2_visualize.py [results.pkl] [out_dir]
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paper"))

import style

ARMS = (("sam", "SAM 2 baseline"), ("memory", "memory oracle"), ("mask", "mask oracle"))
LOCALISATION = 0.5


def window_coverage(overlap, occluded, has_box):
    """Fraction of VISIBLE frames in the memory window whose mask reached IoU >= 0.5.

    The window clears `has_box` up to and including the last occluded frame, so what remains is the
    occluded frames plus the visible tail after the target last disappeared. A clip with no visible frame
    left contributes NaN and is dropped by the nan-aware mean."""

    overlap = np.nan_to_num(np.asarray(overlap, dtype=float), nan=0.0)
    occluded = np.asarray(occluded, dtype=bool)
    kept = np.asarray(has_box, dtype=bool).copy()
    if occluded.any():
        kept[: int(np.where(occluded)[0][-1]) + 1] = False

    visible = kept & ~occluded
    length = min(len(overlap), len(visible))
    overlap, visible = overlap[:length], visible[:length]
    return float((overlap[visible] >= LOCALISATION).mean()) if visible.any() else np.nan


def arm_curve(scored_clips, arm_name, probabilities):
    """One arm's mean window coverage at each corruption probability."""

    return np.array([np.nanmean([window_coverage(record[(arm_name, probability)],
                                                 record["occluded"], record["has_box"])
                                 for record in scored_clips if (arm_name, probability) in record])
                     for probability in probabilities])


def main():
    results_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/claim_2/results.pkl")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/sam-2/claim_2")

    results = pickle.load(open(results_path, "rb"))
    scored_clips, probabilities = results["clips"], results["corruption_ps"]

    style.use_paper_style()
    figure_handle, axis_handle = plt.subplots(figsize=(style.COLUMN, 3.1))

    highest, rows = 0.0, {}
    for position, (arm_name, label) in enumerate(ARMS):
        scores = arm_curve(scored_clips, arm_name, probabilities)
        rows[label] = scores
        axis_handle.plot(probabilities, scores, label=label, zorder=3, **style.series_style(position))
        highest = max(highest, float(np.nanmax(scores)))
        print(f"{label:16s} " + "  ".join(f"p{p:g}={v:.4f}" for p, v in zip(probabilities, scores)))

    axis_handle.set_xticks(probabilities)
    axis_handle.set_xticklabels([f"{probability:g}" for probability in probabilities])
    span = probabilities[-1] - probabilities[0]
    axis_handle.set_xlim(probabilities[0] - 0.04 * span, probabilities[-1] + 0.04 * span)
    style.gridlines(axis_handle, 0.05)
    style.headroom(axis_handle, highest)
    style.style_axes(axis_handle, "corruption probability (per frame)", "coverage@0.5, memory window")
    axis_handle.legend(loc="best")

    out_dir.mkdir(parents=True, exist_ok=True)
    style.save(figure_handle, out_dir / "fig_corruption")
    figure_handle.savefig(out_dir / "fig_corruption.png", dpi=200, bbox_inches="tight")

    with open(out_dir / "table_coverage_by_corruption_window.csv", "w") as handle:
        handle.write("arm," + ",".join(f"p{p:g}" for p in probabilities) + "\n")
        for label, scores in rows.items():
            handle.write(label + "," + ",".join(f"{value:.4f}" for value in scores) + "\n")
        handle.write("n clips," + ",".join(str(len(scored_clips)) for _ in probabilities) + "\n")

    print(f"\n{len(scored_clips)} clips -> {out_dir / 'fig_corruption.png'}")


if __name__ == "__main__":
    main()

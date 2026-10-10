"""claim_6: picking the arm per clip from the gate's own confidence, against picking one arm for the run.

claim_5 trains one gate per corruption prior and reports each as a separate arm, so a deployment has to
commit to one of them -- in practice the one with the best mean coverage, which is a choice made WITH the
labels and on the very clips it is then scored on. claim_6 asks whether the gate can make that choice
itself, per clip, from a quantity it already produces.

THE SIGNAL. For each arm, the mean p(commit) over the frames from that arm's FIRST DETECTED OCCLUSION --
the first frame its own gate declines to commit. The arm with the highest mean is selected. Nothing here
reads an annotation: not the occlusion flags, not the boxes, not the coverage. The detected onset is used
rather than the labelled one precisely so the rule survives deployment, where no occlusion flag exists.

    Note on the stored records: `scored_span` already starts at the LABELLED first occlusion, so in these
    pkls the detected onset lands within a frame or two of index 0 and the window is nearly the whole
    array. The distinction still matters for a system reading frames live, and the onset is reported below
    so the reader can see how much it actually trims.

WHAT IT IS COMPARED AGAINST. Three references, in increasing order of what they are allowed to know:
  * each individual arm, so the question "is selection better than this prior" is answerable per prior;
  * the BEST arm by mean coverage, which is the published choice and already uses the labels;
  * the per-clip ORACLE, which picks the best arm for every clip and bounds what any selector can reach.
The share of the oracle gap that selection recovers is the honest summary -- beating a fixed arm by a
hair is not interesting if perfect selection would have bought ten times that.

HOW DIFFERENCES ARE JUDGED. By magnitude, not by how often one side wins: a clip won by 0.001 and a clip
won by 0.4 are not the same evidence, and a sign test cannot tell them apart. Reported per comparison are
the mean paired difference with a bootstrap interval and the Wilcoxon signed-rank p, which ranks
differences by size. On the memory window many clips tie outright -- the window can be a handful of
frames -- so `MIN_WINDOW` optionally restricts the window tests to clips long enough to distinguish
anything.

    python notebooks/claim_6.py [arms_dir] [out_dir]
    MIN_WINDOW=5 N_BINS=3 python notebooks/claim_6.py data/sam-2/claim_5
"""

import os
import pickle
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOCALISATION = 0.5
N_BINS = int(os.environ.get("N_BINS", "3"))
MIN_WINDOW = int(os.environ.get("MIN_WINDOW", "1"))
BOOTSTRAP = 2000
PERMUTATIONS = 20000


def keyed(results_path):
    clips = pickle.load(open(results_path, "rb"))["clips"]
    return {(record["video"], record["person"]): record for record in clips}


def load_arms(arms_dir):
    """Every arm under `arms_dir`, whichever layout it was written in.

    claim_5 writes `<arm>/results.pkl` when each arm owns a directory and `<arm>_results.pkl` when they
    sit side by side; both appear in the tree, so both are accepted rather than making the caller care."""

    loaded = {path.parent.name: keyed(path) for path in sorted(arms_dir.glob("*/results.pkl"))}
    for path in sorted(arms_dir.glob("*_results.pkl")):
        loaded.setdefault(path.name[: -len("_results.pkl")], keyed(path))
    return loaded


def window_coverage(record, window):
    """Fraction of VISIBLE frames reaching IoU >= 0.5, over the full span or the memory window.

    Identical to claim_5_visualize's, so the numbers here sit beside that table without conversion."""

    overlap = np.nan_to_num(np.asarray(record["samara"], dtype=float), nan=0.0)
    occluded = np.asarray(record["occluded"], dtype=bool)
    kept = np.asarray(record["has_box"], dtype=bool).copy()
    if window and occluded.any():
        kept[: int(np.where(occluded)[0][-1]) + 1] = False

    visible = kept & ~occluded
    length = min(len(overlap), len(visible))
    overlap, visible = overlap[:length], visible[:length]
    return float((overlap[visible] >= LOCALISATION).mean()) if visible.any() else np.nan


def window_length(record):
    """How many VISIBLE frames the memory window holds -- the resolution the window metric has."""

    occluded = np.asarray(record["occluded"], dtype=bool)
    kept = np.asarray(record["has_box"], dtype=bool).copy()
    if occluded.any():
        kept[: int(np.where(occluded)[0][-1]) + 1] = False
    return int((kept & ~occluded).sum())


def commit_probability(record):
    """The gate's p(commit) per frame.

    `commit_probabilities` is stored (n_frames, 3) with exactly one column filled -- the controller scores
    only the proposal SAM's IoU token already chose -- so the row maximum IS that one value, and NaN where
    the gate did not run."""

    probability = np.asarray(record["commit_probability"], dtype=float)
    return np.nanmax(probability, axis=1) if probability.ndim == 2 else probability


def detected_onset(record):
    """First frame the arm's OWN gate declined to commit; 0 when it never declined.

    This is the deployable stand-in for "the target went missing": it needs no annotation, only the
    tracker's own decision."""

    probability = commit_probability(record)
    committed = np.asarray(record["samara_commit"], dtype=bool)[: len(probability)]
    refused = np.where(~committed)[0]
    return int(refused[0]) if len(refused) else 0


def selection_signal(record):
    """Mean p(commit) from the detected onset onward -- the quantity arms are ranked by."""

    probability = commit_probability(record)
    onset = detected_onset(record)
    tail = probability[onset:]
    if not len(tail) or not np.isfinite(tail).any():
        tail = probability
    return float(np.nanmean(tail)) if np.isfinite(tail).any() else np.nan


def select(arm_names, signals, fallback_index):
    """Index of the arm with the highest signal; the fallback when no arm produced one."""

    values = np.asarray([signals[name] for name in arm_names], dtype=float)
    return fallback_index if not np.isfinite(values).any() else int(np.nanargmax(values))


def paired_statistics(differences):
    """Mean difference with a bootstrap interval, and Wilcoxon signed-rank by permutation.

    Both are magnitude-aware: the bootstrap resamples the differences themselves, and Wilcoxon ranks them
    by absolute size. A sign test is deliberately not reported -- it would count a 0.001 win and a 0.4 win
    as the same evidence."""

    differences = differences[np.isfinite(differences)]
    count = len(differences)
    if count < 3:
        return None

    generator = np.random.default_rng(0)
    resampled = np.array([differences[pick].mean()
                          for pick in (generator.integers(0, count, count) for _ in range(BOOTSTRAP))])
    low, high = np.percentile(resampled, [2.5, 97.5])

    non_zero = differences[differences != 0]
    if not len(non_zero):
        return differences.mean(), low, high, 1.0, count, 0

    ranks = np.argsort(np.argsort(np.abs(non_zero))) + 1.0
    statistic = float(np.sum(ranks[non_zero > 0]) - np.sum(ranks[non_zero < 0]))
    null = np.array([float(np.sum(ranks[(signs := generator.choice([-1, 1], len(non_zero))) > 0])
                           - np.sum(ranks[signs < 0])) for _ in range(PERMUTATIONS)])
    probability = float(np.mean(np.abs(null) >= abs(statistic)))
    return differences.mean(), low, high, probability, count, len(non_zero)


def occlusion_bins(counts):
    """`N_BINS` QUANTILE bins on the number of occluded frames, plus a pooled column."""

    edges = np.quantile(counts, np.linspace(0, 1, N_BINS + 1))
    bins, labels = [], []
    for index in range(N_BINS):
        low, high = edges[index], (np.inf if index == N_BINS - 1 else edges[index + 1])
        bins.append((low, high))
        labels.append(f"{int(low)}-{int(counts.max()) if np.isinf(high) else int(high)}")
    return bins + [(0.0, np.inf)], labels + ["all"]


def report(span_name, coverage_per_arm, selected, oracle, best_name, arm_names,
           counts, window, window_lengths, handle=None):
    """One span's table: every arm, the selection, the best arm and the oracle, then the paired tests."""

    bins, labels = occlusion_bins(counts)
    rows = {f"SAMARA {name}": coverage_per_arm[name] for name in arm_names}
    rows["SELECTION"] = selected
    rows["per-clip oracle"] = oracle

    header = f"{'arm':20s}" + "".join(f"{label:>12s}" for label in labels)
    print(f"\n=== {span_name} ===  coverage@{LOCALISATION}")
    print(header)
    print("-" * len(header))
    for label, values in rows.items():
        cells = [np.nanmean(values[(counts >= low) & (counts < high)]) for low, high in bins]
        print(f"{label:20s}" + "".join(f"{value:12.4f}" for value in cells))
        if handle:
            handle.write(label + "," + ",".join(f"{value:.4f}" for value in cells) + "\n")

    finite = np.isfinite(selected)
    averaged = [int(((counts >= low) & (counts < high) & finite).sum()) for low, high in bins]
    print("-" * len(header))
    print(f"{'n averaged':20s}" + "".join(f"{count:12d}" for count in averaged))
    if handle:
        handle.write("n averaged," + ",".join(str(count) for count in averaged) + "\n")

    # The window can be a handful of frames, which makes coverage@0.5 coarse enough that two arms land on
    # the same value and the comparison carries no information. Restricting to longer windows is reported
    # rather than applied silently, so the cost in clips is visible.
    usable = np.ones(len(selected), dtype=bool)
    if window and MIN_WINDOW > 1:
        usable = window_lengths >= MIN_WINDOW
        print(f"\nwindow length: median {np.median(window_lengths):.0f}  "
              f"clips with >= {MIN_WINDOW} frames: {int(usable.sum())} of {len(usable)}")

    print(f"\nselection minus each reference, mean [95% CI] Wilcoxon p   "
          f"(positive = selection better)")
    references = [(f"SAMARA {name}", coverage_per_arm[name]) for name in arm_names]
    references.append((f"best arm ({best_name}), chosen WITH labels", coverage_per_arm[best_name]))
    references.append(("per-clip oracle", oracle))
    for label, values in references:
        statistics = paired_statistics((selected - values)[usable])
        if statistics is None:
            print(f"  {label:42s} too few clips")
            continue
        mean, low, high, probability, count, differing = statistics
        stars = "**" if probability < 0.01 else ("*" if probability < 0.05 else "")
        print(f"  {label:42s} {mean:+.4f} [{low:+.4f}, {high:+.4f}]  p={probability:.4f}{stars}"
              f"   n={count} ({differing} differ)")

    gap = np.nanmean(oracle[usable]) - np.nanmean(coverage_per_arm[best_name][usable])
    gained = np.nanmean(selected[usable]) - np.nanmean(coverage_per_arm[best_name][usable])
    if abs(gap) > 1e-9:
        print(f"\n  recovers {gained / gap * 100:.1f}% of the oracle gap "
              f"({gained:+.4f} of {gap:+.4f} available)")


def main():
    arms_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sam-2/claim_5")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else arms_dir)

    loaded = load_arms(arms_dir)
    if len(loaded) < 2:
        raise SystemExit(f"claim_6 needs at least two arms under {arms_dir}, found {len(loaded)}")
    arm_names = sorted(loaded)

    clip_keys = sorted(set.intersection(*(set(records) for records in loaded.values())))
    if not clip_keys:
        raise SystemExit("the arms share no clips")
    reference = loaded[arm_names[0]]
    counts = np.array([reference[key]["occ_count"] for key in clip_keys], dtype=float)
    window_lengths = np.array([window_length(reference[key]) for key in clip_keys])

    onsets = np.array([detected_onset(reference[key]) for key in clip_keys], dtype=float)
    print(f"{len(clip_keys)} clips shared by {len(arm_names)} arms: {', '.join(arm_names)}")
    print(f"detected onset: median {np.median(onsets):.0f} frames into the stored span "
          f"(the span already begins at the labelled first occlusion)")

    signals = {name: {key: selection_signal(loaded[name][key]) for key in clip_keys}
               for name in arm_names}

    out_dir.mkdir(parents=True, exist_ok=True)
    for window, span_name in ((False, "FULL SPAN"), (True, "MEMORY WINDOW")):
        coverage_per_arm = {
            name: np.array([window_coverage(loaded[name][key], window) for key in clip_keys], dtype=float)
            for name in arm_names}
        best_name = max(arm_names, key=lambda name: np.nanmean(coverage_per_arm[name]))
        fallback = arm_names.index(best_name)

        selected, oracle, chosen = [], [], []
        for index, key in enumerate(clip_keys):
            pick = select(arm_names, {name: signals[name][key] for name in arm_names}, fallback)
            chosen.append(arm_names[pick])
            selected.append(coverage_per_arm[arm_names[pick]][index])
            oracle.append(max(coverage_per_arm[name][index] for name in arm_names))
        selected, oracle = np.array(selected, dtype=float), np.array(oracle, dtype=float)

        name = "table_claim6" + ("_window" if window else "") + ".csv"
        with open(out_dir / name, "w") as handle:
            handle.write("arm," + ",".join(occlusion_bins(counts)[1]) + "\n")
            report(span_name, coverage_per_arm, selected, oracle, best_name, arm_names,
                   counts, window, window_lengths, handle)
        print(f"\n  picks: " + ", ".join(f"{n} {chosen.count(n)}" for n in arm_names)
              + f"   (the best arm {chosen.count(best_name) / len(chosen):.2f} of the time)")
        print(f"  wrote {out_dir / name}")


if __name__ == "__main__":
    main()

import numpy as np


def frame_states(occluded, has_box, include_gaps=False):
    """`(present, evaluated)` boolean masks from the annotation, following the conventions above.

    `include_gaps` folds the annotation gaps -- no box and not marked occluded -- into the ABSENT class
    instead of dropping them. They are ~3% of the post-first-occlusion span and treating them either way
    moves every arm by about 0.004 F without changing the differences between arms."""

    occluded = np.asarray(occluded, dtype=bool)
    has_box = np.asarray(has_box, dtype=bool)
    present = has_box & ~occluded
    evaluated = np.ones(len(present), dtype=bool) if include_gaps else (present | occluded)
    return present, evaluated


def sampled(n_frames, stride=1):
    """Frames kept when evaluating at a lower rate than the video's. OxUvA annotates at 1 Hz and scores
    only those frames, arguing that dense labels over-weight temporal context the consumer never sees; at
    PersonPath's 5 fps that is `stride=5`. `stride=1` scores every frame."""

    return np.arange(n_frames) % stride == 0


class Predictions:
    """One arm's per-frame output over a set of clips, in the form both measures consume.

    A clip contributes its per-frame overlap (0 where the target is absent), its present/evaluated masks
    and its confidence. Frames are POOLED across clips rather than averaged per clip, which is what the
    VOT-LT protocol specifies -- a long clip therefore counts for more than a short one."""

    def __init__(self):
        self.clips = []

    def add(self, overlap, confidence, occluded, has_box, stride=1, include_gaps=False):
        """Record one clip. `overlap` is the per-frame IoU of the emitted mask against the ground truth."""

        overlap = np.nan_to_num(np.asarray(overlap, dtype=float), nan=0.0)
        confidence = np.asarray(confidence, dtype=float)
        present, evaluated = frame_states(occluded, has_box, include_gaps)

        length = min(len(overlap), len(present), len(confidence))
        keep = sampled(length, stride)
        self.clips.append((np.where(present[:length], overlap[:length], 0.0)[keep],
                           present[:length][keep],
                           evaluated[:length][keep],
                           confidence[:length][keep]))
        return self

    def __len__(self):
        return len(self.clips)

    # -----------------------------------------------------------------------------------
    # VOT-LT
    # -----------------------------------------------------------------------------------

    def tracking_scores(self, threshold):
        """`(precision, recall, f_score)` at one confidence threshold."""

        overlap_sum = predicted = present_total = 0
        for overlap, present, evaluated, confidence in self.clips:
            reported = evaluated & (confidence >= threshold)
            overlap_sum += float(overlap[reported].sum())
            predicted += int(reported.sum())
            present_total += int(present.sum())

        precision = overlap_sum / predicted if predicted else 0.0
        recall = overlap_sum / present_total if present_total else 0.0
        denominator = precision + recall
        return precision, recall, (2 * precision * recall / denominator if denominator else 0.0)

    def sequence_curves(self, thresholds, localisation=0.0):
        """Per-sequence `(precision, recall)` at each threshold: two `(clips, thresholds)` arrays.

        VOT-LT's protocol computes the precision-recall graph on EACH sequence and averages the graphs --
        "this guarantees that the result is not dominated by extremely long sequences" (Lukezic et al.).
        A sequence with no prediction at a threshold scores precision 0 there; one with no present frame
        contributes nothing and is dropped.

        `localisation` BINARISES the overlap: 0.0 is the published measure, where a frame contributes its
        raw IoU and a 0.55 mask earns 0.55; a positive value credits 1.0 for a frame at or above it and 0
        below, so precision becomes the fraction of reported frames that are LOCATED rather than their mean
        IoU. That variant makes VOT-LT commensurable with coverage@0.5 -- same overlap criterion -- while
        keeping the thing coverage lacks, namely that reporting on an absent frame still costs precision.
        It is not the published statistic, so anything computed with it must say so."""

        thresholds = np.asarray(thresholds, dtype=float)
        precisions, recalls = [], []
        for overlap, present, evaluated, confidence in self.clips:
            if not present.any():
                continue
            if localisation > 0:
                overlap = (overlap >= localisation).astype(float)
            reported = evaluated[None, :] & (confidence[None, :] >= thresholds[:, None])
            collected = (reported * overlap[None, :]).sum(axis=1)
            predicted = reported.sum(axis=1)
            precisions.append(np.divide(collected, predicted, out=np.zeros_like(collected),
                                        where=predicted > 0))
            recalls.append(collected / present.sum())
        return np.asarray(precisions), np.asarray(recalls)

    def macro_scores(self, thresholds, localisation=0.0):
        """`(threshold, precision, recall, f_score)` at the best F of the SEQUENCE-AVERAGED curve.

        This is the published VOT-LT statistic. `tracking_scores` pools frames instead, which weights a
        300-frame clip three times a 100-frame one and gives systematically higher values. `localisation`
        is passed through to `sequence_curves`."""

        precisions, recalls = self.sequence_curves(thresholds, localisation)
        precision, recall = precisions.mean(axis=0), recalls.mean(axis=0)
        denominator = precision + recall
        f_scores = np.divide(2 * precision * recall, denominator,
                             out=np.zeros_like(precision), where=denominator > 0)
        best = int(np.argmax(f_scores))
        return float(np.asarray(thresholds)[best]), float(precision[best]), float(recall[best]), float(f_scores[best])

    def f_curve(self, thresholds):
        """`(thresholds, precisions, recalls, f_scores)` -- the precision-recall curve to plot."""

        rows = np.array([self.tracking_scores(float(t)) for t in thresholds])
        return np.asarray(thresholds, dtype=float), rows[:, 0], rows[:, 1], rows[:, 2]

    def best_f(self, thresholds):
        """The protocol's ranking statistic: `(threshold, precision, recall, f_score)` at the best F."""

        return max(((float(t),) + self.tracking_scores(float(t)) for t in thresholds),
                   key=lambda row: row[3])

    # -----------------------------------------------------------------------------------
    # OxUvA
    # -----------------------------------------------------------------------------------

    def presence_rates(self, threshold, localisation=0.5):
        """`(tpr, tnr)`. `localisation` is the IoU a present frame must reach to count as located; 0 means
        any overlap at all, which is VOT's own failure definition and the 'relaxed on visible' end."""

        located = 0
        present_total = absent_total = silent_on_absent = 0
        for overlap, present, evaluated, confidence in self.clips:
            reported = evaluated & (confidence >= threshold)
            absent = evaluated & ~present
            hit = overlap > 0 if localisation == 0 else overlap >= localisation
            located += int((present & reported & hit).sum())
            silent_on_absent += int((absent & ~reported).sum())
            present_total += int(present.sum())
            absent_total += int(absent.sum())

        return (located / present_total if present_total else 0.0,
                silent_on_absent / absent_total if absent_total else 0.0)

    def best_max_gm(self, thresholds, localisation=0.5):
        """`(threshold, tpr, tnr, max_gm)` at the threshold maximising MaxGM."""

        return max(((float(t),) + self.presence_rates(float(t), localisation)
                    + (max_gm(*self.presence_rates(float(t), localisation)),) for t in thresholds),
                   key=lambda row: row[3])


def max_gm(tpr, tnr, resolution=2001):
    """OxUvA's `max_p sqrt( ((1-p) TPR) ((1-p) TNR + p) )`, over a grid on p."""

    p = np.linspace(0.0, 1.0, resolution)
    return float(np.max(np.sqrt(((1 - p) * tpr) * ((1 - p) * tnr + p))))

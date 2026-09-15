from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class DatasetExperiment:
    """Per-trajectory dataset entry: metadata + precomputed patch-similarity features.

    Everything needed by `MainDataset` lives in this one pickle — no sibling .npz."""

    video_name: str | None = None
    person_id: int | None = None

    # Original video frame indices for each saved frame
    frame_indices: npt.NDArray[np.int64] | None = None

    # Per-PROPOSAL labels: SAM emits 3 competing masks a frame and the arm keeps one, so every
    # label below carries a proposal axis of 3 and `chosen_index` says which column was tracked with.
    iou_scores: npt.NDArray[np.float32] | None = None        # (n, 3) target pseudo-GT mask IoU -- the label
    distractor_iou: npt.NDArray[np.float32] | None = None    # (n, 3, 3) proposal x nearest-distractor IoU
    proposal_iou_scores: npt.NDArray[np.float32] | None = None   # (n, 3) SAM's IoU token per proposal
    proposal_true_iou: npt.NDArray[np.float32] | None = None     # (n, 3) box IoU per proposal
    chosen_index: npt.NDArray[np.int64] | None = None            # (n,) proposal the arm tracked with, 0-2

    box_iou: npt.NDArray[np.float32] | None = None           # (n,) chosen proposal vs the GT box

    occlusions: npt.NDArray[np.float32] | None = None
    true_bboxes: npt.NDArray[np.float32] | None = None

    predicted_iou: npt.NDArray[np.float32] | None = None     # (n,) SAM's token for the chosen proposal
    object_score: npt.NDArray[np.float32] | None = None      # (n,) raw pre-sigmoid presence logit

    # Precomputed similarity features against the fixed anchor: (n_frames, 3, side, side, 2) float16,
    # one map per proposal, all three built from the same crop so they are directly comparable.
    features: npt.NDArray[np.float16] | None = None

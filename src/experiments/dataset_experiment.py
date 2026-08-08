from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


@dataclass
class DatasetExperiment:
    """Per-trajectory dataset entry: metadata + precomputed patch-similarity features.

    Everything needed by `MainDataset` lives in this one pickle — no sibling .npz."""

    video_name: str | None = None
    person_id: int | None = None

    # Original video frame indices for each saved frame — lets experiments reload
    frame_indices: npt.NDArray[np.int64] | None = None

    # Per-frame labels / SAM 2 outputs. `iou_scores` is the training label (pseudo-GT mask IoU);
    # `box_iou` is kept for tracking-eval reference.
    iou_scores: npt.NDArray[np.float32] | None = None       # target pseudo-GT mask IoU (main label)
    box_iou: npt.NDArray[np.float32] | None = None
    occlusions: npt.NDArray[np.float32] | None = None
    predicted_iou: npt.NDArray[np.float32] | None = None
    true_bboxes: npt.NDArray[np.float32] | None = None

    # Pseudo-GT mask IoU of the proposal vs the K nearest clean distractors: (n_frames, K) float32.
    # Column j is the j-th nearest distractor; unfilled columns (fewer than K present) stay zero.
    distractor_iou: npt.NDArray[np.float32] | None = None

    # Precomputed similarity features against the fixed anchor: (n_frames, 1, side, side, 2) float16.
    features: npt.NDArray[np.float16] | None = None

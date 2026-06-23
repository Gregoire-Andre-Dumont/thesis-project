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

    # Per-frame labels / SAM 2 outputs.
    iou_scores: npt.NDArray[np.float32] | None = None
    occlusions: npt.NDArray[np.float32] | None = None
    predicted_iou: npt.NDArray[np.float32] | None = None
    true_bboxes: npt.NDArray[np.float32] | None = None

    # Precomputed similarity features: (n_frames, n_references, side, side, 2) float16.
    features: npt.NDArray[np.float16] | None = None
    fifo_ious: npt.NDArray[np.float32] | None = None

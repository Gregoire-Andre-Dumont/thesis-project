import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from numpy.typing import NDArray
from decord import VideoReader, cpu

from src.utils.load_occlusion import load_occlusions
from src.utils.load_bboxes import load_bboxes
from src.utils.load_frame_ids import load_frame_ids

from decord import bridge
bridge.set_bridge('torch')


@dataclass
class DetectionData:
    """Per-target data sampled at the dataset's true annotation cadence — no interpolation. All
    per-frame arrays (`occlusions`, `frames`, `bboxes_norm`, `amodal_norm`) are aligned to
    `frame_indices`, the sorted union of amodal- and visible-annotated frames for the target."""

    amodal_directory: str | None = None
    visible_directory: str | None = None
    video_directory: str | None = None

    num_threads: int | None = None
    resize_resolution: int | None = None
    load_frames: bool | None = None

    occlusions: NDArray[np.float32] | None = None
    frames: NDArray[np.float32] | None = None
    bboxes_norm: NDArray[np.float32] | None = None
    amodal_norm: NDArray[np.float32] | None = None
    frame_indices: NDArray[np.int64] | None = None

    def __post_init__(self):
        """Convert the different directories to pathlib."""

        self.amodal_directory = Path(self.amodal_directory)
        self.visible_directory = Path(self.visible_directory)
        self.video_directory = Path(self.video_directory)

    def initialize_target(self, video_name: str, person_id: int,
                            frame_indices: NDArray[np.int64] | None = None):
        """Load the frames and ground truth of the target person at the annotated frames only.

        If `frame_indices` is provided (e.g., from a `DatasetExperiment` pickle saved by
        `create_anchor_dataset.py`), the trajectory is restricted to those indices in
        the given order — every per-frame array is sliced to align with them, and the
        video reader fetches only those frames. This lets experiments reproduce the
        anchor-sliced trajectory the dataset was built on without re-deriving it."""

        self.video_path = str(self.video_directory / video_name)

        amodal_path = self.amodal_directory / (video_name + ".json")
        visible_path = self.visible_directory / (video_name + ".json")

        # Sorted union of annotated frame indices — the single time axis for all per-frame arrays.
        amodal_ids, _, _ = load_frame_ids(amodal_path, person_id)
        visible_ids, _, _ = load_frame_ids(visible_path, person_id)
        full_union = np.unique(np.concatenate([amodal_ids, visible_ids]))

        self.occlusions = load_occlusions(amodal_path, visible_path, person_id)
        self.bboxes_norm = load_bboxes(amodal_path, visible_path, person_id, False)
        self.amodal_norm = load_bboxes(amodal_path, visible_path, person_id, True)

        if frame_indices is None:
            self.frame_indices = full_union
        else:
            self.frame_indices = np.asarray(frame_indices, dtype=np.int64)
            positions = np.searchsorted(full_union, self.frame_indices)
            self.occlusions = self.occlusions[positions]
            self.bboxes_norm = self.bboxes_norm[positions]
            self.amodal_norm = self.amodal_norm[positions]

        # Zero the visible bbox at fully-occluded frames; force occlusion to a clean {0, 1}.
        self.bboxes_norm[self.occlusions > 0.05] = [0, 0, 0, 0]
        self.occlusions[self.occlusions > 0.05] = 1.0

        video_read = VideoReader(self.video_path, ctx=cpu(0), num_threads=self.num_threads)
        if self.load_frames:
            self.frames = video_read.get_batch(self.frame_indices).cpu().numpy()




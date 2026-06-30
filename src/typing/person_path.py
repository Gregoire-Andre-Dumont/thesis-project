import os
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm


def _frame_idx(entity: dict) -> int:
    """Annotated video-frame index of an entity."""

    return entity["blob"]["frame_idx"]


def _bbox_area(bbox) -> float:
    """Area of a `(x, y, w, h)` bbox, clamping negative extents to zero."""

    _, _, width, height = bbox
    return max(0, width) * max(0, height)


@dataclass
class PersonPath:
    """Manages the frames and labels of the person path dataset."""

    main_directory: str | None = None
    random_seed: float | None = None

    # Related to loading the video frames
    num_threads: int | None = None
    resize_resolution: int | None = None

    # Related to the target selections
    non_targets: list[str] | None = None
    occlusion_ranges: list[int] | None = None
    min_frames: int | None = None
    max_frames: int | None = None
    n_after_occlusion: int | None = None
    n_experiments: int | None = None
    total_experiments: int | None = None
    min_visible_ratio: float = 0.9
    min_frames_after_anchor: int = 0

    # Related to all the chosen targets
    selected_video_names: list[str] | None = None
    selected_person_ids: list[int] | None = None
    selected_anchor_video_frames: list[int] | None = None

    def __post_init__(self):
        """Resolve dataset paths, then enumerate and sample the target trajectories."""

        main_directory = Path(self.main_directory)
        self.metadata_path = main_directory / "visible.json"
        self.video_directory = main_directory / "videos"
        self.amodal_directory = main_directory / "amodal"
        self.visible_directory = main_directory / "visible"

        video_names = os.listdir(self.video_directory)
        video_person_ids = self.filter_targets(video_names)

        self.select_targets(video_names, video_person_ids)
        self.total_experiments = self.n_experiments * len(self.occlusion_ranges)

    # -------------------------------------------------------------------------------
    # Target enumeration + sampling
    # -------------------------------------------------------------------------------

    def filter_targets(self, video_names: list[str]) -> list[list[int]]:
        """Per video, the person ids annotated as a valid target in BOTH the amodal and
        visible files (excluding any entity carrying a `non_targets` label)."""

        video_person_ids = []
        for video_name in video_names:
            amodal_entities = self._read_entities(self.amodal_directory, video_name)
            visible_entities = self._read_entities(self.visible_directory, video_name)

            amodal_ids = {e["id"] for e in amodal_entities if self._is_target(e)}
            visible_ids = {e["id"] for e in visible_entities if self._is_target(e)}
            video_person_ids.append(list(amodal_ids & visible_ids))

        return video_person_ids

    def select_targets(self, video_names: list[str], video_person_ids: list[list[int]]):
        """Enumerate every valid target across the dataset (one pass through the JSONs) and
        stratified-sample `n_experiments` per occlusion bin."""

        n_bins = len(self.occlusion_ranges) - 1
        candidates = [[] for _ in range(n_bins)]

        for video_name, person_ids in tqdm(zip(video_names, video_person_ids),
                                           total=len(video_names), desc="Enumerate"):
            amodal_by_pid = self._group_by_id(self._read_entities(self.amodal_directory, video_name))
            visible_by_pid = self._group_by_id(self._read_entities(self.visible_directory, video_name))

            for person_id in person_ids:
                amodal_entities = amodal_by_pid.get(person_id, [])
                visible_entities = visible_by_pid.get(person_id, [])

                bin_idx = self._bin_index(self._occlusion_array(amodal_entities, visible_entities))
                if bin_idx is None:
                    continue

                anchor_video_frame = self._anchor_video_frame(
                    amodal_entities, visible_entities, self.min_visible_ratio)
                if anchor_video_frame is None:
                    continue

                if not self._has_enough_post_anchor_frames(
                        amodal_entities, visible_entities, anchor_video_frame,
                        self.min_frames_after_anchor):
                    continue

                candidates[bin_idx].append((video_name, person_id, anchor_video_frame))

        random_generator = np.random.default_rng(self.random_seed)
        selected = []
        for pool in candidates:
            chosen = random_generator.choice(len(pool), size=self.n_experiments, replace=False)
            selected.extend(pool[i] for i in chosen)
        random_generator.shuffle(selected)

        self.selected_video_names = np.array([video_name for video_name, _, _ in selected])
        self.selected_person_ids = np.array([person_id for _, person_id, _ in selected])
        self.selected_anchor_video_frames = np.array(
            [anchor_frame for _, _, anchor_frame in selected], dtype=np.int64)

    # -------------------------------------------------------------------------------
    # Annotation IO
    # -------------------------------------------------------------------------------

    def _is_target(self, entity: dict) -> bool:
        """True when none of the entity's labels mark it as a non-target."""

        return not any(label in self.non_targets for label in entity["labels"])

    def _read_entities(self, directory: Path, video_name: str) -> list[dict]:
        """Load the `entities` list from `<directory>/<video_name>.json`."""

        with open(directory / (video_name + ".json"), "r") as handle:
            return json.load(handle)["entities"]

    @staticmethod
    def _group_by_id(entities: list[dict]) -> dict[int, list[dict]]:
        """Group entities by their person id."""

        grouped = {}
        for entity in entities:
            grouped.setdefault(entity["id"], []).append(entity)
        return grouped

    # -------------------------------------------------------------------------------
    # Per-target derivations
    # -------------------------------------------------------------------------------

    @staticmethod
    def _anchor_video_frame(amodal_entities, visible_entities, min_visible_ratio):
        """First visible video-frame index where the visible bbox area covers at least
        `min_visible_ratio` of the NEAREST amodal bbox area (by frame index). Falling
        back to the nearest amodal lets a visible-only frame still get a sensible
        reference area when amodal annotations are sparse. Returns `None` if no such
        frame exists."""

        if not amodal_entities or not visible_entities:
            return None

        amodal_sorted = sorted(amodal_entities, key=_frame_idx)
        amodal_frames = np.array([_frame_idx(entity) for entity in amodal_sorted])
        amodal_areas = np.array([_bbox_area(entity["bb"]) for entity in amodal_sorted], dtype=np.float64)
        occluded_frames = {_frame_idx(entity) for entity in amodal_entities
                           if "fully_occluded" in entity["labels"]}

        for entity in sorted(visible_entities, key=_frame_idx):
            visible_frame = int(_frame_idx(entity))
            if visible_frame in occluded_frames:
                continue
            visible_area = _bbox_area(entity["bb"])
            if visible_area <= 0:
                continue

            position = int(np.searchsorted(amodal_frames, visible_frame))
            neighbours = []
            if position < len(amodal_frames):
                neighbours.append(position)
            if position > 0:
                neighbours.append(position - 1)
            nearest = min(neighbours, key=lambda i: abs(amodal_frames[i] - visible_frame))

            amodal_area = float(amodal_areas[nearest])
            if amodal_area <= 0:
                continue

            if visible_area / amodal_area >= min_visible_ratio:
                return visible_frame
        return None

    @staticmethod
    def _has_enough_post_anchor_frames(amodal_entities, visible_entities,anchor_video_frame, min_frames_after_anchor):
        """True when the trajectory's annotated frames (amodal ∪ visible) contain at
        least `min_frames_after_anchor` entries strictly after `anchor_video_frame`."""

        if min_frames_after_anchor <= 0:
            return True
        annotated = PersonPath._annotated_frames(amodal_entities, visible_entities)
        return int(np.sum(annotated > anchor_video_frame)) >= min_frames_after_anchor

    @staticmethod
    def _occlusion_array(amodal_entities, visible_entities):
        """Sparse occlusion vector at the annotated-frame cadence (1 where the amodal entity
        carries `fully_occluded`, 0 elsewhere)."""

        annotated = PersonPath._annotated_frames(amodal_entities, visible_entities)
        occluded = {_frame_idx(entity) for entity in amodal_entities if "fully_occluded" in entity["labels"]}
        return np.array([frame in occluded for frame in annotated], dtype=np.int32)

    @staticmethod
    def _annotated_frames(amodal_entities, visible_entities) -> np.ndarray:
        """Sorted unique union of the amodal- and visible-annotated frame indices."""

        amodal_frames = np.fromiter((_frame_idx(e) for e in amodal_entities), dtype=np.int64)
        visible_frames = np.fromiter((_frame_idx(e) for e in visible_entities), dtype=np.int64)
        return np.unique(np.concatenate([amodal_frames, visible_frames]))

    def _bin_index(self, occlusions):
        """Bin index satisfying every selection condition, or None when one fails."""

        if not (self.min_frames <= len(occlusions) <= self.max_frames):
            return None

        n_occluded = int(occlusions.sum())
        bin_idx = next((i for i in range(len(self.occlusion_ranges) - 1)
                        if self.occlusion_ranges[i] < n_occluded < self.occlusion_ranges[i + 1]), None)
        if bin_idx is None:
            return None

        if self.n_after_occlusion is not None:
            first = int(np.argmax(occlusions > 0))
            if int(np.sum(occlusions[first:] == 0)) <= self.n_after_occlusion:
                return None

        return bin_idx

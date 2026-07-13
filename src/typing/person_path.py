import os
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm


def _group_by_id(entities: list[dict]) -> dict[int, list[dict]]:
    """Entities grouped by person id."""

    grouped: dict[int, list[dict]] = {}
    for entity in entities:
        grouped.setdefault(entity["id"], []).append(entity)
    return grouped


def _occlusion_array(amodal: list[dict], visible: list[dict], since: int = None) -> np.ndarray:
    """Per-annotated-frame flag (over the sorted amodal ∪ visible frames, from `since` onwards),
    1 where the amodal entity is `fully_occluded`."""

    frames = np.unique([entity["blob"]["frame_idx"] for entity in (*amodal, *visible)])
    if since is not None:
        frames = frames[frames >= since]
    occluded = {entity["blob"]["frame_idx"] for entity in amodal if "fully_occluded" in entity["labels"]}
    return np.array([frame in occluded for frame in frames], dtype=np.int32)


def _covered_fraction(box: list, other: list) -> float:
    """Fraction of `box`'s area covered by `other` (both [x, y, w, h] in pixels)."""

    ix = max(0.0, min(box[0] + box[2], other[0] + other[2]) - max(box[0], other[0]))
    iy = max(0.0, min(box[1] + box[3], other[1] + other[3]) - max(box[1], other[1]))
    area = box[2] * box[3]
    return (ix * iy) / area if area > 0 else 0.0


def _nearest_index(sorted_values: np.ndarray, target: int) -> int:
    """Index of the entry in `sorted_values` closest to `target` (ties → the higher index)."""

    position = int(np.searchsorted(sorted_values, target))
    candidates = [i for i in (position, position - 1) if 0 <= i < len(sorted_values)]
    return min(candidates, key=lambda i: abs(sorted_values[i] - target))


def _anchor_video_frame(amodal: list[dict], visible: list[dict], boxes_by_frame: dict, person_id: int,
                        min_visible_ratio: float, min_area: float, max_overlap: float):
    """First visible frame that makes a clean reference: box of at least `min_area` pixels, covering
    at least `min_visible_ratio` of the nearest amodal box on BOTH sides (width and height), and with
    no other person's box covering more than `max_overlap` of it — so the box prompt is unambiguous.
    Requiring each side rather than area stops a target occluded along one dimension from qualifying
    by being oversized on the other. `None` if no frame qualifies."""

    if not amodal or not visible:
        return None

    frame_of = lambda entity: entity["blob"]["frame_idx"]
    amodal = sorted(amodal, key=frame_of)
    amodal_frames = np.array([frame_of(entity) for entity in amodal])
    amodal_wh = np.array([entity["bb"][2:4] for entity in amodal], dtype=np.float64)
    occluded = {frame_of(entity) for entity in amodal if "fully_occluded" in entity["labels"]}

    for entity in sorted(visible, key=frame_of):
        frame = int(frame_of(entity))
        visible_wh = np.array(entity["bb"][2:4], dtype=np.float64)
        if frame in occluded or np.any(visible_wh <= 0) or np.prod(visible_wh) < min_area:
            continue
        amodal_wh_here = amodal_wh[_nearest_index(amodal_frames, frame)]
        if np.any(amodal_wh_here <= 0):
            continue
        if not np.all(visible_wh / amodal_wh_here >= min_visible_ratio):
            continue
        if any(_covered_fraction(entity["bb"], other) > max_overlap
               for other_id, other in boxes_by_frame.get(frame, []) if other_id != person_id):
            continue
        return frame
    return None


@dataclass
class PersonPath:
    """Selects target trajectories from the PersonPath dataset — valid targets, occluded within
    `occlusion_ranges`, each with a clean anchor frame — sampling `n_experiments` of them into the
    `selected_*` arrays (aligned triples of video name, person id and anchor video-frame index)."""

    main_directory: str | None = None
    random_seed: float | None = None

    # Target-selection conditions
    non_targets: list[str] | None = None
    occlusion_ranges: list[int] | None = None
    n_after_occlusion: int | None = None
    n_experiments: int | None = None
    min_visible_ratio: float = 0.9
    resize_resolution: int = 1024          # working resolution the min-area threshold is measured at
    min_visible_area: float = 0.0          # min anchor box area in px² at `resize_resolution` (COCO small = 32²)
    max_distractor_overlap: float = 1.0    # max fraction of the anchor box another person may cover

    # Chosen targets
    total_experiments: int | None = None
    selected_video_names: list[str] | None = None
    selected_person_ids: list[int] | None = None
    selected_anchor_video_frames: list[int] | None = None

    def __post_init__(self):
        """Resolve dataset paths, then enumerate and sample the target trajectories."""

        main_directory = Path(self.main_directory)
        self.amodal_directory = main_directory / "amodal"
        self.visible_directory = main_directory / "visible"
        self.video_directory = main_directory / "videos"

        self.select_targets(sorted(os.listdir(self.video_directory)))
        self.total_experiments = len(self.selected_video_names)

    def _is_target(self, entity: dict) -> bool:
        """True when no label marks the entity as a non-target."""

        return not any(label in self.non_targets for label in entity["labels"])

    def _passes_occlusion(self, occlusions: np.ndarray) -> bool:
        """True when the occlusion count lands strictly inside `occlusion_ranges` and more than
        `n_after_occlusion` visible frames follow the first occlusion."""

        n_occluded = int(occlusions.sum())
        if not (self.occlusion_ranges[0] < n_occluded < self.occlusion_ranges[-1]):
            return False
        first = int(np.argmax(occlusions > 0))
        return int(np.sum(occlusions[first:] == 0)) > self.n_after_occlusion

    def select_targets(self, video_names: list[str]):
        """Enumerate every valid target once and randomly sample `n_experiments` of those meeting
        the occlusion and anchor conditions into the `selected_*` arrays."""

        candidates = []
        for video_name in tqdm(video_names, desc="Enumerate"):
            amodal = json.load(open(self.amodal_directory / f"{video_name}.json"))["entities"]
            visible_json = json.load(open(self.visible_directory / f"{video_name}.json"))
            visible = visible_json["entities"]
            amodal_ids = {entity["id"] for entity in amodal if self._is_target(entity)}
            visible_ids = {entity["id"] for entity in visible if self._is_target(entity)}
            amodal_by_id, visible_by_id = _group_by_id(amodal), _group_by_id(visible)

            resolution = visible_json["metadata"]["resolution"]
            scale = self.resize_resolution / max(resolution["width"], resolution["height"])
            min_area = self.min_visible_area / scale ** 2      # threshold back in raw annotation pixels

            boxes_by_frame = {}                                # every annotated box, to spot distractors
            for entity in visible:
                boxes_by_frame.setdefault(entity["blob"]["frame_idx"], []).append(
                    (entity["id"], entity["bb"]))

            for person_id in list(amodal_ids & visible_ids):
                entities = (amodal_by_id.get(person_id, []), visible_by_id.get(person_id, []))
                anchor = _anchor_video_frame(*entities, boxes_by_frame, person_id, self.min_visible_ratio,
                                             min_area, self.max_distractor_overlap)
                if anchor is None:
                    continue
                # Occlusion conditions apply to the tracked segment, which starts at the anchor.
                if not self._passes_occlusion(_occlusion_array(*entities, since=anchor)):
                    continue
                candidates.append((video_name, person_id, anchor))

        rng = np.random.default_rng(self.random_seed)
        chosen = rng.choice(len(candidates), size=min(self.n_experiments, len(candidates)), replace=False)
        selected = [candidates[i] for i in chosen]
        rng.shuffle(selected)

        videos, pids, anchors = zip(*selected) if selected else ((), (), ())
        self.selected_video_names = np.array(videos)
        self.selected_person_ids = np.array(pids)
        self.selected_anchor_video_frames = np.array(anchors, dtype=np.int64)

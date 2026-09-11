import os
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm


def _group_entities_by_person_id(entities: list[dict]) -> dict[int, list[dict]]:
    """Group annotation entities by their person id."""

    entities_by_person_id: dict[int, list[dict]] = {}
    for entity in entities:
        entities_by_person_id.setdefault(entity["id"], []).append(entity)
    return entities_by_person_id


def _occlusion_flags(amodal_entities: list[dict], visible_entities: list[dict], since_frame: int = None) -> np.ndarray:
    """Per-annotated-frame occlusion flag over the sorted amodal-union-visible frames (from `since_frame`
    onwards), matching DetectionData: a frame is occluded ONLY when the dataset labels it `fully_occluded`.

    A frame that is merely absent from the visible annotations is an unannotated gap, not an occlusion --
    ~22% of box-less frames dataset-wide. Counting those inflated occlusion runs and made trajectory
    selection partly a function of annotation sparsity."""

    all_entities = (*amodal_entities, *visible_entities)
    annotated_frames = np.unique([entity["blob"]["frame_idx"] for entity in all_entities])
    if since_frame is not None:
        annotated_frames = annotated_frames[annotated_frames >= since_frame]

    fully_occluded_frames = {entity["blob"]["frame_idx"] for entity in amodal_entities
                             if "fully_occluded" in entity["labels"]}

    is_occluded = [frame in fully_occluded_frames for frame in annotated_frames]
    return np.array(is_occluded, dtype=np.int32)


def _covered_fraction(target_box: list, other_box: list) -> float:
    """Fraction of `target_box`'s area covered by `other_box` (both [x, y, width, height] in pixels)."""

    target_right = target_box[0] + target_box[2]
    target_bottom = target_box[1] + target_box[3]
    other_right = other_box[0] + other_box[2]
    other_bottom = other_box[1] + other_box[3]

    overlap_left = max(target_box[0], other_box[0])
    overlap_right = min(target_right, other_right)
    overlap_top = max(target_box[1], other_box[1])
    overlap_bottom = min(target_bottom, other_bottom)

    intersection_width = max(0.0, overlap_right - overlap_left)
    intersection_height = max(0.0, overlap_bottom - overlap_top)
    intersection_area = intersection_width * intersection_height
    target_area = target_box[2] * target_box[3]

    return intersection_area / target_area if target_area > 0 else 0.0


def _nearest_index(sorted_frames: np.ndarray, target_frame: int) -> int:
    """Index of the entry in `sorted_frames` closest to `target_frame` (ties resolve to the higher index)."""

    insertion_point = int(np.searchsorted(sorted_frames, target_frame))
    candidate_positions = (insertion_point, insertion_point - 1)
    candidate_indices = [index for index in candidate_positions if 0 <= index < len(sorted_frames)]
    return min(candidate_indices, key=lambda index: abs(sorted_frames[index] - target_frame))


def _anchor_video_frame(amodal_entities: list[dict], visible_entities: list[dict], boxes_by_frame: dict, person_id: int,
                        min_area: float, max_distractor_overlap: float,
                        min_distractor_overlap: float = 0.0, max_area: float = float("inf"),
                        min_visible_ratio: float = 0.0, border_inset: float = 0.0,
                        frame_size: tuple | None = None):
    """First visible frame that makes a clean reference: not fully occluded, with a valid box, the target's
    VISIBLE box area within [`min_area`, `max_area`] pixels, and the closest
    other person's box covering between `min_distractor_overlap` and `max_distractor_overlap` of it. The upper
    overlap bound keeps the box prompt unambiguous; the lower bound (default 0) can instead REQUIRE a nearby
    distractor, to study the confident-drift regime. Every condition -- size and distractor overlap
    -- is evaluated per frame, so the search walks forward to the first frame that satisfies them all; the
    trajectory is skipped only if no frame qualifies. Returns (frame, closest-distractor-overlap), or `None`.

    `min_visible_ratio` > 0 additionally requires visible_area / amodal_area to reach it, i.e. the anchor is at
    most partly occluded. It is OFF by default (0.0) because it reads amodal GEOMETRY, which is unreliable --
    7.3% of amodal boxes are smaller than the visible box they must contain, with outliers up to 194x -- so a
    ratio gate partly filters on corrupt numbers. Set it deliberately, knowing that.

    `border_inset` > 0 (pixels, needs `frame_size`) rejects frames whose box touches the image border, so the
    search walks on to the first frame holding the target WHOLE. This is the only way to catch a target sliced
    by the frame edge: load_bboxes clips amodal and visible boxes alike to the image, so a half-off-screen
    person has amodal == visible and `min_visible_ratio` sees a ratio of 1 no matter how strict it is set.
    Worse, the truncated box is tall and thin, so `min_area` reads such anchors as LARGE and easy when they
    hold only a sliver of the person -- 16.3% of claim_1 anchors touched a border, at 2.3x the median area."""

    if not amodal_entities or not visible_entities:
        return None

    frame_of = lambda entity: entity["blob"]["frame_idx"]
    occluded_entities = [entity for entity in amodal_entities if "fully_occluded" in entity["labels"]]
    fully_occluded_frames = {frame_of(entity) for entity in occluded_entities}

    if min_visible_ratio > 0.0:
        sorted_amodal = sorted(amodal_entities, key=frame_of)
        amodal_frames = np.array([frame_of(entity) for entity in sorted_amodal])
        amodal_width_height = np.array([entity["bb"][2:4] for entity in sorted_amodal], dtype=np.float64)

    for visible_entity in sorted(visible_entities, key=frame_of):  # frame-level search for a clean prompt frame
        frame = int(frame_of(visible_entity))
        visible_width_height = np.array(visible_entity["bb"][2:4], dtype=np.float64)
        if frame in fully_occluded_frames or np.any(visible_width_height <= 0):
            continue

        visible_area = float(np.prod(visible_width_height))
        if not (min_area <= visible_area <= max_area):                     # size gate on the VISIBLE box (frame-based)
            continue

        if border_inset > 0.0 and frame_size is not None:                  # target must be WHOLE in frame
            frame_width, frame_height = frame_size
            x, y = float(visible_entity["bb"][0]), float(visible_entity["bb"][1])
            width, height = float(visible_width_height[0]), float(visible_width_height[1])
            if (x < border_inset or y < border_inset or
                    x + width > frame_width - border_inset or y + height > frame_height - border_inset):
                continue

        if min_visible_ratio > 0.0:                                        # anchor must be mostly unoccluded
            amodal_area = float(np.prod(amodal_width_height[_nearest_index(amodal_frames, frame)]))
            if amodal_area <= 0 or visible_area / amodal_area < min_visible_ratio:
                continue

        frame_boxes = boxes_by_frame.get(frame, [])
        other_boxes = [box for other_id, box in frame_boxes if other_id != person_id]
        distractor_coverages = [_covered_fraction(visible_entity["bb"], box) for box in other_boxes]
        closest_distractor_overlap = max(distractor_coverages) if distractor_coverages else 0.0

        overlap_in_range = min_distractor_overlap <= closest_distractor_overlap <= max_distractor_overlap
        if not overlap_in_range:
            continue

        return frame, closest_distractor_overlap
    return None


@dataclass
class PersonPath:
    """Selects target trajectories from the PersonPath dataset -- valid targets, occluded within
    `occlusion_ranges`, each with a clean anchor frame -- and samples `n_experiments` of them into the
    `selected_*` arrays (aligned per-trajectory: video name, person id, anchor video-frame index, and the
    anchor's distractor overlap and visible-area ratio)."""

    main_directory: str | None = None
    random_seed: float | None = None

    # Target-selection conditions
    non_targets: list[str] | None = None
    occlusion_ranges: list[int] | None = None
    n_after_occlusion: int | None = None
    first_occ_min: int = 0                 # require the first occlusion strictly after this frame past the anchor
    n_experiments: int | None = None
    resize_resolution: int = 1024          # working resolution the min-area threshold is measured at
    min_visible_area: float = 1024         # min anchor VISIBLE box area in px² at `resize_resolution` (COCO small = 32²)
    max_visible_area: float | None = None  # max anchor VISIBLE box area in px² at `resize_resolution` (None = no cap)
    max_distractor_overlap: float = 0.5    # max fraction of the anchor box another person may cover
    min_distractor_overlap: float = 0.0    # min such fraction -- require a nearby distractor at the anchor
    min_visible_ratio: float = 0.0         # min anchor visible/amodal area ratio (0 = off; reads amodal geometry)
    border_inset: float = 0.0              # px the anchor box must keep clear of every image border (0 = off)

    # Chosen targets
    total_experiments: int | None = None
    selected_video_names: list[str] | None = None
    selected_person_ids: list[int] | None = None
    selected_anchor_video_frames: list[int] | None = None
    selected_anchor_overlaps: list[float] | None = None          # closest-distractor overlap at each anchor frame

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

    def _passes_occlusion(self, occlusion_flags: np.ndarray) -> bool:
        """True when the occluded-frame count lands strictly inside `occlusion_ranges`, the first occlusion
        happens strictly after `first_occ_min` frames past the anchor (a clean run-in), and more than
        `n_after_occlusion` visible frames follow the first occlusion."""

        occluded_frame_count = int(occlusion_flags.sum())
        if not (self.occlusion_ranges[0] < occluded_frame_count < self.occlusion_ranges[-1]):
            return False

        first_occluded_index = int(np.argmax(occlusion_flags > 0))
        if first_occluded_index <= self.first_occ_min:
            return False

        visible_frames_after_first_occlusion = int(np.sum(occlusion_flags[first_occluded_index:] == 0))
        return visible_frames_after_first_occlusion > self.n_after_occlusion

    def select_targets(self, video_names: list[str]):
        """Enumerate every valid target once and randomly sample `n_experiments` of those meeting the anchor
        and occlusion conditions into the `selected_*` arrays."""

        candidate_trajectories = []
        for video_name in tqdm(video_names, desc="Enumerate"):
            amodal_annotation = json.load(open(self.amodal_directory / f"{video_name}.json"))
            visible_annotation = json.load(open(self.visible_directory / f"{video_name}.json"))
            amodal_entities = amodal_annotation["entities"]
            visible_entities = visible_annotation["entities"]

            amodal_target_ids = {entity["id"] for entity in amodal_entities if self._is_target(entity)}
            visible_target_ids = {entity["id"] for entity in visible_entities if self._is_target(entity)}
            amodal_entities_by_id = _group_entities_by_person_id(amodal_entities)
            visible_entities_by_id = _group_entities_by_person_id(visible_entities)

            resolution = visible_annotation["metadata"]["resolution"]
            longest_side = max(resolution["width"], resolution["height"])
            resize_scale = self.resize_resolution / longest_side

            min_area_pixels = self.min_visible_area / resize_scale ** 2
            max_area_pixels = self.max_visible_area / resize_scale ** 2 if self.max_visible_area else float("inf")

            boxes_by_frame = {}                                          # every annotated box, to spot distractors
            for entity in visible_entities:
                frame_index = entity["blob"]["frame_idx"]
                boxes_by_frame.setdefault(frame_index, []).append((entity["id"], entity["bb"]))

            target_person_ids = amodal_target_ids & visible_target_ids
            for person_id in target_person_ids:
                person_amodal_entities = amodal_entities_by_id.get(person_id, [])
                person_visible_entities = visible_entities_by_id.get(person_id, [])
                anchor = _anchor_video_frame(person_amodal_entities, person_visible_entities, boxes_by_frame,
                        person_id, min_area_pixels, self.max_distractor_overlap,
                        self.min_distractor_overlap, max_area_pixels, self.min_visible_ratio,
                        self.border_inset, (resolution["width"], resolution["height"]))

                if anchor is None:
                    continue

                anchor_frame, distractor_overlap = anchor
                occlusion_flags = _occlusion_flags(person_amodal_entities, person_visible_entities, anchor_frame)
                if not self._passes_occlusion(occlusion_flags):
                    continue

                trajectory = (video_name, person_id, anchor_frame, distractor_overlap)
                candidate_trajectories.append(trajectory)

        random_generator = np.random.default_rng(self.random_seed)
        sample_size = min(self.n_experiments, len(candidate_trajectories))
        chosen_indices = random_generator.choice(len(candidate_trajectories), size=sample_size, replace=False)
        sampled_trajectories = [candidate_trajectories[index] for index in chosen_indices]
        random_generator.shuffle(sampled_trajectories)

        if sampled_trajectories:
            video_names, person_ids, anchor_frames, overlaps = zip(*sampled_trajectories)
        else:
            video_names, person_ids, anchor_frames, overlaps = (), (), (), ()

        self.selected_video_names = np.array(video_names)
        self.selected_person_ids = np.array(person_ids)
        self.selected_anchor_video_frames = np.array(anchor_frames, dtype=np.int64)

        self.selected_anchor_overlaps = np.array(overlaps, dtype=np.float64)

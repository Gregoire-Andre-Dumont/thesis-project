import cv2
import torch
import numpy as np

from dataclasses import dataclass
from collections import deque
from src.utils.load_bboxes import convert_bbox
from src.typing.detection_data import DetectionData


@dataclass
class MainMemoryV3:
    """`MainMemory` for SAM 3, where a memory entry is ONE object rather than a (pointer, encoding) pair.

    SAM 2's fusion takes four parallel deques -- prompt encodings, prompt pointers, previous encodings,
    previous pointers -- and SAMARA's controller appends to two of them. SAM 3 bundles each entry into a
    single `SAMv3MemoryEncoding` that already carries its own object pointer, so the bank halves to two
    deques. `update_memory(pointer, encoding)` keeps SAM 2's signature and ignores `pointer`, so the
    controller does not have to know which generation it is driving.

    Ordering: entries are appended LEFT, newest first, and `step_video_masking` is told `is_recent_first=True`
    to match. SAM 2's fusion assumed that order implicitly; SAM 3 made it a flag, and getting it backwards
    silently reverses the memory's time axis rather than raising.

    The calibrator anchor is identical to SAM 2's -- it holds Perception Encoder tokens, not SAM features,
    so nothing about it changes with the backbone."""

    max_memory_history: int | None = None
    max_pointer_history: int | None = None          # accepted for config parity; SAM 3 pointers ride along

    prompt_memories: deque | None = None
    frame_memories: deque | None = None

    fixed_reference_foreground: torch.Tensor | None = None
    fixed_reference_background: torch.Tensor | None = None

    def update_reference(self, pointer, encoding):
        """Pin a prompt memory. Never evicted -- this is the anchor SAM 3 keeps re-reading."""

        self.prompt_memories.appendleft(encoding)

    def update_memory(self, pointer, encoding):
        """Commit one frame memory. `pointer` is unused: a SAM 3 memory encoding carries its own."""

        self.frame_memories.appendleft(encoding)

    def seed_calibrator_anchor(self, fixed_foreground, fixed_background):
        """Pin the calibrator's single fixed anchor (fg + bg tokens)."""

        self.fixed_reference_foreground = fixed_foreground
        self.fixed_reference_background = fixed_background

    def gather_calibrator_references(self):
        """Return (foreground, background) reference stacks, each shaped
        `(1, n_patches, feature_dim)` so `compute_patch_similarities`' einsum still
        sees a leading reference dimension. Anchor-only -- no FIFO concatenation."""

        foreground = self.fixed_reference_foreground.unsqueeze(0)
        background = self.fixed_reference_background.unsqueeze(0)
        return foreground, background

    def reset_memory(self):
        self.prompt_memories = deque([])
        self.frame_memories = deque([], maxlen=self.max_memory_history)

        self.fixed_reference_foreground = None
        self.fixed_reference_background = None

    def initialize_references(self, model, detection_data: DetectionData, anchor_index: int = 0):
        """Bootstrap SAM 3's video-masking memory from the chosen anchor's GT bbox.

        Returns the anchor's mask logits, which the controller feeds to `initialize_calibrator_anchor`."""

        reference = detection_data.frames[anchor_index]
        bbox_norm = convert_bbox(detection_data.bboxes_norm[anchor_index])
        model.set_anchor_size_from_normalized(detection_data.bboxes_norm[anchor_index], reference.shape[:2])

        bgr_reference = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
        encoded_image = model.encode_image(bgr_reference)
        init_mask, encoding = model.encode_prompt_memory(encoded_image, bbox_norm, [], [])

        self.update_reference(None, encoding)
        return init_mask

    def initialize_calibrator_anchor(self, model, detection_data: DetectionData, init_mask, anchor_index: int = 0):
        """Seed the calibrator anchor from SAM 3's initialization mask."""

        frame = detection_data.frames[anchor_index]
        mask = init_mask.to(torch.float64).cpu().numpy().squeeze()      # keep logits; extract_crops thresholds

        foreground, background = model.extract_patch_tokens(
            *model.extract_crops(frame[np.newaxis], mask[np.newaxis]))
        self.seed_calibrator_anchor(foreground[0], background[0])

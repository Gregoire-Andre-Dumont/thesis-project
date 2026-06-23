import cv2
import torch
import numpy as np

from dataclasses import dataclass
from collections import deque
from src.utils.load_bboxes import convert_bbox
from src.typing.detection_data import DetectionData


@dataclass
class MainMemory:
    """Memory bank for SAM 2 video object segmentation.

    Also owns the calibrator's reference set (fixed slot + moving FIFO),
    so that both SAM 2's memory bank and the calibrator references are
    updated under the same gate."""

    max_memory_history: int | None = None
    max_pointer_history: int | None = None
    moving_capacity: int | None = None

    reference_encodings: deque | None = None
    reference_pointers: deque | None = None

    previous_encodings: deque | None = None
    previous_pointers: deque | None = None

    # Calibrator references keep BOTH foreground and background tokens
    fixed_reference_foreground: torch.Tensor | None = None
    fixed_reference_background: torch.Tensor | None = None
    
    moving_reference_foregrounds: deque | None = None
    moving_reference_backgrounds: deque | None = None

    def update_reference(self, pointer, encoding):
        self.reference_encodings.appendleft(encoding)
        self.reference_pointers.appendleft(pointer)

    def update_memory(self, pointer, encoding):
        self.previous_encodings.appendleft(encoding)
        self.previous_pointers.appendleft(pointer)

    def seed_calibrator_references(self, fixed_foreground, fixed_background, seed_foregrounds, seed_backgrounds):
        """Pin slot 0 (fixed anchor) and seed the moving FIFO, for both fg and bg tokens."""

        self.fixed_reference_foreground = fixed_foreground
        self.fixed_reference_background = fixed_background
        self.moving_reference_foregrounds = deque(list(seed_foregrounds), maxlen=self.moving_capacity)
        self.moving_reference_backgrounds = deque(list(seed_backgrounds), maxlen=self.moving_capacity)

    def update_calibrator_reference(self, foreground_token, background_token):
        """Push a frame's fg+bg tokens onto the moving FIFO. The deque's maxlen evicts the oldest."""

        self.moving_reference_foregrounds.append(foreground_token)
        self.moving_reference_backgrounds.append(background_token)

    def update_calibrator_anchor(self, foreground_token, background_token):
        """Replace the fixed anchor's fg+bg tokens with a recent high-confidence frame, so the
        anchor tracks the latest clean appearance instead of staying pinned to the warmup frame."""

        self.fixed_reference_foreground = foreground_token
        self.fixed_reference_background = background_token

    def _gather(self, fixed_token, moving_tokens):
        """Stack fixed slot + moving FIFO into a (1 + moving_capacity, n_patches, feature_dim)
        tensor, padding empty FIFO slots with the -5 sentinel so the calibrator can see them as
        invalid via `compute_patch_similarities`'s ref_mask."""

        slots = [fixed_token, *moving_tokens]
        n_padding = self.moving_capacity - len(moving_tokens)
        if n_padding > 0:
            padding = torch.full(fixed_token.shape, -5.0,
                                 device=fixed_token.device, dtype=fixed_token.dtype)
            slots.extend([padding] * n_padding)
        return torch.stack(slots, dim=0)

    def gather_calibrator_references(self):
        """Return (foreground, background) reference stacks, each (1 + moving_capacity, n_patches,
        feature_dim) — matching the offline pipeline's separate fg/bg reference sets."""

        foreground = self._gather(self.fixed_reference_foreground, self.moving_reference_foregrounds)
        background = self._gather(self.fixed_reference_background, self.moving_reference_backgrounds)
        return foreground, background

    def reset_memory(self):
        self.reference_encodings = deque([])
        self.reference_pointers = deque([])

        self.previous_encodings = deque([], maxlen=self.max_memory_history)
        self.previous_pointers = deque([], maxlen=self.max_pointer_history)

        self.fixed_reference_foreground = None
        self.fixed_reference_background = None
        self.moving_reference_foregrounds = deque([], maxlen=self.moving_capacity)
        self.moving_reference_backgrounds = deque([], maxlen=self.moving_capacity)

    def initialize_references(self, model, detection_data: DetectionData):
        """Bootstrap SAM 2's video-masking memory from frame 1's GT bbox. All baseline trackers
        (SAMURAI / SAMiTe / SAM2Long / sam_baseline) match this index so the anchor frame is
        the same across the comparison. Returns the raw mask SAM 2's mask decoder produces for
        the init prompt — `initialize_calibrator_anchor` consumes it to seed the calibrator
        anchor without a second SAM 2 forward pass."""

        reference = detection_data.frames[1]
        bbox_norm = convert_bbox(detection_data.bboxes_norm[1])

        bgr_reference = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
        init_encoded, _, _ = model.encode_image(bgr_reference)
        init_mask, encoding, pointer = model.initialize_video_masking(init_encoded, bbox_norm)

        self.update_reference(pointer, encoding)
        return init_mask

    def initialize_calibrator_anchor(self, model, detection_data: DetectionData, init_mask):
        """Seed the calibrator anchor from SAM 2's initialization mask (the same mask SAM 2's
        decoder produces in `initialize_references` while bootstrapping video masking). The
        moving FIFO starts empty so the warmup / gated loop can begin at frame 0."""

        frame = detection_data.frames[1]
        mask = (init_mask > 0.0).to(torch.float64).cpu().numpy().squeeze()

        foreground, background = model.extract_patch_tokens(
            *model.extract_crops(frame[np.newaxis], mask[np.newaxis]))
        self.seed_calibrator_references(foreground[0], background[0], [], [])

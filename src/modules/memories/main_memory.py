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

    Also owns the calibrator's reference set, which is now a SINGLE fixed anchor —
    the moving FIFO has been removed from the pipeline."""

    max_memory_history: int | None = None
    max_pointer_history: int | None = None

    reference_encodings: deque | None = None
    reference_pointers: deque | None = None

    previous_encodings: deque | None = None
    previous_pointers: deque | None = None

    # Calibrator references keep BOTH foreground and background tokens (anchor only)
    fixed_reference_foreground: torch.Tensor | None = None
    fixed_reference_background: torch.Tensor | None = None

    def update_reference(self, pointer, encoding):
        self.reference_encodings.appendleft(encoding)
        self.reference_pointers.appendleft(pointer)

    def update_memory(self, pointer, encoding):
        self.previous_encodings.appendleft(encoding)
        self.previous_pointers.appendleft(pointer)

    def seed_calibrator_anchor(self, fixed_foreground, fixed_background):
        """Pin the calibrator's single fixed anchor (fg + bg tokens)."""

        self.fixed_reference_foreground = fixed_foreground
        self.fixed_reference_background = fixed_background

    def gather_calibrator_references(self):
        """Return (foreground, background) reference stacks, each shaped
        `(1, n_patches, feature_dim)` so `compute_patch_similarities`' einsum still
        sees a leading reference dimension. Anchor-only — no FIFO concatenation."""

        foreground = self.fixed_reference_foreground.unsqueeze(0)
        background = self.fixed_reference_background.unsqueeze(0)
        return foreground, background

    def reset_memory(self):
        self.reference_encodings = deque([])
        self.reference_pointers = deque([])

        self.previous_encodings = deque([], maxlen=self.max_memory_history)
        self.previous_pointers = deque([], maxlen=self.max_pointer_history)

        self.fixed_reference_foreground = None
        self.fixed_reference_background = None

    def initialize_references(self, model, detection_data: DetectionData, anchor_index: int = 1):
        """Bootstrap SAM 2's video-masking memory from the chosen anchor's GT bbox.

        `anchor_index` selects which position in `detection_data` holds the anchor:
          - `1` (default): used during DATASET CREATION, where `create_anchor_dataset.py`
            puts the chosen anchor at sliced index 1 with a pre-anchor warmup frame at
            index 0.
          - `0`: used at DEPLOY TIME, where the saved trajectory pickle's
            `frame_indices[0]` IS the chosen anchor (the warmup frame was dropped
            post-tracking before saving).

        Returns the raw mask SAM 2's mask decoder produces for the init prompt —
        `initialize_calibrator_anchor` consumes it to seed the calibrator anchor
        without a second SAM 2 forward pass.

        Also sets `model.anchor_amodal_pixels` from the chosen anchor's amodal bbox so
        every subsequent `extract_crops` call sizes its crop from
        anchor amodal × (1 + 2 × pad_ratio) — SiamFC/STARK-style fixed search region."""

        reference = detection_data.frames[anchor_index]
        bbox_norm = convert_bbox(detection_data.bboxes_norm[anchor_index])
        model.set_anchor_amodal_from_normalized(detection_data.amodal_norm[anchor_index], reference.shape[:2])

        bgr_reference = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
        init_encoded, _, _ = model.encode_image(bgr_reference)
        init_mask, encoding, pointer = model.initialize_video_masking(init_encoded, bbox_norm)

        self.update_reference(pointer, encoding)
        return init_mask

    def initialize_calibrator_anchor(self, model, detection_data: DetectionData, init_mask,
                                                            anchor_index: int = 1):
        """Seed the calibrator anchor from SAM 2's initialization mask.

        `anchor_index` selects the frame this method extracts patches from. It
        MUST match the index used by `initialize_references` for the same call —
        otherwise `init_mask` (computed FOR `frames[anchor_index_init]`) is applied
        to a DIFFERENT frame here, and the patches are extracted from positions
        the mask doesn't correspond to. Defaults to 1 to match the dataset-creation
        warmup convention; deploy trackers pass `anchor_index=0`."""

        frame = detection_data.frames[anchor_index]
        mask = (init_mask > 0.0).to(torch.float64).cpu().numpy().squeeze()

        foreground, background = model.extract_patch_tokens(
            *model.extract_crops(frame[np.newaxis], mask[np.newaxis]))
        self.seed_calibrator_anchor(foreground[0], background[0])

import torch
import numpy as np

from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel
from src.utils.load_bboxes import convert_bbox


@dataclass
class SAMBaseline:
    """Baseline module for video object segmentation with SAM 2."""

    model: SamaraHieraModel | None = None
    iou_threshold: float | None = None
    main_memory: MainMemory | None = None
    label_mask_iou: bool = True

    # Memory-bank corruption (claim_2): identical injection to the oracles -- at each commit, with probability
    # `corruption_p`, a CLEAN nearby distractor is written into the bank instead of the target.
    corruption_p: float = 0.0
    corruption_boxes: list | None = None
    corruption_seed: int = 0

    def __post_init__(self):
        """Load and initialize the SAM 2 model with quantization."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    def commit_corruption(self, idx, image_features, rng):
        """With probability `corruption_p`, box-prompt the nearest distractor and push THAT into the memory bank.
        Runs on EVERY frame, independent of `should_commit` -- same injection the oracles use, so all arms face
        identical corruption events."""

        if self.corruption_p <= 0.0 or self.corruption_boxes is None:
            return False
        box = self.corruption_boxes[idx] if idx < len(self.corruption_boxes) else None
        if box is None or float(rng.random()) >= self.corruption_p:
            return False
        _, encoding, pointer = self.model.initialize_video_masking(
            image_features, convert_bbox(np.asarray(box, dtype=np.float32)))
        self.main_memory.update_memory(pointer, encoding)
        return True

    def should_commit(self, object_scores, iou_scores, chosen_mask, frame):
        """Whether to write this frame into the memory bank. Baseline gate = SAM's own confidence.
        Subclasses may add an identity check (e.g. a Perception-Encoder gate)."""
        
        return object_scores > 0.5 and iou_scores > self.iou_threshold

    def predict_masks(self, detection_data: DetectionData):
        """Predict the masks of the target object with the baseline SAM 2."""

        # Reset and initialize the memory bank with the new target (the anchor, same as the oracle --
        # the clip now starts AT the anchor, so it is at index 0).
        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data, anchor_index=0)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)

        # Storage for the predicted IoU and occlusion scores
        self.object_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)
        self.update_memory = torch.zeros(n_frames, dtype=torch.int)
        self.object_pointers = torch.zeros((n_frames, 256), dtype=torch.float64)

        self.corrupted_frames = []                                        # frames where a distractor was committed
        corruption_rng = np.random.default_rng(self.corruption_seed)

        # Optional per-frame image-embedding cache (target-independent), shared across rollouts over the same frames.
        cache = getattr(self, "frame_cache", None)

        for idx, current_frame in enumerate(detection_data.frames):
            reuse = [e.to(self.device) for e in cache[idx]] if cache is not None and idx in cache else None
            (chosen_mask, pointer, encoding, object_scores, iou_scores, _, _, image_features) = self.model.select_best_mask(
                main_memory = self.main_memory,
                current_frame = current_frame,
                encoded_image_features_list = reuse)

            if cache is not None and idx not in cache:
                cache[idx] = [e.detach().cpu() for e in image_features]

            # Update the memory bank with the embeddings and store the mask
            if self.should_commit(object_scores, iou_scores, chosen_mask, current_frame):
                self.main_memory.update_memory(pointer, encoding)
                self.update_memory[idx] = 1

            # Corruption is independent of the gate above: it fires on EVERY frame, occluded or visible, so all
            # arms face the same injected errors regardless of where each one chooses to commit.
            if self.commit_corruption(idx, image_features, corruption_rng):
                self.corrupted_frames.append(idx)


            self.predicted_masks[idx] = chosen_mask
            self.object_scores[idx] = object_scores
            self.iou_scores[idx] = iou_scores
            self.object_pointers[idx] = pointer.squeeze().to(torch.float32).cpu()

        return self.predicted_masks
    
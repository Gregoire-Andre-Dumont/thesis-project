import numpy as np
import torch

from dataclasses import dataclass
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel
from src.utils.load_bboxes import convert_bbox


@dataclass
class MemoryOracle:
    """SAM 2 VOS that commits SAM 2's own predicted mask to the memory bank whenever the prediction is verified
    good against the ground truth.

    The commit gate is mask IoU between the predicted mask and a pseudo-GT mask (SAM box-prompted on the GT box);
    the mask IoU is only scored on visible frames, so occluded frames never commit.

    Corruption (optional): if `corruption_p` > 0 the caller sets a per-frame `corruption_boxes` list on the
    instance (None before the first occlusion). From then on, with probability `corruption_p`, a nearby distractor
    (box-prompted from `corruption_boxes[i]`) is committed to the bank instead of the target -- and, once
    triggered, on the next frame too -- poisoning the bank. `committed_frames` / `committed_masks` record what was
    written."""

    iou_threshold: float | None = None
    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None
    corruption_p: float = 0.0

    def __post_init__(self):
        """Load the SAM 2 model onto the GPU in bfloat16."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

    def predict_masks(self, detection_data: DetectionData):
        """Roll SAM 2 over the sequence, committing its predicted mask (or a distractor when corrupted) to memory."""

        self.main_memory.reset_memory()
        self.main_memory.initialize_references(self.model, detection_data)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.iou_scores = torch.zeros(n_frames, dtype=torch.float64)         # SAM 2's own predicted IoU per frame
        self.mask_iou_scores = torch.zeros(n_frames, dtype=torch.float64)    # predicted-vs-GT mask IoU on visible frames
        self.committed_frames = []                                           # frame indices written to the bank
        self.committed_masks = []                                            # mask committed at each (prediction, or distractor when corrupted)

        rng = np.random.default_rng(0)
        corrupt_burst = 0      

        corruption_boxes = getattr(self, "corruption_boxes", None)       
        cache = getattr(self, "frame_cache", None)

        for current_idx, current_frame in enumerate(detection_data.frames):
            reuse = [e.to(self.device) for e in cache[current_idx]] \
                if cache is not None and current_idx in cache else None
            (chosen_mask, pointer, encoding, _object_score, iou_score,
             _, _, image_features) = self.model.select_best_mask(
                main_memory=self.main_memory,
                current_frame=current_frame,
                encoded_image_features_list=reuse)
            if cache is not None and current_idx not in cache:
                cache[current_idx] = [e.detach().cpu() for e in image_features]

            self.predicted_masks[current_idx] = chosen_mask
            self.iou_scores[current_idx] = iou_score

            # Pseudo-GT mask IoU label, reusing the tracker's image encoding (occluded/boxless frames stay 0).
            bboxes_norm = detection_data.bboxes_norm[current_idx]
            if detection_data.occlusions[current_idx] <= 0.5 and float(bboxes_norm[2]) != 0.0:
                self.mask_iou_scores[current_idx] = self._mask_iou(bboxes_norm, chosen_mask, image_features)

            distractor_box = corruption_boxes[current_idx] if corruption_boxes is not None else None
            quality = float(self.mask_iou_scores[current_idx])

            if (self.corruption_p > 0.0 and distractor_box is not None and (corrupt_burst > 0 or rng.random() < self.corruption_p)):

                corrupt_burst = corrupt_burst - 1 if corrupt_burst > 0 else 1
                distractor_logits, encoding_c, pointer_c = self.model.initialize_video_masking(
                    image_features, convert_bbox(np.asarray(distractor_box, dtype=np.float32)))

                self.main_memory.update_memory(pointer_c, encoding_c)
                self.committed_frames.append(current_idx)
                self.committed_masks.append((distractor_logits.squeeze() > 0.0).to(torch.float64).cpu().numpy())
                
            elif quality > self.iou_threshold and detection_data.occlusions[current_idx] <= 0.5:
                corrupt_burst = 0
                self.main_memory.update_memory(pointer, encoding)
                self.committed_frames.append(current_idx)
                self.committed_masks.append(self.predicted_masks[current_idx].numpy().copy())

        return self.predicted_masks

    def _mask_iou(self, gt_xywh_norm, chosen_mask, image_features):
        """Pseudo-GT mask IoU: box-prompt the GT box (reusing the tracking encoding, no re-encode) and IoU it
        against the predicted mask."""

        mask, _, _ = self.model.initialize_video_masking(image_features, convert_bbox(np.asarray(gt_xywh_norm, dtype=np.float32)))
        target = (mask.squeeze() > 0.0).cpu().numpy()
        predicted = chosen_mask.numpy() > 0
        union = (predicted | target).sum()
        return float((predicted & target).sum() / union) if union > 0 else 0.0

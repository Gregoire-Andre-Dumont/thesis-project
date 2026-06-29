import torch
import hydra
from omegaconf import OmegaConf

from dataclasses import dataclass, field
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.samara_hiera_model import SamaraHieraModel


@dataclass
class SamaraFixed:
    """Calibrated SAM 2 tracker for the CNN-Fixed calibrator: no warmup phase, no moving FIFO.
    The calibrator only consumes the anchor reference (frame 0's features), so commits are
    gated from frame 1 onward by `samara_iou > iou_threshold AND iou_score > pred_iou_threshold`.
    Frame 0 is processed with a standard SAM 2 step to obtain the initial mask + memory entry,
    after which the anchor features are extracted from that mask."""

    iou_threshold: float = 0.5
    pred_iou_threshold: float = -1.0
    trainer_config_path: str | None = None

    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None

    _trajectory_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Move SAM 2 to GPU and load the calibrator from `trainer_config_path`
        via epochalyst's cache."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)

        if self.trainer_config_path is not None:
            trainer_config = OmegaConf.load(self.trainer_config_path)
            trainer = hydra.utils.instantiate(trainer_config.main_trainer)
            trainer._load_model()
            self.model.controller = trainer.model
            self.model.eval()

    @torch.inference_mode()
    def predict_masks(self, detection_data: DetectionData):
        """Predict the masks of the target object with the calibrated tracker (no warmup)."""

        self.main_memory.reset_memory()
        init_mask = self.main_memory.initialize_references(self.model, detection_data, anchor_index=0)
        self.main_memory.initialize_calibrator_anchor(self.model, detection_data, init_mask, anchor_index=0)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.predicted_iou_trace = [float("nan")] * n_frames

        # Gated loop from frame 0 — calibrator scores against the SAM 2 init mask as the anchor.
        for current_idx in range(n_frames):
            reference_foreground, reference_background = self.main_memory.gather_calibrator_references()
            result = self.model.select_best_mask_gated(
                current_frame=detection_data.frames[current_idx],
                main_memory=self.main_memory,
                reference_foreground=reference_foreground,
                reference_background=reference_background)

            self.predicted_masks[current_idx] = result["chosen_mask"]
            self.predicted_iou_trace[current_idx] = result["samara_iou"]

            if (result["samara_iou"] > self.iou_threshold and float(result["iou_score"]) > self.pred_iou_threshold):
                self.main_memory.update_memory(result["pointer"], result["encoding"])

        return self.predicted_masks

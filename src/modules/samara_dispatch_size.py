import torch
import hydra
from omegaconf import OmegaConf

from dataclasses import dataclass, field
from src.typing.detection_data import DetectionData
from src.modules.memories.main_memory import MainMemory
from src.modules.models.samara_hiera_model import SamaraHieraModel


SAM_RESIZE = 1024


@dataclass
class SamaraDispatchSize:
    """Dispatch tracker that swaps the memory-commit gate based on the anchor's size:

      - SMALL anchor (visible-bbox area at 1024-resize < `small_anchor_max_area`) →
        gate by the SAMARA calibrator's score (`samara_iou > samara_iou_threshold`).
      - LARGE anchor → gate by SAM 2's predicted IoU (`iou_score > pred_iou_threshold`),
        same logic as `sam_baseline`.

    The size decision is made ONCE per trajectory at init (from `detection_data.frames[0]`
    + `detection_data.bboxes_norm[0]`), then applied uniformly to every per-frame commit
    decision. Anchor-aware initialization (`set_amodal_anchor`) runs unchanged."""

    samara_iou_threshold: float = 0.5
    pred_iou_threshold: float = 0.3
    small_anchor_max_area: float = 48 * 48        # NATIVE px²; visible bbox at anchor
    trainer_config_path: str | None = None

    model: SamaraHieraModel | None = None
    main_memory: MainMemory | None = None

    _trajectory_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.main_memory.moving_capacity = 0

        if self.trainer_config_path is not None:
            trainer_config = OmegaConf.load(self.trainer_config_path)
            trainer = hydra.utils.instantiate(trainer_config.main_trainer)
            trainer._load_model()
            self.model.controller = trainer.model
            self.model.eval()

    @torch.inference_mode()
    def predict_masks(self, detection_data: DetectionData):
        frame_height, frame_width = detection_data.frames[0].shape[:2]
        self.model.set_amodal_anchor(detection_data.amodal_norm[0],
                                       frame_height=frame_height, frame_width=frame_width)

        # Decide gate strategy from the anchor visible-bbox area in NATIVE pixels.
        _, _, w_norm, h_norm = detection_data.bboxes_norm[0]
        anchor_area = (w_norm * frame_width) * (h_norm * frame_height)
        self.use_samara_gate = anchor_area < self.small_anchor_max_area

        self.main_memory.reset_memory()
        init_mask = self.main_memory.initialize_references(self.model, detection_data)
        self.main_memory.initialize_calibrator_anchor(self.model, detection_data, init_mask)

        n_frames = detection_data.frames.shape[0]
        self.predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        self.predicted_iou_trace = [float("nan")] * n_frames

        for current_idx in range(n_frames):
            reference_foreground, reference_background = self.main_memory.gather_calibrator_references()
            result = self.model.select_best_mask_gated(
                current_frame=detection_data.frames[current_idx],
                main_memory=self.main_memory,
                reference_foreground=reference_foreground,
                reference_background=reference_background)

            self.predicted_masks[current_idx] = result["chosen_mask"]
            self.predicted_iou_trace[current_idx] = result["samara_iou"]

            if self.use_samara_gate:
                gate_passes = result["samara_iou"] > self.samara_iou_threshold
            else:
                gate_passes = float(result["iou_score"]) > self.pred_iou_threshold
            if gate_passes:
                self.main_memory.update_memory(result["pointer"], result["encoding"])

        return self.predicted_masks

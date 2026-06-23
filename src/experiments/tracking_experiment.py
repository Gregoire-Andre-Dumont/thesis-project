import torch
import cv2
import numpy as np
import ipdb
import torch.nn.functional as F
from dataclasses import dataclass, field

@dataclass
class TrackingExperiment:
    """Manages the results and visualizations of the test tracking"""

    video_path: str | None = None

    success: list = field(default_factory=list)
    true_iou: list = field(default_factory=list)
    true_iou_per_frame: list = field(default_factory=list)
    pred_iou: list = field(default_factory=list)
    true_visibility: list = field(default_factory=list)
    true_occlusions: list = field(default_factory=list)
    pred_occlusions: list = field(default_factory=list)
    video_names: list = field(default_factory=list)
    person_ids: list = field(default_factory=list)
    coverage: list = field(default_factory=list)
    auc_scores: list = field(default_factory=list)

"""SENTRY baseline: consistency-validated memory writes on top of SAM 2.

SENTRY (Alshehri et al., ECCV 2026 -- SAM2-Enhanced Neighbor-Aware and Temporally Reasoned Memory) is a
training-free refine-before-write module. Each frame it builds a candidate set from the decoder's masks
(optionally plus AMG proposals and a Kalman motion prior), re-segments the previous tau frames backwards
from each candidate to get a short tracklet, and scores each tracklet by mean box IoU against the recent
trajectory of the target. The most temporally consistent candidate is written, not the most confident one.

This is a THIN WRAPPER around the authors' released implementation, not a reimplementation: the released
package is imported from `sentry_root` and driven through its public `SENTRYTracker` API, so the policy and
its published defaults are the authors' own. The only thing here is the adaptation to this repo's
conventions -- reading the clip's frames, prompting with the anchor box, and returning masks in the
(frames, 256, 256) form every other baseline returns.

The release ships its own SAM 2 backend, which builds the official `sam2` video predictor. That predictor
takes PIL frames one at a time rather than a directory of JPEGs, which is why the loop below decodes with
cv2 and hands frames over individually.

ANCHOR CONVENTION. `detection_data` frame 0 is the anchor, matching SAMARA, sam_baseline and the other
baselines: SENTRY is initialised there with the ground-truth box and tracks forward from it.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.typing.detection_data import DetectionData


@dataclass
class Sentry:
    """SENTRY's tracker over one clip, exposing this repo's `predict_masks` interface."""

    checkpoint: str = "tm/sam_large.pt"
    # A config name resolved by the official `sam2` package's own hydra search path, NOT a path in this
    # repo: the release builds its predictor with `sam2.build_sam.build_sam2_video_predictor`.
    model_config: str = "configs/sam2/sam2_hiera_l.yaml"
    # The authors' released checkout. `policy` names one of its `configs/sentry/*.yaml` files, so the
    # thresholds stay the published ones rather than being restated here.
    sentry_root: str = "/workspace/SENTRY"
    policy: str = "default"
    device: str = "cuda:0"

    def __post_init__(self):
        """Put the release on the import path and build its tracker once, for reuse across clips."""

        root = Path(self.sentry_root)
        source = root / "src"
        if not source.is_dir():
            raise FileNotFoundError(
                f"SENTRY release not found at {root}. Clone https://github.com/HamadYA/SENTRY and point "
                f"`sentry_root` at it.")

        if str(source) not in sys.path:
            sys.path.insert(0, str(source))

        from sentry_tracking import SENTRYConfig, SENTRYTracker
        from sentry_tracking.backends.sam2 import SAM2Backend

        policy_path = root / "configs" / "sentry" / f"{self.policy}.yaml"
        if not policy_path.exists():
            raise FileNotFoundError(f"no SENTRY policy {policy_path}")

        backend = self._build_backend(SAM2Backend)
        self.config = SENTRYConfig.from_yaml(policy_path)
        self.tracker = SENTRYTracker(backend, self.config)

    def _build_backend(self, backend_class):
        """Build the release's SAM 2 backend under sam2's OWN hydra config search path.

        `build_sam2_video_predictor` composes its model config through hydra, expecting the `sam2` package's
        config module to be registered. Our experiment scripts run under `@hydra.main`, which has already
        installed this repo's `conf/` as the one and only search path, so sam2's config name does not
        resolve and the build fails with MissingConfigException.

        The global hydra state is therefore swapped for sam2's while the predictor is built and cleared
        afterwards. Safe here because the experiment's own config is a plain resolved DictConfig by this
        point -- `hydra.utils.instantiate` does not read the global state."""

        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        initialize_config_module("sam2", version_base="1.2")
        try:
            return backend_class(checkpoint=self.checkpoint, model_config=self.model_config,
                                 device=self.device)
        finally:
            GlobalHydra.instance().clear()

    def _frames(self, detection_data: DetectionData):
        """The clip's frames as PIL images, in `frame_indices` order -- what the release's backend takes."""

        capture = cv2.VideoCapture(detection_data.video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        for index in detection_data.frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not read frame {index} of {detection_data.video_path}")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        capture.release()
        return frames, width, height

    def _anchor_box(self, detection_data: DetectionData, width, height):
        """The anchor's ground-truth box in pixel XYWH.

        The release's `_estimate_mask_from_box` builds its corners as (x, y, x + w, y + h), so it takes
        xywh -- NOT the xyxy the other baselines in this repo pass to SAM's box prompt. Passing xyxy here
        prompts a box running from the target's top-left to (x_max + width, y_max + height) and the tracker
        never finds the target at all."""

        x_min, y_min, box_width, box_height = detection_data.bboxes_norm[0]
        return np.array([width * x_min, height * y_min,
                         width * box_width, height * box_height], dtype=np.float32)

    def predict_masks(self, detection_data: DetectionData):
        """Roll SENTRY over the clip and return its per-frame masks at 256x256."""

        frames, width, height = self._frames(detection_data)
        n_frames = len(frames)
        predicted_masks = torch.zeros((n_frames, 256, 256), dtype=torch.float64)

        # Kept so a rollout can be read back the way the other arms' diagnostics are: which branch of
        # SENTRY's three-tier selection produced each frame, and where it declared a severe failure.
        self.sources = []
        self.severe_failures = []

        result = self.tracker.initialize(frames[0], self._anchor_box(detection_data, width, height))
        for index, frame in enumerate(frames):
            if index > 0:
                result = self.tracker.track(frame)

            self.sources.append(getattr(result, "source", None))
            self.severe_failures.append(getattr(result, "severe_failure", None))
            predicted_masks[index] = self._resize(result.mask)

        # SENTRY refines WHICH mask is written, not whether to write: the selected mask goes into memory
        # under SAM 2's default schedule. The flag is all-True; `self.sources` records which branch of its
        # three-tier selection produced each frame, which is where its decision actually shows up.
        self.update_memory = torch.ones(n_frames, dtype=torch.bool)

        return predicted_masks

    @staticmethod
    def _resize(mask):
        """One frame's mask as a 256x256 binary map; an empty prediction stays all zeros."""

        if mask is None:
            return torch.zeros((256, 256), dtype=torch.float64)

        tensor = torch.as_tensor(np.asarray(mask), dtype=torch.float32)[None, None]
        resized = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
        return (resized.squeeze() > 0.5).to(torch.float64)

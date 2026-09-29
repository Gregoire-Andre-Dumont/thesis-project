"""SAMURAI-3: SAMURAI's motion-and-memory strategy re-implemented on the SAM 3 backbone.

From SAM3_Tracking_Zoo, which is the SAM 3 edition of the same collection that supplies this repo's SAM 2
forks -- same layout (`DAM4SAM/`, `SAMITE/`, `SAMURAI/`, `unified/`, `utils/`), with `sam3` where the other
has `sam2`. These are THIRD-PARTY re-implementations, not the SAMURAI authors' own release, which is worth
stating wherever the numbers appear.

Two things differ from the SAM 2 forks and shape this wrapper:

  * The API is STREAMING -- `initialize(image, mask, bbox)` then `track(image)` per frame -- rather than
    SAM 2's `init_state` over a whole video plus a `propagate_in_video` generator. That removes the need
    for the decord frame-window patch every SAM 2 fork required.
  * Their `track()` discards the per-frame confidence: the generator yields
    `(frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)` and their loop binds the last to
    `_`. Without it the arm cannot be swept for VOT-LT, so `ScoredTracker` below keeps it.
"""

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.typing.detection_data import DetectionData

ZOO_ROOT = "/workspace/SAM3_Tracking_Zoo"

# Precision parity with the SAM 2 arms, which all enter this same context at import. Without it this arm
# runs fp32 while every arm it is tabled against runs bf16, so a difference in the table could be read off
# the precision rather than the backbone. The zoo's own entry points wrap their `main` in
# `torch.cuda.amp.autocast()`, i.e. fp16; bf16 matches this repo and has the wider exponent range.
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


def _stub_vot():
    """Satisfy `utils.utils`'s import of the VOT toolkit without installing it.

    That module imports `vot.region.RegionType` and `vot.region.raster.calculate_overlaps` at module level
    purely for its own evaluation helpers; the only names this wrapper needs from it are
    `keep_largest_component` and `determine_tracker`, which touch neither. Installing the toolkit to
    satisfy a dead import risks pulling conflicting dependencies into an environment that is already
    juggling several vendored `sam2`/`sam3` packages, so the unused surface is stubbed instead. If
    anything ever calls through these, it raises rather than silently returning a wrong overlap."""

    if "vot" in sys.modules:
        return

    def unavailable(*args, **kwargs):
        raise RuntimeError("vot toolkit is stubbed out in this wrapper; it is not needed for tracking")

    vot = types.ModuleType("vot")
    region = types.ModuleType("vot.region")
    raster = types.ModuleType("vot.region.raster")
    region.RegionType = type("RegionType", (), {"RECTANGLE": "rectangle", "MASK": "mask"})
    raster.calculate_overlaps = unavailable
    region.raster = raster
    vot.region = region
    sys.modules.update({"vot": vot, "vot.region": region, "vot.region.raster": raster})


def _load_zoo(zoo_root, fork):
    """Import the fork's own `tracker` module, with its directory first on the path.

    The forks each ship a module called `tracker`, so the ORDER matters: whichever directory leads
    `sys.path` decides which one is imported. One fork per process, as with the SAM 2 forks."""

    _stub_vot()
    for path in (str(Path(zoo_root) / fork), zoo_root):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    from tracker import SAMTracker
    return SAMTracker


@dataclass
class Samurai3:
    """SAMURAI on SAM 3, anchored on frame 0 like every other arm so records join on (video, person)."""

    zoo_root: str = ZOO_ROOT
    fork: str = "SAMURAI"
    tracker_name: str = "sam3"

    def __post_init__(self):
        base = _load_zoo(self.zoo_root, self.fork)

        class ScoredTracker(base):
            """`track()` that also returns the object score the upstream loop throws away."""

            @torch.inference_mode()
            def track_scored(self, image, init=False):
                """`init=True` reproduces upstream's first-frame call, which does NOT advance the frame
                index: the prompted frame still has to be run through the tracker to get its refined mask.
                Taking `initialize`'s return value instead gives the raw box-prompt mask -- on the anchor
                frame that scored IoU 0.324 where the refined mask reaches 0.8."""

                prepared = self._prepare_image(image)
                if not init:
                    self.frame_index += 1
                self.inference_state["num_frames"] += 1
                self.inference_state["images"][self.frame_index] = prepared

                mask, score = None, float("nan")
                run_preflight = self._needs_preflight
                for _, _, _, video_res_masks, obj_scores in self.predictor.propagate_in_video(
                        inference_state=self.inference_state, start_frame_idx=self.frame_index,
                        max_frame_num_to_track=0, reverse=False, propagate_preflight=run_preflight):
                    mask = (video_res_masks[0, 0] > 0).float().cpu().numpy().astype(np.uint8)
                    if obj_scores is not None:
                        score = float(torch.as_tensor(obj_scores).reshape(-1)[0])

                self._needs_preflight = False
                self.inference_state["images"].pop(self.frame_index, None)
                if run_preflight:
                    self.inference_state["images"].pop(0, None)
                return mask, score

        self.tracker = ScoredTracker(tracker_name=self.tracker_name)

    def predict_masks(self, detection_data: DetectionData):
        """Roll the clip: prompt with the anchor box on frame 0, then stream the rest."""

        from PIL import Image

        frames = detection_data.frames
        boxes = detection_data.bboxes_norm
        n_frames = len(frames)

        first = np.asarray(frames[0])
        height, width = first.shape[:2]
        # XYWH in pixels, NOT xyxy: their `initialize` builds the box as
        #   [x / W, y / H, (x + bbox[2]) / W, (y + bbox[3]) / H]
        # so bbox[2:] are read as width and height. Passing xyxy makes the box about twice as wide as the
        # target and cost 0.4 IoU on the prompted frame.
        x, y, w, h = boxes[0]
        box = np.array([x * width, y * height, w * width, h * height], dtype=np.float32)

        predicted = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        scores = torch.full((n_frames, 2), float("nan"), dtype=torch.float64)

        self.tracker.initialize(Image.fromarray(first), None, bbox=box)
        mask, score = self.tracker.track_scored(Image.fromarray(first), init=True)
        masks = [mask]
        scores[0, 1] = score

        for index in range(1, n_frames):
            mask, score = self.tracker.track_scored(Image.fromarray(np.asarray(frames[index])))
            masks.append(mask)
            scores[index, 1] = score

        for index, mask in enumerate(masks):
            if mask is None:
                continue
            tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
            resized = F.interpolate(tensor, size=(256, 256), mode="nearest")
            predicted[index] = (resized.squeeze() > 0).to(torch.float64)

        # SAM 3 in single-object mode inherits SAM 2's policy: every frame is written to memory, so there
        # is no commit decision to read back. Column 0 (the IoU token) stays NaN -- this fork exposes only
        # the object score -- and `confidence_signals` reads both columns.
        self.update_memory = torch.ones(n_frames, dtype=torch.bool)
        self.memory_scores = scores
        return predicted

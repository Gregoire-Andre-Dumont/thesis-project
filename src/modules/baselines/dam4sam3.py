"""DAM4SAM on SAM 3: the distractor-aware memory, ported to the newer backbone.

From SAM3_Tracking_Zoo, like `samurai3`. Two things make this wrapper different from that one, and both
come from WHERE each method's contribution lives:

  * SAMURAI's motion model sits inside the predictor, so `samurai3` can drive `propagate_in_video` itself
    and still get SAMURAI behaviour. DAM4SAM's contribution -- deciding when a frame goes into the
    distractor-resolving memory (`add_to_drm`) -- sits in `track()`, AFTER propagation returns. Calling
    propagate directly would silently produce plain SAM 3 with a DAM4SAM label on it. So this wrapper
    calls their `track()` unmodified and captures the object score by wrapping the generator instead.

  * `track()` really does call the VOT toolkit (`Mask`, `RegionType`, `calculate_overlaps`) on every frame
    where alternative masks survive filtering. `samurai3._stub_vot` installs stubs that RAISE, which is
    correct there -- nothing calls them -- and would crash here. The shim below implements the three
    entry points for real, over rectangles, which is the only geometry this code path uses.
"""

import sys
import types
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.typing.detection_data import DetectionData
from src.modules.baselines.samurai3 import ZOO_ROOT, _load_zoo

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


# ---------------------------------------------------------------------------------------
# the VOT surface DAM4SAM actually uses
# ---------------------------------------------------------------------------------------

def _install_vot_shim():
    """Provide `Mask`, `RegionType` and `calculate_overlaps` with working implementations.

    DAM4SAM's distractor test converts the chosen mask and each alternative mask to bounding boxes and
    compares them by IoU; a box overlap below 0.7 means the alternatives disagree enough with the chosen
    mask that a distractor is probably present, and the frame is added to the distractor memory. Only
    rectangle geometry is ever exercised, so that is what this implements -- faithfully, rather than
    approximately, because the 0.7 threshold is a published constant and a wrong IoU would move it.

    Replaces `samurai3`'s raising stubs if those were installed first (one fork per process, but the
    import order is not guaranteed)."""

    class RegionType:
        RECTANGLE = "rectangle"
        MASK = "mask"

    class Rectangle:
        __slots__ = ("x", "y", "width", "height")

        def __init__(self, x, y, width, height):
            self.x, self.y, self.width, self.height = float(x), float(y), float(width), float(height)

        def bounds(self):
            return self.x, self.y, self.x + self.width, self.y + self.height

    class Mask:
        """A binary mask region. `rasterize` renders it into the requested bounds."""

        def __init__(self, mask, offset=(0, 0)):
            self._mask = np.asarray(mask).astype(np.uint8)
            self._offset = offset

        def rasterize(self, bounds):
            """`bounds` is (left, top, right, bottom) INCLUSIVE, as the toolkit defines it."""

            left, top, right, bottom = (int(round(value)) for value in bounds)
            height, width = bottom - top + 1, right - left + 1
            output = np.zeros((height, width), dtype=np.uint8)

            source_top = max(top, 0)
            source_left = max(left, 0)
            source_bottom = min(bottom + 1, self._mask.shape[0])
            source_right = min(right + 1, self._mask.shape[1])
            if source_bottom <= source_top or source_right <= source_left:
                return output

            patch = self._mask[source_top:source_bottom, source_left:source_right]
            output[source_top - top:source_top - top + patch.shape[0],
                   source_left - left:source_left - left + patch.shape[1]] = patch
            return output

        def convert(self, region_type):
            if region_type != RegionType.RECTANGLE:
                raise NotImplementedError(f"this shim only converts to RECTANGLE, not {region_type}")
            coordinates = cv2.findNonZero(self._mask)
            if coordinates is None:
                return Rectangle(0, 0, 0, 0)
            x, y, width, height = cv2.boundingRect(coordinates)
            return Rectangle(x, y, width, height)

    def calculate_overlaps(first, second):
        """Pairwise IoU between two equal-length sequences of rectangles."""

        overlaps = []
        for a, b in zip(first, second):
            a_left, a_top, a_right, a_bottom = a.bounds()
            b_left, b_top, b_right, b_bottom = b.bounds()
            width = max(0.0, min(a_right, b_right) - max(a_left, b_left))
            height = max(0.0, min(a_bottom, b_bottom) - max(a_top, b_top))
            intersection = width * height
            union = a.width * a.height + b.width * b.height - intersection
            overlaps.append(intersection / union if union > 0 else 0.0)
        return overlaps

    vot = types.ModuleType("vot")
    region = types.ModuleType("vot.region")
    raster = types.ModuleType("vot.region.raster")
    shapes = types.ModuleType("vot.region.shapes")

    region.RegionType = RegionType
    shapes.Mask, shapes.Rectangle = Mask, Rectangle
    raster.calculate_overlaps = calculate_overlaps
    region.raster, region.shapes = raster, shapes
    sys.modules.update({"vot": vot, "vot.region": region,
                        "vot.region.raster": raster, "vot.region.shapes": shapes})
    vot.region = region


# ---------------------------------------------------------------------------------------
# the arm
# ---------------------------------------------------------------------------------------

@dataclass
class Dam4Sam3:
    """DAM4SAM's distractor-resolving memory on SAM 3, anchored on frame 0 like every other arm."""

    zoo_root: str = ZOO_ROOT
    fork: str = "DAM4SAM"
    tracker_name: str = "sam3"

    def __post_init__(self):
        _install_vot_shim()
        base = _load_zoo(self.zoo_root, self.fork)
        _install_vot_shim()        # `_load_zoo` calls `_stub_vot`, whose raising stubs must not win

        class ScoredTracker(base):
            """Their `track()` verbatim, with the object score captured on the way past.

            `propagate_in_video` is wrapped rather than replaced: the generator's values reach `track()`
            untouched, so every DAM4SAM decision -- including `add_to_drm` -- happens exactly as upstream
            wrote it, and the score is read off as a side effect."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.last_object_score = float("nan")
                inner = self.predictor.propagate_in_video

                def capturing(*call_args, **call_kwargs):
                    for output in inner(*call_args, **call_kwargs):
                        if len(output) >= 5 and output[4] is not None:
                            self.last_object_score = float(torch.as_tensor(output[4]).reshape(-1)[0])
                        yield output

                self.predictor.propagate_in_video = capturing

        self.tracker = ScoredTracker(tracker_name=self.tracker_name)

    def predict_masks(self, detection_data: DetectionData):
        """Roll the clip: prompt with the anchor box on frame 0, then stream the rest."""

        from PIL import Image

        frames = detection_data.frames
        boxes = detection_data.bboxes_norm
        n_frames = len(frames)

        first = np.asarray(frames[0])
        height, width = first.shape[:2]
        # XYWH in pixels, NOT xyxy -- see samurai3 for what passing xyxy costs (0.4 IoU on the anchor).
        x, y, w, h = boxes[0]
        box = np.array([x * width, y * height, w * width, h * height], dtype=np.float32)

        predicted = torch.zeros((n_frames, 256, 256), dtype=torch.float64)
        scores = torch.full((n_frames, 2), float("nan"), dtype=torch.float64)

        self.tracker.initialize(Image.fromarray(first), None, bbox=box)
        masks = []
        for index in range(n_frames):
            # `init=True` on frame 0 reproduces upstream's own first call: it runs the prompted frame
            # through the tracker instead of taking the raw box-prompt mask, without advancing the index.
            image = Image.fromarray(np.asarray(frames[index]))
            output = self.tracker.track(image, init=(index == 0))
            masks.append(output.get("pred_mask"))
            scores[index, 1] = self.tracker.last_object_score

        for index, mask in enumerate(masks):
            if mask is None:
                continue
            tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
            resized = F.interpolate(tensor, size=(256, 256), mode="nearest")
            predicted[index] = (resized.squeeze() > 0).to(torch.float64)

        # DAM4SAM ADDS a second memory rather than withholding from the first, so every frame still
        # enters the ordinary bank. Unlike `samurai3`, this is a property of the method rather than an
        # unread value: the distractor memory is the intervention, and `add_to_drm` is not a refusal.
        self.update_memory = torch.ones(n_frames, dtype=torch.bool)
        self.memory_scores = scores
        return predicted

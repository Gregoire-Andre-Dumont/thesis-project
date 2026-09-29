"""DAM4SAM's video predictor, taught to decode only a trajectory's frames.

The fork ships the STOCK SAM 2 `init_state(video_path, offload_video_to_cpu=..., ...)`, which loads an
entire video. Every other arm in this repo takes `(video_path, frame_indices)` and decodes just the clip
window -- a large speedup on PersonPath's long videos, and the reason a positional `frame_indices` landed
in `offload_video_to_cpu` and blew up on `if not <array>`.

Rather than copy the parent's state construction, this swaps out only the frame loader and lets DAM4SAM
build its own inference state -- so its distractor-memory keys (`adds_in_drm_per_obj` and friends) are
created by its own code and cannot drift out of sync with the upstream fork.
"""

import decord
import torch

from sam2.sam2_video_predictor import SAM2VideoPredictor
import sam2.sam2_video_predictor as predictor_module

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def decode_frames(video_path, frame_indices, image_size, device, offload_to_cpu):
    """The clip's frames only, resized and normalised exactly as `load_video_frames` returns them."""

    decord.bridge.set_bridge("torch")
    video_height, video_width, _ = decord.VideoReader(video_path)[0].shape

    reader = decord.VideoReader(video_path, width=image_size, height=image_size)
    images = reader.get_batch([int(index) for index in frame_indices]).permute(0, 3, 1, 2).float() / 255.0

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    if not offload_to_cpu:
        images, mean, std = images.to(device), mean.to(device), std.to(device)

    return (images - mean) / std, video_height, video_width


class Dam4SamVideoPredictor(SAM2VideoPredictor):
    """SAM 2.1++ with this repo's frame-window calling convention."""

    @torch.inference_mode()
    def init_state(self, video_path, frame_indices, offload_video_to_cpu=False, **kwargs):
        images, video_height, video_width = decode_frames(
            video_path, frame_indices, self.image_size, self.device, offload_video_to_cpu)

        original_loader = predictor_module.load_video_frames
        predictor_module.load_video_frames = lambda **_: (images, video_height, video_width)
        try:
            return super().init_state(video_path, offload_video_to_cpu=offload_video_to_cpu, **kwargs)
        finally:
            predictor_module.load_video_frames = original_loader

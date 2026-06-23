import torch
import torch.nn.functional as F

def resize_masks(object_masks: torch.Tensor, resolution: int):
        """Resize the masks of the target object to the correct size."""

        resized_masks = F.interpolate(
            object_masks.unsqueeze(1).float(),
            mode="bicubic",
            align_corners=False,
            size=(resolution, resolution))

        return resized_masks.squeeze(1).byte()
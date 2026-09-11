"""Connected-component decomposition of SAM 2 mask proposals.

When SAM 2 is uncertain -- typically right after an occlusion with a distractor nearby -- a single proposal can
cover the target AND a second person as two separate blobs. Committing that blob writes the distractor's
appearance into the memory bank. These helpers split a proposal into components so an oracle can keep only the
part it wants.

A person is often legitimately split into several components (occluded by a pole, a railing, another person), so
single components are not enough: every non-empty UNION of components is a candidate too.
"""
import cv2
import numpy as np
import torch

MIN_COMPONENT_AREA = 16      # px @256: ignore specks so the subset enumeration stays small
MAX_COMPONENTS = 4           # subsets of at most this many largest components -> at most 2^4-1 = 15 candidates
SUPPRESSED_LOGIT = -32.0     # dropped pixels become confident background (proposals are LOGITS; 0 is the boundary)


def mask_components(mask_logits, min_area=MIN_COMPONENT_AREA, max_components=MAX_COMPONENTS):
    """Connected components of one proposal's positive region, largest first. Returns a list of boolean
    (H, W) arrays -- empty when the proposal has no component above `min_area`."""

    binary = (mask_logits > 0.0).cpu().numpy().astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)
             if stats[i, cv2.CC_STAT_AREA] >= min_area]
    areas.sort(reverse=True)
    return [labels == index for _, index in areas[:max_components]]


def component_subsets(components):
    """Every non-empty union of `components`, so a target split across blobs can still be recovered whole."""

    subsets = []
    for bits in range(1, 1 << len(components)):
        keep = np.zeros_like(components[0])
        for position, component in enumerate(components):
            if bits >> position & 1:
                keep |= component
        subsets.append(keep)
    return subsets


def filter_logits(mask_logits, keep):
    """Drop everything outside `keep` by pushing it to a confident-background logit -- not 0, which is the
    threshold the memory encoder decides on."""

    keep_tensor = torch.as_tensor(np.ascontiguousarray(keep), device=mask_logits.device)
    return torch.where(keep_tensor, mask_logits, torch.full_like(mask_logits, SUPPRESSED_LOGIT))

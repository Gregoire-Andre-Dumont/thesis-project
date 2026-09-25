import torch

TRACKER_PREFIX = "tracker."
BACKBONE_PREFIX = "detector.backbone."

def merge_tracker_weights(state_dict):
    """The tracker's weights, keyed as a standalone tracker expects them.

    The Perception Encoder trunk is SHARED with the detector and therefore stored under the detector's
    prefix; the tracker's own layers sit under `tracker.`. A tracker built on its own wants both, with
    the prefixes stripped. Everything else -- the detector head, the language tower -- is dropped."""

    tracker_weights = {key[len(TRACKER_PREFIX):]: value for key, value in state_dict.items() if key.startswith(TRACKER_PREFIX)}
    backbone_weights = {key[len(BACKBONE_PREFIX):]: value for key, value in state_dict.items() if key.startswith(BACKBONE_PREFIX)}

    merged_weights = dict(tracker_weights)
    merged_weights.update({f"backbone.{name}": value for name, value in backbone_weights.items()})
    return merged_weights


def build_sam3_video_predictor(memory_selection=False, checkpoint_version="sam3", device="cuda"):
    """A SAM 3 tracker with its weights loaded, ready to prompt with a box on frame 0.

    Raises rather than warns when a weight is missing: a tracker that silently keeps randomly initialised
    layers would still produce masks, and those masks would look like a baseline result."""

    from sam3.model_builder import build_tracker, download_ckpt_from_hf
    predictor = build_tracker(apply_temporal_disambiguation=memory_selection, with_backbone=True)

    checkpoint_path = download_ckpt_from_hf(version=checkpoint_version)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    missing_weights, _ = predictor.load_state_dict(merge_tracker_weights(state_dict), strict=False)
    return predictor.to(device).eval()

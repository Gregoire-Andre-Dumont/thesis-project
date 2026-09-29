"""Build DAM4SAM's predictor with this repo's frame-window `init_state`."""

from sam2.build_sam import build_sam2_video_predictor

TARGET = "sam_2.dam4sam.dam4sam_video_predictor.Dam4SamVideoPredictor"


def build_dam4sam_video_predictor(config_file, checkpoint=None, device="cuda", **kwargs):
    """`build_sam2_video_predictor` with the predictor class swapped for the windowed subclass.

    The override is passed through `hydra_overrides_extra`, which the fork appends AFTER its own
    `++model._target_`, so ours is the one that takes effect."""

    return build_sam2_video_predictor(
        config_file, ckpt_path=checkpoint, device=device,
        hydra_overrides_extra=[f"++model._target_={TARGET}"], **kwargs)

"""Hydra-based builder for SAM2Long, mirroring the SAMURAI / SAMITE pattern. Overrides the
model `_target_` to point at the local subclass that adds the `frame_indices` patch on
`init_state` and the tree-search hyperparameters (`num_pathway`, `iou_thre`, `uncertainty`)."""

import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_sam2long_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    num_pathway=3,
    iou_thre=0.1,
    uncertainty=2,
    **kwargs,
):
    if hydra_overrides_extra is None:
        hydra_overrides_extra = []
    hydra_overrides = [
        "++model._target_=muggled_sam.sam2long.sam2long_video_predictor.SAM2LongVideoPredictor",
        f"++model.num_pathway={num_pathway}",
        f"++model.iou_thre={iou_thre}",
        f"++model.uncertainty={uncertainty}",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = list(hydra_overrides_extra) + [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is None:
        return
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        logging.error(missing_keys)
        raise RuntimeError("missing keys in SAM2Long state_dict")
    if unexpected_keys:
        logging.error(unexpected_keys)
        raise RuntimeError("unexpected keys in SAM2Long state_dict")
    logging.info("Loaded SAM2Long checkpoint successfully")

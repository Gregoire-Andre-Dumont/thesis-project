"""Thin subclass of upstream SAM2Long's `SAM2VideoPredictor` (pip-installed via
`Mark12Ding/SAM2Long`). Patches `init_state` to accept a `frame_indices` array — same shape as
the SAMURAI wrapper — so callers can drop `detection_data.frame_indices[0]` and have the
predictor's local frame 0 align with `detection_data.frames[1]`. Also stores SAM2Long's
tree-search hyperparameters (`num_pathway`, `iou_thre`, `uncertainty`) on the returned state
so `propagate_in_video` picks them up without an extra wrapper call."""

from collections import OrderedDict

import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import load_video_frames


class SAM2LongVideoPredictor(SAM2VideoPredictor):
    def __init__(self, num_pathway=3, iou_thre=0.1, uncertainty=2, **kwargs):
        super().__init__(**kwargs)
        self._sam2long_num_pathway = int(num_pathway)
        self._sam2long_iou_thre = float(iou_thre)
        self._sam2long_uncertainty = float(uncertainty)

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        frame_indices,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    ):
        compute_device = self.device
        # Decode ONLY the trajectory's frames instead of the whole video (big speedup on long clips).
        import decord
        decord.bridge.set_bridge("torch")
        _mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        _std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        video_height, video_width, _ = decord.VideoReader(video_path)[0].shape
        _reader = decord.VideoReader(video_path, width=self.image_size, height=self.image_size)
        images = _reader.get_batch([int(i) for i in frame_indices]).permute(0, 3, 1, 2).float() / 255.0
        if not offload_video_to_cpu:
            images = images.to(compute_device); _mean = _mean.to(compute_device); _std = _std.to(compute_device)
        images = (images - _mean) / _std
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(inference_state["images"])
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = True
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}

        inference_state["num_pathway"] = self._sam2long_num_pathway
        inference_state["iou_thre"] = self._sam2long_iou_thre
        inference_state["uncertainty"] = self._sam2long_uncertainty

        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

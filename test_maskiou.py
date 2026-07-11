import hydra, numpy as np
from hydra import initialize, compose
with initialize(config_path="conf", version_base=None):
    cfg = compose(config_name="create_anchor_dataset")
detection_data = hydra.utils.instantiate(cfg.detection_data)
person_path = hydra.utils.instantiate(cfg.person_path)
tracker = hydra.utils.instantiate(cfg.oracle.tracker)
from create_anchor_dataset import shard_by_video, track_trajectory
from src.experiments.dataset_experiment import DatasetExperiment
_, pairs = shard_by_video(person_path, cfg.shard_index, cfg.num_shards)
print("loaded; testing trajectories", flush=True)
for v,p,anchor in pairs[:6]:
    result = track_trajectory(tracker, detection_data, v, p, anchor, cfg)
    if result is None:
        print(f"{v} p{p}: filtered", flush=True); continue
    meta, frames, masks = result
    DatasetExperiment(video_name=v, person_id=p, features=np.zeros((len(frames),1,32,32,2),np.float16), **meta)
    print(f"{v} p{p}: OK  mask-IoU label mean {meta['iou_scores'].mean():.3f}  box-IoU mean {meta['box_iou'].mean():.3f}  frames {len(frames)}", flush=True)
    break
print("DONE", flush=True)

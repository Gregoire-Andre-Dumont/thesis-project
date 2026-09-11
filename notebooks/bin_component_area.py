"""Bin the connected-component comparison by the anchor's VISIBLE box area (px^2 @1024).

Shows, per area bin, what component filtering did to each oracle: `plain` vs `components` for the memory arm
(components change only what is COMMITTED) and the mask arm (components widen the SELECTION candidate set).
The `fired` columns are the mean number of frames narrowed per clip -- a bin with fired~0 cannot show an effect.
"""
import sys
import json
import pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.typing.person_path import PersonPath, _nearest_index

RESIZE = 1024
NBINS = 4

pp = PersonPath(main_directory="data/person_path", occlusion_ranges=[5, 75], n_experiments=1200,
                n_after_occlusion=30, first_occ_min=5, random_seed=44, min_visible_ratio=0.5,
                min_visible_area=400, max_distractor_overlap=0.5,
                non_targets=["crowd", "person_in_vehicle", "reflection", "person_in_background", "severly_occluded_person"])
anchor_of = {(v, int(p)): int(a) for v, p, a in zip(
    pp.selected_video_names.tolist(), pp.selected_person_ids.tolist(), pp.selected_anchor_video_frames.tolist())}

_cache = {}


def visible_area(video, pid, anchor):
    if video not in _cache:
        _cache[video] = json.load(open(f"data/person_path/visible/{video}.json"))
    visible = _cache[video]
    scale = RESIZE / max(float(visible["metadata"]["resolution"]["width"]),
                         float(visible["metadata"]["resolution"]["height"]))
    ents = sorted([e for e in visible["entities"] if e["id"] == pid], key=lambda e: e["blob"]["frame_idx"])
    if not ents:
        return np.nan
    frames = np.array([e["blob"]["frame_idx"] for e in ents])
    wh = np.array([e["bb"][2:4] for e in ents], dtype=np.float64)
    return float(np.prod(wh[_nearest_index(frames, anchor)]) * scale ** 2)


rows = []
for r in pickle.load(open("data/claim_1/component_compare.pkl", "rb")):
    sam_c, mem_p, mem_c, msk_p, msk_c, mem_n, msk_n, video, person = r
    anchor = anchor_of.get((video, int(person)))
    if anchor is None:
        continue
    area = visible_area(video, int(person), anchor)
    if np.isfinite(area):
        rows.append((area, sam_c, mem_p, mem_c, msk_p, msk_c, mem_n, msk_n))

data = np.array([r[1:] for r in rows], dtype=float)
areas = np.array([r[0] for r in rows], dtype=float)
edges = np.unique(np.quantile(areas, np.linspace(0, 1, NBINS + 1)))
bin_of = np.clip(np.digitize(areas, edges[1:-1]), 0, len(edges) - 2)

print(f"n={len(rows)}  visible area px^2@{RESIZE}: {areas.min():.0f}-{areas.max():.0f}")
print(f"\npooled  sam={data[:, 0].mean():.3f} | "
      f"memory {data[:, 1].mean():.3f}->{data[:, 2].mean():.3f} ({data[:, 2].mean()-data[:, 1].mean():+.4f}) | "
      f"mask {data[:, 3].mean():.3f}->{data[:, 4].mean():.3f} ({data[:, 4].mean()-data[:, 3].mean():+.4f})")
print(f"\n{'vis area bin':>16} {'n':>3} {'sam':>6} {'memP':>6} {'memC':>6} {'d_mem':>8} "
      f"{'mskP':>6} {'mskC':>6} {'d_msk':>8} {'fireM':>6} {'fireK':>6}")
for b in range(len(edges) - 1):
    sel = bin_of == b
    if not sel.any():
        continue
    d = data[sel]
    print(f"{int(edges[b]):>6}-{int(edges[b+1]):<9} {int(sel.sum()):>3} {d[:, 0].mean():>6.3f} "
          f"{d[:, 1].mean():>6.3f} {d[:, 2].mean():>6.3f} {d[:, 2].mean()-d[:, 1].mean():>+8.4f} "
          f"{d[:, 3].mean():>6.3f} {d[:, 4].mean():>6.3f} {d[:, 4].mean()-d[:, 3].mean():>+8.4f} "
          f"{d[:, 5].mean():>6.1f} {d[:, 6].mean():>6.1f}")

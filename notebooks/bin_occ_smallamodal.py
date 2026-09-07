"""Occlusion graph after DROPPING Q4 of amodal area (the large, saturated targets)."""
import sys
import json
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.typing.person_path import PersonPath

RESIZE = 1024
THR = 0.4
N_BINS = 4

s = pickle.load(open("data/claim_1/results.pkl", "rb"))
clips = s["clips"]


def cov(a, t=0.5):
    a = np.asarray(a)
    return float((a >= t).mean()) if len(a) else np.nan


pp = PersonPath(main_directory="data/person_path", occlusion_ranges=[2, 50], n_experiments=1200,
                n_after_occlusion=40, first_occ_min=2, random_seed=44, min_visible_area=500,
                max_distractor_overlap=0.5,
                non_targets=["crowd", "person_in_vehicle", "reflection", "person_in_background", "severly_occluded_person"])
anchor_of = {(v, int(p)): int(a) for v, p, a in zip(
    pp.selected_video_names.tolist(), pp.selected_person_ids.tolist(), pp.selected_anchor_video_frames.tolist())}


def amodal_area_1024(video, pid, anchor):
    d = json.load(open(f"data/person_path/amodal/{video}.json"))
    res = d["metadata"]["resolution"]; W, H = float(res["width"]), float(res["height"])
    ents = [e for e in d["entities"] if e["id"] == pid]
    if not ents:
        return np.nan
    frames = np.array([e["blob"]["frame_idx"] for e in ents])
    whs = np.array([e["bb"][2:4] for e in ents], dtype=np.float64)
    nearest = int(np.argmin(np.abs(frames - anchor)))
    w, h = whs[nearest]; scale = RESIZE / max(W, H)
    return float((w * scale) * (h * scale))


occ, area, sam, mem, mask = [], [], [], [], []
for c in clips:
    a = anchor_of.get((c["video"], int(c["person"])))
    if a is None:
        continue
    ar = amodal_area_1024(c["video"], int(c["person"]), a)
    if not np.isfinite(ar):
        continue
    occ.append(c["occ_count"]); area.append(ar)
    sam.append(cov(c["sam"])); mem.append(cov(c["memory"][THR])); mask.append(cov(c["mask"][THR]))

occ = np.array(occ); area = np.array(area)
sam = np.array(sam); mem = np.array(mem); mask = np.array(mask)
m = ~np.isnan(sam)
occ, area, sam, mem, mask = occ[m], area[m], sam[m], mem[m], mask[m]

q3 = np.quantile(area, 0.75)                                       # drop Q4 (top 25% by amodal area)
keep = area <= q3
occ, sam, mem, mask = occ[keep], sam[keep], mem[keep], mask[keep]
n = len(occ)

edges = np.unique(np.quantile(occ, np.linspace(0, 1, N_BINS + 1)))
which = np.clip(np.digitize(occ, edges[1:-1]), 0, len(edges) - 2)
print(f"n={n} (dropped Q4 amodal area, area<= {q3:.0f} px^2@{RESIZE})  commit thr={THR}")
print(f"{'occ bin':>10} {'n':>4} {'sam':>6} {'mem':>6} {'mask':>6} {'mem-sam':>8} {'mask-sam':>9}")
centers, s_m, e_m, k_m = [], [], [], []
for b in range(len(edges) - 1):
    sel = which == b
    if not sel.any():
        continue
    print(f"{int(edges[b]):>4}-{int(edges[b+1]):<5} {sel.sum():>4} {sam[sel].mean():>6.3f} {mem[sel].mean():>6.3f} "
          f"{mask[sel].mean():>6.3f} {mem[sel].mean()-sam[sel].mean():>+8.3f} {mask[sel].mean()-sam[sel].mean():>+9.3f}")
    centers.append(occ[sel].mean()); s_m.append(sam[sel].mean()); e_m.append(mem[sel].mean()); k_m.append(mask[sel].mean())

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(centers, k_m, "-s", color="#c0392b", lw=2.2, ms=8, label="mask oracle")
ax.plot(centers, e_m, "-o", color="#2471a3", lw=2.2, ms=8, label="memory oracle")
ax.plot(centers, s_m, "--^", color="#7f8c8d", lw=2.2, ms=8, label="sam baseline")
ax.set_xlabel("number of occluded frames")
ax.set_ylabel("post-occlusion coverage (IoU >= 0.5)")
ax.set_title(f"coverage vs occlusion  |  amodal Q4 dropped (n={n}, thr={THR})")
ax.set_ylim(0, 1.02); ax.grid(True, ls=":", alpha=0.5); ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig("data/claim_1/claim_1_occ_smallamodal.png", dpi=130)
print("saved data/claim_1/claim_1_occ_smallamodal.png")

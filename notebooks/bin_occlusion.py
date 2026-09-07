"""Bin claim_1 coverage by NUMBER OF OCCLUDED FRAMES (occ_count), at a chosen commit-gate threshold."""
import pickle
import numpy as np
import matplotlib.pyplot as plt

THR = 0.4

s = pickle.load(open("data/claim_1/results.pkl", "rb"))
clips = s["clips"]
n_bins = int(s.get("n_bins", 4))


def cov(a, t=0.5):
    a = np.asarray(a)
    return float((a >= t).mean()) if len(a) else np.nan


def arm(c, key):
    v = c[key]
    return v[THR] if isinstance(v, dict) else v                   # dict = swept; else single array (sam)


occ = np.array([c["occ_count"] for c in clips])
sam = np.array([cov(arm(c, "sam")) for c in clips])
mem = np.array([cov(arm(c, "memory")) for c in clips])
mask = np.array([cov(arm(c, "mask")) for c in clips])
m = ~np.isnan(sam)
occ, sam, mem, mask = occ[m], sam[m], mem[m], mask[m]
n = len(occ)

edges = np.unique(np.quantile(occ, np.linspace(0, 1, n_bins + 1)))
which = np.clip(np.digitize(occ, edges[1:-1]), 0, len(edges) - 2)
print(f"n={n}  threshold={THR}  occ_count {occ.min()}-{occ.max()}")
print(f"{'occ bin':>12} {'n':>4} {'sam':>6} {'mem':>6} {'mask':>6} {'mem-sam':>8} {'mask-sam':>9}")
centers, s_m, e_m, k_m = [], [], [], []
for b in range(len(edges) - 1):
    sel = which == b
    if not sel.any():
        continue
    print(f"{int(edges[b]):>5}-{int(edges[b+1]):<6} {sel.sum():>4} {sam[sel].mean():>6.3f} "
          f"{mem[sel].mean():>6.3f} {mask[sel].mean():>6.3f} "
          f"{mem[sel].mean()-sam[sel].mean():>+8.3f} {mask[sel].mean()-sam[sel].mean():>+9.3f}")
    centers.append(occ[sel].mean())
    s_m.append(sam[sel].mean()); e_m.append(mem[sel].mean()); k_m.append(mask[sel].mean())

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(centers, k_m, "-s", color="#c0392b", lw=2.2, ms=8, label="mask oracle")
ax.plot(centers, e_m, "-o", color="#2471a3", lw=2.2, ms=8, label="memory oracle")
ax.plot(centers, s_m, "--^", color="#7f8c8d", lw=2.2, ms=8, label="sam baseline")
ax.set_xlabel("number of occluded frames")
ax.set_ylabel("post-occlusion coverage (IoU >= 0.5)")
ax.set_title(f"coverage vs occlusion (n={n}, commit thr={THR})")
ax.set_ylim(0, 1.02)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig("data/claim_1/claim_1_occlusion.png", dpi=130)
print("saved data/claim_1/claim_1_occlusion.png")

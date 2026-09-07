"""Bin claim_1 coverage by OCCLUSION RATE = occ_count / (n_frames - first_occlusion)."""
import pickle
import numpy as np
import matplotlib.pyplot as plt

THR = 0.2

s = pickle.load(open("data/claim_1/results.pkl", "rb"))
clips = s["clips"]
n_bins = int(s.get("n_bins", 4))


def cov(a, t=0.5):
    a = np.asarray(a)
    return float((a >= t).mean()) if len(a) else np.nan


def arm(c, key):
    v = c[key]
    return v[THR] if isinstance(v, dict) else v


def occ_rate(c):
    denom = c["n_frames"] - c["first_occlusion"]
    return c["occ_count"] / denom if denom > 0 else np.nan


rate = np.array([occ_rate(c) for c in clips])
sam = np.array([cov(arm(c, "sam")) for c in clips])
mem = np.array([cov(arm(c, "memory")) for c in clips])
mask = np.array([cov(arm(c, "mask")) for c in clips])
m = ~np.isnan(sam) & ~np.isnan(rate)
rate, sam, mem, mask = rate[m], sam[m], mem[m], mask[m]
n = len(rate)

edges = np.unique(np.quantile(rate, np.linspace(0, 1, n_bins + 1)))
which = np.clip(np.digitize(rate, edges[1:-1]), 0, len(edges) - 2)
print(f"n={n}  threshold={THR}  occ_rate {rate.min():.2f}-{rate.max():.2f}")
print(f"{'rate bin':>14} {'n':>4} {'sam':>6} {'mem':>6} {'mask':>6} {'mem-sam':>8} {'mask-sam':>9}")
centers, s_m, e_m, k_m = [], [], [], []
for b in range(len(edges) - 1):
    sel = which == b
    if not sel.any():
        continue
    print(f"{edges[b]:.2f}-{edges[b+1]:.2f}   {sel.sum():>4} {sam[sel].mean():>6.3f} "
          f"{mem[sel].mean():>6.3f} {mask[sel].mean():>6.3f} "
          f"{mem[sel].mean()-sam[sel].mean():>+8.3f} {mask[sel].mean()-sam[sel].mean():>+9.3f}")
    centers.append(rate[sel].mean())
    s_m.append(sam[sel].mean()); e_m.append(mem[sel].mean()); k_m.append(mask[sel].mean())

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(centers, k_m, "-s", color="#c0392b", lw=2.2, ms=8, label="mask oracle")
ax.plot(centers, e_m, "-o", color="#2471a3", lw=2.2, ms=8, label="memory oracle")
ax.plot(centers, s_m, "--^", color="#7f8c8d", lw=2.2, ms=8, label="sam baseline")
ax.set_xlabel("occlusion rate  (occluded / post-occlusion frames)")
ax.set_ylabel("post-occlusion coverage (IoU >= 0.5)")
ax.set_title(f"coverage vs occlusion rate (n={n}, commit thr={THR})")
ax.set_ylim(0, 1.02)
ax.grid(True, ls=":", alpha=0.5)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig("data/claim_1/claim_1_occrate.png", dpi=130)
print("saved data/claim_1/claim_1_occrate.png")

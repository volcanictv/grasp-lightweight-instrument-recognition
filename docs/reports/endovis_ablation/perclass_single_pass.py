import sys

import numpy as np

sys.path[:0] = [r"C:\Users\aryan\Classification Surgurical Tools\.claude\worktrees\frame-decode-rarecls\src", r"C:\Users\aryan\Classification Surgurical Tools\.claude\worktrees\frame-decode-rarecls\scripts"]
from scipy.special import softmax

from evidential_seeds_e2e_eval import CONFIGS, alpha_mix

NAMES = ["BF", "PF", "LND", "MCS", "UP", "SI", "CA"]
z = np.load("e17_single_ens_s42.npz")
y = z["y"]
a = alpha_mix(lambda k: z[f"det_{k}"], CONFIGS["four"])
ens = (a / a.sum(1, keepdims=True)).argmax(1)
d2 = softmax(np.load("val_logits_dinov2-base_s42.npy"), 1).argmax(1)
d3 = softmax(np.load("val_logits_dinov3-base_s42.npy"), 1).argmax(1)
print("class  n_val  | recall: ens  DINOv2  DINOv3 | precision: ens  DINOv2  DINOv3")
for c in range(7):
    m = y == c
    r = lambda p: 100 * (p[m] == c).mean() if m.any() else float("nan")
    pr = lambda p: 100 * (y[p == c] == c).mean() if (p == c).any() else float("nan")
    print(f"{NAMES[c]:>4} {m.sum():6d}  |  {r(ens):6.1f} {r(d2):6.1f} {r(d3):6.1f}  |  {pr(ens):6.1f} {pr(d2):6.1f} {pr(d3):6.1f}")
print("accuracy", [round(100 * (p == y).mean(), 2) for p in (ens, d2, d3)])

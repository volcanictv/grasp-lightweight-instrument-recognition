import json
import sys

import numpy as np

d = np.load("feats.npz")
rows = json.load(open("rows.json"))
idx, f, lg, W, b = d["idx"], d["feat"], d["logits"], d["head_w"], d["head_b"]
CL = ["BF", "PF", "LND", "MCS", "SI", "CA", "LG"]
sets = {r["idx"]: r["set"] for r in rows}
true = {r["idx"]: r["true"] for r in rows}
S = np.array([sets[i] for i in idx])
member_pred = lg.argmax(1)
print("resnet50_320 member alone: prediction per set (true LG sets A, B, D; SI set C)")
for s in sorted(set(S)):
    m = S == s
    print(" ", s, [CL[p] for p in member_pred[m]])
# margin LG - SI decomposed into channels
w = W[6] - W[4]
contrib = f * w[None, :]
margin = contrib.sum(1) + (b[6] - b[4])
assert np.allclose(margin, lg[:, 6] - lg[:, 4], atol=1e-3), "decomposition mismatch"
good = S == "B_LG_correct_all_seeds"
bad = np.isin(S, ["A_LG_missed_TAPIS_right", "D_LG_called_SI_s42"]) & (member_pred != 6)
print(f"\nLG-minus-SI margin: correct LG mean {margin[good].mean():.1f}, missed LG mean {margin[bad].mean():.1f}  (n={good.sum()} vs {bad.sum()})")
delta = contrib[good].mean(0) - contrib[bad].mean(0)  # channels that separate the two groups
order = np.argsort(-np.abs(delta))
tot = np.abs(delta).sum()
for k in (10, 20, 50, 100, 200, 500):
    print(f"  top {k:4d} of 2048 channels carry {100 * np.abs(delta[order[:k]]).sum() / tot:5.1f}% of the absolute group difference; signed sum of the top {k}: {delta[order[:k]].sum():.1f} of total {delta.sum():.1f}")
# how much does the margin of the missed instruments move if the top-k separating channels are set to the correct-group mean?
for k in (20, 100, 500):
    ch = order[:k]
    f2 = f.copy()
    f2[np.ix_(np.where(bad)[0], ch)] = f[good][:, ch].mean(0)
    m2 = (f2 * w[None, :]).sum(1) + (b[6] - b[4])
    print(f"  set the top {k} channels of the missed instruments to the correct-group mean: margin {margin[bad].mean():.1f} -> {m2[bad].mean():.1f}; flips to LG-over-SI: {(m2[bad] > 0).sum()} of {bad.sum()} (diagnostic on the same 28, not a result)")
# are these channels also what the SI instruments use?
si = S == "C_SI_correct"
print(f"\nmargin for correct SI (should be negative): mean {margin[si].mean():.1f}; with the same top-100 channel swap applied to SI: ", end="")
ch = order[:100]
f3 = f.copy()
f3[np.ix_(np.where(si)[0], ch)] = f[good][:, ch].mean(0)
m3 = (f3 * w[None, :]).sum(1) + (b[6] - b[4])
print(f"{m3[si].mean():.1f}; SI flipped to LG: {(m3[si] > 0).sum()} of {si.sum()}")

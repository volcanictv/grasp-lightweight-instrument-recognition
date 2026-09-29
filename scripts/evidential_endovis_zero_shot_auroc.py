import json
import sys
from pathlib import Path

ROOT = Path("/home/yzx/Desktop/Classification Surgurical Tools")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import evidential_analyze as ea

out = {}
for tname in ("endovis2018", "endovis2017"):
    dc = ea.load(ROOT / f"experiments_edl/extract/C_{tname}_fold1_s42.npz")
    de = ea.load(ROOT / f"experiments_edl/extract/E_{tname}_fold1_s42.npz")
    c = ea.arm_c(dc)
    e = ea.arm_e(de, c["signals"]["B1_mc_vote_disagreement"])
    present = sorted(set(c["y"].tolist()))
    rc = ea.evaluate_arm(c, present)
    re_ = ea.evaluate_arm(e, present)
    out[tname] = {"n": rc["n"], "acc_C": rc["accuracy"], "acc_E": re_["accuracy"],
                  "errors_C": rc["errors"], "errors_E": re_["errors"], "signals": {}}
    print(f"\n== {tname}: n={rc['n']}  acc CE-MC arm {rc['accuracy']:.3f} ({rc['errors']} errors), "
          f"evidential arm {re_['accuracy']:.3f} ({re_['errors']} errors)")
    print(f"{'signal':<38}{'AUROC':>8}  {'recall@10%':>10}{'recall@20%':>11}{'recall@30%':>11}{'recall@50%':>11}")
    for arm_name, r, a in (("C", rc, c), ("E", re_, e)):
        for sig, v in r["signals"].items():
            r50 = ea.recall_at_budget(a["signals"][sig], a["err"], 0.50)
            print(f"{sig:<38}{v['auroc']:>8.3f}  {v['error_recall_at_10pct']:>10.3f}{v['error_recall_at_20pct']:>11.3f}"
                  f"{v['error_recall_at_30pct']:>11.3f}{r50:>11.3f}")
            out[tname]["signals"][sig] = {"arm": arm_name, "auroc": v["auroc"], "auroc_ci95": v["auroc_ci95"],
                                          "recall_10": v["error_recall_at_10pct"], "recall_20": v["error_recall_at_20pct"],
                                          "recall_30": v["error_recall_at_30pct"], "recall_50": r50}
Path("/tmp/endovis_zero_shot_auroc.json").write_text(json.dumps(out, indent=1))
print("\nwrote /tmp/endovis_zero_shot_auroc.json")

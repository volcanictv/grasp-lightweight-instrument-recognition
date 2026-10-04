"""The pre-registered choice step of scripts/overnight_yolo_arms.sh, run on its own: an arm (Y or NY) is confirmed only if it is an IMPROVEMENT on
fold1 (three seeds) and its accuracy delta against the baseline is positive on fold2. The best confirmed arm by fold1 accuracy is the one
whose official-split members would be trained; NONE if no arm is confirmed. Run again after the scoring finished because the driver read
yolo_fold1.json before the (separately run) fold1 scoring had written it.
Usage (repo root): python scripts/choose_yolo_arm.py
"""
import json

f1 = json.load(open("docs/reports/training_arms/yolo_fold1.json"))["summary"]
f2 = json.load(open("docs/reports/training_arms/yolo_fold2.json"))["summary"]
for arm in ("baseline", "N", "Y", "NY"):
    for name, s in (("fold1", f1), ("fold2", f2)):
        if arm in s:
            a = s[arm]
            print(f"{name} {arm:<9} accuracy {a['accuracy'][0]:.4f} +- {a['accuracy'][1]:.4f}  delta {a['delta_accuracy_vs_baseline']:+.4f}  {a['verdict']}")
ok = [a for a in ("Y", "NY") if a in f1 and a in f2 and f1[a]["verdict"] == "IMPROVEMENT" and f2[a]["delta_accuracy_vs_baseline"] > 0]
print("choice:", max(ok, key=lambda a: (round(f1[a]["accuracy"][0], 4), a == "NY")) if ok else "NONE")

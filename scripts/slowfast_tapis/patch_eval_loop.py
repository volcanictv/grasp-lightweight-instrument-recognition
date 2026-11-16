import os, shutil
p = os.path.expanduser("~/baselines/GraSP/TAPIS_gtbox/tools/train_net.py")
if not os.path.exists(p + ".orig"): shutil.copy(p, p + ".orig")
s = open(p).read()
a = "            inputs[0] = inputs[0].cuda(non_blocking=True)\n            if cfg.MODEL.PRECISION == 64:\n                inputs[0] = inputs[0].double()\n\n            for key, val in data.items():\n                data[key] = val.cuda(non_blocking=True)"
assert s.count(a) == 1, s.count(a)
s = s.replace(a, "            inputs = [input.cuda(non_blocking=True) for input in inputs]  # every pathway (SlowFast has two)\n            if cfg.MODEL.PRECISION == 64:\n                inputs[0] = inputs[0].double()\n\n            for key, val in data.items():\n                data[key] = val.cuda(non_blocking=True)")
open(p, "w").write(s); print("patched eval loop")

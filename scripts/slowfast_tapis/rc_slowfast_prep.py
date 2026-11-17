"""Stage the 30 fps GraSP frames for the SlowFast/TAPIS loader and build one self-contained data directory per run.

The loader reads, around every annotated keyframe, NUM_FRAMES frames at a step of SAMPLING_RATE (indices clamped at the video ends). Only those frames are extracted from the per-case
archives (about 20 GB instead of the full 260 GB). The centre index follows grasp.py exactly: videos stored at 30 fps use the frame number itself, CASE014 is looked up in the frame list,
the others use round(frame_number * 30 / 45).

Run directories (data/run_<name>/{annotations,features,frame_lists}) use the same file names for every run so one runner script serves all of them:
  annotations/gt_{train,test}.json   features/{train,test}.pth   frame_lists/{train,test}.csv   annotations/preds_{train,test}.json
usage: rc_slowfast_prep.py --archives DIR --release-annotations DIR --data DIR --windows DIR --runs official,fold1,fold2[,smoke]
"""
import argparse, csv, hashlib, json, os, subprocess, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FPS30 = {"CASE021", "CASE041", "CASE047", "CASE050", "CASE051", "CASE053"}


def read_lists(path):
    lists = defaultdict(list)
    with open(path) as f:
        for line in f:
            row = line.split()
            assert len(row) == 4, f"{path}: unexpected row {line!r}"
            lists[row[0]].append(row[3])
    return lists


def center_index(video, frame_num, paths):
    if video in FPS30:
        return frame_num
    if video == "CASE014":
        return paths.index(f"{video}/{frame_num:09d}.jpg")
    return round(frame_num * 30 / 45)


def window(center, half_len, rate, n):
    return [min(max(i, 0), n - 1) for i in range(center - half_len, center + half_len, rate)]


def needed_frames(gt_json, lists, num_frames, rate, cases):
    gt = json.load(open(gt_json)); need = defaultdict(set); missing = 0
    for im in gt["images"]:
        v = im["video_name"]
        if cases and v not in cases: continue
        if v not in lists: missing += 1; continue
        paths = lists[v]; c = center_index(v, im["frame_num"], paths)
        for i in window(c, num_frames * rate // 2, rate, len(paths)): need[v].add(paths[i])
    return need, missing


def extract(archive, names, out, listing):
    Path(listing).write_text("\n".join(sorted(names)) + "\n")
    r = subprocess.run(["tar", "-xzf", str(archive), "-C", str(out), "--no-same-owner", "-T", str(listing)], capture_output=True, text=True)
    return r.returncode, r.stderr[-400:]


def link(src, dst):
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists(): dst.unlink()
    os.symlink(os.path.abspath(src), dst)


def subset(src, dst, cases):
    """filtered copy of one run file restricted to the videos in cases (smoke test only)"""
    src, dst = Path(src), Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".csv":
        with open(src) as f, open(dst, "w") as g:
            for line in f:
                if line.split()[0] in cases: g.write(line)
    elif src.suffix == ".json":
        d = json.load(open(src)); keep = {i["id"] for i in d["images"] if i["video_name"] in cases}
        d["images"] = [i for i in d["images"] if i["id"] in keep]; d["annotations"] = [x for x in d["annotations"] if x["image_id"] in keep]
        json.dump(d, open(dst, "w"))
    else:
        import torch
        torch.save([f for f in torch.load(src) if f["file_name"].split("/")[0] in cases], dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", required=True); ap.add_argument("--release-annotations", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--windows", required=True); ap.add_argument("--runs", default="official,fold1,fold2")
    ap.add_argument("--num-frames", type=int, default=16); ap.add_argument("--rate", type=int, default=15); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--smoke-train", default="CASE001,CASE003,CASE007"); ap.add_argument("--smoke-test", default="CASE041,CASE047,CASE051")
    ap.add_argument("--smoke-from", choices=["official", "test"], default="official", help="test: build the smoke training part from the test split too, for use before the train annotations are available")
    a = ap.parse_args()
    data, ann, win, arch = Path(a.data), Path(a.release_annotations), Path(a.windows), Path(a.archives)
    gt = lambda n: next(iter(sorted(ann.rglob(f"grasp_short-term_{n}.json"))), None)
    src = {  # run -> {train|test: (gt json, preds json, features, frame list)}
        "official": {"train": (gt("train"), data / "GraSP/annotations/train_train_preds.json", data / "GraSP/features/train_train_region_features.pth", data / "GraSP/frame_lists/train.csv"),
                     "test": (gt("test"), data / "GraSP/annotations/test_val_preds.json", data / "GraSP/features/test_val_region_features.pth", data / "GraSP/frame_lists/test.csv")},
    }
    for f, o in ((1, 2), (2, 1)):  # train on the other fold, as the released two-fold TAPIS models do
        F = data / "GraSP_folds"
        src[f"fold{f}"] = {"train": (gt(f"fold{o}"), F / f"annotations/fold{o}_val_preds.json", F / f"features/fold{o}_val_region_features.pth", F / f"frame_lists/fold{o}.csv"),
                           "test": (gt(f"fold{f}"), F / f"annotations/fold{f}_val_preds.json", F / f"features/fold{f}_val_region_features.pth", F / f"frame_lists/fold{f}.csv")}
    src["smoke"] = {"train": src["official"]["test" if a.smoke_from == "test" else "train"], "test": src["official"]["test"]}
    runs = a.runs.split(",")
    if a.smoke_from == "test": smoke_cases = {"train": set(a.smoke_test.split(",")), "test": set(a.smoke_test.split(","))}
    else: smoke_cases = {"train": set(a.smoke_train.split(",")), "test": set(a.smoke_test.split(","))}
    for r in runs:
        for part, files in src[r].items():
            for p in files: assert p is not None and Path(p).exists(), f"missing input for {r}/{part}: {p}"
    need_all = defaultdict(set); problems = {}
    for r in runs:
        cases = smoke_cases if r == "smoke" else {"train": None, "test": None}
        rd = data / f"run_{r}"
        for part, (gtj, preds, feats, csvp) in src[r].items():
            lists = read_lists(csvp)
            need, miss = needed_frames(gtj, lists, a.num_frames, a.rate, cases[part])
            for v, s in need.items(): need_all[v] |= s
            problems[f"{r}/{part}"] = {"videos": len(need), "frames": sum(len(s) for s in need.values()), "annotated_images_without_frame_list": miss}
            for dirn, fname, srcp in (("annotations", f"gt_{part}.json", gtj), ("annotations", f"preds_{part}.json", preds), ("features", f"{part}.pth", feats), ("frame_lists", f"{part}.csv", csvp)):
                if cases[part]: subset(srcp, rd / dirn / fname, cases[part])
                else: link(srcp, rd / dirn / fname)
    print(json.dumps(problems, indent=1), flush=True)
    win.mkdir(parents=True, exist_ok=True); todo = []
    for v, s in sorted(need_all.items()):
        left = {n for n in s if not (win / n).exists()}
        if left: todo.append((v, left))
    absent = [v for v, _ in todo if not (arch / f"{v}.tar.gz.ok").exists()]
    if absent: sys.exit(f"archives not verified yet (no .ok marker): {absent}")
    print(f"extracting {sum(len(s) for _, s in todo)} frames from {len(todo)} archives ({a.jobs} at a time)", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda t: (t[0],) + extract(arch / f"{t[0]}.tar.gz", t[1], win, win / f"_list_{t[0]}.txt"), todo))
    for v, rc, err in res: print(v, "tar exit", rc, err.strip()[:200], flush=True)
    miss = {v: sorted(n for n in s if not (win / n).exists()) for v, s in need_all.items()}
    miss = {v: m for v, m in miss.items() if m}
    json.dump({"problems": problems, "missing_frames": {v: len(m) for v, m in miss.items()}}, open(win / "_prep_report.json", "w"), indent=1)
    if miss: sys.exit("frames missing after extraction: " + json.dumps({v: m[:3] + [f"... {len(m)} total"] for v, m in miss.items()}))
    print("all needed frames present", flush=True)
    if {"official", "fold1", "fold2"} <= set(runs):  # marker for the transfer to titanxp: only written for the complete set, never for the smoke subset
        rel = sorted(n for s in need_all.values() for n in s)
        nl = chr(10)
        (win / "_files.txt").write_text(nl.join(rel) + nl)
        with ThreadPoolExecutor(a.jobs) as ex: sums = list(ex.map(lambda n: hashlib.sha256((win / n).read_bytes()).hexdigest(), rel))
        (win / "_manifest.sha256").write_text("".join(f"{h}  {n}{nl}" for h, n in zip(sums, rel)))
        (win / "_ready").write_text(f"{len(rel)} files{nl}"); print(f"manifest of {len(rel)} files written; windows ready", flush=True)


if __name__ == "__main__":
    main()

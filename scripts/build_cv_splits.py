"""Deterministic, size-balanced, case-level K-fold cross-validation splits over the 13 GraSP cases (the 8 official training cases and the 5 official test cases).

The official split files are only read. For every fold k the script writes two new annotation files in the same layout as the official ones,
    grasp_short-term_cv<K>_f<k>_train.json   the cases outside fold k
    grasp_short-term_cv<K>_f<k>_test.json    the cases of fold k (held out)
and a registry, cv<K>_splits.json, that scripts/ read through the GRASP_EXTRA_SPLITS variable (src/surgical_ai/data/splits.py). Cases are assigned by a fixed rule (largest case first,
into the fold with the fewest instruments so far; ties by fold number), so the folds are reproducible without a random seed. The official test file's image and annotation ids start at 1 like the training
file's, so its ids are shifted (images +10000, annotations +100000) to stay unique inside a fold file.

Results from these folds are cross-validation results over 13 cases; they are never mixed with the official benchmark numbers.

Usage:
    python scripts/build_cv_splits.py --data-root GraSP --folds 5
    GRASP_EXTRA_SPLITS=GraSP/annotations/cv5_splits.json python scripts/train.py ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

IMG_SHIFT, ANN_SHIFT = 10_000, 100_000
KEYS = ("info", "categories", "actions_categories", "phases_categories", "steps_categories")


def assign(case_sizes: dict[str, int], k: int) -> list[list[str]]:
    folds: list[list[str]] = [[] for _ in range(k)]
    load = [0] * k
    for case, n in sorted(case_sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        j = min(range(k), key=lambda i: (load[i], i))
        folds[j].append(case)
        load[j] += n
    return [sorted(f) for f in folds]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="print the assignment and the file sizes, write nothing")
    args = ap.parse_args()
    ann = args.data_root / "annotations"
    docs = {n: json.loads((ann / f"grasp_short-term_{n}.json").read_text(encoding="utf-8")) for n in ("train", "test")}
    # every image/annotation tagged with its source file, ids shifted for the test file
    images, annotations = [], []
    for name, doc in docs.items():
        s_img, s_ann = (0, 0) if name == "train" else (IMG_SHIFT, ANN_SHIFT)
        for im in doc["images"]:
            images.append({**im, "id": im["id"] + s_img})
        for a in doc["annotations"]:
            annotations.append({**a, "id": a["id"] + s_ann, "image_id": a["image_id"] + s_img})
    case_of_image = {im["id"]: im["video_name"] for im in images}
    sizes = Counter(case_of_image[a["image_id"]] for a in annotations)
    assert len(sizes) == 13 and sum(sizes.values()) == 6170 + 2861, (len(sizes), sum(sizes.values()))
    folds = assign(dict(sizes), args.folds)
    print("fold assignment (instruments):")
    for k, cs in enumerate(folds):
        print(f"  fold {k}: {', '.join(cs)}  ({sum(sizes[c] for c in cs)} instruments)")
    assert sorted(c for f in folds for c in f) == sorted(sizes), "every case must be held out exactly once"
    registry = {"note": f"{args.folds}-fold case-level cross-validation over the 13 GraSP cases; reported separately from the official benchmark",
                "folds": {str(k): cs for k, cs in enumerate(folds)}, "splits": {}, "train_val": {}, "sha256": {}}
    def write(name: str, keep_img: list[dict]) -> None:
        ids = {im["id"] for im in keep_img}
        keep_ann = [a for a in annotations if a["image_id"] in ids]
        out = {key: docs["train"][key] for key in KEYS} | {"images": keep_img, "annotations": keep_ann}
        fname = f"grasp_short-term_{name}.json"
        registry["splits"][name] = fname
        print(f"  {name}: {len({im['video_name'] for im in keep_img})} cases, {len(keep_img)} images, {len(keep_ann)} instruments")
        if not args.dry_run:
            text = json.dumps(out)
            (ann / fname).write_text(text, encoding="utf-8")
            registry["sha256"][fname] = hashlib.sha256(text.encode()).hexdigest()

    for k, held in enumerate(folds):
        train_cases = [c for c in sizes if c not in held]
        for role, cases in (("train", train_cases), ("test", held)):
            write(f"cv{args.folds}_f{k}_{role}", [im for im in images if im["video_name"] in cases])
        # a small split inside the training data, the smallest training case: it is only reported during a fixed-schedule training, it never selects anything
        smallest = min(train_cases, key=lambda c: (sizes[c], c))
        write(f"cv{args.folds}_f{k}_dev", [im for im in images if im["video_name"] == smallest])
        registry["train_val"][f"cv{args.folds}_f{k}"] = [f"cv{args.folds}_f{k}_train", f"cv{args.folds}_f{k}_test"]

    # tiny splits for the smoke test of the whole chain: every 25th image of fold 0's training cases, the smallest of them as dev, every 20th image of fold 0's held-out cases
    train0 = [c for c in sizes if c not in folds[0]]
    pick = lambda cs, step: [im for c in sorted(cs) for im in sorted((i for i in images if i["video_name"] == c), key=lambda i: i["file_name"])[::step]]
    write(f"cv{args.folds}_smoke_train", pick(train0, 25))
    write(f"cv{args.folds}_smoke_dev", pick([min(train0, key=lambda c: (sizes[c], c))], 40))
    write(f"cv{args.folds}_smoke_test", pick(folds[0], 20))
    registry["train_val"][f"cv{args.folds}_smoke"] = [f"cv{args.folds}_smoke_train", f"cv{args.folds}_smoke_test"]
    if not args.dry_run:
        (ann / f"cv{args.folds}_splits.json").write_text(json.dumps(registry, indent=1), encoding="utf-8")
        print("wrote", ann / f"cv{args.folds}_splits.json")


if __name__ == "__main__":
    main()

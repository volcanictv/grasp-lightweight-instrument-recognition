# Surgical instrument recognition on GraSP

Code for recognizing surgical instruments (7 classes) in GraSP, a robot-assisted
radical prostatectomy video dataset. Four ways of asking "what instrument is
that," in increasing order of how precisely they answer it:

- **multilabel_frame** — which instruments are somewhere in this frame.
- **region_classification** — given a cropped instrument (from its ground-truth
  mask), which class is it.
- **detection** — draw a box around every instrument and label it (Faster R-CNN).
- **instance_segmentation** — color in every instrument's exact pixels
  (Mask R-CNN, plus a from-scratch centroid/offset architecture in
  `models/segmenters/`).

Full results, figures, and the actual accuracy numbers live in the published
report, not here. This file is setup and how-to-run only. I promise I will not
make you read a research paper to figure out how to install torch.

## Setup

Python 3.11+. Not tested below that, might work, not going to find out for you.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # pick your own CUDA build
pip install -r requirements.txt
```

If you don't have a CUDA GPU, drop the `--index-url` line and install the CPU
build instead. Everything still runs, just slower, and you'll want `--device cpu`
on every script below.

Get the GraSP dataset yourself (not redistributed here, it's tens of GB) and
either put it at `./GraSp` in the repo root, or point `GRASP_DATA_ROOT` at
wherever you actually put it:

```bash
export GRASP_DATA_ROOT=/path/to/GraSP
```

Expected layout, because someone will ask:

```
GraSP/
  frames-001/frames/<CASE_ID>/<frame>.jpg
  annotations/
    segmentations/<CASE_ID>/<frame>.png
    grasp_short-term_train.json
    grasp_short-term_test.json
    grasp_short-term_fold1.json
    grasp_short-term_fold2.json
```

Verify it actually works before you trust anything downstream of it:

```bash
python -m pytest tests/ -q
```

81 tests, should all pass, takes under 20 seconds. If it doesn't pass, stop and
fix that first — nothing past this point means anything if the basics are broken.

## Running things

Every experiment is one YAML config. No hyperparameters live in Python, no CLI
flags change results — only paths and device do.

```bash
python scripts/train.py configs/<name>.yaml --data-root ./GraSP --device cuda:0
```

Each run writes `experiments/<run_id>/` with a `manifest.json` (full config,
git commit, seed, split checksums, package versions, final metrics — everything
needed to answer "how did we get this number" without guessing) and `best.pt`.
`configs/` has ~36 examples covering every task type above; copy the closest one
and change what you need, don't start from scratch.

Splits: `official` (train on the 8 official-train cases, test on the 5
official-test cases — use this for one final confirmatory run, not for
picking hyperparameters), `fold1`/`fold2` (case-level cross-validation carved
out of the 8 train cases, official test never touched — use these for
everything else). This distinction matters and got violated once already
this project; don't repeat that.

Other scripts, all under `scripts/`:

- `inspect_dataset.py` — dataset stats, class histograms, mask overlays.
- `build_frame_cache.py` — resized on-disk JPEG cache of the annotated frames
  only, if your dataloader turns out to be the bottleneck (it probably will
  be, native resolution is 800x1280 and there's no getting around decoding
  that every epoch otherwise).
- `benchmark.py` / `benchmark_ensemble_latency.py` — parameter count, FLOPs,
  model size, GPU/CPU latency. `train.py` already reports final metrics
  per-run, there's no separate "load a checkpoint and re-evaluate" script —
  if you need that, it's a small addition, not a missing feature.
- `evaluate_tracking*.py` — IOU-tracker-by-detection, on top of a trained
  detector or segmenter.
- `evaluate_maskrcnn_ensemble.py` — weighted box+mask fusion across N
  Mask R-CNN checkpoints (`--model checkpoint_path:registry_name`, repeatable).
- `evaluate_sam2_boxprompt.py` / `finetune_sam2_decoder.py` /
  `evaluate_maskrcnn_sam2_ensemble.py` — SAM2 as a box-prompted mask generator,
  zero-shot and fine-tuned. Needs SAM2 installed separately (it's not a normal
  pip package):

  ```bash
  git clone https://github.com/facebookresearch/sam2.git && cd sam2 && pip install -e .
  ```

  Then grab a checkpoint from Meta's own download links in that repo's README.
  Point the scripts at wherever you put it.

## Adding a model

Models live behind a registry per task type (`models/registry.py` for
classifiers, `models/detectors/registry.py`, `models/segmenters/registry.py`). Add a file,
register it with a decorator, import it once in that registry's `__init__`/
bottom-of-file import list. That's the whole contract — training, data
loading, and evaluation code never need to know a new model exists. If you're
touching `data/` or `training/` to add a model, you're doing it wrong; go
look at how an existing one is wired up instead.

## Repo layout

```
configs/            one YAML per experiment
src/surgical_ai/
  data/             dataset classes, transforms, official splits
  models/           classifiers/, detectors/, segmenters/, all registry-based
  training/         trainer, losses, samplers
  evaluation/       classification/detection/segmentation metrics, latency benchmarking
  inference/        tracking-by-detection, model ensembling
scripts/            everything you actually run
experiments/        one dir per run, git-ignored except manifest.json
docs/reports/       the actual report with actual numbers -- go read that instead
tests/
```

Go read the report for results. This file was already longer than I wanted it
to be.

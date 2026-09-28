# Dead code audit, 2026-09-29: nothing deleted, and why

Requested: clean up dead code across `scripts/`, `configs/`, and `src/surgical_ai/`. Outcome:
**zero files removed**, after a real audit, not a decision to skip the work. Every candidate a
naive "not referenced anywhere" search surfaced turned out to be real, intentional, provenance
code once checked against git history and actual usage rather than text search alone.

## Why the naive signal fails on this repo specifically

Cross-referencing a script or config's filename against docs, other scripts, and other configs is
the normal way to find dead code, and it does not work here, because of two things this project
does deliberately:

1. **Configs reference each other by generated checkpoint-directory name, not by config filename.**
   An ensemble config like `region_ensemble_deepdropout_fold1.yaml` lists
   `experiments/region_baseline_deepdropout_fold1_20260920-234547/best.pt` as a member checkpoint.
   That path contains the training config's stem (`region_baseline_deepdropout_fold1`) with a
   run timestamp appended, not the literal `.yaml` filename. A plain filename search finds
   nothing; a stem-substring search finds it. This one config
   (`region_baseline_deepdropout_fold1.yaml`) is the source of the gate-confirmation checkpoint
   behind the report's current leave-one-case-out and 15%-gate numbers — a filename search alone
   would have called it dead.
2. **Docs describe experiments by parameter, not by filename.** `docs/findings.md`'s 5-seed
   reproducibility table says "retrained all 4 ensemble members at seeds 43-46"; it never writes
   `region_baseline_seed43.yaml`. All sixteen `*_seed43/44/45/46.yaml` configs (four architectures
   x four seeds) are the literal source of that cited, tabulated result, and none of them contain
   any recognizable filename string in any doc.

## What was actually checked

- **Every `scripts/*.py` and `configs/*.yaml`** cross-referenced against: all docs (tracked and
  the local-only ones in the main checkout — CLAUDE.md, PROJECT_SPEC.md, DECISIONS.md,
  findings.md, imbalance_notes.md, detection_literature_notes.md, generalization_datasets.md,
  dataset_report.md, environment.md, error_analysis.md, literature_positioning.md), every other
  script, every other config, the 40 tracked `experiments/*/manifest.json` files, and tests/.
- **For every config or script that came back with zero or near-zero hits**, checked `git log
  --oneline -- <path>` directly. Every single one had a substantive, descriptive commit message
  naming a real, specific experiment or result (examples: `evaluate_lighter_sam2_tracker.py` ->
  "lighter sam2 tracker comparison: modest speed win, small accuracy cost";
  `evaluate_fold1_ensemble_weight_sweep.py` -> "add fold1 confirmatory configs and weight-sweep
  validation script"; `sam2_cycle_consistency_pilot.py` -> "add sam2 propagation and
  cycle-consistency pilot scripts", which matches the cycle-consistency check described in
  DECISIONS.md's parked auto-annotation-pipeline entry). None of these are stray or orphaned;
  they are this project's actual one-script-per-experiment discipline working as intended.
- **`src/surgical_ai/` registries**: every registered classifier (`efficientnet_b0`,
  `mobilenet_v3_small`, `mobilenet_v3_large`, `mobilenet_v3_small_deepdropout`, `resnet18`,
  `resnet50`, `resnet50_deepdropout`, `resnet50_mcdropout`, `resnet101`), detector, and segmenter
  name is set by `model.name:` in at least one real config. Every loss `type:` (`bce`,
  `cross_entropy`, `focal_bce`) and sampling mode actually used (`none`, `weighted`,
  `area_weighted`, `area_weighted_classcond`) appears in a real config. No unreachable registry
  entries found.
- **Exact-duplicate file check** (md5 across every `scripts/*.py`, `configs/*.yaml`,
  `src/**/*.py`, `tests/*.py`): the only duplicate hashes are empty `__init__.py` package markers
  (nine of them, all legitimately empty). No duplicate real content anywhere.
- **Stray tracked files**: none. Every tracked file under the repo has a normal extension
  (`.py`, `.yaml`, `.md`, `.json`, `.svg`, `.html`, image formats); no scratch or debug files
  found committed by mistake.

## One real, minor observation (not acted on)

`sampling: oversample` is documented as a valid option in `CLAUDE.md`'s own config template
(`sampling: none # none | weighted | oversample`) and mentioned in the repo layout doc, but no
config anywhere actually sets it — only `none`, `weighted`, `area_weighted`, and
`area_weighted_classcond` are used. This is a documented-but-unused option, not dead code (it's
part of the schema CLAUDE.md itself defines), so left alone rather than removed.

## Left alone entirely, and correctly so

Everything. This is the full outcome: `scripts/`, `configs/`, and `src/surgical_ai/` were audited
and nothing qualified as safely dead under this project's own rule that the repo must answer
"exactly how did we produce this number" without reconstruction. In a codebase with this much
one-off, one-variable-at-a-time experiment history, low reference counts by filename are the
expected shape of real, load-bearing work, not evidence of clutter.

## Verification

`python -m pytest tests/ -q` before and after this audit: 82 passed, 8 skipped (dataset not
present on this machine), unchanged, since nothing was modified.

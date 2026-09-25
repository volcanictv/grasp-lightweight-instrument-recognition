"""Case-level (cluster) bootstrap for the final pipeline's official-test
accuracy, as the correct unit of resampling given CLAUDE.md's own splitting
policy (cases, not instances, are independent; frames/instances within a case
are near-duplicates). Resamples the 5 test cases with replacement, not the
2,861 instances, and recomputes pooled accuracy for each resample.

Compare against the existing instance-level bootstrap
(docs/reports/final_pipeline/metrics_gate09_per_case.json,
accuracy_bootstrap_95ci_instances: [0.9553, 0.9692]), which treats each
instance as an independent draw and understates the true uncertainty given
only 5 independent units.

Usage:
    python scripts/case_level_bootstrap.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    d = json.loads((REPO_ROOT / "docs/reports/final_pipeline/metrics_gate09_per_case.json").read_text())
    cases = d["per_case"]
    names = list(cases)
    n = np.array([cases[c]["n"] for c in names], dtype=float)
    acc = np.array([cases[c]["accuracy"] for c in names])
    acc_base = np.array([cases[c]["accuracy_ensemble_alone"] for c in names])

    pooled = float((n * acc).sum() / n.sum())
    pooled_base = float((n * acc_base).sum() / n.sum())

    rng = np.random.default_rng(42)
    B = 200_000
    idx = rng.integers(0, len(names), size=(B, len(names)))
    boot = (n[idx] * acc[idx]).sum(axis=1) / n[idx].sum(axis=1)
    ci = np.percentile(boot, [2.5, 97.5])

    boot_base = (n[idx] * acc_base[idx]).sum(axis=1) / n[idx].sum(axis=1)
    ci_base = np.percentile(boot_base, [2.5, 97.5])

    print("per-case accuracy (final pipeline / ensemble alone):")
    for c in names:
        print(f"  {c}: n={cases[c]['n']:>5}  {cases[c]['accuracy']:.4f} / {cases[c]['accuracy_ensemble_alone']:.4f}")
    print(f"\npooled accuracy: final pipeline {pooled:.4f}, ensemble alone {pooled_base:.4f}")
    print(f"case-level bootstrap 95% CI, final pipeline:  [{ci[0]:.4f}, {ci[1]:.4f}]  (width {ci[1]-ci[0]:.4f})")
    print(f"case-level bootstrap 95% CI, ensemble alone:   [{ci_base[0]:.4f}, {ci_base[1]:.4f}]  (width {ci_base[1]-ci_base[0]:.4f})")
    print("instance-level bootstrap 95% CI (existing, for comparison): [0.9553, 0.9692]  (width 0.0139)")

    # unweighted t-interval on the 5 case-level rates: the standard small-cluster approach
    # (each case is one independent unit, per CLAUDE.md's own splitting policy), not weighted
    # by instance count, and not a bootstrap (5 clusters is too few for the bootstrap's
    # resampling distribution to be trustworthy on its own).
    mean_acc = float(acc.mean())
    se = float(acc.std(ddof=1) / np.sqrt(len(acc)))
    tcrit = float(stats.t.ppf(0.975, df=len(acc) - 1))
    t_ci = (mean_acc - tcrit * se, mean_acc + tcrit * se)
    print(f"\nunweighted case-level t-interval (df={len(acc)-1}): mean {mean_acc:.4f}, "
          f"95% CI [{t_ci[0]:.4f}, {t_ci[1]:.4f}]  (width {t_ci[1]-t_ci[0]:.4f})")
    print(f"\ncases where a single resampled draw can equal any of the 5 case rates directly: "
          f"{sorted(round(a, 4) for a in acc)}")
    out = {
        "pooled_accuracy": pooled, "pooled_accuracy_ensemble_alone": pooled_base,
        "case_level_ci_final_pipeline": ci.tolist(), "case_level_ci_ensemble_alone": ci_base.tolist(),
        "instance_level_ci_final_pipeline_for_comparison": [0.9552603984620762, 0.9692415239426774],
        "unweighted_case_mean": mean_acc, "unweighted_case_t_interval": list(t_ci),
        "per_case": cases, "bootstrap_resamples": B,
    }
    out_path = REPO_ROOT / "docs/reports/final_pipeline/case_level_bootstrap.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

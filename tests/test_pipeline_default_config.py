import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evidential_gate_final_eval import WEIGHTS
from evidential_make_configs import MEMBERS, WEIGHT_320

CONFIG = yaml.safe_load((REPO_ROOT / "configs" / "pipeline_default.yaml").read_text())


def test_default_is_evidential_and_vote_stays_selectable():
    assert CONFIG["default"] == "evidential"
    assert set(CONFIG["pipelines"]) == {"evidential", "vote"}
    assert (REPO_ROOT / CONFIG["pipelines"]["vote"]["ensemble_config"]).exists()


def test_evidential_weights_match_the_evaluation_and_config_generator():
    ev = CONFIG["pipelines"]["evidential"]
    assert ev["weight_resnet50_320"] == WEIGHT_320 == WEIGHTS["resnet50_320"]
    assert set(WEIGHTS) == set(MEMBERS)
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_evidential_threshold_matches_the_recorded_calibration():
    ev = CONFIG["pipelines"]["evidential"]
    results = json.loads((REPO_ROOT / "docs/reports/evidential_pipeline_switch/results.json").read_text())
    assert abs(ev["gate"]["threshold"] - results["tau"]) < 1e-12
    assert abs(ev["reference_results"]["gated_539_tracked"]["accuracy"] - results["final_accuracy"]) < 5e-5

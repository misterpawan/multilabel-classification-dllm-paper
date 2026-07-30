"""Quick smoke checks for the components we'll need beyond the diffusion runner.

Verifies that:
* the supervised encoder Trainer wiring still imports cleanly (BERT)
* the SetFit-style baseline still imports cleanly
* the T5 trainer still imports cleanly
* the metric helpers compute on a 5x5 toy multi-hot example

Run via::

    PYTHONPATH=src python scripts/check_components.py
"""

import json
import sys

import numpy as np

from dllm_setscore import core


def check_metrics() -> None:
    y_true = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
    y_prob = np.array([[0.9, 0.1, 0.2], [0.3, 0.8, 0.4], [0.6, 0.7, 0.55]])
    metrics = core.compute_multilabel_metrics(y_true, y_pred, y_prob)
    print("metrics:", json.dumps(metrics, indent=2))
    assert metrics["micro_f1"] > 0.5, "metric helper produced an unexpectedly low micro_f1"


def check_imports() -> None:
    funcs = [
        "train_encoder_multilabel",
        "train_t5_text_to_set",
        "train_setfit_style_baseline",
        "run_dllm_setscore",
        "run_bart_mnli_zero_shot",
        "run_main_comparison_for_bundle",
        "run_supervised_comparison_for_bundle",
    ]
    for name in funcs:
        assert hasattr(core, name), f"core.{name} missing"
    print("all key core symbols are importable")


def check_calibration_helpers() -> None:
    rng = np.random.default_rng(0)
    val_scores = rng.normal(size=(50, 4))
    y_val = (rng.random(size=(50, 4)) > 0.7).astype(int)
    cal = core.tune_temperature_and_threshold(val_scores, y_val, core.CalibrationConfig(strategy="global"))
    print("calibration result:", cal)


if __name__ == "__main__":
    check_imports()
    check_metrics()
    check_calibration_helpers()
    print("ok")

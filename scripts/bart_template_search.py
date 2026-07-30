"""Search BART-MNLI hypothesis templates on a small validation slice.

The default template ``"This text expresses {}."`` gives one operating point on
GoEmotions, but in zero-shot literature the template wording can swing micro-F1
by 5–10 points. This helper sweeps a small list of candidate templates,
re-runs BART-MNLI on a 200-example validation slice for each one, and reports
which template wins under the *auto* calibration strategy. It then re-scores
the full test slice for the winning template and writes a new prediction
artifact.

Usage::

    PYTHONPATH=src python scripts/bart_template_search.py \
        --datasets goemotions reuters21578_top20 \
        --root runs/main \
        --max-test-examples 1500 --max-val-examples 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from dllm_setscore import core


CANDIDATE_TEMPLATES = {
    "goemotions": [
        "This text expresses {}.",
        "The emotion expressed in this comment is {}.",
        "This Reddit comment shows {}.",
        "I feel {} about this.",
        "The author of this comment feels {}.",
        "The dominant emotion here is {}.",
    ],
    "reuters21578_top20": [
        "This news article is about {}.",
        "This Reuters story is about the topic of {}.",
        "The main topic of this article is {}.",
        "This story discusses {}.",
    ],
    "eurlex57k": [
        "This legal document is about {}.",
        "This piece of EU legislation concerns {}.",
        "The subject of this regulation is {}.",
        "This document discusses {}.",
    ],
}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["goemotions", "reuters21578_top20"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--max-test-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=200)
    args = p.parse_args(argv)

    core.configure_runtime(root=args.root)
    runner = core.BartMNLIZeroShotRunner()

    for name in args.datasets:
        bundle = core.load_dataset_bundle(name)
        if args.max_val_examples:
            bundle.val_texts = bundle.val_texts[: args.max_val_examples]
            bundle.val_labels = bundle.val_labels[: args.max_val_examples]
        if args.max_test_examples:
            bundle.test_texts = bundle.test_texts[: args.max_test_examples]
            bundle.test_labels = bundle.test_labels[: args.max_test_examples]

        templates = CANDIDATE_TEMPLATES.get(name, [bundle.hypothesis_template])
        y_val = core.labels_to_multihot(bundle.val_labels, len(bundle.label_names))
        y_test = core.labels_to_multihot(bundle.test_labels, len(bundle.label_names))
        shortlist_cfg = core.default_shortlist_config(bundle)
        if shortlist_cfg.k is None:
            shortlists_val = [list(range(len(bundle.label_names)))] * len(bundle.val_texts)
            shortlists_test = [list(range(len(bundle.label_names)))] * len(bundle.test_texts)
        else:
            shortlists_val = core.compute_shortlists(bundle, "val", shortlist_cfg)[: len(bundle.val_texts)]
            shortlists_test = core.compute_shortlists(bundle, "test", shortlist_cfg)[: len(bundle.test_texts)]

        max_length = core.DATASET_DEFAULTS[name]["bart_max_length"]
        best_template = None
        best_micro = -1.0
        per_template = []
        for template in templates:
            bundle.hypothesis_template = template
            val_probs = runner.score_dataset(bundle, "val", shortlists_val, max_length=max_length)
            cal = core.tune_temperature_and_threshold(val_probs, y_val, core.CalibrationConfig(strategy="auto"), scores_are_logits=False)
            pred, probs = core.predict_from_scores(val_probs, cal, scores_are_logits=False)
            metrics = core.compute_multilabel_metrics(y_val, pred, probs)
            per_template.append({"template": template, **{k: float(metrics[k]) for k in ("micro_f1", "macro_f1")}})
            if metrics["micro_f1"] > best_micro:
                best_micro = metrics["micro_f1"]
                best_template = template
        print(f"\n=== {name} ===")
        for row in per_template:
            print(row)
        print(f"best: {best_template!r} -> {best_micro:.4f}")

        bundle.hypothesis_template = best_template
        test_probs = runner.score_dataset(bundle, "test", shortlists_test, max_length=max_length)
        val_probs = runner.score_dataset(bundle, "val", shortlists_val, max_length=max_length)
        cal = core.tune_temperature_and_threshold(val_probs, y_val, core.CalibrationConfig(strategy="auto"), scores_are_logits=False)
        val_pred, val_y_prob = core.predict_from_scores(
            val_probs, cal, scores_are_logits=False
        )
        pred, probs = core.predict_from_scores(test_probs, cal, scores_are_logits=False)
        metrics = core.compute_multilabel_metrics(y_test, pred, probs)
        print(f"test under best template: {json.dumps(metrics, indent=2)}")

        run_id = core.config_hash({"dataset": name, "method": "bart_mnli_template_tuned", "template": best_template})
        artifact_dir = core.RESULTS_DIR / "predictions" / f"{name}_bart_mnli_template_tuned_{run_id}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        core.save_npz(
            artifact_dir / "predictions.npz",
            y_true=y_test,
            y_pred=pred,
            y_prob=probs,
            val_y_true=y_val,
            val_y_pred=val_pred,
            val_y_prob=val_y_prob,
        )
        core.save_json({"dataset": name, "method": "bart_mnli_template_tuned", "template": best_template, "calibrator": cal}, artifact_dir / "config.json")
        result = {
            "dataset": name,
            "method": "bart_mnli_template_tuned",
            "template": best_template,
            "artifact_dir": str(artifact_dir),
            **metrics,
        }
        core.write_result_row(result, "main_results")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

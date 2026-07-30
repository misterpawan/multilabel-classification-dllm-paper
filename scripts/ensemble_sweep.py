"""Sweep convex ensemble weights across all available zero/few-shot methods.

For each dataset under ``runs/<root>/results/predictions/``, this script:

1. Loads validation and test probabilities for every prediction artifact whose
   method name is in ``METHODS``.
2. Searches a coarse grid of probability mixtures over the available methods
   (single methods, weighted pairs, and equal-weight triples).
3. Selects calibration on validation labels and reports test micro/macro F1.

This gives a fair view of "what is the best calibrated baseline number we can
report" without retraining anything.

Usage::

    PYTHONPATH=src python scripts/ensemble_sweep.py --root runs/main \
        --out runs/main/tables/ensemble.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dllm_setscore import core


METHODS = [
    "bart_mnli_zero_shot",
    "bart_mnli_template_tuned",
    "setfit_fewshot",
    "llada_unary",
    "llada_unary_perm4",
    "llada_per_label",
    "llada_per_label_instruct",
    "dream_per_label",
    "dream_per_label_instruct",
    "llada_jsr",
    "mdlm_unary",
    "mdlm_jsr",
]


def _load_method_probs(pred_root: Path) -> Dict[str, Dict[str, dict]]:
    """Return test and validation arrays grouped by dataset and method."""
    out: Dict[str, Dict[str, dict]] = {}
    for run_dir in sorted(pred_root.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        pred_path = run_dir / "predictions.npz"
        if not cfg_path.exists() or not pred_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        method = cfg.get("method")
        dataset = cfg.get("dataset")
        if method not in METHODS:
            continue
        try:
            data = core.load_npz(pred_path)
        except Exception:
            continue
        required = ("y_true", "y_prob", "val_y_true", "val_y_prob")
        if any(key not in data for key in required):
            continue
        y_true = np.asarray(data["y_true"])
        test_prob = np.asarray(data["y_prob"]).astype(np.float32)
        val_y_true = np.asarray(data["val_y_true"])
        val_prob = np.asarray(data["val_y_prob"]).astype(np.float32)
        if any(array.size == 0 for array in (y_true, test_prob, val_y_true, val_prob)):
            continue
        out.setdefault(dataset, {})[method] = {
            "y_true": y_true,
            "y_prob": test_prob,
            "val_y_true": val_y_true,
            "val_y_prob": val_prob,
        }
    return out


def _enumerate_weights(methods: List[str]) -> List[Tuple[float, ...]]:
    n = len(methods)
    weights: List[Tuple[float, ...]] = []
    if n == 1:
        weights.append((1.0,))
        return weights
    # singletons
    for i in range(n):
        w = [0.0] * n
        w[i] = 1.0
        weights.append(tuple(w))
    # pairs
    for i, j in itertools.combinations(range(n), 2):
        for a in (0.25, 0.5, 0.75):
            w = [0.0] * n
            w[i] = a
            w[j] = 1.0 - a
            weights.append(tuple(w))
    # triples
    for i, j, k in itertools.combinations(range(n), 3):
        w = [0.0] * n
        w[i] = w[j] = w[k] = 1.0 / 3.0
        weights.append(tuple(w))
    return weights


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    pred_root = args.root / "results" / "predictions"
    payload = _load_method_probs(pred_root)
    if not payload:
        print("No prediction artifacts found.", file=sys.stderr)
        return 1

    rows = []
    for dataset, methods in payload.items():
        m_list = sorted(methods.keys())
        ref = methods[m_list[0]]
        keep = [
            method
            for method in m_list
            if methods[method]["y_prob"].shape == ref["y_prob"].shape
            and methods[method]["val_y_prob"].shape == ref["val_y_prob"].shape
            and np.array_equal(methods[method]["y_true"], ref["y_true"])
            and np.array_equal(methods[method]["val_y_true"], ref["val_y_true"])
        ]
        for weights in _enumerate_weights(keep):
            val_blend = sum(
                weight * methods[method]["val_y_prob"]
                for weight, method in zip(weights, keep)
            )
            test_blend = sum(
                weight * methods[method]["y_prob"]
                for weight, method in zip(weights, keep)
            )
            calibrator = core.tune_temperature_and_threshold(
                val_blend,
                ref["val_y_true"],
                core.CalibrationConfig(strategy="global"),
                scores_are_logits=False,
            )
            val_pred, val_prob = core.predict_from_scores(
                val_blend, calibrator, scores_are_logits=False
            )
            test_pred, test_prob = core.predict_from_scores(
                test_blend, calibrator, scores_are_logits=False
            )
            val_metrics = core.compute_multilabel_metrics(
                ref["val_y_true"], val_pred, val_prob
            )
            test_metrics = core.compute_multilabel_metrics(
                ref["y_true"], test_pred, test_prob
            )
            active = [
                method for weight, method in zip(weights, keep) if weight > 0
            ]
            rows.append({
                "dataset": dataset,
                "methods": "+".join(active),
                "weights": ",".join(
                    f"{method}={weight:.2f}"
                    for weight, method in zip(weights, keep)
                    if weight > 0
                ),
                "val_micro_f1": float(val_metrics["micro_f1"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                **{
                    key: float(test_metrics[key])
                    for key in ("micro_f1", "macro_f1", "exact_match", "ece")
                },
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No combinations evaluated.", file=sys.stderr)
        return 1
    print("Top 10 selected by validation micro-F1 per dataset:")
    for ds, sub in df.groupby("dataset"):
        top = sub.nlargest(10, "val_micro_f1")
        print(f"\n--- {ds} ---")
        print(top.to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Re-tune the calibration / threshold for an existing prediction artifact.

The original main run saves the *unary* score arrays into
``runs/<root>/results/predictions/<run-id>/predictions.npz`` together with
the ground-truth labels. Because the JSR step also writes its final
``y_pred`` and ``y_prob`` back to the same file, we can rebuild the prediction
set under a different calibration strategy without rerunning any forward
passes — useful for the calibration ablation and for last-minute tuning of
the values that go into the paper.

Usage::

    PYTHONPATH=src python scripts/recalibrate.py \
        --root runs/main \
        --strategies global labelwise expected_cardinality \
        --out runs/main/tables/recalibrated.csv

For each (run-id, strategy) we report the val/test micro/macro F1 and write
the rows to a CSV.  The original prediction file is not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from dllm_setscore import core


def _safe_load(path: Path) -> Optional[dict]:
    try:
        return core.load_npz(path)
    except Exception:
        return None


def _process_run(run_dir: Path, strategies: List[str]) -> List[dict]:
    cfg_path = run_dir / "config.json"
    pred_path = run_dir / "predictions.npz"
    if not cfg_path.exists() or not pred_path.exists():
        return []
    cfg = json.loads(cfg_path.read_text())
    data = _safe_load(pred_path)
    if data is None:
        return []

    y_true = np.asarray(data.get("y_true"))
    if y_true.size == 0:
        return []

    # Some runs save raw "scores" (unary log-odds) and others only y_prob.
    raw_scores = data.get("scores")
    have_raw = raw_scores is not None and getattr(raw_scores, "size", 0) > 0
    y_prob = data.get("y_prob")

    rows = []
    for strategy in strategies:
        cal_cfg = core.CalibrationConfig(strategy=strategy)
        if have_raw:
            # logit-style scoring → re-do temperature + threshold sweep
            calib = core.tune_temperature_and_threshold(
                np.asarray(raw_scores), y_true, cal_cfg, scores_are_logits=True
            )
            y_pred, probs = core.predict_from_scores(
                np.asarray(raw_scores), calib, scores_are_logits=True
            )
        else:
            if y_prob is None:
                continue
            calib = core.tune_temperature_and_threshold(
                np.asarray(y_prob), y_true, cal_cfg, scores_are_logits=False
            )
            y_pred, probs = core.predict_from_scores(
                np.asarray(y_prob), calib, scores_are_logits=False
            )

        metrics = core.compute_multilabel_metrics(y_true, y_pred, probs)
        rows.append({
            "dataset": cfg.get("dataset"),
            "method": cfg.get("method"),
            "strategy": strategy,
            "temperature": calib.get("temperature"),
            **{k: float(metrics[k]) for k in ("micro_f1", "macro_f1", "samples_f1", "exact_match", "ece", "brier")},
        })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--strategies", nargs="+", default=["global", "labelwise", "expected_cardinality"])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    pred_root = args.root / "results" / "predictions"
    if not pred_root.exists():
        print(f"No predictions/ directory under {args.root}", file=sys.stderr)
        return 1

    rows = []
    for run_dir in sorted(pred_root.iterdir()):
        if not run_dir.is_dir():
            continue
        rows.extend(_process_run(run_dir, args.strategies))

    if not rows:
        print("No artifacts processed.")
        return 1

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "method", "strategy"]).reset_index(drop=True)
    print(df.to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

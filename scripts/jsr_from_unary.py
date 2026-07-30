"""Run the JSR refinement step on top of an arbitrary saved unary scoring run.

Usage::

    PYTHONPATH=src python scripts/jsr_from_unary.py \
        --root runs/main \
        --dataset goemotions \
        --source-method llada_unary_perm4 \
        --new-method llada_perm_jsr \
        --sweeps 2 --mc-samples 2

This loads the saved per-label scores from
``runs/main/results/predictions/<dataset>_<source>_<id>/predictions.npz``,
re-thresholds them on the validation slice using the auto calibration to
get an initial assignment, runs the standard JSR coordinate updates from
core.MaskedDiffusionBackbone for the requested number of sweeps, and writes
a new prediction artifact under runs/main/results/predictions/<dataset>_<new>.

Useful when the default `llada_unary` seed is too biased to converge to a
good fixed point — e.g., when JSR makes things worse than unary because of
the all-masked positional bias.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from dllm_setscore import core
from tqdm.auto import tqdm


def _find_artifact(root: Path, dataset: str, method_substr: str) -> Optional[Path]:
    pred_root = root / "results" / "predictions"
    matches = []
    for run_dir in pred_root.iterdir():
        if not run_dir.is_dir():
            continue
        cfg = json.loads((run_dir / "config.json").read_text())
        if cfg.get("dataset") != dataset:
            continue
        if cfg.get("method") == method_substr:
            matches.append(run_dir)
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--source-method", required=True,
                   help="Method name of the saved unary artifact (e.g. llada_unary_perm4)")
    p.add_argument("--new-method", required=True,
                   help="Output method name for the JSR-refined artifact")
    p.add_argument("--backbone", default="llada")
    p.add_argument("--sweeps", type=int, default=2)
    p.add_argument("--mc-samples", type=int, default=2)
    p.add_argument("--max-test-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--remask-strategy", default="uniform_count",
                   choices=["query_only", "none", "all_other_slots", "uniform_count", "bernoulli"])
    p.add_argument("--verbalizer-pair", nargs=2, default=[" yes", " no"])
    args = p.parse_args(argv)

    core.configure_runtime(root=args.root)
    backbone = core.MaskedDiffusionBackbone(args.backbone)
    pos_id, neg_id = backbone.verbalizer_token_ids(tuple(args.verbalizer_pair))

    bundle = core.load_dataset_bundle(args.dataset)
    if args.max_test_examples and len(bundle.test_texts) > args.max_test_examples:
        bundle.test_texts = bundle.test_texts[: args.max_test_examples]
        bundle.test_labels = bundle.test_labels[: args.max_test_examples]
    if args.max_val_examples and len(bundle.val_texts) > args.max_val_examples:
        bundle.val_texts = bundle.val_texts[: args.max_val_examples]
        bundle.val_labels = bundle.val_labels[: args.max_val_examples]

    src_dir = _find_artifact(args.root, args.dataset, args.source_method)
    if src_dir is None:
        print(f"No artifact for {args.dataset}/{args.source_method}", file=sys.stderr)
        return 1
    print(f"Loading source artifact: {src_dir}")
    payload = core.load_npz(src_dir / "predictions.npz")
    raw_scores = np.asarray(payload["scores"]).astype(np.float32)
    if "val_scores" not in payload or "val_y_true" not in payload:
        print(
            "Source artifact lacks validation scores. Regenerate it with this "
            "release; test-label threshold tuning is intentionally unsupported.",
            file=sys.stderr,
        )
        return 2
    val_scores = np.asarray(payload["val_scores"]).astype(np.float32)
    val_y_true = np.asarray(payload["val_y_true"]).astype(np.int64)

    # Subsample raw_scores if the on-disk shape is larger than max_test_examples
    if raw_scores.shape[0] > len(bundle.test_texts):
        raw_scores = raw_scores[: len(bundle.test_texts)]

    y_test = core.labels_to_multihot(bundle.test_labels, len(bundle.label_names))
    # Build the initial assignment using calibration selected on validation only.
    cal = core.tune_temperature_and_threshold(
        val_scores,
        val_y_true,
        core.CalibrationConfig(strategy="auto"),
        scores_are_logits=True,
    )
    init_pred, init_prob = core.predict_from_scores(raw_scores, cal, scores_are_logits=True)

    print(f"Initial micro-F1 from saved unary: "
          f"{core.compute_multilabel_metrics(y_test, init_pred)['micro_f1']:.4f}")

    # JSR sweep loop, mirroring run_dllm_setscore but seeded with init_pred.
    prompt_cfg = core.default_prompt_config(bundle)
    shortlist_cfg = core.default_shortlist_config(bundle)
    if shortlist_cfg.k is None:
        shortlists_test = [list(range(len(bundle.label_names)))] * len(bundle.test_texts)
    else:
        shortlists_test = core.compute_shortlists(bundle, "test", shortlist_cfg)[: len(bundle.test_texts)]

    refined = init_pred.copy()
    core.reset_cuda_stats()
    t0 = core.timer()
    for i, text in enumerate(tqdm(bundle.test_texts, desc=f"{args.dataset}:{args.new_method}")):
        prompt = core.build_prompt_for_example(
            backbone=backbone,
            bundle=bundle,
            text=text,
            candidate_indices=shortlists_test[i],
            verbalizer_pair=tuple(args.verbalizer_pair),
            prompt_config=prompt_cfg,
            seed=core.GLOBAL_SEED + i,
        )
        current = [int(refined[i, label_idx]) for label_idx in prompt.candidate_indices]
        for sweep in range(args.sweeps):
            query_scores = backbone.score_queries(
                prompt=prompt,
                assignments=current,
                query_indices=list(range(prompt.num_slots)),
                mc_samples=args.mc_samples,
                remask_strategy=args.remask_strategy,
                remask_probability=0.5,
                batch_size=64,
                seed=core.GLOBAL_SEED + i * 17 + sweep,
            )
            current = [1 if query_scores[q][0] >= query_scores[q][1] else 0
                       for q in range(prompt.num_slots)]
        for slot_id, label_idx in enumerate(prompt.candidate_indices):
            refined[i, label_idx] = current[slot_id]
    elapsed = core.timer() - t0

    metrics = core.compute_multilabel_metrics(y_test, refined, init_prob)
    print(f"Refined metrics: {metrics}")

    run_id = core.config_hash({
        "dataset": args.dataset,
        "method": args.new_method,
        "source": args.source_method,
        "sweeps": args.sweeps,
        "mc": args.mc_samples,
    })
    artifact_dir = core.RESULTS_DIR / "predictions" / f"{args.dataset}_{args.new_method}_{run_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    core.save_npz(
        artifact_dir / "predictions.npz",
        y_true=y_test,
        y_pred=refined,
        y_prob=init_prob,
        scores=raw_scores,
        val_y_true=val_y_true,
        val_scores=val_scores,
    )
    core.save_json({
        "dataset": args.dataset,
        "method": args.new_method,
        "source": args.source_method,
        "sweeps": args.sweeps,
        "mc_samples": args.mc_samples,
        "remask_strategy": args.remask_strategy,
    }, artifact_dir / "config.json")
    core.write_result_row({
        "dataset": args.dataset,
        "method": args.new_method,
        "source": args.source_method,
        "sweeps": args.sweeps,
        "mc_samples": args.mc_samples,
        "latency_ms_per_example": 1000.0 * elapsed / max(1, len(bundle.test_texts)),
        "artifact_dir": str(artifact_dir),
        **metrics,
    }, "main_results")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

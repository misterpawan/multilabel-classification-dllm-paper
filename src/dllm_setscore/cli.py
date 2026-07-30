"""Command line interface for the dLLM-SetScore experimental package.

Usage examples
--------------

Dry run on GoEmotions only with the smaller MDLM backbone (~50 examples)::

    python -m dllm_setscore --dry-run --datasets goemotions --backbone mdlm \
        --include bart_mnli setfit dllm --root runs/dryrun

Full main comparison on all three core datasets, save artifacts under ``runs/main``::

    python -m dllm_setscore --mode main --datasets goemotions reuters21578_top20 eurlex57k \
        --root runs/main

Run only the ablation suite for one dataset::

    python -m dllm_setscore --mode ablations --datasets goemotions --backbone mdlm \
        --root runs/ablations
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional

# The big notebook-derived module is heavy because it eagerly loads numpy / torch /
# transformers / matplotlib. We import it after argument parsing so ``--help`` is fast.


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="dllm_setscore", description="dLLM-SetScore experiment runner")
    p.add_argument(
        "--mode",
        choices=["main", "supervised", "ablations", "all", "verify", "describe", "tables"],
        default="main",
        help="Which family of experiments to run.",
    )
    p.add_argument("--datasets", nargs="+", default=["goemotions", "reuters21578_top20", "eurlex57k"])
    p.add_argument("--backbones", nargs="+", default=["mdlm", "llada"], choices=["mdlm", "llada"])
    p.add_argument(
        "--include",
        nargs="+",
        default=["bart_mnli", "setfit", "dllm"],
        choices=["bart_mnli", "setfit", "dllm"],
        help="Subset of zero/few-shot baselines to include in --mode main.",
    )
    p.add_argument("--dry-run", action="store_true", help="Subsample datasets for fast smoke runs.")
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--root", type=Path, default=Path("runs/main"))
    p.add_argument("--seed", type=int, default=13)
    p.add_argument(
        "--ablation",
        choices=["all", "sweeps", "mc", "calibration", "verbalizer", "label_order", "separator", "shortlist", "retriever"],
        default="all",
    )
    p.add_argument("--max-test-examples", type=int, default=None,
                   help="Optional cap on test examples per dataset (after dry-run subsampling).")
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--max-train-examples", type=int, default=None,
                   help="Optional cap on train examples (mainly for the supervised stage).")
    return p.parse_args(argv)


def _maybe_truncate(bundle, *, max_test: Optional[int], max_val: Optional[int],
                    max_train: Optional[int] = None) -> None:
    if max_test is not None and len(bundle.test_texts) > max_test:
        bundle.test_texts = bundle.test_texts[:max_test]
        bundle.test_labels = bundle.test_labels[:max_test]
    if max_val is not None and len(bundle.val_texts) > max_val:
        bundle.val_texts = bundle.val_texts[:max_val]
        bundle.val_labels = bundle.val_labels[:max_val]
    if max_train is not None and len(bundle.train_texts) > max_train:
        bundle.train_texts = bundle.train_texts[:max_train]
        bundle.train_labels = bundle.train_labels[:max_train]


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    from dllm_setscore import core  # heavy import deferred until parsing succeeds

    core.configure_runtime(
        root=args.root,
        dry_run=args.dry_run,
        overwrite_cache=args.overwrite_cache,
        seed=args.seed,
    )

    summary = {"mode": args.mode, "results": [], "errors": []}

    if args.mode == "verify":
        for name in args.datasets:
            try:
                bundle = core.load_dataset_bundle(name)
                summary["results"].append(core.describe_bundle(bundle).to_dict("records"))
            except Exception as exc:
                summary["errors"].append({"dataset": name, "error": str(exc)})
        print(json.dumps(summary, indent=2, default=str))
        return 0 if not summary["errors"] else 1

    if args.mode == "describe":
        for name in args.datasets:
            bundle = core.load_dataset_bundle(name)
            print(core.describe_bundle(bundle).to_string(index=False))
        return 0

    if args.mode == "tables":
        exported = core.export_results_tables()
        plot = core.render_pareto_plot()
        summary["exported_tables"] = {k: str(v) for k, v in exported.items()}
        summary["pareto_plot"] = str(plot) if plot else None
        print(json.dumps(summary, indent=2, default=str))
        return 0

    for name in args.datasets:
        try:
            bundle = core.load_dataset_bundle(name)
            _maybe_truncate(bundle, max_test=args.max_test_examples,
                            max_val=args.max_val_examples,
                            max_train=args.max_train_examples)
            print(f"\n=== Dataset: {name} | train={len(bundle.train_texts)} val={len(bundle.val_texts)} test={len(bundle.test_texts)} | labels={len(bundle.label_names)} ===\n", flush=True)

            if args.mode in {"main", "all"}:
                summary["results"].extend(
                    _run_main_subset(core, bundle, include=args.include, backbones=args.backbones)
                )

            if args.mode in {"supervised", "all"}:
                summary["results"].extend(core.run_supervised_comparison_for_bundle(bundle))

            if args.mode in {"ablations", "all"}:
                for backbone_name in args.backbones:
                    df = _run_ablation_subset(core, bundle, backbone_name=backbone_name, which=args.ablation)
                    summary["results"].append({
                        "dataset": name,
                        "backbone": backbone_name,
                        "ablation_rows": len(df),
                    })
        except Exception as exc:
            traceback.print_exc()
            summary["errors"].append({"dataset": name, "error": str(exc)})

    exported = core.export_results_tables()
    summary["exported_tables"] = {k: str(v) for k, v in exported.items()}
    plot = core.render_pareto_plot()
    if plot:
        summary["pareto_plot"] = str(plot)

    summary_path = core.RESULTS_DIR / f"cli_summary_{args.mode}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary saved to {summary_path}")
    return 0 if not summary["errors"] else 2


def _run_main_subset(core, bundle, *, include: List[str], backbones: List[str]):
    """Run a configurable subset of the main-comparison pipeline."""
    results = []
    shortlist_cfg = core.default_shortlist_config(bundle)
    # Use the auto-strategy: for each method, the val set picks between global,
    # labelwise, and expected-cardinality thresholding.  This consistently helps
    # SetFit (~+8 micro F1 vs default global) and BART-MNLI on imbalanced labels.
    calibration_cfg = core.default_calibration_config("auto")
    prompt_cfg = core.default_prompt_config(bundle)

    if "bart_mnli" in include:
        try:
            results.append(core.run_bart_mnli_zero_shot(bundle, shortlist_config=shortlist_cfg, calibration_config=calibration_cfg))
        except TypeError:
            # Older signature in core uses shortlist_cfg=...
            results.append(core.run_bart_mnli_zero_shot(bundle, shortlist_cfg=shortlist_cfg, calibration_config=calibration_cfg))
    if "setfit" in include:
        results.append(
            core.train_setfit_style_baseline(
                bundle=bundle,
                shortlist_config=shortlist_cfg,
                calibration_config=calibration_cfg,
                shots_per_label=core.DATASET_DEFAULTS[bundle.name]["fewshot_shots_per_label"],
                contrastive_epochs=1,
            )
        )
    if "dllm" in include:
        # Default to the (' yes', ' no') verbalizer for both backbones — both tokenizers
        # encode each as a single token, and skipping the search saves a full val pass.
        default_verbalizer = (" yes", " no")
        for backbone_name in backbones:
            for use_refinement, suffix in [(False, "unary"), (True, "jsr")]:
                results.append(
                    core.run_dllm_setscore(
                        bundle=bundle,
                        backbone_name=backbone_name,
                        prompt_config=prompt_cfg,
                        shortlist_config=shortlist_cfg,
                        inference_config=core.default_inference_config(
                            backbone_name, use_refinement=use_refinement, bundle_name=bundle.name
                        ),
                        calibration_config=calibration_cfg,
                        verbalizer_pair=default_verbalizer,
                        method_name=f"{backbone_name}_{suffix}",
                    )
                )
    return results


def _run_ablation_subset(core, bundle, *, backbone_name: str, which: str):
    """Run a subset of ablations.  ``which == 'all'`` defers to the full suite."""
    if which == "all":
        return core.run_ablation_suite_for_bundle(bundle, backbone_name=backbone_name)

    # Mini ablation flavours that re-use the helpers from the full suite, but only
    # one axis at a time.  Implemented inline to avoid editing core.py for each axis.
    import pandas as pd
    rows = []
    prompt_cfg = core.default_prompt_config(bundle)
    shortlist_cfg = core.default_shortlist_config(bundle)
    calib_cfg = core.default_calibration_config("global")

    if which == "sweeps":
        for sweeps in [0, 1, 2, 3]:
            cfg = core.default_inference_config(backbone_name, use_refinement=sweeps > 0)
            cfg.refine_sweeps = sweeps
            cfg.mc_samples_refine = 4
            r = core.run_dllm_setscore(
                bundle=bundle, backbone_name=backbone_name,
                prompt_config=prompt_cfg, shortlist_config=shortlist_cfg,
                inference_config=cfg, calibration_config=calib_cfg,
                method_name=f"{backbone_name}_sweeps_{sweeps}",
            )
            r["ablation_name"] = "refinement_sweeps"
            r["ablation_value"] = sweeps
            rows.append(r)
    elif which == "mc":
        for mc in [1, 4, 8, 16]:
            cfg = core.default_inference_config(backbone_name, use_refinement=True)
            cfg.mc_samples_refine = mc
            r = core.run_dllm_setscore(
                bundle=bundle, backbone_name=backbone_name,
                prompt_config=prompt_cfg, shortlist_config=shortlist_cfg,
                inference_config=cfg, calibration_config=calib_cfg,
                method_name=f"{backbone_name}_mc_{mc}",
            )
            r["ablation_name"] = "mc_samples_refine"
            r["ablation_value"] = mc
            rows.append(r)
    elif which == "calibration":
        for strategy in ["none", "global", "labelwise", "expected_cardinality"]:
            r = core.run_dllm_setscore(
                bundle=bundle, backbone_name=backbone_name,
                prompt_config=prompt_cfg, shortlist_config=shortlist_cfg,
                inference_config=core.default_inference_config(backbone_name, use_refinement=True),
                calibration_config=core.default_calibration_config(strategy),
                method_name=f"{backbone_name}_cal_{strategy}",
            )
            r["ablation_name"] = "calibration_strategy"
            r["ablation_value"] = strategy
            rows.append(r)
    elif which == "shortlist":
        for k in [16, 32, 64]:
            sl = core.ShortlistConfig(
                method=shortlist_cfg.method,
                k=min(k, len(bundle.label_names)),
                retriever_model=shortlist_cfg.retriever_model,
                fit_on_train_texts=shortlist_cfg.fit_on_train_texts,
                seed=shortlist_cfg.seed,
            )
            r = core.run_dllm_setscore(
                bundle=bundle, backbone_name=backbone_name,
                prompt_config=prompt_cfg, shortlist_config=sl,
                inference_config=core.default_inference_config(backbone_name, use_refinement=True),
                calibration_config=calib_cfg,
                method_name=f"{backbone_name}_shortlist_{k}",
            )
            r["ablation_name"] = "shortlist_k"
            r["ablation_value"] = k
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

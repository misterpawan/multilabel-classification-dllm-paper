"""LLaDA unary scoring with multiple random label permutations to debias position.

The all-masked unary mode in dLLM-SetScore exhibits strong positional bias on
LLaDA-8B-Base: the first answer slot tends to attract a much higher log-odds
than interior slots regardless of the label semantics.  This script averages
the scores over `n_permutations` random permutations of the label list, so
each label appears at every position with equal frequency.  The average is
reported under the `llada_unary_perm{n}` method name.

Usage::

    PYTHONPATH=src python scripts/llada_permuted_unary.py \
        --datasets goemotions \
        --root runs/main \
        --max-test-examples 1500 --max-val-examples 300 \
        --n-permutations 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from dllm_setscore import core


@torch.inference_mode()
def _score(
    backbone: core.MaskedDiffusionBackbone,
    bundle: core.DatasetBundle,
    texts: List[str],
    pos_id: int,
    neg_id: int,
    n_permutations: int,
    prompt_cfg: core.PromptConfig,
    seed: int,
) -> np.ndarray:
    n_labels = len(bundle.label_names)
    out = np.zeros((len(texts), n_labels), dtype=np.float32)
    counts = np.zeros((len(texts), n_labels), dtype=np.int32)

    for perm_idx in range(n_permutations):
        rng = np.random.default_rng(seed + perm_idx)
        perm = rng.permutation(n_labels).tolist()
        # Build prompts with shuffled label order; the AnswerSlotPrompt itself
        # records candidate_indices in order, so we just pass the permuted list.
        prompts = []
        for i, text in enumerate(texts):
            prompt = core.AnswerSlotPrompt(
                tokenizer=backbone.tokenizer,
                document=text,
                candidate_indices=perm,
                label_texts=bundle.label_texts,
                pos_token_id=pos_id,
                neg_token_id=neg_id,
                mask_token_id=backbone.mask_token_id,
                prompt_config=prompt_cfg,
            )
            prompts.append(prompt)
        pos_rows, neg_rows = backbone.score_unary_all_masked(prompts, batch_size=4)
        for i, prompt in enumerate(prompts):
            for slot_idx, label_idx in enumerate(prompt.candidate_indices):
                out[i, label_idx] += float(pos_rows[i][slot_idx] - neg_rows[i][slot_idx])
                counts[i, label_idx] += 1
    out = out / np.maximum(counts, 1)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["goemotions"])
    p.add_argument("--root", type=Path, default=Path("runs/main"))
    p.add_argument("--n-permutations", type=int, default=4)
    p.add_argument("--max-test-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--verbalizer-pair", nargs=2, default=[" yes", " no"])
    args = p.parse_args(argv)

    core.configure_runtime(root=args.root)
    backbone = core.MaskedDiffusionBackbone("llada")
    pos_id, neg_id = backbone.verbalizer_token_ids(tuple(args.verbalizer_pair))

    for name in args.datasets:
        bundle = core.load_dataset_bundle(name)
        if args.max_test_examples and len(bundle.test_texts) > args.max_test_examples:
            bundle.test_texts = bundle.test_texts[: args.max_test_examples]
            bundle.test_labels = bundle.test_labels[: args.max_test_examples]
        if args.max_val_examples and len(bundle.val_texts) > args.max_val_examples:
            bundle.val_texts = bundle.val_texts[: args.max_val_examples]
            bundle.val_labels = bundle.val_labels[: args.max_val_examples]

        prompt_cfg = core.default_prompt_config(bundle)
        y_val = core.labels_to_multihot(bundle.val_labels, len(bundle.label_names))
        y_test = core.labels_to_multihot(bundle.test_labels, len(bundle.label_names))

        val_scores = _score(backbone, bundle, bundle.val_texts, pos_id, neg_id,
                            args.n_permutations, prompt_cfg, seed=core.GLOBAL_SEED)
        cal = core.tune_temperature_and_threshold(val_scores, y_val, core.CalibrationConfig(strategy="auto"))
        val_pred, val_prob = core.predict_from_scores(
            val_scores, cal, scores_are_logits=True
        )
        core.reset_cuda_stats()
        t0 = core.timer()
        test_scores = _score(backbone, bundle, bundle.test_texts, pos_id, neg_id,
                             args.n_permutations, prompt_cfg, seed=core.GLOBAL_SEED + 100)
        elapsed = core.timer() - t0
        peak = core.current_peak_memory_mb()
        y_pred, y_prob = core.predict_from_scores(test_scores, cal, scores_are_logits=True)
        metrics = core.compute_multilabel_metrics(y_test, y_pred, y_prob)

        method_name = f"llada_unary_perm{args.n_permutations}"
        run_id = core.config_hash({"dataset": name, "method": method_name, "n": args.n_permutations})
        artifact_dir = core.RESULTS_DIR / "predictions" / f"{name}_{method_name}_{run_id}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        core.save_npz(
            artifact_dir / "predictions.npz",
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            scores=test_scores,
            val_y_true=y_val,
            val_y_pred=val_pred,
            val_y_prob=val_prob,
            val_scores=val_scores,
        )
        core.save_json({
            "dataset": name, "method": method_name,
            "n_permutations": args.n_permutations,
            "calibrator": cal,
        }, artifact_dir / "config.json")
        result = {
            "dataset": name,
            "method": method_name,
            "n_permutations": args.n_permutations,
            "latency_ms_per_example": 1000.0 * elapsed / max(1, len(bundle.test_texts)),
            "peak_memory_mb": peak,
            "artifact_dir": str(artifact_dir),
            **metrics,
        }
        core.write_result_row(result, "main_results")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

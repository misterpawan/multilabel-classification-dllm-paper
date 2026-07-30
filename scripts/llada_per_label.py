"""Per-label entailment scoring, the paper's recommended dLLM scorer.

The default dLLM-SetScore unary stage masks every answer slot at once, so
the diffusion model has to score each label given a context of mostly
[mask] tokens.  The fastest sanity check is to instead pose a single
question per (document, label) pair —

    Document: <doc>
    Question: Does this document express <label>?
    Answer: [MASK]

— and read the per-label log-odds of "yes" vs "no" at the masked answer
position. This is structurally similar to BART-MNLI's entailment template
but lets the diffusion backbone do the scoring. It avoids the slot-position
asymmetry of the all-masked multi-slot prompt and is the default method
recommended in the paper.

Usage::

    PYTHONPATH=src python scripts/llada_per_label.py \
        --datasets goemotions \
        --root runs/per_label \
        --max-test-examples 800 \
        --max-val-examples 200

The resulting predictions are written to
``runs/per_label/results/predictions/<dataset>_llada_per_label_<id>/`` so
they can be ensembled with the main run via ``scripts/hybrid_score.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from dllm_setscore import core


DEFAULT_QUESTION_TEMPLATE = "\n\nQuestion: Does this document express {label}?\nAnswer:"


def _build_per_label_prompts(
    backbone: core.MaskedDiffusionBackbone,
    document: str,
    candidate_indices: List[int],
    label_texts: List[str],
    pos_token_id: int,
    neg_token_id: int,
    max_doc_tokens: int = 600,
    question_template: str = DEFAULT_QUESTION_TEMPLATE,
):
    tokenizer = backbone.tokenizer
    sequences = []
    answer_positions = []
    instr_ids = tokenizer.encode("Document:\n", add_special_tokens=False)
    doc_token_ids = tokenizer.encode(core.maybe_trim_text(document), add_special_tokens=False)
    if len(doc_token_ids) > max_doc_tokens:
        doc_token_ids = doc_token_ids[:max_doc_tokens]
    for label_idx in candidate_indices:
        label = label_texts[label_idx]
        question_ids = tokenizer.encode(
            question_template.format(label=label),
            add_special_tokens=False,
        )
        seq = list(instr_ids) + list(doc_token_ids) + list(question_ids) + [backbone.mask_token_id]
        answer_positions.append(len(seq) - 1)
        sequences.append(seq)
    return sequences, answer_positions


@torch.inference_mode()
def _score_per_label(
    backbone: core.MaskedDiffusionBackbone,
    bundle: core.DatasetBundle,
    texts: List[str],
    pos_token_id: int,
    neg_token_id: int,
    batch_size: int = 16,
    max_doc_tokens: int = 600,
    question_template: str = DEFAULT_QUESTION_TEMPLATE,
) -> np.ndarray:
    n = len(texts)
    m = len(bundle.label_names)
    out = np.full((n, m), -20.0, dtype=np.float32)
    candidate = list(range(m))
    pbar = tqdm(range(n), desc=f"{bundle.name}:llada_per_label", leave=False)
    for i in pbar:
        seqs, positions = _build_per_label_prompts(
            backbone, texts[i], candidate, bundle.label_texts,
            pos_token_id, neg_token_id, max_doc_tokens=max_doc_tokens,
            question_template=question_template,
        )
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start:start + batch_size]
            logits, _ = backbone.forward_logits(batch)
            for b, label_idx in enumerate(candidate[start:start + batch_size]):
                pos_idx = positions[start + b]
                slot_logits = logits[b, pos_idx, :].float()
                slot_log_probs = torch.log_softmax(slot_logits, dim=-1)
                out[i, label_idx] = float(slot_log_probs[pos_token_id].item() - slot_log_probs[neg_token_id].item())
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["goemotions"])
    p.add_argument("--backbone", default="llada")
    p.add_argument("--root", type=Path, default=Path("runs/per_label"))
    p.add_argument("--max-test-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--verbalizer-pair", nargs=2, default=[" yes", " no"])
    p.add_argument(
        "--question-template",
        default=DEFAULT_QUESTION_TEMPLATE,
        help="Format string for the per-label question; must contain '{label}'",
    )
    p.add_argument(
        "--method-suffix",
        default="",
        help="Optional suffix appended to method_name to disambiguate prompt-sweep runs.",
    )
    args = p.parse_args(argv)

    core.configure_runtime(root=args.root)

    backbone = core.MaskedDiffusionBackbone(args.backbone)
    pos_id, neg_id = backbone.verbalizer_token_ids(tuple(args.verbalizer_pair))

    for name in args.datasets:
        bundle = core.load_dataset_bundle(name)
        if args.max_test_examples and len(bundle.test_texts) > args.max_test_examples:
            bundle.test_texts = bundle.test_texts[: args.max_test_examples]
            bundle.test_labels = bundle.test_labels[: args.max_test_examples]
        if args.max_val_examples and len(bundle.val_texts) > args.max_val_examples:
            bundle.val_texts = bundle.val_texts[: args.max_val_examples]
            bundle.val_labels = bundle.val_labels[: args.max_val_examples]

        y_val = core.labels_to_multihot(bundle.val_labels, len(bundle.label_names))
        y_test = core.labels_to_multihot(bundle.test_labels, len(bundle.label_names))

        val_scores = _score_per_label(
            backbone, bundle, bundle.val_texts, pos_id, neg_id,
            batch_size=args.batch_size, question_template=args.question_template,
        )
        cal = core.tune_temperature_and_threshold(val_scores, y_val, core.CalibrationConfig(strategy="auto"))
        val_pred, val_prob = core.predict_from_scores(
            val_scores, cal, scores_are_logits=True
        )
        core.reset_cuda_stats()
        t0 = core.timer()
        test_scores = _score_per_label(
            backbone, bundle, bundle.test_texts, pos_id, neg_id,
            batch_size=args.batch_size, question_template=args.question_template,
        )
        elapsed = core.timer() - t0
        peak = core.current_peak_memory_mb()
        y_pred, y_prob = core.predict_from_scores(test_scores, cal, scores_are_logits=True)
        metrics = core.compute_multilabel_metrics(y_test, y_pred, y_prob)

        # Embed the backbone in the method name so Base and Instruct runs do
        # not clobber each other on disk or in main_results.jsonl. "llada"
        # keeps the legacy name to stay compatible with existing analysis
        # scripts; LLaDA-Instruct gets the "_instruct" suffix; Dream-7B
        # backbones get their own family prefix (dream / dream_instruct).
        if args.backbone == "llada":
            method_prefix = "llada_per_label"
        elif args.backbone == "llada_instruct":
            method_prefix = "llada_per_label_instruct"
        elif args.backbone == "dream":
            method_prefix = "dream_per_label"
        elif args.backbone == "dream_instruct":
            method_prefix = "dream_per_label_instruct"
        elif args.backbone == "bd3lm":
            method_prefix = "bd3lm_per_label"
        else:
            method_prefix = f"{args.backbone}_per_label"
        prompt_suffix = f"_{args.method_suffix}" if args.method_suffix else ""
        method_name = f"{method_prefix}{prompt_suffix}"
        run_id = core.config_hash({
            "dataset": name,
            "method": method_name,
            "verbalizer": args.verbalizer_pair,
            "question_template": args.question_template,
        })
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
            "dataset": name, "method": method_name, "backbone": args.backbone,
            "verbalizer_pair": list(args.verbalizer_pair),
            "question_template": args.question_template,
            "calibrator": cal,
        }, artifact_dir / "config.json")
        result = {
            "dataset": name,
            "method": method_name,
            "backbone": args.backbone,
            "latency_ms_per_example": 1000.0 * elapsed / max(1, len(bundle.test_texts)),
            "peak_memory_mb": peak,
            "verbalizer_pair": "/".join(args.verbalizer_pair),
            "artifact_dir": str(artifact_dir),
            **metrics,
        }
        core.write_result_row(result, "main_results")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Plot the per-slot score distribution from an all-masked unary run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dllm_setscore import core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to a predictions.npz file produced by an all-masked unary run.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title-prefix", default="LLaDA all-masked unary")
    args = parser.parse_args()

    data = np.load(args.predictions)
    if "scores" not in data:
        raise KeyError(f"{args.predictions} does not contain a 'scores' array")
    scores = np.asarray(data["scores"])
    bundle = core.load_dataset_bundle(args.dataset)
    labels = bundle.label_names
    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError(
            f"score shape {scores.shape} does not match "
            f"{len(labels)} labels for {args.dataset}"
        )

    mean_log_odds = scores.mean(axis=0)
    positive_rate = (scores > 0).mean(axis=0)
    order = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(order, mean_log_odds, color="#2b6cb0")
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_xticks(order)
    axes[0].set_xticklabels(labels, rotation=70, fontsize=7)
    axes[0].set_ylabel("mean log-odds (yes vs no)")
    axes[0].set_title(f"{args.title_prefix}: per-slot mean log-odds")

    axes[1].bar(order, positive_rate * 100, color="#c05621")
    axes[1].axhline(50, color="black", linewidth=0.4, linestyle="--")
    axes[1].set_xticks(order)
    axes[1].set_xticklabels(labels, rotation=70, fontsize=7)
    axes[1].set_ylabel("% of inputs predicted positive")
    axes[1].set_title(f"{args.title_prefix}: per-slot positive rate")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")
    for idx in range(len(labels)):
        print(
            f"slot {idx} ({labels[idx]}): "
            f"mean_log_odds={mean_log_odds[idx]:.3f}, "
            f"positive_rate={positive_rate[idx] * 100:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

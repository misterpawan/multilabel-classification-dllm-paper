"""Build a markdown summary of every result row in a runs/<root> directory.

Walks ``results/main_results.jsonl`` and ``results/supervised_results.jsonl``,
groups by dataset, and emits a markdown table that can be pasted into the
README's "Current results" section.

Usage::

    PYTHONPATH=src python scripts/final_summary.py --root runs/main \
        --out runs/main/tables/summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd


def load(root: Path) -> pd.DataFrame:
    rows = []
    for name in ["main_results.jsonl", "supervised_results.jsonl"]:
        path = root / "results" / name
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["dataset", "method"], keep="last")
    return df


def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{float(x) * 100:.2f}"


def fmt_ms(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):.1f}"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    df = load(args.root)
    if df.empty:
        print(f"no results in {args.root}")
        return 1

    lines: List[str] = []
    for dataset in sorted(df["dataset"].dropna().unique()):
        sub = df[df["dataset"] == dataset].sort_values("micro_f1", ascending=False)
        lines.append(f"### {dataset}\n")
        lines.append("| Method | micro F1 | macro F1 | samples F1 | exact match | ECE | ms / example |")
        lines.append("|--------|---------:|---------:|-----------:|------------:|----:|-------------:|")
        for _, row in sub.iterrows():
            lines.append(
                f"| {row.get('method', '')} | {fmt_pct(row.get('micro_f1'))} "
                f"| {fmt_pct(row.get('macro_f1'))} "
                f"| {fmt_pct(row.get('samples_f1'))} "
                f"| {fmt_pct(row.get('exact_match'))} "
                f"| {row.get('ece', float('nan')):.3f} "
                if not pd.isna(row.get('ece', float('nan')))
                else
                f"| {row.get('method', '')} | {fmt_pct(row.get('micro_f1'))} "
                f"| {fmt_pct(row.get('macro_f1'))} "
                f"| {fmt_pct(row.get('samples_f1'))} "
                f"| {fmt_pct(row.get('exact_match'))} "
                f"| — "
            )
            lines[-1] += f"| {fmt_ms(row.get('latency_ms_per_example'))} |"
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

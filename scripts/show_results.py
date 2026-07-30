"""Pretty-print the latest results in a runs/<root>/ directory."""

import argparse
import json
import sys
from pathlib import Path
from typing import List

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


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("runs/main"))
    args = p.parse_args(argv)

    df = load(args.root)
    if df.empty:
        print(f"No results in {args.root}")
        return 1

    cols = [c for c in ["dataset", "method", "micro_f1", "macro_f1", "samples_f1", "exact_match", "ece", "latency_ms_per_example"] if c in df.columns]
    df = df.sort_values(["dataset", "method"])[cols]
    df = df.reset_index(drop=True)
    for col in ["micro_f1", "macro_f1", "samples_f1", "exact_match", "ece"]:
        if col in df.columns:
            df[col] = df[col].astype(float).round(4)
    if "latency_ms_per_example" in df.columns:
        df["latency_ms_per_example"] = df["latency_ms_per_example"].astype(float).round(1)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Aggregate the 3-seed P0a runs into mean/std for the headline cells.

Walks runs/main/results/main_results.jsonl and groups rows by
(dataset, base_method) where base_method is the method name with the
"_seedNN" suffix stripped. For each group, prints micro/macro F1 mean
and standard deviation over seeds.

Usage:
    PYTHONPATH=src python scripts/aggregate_seeds.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SEED_RE = re.compile(r"_seed(\d+)$")


def _strip_seed(method: str) -> tuple[str, int | None]:
    m = SEED_RE.search(method)
    seed: int | None
    if not m:
        seed = 13  # the default DLLM_SEED in core.py
        base = method
    else:
        base = method[: m.start()]
        seed = int(m.group(1))
    # Normalise the prompt-sweep aliases so the original headline run
    # (recorded under the "_promptsweep_<name>" suffix from the standalone
    # sweep script) lines up with the multi-seed runs (recorded under the
    # "_<name>" suffix from the multi-seed chain script).
    base = base.replace("_promptsweep_topic", "_topic")
    base = base.replace("_promptsweep_isabout", "_isabout")
    base = base.replace("_promptsweep_concerns", "_concerns")
    base = base.replace("_promptsweep_violation", "_violation")
    return base, seed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("runs/main"))
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Restrict to these datasets (default: all)")
    args = p.parse_args()

    jsonl = args.root / "results" / "main_results.jsonl"
    if not jsonl.exists():
        print(f"(no results file at {jsonl})")
        return 1

    # Group by (dataset, base_method); collect (seed, micro, macro).
    by_cell: dict[tuple[str, str], list[tuple[int | None, float, float]]] = defaultdict(list)
    with jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ds = r.get("dataset", "")
            method = r.get("method", "")
            mi = r.get("micro_f1")
            ma = r.get("macro_f1")
            if mi is None or ma is None:
                continue
            base, seed = _strip_seed(method)
            if args.datasets and ds not in args.datasets:
                continue
            by_cell[(ds, base)].append((seed, float(mi) * 100, float(ma) * 100))

    # Pretty-print one section per cell that has >= 2 entries.
    print()
    print(f"{'dataset':<22} {'method':<48} {'n':>3}  {'micro mean ± std':>20}  {'macro mean ± std':>20}")
    print("-" * 122)
    for key in sorted(by_cell.keys()):
        rows = by_cell[key]
        if len(rows) < 2:
            continue
        mis = [r[1] for r in rows]
        mas = [r[2] for r in rows]
        ds, method = key
        mi_mean, mi_std = mean(mis), stdev(mis) if len(mis) > 1 else 0.0
        ma_mean, ma_std = mean(mas), stdev(mas) if len(mas) > 1 else 0.0
        print(f"{ds:<22} {method:<48} {len(rows):>3}  {mi_mean:7.2f} \u00b1 {mi_std:5.2f}    {ma_mean:7.2f} \u00b1 {ma_std:5.2f}")
    print()
    print("(seeds present per cell — for cells with n=1, no stdev computed)")
    print()
    for key in sorted(by_cell.keys()):
        rows = by_cell[key]
        seeds = [str(r[0]) if r[0] is not None else "default" for r in rows]
        ds, method = key
        print(f"  {ds}/{method}: seeds={','.join(seeds)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

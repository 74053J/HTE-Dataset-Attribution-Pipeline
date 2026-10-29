#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_results.py - regenerate the result tables from the CSV files in data/.

    python scripts/build_results.py --grid      # results/grid13_v18.csv      (13 x 12 cells)
    python scripts/build_results.py --tables    # results/table1_v18_final.csv
    python scripts/build_results.py --all       # both, then run verify_numbers.py

Models, splits and feature construction come from the ``hte`` package, which is the
factored form of ``single_dataset_full_benchmark.py``. Figures are produced afterwards by
``scripts/figures.py`` and ``scripts/figures_paper2.py``.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# the thirteen datasets the manuscript reports -- ``corpus: true`` in
# data/alias_mapping.yaml (the synthetic benchmark, DoyleHTE_CN and the merged
# Roche66k_BH+SM file are marked corpus: false there)
from hte.datasets import corpus as _corpus  # noqa: E402

DATASETS = _corpus()


def dataset_pairs() -> list:
    """(alias, source stem) pairs read from data/alias_mapping.yaml."""
    text = (DATA / "alias_mapping.yaml").read_text(encoding="utf-8")
    mapping = {}
    for line in text.splitlines():
        m = re.match(r'\s*"?([A-Za-z0-9_+\-]+)"?:\s*\{source:\s*"([^"]+)"', line)
        if m:
            mapping[m.group(1)] = m.group(2)
    missing = [a for a in DATASETS if a not in mapping]
    if missing:
        raise SystemExit("alias_mapping.yaml has no entry for: %s" % missing)
    return [(a, mapping[a]) for a in DATASETS]


def build_grid() -> pd.DataFrame:
    from hte.grid import run_grid

    pairs = dataset_pairs()
    print("[grid] %d datasets: %s" % (len(pairs), ", ".join(a for a, _ in pairs)))
    out = RES / "grid13_v18.csv"
    grid = run_grid(datasets=pairs, data_dir=str(DATA), out_path=str(out), verbose=True)
    print("\n[grid] %d rows x %d datasets -> %s" % (len(grid), grid.dataset.nunique(), out.name))
    return grid


def build_tables(grid: pd.DataFrame) -> pd.DataFrame:
    """Not used: Table 1 has ONE producer, ``scripts/build_grid_ceiling.py``.

    This function used to pick each dataset's row by ``r2_test.idxmax()``, i.e.
    the maximum over the twelve configurations ON THE TEST SPLIT.  The paper
    selects the configuration on an inner validation split instead, and states
    that the test maximum exceeds the selected value by a mean of 0.0145 R2, so
    the two are not the same number.  Reproducing the inner-validation winner
    needs the refits that step 1 performs, so the table lives there.
    """
    raise SystemExit(
        "results/table1_v18_final.csv is produced by scripts/build_grid_ceiling.py\n"
        "(it needs the inner-validation refits that a plain grid pass does not do).\n"
        "Run:  python scripts/build_grid_ceiling.py")


def compare_with_shipped() -> None:
    """Compare the rebuilt tables with the copies shipped before this run (*.shipped.csv)."""
    for name in ("grid13_v18.csv", "table1_v18_final.csv"):
        rebuilt = RES / name
        shipped = RES / name.replace(".csv", ".shipped.csv")
        if not (rebuilt.exists() and shipped.exists()):
            continue
        a = pd.read_csv(shipped)
        b = pd.read_csv(rebuilt)
        print("\n[compare] %s: shipped %d rows, rebuilt %d rows, same columns: %s"
              % (name, len(a), len(b), list(a.columns) == list(b.columns)))
        if len(a) == len(b):
            for c in a.columns:
                if c in b.columns and pd.api.types.is_numeric_dtype(a[c]):
                    d = np.nanmax(np.abs(a[c].values - b[c].values))
                    print("    %-14s max|diff| = %.4g" % (c, d))


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild the paper's result tables from data/*.csv")
    ap.add_argument("--grid", action="store_true", help="rebuild results/grid13_v18.csv")
    ap.add_argument("--tables", action="store_true",
                    help="(moved) Table 1 is produced by scripts/build_grid_ceiling.py")
    ap.add_argument("--all", action="store_true", help="grid, then verify against the manuscript")
    args = ap.parse_args()
    if args.tables:
        build_tables(None)
    if not (args.grid or args.all):
        ap.print_help()
        return 0

    grid = None
    if args.grid or args.all:
        grid = build_grid()

    if args.all:
        compare_with_shipped()
        print("\n[verify] running scripts/verify_numbers.py")
        return subprocess.call([sys.executable, str(ROOT / "scripts" / "verify_numbers.py")])
    return 0


if __name__ == "__main__":
    sys.exit(main())

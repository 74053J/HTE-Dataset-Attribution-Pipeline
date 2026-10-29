#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 12 -- rebuild the aggregate ledger the gate reads.

``outputs/final_numbers/final_numbers_v2.json`` re-states the headline numbers
(capture mean/median, the correlation of ceiling with R², the R² range and the
correlation of dataset size with R²) so that downstream readers have one small file
instead of five tables.  It used to be maintained by hand, which meant it could drift
away from the tables it summarises; this script derives it from them instead.

Every value is recomputed from ``results/table1_v18_final.csv`` and
``results/grid13_v18.csv``; ``meta`` is preserved from the previous file when it exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

OUT = ROOT / "outputs" / "final_numbers" / "final_numbers_v2.json"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t1 = pd.read_csv(ROOT / "results" / "table1_v18_final.csv")
    grid = pd.read_csv(ROOT / "results" / "grid13_v18.csv")

    cap = pd.to_numeric(t1["capture"], errors="coerce").dropna()
    sub = t1[pd.to_numeric(t1["ceiling"], errors="coerce").notna()]
    rho_ceiling = float(stats.spearmanr(sub["ceiling"], sub["r2_best"]).statistic)
    rho_size = float(stats.spearmanr(sub["n_rows"], sub["r2_best"]).statistic)

    old = {}
    if OUT.is_file():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            old = {}

    doc = {"meta": old.get("meta", {}),
           "corpora": {"n13": {
               "ceiling_capture": {
                   "spearman_rho": {"value": rho_ceiling},
                   "median_capture": {"value": float(cap.median())},
                   "mean_capture": {"value": float(cap.mean())},
               },
               "best_r2": {"min": {"value": float(t1["r2_best"].min())},
                           "max": {"value": float(t1["r2_best"].max())}},
               "size_correlation": {"new": {"value": rho_size}},
               "n_datasets": int(len(t1)),
               "n_cells": int(len(grid)),
           }}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[final-numbers] capture mean %.6f median %.6f | rho(ceiling,R2) %+.6f | "
          "rho(n_rows,R2) %+.6f" % (cap.mean(), cap.median(), rho_ceiling, rho_size))
    print("[final-numbers] wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 9 -- every section-3.1 number, recomputed from results/grid13_v18.csv.

The paper's §3.1 quotes, per architecture pair, a mean paired gap, a two-sided
Wilcoxon signed-rank p, a Benjamini-Hochberg q and a Bonferroni p, plus two win
counts.  Those came from a scratch file that is not in the package; this script
reproduces them from the grid alone.

Conventions, calibrated against the published values on the pre-existing grid:

  * paired two-sided ``scipy.stats.wilcoxon`` (exact for n <= 25, zeros dropped);
  * BH family size **6** (the five comparisons plus the encoding effect, the
    legacy ``encoding_effect_bonferroni_6`` in the ledger), q = p * 6 / rank,
    ties share their **average** rank, no monotone back-fill;
  * Bonferroni p = min(1, 6 p).

Writes
  results/grid_stats.csv     one row per comparison, plus the win counts
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = HERE.parent
RES = ROOT / "results"
FAMILY = 6

PAIRS = [
    ("encoding_effect", "MLP", "one-hot", "MLP", "integer"),
    ("mlp_int_vs_xgb", "MLP", "integer", "XGBoost", "integer"),
    ("histgb_vs_xgb", "HistGB", "integer", "XGBoost", "integer"),
    ("gb_vs_xgb", "Gradient Boosting", "integer", "XGBoost", "integer"),
    ("rf_vs_xgb", "Random Forest", "integer", "XGBoost", "integer"),
]


def _bh(pvals):
    """q = p * m / average-rank among ties."""
    p = np.asarray(pvals, float)
    ranks = stats.rankdata(p, method="average")
    return np.clip(p * len(p) / ranks, 0.0, 1.0)


def main() -> int:
    grid = pd.read_csv(RES / "grid13_v18.csv")
    w = grid.pivot_table(index="dataset", columns=["model", "encoding"],
                         values="r2_test")
    rows = []
    for name, ma, ea, mb, eb in PAIRS:
        a, b = w[(ma, ea)], w[(mb, eb)]
        d = (a - b).dropna()
        rows.append(dict(comparison=name, n=len(d), gap=float(d.mean()),
                         n_positive=int((d > 0).sum()),
                         p_raw=float(stats.wilcoxon(d).pvalue)))
    q = _bh([r["p_raw"] for r in rows])
    for r, qv in zip(rows, q):
        r["q_bh_family6"] = float(qv)
        r["p_bonferroni6"] = float(min(1.0, r["p_raw"] * FAMILY))
    # HistGB native categorical against integer, the "0.041, 11 of 13" claim
    hn, hi = w[("HistGB", "native categorical")], w[("HistGB", "integer")]
    dh = (hn - hi).dropna()
    rows.append(dict(comparison="histgb_native_vs_integer", n=len(dh),
                     gap=float(dh.mean()), n_positive=int((dh > 0).sum()),
                     p_raw=float(stats.wilcoxon(dh).pvalue),
                     q_bh_family6=np.nan, p_bonferroni6=np.nan))

    out = pd.DataFrame(rows)
    out.to_csv(RES / "grid_stats.csv", index=False, encoding="utf-8")
    means = w.mean(axis=0)
    print("cell means: MLP int %.4f  MLP one-hot %.4f  XGB int %.4f"
          % (means[("MLP", "integer")], means[("MLP", "one-hot")],
             means[("XGBoost", "integer")]))
    print(out.to_string(index=False))
    print("\n[grid_stats] -> %s" % (RES / "grid_stats.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

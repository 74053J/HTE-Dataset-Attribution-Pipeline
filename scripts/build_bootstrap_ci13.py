#!/usr/bin/env python
"""Bootstrap intervals for the 13-dataset corpus -> ``results/bootstrap_ci13.csv``.

Why this file exists
--------------------
``results/bootstrap_ci.csv`` is the legacy **16-dataset** artefact: its per-dataset
rows cover the 12 datasets of ``results/legacy/ceiling_stats_v16.csv`` that have a
computable ceiling (``Roche66k_BH+SM`` among them) and its aggregate rows were
bootstrapped over those 12.  The manuscript uses the 13-dataset corpus of
``results/table1_v18_final.csv``, 11 of which have a ceiling, so the intervals are
regenerated here against exactly those numbers.

Method -- the reviewed implementation, not a copy
-------------------------------------------------
* per dataset: parametric chi-square sampling distribution of the pooled
  within-condition variance (``scripts/bootstrap_ci.py:per_dataset_tables``),
  which is the shipped method when the replicate rows are not redistributed;
* aggregates: dataset-level bootstrap resampling the 11 datasets with
  replacement, recomputing rho(ceiling, R2) and the mean/median capture ratio
  (``scripts/bootstrap_ci.py:dataset_level_bootstrap``);
* seed 42, 1000 resamples, 95 % intervals.

Output columns are ``dataset, statistic, estimate, ci_low, ci_high, method``;
aggregate rows use ``dataset == "TABLE1_11"``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
sys.path.insert(0, str(SCRIPTS))

import bootstrap_ci as bc  # noqa: E402  (the frozen, reviewed implementation)

OUT_PATH = RESULTS / "bootstrap_ci13.csv"


def load_corpus() -> pd.DataFrame:
    t = pd.read_csv(RESULTS / "table1_v18_final.csv")

    # Table 1 carries the validation-selected best R2 and the capture ratio computed
    # from it; grid13_v18.csv holds the twelve-configuration test-split maximum, which
    # is a different quantity by construction (mean gap 0.0117 R2), so it is not used
    # as a cross-check here.
    t["r2_best_grid"] = t["r2_best"].astype(float)
    t["n"] = t["n_rows"].astype(float)
    n_all = len(t)
    t = t[np.isfinite(t["ceiling"])].copy()
    t["capture_grid"] = t["r2_best_grid"] / t["ceiling"]
    t = t.sort_values("dataset").reset_index(drop=True)
    print("corpus: %d datasets, %d with a computable ceiling" % (n_all, len(t)))
    return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT_PATH), help="output csv")
    args = ap.parse_args(argv)

    t = load_corpus()

    rng = np.random.default_rng(bc.SEED)
    per = bc.per_dataset_tables(t, rng)
    agg = bc.dataset_level_bootstrap(t, np.random.default_rng(bc.SEED))

    rows: list[dict] = []
    for d in sorted(per, key=lambda x: x["dataset"]):
        rows.append(dict(dataset=d["dataset"], statistic="ceiling",
                         estimate=d["ceiling"], ci_low=d["ceil_lo"], ci_high=d["ceil_hi"],
                         method=d["ceil_method"]))
        rows.append(dict(dataset=d["dataset"], statistic="capture",
                         estimate=d["capture"], ci_low=d["cap_lo"], ci_high=d["cap_hi"],
                         method=d["cap_method"]))
    for stat, val, lo, hi, method in (
            ("spearman_rho", agg["rho"], agg["rho_lo"], agg["rho_hi"], bc.METHOD_RHO),
            ("median_capture", agg["median"], agg["med_lo"], agg["med_hi"], bc.METHOD_MEDIAN),
            ("mean_capture", agg["mean"], agg["mean_lo"], agg["mean_hi"], bc.METHOD_RHO),
            ("mean_capture_percentile", agg["mean"], agg["mean_lo"], agg["mean_hi"], bc.METHOD_RHO),
            ("median_capture_percentile", agg["median"], agg["med_pct_lo"], agg["med_pct_hi"],
             bc.METHOD_RHO)):
        rows.append(dict(dataset="TABLE1_11", statistic=stat, estimate=val,
                         ci_low=lo, ci_high=hi, method=method))

    frame = pd.DataFrame(rows, columns=bc.CSV_COLUMNS)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, float_format="%.6f")

    bh = frame[(frame.dataset == "bh-hte-ood_BH") & (frame.statistic == "ceiling")].iloc[0]
    print("wrote %s (%d rows, %d datasets)" % (args.out, len(frame), len(per)))
    print("  bh-hte-ood_BH ceiling = %.9f  [%.6f, %.6f]"
          % (bh["estimate"], bh["ci_low"], bh["ci_high"]))
    print("  TABLE1_11  rho       = %.6f [%.4f, %.4f]" % (agg["rho"], agg["rho_lo"], agg["rho_hi"]))
    print("  TABLE1_11  mean cap  = %.4f%% [%.4f, %.4f]"
          % (agg["mean"] * 100, agg["mean_lo"] * 100, agg["mean_hi"] * 100))
    print("  TABLE1_11  median cap= %.4f%% [%.4f, %.4f]"
          % (agg["median"] * 100, agg["med_pct_lo"] * 100, agg["med_pct_hi"] * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

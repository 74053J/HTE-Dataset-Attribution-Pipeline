#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 5 -- the enrichment audit (k = 10 / 50 / 1000) on the 13-dataset corpus.

Every number here comes from ``results/table1_v18_final.csv`` plus the packaged
loader; nothing is hard-coded per dataset.

Protocol (recovered 2026-10-28 from the file that produced
``results/enrichment13.csv``: a scratch script outside this repository,
written 2026-10-24 14:59):

  * each dataset is ranked by the cross-fitted prediction of its own Table-1
    best (model, encoding) cell;
  * cross-fitting is ``KFold(5, shuffle=True, random_state=42)`` over ALL rows,
    so every measured row is scored by a model that never saw it;
  * the whole dataset is sorted by that prediction (stable) and the top-k
    measured yields are averaged;
  * ``base_rate`` is the mean measured yield over ALL rows of the dataset (the
    ``mean_yield`` column of Table 1), and ``ef_k = mean_top_k / base_rate``.

A file shorter than k is taken whole, which is why the two files with fewer than
1,000 rows return exactly 1.00 at k = 1,000.

Writes
  results/enrichment13.csv      the 13-row deliverable
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from hte import params as P                          # noqa: E402
from hte.data import load_real_dataset               # noqa: E402
from hte.features import get_split_positions, prepare_features  # noqa: E402
from hte.grid import build_frames                    # noqa: E402
from hte.models import grid_cells                    # noqa: E402

DATA = ROOT / "data"
RES = ROOT / "results"

KS = (10, 50, 1000)
SEED = 42
N_FOLDS = 5


def _fit(cell, Xtr, ytr):
    mdl = cell.factory()
    is_rf = type(mdl).__name__ == "RandomForestRegressor"
    if is_rf:
        mdl.n_jobs = P.RF_N_JOBS
    mdl.fit(Xtr, ytr)
    if is_rf:
        mdl.n_jobs = None
    return mdl


def _topk(pred, k):
    return np.argsort(-np.asarray(pred, float), kind="stable")[:min(int(k), len(pred))]


def run_one(alias, stem, row):
    df, raw = load_real_dataset(str(DATA / (stem + ".csv")), str(DATA))
    n = len(df)
    tr, _te = get_split_positions(n)
    X, y, fcols, _disp, _enc, _sc = prepare_features(df, raw, fit_rows=tr)
    cat_idx = [j for j, c in enumerate(fcols) if str(c).endswith("_encoded")]
    model_name, enc_name = row["best_model"], row["best_encoding"]
    hits = [c for c in grid_cells(cat_idx)
            if c.architecture == model_name and c.encoding == enc_name]
    if not hits:
        raise SystemExit("cell %s / %s not in the grid for %s"
                         % (model_name, enc_name, alias))
    cell = hits[0]

    yv = np.asarray(y, float)
    base = float(yv.mean())
    pred = np.full(n, np.nan)
    for trf, tef in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.arange(n)):
        packs, _cc, _ci = build_frames(X, fcols, trf, tef)
        Xtr, Xte = packs[cell.frame]
        mdl = _fit(cell, Xtr, y.iloc[trf])
        pred[tef] = np.asarray(mdl.predict(Xte), float).ravel()
    if not np.isfinite(pred).all():
        raise SystemExit("cross-fitting left %d rows unscored (%s)"
                         % (int((~np.isfinite(pred)).sum()), alias))

    rec = dict(alias=alias, n=n, base_rate=base, best_r2=float(row["r2_best"]))
    for k in KS:
        m = float(yv[_topk(pred, k)].mean())
        rec["mean_yield_top%d" % k] = m
        rec["ef_%d" % k] = m / base
    return rec


def _stems() -> dict:
    """alias -> source stem, from the one table that holds it."""
    from protocol import stems
    return stems()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=None,
                    help="comma separated aliases (default: every row of Table 1)")
    ap.add_argument("--out", default=None, help="override the output CSV path")
    args = ap.parse_args()

    t1 = pd.read_csv(RES / "table1_v18_final.csv").set_index("dataset")
    stems = _stems()
    want = ([a.strip() for a in args.datasets.split(",")] if args.datasets
            else list(t1.index))

    rows = []
    t0 = time.perf_counter()
    for i, alias in enumerate(want, 1):
        rec = run_one(alias, stems[alias], t1.loc[alias])
        rows.append(rec)
        print("[enrich %2d/%d] %-16s n=%-6d base=%6.2f  top1000=%6.2f (EF %.3f)  "
              "top50=%6.2f (EF %.3f)  [%.0fs]"
              % (i, len(want), alias, rec["n"], rec["base_rate"],
                 rec["mean_yield_top1000"], rec["ef_1000"],
                 rec["mean_yield_top50"], rec["ef_50"],
                 time.perf_counter() - t0), flush=True)

    out = pd.DataFrame(rows).sort_values("best_r2").reset_index(drop=True)
    cols = (["alias", "n", "base_rate"]
            + ["mean_yield_top%d" % k for k in KS]
            + ["ef_%d" % k for k in KS] + ["best_r2"])
    out = out[cols]
    dst = Path(args.out) if args.out else RES / "enrichment13.csv"
    out.to_csv(dst, index=False, encoding="utf-8")
    print("\n[enrich] %d rows -> %s" % (len(out), dst))
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

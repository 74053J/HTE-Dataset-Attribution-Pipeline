#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 7 -- §3.6, part 1: unseen reactant PAIRS of known reactants.

Ported from the script that produced the numbers in §3.6
(``_work/v248/q4_multiseed.py``, 2026-10-26 19:50).  The question it answers:
holding out whole reactant pairs and keeping only pairs whose two reactants both
occur in training, how well does each model interpolate over conditions?

Protocol (unchanged from the original):

  * pair = (reactant_1, reactant_2); split by ``GroupShuffleSplit(test_size=0.25,
    random_state=seed)`` so a pair never straddles the two sides;
  * keep only held-out rows whose BOTH reactants appear in training -- that is
    what makes the task "unseen combination of known partners";
  * five seeds (42, 1, 2, 3, 4); reported value = mean ± 95 % CI over the seeds;
  * four arms: the training mean, reactant-1 + reactant-2 main effects,
    reactant-1's historical mean, XGBoost one-hot, XGBoost native categorical;
  * metrics per arm: row-level R2, the R2 after centering each held-out pair,
    MAE, and the pair-level Spearman correlation.

The dataset is the one flagged ``pair_extrapolation: true`` in
``data/alias_mapping.yaml``; nothing here names a dataset.  The estimator
settings are the paper's (§3.6 uses XGBoost, 800 trees, depth 6, lr 0.05).

Writes
  results/novel_pairs.csv
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from hte import params as P                     # noqa: E402
from hte.data import load_real_dataset          # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data"
RES = ROOT / "results"

SEEDS = (42, 1, 2, 3, 4)
TEST_SIZE = 0.25
ARMS = ("global", "additive", "r1_mean", "xgb_onehot", "xgb_native")
XGB = dict(n_estimators=800, max_depth=6, learning_rate=0.05,
           subsample=0.9, colsample_bytree=0.9, n_jobs=4)


def _dataset() -> str:
    from hte.datasets import _entries
    hits = [a for a, s in _entries(DATA / "alias_mapping.yaml")
            if bool(s.get("pair_extrapolation", False))]
    if len(hits) != 1:
        raise SystemExit("alias_mapping.yaml must flag exactly one "
                         "pair_extrapolation dataset, found %d: %s"
                         % (len(hits), hits))
    return hits[0]


def _features(df: pd.DataFrame) -> list:
    """The condition slots that vary: every categorical slot present + time/temperature."""
    cats = [c for c in P.CATEGORICAL_SLOTS
            if c in df.columns and df[c].astype(str).nunique() > 1]
    nums = [c for c in ("temperature", "time")
            if c in df.columns and df[c].nunique() > 1]
    return cats + nums


def run(seed, df, feat, Xall):
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    tr_i, te_i = next(gss.split(df, groups=df["pair"]))
    tr, te = df.iloc[tr_i], df.iloc[te_i]
    keep = (te["reactant_1"].isin(set(tr["reactant_1"]))
            & te["reactant_2"].isin(set(tr["reactant_2"])))
    te = te[keep]
    if len(te) < 200:
        return None
    y_tr = tr["yield"].astype(float).values
    y_te = te["yield"].astype(float).values
    out = {}

    def rec(name, pred):
        r2 = 1 - np.sum((y_te - pred) ** 2) / np.sum((y_te - y_te.mean()) ** 2)
        d = pd.DataFrame({"pair": te["pair"].values, "y": y_te, "p": pred})
        cy = d.y - d.groupby("pair").y.transform("mean")
        cp = d.p - d.groupby("pair").p.transform("mean")
        r2c = 1 - np.sum((cy - cp) ** 2) / np.sum(cy ** 2)
        pr = d.groupby("pair")[["y", "p"]].mean()
        out[name] = (float(r2), float(r2c),
                     float(np.mean(np.abs(y_te - pred))),
                     float(stats.spearmanr(pr.y, pr.p).statistic)
                     if len(pr) >= 3 else float("nan"))

    import xgboost as xgb

    m = y_tr.mean()
    rec("global", np.full(len(te), m))
    a1 = tr.groupby("reactant_1")["yield"].mean() - m
    b2 = tr.groupby("reactant_2")["yield"].mean() - m
    rec("additive", m + te.reactant_1.map(a1).fillna(0).values
        + te.reactant_2.map(b2).fillna(0).values)
    rec("r1_mean", te.reactant_1.map(tr.groupby("reactant_1")["yield"].mean())
        .fillna(m).values)

    Xtr = Xall.iloc[tr_i]
    Xte = Xall.loc[te.index]
    Xc, Xt = Xtr.copy(), Xte.copy()
    for c in feat:
        Xc[c] = Xc[c].astype("category")
        Xt[c] = pd.Categorical(Xt[c], categories=Xc[c].cat.categories)
    mdl = xgb.XGBRegressor(random_state=seed, enable_categorical=True,
                           tree_method="hist", **XGB)
    mdl.fit(Xc, y_tr)
    rec("xgb_native", mdl.predict(Xt))

    Xo = pd.get_dummies(Xtr)
    Xo2 = pd.get_dummies(Xte).reindex(columns=Xo.columns, fill_value=0)
    mdl2 = xgb.XGBRegressor(random_state=seed, **XGB)
    mdl2.fit(np.asarray(Xo, dtype=np.float32), y_tr)
    rec("xgb_onehot", mdl2.predict(np.asarray(Xo2, dtype=np.float32)))
    return out, len(te), int(te["pair"].nunique())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="override the flagged dataset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from hte.datasets import stems
    alias = args.dataset or _dataset()
    stem = stems()[alias]
    t0 = time.perf_counter()
    df, _raw = load_real_dataset(str(DATA / (stem + ".csv")), str(DATA))
    df = df.dropna(subset=["reactant_1", "reactant_2", "yield"]).copy()
    for c in ("reactant_1", "reactant_2"):
        df[c] = df[c].astype(str)
    df["pair"] = df["reactant_1"] + " || " + df["reactant_2"]
    feat = _features(df)
    Xall = df[feat].astype(str)
    print("%s: %d rows, %d distinct pairs, %d feature columns (%s)"
          % (alias, len(df), df["pair"].nunique(), len(feat), ", ".join(feat)))

    rows = []
    for seed in SEEDS:
        r = run(seed, df, feat, Xall)
        if r is None:
            print("  seed %-3d held-out set smaller than 200 rows -- skipped" % seed)
            continue
        out, n_rows, n_pairs = r
        rows.append((seed, n_rows, n_pairs, out))
        print("  seed %-3d held out %4d rows / %4d pairs | xgb_native R2 %.3f | "
              "additive %.3f" % (seed, n_rows, n_pairs, out["xgb_native"][0],
                                 out["additive"][0]))
    if not rows:
        raise SystemExit("no seed produced a usable held-out set")

    table = []
    for nm in ARMS:
        vals = np.array([[r[3][nm][i] for i in range(4)] for r in rows], dtype=float)
        mean, sd = vals.mean(axis=0), vals.std(axis=0, ddof=1)
        ci = 1.96 * sd / np.sqrt(len(rows))
        table.append(dict(dataset=alias, arm=nm, n_splits=len(rows),
                          r2_mean=mean[0], r2_ci95=ci[0],
                          r2_centered_mean=mean[1], r2_centered_ci95=ci[1],
                          mae=mean[2], pair_spearman=mean[3]))
    out = pd.DataFrame(table)
    dst = Path(args.out) if args.out else RES / "novel_pairs.csv"
    out.to_csv(dst, index=False, encoding="utf-8")
    print("\n%-12s %-22s %-22s %-8s %s"
          % ("arm", "row R2 (mean ± 95%CI)", "centered R2", "MAE", "pair rho"))
    for r in table:
        print("%-12s %+8.3f ± %.3f        %+8.3f ± %.3f     %6.2f    %+.3f"
              % (r["arm"], r["r2_mean"], r["r2_ci95"], r["r2_centered_mean"],
                 r["r2_centered_ci95"], r["mae"], r["pair_spearman"]))
    print("\n[novel_pairs] %d arms x %d seeds -> %s  (%.0fs)"
          % (len(table), len(rows), dst, time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

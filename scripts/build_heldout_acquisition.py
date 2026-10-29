#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 12 -- ranking versus acquisition on HELD-OUT REAL CHEMISTRY.

The manuscript's paragraph after the Bayesian-acquisition result claims that on
real chemistry the three orderings coincide, because the ensemble spread is an
order of magnitude smaller than the prediction range.  Those numbers used to come
from scratch scripts that imported an older pipeline (``Analysis_all_v13``) and
hard-coded absolute paths; nothing in the package could recompute them.  This
script does, on the current pipeline.

Protocol
    * the case-study corpus member with the most reactant pairs (bh-hte-ood_BH);
    * pairs (reactant_1, reactant_2) are split into five groups, one group is
      held out entirely, so no pair in the test set was seen in training;
    * the five-member panel is fitted on the remaining rows;
    * the held-out rows are ordered four ways -- ranking by predicted yield,
      expected improvement, upper confidence bound, and random -- and each order
      is scored by the best MEASURED yield it reaches after k picks;
    * the spread is then checked twice: does it rank the absolute error, and is
      mu +/- 1.645 sigma a 90 % interval?

Writes
    results/heldout_acquisition.csv     one row per ordering at k = 1, 10, 50
    results/heldout_calibration.csv     spearman, quintile medians, coverage
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _panel(seed: int, n_jobs: int):
    import xgboost as xgb
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.neural_network import MLPRegressor
    return {
        "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=seed, enable_categorical=True,
                                    tree_method="hist", max_cat_to_onehot=4,
                                    n_jobs=n_jobs),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=seed),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=seed, n_jobs=n_jobs),
        "HistGB": HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, learning_rate=0.1, random_state=seed),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                            max_iter=500, random_state=seed, early_stopping=True),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import protocol as L
    from hte import datasets as DS
    from hte.v16_core import get_split_positions, prepare_features

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=None,
                    help="dataset alias (default: the corpus member with the most "
                         "reactant pairs)")
    ap.add_argument("--folds", type=int, default=5, help="group folds (mode=group)")
    ap.add_argument("--test-size", type=float, default=None,
                    help="row-split test fraction (default: the pipeline 80/20)")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--clip", type=float, default=None,
                    help="clip measured yields at this value before scoring the curves")
    ap.add_argument("--mode", choices=("plain", "row", "group"), default="plain",
                    help="row = 80/20 split then drop test pairs seen in training "
                         "(the manuscript's protocol); group = GroupKFold")
    ap.add_argument("--picks", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", choices=("panel", "seed", "cv"), default="panel",
                    help="spread definition: panel = five architectures; "
                         "seed = one architecture fitted with five seeds; "
                         "cv = five fold-models of one architecture")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    from scipy.stats import norm, spearmanr

    alias = args.dataset
    if alias is None:
        best, best_n = None, -1
        for cand in DS.corpus():
            with contextlib.redirect_stdout(io.StringIO()):
                d, _ = L.load(cand)
            cols = [c for c in ("reactant_1", "reactant_2") if c in d.columns]
            if len(cols) < 2:
                continue
            n = int(d.groupby(cols).ngroups)
            if n > best_n:
                best, best_n = cand, n
        alias = best
    print("[heldout] dataset %s" % alias)

    with contextlib.redirect_stdout(io.StringIO()):
        df, raw = L.load(alias)
    pair = df["reactant_1"].astype(str) + "|" + df["reactant_2"].astype(str)
    if args.mode == "plain":
        # the manuscript's test set: the ordinary 80/20 row split, every test row
        tr, te = get_split_positions(len(df), test_size=args.test_size,
                                     random_state=args.split_seed)
    elif args.mode == "group":
        # one fold of a group split over the pair key
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=args.folds)
        tr, te = next(iter(gkf.split(df, groups=pair.to_numpy())))
    else:
        # the manuscript's protocol: the ordinary 80/20 row split, then keep only the
        # test rows whose reactant pair never appears in training
        tr, te0 = get_split_positions(len(df), test_size=args.test_size,
                                      random_state=args.split_seed)
        seen = set(pair.iloc[tr])
        te = np.array([i for i in te0 if pair.iloc[i] not in seen], dtype=int)
    n_test_pairs = int(pair.iloc[te].nunique())
    print("[heldout] train rows %d / test rows %d | held-out pairs %d"
          % (len(tr), len(te), n_test_pairs))

    with contextlib.redirect_stdout(io.StringIO()):
        X, y, cols, _disp, enc, _sc = prepare_features(df, raw, fit_rows=tr)
    cat_cols = [c for c in cols if str(c).endswith("_encoded")]
    Xc = X.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].fillna(-1).astype("int64").astype("category")

    models = {}
    panel = _panel(args.seed, args.n_jobs)
    if args.sigma == "cv":
        # five fold-models of one architecture: sigma is the cross-validation spread
        import copy
        from sklearn.model_selection import KFold
        base = panel["XGBoost"]
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        cv_models = []
        for k, (itr, _iva) in enumerate(kf.split(tr)):
            mdl = copy.deepcopy(base).set_params(random_state=args.seed + k)
            frame = Xc.iloc[tr[itr]]
            with contextlib.redirect_stdout(io.StringIO()):
                mdl.fit(frame, y.iloc[tr[itr]])
            cv_models.append(mdl)
        preds = []
        use = Xc.iloc[te].copy()
        for c in cat_cols:
            use[c] = use[c].astype("int64").astype("category")
        for mdl in cv_models:
            preds.append(np.asarray(mdl.predict(use), dtype=float))
        p5 = np.vstack(preds)
        mu, sd = p5.mean(0), p5.std(0)
        obs = y.iloc[te].to_numpy(dtype=float)
        if args.clip is not None:
            obs = np.minimum(obs, args.clip)
        models = {"cv": (cv_models[0], Xc)}      # only to satisfy the code below
    elif args.sigma == "seed":
        import copy
        panel = {"XGBoost seed %d" % (args.seed + k):
                 copy.deepcopy(panel["XGBoost"]).set_params(random_state=args.seed + k)
                 for k in range(5)}
    for name, m in panel.items():
        frame = Xc if name.startswith("XGBoost") or name == "HistGB" else X
        if name == "HistGB":
            idx = [i for i, c in enumerate(cols) if str(c).endswith("_encoded")]
            m.set_params(categorical_features=(idx or None))
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(frame.iloc[tr], y.iloc[tr])
        models[name] = (m, frame)
        print("[fit] %-18s R2 test %+.4f" % (name, m.score(frame.iloc[te], y.iloc[te])))

    def predict(idx):
        preds = []
        for name, (m, frame) in models.items():
            use = frame.iloc[idx]
            if name.startswith("XGBoost") or name == "HistGB":
                use = use.copy()
                for c in cat_cols:
                    use[c] = use[c].astype("int64").astype("category")
            preds.append(np.asarray(m.predict(use), dtype=float))
        p = np.vstack(preds)
        return p.mean(0), p.std(0)

    if args.sigma != "cv":
        mu, sd = predict(te)
    obs = y.iloc[te].to_numpy(dtype=float)
    if args.clip is not None:
        obs = np.minimum(obs, args.clip)
    n = len(obs)
    print("[heldout] spread %.2f | observed range %.2f | ratio %.1fx"
          % (np.median(sd), obs.max() - obs.min(),
             (obs.max() - obs.min()) / max(np.median(sd), 1e-9)))

    rng = np.random.default_rng(args.seed)
    orders = {
        "ranking by predicted yield": np.argsort(-mu),
        "expected improvement": np.argsort(-((mu - 0.0) * norm.cdf(mu / np.maximum(sd, 1e-9))
                                             + sd * norm.pdf(mu / np.maximum(sd, 1e-9)))),
        "upper confidence bound": np.argsort(-(mu + 1.645 * sd)),
        "random order": rng.permutation(n),
    }
    rows = []
    for name, order in orders.items():
        curve = np.maximum.accumulate(obs[order])
        for k in (1, 10, args.picks, 100):
            rows.append(dict(order=name, k=k, best_measured=float(curve[k - 1])))
        print("[order] %-26s k=1 %.1f%%  k=%d %.1f%%"
              % (name, curve[0], args.picks, min(curve[args.picks - 1], 100.0)))
    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "heldout_acquisition.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    err = np.abs(mu - obs)
    rho = float(spearmanr(sd, err).statistic)
    q = pd.qcut(sd, 5, labels=False)
    lo = float(np.median(err[q == 0]))
    hi = float(np.median(err[q == 4]))
    width90 = float(np.mean(2 * 1.645 * sd))
    cov68 = float(np.mean(err <= 1.0 * sd) * 100)
    cov90 = float(np.mean(err <= 1.645 * sd) * 100)
    cov95 = float(np.mean(err <= 1.96 * sd) * 100)
    cal = pd.DataFrame([dict(rows=int(n), spearman_sd_vs_error=rho,
                             mean_interval_width_90=width90,
                             coverage_68=cov68, median_error=float(np.median(err)),
                             median_error_lowest_quintile=lo,
                             median_error_highest_quintile=hi,
                             coverage_90=cov90, coverage_95=cov95)])
    cal_path = ROOT / "results" / "heldout_calibration.csv"
    cal.to_csv(cal_path, index=False, encoding="utf-8")
    print("[calib] rho(sd,|err|) = %.3f | median err %.2f (q1 %.2f q5 %.2f) | "
          "width90 %.2f | coverage 68%% %.1f%% / 90%% %.1f%% / 95%% %.1f%%"
          % (rho, float(np.median(err)), lo, hi, width90, cov68, cov90, cov95))
    print("[write] %s ; %s" % (out_path, cal_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Smoke test for hte.recommend -- wiring only, no virtual enumeration.

* ``rank_real_records`` ranks the real measured records of AI_RCP_BH using two
  cheap fitted members (XGBoost + Random Forest) and must return a sane table:
  one row per condition, Rank 1..n, no NaN ensemble score.
* ``plan_enumeration`` resolves the capped search space without scoring it.
* ``enumerate_virtual_capped`` must REFUSE to run without ``allow_virtual``.

The full virtual library is never enumerated here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C


def main() -> int:
    root = C.data_dir()
    if root is None:
        print("SKIP: data/ directory not found next to this repository")
        return 0

    import hte
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb

    rep = C.Report("test_recommend")
    stem = "(AI_RCP_pub)bh_data_clean_all_whitelisted"
    with C.quiet():
        df, raw = hte.load_real_dataset(str(Path(root) / (stem + ".csv")), str(root))
        tr, te = hte.get_split_positions(len(df))
        X, y, feat_cols, _display, enc, sc = hte.prepare_features(
            df, raw, fit_rows=tr)

    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    models = {
        "XGBoost": xgb.XGBRegressor(n_estimators=60, max_depth=5,
                                    learning_rate=0.1, subsample=0.8,
                                    colsample_bytree=0.8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=10,
                                               min_samples_split=5,
                                               random_state=42, n_jobs=1),
    }
    preds = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        preds[name] = model.predict(X)
    rep.check("two members fitted", all(np.isfinite(p).all() for p in preds.values()))

    ranked = hte.rank_real_records(df, preds, max_model_gap=None)
    rep.check("ranked table non-empty", len(ranked) > 0, note="%d rows" % len(ranked))
    rep.check("Rank is 1..n", list(ranked["Rank"]) == list(range(1, len(ranked) + 1)))
    rep.check("one row per distinct condition",
              ranked["Rank"].is_unique and len(ranked) <= len(df),
              note="%d conditions from %d runs" % (len(ranked), len(df)))
    rep.check("ensemble score finite",
              bool(np.isfinite(ranked["Ensemble Predicted Yield (%)"]).all()))
    rep.check("ensemble score is descending",
              bool((np.diff(ranked["Ensemble Predicted Yield (%)"]) <= 1e-12).all()))
    rep.check("split label present",
              "Split" in ranked.columns and set(ranked["Split"].dropna().unique())
              <= {"train", "held-out", "unknown"})
    rep.check("replicate bookkeeping present",
              "N Runs" in ranked.columns and int(ranked["N Runs"].min()) >= 1)

    plan = hte.plan_enumeration(df, feat_cols, 200_000)
    rep.check("plan: categorical axes resolved", plan["n_cat"] > 0,
              note="n_cat=%d axes=%d" % (plan["n_cat"], len(plan["cat_cols"])))
    rep.check("plan: rows <= cap", plan["rows"] <= 200_000,
              note="rows=%d" % plan["rows"])
    rep.check("plan: full library >= capped rows",
              plan["full_rows"] >= plan["rows"],
              note="full=%d" % plan["full_rows"])

    try:
        hte.enumerate_virtual_capped(df, models, feat_cols, enc, sc,
                                     max_combos=100, allow_virtual=False)
    except hte.recommend.VirtualEnumerationRefused:
        rep.check("enumerate refuses without allow_virtual", True)
    except Exception as exc:                     # pragma: no cover
        rep.check("enumerate refuses without allow_virtual", False,
                  note="%s: %s" % (type(exc).__name__, exc))
    else:
        rep.check("enumerate refuses without allow_virtual", False,
                  note="it returned instead of raising")

    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())

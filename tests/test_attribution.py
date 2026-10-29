#!/usr/bin/env python
"""Equivalence check (extra): hte.attribution vs v16 on the same fitted models.

Not one of the three gate items, but cheap (a few seconds) and it covers the
module used to build ``results/attrib12_long.csv``:

* XGBoost TreeSHAP (v16's ``pred_contribs`` path) must agree bit-exactly;
* sklearn RandomForest TreeSHAP must agree bit-exactly;
* ``analyze_feature_importance`` must return the same table (order included).

The MLP ``PermutationExplainer`` path is deliberately not gated: permutation
sampling is stochastic, so bit-exact agreement across two calls is not a
meaningful requirement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

    stem = "(AI_RCP_pub)bh_data_clean_all_whitelisted"
    with C.quiet():
        df, raw = hte.load_real_dataset(str(Path(root) / (stem + ".csv")), str(root))
        tr, te = hte.get_split_positions(len(df))
        X, y, feat_cols, display, _enc, _sc = hte.prepare_features(
            df, raw, fit_rows=tr)

    rep = C.Report("test_attribution")
    v16 = C.load_v16()
    Xtr = X.iloc[tr].copy()
    ytr = y.iloc[tr].copy()
    Xs = X.iloc[tr[:300]].copy()

    print("\n[xgboost] fit on %d rows" % len(Xtr))
    xm = xgb.XGBRegressor(n_estimators=40, max_depth=4, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
    xm.fit(Xtr, ytr)
    sv_v, base_v = v16.get_shap_values(xm, Xs, feature_names=display)
    sv_h, base_h = hte.get_shap_values(xm, Xs, feature_names=display)
    rep.check("xgboost: shap matrix shape", sv_v.shape == sv_h.shape,
              note=str(sv_h.shape))
    rep.diff("xgboost: shap values", sv_v, sv_h)
    rep.diff("xgboost: base value", [base_v], [base_h])

    print("\n[random forest] fit on %d rows" % len(Xtr))
    rf = RandomForestRegressor(n_estimators=30, max_depth=8,
                               min_samples_split=5, random_state=42, n_jobs=1)
    rf.fit(Xtr, ytr)
    sv_v, base_v = v16.get_shap_values(rf, Xs, feature_names=display)
    sv_h, base_h = hte.get_shap_values(rf, Xs, feature_names=display)
    rep.check("random forest: shap matrix shape", sv_v.shape == sv_h.shape,
              note=str(sv_h.shape))
    rep.diff("random forest: shap values", sv_v, sv_h)
    rep.diff("random forest: base value", [base_v], [base_h])

    iv = v16.analyze_feature_importance(rf, display)
    ih = hte.analyze_feature_importance(rf, display)
    rep.equal("random forest: analyze_feature_importance columns",
              list(iv.columns), list(ih.columns))
    rep.equal("random forest: analyze_feature_importance feature order",
              list(iv["Feature"]), list(ih["Feature"]))
    rep.diff("random forest: analyze_feature_importance values",
             iv["Importance"].to_numpy(float), ih["Importance"].to_numpy(float))

    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())

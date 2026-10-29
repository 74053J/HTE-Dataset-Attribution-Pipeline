"""Feature attribution (SHAP) and native feature importance.

``get_shap_values`` and ``analyze_feature_importance`` are copied verbatim from
``archive/Analysis_all_v16.py`` (v16 lines 1702 and 1773).  They read no
module globals; the only change is that ``shap`` is a hard import here (it is in
``requirements.txt``) rather than an optional one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def get_shap_values(model, X, feature_names=None):
    """
    Exact TreeSHAP values for tree models, model-agnostic values for everything else.

    Routing:
      * xgboost  -> booster.predict(DMatrix, pred_contribs=True): exact TreeSHAP,
                    no shap dependency at all (works around shap 0.49 failing on
                    xgboost >= 3.0's bracketed vector base_score).
      * sklearn tree -> shap.TreeExplainer.
      * anything else (MLPRegressor) -> shap.PermutationExplainer.

    Returns (shap_values[n_rows, n_features], base_value).
    """
    X = pd.DataFrame(X)
    if feature_names is not None:
        X.columns = list(feature_names)

    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        names = booster.feature_names if booster.feature_names else list(X.columns)
        dm = xgb.DMatrix(X, feature_names=list(names))
        contribs = np.asarray(booster.predict(dm, pred_contribs=True), dtype=np.float64)
        return contribs[:, :-1], float(np.asarray(contribs[:, -1]).ravel()[0])

    # sklearn tree ensembles
    if hasattr(model, 'estimators_') or hasattr(model, 'tree_'):
        e = shap.TreeExplainer(model)
        sv = np.asarray(e.shap_values(X), dtype=np.float64)
        if sv.ndim == 3:
            sv = sv[:, :, 0]
        base = getattr(e, 'expected_value', 0.0)
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).ravel()[0])
        return sv, float(base)

    # model-agnostic fallback (neural networks and anything else)
    #
    # Feed the explainer a plain float array, not a DataFrame: X.columns was just
    # overwritten with DISPLAY names, and sklearn validates a DataFrame's column
    # names against those seen during fit for MLPRegressor.predict.
    _Xv = np.asarray(X, dtype=float)
    _raw_predict = getattr(model, 'predict', model)

    def _flat(A):
        return np.asarray(_raw_predict(A), dtype=float).ravel()

    ex = shap.PermutationExplainer(_flat, _Xv)
    exp = ex(_Xv, max_evals=int(2 * _Xv.shape[1] + 1), silent=True)
    sv = np.asarray(exp.values, dtype=np.float64)
    if sv.ndim == 3:
        sv = sv[:, :, 0]
    base = np.asarray(exp.base_values).ravel()
    return sv, float(base[0]) if base.size else 0.0


def analyze_feature_importance(model, display_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        importances = np.full(len(display_names), np.nan)
        # v15: was np.random.uniform(...) -- FABRICATED bars for models with no
        # native importance.  NaN keeps any future caller honest.

    return pd.DataFrame({'Feature': display_names, 'Importance': importances}).sort_values('Importance', ascending=False)

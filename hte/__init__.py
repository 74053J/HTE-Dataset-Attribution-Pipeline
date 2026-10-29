"""``hte`` -- standalone package extracted from the frozen v16 pipeline.

The package reproduces the numerics of ``archive/Analysis_all_v16.py``
for the parts the paper needs (data loading, feature preparation, noise
ceiling, the 12-cell benchmark grid, SHAP attribution and condition
recommendation) without importing the frozen module.

Modules
-------
:mod:`hte.data`          load_real_dataset, create_default_col_mappings
                         (+ loading-axis resolution); per-file routing comes
                         from data/col_mappings.csv, not from code
:mod:`hte.features`      prepare_features, get_split_positions
:mod:`hte.ceiling`       estimate_noise_ceiling
:mod:`hte.models`        estimator constructors (v16 members and the 12 cells)
:mod:`hte.grid`          the 12-configuration grid runner
:mod:`hte.attribution`   get_shap_values, analyze_feature_importance
:mod:`hte.recommend`     real-record ranking + capped virtual enumeration

Parameters are read from ``params.json`` (see :mod:`hte.params`).  The
BLAS/OpenMP thread caps are set here, before numpy/xgboost are imported, exactly
as v16 does at lines 30-34; without them the XGBoost grid cells drift from the
shipped reference values.
"""
from __future__ import annotations

import importlib
import os

# Thread cap -- must happen before numpy / xgboost / sklearn are imported.
# Mirrors v16 lines 30-34.  Set DSH_SWEEP_THREADS=0 to disable.
_SWEEP_THREADS = os.environ.get("DSH_SWEEP_THREADS", "4")
if _SWEEP_THREADS not in ("0", "", "none", "None"):
    for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_tv, _SWEEP_THREADS)

__version__ = "0.1.0"

_MODULES = ("params", "data", "features", "ceiling", "models", "grid",
            "attribution", "recommend")

# (module, attribute) for the public one-line API.
_API = {
    "load_real_dataset": ("data", "load_real_dataset"),
    "create_default_col_mappings": ("data", "create_default_col_mappings"),
    "resolve_loading_axis": ("data", "resolve_loading_axis"),
    "apply_loading_axis_alias": ("data", "apply_loading_axis_alias"),
    "loading_axis_available": ("data", "loading_axis_available"),
    "prepare_features": ("features", "prepare_features"),
    "get_split_positions": ("features", "get_split_positions"),
    "estimate_noise_ceiling": ("ceiling", "estimate_noise_ceiling"),
    "get_shap_values": ("attribution", "get_shap_values"),
    "analyze_feature_importance": ("attribution", "analyze_feature_importance"),
    "rank_real_records": ("recommend", "rank_real_records"),
    "plan_enumeration": ("recommend", "plan_enumeration"),
    "enumerate_virtual_capped": ("recommend", "enumerate_virtual_capped"),
    "PARAMS": ("params", "PARAMS"),
}

__all__ = list(_API) + list(_MODULES) + ["__version__"]


def __getattr__(name: str):
    if name in _MODULES:
        return importlib.import_module("." + name, __name__)
    if name in _API:
        module, attr = _API[name]
        mod = importlib.import_module("." + module, __name__)
        return getattr(mod, attr)
    raise AttributeError("module 'hte' has no attribute %r" % name)


def __dir__():
    return sorted(set(globals()) | set(__all__))

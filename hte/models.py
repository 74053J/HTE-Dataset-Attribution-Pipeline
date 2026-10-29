"""Estimator constructors.

The hyper-parameters are copied verbatim from
``archive/Analysis_all_v16.py``:

* ``v16_members`` reproduces the six estimators inlined in
  ``train_and_evaluate_models`` (v16 lines 1503-1545);
* ``grid_cells`` reproduces the twelve architecture/encoding cells that built
  ``results/grid_all16.csv`` (the cell table that lived in
  ``scripts/tables.py:build_grid_all16``; that script used the same literals).

Both panels are provided because the paper uses both: the six-member ensemble is
the virtual-library / recommendation member panel, while the 12-cell grid is the
Table I benchmark (5 architectures x 3 encodings, minus the 3 cells that are not
defined: Gradient Boosting, Random Forest and MLP have no native-categorical
cell).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from . import params as P


# individual estimators (literals from v16 / tables.py)
def xgboost_regressor(**extra) -> xgb.XGBRegressor:
    """v16 line 1504 / tables.py ``_xgb``."""
    return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, **extra)


def gradient_boosting() -> GradientBoostingRegressor:
    """v16 line 1505 / tables.py ``_gb``."""
    return GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                     learning_rate=0.1, random_state=42)


def multilayer_perceptron() -> MLPRegressor:
    """v16 line 1506 / tables.py ``_mlp``."""
    return MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                        max_iter=500, random_state=42, early_stopping=True)


def random_forest(n_jobs=None) -> RandomForestRegressor:
    """v16 line 1507 / tables.py ``_rf``."""
    return RandomForestRegressor(n_estimators=int(P.RF_N_ESTIMATORS),
                                 max_depth=15, min_samples_split=5,
                                 random_state=42, n_jobs=n_jobs)


def histgb_native(cat_idx) -> HistGradientBoostingRegressor:
    """v16 line 1525 / tables.py ``_histgb_native`` (categorical_features)."""
    return HistGradientBoostingRegressor(
        max_iter=int(P.HISTGB_MAX_ITER), max_depth=6, learning_rate=0.1,
        random_state=42, categorical_features=(cat_idx if cat_idx else None))


def histgb_plain() -> HistGradientBoostingRegressor:
    """tables.py ``_histgb_plain`` (no native categorical declaration)."""
    return HistGradientBoostingRegressor(
        max_iter=int(P.HISTGB_MAX_ITER), max_depth=6, learning_rate=0.1,
        random_state=42)


def mlp_onehot(cat_idx):
    """v16 line 1528 / tables.py ``_mlp_onehot`` (one-hot inside the pipeline)."""
    return make_pipeline(
        ColumnTransformer(
            [('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_idx)],
            remainder='passthrough'),
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                     max_iter=500, random_state=42, early_stopping=True))


def categorical_indices(feature_cols) -> list[int]:
    """Positions of the ``*_encoded`` categorical columns (v16 line 1499)."""
    return [i for i, c in enumerate(feature_cols) if str(c).endswith('_encoded')]


# the six v16 ensemble members
def v16_members(cat_idx=None) -> dict:
    """The six estimators of ``train_and_evaluate_models``, in v16 order."""
    return {
        'XGBoost': xgboost_regressor(),
        'Gradient Boosting': gradient_boosting(),
        'Multilayer Perceptron': multilayer_perceptron(),
        'Random Forest': random_forest(),
        'HistGB-NativeCat': histgb_native(cat_idx),
        'MLP-OneHot': mlp_onehot(cat_idx),
    }


# the 12 architecture/encoding cells of results/grid_all16.csv
@dataclass(frozen=True)
class GridCell:
    """One row of the paper's 5x3 benchmark grid.

    ``frame`` says which feature matrix the estimator must see:

    * ``INT`` -- the integer-coded matrix returned by ``prepare_features``;
    * ``OH``  -- a one-hot matrix built OUTSIDE the estimator;
    * ``NAT`` -- the integer matrix with the categorical columns cast to
                 pandas ``category`` dtype (xgboost native categorical).
    """

    architecture: str
    encoding: str
    factory: Callable[[], object]
    frame: str


def grid_cells(cat_idx) -> list[GridCell]:
    """The twelve cells of ``results/grid_all16.csv`` (tables.py ``cells_for``)."""
    return [
        GridCell("XGBoost", "integer", xgboost_regressor, "INT"),
        GridCell("XGBoost", "one-hot", xgboost_regressor, "OH"),
        GridCell("XGBoost", "native categorical",
                 lambda: xgboost_regressor(enable_categorical=True,
                                           tree_method="hist",
                                           max_cat_to_onehot=4), "NAT"),
        GridCell("Gradient Boosting", "integer", gradient_boosting, "INT"),
        GridCell("Gradient Boosting", "one-hot", gradient_boosting, "OH"),
        GridCell("Random Forest", "integer", random_forest, "INT"),
        GridCell("Random Forest", "one-hot", random_forest, "OH"),
        GridCell("HistGB", "integer", histgb_plain, "INT"),
        GridCell("HistGB", "one-hot", histgb_plain, "OH"),
        GridCell("HistGB", "native categorical",
                 lambda: histgb_native(cat_idx), "INT"),
        GridCell("MLP", "integer", multilayer_perceptron, "INT"),
        GridCell("MLP", "one-hot", lambda: mlp_onehot(cat_idx), "INT"),
    ]

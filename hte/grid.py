"""The 12-configuration benchmark grid (5 architectures x 3 encodings).

This module reproduces ``results/grid_all16.csv``.  The cell definitions come
from ``scripts/tables.py:build_grid_all16`` (the script that rebuilt the shipped
table from the frozen v16 loader); the estimators come from
:mod:`hte.models`, and the loader / split / features from :mod:`hte.data` and
:mod:`hte.features`, all copied from v16.

The grid is 12 cells, not 15: Gradient Boosting, Random Forest and MLP have no
native-categorical cell.

Frame conventions (identical to tables.py)
------------------------------------------
``INT``  the integer-coded matrix returned by ``prepare_features``
``OH``   one-hot blocks built OUTSIDE the estimator (fit on training rows only)
``NAT``  the integer matrix with categorical columns cast to ``category`` dtype
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from . import params as P
from .features import get_split_positions, prepare_features
from .models import GridCell, categorical_indices, grid_cells

GRID_COLUMNS = ["dataset", "n_rows", "model", "encoding", "n_features",
                "r2_train", "r2_test", "rmse_train", "rmse_test",
                "fit_seconds", "status"]

#: (short alias, source stem) for every file in ``data/alias_mapping.yaml``.
#: The list used to be written out here; it now comes from that one table, so a
#: new dataset is added by editing the mapping -- never this module.
from .datasets import dataset_order as _dataset_order  # noqa: E402

DATASET_ORDER = _dataset_order()


def build_frames(X, feature_cols, tr_pos, te_pos=None):
    """Build the INT / OH / NAT frames exactly as tables.py does.

    Returns ``(packs, cat_cols, cat_idx)`` where ``packs[frame_key]`` is a
    ``(X_train, X_test)`` tuple.

    ``te_pos`` is passed through rather than recomputed with ``setdiff1d``: the
    test indices returned by ``train_test_split`` are in shuffled order, and
    keeping them in that exact order makes the row summation in ``r2_score``
    bit-identical to the reference run.
    """
    cat_cols = [c for c in feature_cols if str(c).endswith("_encoded")]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    cat_idx = categorical_indices(feature_cols)
    tr_pos = np.asarray(tr_pos, dtype=int)
    if te_pos is None:
        te_pos = np.setdiff1d(np.arange(len(X)), tr_pos)
    te_pos = np.asarray(te_pos, dtype=int)

    X_int = X.copy()

    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    oh.fit(X.iloc[tr_pos][cat_cols])
    blocks = [oh.transform(X[cat_cols])]
    if num_cols:
        blocks.append(X[num_cols].to_numpy(dtype=float))
    X_oh = pd.DataFrame(np.hstack(blocks).astype(np.float64),
                        columns=list(oh.get_feature_names_out(cat_cols)) + num_cols,
                        index=X.index)

    X_nat = X.copy()
    for c in cat_cols:
        # XGBoost's native-categorical path rejects float category codes, which is
        # what pandas produces when a column carries a NaN.  Fill, then declare
        # integer codes.
        X_nat[c] = X_nat[c].fillna(-1).astype("int64").astype("category")

    packs = {
        "INT": (X_int.iloc[tr_pos].copy(), X_int.iloc[te_pos].copy()),
        "OH": (X_oh.iloc[tr_pos].copy(), X_oh.iloc[te_pos].copy()),
        "NAT": (X_nat.iloc[tr_pos].copy(), X_nat.iloc[te_pos].copy()),
    }
    return packs, cat_cols, cat_idx


def run_dataset(df, dataset_key, raw_feature_names=None, cells=None) -> pd.DataFrame:
    """Run all 12 cells on one loaded dataset and return the grid rows."""
    tr_pos, te_pos = get_split_positions(len(df))
    X, y, feature_cols, _display, _encoders, _scaler = prepare_features(
        df, raw_feature_names, fit_rows=tr_pos)
    packs, _cat_cols, cat_idx = build_frames(X, feature_cols, tr_pos, te_pos)

    y_tr, y_te = y.iloc[tr_pos].copy(), y.iloc[te_pos].copy()

    rows = []
    for cell in (cells if cells is not None else grid_cells(cat_idx)):
        Xtr, Xte = packs[cell.frame]
        rec = dict(dataset=dataset_key, n_rows=len(df), model=cell.architecture,
                   encoding=cell.encoding, n_features=int(Xtr.shape[1]),
                   r2_train=np.nan, r2_test=np.nan, rmse_train=np.nan,
                   rmse_test=np.nan, fit_seconds=np.nan, status="ok")
        t0 = time.perf_counter()
        try:
            mdl = cell.factory()
            fast_rf = (hasattr(mdl, "n_jobs")
                       and type(mdl).__name__ == "RandomForestRegressor"
                       and P.RF_N_JOBS and P.RF_N_JOBS > 1)
            if fast_rf:
                mdl.n_jobs = P.RF_N_JOBS
            mdl.fit(Xtr, y_tr)
            if fast_rf:
                mdl.n_jobs = None
            p_tr = np.asarray(mdl.predict(Xtr), dtype=float).ravel()
            p_te = np.asarray(mdl.predict(Xte), dtype=float).ravel()
            rec["r2_train"] = float(r2_score(y_tr, p_tr))
            rec["r2_test"] = float(r2_score(y_te, p_te))
            rec["rmse_train"] = float(np.sqrt(mean_squared_error(y_tr, p_tr)))
            rec["rmse_test"] = float(np.sqrt(mean_squared_error(y_te, p_te)))
            rec["fit_seconds"] = time.perf_counter() - t0
        except Exception as exc:                     # keep the cell, record why
            rec["status"] = "ERROR: %s: %s" % (type(exc).__name__, exc)
            rec["fit_seconds"] = time.perf_counter() - t0
        rows.append(rec)
    return pd.DataFrame(rows, columns=GRID_COLUMNS)


def run_grid(datasets=None, data_dir=None, out_path=None, verbose=True) -> pd.DataFrame:
    """Run the grid for ``datasets`` (default: all 16) and optionally write CSV.

    ``datasets`` is an iterable of ``(alias, source_stem)`` pairs; the default is
    :data:`DATASET_ORDER`.  ``data_dir`` is the directory holding the raw CSVs
    and ``col_mappings.csv``; it is required because raw data are not
    redistributed with the repository.
    """
    from pathlib import Path

    from .data import load_real_dataset

    if data_dir is None:
        raise ValueError("data_dir is required: point it at the directory holding "
                         "the raw dataset CSVs and col_mappings.csv")
    datasets = list(datasets) if datasets is not None else list(DATASET_ORDER)
    data_dir = Path(data_dir)
    parts = []
    t_all = time.perf_counter()
    for i, (key, stem) in enumerate(datasets, 1):
        if verbose:
            print("[grid %2d/%d] %s" % (i, len(datasets), key), flush=True)
        df, raw = load_real_dataset(str(data_dir / (stem + ".csv")), str(data_dir))
        parts.append(run_dataset(df, key, raw))
    grid = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=GRID_COLUMNS)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        grid.to_csv(out_path, index=False, encoding="utf-8")
        if verbose:
            print("[grid] wrote %s (%d rows) in %.1f s"
                  % (out_path, len(grid), time.perf_counter() - t_all))
    return grid

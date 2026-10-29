"""Feature preparation and the canonical train/test split.

Copied verbatim from ``archive/Analysis_all_v16.py`` (``get_split_positions``
and ``prepare_features``).  The only change is that the module-global
``FIT_PREPROCESSING_ON_TRAIN_ONLY`` / ``TEST_SIZE`` are read from
:mod:`hte.params`.
"""
from __future__ import annotations
from .v16_core import (get_shap_values, get_split_positions, prepare_features)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import params as P

# Columns whose constant-value warning has already been printed in this process
_REPORTED_CONST = set()


def _local_get_split_positions(n_rows, test_size=None, random_state=42):
    """
    The ONE definition of the train/test split, as positional indices.

    Every train_test_split call in the pipeline uses
    (test_size=TEST_SIZE, random_state=42), and splitting `np.arange(n_rows)`
    returns positional indices, so this reproduces the split of the feature
    matrix exactly.
    """
    if test_size is None:
        test_size = P.TEST_SIZE
    idx_tr, idx_te = train_test_split(np.arange(int(n_rows)), test_size=test_size,
                                      random_state=random_state)
    return np.asarray(idx_tr), np.asarray(idx_te)


def _local_prepare_features(df, raw_feature_names=None, fit_rows=None):
    """
    Encode + scale the feature matrix.

    fit_rows
    --------
    Positional row indices on which to FIT the StandardScaler.  Pass the training
    rows so the scaler never sees the held-out set -- the scikit-learn convention:
    fit on train, transform everything.  The RETURNED matrix is still the transform
    of the whole frame, identical in shape to before; only the two numbers the
    scaler learned change.

    fit_rows=None reproduces the old (full-data) behaviour exactly.
    """
    df_encoded = df.copy()
    categorical_cols = list(P.CATEGORICAL_SLOTS)
    numerical_cols = list(P.NUMERIC_SLOTS)

    categorical_cols = [c for c in categorical_cols if c in df_encoded.columns]

    # v22e: prepare_features runs more than once per session (main, then the figure path
    # that needs the matrix).  Each run printed the same constant-slot lines again, so the
    # console showed every warning twice.  Report each column ONCE per process.
    global _REPORTED_CONST

    # A slot that is constant within a dataset (always absent, or always the same
    # value) carries no information: drop it instead of feeding a fake column.
    valid_categorical_cols = []
    for col in categorical_cols:
        n = df_encoded[col].astype(str).nunique(dropna=False)
        if n > 1:
            valid_categorical_cols.append(col)
        else:
            if col not in _REPORTED_CONST:
                _REPORTED_CONST.add(col)
                print(f" 槽 '{col}' 在本数据集为常量（唯一值为 "
                      f"{df_encoded[col].astype(str).iloc[0]}），不参与建模。")
    categorical_cols = valid_categorical_cols

    valid_numerical_cols = []
    for col in numerical_cols:
        if col in df_encoded.columns:
            if df_encoded[col].nunique() > 1:
                valid_numerical_cols.append(col)
            else:
                if col not in _REPORTED_CONST:
                    _REPORTED_CONST.add(col)
                    print(f" 特征 '{col}' 为常量（唯一值为 {df_encoded[col].iloc[0]}），"
                          f"已自动从模型训练集中剔除。")

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

    feature_cols = [col + '_encoded' for col in categorical_cols] + valid_numerical_cols
    X = df_encoded[feature_cols]
    y = df_encoded['yield']

    scaler = StandardScaler()
    numerical_indices = [feature_cols.index(col) for col in valid_numerical_cols]
    X_scaled = X.copy()
    if len(numerical_indices) > 0:
        num_block = X.iloc[:, numerical_indices]

        # v14: MEDIAN-IMPUTE missing numeric features before scaling.
        # Real HTE exports have gaps; StandardScaler would propagate NaN and
        # GradientBoostingRegressor raises "Input X contains NaN".  The fill values
        # come from the TRAINING rows only, matching the scaler policy above.
        _na_mask = num_block.isna()
        if _na_mask.to_numpy().any():
            if P.FIT_PREPROCESSING_ON_TRAIN_ONLY and fit_rows is not None and len(fit_rows) > 0:
                _r = np.asarray(fit_rows)
                _r = _r[(_r >= 0) & (_r < len(num_block))]
                _fill = num_block.iloc[_r].median()
            else:
                _fill = num_block.median()
            _fill = _fill.fillna(0.0)          # a column that is entirely NaN
            _counts = _na_mask.sum()
            num_block = num_block.fillna(_fill)
            X_scaled.iloc[:, numerical_indices] = num_block
            print(f"  [NaN] imputed {int(_counts.sum())} missing numeric feature value(s) with "
                  f"the training median: "
                  + ', '.join(f'{c}={_counts[c]} ({100*_counts[c]/len(num_block):.0f}%)'
                              for c in num_block.columns if _counts[c] > 0))

        if P.FIT_PREPROCESSING_ON_TRAIN_ONLY and fit_rows is not None and len(fit_rows) > 0:
            rows = np.asarray(fit_rows)
            rows = rows[(rows >= 0) & (rows < len(num_block))]
            scaler.fit(num_block.iloc[rows])
        else:
            scaler.fit(num_block)
        X_scaled = X_scaled.astype(float)
        X_scaled.iloc[:, numerical_indices] = scaler.transform(num_block)

    display_feature_names = []
    for f in feature_cols:
        base_col = f.replace('_encoded', '')
        if raw_feature_names and base_col in raw_feature_names:
            display_feature_names.append(raw_feature_names[base_col])
        else:
            display_feature_names.append(base_col)

    # v16: these labels are the RAW column names, and splitting `catalyst` from
    # `ligand` makes a collision possible.  Two identical labels turn
    # `X.columns = list(feature_names)` inside get_shap_values into a frame with
    # DUPLICATE columns.  Disambiguate the LABELS only.
    _seen_nm = {}
    _uniq_nm = []
    for _nm in display_feature_names:
        _k = _seen_nm.get(_nm, 0)
        _seen_nm[_nm] = _k + 1
        _uniq_nm.append(_nm if _k == 0 else '%s [%d]' % (_nm, _k + 1))
    display_feature_names = _uniq_nm

    return X_scaled, y, feature_cols, display_feature_names, label_encoders, scaler

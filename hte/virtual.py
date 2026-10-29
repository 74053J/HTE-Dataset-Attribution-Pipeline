# -*- coding: utf-8 -*-
"""Virtual-library machinery, extracted from the frozen pipeline.

These helpers and constants used to be fetched from ``single_dataset_full_benchmark.py``
through a loader at run time, which kept the recommendation route dependent
on the monolith.  They are copied here verbatim so that the package owns them; the
frozen pipeline is then only a historical reference.

Extracted 2026-10-28 from single_dataset_full_benchmark.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_LOADING_POINTS = 100

ENSEMBLE_MIN_WEIGHT_R2 = 0.05

ENSEMBLE_WEIGHT_POWER = 1.0

ENSEMBLE_MAX_MODEL_GAP = 20.0

PREDICTION_MIN = 0.0

PREDICTION_MAX = 100.0

PREDICT_BATCH_ROWS = 250_000

RF_N_ESTIMATORS = 100

HISTGB_MAX_ITER = 100# Trees for RandomForestRegressor, down from 200.  RF is 15.6% of the real sweep's predict

CLIP_MEMBER_PREDICTIONS = True

EI_INCUMBENT_MODE = 'quantile'      # 'quantile' | 'max'

EI_INCUMBENT_QUANTILE = 0.90        # used only when EI_INCUMBENT_MODE == 'quantile'

ENUM_AXES = ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1']

LOADING_AXIS = {
    'col': None, 'source_col': None, 'role': 'unavailable', 'label': None,
    'unit': None, 'n_distinct': 0, 'reason': 'not resolved yet',
}

def get_enumeration_spec(df, feat_cols):
    """
    THE single definition of the enumerated space.  Both `stream_virtual_library` and the
    parallel feature cache must agree on it exactly, so it lives in one place.

    Returns (categorical_cols, value_lists, n_cat, inner).

    `inner[p]` is the stride of column p in the flat combination index: the LAST category
    varies fastest, which matches the row order the in-memory builder produced.

    Do NOT try to derive the category lists from a small generated chunk: `category`
    dtype only records the values that actually appeared, so a 1000-combination probe
    reports 1 catalyst instead of 37 and silently shrinks the whole enumeration.
    """
    categorical_cols = [
        c for c in ENUM_AXES
        if c in df.columns and c + '_encoded' in feat_cols
    ]

    value_lists = []
    for col in categorical_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        value_lists.append(vals if vals else ['Unknown'])

    n_cat = 1
    for v in value_lists:
        n_cat *= len(v)
    inner = [int(np.prod([len(v) for v in value_lists[p + 1:]], dtype=np.int64))
             for p in range(len(value_lists))]
    return categorical_cols, value_lists, int(n_cat), inner



def get_loading_grid(df, n_points=N_LOADING_POINTS, quantile_range=(0.002, 0.998)):
    """
    Loading values to enumerate: `n_points` EVENLY SPACED values between two quantiles.

    Two things this is NOT, both of which are easy to assume wrongly:

    1. It is not a 0..1 grid.  The range is a quantile band of the historical loadings
       (default 0.2 %..99.8 %), which on this dataset is 2.67e-06 .. 0.4482 eq.  Using
       0..1 would spend most of the grid on loadings that never occur: only 3 of 10,138
       experimental rows (0.03 %) use more than 1 eq.

    2. It is not a quantile SAMPLING.  Only the two ENDPOINTS are quantiles; the interior
       is `np.linspace`, i.e. uniformly spaced.  That is deliberate.  The loading
       distribution is extremely right-skewed (median 0.0733, p75 0.0811, but the 0.2 %
       point is 2.67e-06), so sampling true quantiles would put ~80 of 100 points below the
       median and leave the upper half of the axis unresolved.  Uniform spacing over a
       quantile-bounded interval keeps the whole reachable loading axis covered.

    The band width is a real trade-off: narrowing the trim does NOT narrow the grid -- going
    from a 0.5 % trim to 0.2 % grows the span 1.45x, because the lower bound collapses
    toward 0 while the upper bound only rises.  Tighter trims buy a few more legitimate
    high-loading points at the cost of spending low-grid points on a nearly empty region.

    v14: `catalyst_loading` is absent when no usable amount axis exists (see
    resolve_loading_axis).  In that case the returned single value is a PLACEHOLDER: it is
    not a model feature, so the sweep would emit N_LOADING_POINTS identical rows per reagent
    combination.  get_loading_grid_for() reads LOADING_AXIS and collapses the grid to ONE
    point in that situation, which avoids the duplicate work; the value itself is irrelevant.
    """
    if not loading_axis_available() or 'catalyst_loading' not in df.columns:
        return np.array([0.0], dtype=float)

    vals = pd.to_numeric(df['catalyst_loading'], errors='coerce').dropna()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([0.0], dtype=float)

    lo = float(vals.quantile(quantile_range[0]))
    hi = float(vals.quantile(quantile_range[1]))
    if np.isclose(lo, hi):
        return np.array([float(vals.median())], dtype=float)

    # anchored on real measured loadings, uniformly spaced over the dense range
    return np.linspace(lo, hi, n_points)


def get_representative_conditions(df):
    representative = {}
    # reactant_2 / reactant_1 are the current slot names; the retired names hold only
    # 'Unknown' placeholders, so a scored condition could not report its partner pair.
    for col in ['metal', 'catalyst', 'ligand', 'base', 'solvent',
                'reactant_2', 'reactant_1']:
        if col in df.columns:
            mode = df[col].dropna().astype(str).mode()
            representative[col] = mode.iloc[0] if len(mode) else 'Unknown'

    for col in ['temperature', 'time', 'catalyst_loading']:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            representative[col] = float(vals.median()) if len(vals) else 0.0

    return representative


def build_virtual_feature_frame(df, feat_cols, label_encoders, scaler, records):
    raw = pd.DataFrame(records).copy()
    # derive the encoded slots from feat_cols, not from a hard-coded list of the
    # retired names: the encoder keys are the current slot names, so the old
    # list left reactant_1/2_encoded as NaN.
    categorical_cols = [str(c)[:-len('_encoded')] for c in feat_cols
                        if str(c).endswith('_encoded')]
    # derive the amount axes from feat_cols: the hard-coded list still used the
    # retired names (reactant_2_eq / base_eq), so the new *_eq columns were left
    # as NaN and the model refused to predict.
    numerical_cols = [c for c in feat_cols
                      if not str(c).endswith('_encoded') and c in df.columns]

    encoded = pd.DataFrame(index=raw.index)

    for col in categorical_cols:
        encoded_name = col + '_encoded'
        if encoded_name not in feat_cols:
            continue
        le = label_encoders.get(col)
        if le is None:
            continue
        if col not in raw.columns:
            # The enumerated frame only carries the search axes; a slot that is held
            # fixed has no column here, so pin it to its modal value.
            _mode = df[col].astype(str).mode()
            _default = _mode.iloc[0] if len(_mode) else str(le.classes_[0])
            encoded[encoded_name] = le.transform([_default] * len(raw))
            continue
        values = raw[col].astype(str)
        encoded[encoded_name] = le.transform(values)

    for col in numerical_cols:
        if col not in raw.columns:
            # held-fixed amount axis: pin it to the dataset median
            _m = pd.to_numeric(df[col], errors="coerce")
            encoded[col] = float(_m.median()) if _m.notna().any() else 0.0
            continue
        encoded[col] = pd.to_numeric(raw[col], errors='coerce')
        fallback = float(df[col].median()) if pd.to_numeric(df[col], errors='coerce').notna().any() else 0.0
        encoded[col] = encoded[col].fillna(fallback)

    encoded = encoded.reindex(columns=feat_cols)
    numerical_indices = [i for i, c in enumerate(feat_cols) if not c.endswith('_encoded')]
    if numerical_indices:
        encoded_values = encoded.iloc[:, numerical_indices].astype(float).values
        encoded.iloc[:, numerical_indices] = scaler.transform(encoded_values)

    return encoded.astype(float)



def predict_in_batches(model, X, batch_size=5000000, desc='Prediction', show_progress=False):
    """
    Predict in chunks and concatenate.

    TIP: for a 45.7e6-row virtual library this returns a 0.34 GiB float64 array per
    model.  If you only need ensemble mean/spread, use `ensemble_stats_streaming`
    instead, which never holds more than one batch per model.
    """
    outputs = []
    pbar = tqdm(total=len(X), desc=desc, unit='rows', dynamic_ncols=True) if show_progress else None

    try:
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            batch_pred = np.asarray(model.predict(X.iloc[start:stop]), dtype=float)
            outputs.append(batch_pred)
            if pbar is not None:
                pbar.update(stop - start)
    finally:
        if pbar is not None:
            pbar.close()

    return np.concatenate(outputs) if outputs else np.array([], dtype=float)



def calculate_expected_improvement(mean_prediction, uncertainty, best_observed_yield, xi=0.01):
    from scipy.stats import norm

    mu = np.asarray(mean_prediction, dtype=float)
    sigma = np.asarray(uncertainty, dtype=float)
    sigma_safe = np.maximum(sigma, 1e-12)

    improvement = mu - float(best_observed_yield) - xi
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)

    zero_sigma = sigma <= 1e-12
    if np.any(zero_sigma):
        ei[zero_sigma] = np.maximum(improvement[zero_sigma], 0.0)

    return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)


def resolve_ei_incumbent(df):
    """
    Return (f_star, all_time_record) for the expected-improvement incumbent.

    Single source of truth: both the virtual-library sweep and Figure 10b call this, so the
    EI column and the EI-ranked table can never disagree about what they were measured
    against.  See EI_INCUMBENT_MODE for why the choice matters.
    """
    _y = pd.to_numeric(df['yield'], errors='coerce').dropna()
    _record = float(_y.max()) if len(_y) else 0.0
    if EI_INCUMBENT_MODE == 'quantile' and len(_y):
        return float(np.percentile(_y.to_numpy(), EI_INCUMBENT_QUANTILE * 100.0)), _record
    return _record, _record


def audit_condition_scores(pool, df, models, feat_cols, label_encoders, scaler,
                           weights=None, tol=0.01, tag='', repair=True):
    """
    Verify that every row's stored ensemble score actually belongs to its stored
    condition, by RE-SCORING those conditions from scratch and comparing.

    Why this exists
    ---------------
    While building the parallel engine I found that the older streaming engine
    (`generate_bo_results_streaming`) can return a pool in which the scores are attached
    to the WRONG conditions: re-scoring 5,000 returned rows independently disagreed for
    4,501 of them, by up to 51 points.  The feature builders themselves are provably
    identical (max|diff| = 0.00e+00, and the member predictions agree exactly), so the
    corruption is in that engine's pool assembly, not in the encodings.

    A silent condition/score mismatch is the worst possible failure mode for a paper: the
    table looks fine and the chemistry is wrong.  Rather than rely on any single code path
    being correct, every engine now audits its own output and (by default) DROPS rows that
    fail, so a returned pool is always self-consistent.

    Returns (pool, n_bad, n_total).
    """
    if pool is None or len(pool) == 0:
        return pool, 0, 0
    cond_cols = [c for c in ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2',
                             'reactant_1', 'temperature', 'time', 'catalyst_loading']
                 if c in pool.columns]
    if len(cond_cols) < 5 or 'Ensemble Predicted Yield (%)' not in pool.columns:
        return pool, 0, len(pool)
    try:
        feat = build_virtual_feature_frame(df, feat_cols, label_encoders, scaler,
                                           pool[cond_cols])
        items = list(models.items())
        if weights is None:
            wv = np.full(len(items), 1.0 / len(items))
        else:
            wv = np.asarray([float(x) for x in weights], dtype=float)
            wv = np.clip(wv, 0.0, None)
            wv = wv / wv.sum()
        if len(wv) != len(items):
            return pool, 0, len(pool)
        acc = np.zeros(len(pool))
        for j, (nm, mdl) in enumerate(items):
            p = np.asarray(mdl.predict(feat), dtype=np.float64)
            if CLIP_MEMBER_PREDICTIONS:
                p = np.clip(p, PREDICTION_MIN, PREDICTION_MAX)
            acc += wv[j] * p
        stored = pool['Ensemble Predicted Yield (%)'].to_numpy(float)
        bad = np.abs(stored - acc) > float(tol)
        n_bad = int(bad.sum())
        if n_bad:
            print(f"  [audit{(' ' + tag) if tag else ''}] {n_bad:,} / {len(pool):,} rows had a "
                  f"score that does not match their stated condition "
                  f"(max error {np.abs(stored - acc).max():.2f} pp) -- "
                  f"{'dropping them' if repair else 'keeping them'}.")
            if repair:
                pool = pool.loc[~bad].reset_index(drop=True)


        return pool, n_bad, len(stored)
    except Exception as e:
        print(f"  [audit{(' ' + tag) if tag else ''}] could not verify "
              f"({'/'.join(str(e).splitlines()[:1])}).")
        return pool, 0, len(pool)



def loading_axis_available():
    return LOADING_AXIS.get('col') is not None



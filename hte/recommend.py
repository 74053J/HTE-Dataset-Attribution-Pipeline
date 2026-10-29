"""Condition recommendation: rank real records, and a capped virtual-library wrapper.

Two routes, mirroring the paper:

``rank_real_records``
    Rank the REAL, measured reaction records by the ensemble mean predicted
    yield.  The ranking core is copied from
    ``Analysis_all_v16.generate_figure6_direct_ranking`` (ensemble weighting,
    member clipping, the conservative per-condition train/held-out label,
    ``dedupe_conditions``, the model-agreement gate and the final sort).  The
    matplotlib rendering is deliberately not part of this module.

``enumerate_virtual_capped``
    A hard-capped wrapper around v16's virtual-library path.  It uses v16's
    ``get_enumeration_spec`` / ``get_loading_grid`` / ``get_representative_conditions``
    / ``build_virtual_feature_frame`` / ``predict_in_batches`` /
    ``calculate_expected_improvement`` (all copied below) and stops after
    ``max_combos`` rows.  It REFUSES to run unless ``allow_virtual=True``.
    ``plan_enumeration`` reports the search space and cost without scoring.

NOTE: the full enumeration is never executed by the test suite or by
``run_all_fast.bat``.  Only ``plan_enumeration`` / a tiny smoke cap should be
used outside ``run_all.bat``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from . import params as P
from .data import ENUM_AXES, loading_axis_available
from .features import get_split_positions


class VirtualEnumerationRefused(RuntimeError):
    """Raised when the virtual path is called without ``allow_virtual=True``."""


# ensemble weighting (v16 / recommend_virtual.py rule)
def select_members(r2_map: dict, top_k: int = 4, allowed=None):
    """Rank members by held-out R^2 and apply v16's ensemble-weight rule."""
    floor = float(P.ENSEMBLE_MIN_WEIGHT_R2)
    power = float(P.ENSEMBLE_WEIGHT_POWER)
    candidates = list(allowed or r2_map)
    names = [n for n in candidates
             if n in r2_map and np.isfinite(float(r2_map[n]))]
    if names:
        scores = np.array([float(r2_map[n]) for n in names], dtype=float)
    else:
        names = [n for n in candidates if n in r2_map]
        if not names:
            raise ValueError("no ensemble member has a usable held-out R^2")
        scores = np.full(len(names), floor, dtype=float)
    order = np.argsort(-scores, kind="stable")
    names = [names[i] for i in order[:max(1, int(top_k))]]
    scores = scores[order[:max(1, int(top_k))]]
    raw_w = np.power(np.maximum(scores, floor), power)
    return names, raw_w / raw_w.sum()


# replicate collapsing (verbatim from v16:5763)
def dedupe_conditions(pool, condition_cols, verbose=True):
    """
    Collapse replicate experiments: one row per distinct reaction CONDITION.

    Grouped on the full condition key.  Because the model is a function of the
    condition, all prediction columns agree within a group, so they are taken
    as-is (max).  The measured columns are AVERAGED -- ``mean(yield | condition)``
    is exactly the quantity the regressor estimates -- and the within-condition
    spread is kept for transparency (``N Runs``, ``Yield SD (pp)``).
    """
    key = [c for c in condition_cols if c in pool.columns]
    if not key or len(pool) == 0:
        return pool, 0

    n_before = len(pool)
    present = set(pool.columns)
    agg = {}
    for c in present:
        if c in key:
            continue
        if c == 'yield':
            agg[c] = 'mean'
        elif pd.api.types.is_numeric_dtype(pool[c]):
            agg[c] = 'max'          # identical within a condition (model output)
        else:
            agg[c] = 'first'        # descriptors etc.

    grouped = pool.groupby(key, dropna=False, as_index=False).agg(agg)

    # replicate bookkeeping
    runs = pool.groupby(key, dropna=False)['yield'].agg(['count', 'std']).reset_index()
    grouped = grouped.merge(runs, on=key, how='left')
    grouped = grouped.rename(columns={'count': 'N Runs', 'std': 'Yield SD (pp)'})
    grouped['N Runs'] = grouped['N Runs'].fillna(1).astype(int)

    n_after = len(grouped)
    if verbose:
        print(f"  [dedupe] {n_before:,} measured runs -> {n_after:,} distinct conditions "
              f"({n_before - n_after:,} replicate rows collapsed; "
              f"{(n_before - n_after) / max(n_before, 1):.0%} were repeats)")
    return grouped, n_before - n_after


# per-condition split label (verbatim from v16:5826)
def _condition_split_table(df, train_idx, condition_cols):
    """Per-CONDITION train membership, keyed so it can be merged onto the pool.

    The rule is the CONSERVATIVE one: a condition is 'train' only if EVERY one of
    its replicate rows was in the training split.
    """
    key_cols = [c for c in condition_cols if c in df.columns]
    if not key_cols:
        return pd.DataFrame(columns=['_ckey', 'Split']), key_cols

    def _canon(cols):
        out = pd.DataFrame(index=df.index)
        for c in cols:
            out[c] = df[c].astype(str)
        return out

    is_train = np.zeros(len(df), dtype=bool)
    idx = np.asarray(train_idx, dtype=int)
    idx = idx[(idx >= 0) & (idx < len(df))]
    is_train[idx] = True

    work = _canon(key_cols)
    work['_ckey'] = work[key_cols].agg('|'.join, axis=1)
    work['_is_tr'] = is_train
    all_tr = work.groupby('_ckey', sort=False)['_is_tr'].apply(lambda s: bool(s.all()))
    split_tbl = pd.DataFrame({'_ckey': all_tr.index,
                              'Split': np.where(all_tr.to_numpy(), 'train', 'held-out')})
    out = pd.DataFrame({'_ckey': np.unique(work['_ckey'].to_numpy())})
    return out.merge(split_tbl, on='_ckey', how='left'), key_cols


# route 1: rank real records
def rank_real_records(df, predictions, members=None, weights=None,
                      max_model_gap=None, condition_cols=None,
                      add_split=True, verbose=True):
    """Rank real measured reactions by the weighted ensemble predicted yield.

    Parameters
    ----------
    df : DataFrame returned by an ``hte`` loader (must contain ``yield``).
    predictions : mapping member name -> array-like of length ``len(df)``.
        Values are clipped to ``[PREDICTION_MIN, PREDICTION_MAX]`` exactly as v16
        does before combining members.
    members : ordered list of member names (default: ``list(predictions)``).
    weights : non-negative weights aligned with ``members`` (default: equal).
    max_model_gap : float or None.  Gate A from v16; None disables the gate.
        v16's published default is ``params.ENSEMBLE_MAX_MODEL_GAP`` (20 pp),
        which the caller can pass explicitly.
    condition_cols : dedupe key (default: ``params.CONDITION_COLS``).
    add_split : attach the conservative per-condition train/held-out label.

    Returns the ranked DataFrame (one row per distinct condition), best first.
    """
    members = list(members) if members is not None else list(predictions)
    if len(members) < 2:
        raise ValueError("rank_real_records needs at least two ensemble members")
    cond_cols = list(condition_cols) if condition_cols is not None else list(P.CONDITION_COLS)

    pool = df.copy()
    mat = np.column_stack([np.asarray(predictions[m], dtype=float) for m in members])
    if len(mat) != len(pool):
        raise ValueError("prediction length %d != dataset length %d"
                         % (len(mat), len(pool)))
    if not np.isfinite(mat).all():
        raise ValueError("prediction array contains non-finite values")

    if P.CLIP_MEMBER_PREDICTIONS:
        n_bad = int(np.count_nonzero((mat < P.PREDICTION_MIN) | (mat > P.PREDICTION_MAX)))
        mat = np.clip(mat, P.PREDICTION_MIN, P.PREDICTION_MAX)
        if n_bad and verbose:
            print(f"  [rank-real] clipped {n_bad:,} member predictions to "
                  f"[{P.PREDICTION_MIN:g}, {P.PREDICTION_MAX:g}]")

    if weights is not None and len(weights) == mat.shape[1]:
        wv = np.asarray(weights, dtype=float)
        wv = np.clip(wv, 0.0, None)
        wv = wv / wv.sum()
        mean_v = (mat * wv[None, :]).sum(axis=1)
        var_v = (wv[None, :] * (mat - mean_v[:, None]) ** 2).sum(axis=1) / \
            max(1.0 - float((wv ** 2).sum()), 1e-12)
        if verbose:
            print("  [rank-real] weighted ensemble mean, weights "
                  + str(dict(zip(members, wv.round(4)))))
    else:
        mean_v = mat.mean(axis=1)
        var_v = mat.std(axis=1, ddof=1)
        if verbose:
            print("  [rank-real] unweighted ensemble mean (no weights supplied)")

    pool['Ensemble Predicted Yield (%)'] = mean_v
    pool['Prediction Uncertainty (%)'] = np.sqrt(np.maximum(var_v, 0.0))
    pool['Model Gap (%)'] = mat.max(axis=1) - mat.min(axis=1)
    pool['Model Min (%)'] = mat.min(axis=1)
    pool['Model Max (%)'] = mat.max(axis=1)
    pool['Model Preset'] = ', '.join(members)
    for j, nm in enumerate(members):
        pool[f'Pred_{nm}'] = mat[:, j]

    if add_split:
        try:
            idx_tr, _idx_te = get_split_positions(len(df))
            _split_tbl, _key_cols = _condition_split_table(df, idx_tr, cond_cols)
            if _key_cols:
                _pool_key = np.asarray(
                    df[_key_cols].astype(str).agg('|'.join, axis=1).reindex(pool.index))
                pool = pool.merge(_split_tbl, left_on=_pool_key, right_on='_ckey',
                                  how='left')
                pool = pool.drop(columns=['_ckey'])
                pool['Split'] = pool['Split'].fillna('unknown')
        except Exception as exc:                     # never fail the ranking for a label
            if verbose:
                print(f"  [rank-real] could not reconstruct the split: {exc}")
            pool['Split'] = 'unknown'

    n_pool_rows = len(pool)
    pool, n_replicates = dedupe_conditions(pool, condition_cols=cond_cols,
                                           verbose=verbose)
    if verbose and n_pool_rows != len(pool):
        print(f"  [rank-real] {n_replicates:,} replicate row(s) collapsed")

    if max_model_gap is not None:
        before = len(pool)
        pool = pool[pool['Model Gap (%)'] <= float(max_model_gap)].copy()
        if verbose:
            print(f"  [Gate A] Model Gap <= {max_model_gap:g} pp: kept "
                  f"{len(pool):,} / {before:,} conditions")
    if len(pool) == 0:
        return pool

    ranked = pool.sort_values('Ensemble Predicted Yield (%)',
                              ascending=False).reset_index(drop=True)
    ranked.insert(0, 'Rank', np.arange(1, len(ranked) + 1))
    ranked['Prediction Error (%)'] = (ranked['Ensemble Predicted Yield (%)']
                                      - ranked['yield'])
    return ranked


# virtual-library helpers (verbatim from v16)
def get_representative_conditions(df):
    representative = {}
    for col in P.CATEGORICAL_SLOTS:
        if col in df.columns:
            mode = df[col].dropna().astype(str).mode()
            representative[col] = mode.iloc[0] if len(mode) else 'Unknown'

    for col in P.NUMERIC_SLOTS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            representative[col] = float(vals.median()) if len(vals) else 0.0

    return representative


def get_loading_grid(df, n_points=None, quantile_range=(0.002, 0.998)):
    """``n_points`` evenly spaced values between two quantiles of measured loadings.

    Returns a single placeholder 0.0 when no usable amount axis exists (v16
    behaviour: the value is not a model feature, so it is irrelevant).
    """
    if n_points is None:
        n_points = int(P.N_LOADING_POINTS)
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

    return np.linspace(lo, hi, n_points)


def get_loading_grid_for(df, max_combinations, n_cat, n_points=None):
    """The loading grid a given cap actually permits (shared by both engines)."""
    requested = get_loading_grid(df, n_points=n_points)
    allowed = max(1, int(max_combinations // max(n_cat, 1)))
    if allowed >= len(requested):
        return requested
    ix = np.unique(np.linspace(0, len(requested) - 1, allowed).round().astype(int))
    return requested[ix]


def get_enumeration_spec(df, feat_cols):
    """THE single definition of the enumerated space (verbatim from v16:2295).

    Returns (categorical_cols, value_lists, n_cat, inner).  ``inner[p]`` is the
    stride of column p in the flat combination index: the LAST category varies
    fastest.
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


def build_virtual_feature_frame(df, feat_cols, label_encoders, scaler, records):
    raw = pd.DataFrame(records).copy()
    categorical_cols = [c for c in P.CATEGORICAL_SLOTS
                        if c in df.columns and c in feat_cols]
    numerical_cols = [c for c in P.NUMERIC_SLOTS if c in df.columns and c in feat_cols]

    encoded = pd.DataFrame(index=raw.index)

    for col in categorical_cols:
        encoded_name = col + '_encoded'
        if encoded_name not in feat_cols:
            continue
        le = label_encoders.get(col)
        if le is None:
            continue
        values = raw[col].astype(str)
        encoded[encoded_name] = le.transform(values)

    for col in numerical_cols:
        encoded[col] = pd.to_numeric(raw[col], errors='coerce')
        fallback = float(df[col].median()) if pd.to_numeric(df[col], errors='coerce').notna().any() else 0.0
        encoded[col] = encoded[col].fillna(fallback)

    encoded = encoded.reindex(columns=feat_cols)
    numerical_indices = [i for i, c in enumerate(feat_cols) if not c.endswith('_encoded')]
    if numerical_indices:
        encoded_values = encoded.iloc[:, numerical_indices].astype(float).values
        encoded.iloc[:, numerical_indices] = scaler.transform(encoded_values)

    return encoded.astype(float)


def predict_in_batches(model, X, batch_size=None, desc='Prediction', show_progress=False):
    """Predict in chunks and concatenate (verbatim from v16:2550)."""
    if batch_size is None:
        batch_size = int(P.PREDICT_BATCH_SIZE)
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
    """Expected improvement (verbatim from v16:3793)."""
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


# capped enumeration plan
def plan_enumeration(df, feat_cols, max_combos):
    """Resolve the enumerable space and the exact rows a cap would score.

    Mirrors ``scripts/recommend_virtual.py:_enumeration_plan``: the cap first
    covers every categorical combination, then spends the remainder on the
    loading axis; if it cannot cover the product it samples the product in
    index space at one loading point.
    """
    max_combos = int(max_combos)
    if max_combos <= 0:
        raise ValueError("max_combos must be a positive integer")

    cat_cols, value_lists, n_cat, inner = get_enumeration_spec(df, feat_cols)
    n_cat = int(n_cat)
    grid_full = np.asarray(get_loading_grid(df, n_points=int(P.N_LOADING_POINTS)),
                           dtype=float)
    if grid_full.size == 0:
        grid_full = np.array([0.0], dtype=float)

    full_rows = int(n_cat) * int(grid_full.size)
    effective_cap = min(max_combos, full_rows) if full_rows else 0

    if effective_cap >= n_cat:
        n_load = max(1, min(int(grid_full.size), effective_cap // max(n_cat, 1)))
        mode = "full-product"
    else:
        n_load = 1
        mode = "sampled"

    if n_load >= grid_full.size:
        load_values = grid_full
    elif n_load == 1:
        median = 0.0
        if "catalyst_loading" in df.columns:
            vals = pd.to_numeric(df["catalyst_loading"], errors="coerce").dropna()
            if len(vals):
                median = float(vals.median())
        load_values = np.array(
            [float(grid_full[int(np.argmin(np.abs(grid_full - median)))])],
            dtype=float)
    else:
        idx = np.unique(
            np.linspace(0, grid_full.size - 1, n_load).round().astype(int))
        load_values = grid_full[idx]

    rows = int(n_cat) * int(load_values.size)
    if mode == "sampled":
        rows = int(effective_cap)
    return {
        "feat_cols": list(feat_cols),
        "cat_cols": list(cat_cols),
        "value_lists": [list(v) for v in value_lists],
        "inner": [int(v) for v in inner],
        "n_cat": n_cat,
        "grid_full": grid_full,
        "load_values": load_values,
        "n_load": int(load_values.size),
        "full_rows": full_rows,
        "cap": max_combos,
        "effective_cap": int(effective_cap),
        "rows": int(rows),
        "mode": mode,
    }


def _iter_plan_chunks(df, plan, chunk_rows):
    """Yield bounded DataFrames covering exactly ``plan['rows']`` combinations."""
    representative = get_representative_conditions(df)
    out_cols = list(representative.keys())
    cat_cols = plan["cat_cols"]
    value_lists = plan["value_lists"]
    inner = plan["inner"]
    n_cat = plan["n_cat"]
    grid = np.asarray(plan["load_values"], dtype=float)
    n_load = int(plan["n_load"])
    chunk_rows = max(1, int(chunk_rows))

    def _build(combo_idx: np.ndarray, load_idx: np.ndarray) -> pd.DataFrame:
        data = {}
        for pos, col in enumerate(cat_cols):
            levels = np.asarray(value_lists[pos], dtype=object)
            data[col] = levels[(combo_idx // inner[pos]) % len(levels)]
        chunk = pd.DataFrame(data)
        if n_load > 1:
            # Each categorical combination is repeated over the loading grid,
            # matching v16's stream order (loading varies fastest).
            chunk = chunk.loc[chunk.index.repeat(n_load)].reset_index(drop=True)
        if n_load:
            chunk["catalyst_loading"] = grid[load_idx]
        for col in out_cols:
            if col not in chunk.columns:
                chunk[col] = representative[col]
        return chunk[out_cols]

    if plan["mode"] == "full-product":
        chunk_combos = max(1, chunk_rows // max(n_load, 1))
        for start in range(0, n_cat, chunk_combos):
            stop = min(start + chunk_combos, n_cat)
            combo_idx = np.arange(start, stop, dtype=np.int64)
            load_idx = np.tile(np.arange(n_load, dtype=np.int64), stop - start)
            yield _build(combo_idx, load_idx)
    else:
        # Uniform stride through the flat product index: every level of the
        # fast-varying axes and a spread of the slow-varying ones is covered.
        total = int(plan["rows"])
        combo_idx = np.linspace(0, max(n_cat - 1, 0), total).round().astype(np.int64)
        zeros = np.zeros(total, dtype=np.int64)
        for start in range(0, total, chunk_rows):
            stop = min(start + chunk_rows, total)
            yield _build(combo_idx[start:stop], zeros[start:stop])


def enumerate_virtual_capped(df, models, feat_cols, label_encoders, scaler,
                             max_combos=200_000, allow_virtual=False,
                             chunk_rows=250_000, keep_top=1_000,
                             weights=None, verbose=True):
    """Score at most ``max_combos`` virtual conditions and return the top rows.

    GATED: raises :class:`VirtualEnumerationRefused` unless ``allow_virtual`` is
    True.  The full library (hundreds of millions of rows) is never materialised;
    at most ``chunk_rows`` rows are alive at once.

    Scope note: this wrapper reproduces v16's enumeration/feature/prediction/EI
    path, but not the post-hoc ``audit_condition_scores`` re-score or the
    experimental-support gate (``apply_support_gate``).  Those live in the frozen
    pipeline and are applied by ``scripts/recommend_virtual.py`` when the full
    route is used.

    Returns a DataFrame of the ``keep_top`` rows with the highest expected
    improvement, or an empty DataFrame if nothing was scored.
    """
    if not allow_virtual:
        raise VirtualEnumerationRefused(
            "enumerate_virtual_capped enumerates a virtual library and is gated. "
            "Pass allow_virtual=True (CLI: --allow-virtual) and a bounded max_combos.")

    plan = plan_enumeration(df, feat_cols, max_combos)
    n_cat = plan["n_cat"]
    if n_cat <= 0:
        return pd.DataFrame()

    members = list(models)
    if weights is None:
        weights = np.full(len(members), 1.0 / max(len(members), 1))
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(members):
        raise ValueError("weights length must match the number of models")
    weights = weights / weights.sum()
    wsumsq = float((weights ** 2).sum())

    # EI incumbent: v16 defaults to the 90th percentile of measured yields.
    y = pd.to_numeric(df['yield'], errors='coerce').dropna()
    if P.EI_INCUMBENT_MODE == 'quantile' and len(y):
        incumbent = float(np.percentile(y.to_numpy(), P.EI_INCUMBENT_QUANTILE * 100.0))
    else:
        incumbent = float(y.max()) if len(y) else 0.0

    gap_thr = float(P.ENSEMBLE_MAX_MODEL_GAP)
    keep_top = max(1, int(keep_top))
    buffered = []
    n_scored = n_chunks = 0
    t0 = time.perf_counter()

    for chunk in _iter_plan_chunks(df, plan, chunk_rows):
        n = len(chunk)
        feature = build_virtual_feature_frame(df, feat_cols, label_encoders,
                                              scaler, chunk)
        total = np.zeros(n, dtype=np.float64)
        total_sq = np.zeros(n, dtype=np.float64)
        vmin = np.full(n, np.inf, dtype=np.float64)
        vmax = np.full(n, -np.inf, dtype=np.float64)
        predictions = {}
        for j, (name, model) in enumerate(models.items()):
            p = np.asarray(predict_in_batches(
                model, feature, batch_size=int(P.PREDICT_BATCH_ROWS),
                desc="predict " + name, show_progress=False), dtype=np.float64)
            p = np.clip(p, float(P.PREDICTION_MIN), float(P.PREDICTION_MAX))
            predictions[name] = p
            total += weights[j] * p
            total_sq += weights[j] * p * p
            np.minimum(vmin, p, out=vmin)
            np.maximum(vmax, p, out=vmax)
        mean = total
        std = np.sqrt(np.maximum(total_sq - mean * mean, 0.0)
                      / max(1.0 - wsumsq, 1e-12))
        gap = vmax - vmin
        chunk["Ensemble Predicted Yield (%)"] = mean
        chunk["Prediction Uncertainty (%)"] = std
        chunk["Model Gap (%)"] = gap
        chunk["Model Min (%)"] = vmin
        chunk["Model Max (%)"] = vmax
        chunk["Expected Improvement"] = calculate_expected_improvement(
            mean, std, incumbent, xi=0.01)
        chunk["UCB (%)"] = mean + 1.96 * std
        chunk["Passes Ensemble Gate"] = gap <= gap_thr
        for name in models:
            chunk["Pred_" + name] = predictions[name]

        n_scored += n
        n_chunks += 1
        buffered.append(chunk.nlargest(min(keep_top, n), "Expected Improvement"))
        if len(buffered) > 8:
            pooled = pd.concat(buffered, ignore_index=True)
            buffered = [pooled.nlargest(keep_top, "Expected Improvement")]
        del feature, predictions, total, total_sq, vmin, vmax, mean, std

    if not buffered:
        return pd.DataFrame()
    top = pd.concat(buffered, ignore_index=True)
    top = top.nlargest(min(keep_top, len(top)),
                       "Expected Improvement").reset_index(drop=True)
    top = top.sort_values("Expected Improvement", ascending=False).reset_index(drop=True)
    top.insert(0, "Rank", np.arange(1, len(top) + 1))
    if verbose:
        print("[enumerate] scored %s rows in %d chunk(s) in %.2f s"
              % (f"{n_scored:,}", n_chunks, time.perf_counter() - t0))
    return top

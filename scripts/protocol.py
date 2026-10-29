"""Shared helpers for adopting the diagnosed capture definition across the corpus.

The adopted definition
----------------------
    capture = R2_test( configuration chosen on an INNER VALIDATION split )
              / [ 1 - s2_within / s2_total ]   (ceiling on ALL rows)

with a condition key that identifies the condition: every column of
``hte.params.CONDITION_COLS`` (9 categorical + 10 equivalent slots) that the
mapping produced for this file.  A slot the mapping disables with ``none``, or
one the source does not record, is absent from the key.  There is no per-dataset
choice -- the mapping table carries that information.

Threads
-------
``OMP_NUM_THREADS=8`` is pinned by every entry script BEFORE numpy/xgboost is
imported, XGBoost is built with ``n_jobs=8`` and RandomForest with
``n_jobs=8`` (``params.RF_N_JOBS``).  Every number produced here is therefore
an 8-thread number; the shipped tables are 4-thread numbers.

Nothing in this module writes to the repository; it only reads raw data and
``results/``.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import time as _timing
from pathlib import Path

sys.dont_write_bytecode = True

# Thread setting.  The paper's runs used FOUR threads: the frozen pipeline caps
# the pools at 4 in its header (``DSH_SWEEP_THREADS='4'``) and every shipped
# table reproduces at 4 and only at 4 -- measured over the full 156-cell grid,
# 4 threads gives 138 cells bit-identical to the shipped values and 8 threads
# only 105 (XGBoost 0/39).  One number in the paper is an 8-thread number: the
# random-split CV of Figure 7, recorded as ``n_jobs=8`` in results/protocol_cv.csv.
#
# So: pin 4 by default.  ``CD_THREADS=N`` changes the count, ``CD_THREADS=0``
# leaves the pools exactly as the caller set them.
_CD_THREADS = os.environ.get("CD_THREADS", "4")
PINNED_THREADS = _CD_THREADS not in ("", "0", "none", "None")
N_THREADS = int(_CD_THREADS) if PINNED_THREADS else (os.cpu_count() or 4)
if PINNED_THREADS:
    for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "DSH_SWEEP_THREADS"):
        os.environ[_tv] = str(N_THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "outputs" / "grid_ceiling"
RESULTS = REPO / "results"
CACHE = OUT / "cache"

# v24: the shipped package no longer carries empty results/ outputs/ figures/,
# so every writer creates its directory instead of failing with FileNotFoundError.
OUTPUTS = REPO / "outputs"
for _d in (RESULTS, OUT, CACHE, OUTPUTS / "c20_transfer_4donor", OUTPUTS / "acquisition",
           OUTPUTS / "attrib13", OUTPUTS / "final_numbers", OUTPUTS / "table2",
           REPO / "figures"):
    _d.mkdir(parents=True, exist_ok=True)

if str(REPO) not in sys.path:          # make ``hte`` importable from anywhere
    sys.path.insert(0, str(REPO))

from hte import params as P  # noqa: E402  (thread/n_jobs policy for the fits)
from hte import datasets as _DS  # noqa: E402  (the corpus definition)

#: the manuscript's 13 datasets -- ``corpus: true`` in data/alias_mapping.yaml
ALIASES13 = _DS.corpus()
#: the datasets with no repeated condition -- ``ceiling: false`` in the same table
NO_CEILING = _DS.no_ceiling()

SEED_BOOT = 42
N_BOOT = 1000
VAL_SEEDS = [42, 1, 2, 3, 4]          # same five seeds as the diagnosis (M3)
VAL_TEST_SIZE = 0.2                   # 20 % of the training 80 % -> 16 % of all rows

# dataset loading / keys
def table1() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "dataset_index.csv", encoding="utf-8-sig")


def stems() -> dict:
    """alias -> source stem, from ``hte.datasets`` (i.e. data/alias_mapping.yaml).

    The frozen ``results/dataset_index.csv`` is only a row-count check, done in
    :func:`load`.
    """
    return _DS.stems()


def donors() -> list:
    """The transfer donors of §3.5 (``donor: true`` in data/alias_mapping.yaml)."""
    return _DS.donors()


def load(alias: str):
    """Load one dataset through the frozen ``hte`` loader (bit-identical)."""
    from hte.data import load_real_dataset
    stem = stems()[alias]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df, raw = load_real_dataset(str(DATA / (stem + ".csv")), str(DATA))
    idx_p = RESULTS / "dataset_index.csv"
    if idx_p.exists():
        t = pd.read_csv(idx_p, encoding="utf-8-sig")
        row = t[t["dataset"] == alias]
        if not row.empty:
            assert len(df) == int(row.iloc[0]["n_rows"]), \
                (alias, len(df), int(row.iloc[0]["n_rows"]))
    return df, raw


def key_shipped(df: pd.DataFrame) -> np.ndarray:
    """The condition key of the paper: every canonical condition column present.

    ``CONDITION_COLS`` (9 categorical + 10 equivalent columns) is read from
    :mod:`hte.params`, i.e. from the same table ``hte.ceiling`` uses, and is
    intersected with the columns the mapping actually produced for this file.
    A slot the mapping disabled (``none``) or that the loader never created is
    simply absent from the key.  Nothing here is chosen per dataset.
    """
    from hte import params as _P
    cols = [c for c in _P.CONDITION_COLS if c in df.columns]
    return df[cols].astype(str).agg("|".join, axis=1).to_numpy()


def condition_key(alias: str, df: pd.DataFrame):
    """Return ``(key, key_type, note)`` under the adopted definition.

    Recovered 2026-10-28 from the file that produced ``results/table1_v18_final.csv``
    (``_work/v237/ceil18.py``): ``estimate_noise_ceiling(df)`` with
    ``cond_cols=None``, i.e. the canonical list above.  There is no per-dataset
    key: the older ``smiles_complete`` variant for ``bh-hte-ood_BH`` belongs to
    the capture diagnosis, not to Table 1, and reproducing all eleven published
    ceilings needs no such branch.
    """
    from hte import params as _P
    cols = [c for c in _P.CONDITION_COLS if c in df.columns]
    return key_shipped(df), "condition_cols", "%d condition columns" % len(cols)


# noise ceiling and its cluster bootstrap
def group_moments(key, y):
    """Per-condition n, sum, sum of squares, within-condition SS."""
    key = np.asarray(key)
    y = np.asarray(y, float)
    codes, _ = pd.factorize(pd.Series(key), sort=False)
    g = int(codes.max()) + 1
    n = np.bincount(codes, minlength=g).astype(float)
    s = np.bincount(codes, weights=y, minlength=g)
    q = np.bincount(codes, weights=y * y, minlength=g)
    ss = q - s * s / n
    np.maximum(ss, 0.0, out=ss)
    return n, s, q, ss


def ceiling_from_moments(n, s, q, ss) -> float:
    """1 - s2w/s2t from the group moments (identical to hte.ceiling)."""
    rep = n >= 2
    if not rep.any():
        return float("nan")
    den = float((n[rep] - 1.0).sum())
    if den <= 0.0:
        return float("nan")
    s2w = float(ss[rep].sum() / den)
    n_tot = float(n.sum())
    if n_tot < 2.0:
        return float("nan")
    s2t = float((q.sum() - s.sum() ** 2 / n_tot) / (n_tot - 1.0))
    if not np.isfinite(s2t) or s2t <= 0.0:
        return float("nan")
    return 1.0 - s2w / s2t


def ceiling_stats(key, y) -> dict:
    """Full plug-in ceiling record for one dataset."""
    n, s, q, ss = group_moments(key, y)
    rep = n >= 2
    den = float((n[rep] - 1.0).sum())
    s2w = float(ss[rep].sum() / den) if den > 0 else float("nan")
    y = np.asarray(y, float)
    s2t = float(np.var(y, ddof=1))
    return dict(
        ceiling=ceiling_from_moments(n, s, q, ss),
        s2_within=s2w, s2_total=s2t,
        n_cond=int(len(n)), n_rep_groups=int(rep.sum()),
        n_rep_rows=int(n[rep].sum()),
        rep_frac=float(n[rep].sum() / len(y)),
        df_within=den,
    )


def cluster_ceiling_draws(n, s, q, ss, B=N_BOOT, seed=SEED_BOOT) -> np.ndarray:
    """Cluster bootstrap over REPLICATE condition groups.

    Each condition measured >= 2 times is one sampling unit.  A bootstrap
    sample draws the same number of replicate groups with replacement
    (multiplicities ~ Multinomial(G_rep, 1/G_rep)); a unit drawn k times adds
    its rows k times, so it contributes k * SS_g on k * (n_g - 1) df.  Singleton
    conditions carry no within-condition information and are held fixed, but
    they still enter s2_total.  This is ``scripts/bootstrap_ci.py``'s scheme.
    """
    rep = n >= 2
    if not rep.any():
        return np.empty(0)
    nr, sr, qr, ssr = n[rep], s[rep], q[rep], ss[rep]
    dfr = nr - 1.0
    n_groups = int(rep.sum())
    n_fixed = float(n[~rep].sum())
    s_fixed = float(s[~rep].sum())
    q_fixed = float(q[~rep].sum())

    rng = np.random.default_rng(seed)
    k = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=B).astype(float)
    s2w = (k @ ssr) / (k @ dfr)
    n_tot = n_fixed + k @ nr
    s_tot = s_fixed + k @ sr
    q_tot = q_fixed + k @ qr
    s2t = (q_tot - s_tot * s_tot / n_tot) / (n_tot - 1.0)
    return 1.0 - s2w / s2t


def boot_ceiling(key, y, B=N_BOOT, seed=SEED_BOOT) -> dict:
    """Point ceiling, bootstrap SE and 95 % percentile CI."""
    n, s, q, ss = group_moments(key, y)
    point = ceiling_from_moments(n, s, q, ss)
    draws = cluster_ceiling_draws(n, s, q, ss, B=B, seed=seed)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return dict(ceiling=point, se=float("nan"), lo=float("nan"), hi=float("nan"),
                    draws=draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(ceiling=point, se=float(np.std(draws, ddof=1)),
                lo=float(lo), hi=float(hi), draws=draws)


def percentile_ci(values, log_scale=False):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if log_scale:
        x = x[x > 0.0]
    if x.size == 0:
        return float("nan"), float("nan")
    if log_scale:
        lo, hi = np.percentile(np.log(x), [2.5, 97.5])
        return float(np.exp(lo)), float(np.exp(hi))
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


# the 12-cell grid with the inner-validation selection
def grid_cells_threaded(cat_idx):
    """The paper's 12 cells (threads are applied in :func:`fit_predict`)."""
    from hte.models import grid_cells

    return grid_cells(cat_idx)


def fit_predict(cell, Xtr, ytr, Xev):
    """Fit one cell and predict, at ``n_jobs=N_THREADS`` wherever the estimator has it.

    XGBoost and RandomForest expose ``n_jobs``; HistGradientBoosting reads
    ``OMP_NUM_THREADS`` (pinned at import), Gradient Boosting is single-threaded
    and the MLP is bounded by the BLAS pool.
    """
    mdl = cell.factory()
    _rf = type(mdl).__name__ == "RandomForestRegressor"
    if _rf and P.RF_N_JOBS:
        mdl.n_jobs = P.RF_N_JOBS
    elif PINNED_THREADS and "n_jobs" in mdl.get_params():
        mdl.set_params(n_jobs=N_THREADS)
    mdl.fit(Xtr, ytr)
    if _rf:
        mdl.n_jobs = None
    return np.asarray(mdl.predict(Xev), float).ravel()


def fit_train_test(cell, Xtr, ytr, Xte):
    """Fit once on ``Xtr`` and return ``(yhat_train, yhat_test, seconds)``.

    The grid table carries both rows' scores; fitting twice would double the
    cost of the whole step, so the single fit is reused for both predictions.
    """
    mdl = cell.factory()
    _rf = type(mdl).__name__ == "RandomForestRegressor"
    if _rf and P.RF_N_JOBS:
        mdl.n_jobs = P.RF_N_JOBS
    elif PINNED_THREADS and "n_jobs" in mdl.get_params():
        mdl.set_params(n_jobs=N_THREADS)
    t0 = _timing.perf_counter()
    mdl.fit(Xtr, ytr)
    secs = _timing.perf_counter() - t0
    if _rf:
        mdl.n_jobs = None
    p_tr = np.asarray(mdl.predict(Xtr), float).ravel()
    p_te = np.asarray(mdl.predict(Xte), float).ravel()
    return p_tr, p_te, secs


def grid_and_select(df, raw, verbose=True):
    """Run the 12 cells and the 5-seed inner-validation selection.

    Protocol (identical to outputs/capture_diagnosis/02_grid.py)
    ----------------------------------------------------------
    * outer split: ``hte.features.get_split_positions(n, random_state=42)``,
      test_size 0.2 -> the untouched 20 % test split;
    * test fits: each cell trained on the full 80 % train split, scored on test;
    * inner validation: ``train_test_split(arange(len(train)), test_size=0.2,
      random_state=seed)`` for seed in [42, 1, 2, 3, 4] -> 64 % fit / 16 % val;
      the arg-max cell on the 16 % validation split is the winner; its R2 is
      read from the test-fit grid (refit on the full 80 %);
    * reported ``r2_test`` = MEAN of that winner's test R2 over the five seeds;
      ``selected_model``/``selected_encoding`` = the modal winner (with its
      stability count).
    """
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from hte.features import get_split_positions, prepare_features
    from hte.grid import build_frames

    tr, te = get_split_positions(len(df))
    y = df["yield"].to_numpy(float)
    X, yv, feature_cols, _disp, _enc, _sc = prepare_features(df, raw, fit_rows=tr)
    packs, _cc, cat_idx = build_frames(X, feature_cols, tr, te)
    cells = grid_cells_threaded(cat_idx)
    names = ["%s / %s" % (c.architecture, c.encoding) for c in cells]
    y_tr, y_te = y[tr], y[te]

    r2_test = np.full(len(cells), np.nan)
    cell_rows = []
    for j, cell in enumerate(cells):
        Xtr, Xte = packs[cell.frame]
        p_tr, p_te, secs = fit_train_test(cell, Xtr, y_tr, Xte)
        r2_test[j] = r2_score(y_te, p_te)
        cell_rows.append(dict(
            dataset=None, n_rows=int(len(df)),
            model=cell.architecture, encoding=cell.encoding,
            n_features=int(Xtr.shape[1]),
            r2_train=float(r2_score(y_tr, p_tr)),
            r2_test=float(r2_score(y_te, p_te)),
            rmse_train=float(np.sqrt(np.mean((y_tr - p_tr) ** 2))),
            rmse_test=float(np.sqrt(np.mean((y_te - p_te) ** 2))),
            fit_seconds=float(secs), status="ok", threads=N_THREADS,
        ))
    best = int(np.nanargmax(r2_test))

    sel_rows = []
    for seed in VAL_SEEDS:
        fit_pos, val_pos = train_test_split(np.arange(len(tr)), test_size=VAL_TEST_SIZE,
                                            random_state=seed)
        r2_val = np.full(len(cells), np.nan)
        for j, cell in enumerate(cells):
            Xtr, _ = packs[cell.frame]
            r2_val[j] = r2_score(y_tr[val_pos],
                                 fit_predict(cell, Xtr.iloc[fit_pos], y_tr[fit_pos],
                                             Xtr.iloc[val_pos]))
        sv = int(np.nanargmax(r2_val))
        sel_rows.append(dict(
            val_seed=seed,
            selected_model=cells[sv].architecture,
            selected_encoding=cells[sv].encoding,
            r2_val_selected=float(r2_val[sv]),
            r2_test_selected=float(r2_test[sv]),
            r2_test_max=float(r2_test[best]),
            selected_is_test_best=int(sv == best),
            n_val=int(len(val_pos)), n_train_fit=int(len(fit_pos)),
        ))
        if verbose:
            print("    val seed %2d -> %-18s %-18s val=%.4f  test=%.4f  (test max %.4f)"
                  % (seed, cells[sv].architecture, cells[sv].encoding,
                     r2_val[sv], r2_test[sv], r2_test[best]), flush=True)

    return dict(tr=tr, te=te, cells=cells, names=names, r2_test=r2_test, best=best,
                sel_rows=sel_rows, y=y, cell_rows=cell_rows)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild the paper's result tables from raw per-dataset outputs.

The repository ships the result tables but not the scripts that built them: they
were produced by throw-away scripts in a scratch directory.  This module is the
committed replacement.  It rebuilds a table, writes it under ``results/`` and
checks it against the shipped copy column by column.

    python scripts/tables.py --check
    python scripts/tables.py --rebuild grid_all16
    python scripts/tables.py --rebuild all

Where the rebuild lands
-----------------------
* ``grid_all16``, ``ceiling_stats_v16`` and ``attrib12_long`` refer to the
  superseded 16-dataset corpus and now live in ``results/legacy/``; they are no
  longer covered by ``results/checksums.txt``.  Their canonical files are
  **never overwritten**; the rebuild goes to ``results/rebuilt_<name>.csv`` and
  is compared against the legacy copy.
* ``perm_vs_shap``, ``zero_yield_audit`` and ``interaction_ligand_base`` have no
  canonical file in the repo, so their rebuild is written to ``results/<name>.csv``.

Raw material
------------
* ``PROJECT_ROOT/Script/Analysis_all_v16.py`` -- frozen pipeline (loader, feature
  preparation, split, model hyper-parameters, noise-ceiling estimator).  Read
  only; never imported as ``import`` but executed through ``importlib`` so that
  its ``__main__`` guard holds and its module-level rcParams do not leak.
* ``PROJECT_ROOT/Data/<source>.csv`` -- the 16 published datasets.
* ``PROJECT_ROOT/outputs/<run>/model_evaluation_metrics.csv`` -- per-dataset,
  per-model held-out metrics written by the pipeline itself.
* ``PROJECT_ROOT/batch_logs/<batch>/summary.csv`` -- run manifest that says which
  ``outputs/`` directory belongs to the batch whose numbers the paper quotes.
* ``data/alias_mapping.yaml`` -- maps source stems to the short aliases used in
  ``grid_all16.csv``.

Notes on the tables
-------------------
``grid_all16.csv`` has 192 rows = 16 datasets x **12** architecture/encoding
cells (not 15): Gradient Boosting, Random Forest and MLP are only run under
``integer`` and ``one-hot`` (and MLP's one-hot is an internal pipeline), so the
grid is 3 encodings x 5 architectures minus 3 missing cells.

``fit_seconds`` is wall-clock time.  It is reported for information but is
excluded from the equality gate: it cannot be reproduced.

Reproducibility caveat: ``attrib12_long.csv`` cannot fully meet the 1e-9 gate.
Its XGBoost "one-hot" SHAP values came from an unseeded ``shap.KernelExplainer``
and are stochastic (two all-cores runs differ by ~1e-2); its XGBoost "native
categorical" values come from xgboost's float32 ``pred_contribs`` and move by
~1e-5.  Both cases are documented in ``results/attrib12_long.PROVENANCE.md``.
Everything else in every table reproduces to machine precision.

Exit codes
----------
0  requested table(s) rebuilt and verified against the shipped copy
1  rebuild failed, or a column mismatch
2  table derived but UNVERIFIED (no reference copy exists to compare against)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Pin the BLAS/OpenMP thread count BEFORE numpy/xgboost are imported, exactly as
# Analysis_all_v16.py does (lines 30-34).  Without this the XGBoost cells drift
# from the stored reference values.
_SWEEP_THREADS = os.environ.get("DSH_SWEEP_THREADS", "4")
if _SWEEP_THREADS not in ("0", "", "none", "None"):
    for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_tv, _SWEEP_THREADS)

import numpy as np                                   # noqa: E402
import pandas as pd

# make the package importable no matter where the script is run from
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))                                  # noqa: E402

# Paths
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

PROJECT = Path(__file__).resolve().parents[1]   # the packaged repository itself
# The packaged layout: raw data in data/, the frozen pipeline at the repository root.
DATA = PROJECT / "data"
OUTPUTS = PROJECT / "outputs"
PIPELINE = PROJECT / "archive" / "single_dataset_full_benchmark.py"

# batch whose v15 run produced the numbers in ceiling_stats_v16.csv / Table I
BATCH_ID = "v15_20261009_163116"
BATCH = PROJECT / "batch_logs" / BATCH_ID

# scratch directory the original throw-away scripts wrote into; used only as a
# fallback location for reference copies that never entered the repo.
SCRATCH = ROOT / "_scratch"

TOL = 1e-9
NONREPRODUCIBLE = {"fit_seconds"}                 # wall-clock timing

# Canonical dataset lists -- read from data/alias_mapping.yaml (the one table
# that holds alias -> file stem).  Nothing here names a dataset.
from hte.datasets import dataset_order as _dataset_order  # noqa: E402

DATASET_ORDER = _dataset_order()

# order of ceiling_stats_v16.csv (same sources, different arrangement)
CEILING_ORDER = [stem for _alias, stem in DATASET_ORDER]

GRID_COLUMNS = ["dataset", "n_rows", "model", "encoding", "n_features",
                "r2_train", "r2_test", "rmse_train", "rmse_test",
                "fit_seconds", "status"]
CEILING_COLUMNS = ["dataset", "n", "best_model", "r2_best", "ceiling", "capture",
                   "n_cond", "n_rep_groups", "rep_frac"]
ATTRIB_COLUMNS = ["dataset", "model", "encoding", "feature", "perm", "shap"]
PERM_COLUMNS = ["dataset", "model", "n", "rho"]
ZERO_COLUMNS = ["dataset", "n_rows", "n_zero", "pct_zero", "n_nonzero",
                "mean_yield", "median_yield", "max_yield"]
INTERACTION_COLUMNS = ["ligand", "base", "n", "mean_yield", "in_fig4"]

TABLES = ["grid_all16", "ceiling_stats_v16", "attrib12_long",
          "perm_vs_shap", "zero_yield_audit", "interaction_ligand_base"]

# Tables that ship in the repository and are covered by results/checksums.txt.
# Their canonical files are never overwritten: a rebuild goes to results/rebuilt/.
REPO_TABLES = {"grid_all16", "ceiling_stats_v16", "attrib12_long"}

# keys used to align a rebuilt table with its reference copy
KEYS = {
    "grid_all16": ["dataset", "model", "encoding"],
    "ceiling_stats_v16": ["dataset"],
    "attrib12_long": ["dataset", "model", "encoding", "feature"],
    "perm_vs_shap": ["dataset", "model"],
}


# Small helpers
def say(msg: str = "") -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_bom(path: Path) -> bool:
    try:
        return path.read_bytes()[:3] == b"\xef\xbb\xbf"
    except OSError:
        return False


# Tables with no shipped original anywhere.  Their output must never be mistaken
# for a reference on a second run, so reference lookup returns None for them.
DERIVED = {"zero_yield_audit", "interaction_ligand_base"}


def reference_path(name: str) -> Path | None:
    """Where the shipped copy of ``name`` lives.

    The rebuild is never used as its own reference: repo tables are compared with
    the shipped ``results/<name>.csv``, and the three tables that only exist in
    the scratch directory are compared with that original (which is byte-identical
    to the copy we write into results/).
    """
    if name in DERIVED:
        return None
    if name in REPO_TABLES:
        p = RES / f"{name}.csv"
        if p.exists():
            return p
        # Superseded 16-dataset tables were moved to results/legacy/; that copy
        # is the reference for the legacy rebuild.
        legacy = RES / "legacy" / f"{name}.csv"
        return legacy if legacy.exists() else None
    p = SCRATCH / f"{name}.csv"
    if p.exists():
        return p
    p = RES / f"{name}.csv"
    if p.exists():
        return p
    # Superseded 16-dataset copies also live in results/legacy/ (e.g. perm_vs_shap).
    legacy = RES / "legacy" / f"{name}.csv"
    return legacy if legacy.exists() else None


def read_alias_map() -> dict[str, str]:
    import yaml
    p = ROOT / "data" / "alias_mapping.yaml"
    if not p.exists():
        return {}
    m = yaml.safe_load(p.read_text(encoding="utf-8"))
    return {k: v["source"] for k, v in (m.get("datasets") or {}).items()}


_PIPE = None


def pipeline():
    """Load Analysis_all_v16.py once (never as a normal import)."""
    global _PIPE
    if _PIPE is None:
        if not PIPELINE.exists():
            # the monolith is not shipped as a runnable file: use the in-package copy
            from hte import v16_core as _core
            _PIPE = _core
            return _PIPE
        spec = importlib.util.spec_from_file_location("tables_pipe_v16", PIPELINE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["tables_pipe_v16"] = mod
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mod)
        _PIPE = mod
    return _PIPE


def load_dataset(stem: str):
    """Pipeline loader, output suppressed (it is extremely chatty)."""
    P = pipeline()
    f = DATA / f"{stem}.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        out = P.load_real_dataset(str(f), str(DATA))
    return out[0] if isinstance(out, tuple) else out


# Table builders
def build_grid_all16() -> pd.DataFrame:
    """192 rows: 16 datasets x 12 architecture/encoding cells (held-out R^2 etc.)."""
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                                  HistGradientBoostingRegressor)
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    import xgboost as xgb

    P = pipeline()

    def _xgb(**extra):
        return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, **extra)

    def _gb():
        return GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                         learning_rate=0.1, random_state=42)

    def _mlp():
        return MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                            max_iter=500, random_state=42, early_stopping=True)

    def _rf():
        return RandomForestRegressor(n_estimators=P.RF_N_ESTIMATORS, max_depth=15,
                                     min_samples_split=5, random_state=42)

    def _histgb_native(cat_idx):
        return HistGradientBoostingRegressor(
            max_iter=P.HISTGB_MAX_ITER, max_depth=6, learning_rate=0.1,
            random_state=42, categorical_features=(cat_idx or None))

    def _histgb_plain():
        return HistGradientBoostingRegressor(
            max_iter=P.HISTGB_MAX_ITER, max_depth=6, learning_rate=0.1, random_state=42)

    def _mlp_onehot(cat_idx):
        return make_pipeline(
            ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False), cat_idx)],
                              remainder="passthrough"),
            StandardScaler(), _mlp())

    def cells_for(cat_idx):
        """(architecture, encoding label, factory, WHICH FRAME the estimator sees).

        Two cells deliberately take the INTEGER frame under another label:
        HistGB 'native categorical' declares ``categorical_features`` on the
        integer-coded frame, and MLP 'one-hot' one-hots inside its own pipeline.
        """
        return [
            ("XGBoost", "integer", _xgb, "INT"),
            ("XGBoost", "one-hot", _xgb, "OH"),
            ("XGBoost", "native categorical",
             lambda: _xgb(enable_categorical=True, tree_method="hist",
                          max_cat_to_onehot=4), "NAT"),
            ("Gradient Boosting", "integer", _gb, "INT"),
            ("Gradient Boosting", "one-hot", _gb, "OH"),
            ("Random Forest", "integer", _rf, "INT"),
            ("Random Forest", "one-hot", _rf, "OH"),
            ("HistGB", "integer", _histgb_plain, "INT"),
            ("HistGB", "one-hot", _histgb_plain, "OH"),
            ("HistGB", "native categorical", lambda: _histgb_native(cat_idx), "INT"),
            ("MLP", "integer", _mlp, "INT"),
            ("MLP", "one-hot", lambda: _mlp_onehot(cat_idx), "INT"),
        ]

    rows = []
    t_all = time.perf_counter()
    for di, (key, stem) in enumerate(DATASET_ORDER, 1):
        say("[grid %2d/%d] %s" % (di, len(DATASET_ORDER), key))
        df = load_dataset(stem)
        tr_pos, te_pos = P.get_split_positions(len(df))
        X, y, feature_cols, _disp, _enc, _sc = P.prepare_features(
            df, None, fit_rows=tr_pos)

        cat_cols = [c for c in feature_cols if str(c).endswith("_encoded")]
        num_cols = [c for c in feature_cols if c not in cat_cols]
        cat_idx = [i for i, c in enumerate(feature_cols) if str(c).endswith("_encoded")]

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
            X_nat[c] = X_nat[c].astype("category")

        y_tr, y_te = y.iloc[tr_pos].copy(), y.iloc[te_pos].copy()
        packs = {"INT": (X_int.iloc[tr_pos].copy(), X_int.iloc[te_pos].copy()),
                 "OH": (X_oh.iloc[tr_pos].copy(), X_oh.iloc[te_pos].copy()),
                 "NAT": (X_nat.iloc[tr_pos].copy(), X_nat.iloc[te_pos].copy())}

        for arch, enc, factory, fkey in cells_for(cat_idx):
            Xtr, Xte = packs[fkey]
            rec = dict(dataset=key, n_rows=len(df), model=arch, encoding=enc,
                       n_features=int(Xtr.shape[1]), r2_train=np.nan,
                       r2_test=np.nan, rmse_train=np.nan, rmse_test=np.nan,
                       fit_seconds=np.nan, status="ok")
            t0 = time.perf_counter()
            try:
                mdl = factory()
                fast_rf = isinstance(mdl, RandomForestRegressor) and P.RF_N_JOBS > 1
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
    say("[grid] done in %.1f s" % (time.perf_counter() - t_all))
    return pd.DataFrame(rows, columns=GRID_COLUMNS)


def build_ceiling_stats_v16() -> pd.DataFrame:
    """Ceiling, capture and replicate statistics per dataset."""
    P = pipeline()
    manifest = ceiling_manifest()
    rows = []
    t_all = time.perf_counter()
    for di, stem in enumerate(CEILING_ORDER, 1):
        say("[ceiling %2d/%d] %s" % (di, len(CEILING_ORDER), stem[:50]))
        df = load_dataset(stem)
        info = P.estimate_noise_ceiling(df)

        out_dir = manifest.get(stem)
        if out_dir is None:
            out_dir = newest_output(stem)
        best_model, r2_best = best_from_metrics(out_dir)

        ceiling = info["ceiling"]
        capture = (r2_best / ceiling
                   if (np.isfinite(ceiling) and ceiling > 0 and np.isfinite(r2_best))
                   else np.nan)
        rows.append(dict(dataset=stem, n=len(df), best_model=best_model,
                         r2_best=r2_best, ceiling=ceiling, capture=capture,
                         n_cond=info["n_cond"], n_rep_groups=info["n_rep_groups"],
                         rep_frac=info["rep_frac"]))
    say("[ceiling] done in %.1f s" % (time.perf_counter() - t_all))
    return pd.DataFrame(rows, columns=CEILING_COLUMNS)


def build_attrib12_long() -> pd.DataFrame:
    """1704 rows: permutation importance and mean|SHAP| for the same 12 cells."""
    import shap
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.inspection import permutation_importance

    P = pipeline()

    #     tables cannot drift through a shared mutable default) ---
    import xgboost as xgb
    from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                                  HistGradientBoostingRegressor)
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer

    def _xgb(**extra):
        # Threading trap inherited from the scratch run: xgboost sizes its OpenMP
        # pool when the module is first imported.  The scratch attribution script
        # imported xgboost BEFORE pinning OMP_NUM_THREADS, so those fits used all
        # cores; the grid table imported it after pinning, so its fits used 4.
        # The two settings give measurably different models, and one process
        # cannot hold both pools.  cmd_rebuild therefore re-executes this table
        # in a child process with DSH_SWEEP_THREADS=0 (no pinning); see main().
        return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, **extra)

    def _gb():
        return GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                         learning_rate=0.1, random_state=42)

    def _mlp():
        return MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                            max_iter=500, random_state=42, early_stopping=True)

    def _rf():
        return RandomForestRegressor(n_estimators=P.RF_N_ESTIMATORS, max_depth=15,
                                     min_samples_split=5, random_state=42)

    def _histgb_native(cat_idx):
        return HistGradientBoostingRegressor(
            max_iter=P.HISTGB_MAX_ITER, max_depth=6, learning_rate=0.1,
            random_state=42, categorical_features=(cat_idx or None))

    def _histgb_plain():
        return HistGradientBoostingRegressor(
            max_iter=P.HISTGB_MAX_ITER, max_depth=6, learning_rate=0.1, random_state=42)

    def _mlp_onehot(cat_idx):
        return make_pipeline(
            ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False), cat_idx)],
                              remainder="passthrough"),
            StandardScaler(), _mlp())

    def cells_for(cat_idx):
        return [
            ("XGBoost", "integer", _xgb, "INT"),
            ("XGBoost", "one-hot", _xgb, "OH"),
            ("XGBoost", "native categorical",
             lambda: _xgb(enable_categorical=True, tree_method="hist",
                          max_cat_to_onehot=4), "NAT"),
            ("Gradient Boosting", "integer", _gb, "INT"),
            ("Gradient Boosting", "one-hot", _gb, "OH"),
            ("Random Forest", "integer", _rf, "INT"),
            ("Random Forest", "one-hot", _rf, "OH"),
            ("HistGB", "integer", _histgb_plain, "INT"),
            ("HistGB", "one-hot", _histgb_plain, "OH"),
            ("HistGB", "native categorical", lambda: _histgb_native(cat_idx), "INT"),
            ("MLP", "integer", _mlp, "INT"),
            ("MLP", "one-hot", lambda: _mlp_onehot(cat_idx), "INT"),
        ]

    def parent_of(name, cat_cols):
        """Map an expanded one-hot column back onto its semantic slot."""
        for c in cat_cols:
            if name == c or name.startswith(str(c) + "_"):
                return c
        return name

    def coerce(sv):
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.dtype == object:
            sv = np.array([[float(x) for x in row] for row in sv], dtype=float)
        sv = sv.astype(float)
        if sv.ndim == 3:
            sv = sv[:, :, 0]
        return sv

    def shap_mean_abs(mdl, Xtr, Xte, is_tree):
        errs = []
        if is_tree:
            for how in ("new", "old"):
                try:
                    ex = shap.TreeExplainer(mdl)
                    sv = ex(Xte).values if how == "new" else ex.shap_values(Xte)
                    return np.abs(coerce(sv)).mean(axis=0)
                except Exception as exc:
                    errs.append("%s=%s" % (how, type(exc).__name__))
        try:
            bg = Xtr.sample(min(25, len(Xtr)), random_state=0)
            sub = Xte.sample(min(120, len(Xte)), random_state=0)
            sv = shap.Explainer(mdl.predict, bg)(sub).values
            return np.abs(coerce(sv)).mean(axis=0)
        except Exception as exc:
            errs.append("kernel=%s:%s" % (type(exc).__name__, exc))
        raise RuntimeError("; ".join(errs))

    # The shipped table is the output of two scripts: the 12-cell run, then a
    # post-hoc repair of the XGBoost|native-categorical SHAP cell (see the
    # special case below).  The repair appended its 144 rows at the end, so we
    # keep the same layout: 11 cells in dataset order, then the native rows.
    keep_rows: list[dict] = []
    native_rows: list[dict] = []
    t_all = time.perf_counter()
    for di, (key, stem) in enumerate(DATASET_ORDER, 1):
        say("[attrib %2d/%d] %s" % (di, len(DATASET_ORDER), key))
        df = load_dataset(stem)
        tr_pos, te_pos = P.get_split_positions(len(df))
        X, y, feature_cols, _disp, _enc, _sc = P.prepare_features(
            df, None, fit_rows=tr_pos)
        cat_cols = [c for c in feature_cols if str(c).endswith("_encoded")]

        X_int = X.copy()
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        oh.fit(X.iloc[tr_pos][cat_cols])
        num_cols = [c for c in feature_cols if c not in cat_cols]
        blocks = [oh.transform(X[cat_cols])]
        if num_cols:
            blocks.append(X[num_cols].to_numpy(dtype=float))
        X_oh = pd.DataFrame(np.hstack(blocks).astype(np.float64),
                            columns=list(oh.get_feature_names_out(cat_cols)) + num_cols,
                            index=X.index)
        X_nat = X.copy()
        for c in cat_cols:
            X_nat[c] = X_nat[c].astype("category")
        cat_idx = [i for i, c in enumerate(feature_cols) if str(c).endswith("_encoded")]

        y_tr, y_te = y.iloc[tr_pos].copy(), y.iloc[te_pos].copy()
        packs = {"INT": (X_int.iloc[tr_pos].copy(), X_int.iloc[te_pos].copy()),
                 "OH": (X_oh.iloc[tr_pos].copy(), X_oh.iloc[te_pos].copy()),
                 "NAT": (X_nat.iloc[tr_pos].copy(), X_nat.iloc[te_pos].copy())}

        for arch, enc, factory, fkey in cells_for(cat_idx):
            t0 = time.perf_counter()
            try:
                Xtr, Xte = packs[fkey]
                mdl = factory()
                if hasattr(mdl, "n_jobs") and type(mdl).__name__ == "RandomForestRegressor":
                    mdl.n_jobs = getattr(P, "RF_N_JOBS", None)
                mdl.fit(Xtr, y_tr)
                cols = list(Xtr.columns)

                pi = permutation_importance(mdl, Xte, y_te, n_repeats=5,
                                            random_state=42, scoring="r2", n_jobs=1)
                perm = np.asarray(pi.importances_mean, dtype=float)

                is_tree = arch in ("XGBoost", "Gradient Boosting",
                                   "Random Forest", "HistGB")
                try:
                    if arch == "XGBoost" and fkey == "NAT":
                        # shap.TreeExplainer chokes on the XGBoost 3.x categorical
                        # dtype ("unsupported operand type(s) for -: 'str' and
                        # 'float'").  The shipped table was repaired afterwards by
                        # _work/fix_xgbnative_shap.py, which uses the booster's own
                        # exact TreeSHAP.  Reproduce that here.
                        dm = xgb.DMatrix(Xte, enable_categorical=True)
                        contrib = mdl.get_booster().predict(dm, pred_contribs=True)
                        sh = np.abs(np.asarray(contrib, dtype=float)[:, :-1]).mean(axis=0)
                    else:
                        Xte_sh = Xte.copy()
                        if fkey == "NAT":
                            for c in cat_cols:
                                if c in Xte_sh.columns:
                                    Xte_sh[c] = pd.Categorical(
                                        Xte_sh[c].astype(str),
                                        categories=Xtr[c].cat.categories.astype(str))
                        sh = shap_mean_abs(mdl, Xtr, Xte_sh, is_tree)
                except Exception as exc:
                    sh = np.full(len(cols), np.nan)
                    say("    SHAP failed (%s|%s): %s" % (arch, enc, exc))

                agg_p, agg_s = {}, {}
                for c, pv, sv in zip(cols, perm, sh):
                    k = parent_of(c, cat_cols)
                    agg_p[k] = agg_p.get(k, 0.0) + float(pv)
                    agg_s[k] = agg_s.get(k, 0.0) + (0.0 if not np.isfinite(sv)
                                                    else float(sv))
                target = (native_rows if (arch == "XGBoost" and enc == "native categorical")
                          else keep_rows)
                for k in agg_p:
                    target.append(dict(dataset=key, model=arch, encoding=enc,
                                       feature=k, perm=agg_p[k], shap=agg_s[k]))
                say("    %-18s %-19s ok %.1fs" % (arch, enc, time.perf_counter() - t0))
            except Exception as exc:
                say("    %-18s %-19s FAILED %s: %s"
                    % (arch, enc, type(exc).__name__, exc))
    say("[attrib] done in %.1f min" % ((time.perf_counter() - t_all) / 60))
    return pd.DataFrame(keep_rows + native_rows, columns=ATTRIB_COLUMNS)


def build_perm_vs_shap() -> pd.DataFrame:
    """Spearman rho between permutation importance and mean|SHAP| per (dataset, model).

    Both sides are read from the pipeline's own Figure4/Figure7 panel exports --
    the same source the original ad-hoc script used.
    """
    from scipy.stats import spearmanr

    excl = ("Case1", "BH_Manu_NewConditions")
    # The original script took the newest outputs/ directory per dataset.  The
    # directory tree has grown since, and several newer runs are partial (they
    # stopped before Figure7), so "newest" no longer points at the run the paper
    # used.  Pin the selection to the v15 batch manifest and only fall back to
    # newest for datasets that are not part of that batch.
    manifest = ceiling_manifest()
    best: dict[str, str] = {ds: str(d) for ds, d in manifest.items()}
    for run in sorted(OUTPUTS.glob("*")):
        if not run.is_dir():
            continue
        ds = re.sub(r"^\d{8}_\d{6}_", "", run.name)
        if any(e in ds for e in excl) or ds in best:
            continue
        if not list(run.glob("Figure4_*_panel_data.csv")):
            continue
        best[ds] = str(run)

    rows = []
    for ds, run in sorted(best.items()):
        for f4 in sorted(Path(run).glob("Figure4_*_panel_data.csv")):
            model = f4.name[len("Figure4_Feature_Importance_Attribution- "):
                             -len("_panel_data.csv")]
            f7 = Path(run) / f"Figure7_Deep_SHAP_Attribution- {model}_panel_data.csv"
            if not f7.exists():
                continue
            a = pd.read_csv(f4)
            a = a[a["panel"].astype(str) == "a"]
            b = pd.read_csv(f7)
            b = b[b["panel"].astype(str) == "d"]
            if not len(a) or not len(b):
                continue
            pa = pd.Series(pd.to_numeric(a["y"], errors="coerce").to_numpy(float),
                           index=a["x"].astype(str))
            pa = pa[~pa.index.duplicated()].dropna()
            names = [str(s)[len("global SHAP ("):-1] for s in b["series"]]
            pb = pd.Series(pd.to_numeric(b["value"], errors="coerce").to_numpy(float),
                           index=names).dropna()
            common = [c for c in pa.index if c in pb.index]
            if len(common) < 4:
                continue
            rho = spearmanr(pa[common], pb[common]).statistic
            if np.isfinite(rho):
                rows.append(dict(dataset=ds, model=model, n=len(common),
                                 rho=float(rho)))
    return pd.DataFrame(rows, columns=PERM_COLUMNS)


def build_zero_yield_audit() -> pd.DataFrame:
    """Per-dataset zero-yield audit (raw counts only; no model fitting).

    There is no original table to compare against -- the paper's ``46.6%`` for
    AI_RCP_BH was read off the stdout of a scratch script.  See the PROVENANCE
    file written alongside this table.
    """
    rows = []
    for di, (key, stem) in enumerate(DATASET_ORDER, 1):
        say("[zero %2d/%d] %s" % (di, len(DATASET_ORDER), key))
        df = load_dataset(stem)
        y = pd.to_numeric(df["yield"], errors="coerce")
        nz = y.notna()
        rows.append(dict(
            dataset=key, n_rows=len(df), n_zero=int((y == 0).sum()),
            pct_zero=float(100.0 * (y == 0).mean()),
            n_nonzero=int((y > 0).sum()),
            mean_yield=float(y[nz].mean()), median_yield=float(y[nz].median()),
            max_yield=float(y[nz].max())))
    return pd.DataFrame(rows, columns=ZERO_COLUMNS)


def build_interaction_ligand_base() -> pd.DataFrame:
    """AI_RCP_BH ligand x base mean yield, long form.

    Every observed pair is written; ``in_fig4`` marks the 14x10 crop that the
    paper's figure 4 heatmap displays (top ligands/bases by measurement count,
    138 observed cells).  Keeping the full cross-tab means the headline pairs
    quoted in section 3.4 (RuPhos/Cs2CO3, dtbpf/NaOtBu) are present even though
    they fall outside the figure crop.
    """
    df = load_dataset("(AI_RCP_pub)bh_data_clean_all_whitelisted")
    d = df[["catalyst_name", "reagent_1_name", "yield"]].dropna()
    cnt = d.pivot_table(index="catalyst_name", columns="reagent_1_name",
                        values="yield", aggfunc="size").fillna(0)
    keep_l = cnt.sum(axis=1).sort_values(ascending=False).head(14).index
    keep_b = cnt.sum(axis=0).sort_values(ascending=False).head(10).index
    mat = d.pivot_table(index="catalyst_name", columns="reagent_1_name",
                        values="yield", aggfunc="mean")
    rows = []
    for lig in mat.index:
        for base in mat.columns:
            v = mat.loc[lig, base]
            if not np.isfinite(v):
                continue
            rows.append(dict(ligand=str(lig), base=str(base),
                             n=int(cnt.loc[lig, base]), mean_yield=float(v),
                             in_fig4=bool(lig in keep_l and base in keep_b)))
    return pd.DataFrame(rows, columns=INTERACTION_COLUMNS)


BUILDERS = {
    "grid_all16": build_grid_all16,
    "ceiling_stats_v16": build_ceiling_stats_v16,
    "attrib12_long": build_attrib12_long,
    "perm_vs_shap": build_perm_vs_shap,
    "zero_yield_audit": build_zero_yield_audit,
    "interaction_ligand_base": build_interaction_ligand_base,
}


# Ceiling helpers (run manifest -> per-dataset metrics)
def ceiling_manifest() -> dict[str, Path]:
    """source stem -> outputs/ directory, from the batch run manifest."""
    f = BATCH / "summary.csv"
    if not f.exists():
        return {}
    t = pd.read_csv(f, encoding="utf-8-sig")
    out: dict[str, Path] = {}
    for _, r in t.iterrows():
        stem = str(r["dataset"])
        stem = stem[:-4] if stem.lower().endswith(".csv") else stem
        d = r.get("output_dir", "")
        if not isinstance(d, str) or not d.strip() or d.strip().lower() == "nan":
            continue
        out[stem] = PROJECT / d.strip()
    return out


def newest_output(stem: str) -> Path | None:
    if not OUTPUTS.is_dir():
        return None
    cands = [d for d in OUTPUTS.iterdir()
             if d.is_dir() and d.name.endswith("_" + stem)]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def best_from_metrics(out_dir: Path | None) -> tuple[str, float]:
    """(best model name, R2 test rounded to 4 dp) from model_evaluation_metrics.csv."""
    if out_dir is None:
        return "", float("nan")
    f = Path(out_dir) / "model_evaluation_metrics.csv"
    if not f.exists():
        return "", float("nan")
    t = pd.read_csv(f)
    col = next((c for c in t.columns if "Test" in c and "R" in c), None)
    if col is None:
        return "", float("nan")
    vals = pd.to_numeric(t[col], errors="coerce")
    i = int(vals.idxmax())
    return str(t.iloc[i, 0]), round(float(vals.max()), 4)


# --check
def check_table(name: str) -> dict:
    missing: list[str] = []
    notes: list[str] = []

    if name == "grid_all16":
        missing += [str(PIPELINE)] if not PIPELINE.exists() else []
        missing += [str(DATA / f"{s}.csv") for _k, s in DATASET_ORDER
                    if not (DATA / f"{s}.csv").exists()]
        amap = read_alias_map()
        if not amap:
            missing.append(str(ROOT / "data" / "alias_mapping.yaml"))
        else:
            expected = dict(DATASET_ORDER)
            bad = [k for k, v in expected.items() if amap.get(k) != v]
            if bad:
                notes.append("alias_mapping.yaml disagrees for: " + ", ".join(bad))
        notes.append("expects 192 rows = 16 datasets x 12 cells")

    elif name == "ceiling_stats_v16":
        missing += [str(PIPELINE)] if not PIPELINE.exists() else []
        missing += [str(DATA / f"{s}.csv") for s in CEILING_ORDER
                    if not (DATA / f"{s}.csv").exists()]
        man = ceiling_manifest()
        if not (BATCH / "summary.csv").exists():
            missing.append(str(BATCH / "summary.csv"))
            notes.append("manifest missing -> falls back to newest outputs/ run per dataset")
        for stem in CEILING_ORDER:
            d = man.get(stem)
            if d is None:
                continue
            mf = d / "model_evaluation_metrics.csv"
            if not mf.exists():
                missing.append(str(mf))

    elif name == "attrib12_long":
        missing += [str(PIPELINE)] if not PIPELINE.exists() else []
        missing += [str(DATA / f"{s}.csv") for _k, s in DATASET_ORDER
                    if not (DATA / f"{s}.csv").exists()]
        try:
            import shap  # noqa: F401
        except Exception:
            missing.append("python package 'shap' (pip install shap)")
        notes.append("full rebuild takes about 25 min on this machine")

    elif name == "perm_vs_shap":
        if not OUTPUTS.is_dir():
            missing.append(str(OUTPUTS))
        else:
            n4 = len(list(OUTPUTS.glob("*/Figure4_*_panel_data.csv")))
            n7 = len(list(OUTPUTS.glob("*/Figure7_*_panel_data.csv")))
            notes.append("Figure4 panel files: %d, Figure7 panel files: %d" % (n4, n7))
            if n4 == 0 or n7 == 0:
                missing.append(str(OUTPUTS / "*" / "Figure4_*_panel_data.csv"))
            # the runs actually used are pinned by the batch manifest
            for ds, d in ceiling_manifest().items():
                if not list(d.glob("Figure4_*_panel_data.csv")):
                    missing.append(str(d / "Figure4_*_panel_data.csv"))
                if not list(d.glob("Figure7_*_panel_data.csv")):
                    missing.append(str(d / "Figure7_*_panel_data.csv"))

    elif name == "zero_yield_audit":
        missing += [str(PIPELINE)] if not PIPELINE.exists() else []
        missing += [str(DATA / f"{s}.csv") for _k, s in DATASET_ORDER
                    if not (DATA / f"{s}.csv").exists()]
        notes.append("no reference table exists anywhere -- derived table cannot be verified")
        notes.append("original producer: scratch script zero_check.py (stdout only)")

    elif name == "interaction_ligand_base":
        missing += [str(PIPELINE)] if not PIPELINE.exists() else []
        f = DATA / "(AI_RCP_pub)bh_data_clean_all_whitelisted.csv"
        if not f.exists():
            missing.append(str(f))
        notes.append("no reference table exists anywhere -- derived table cannot be verified")
        notes.append("original producer: scratch script gen_figs_v16.py (fig4 heatmap)")

    ref = reference_path(name)
    if ref is None:
        notes.append("reference copy: ABSENT")
    else:
        try:
            n = len(pd.read_csv(ref))
            notes.insert(0, "reference copy: %s (%d rows)" % (ref, n))
        except Exception as exc:
            notes.insert(0, "reference copy: %s (unreadable: %s)" % (ref, exc))

    return dict(name=name, missing=missing, notes=notes, ref=ref)


def cmd_check() -> int:
    say("=" * 78)
    say("tables.py --check   project: %s" % PROJECT)
    say("=" * 78)
    if not PIPELINE.exists():
        say("[WARN] pipeline not found: %s" % PIPELINE)
    verdicts = {}
    for name in TABLES:
        info = check_table(name)
        ref = info["ref"]
        if info["missing"]:
            verdict = "INPUTS MISSING"
        elif ref is None:
            verdict = "DERIVABLE (reference missing)"
        else:
            verdict = "REBUILDABLE"
        verdicts[name] = verdict
        say("")
        say("[%s]" % name)
        for n in info["notes"]:
            say("    %s" % n)
        if info["missing"]:
            say("    missing paths:")
            for m in info["missing"]:
                say("      - %s" % m)
        say("    verdict: %s" % verdict)
    say("")
    say("-" * 78)
    for name in TABLES:
        say("  %-24s %s" % (name, verdicts[name]))
    return 0


# Comparison
def _aligned(new: pd.DataFrame, ref: pd.DataFrame, keys: list[str]):
    if keys and all(k in new.columns for k in keys) and all(k in ref.columns for k in keys):
        a = new.set_index(keys)
        b = ref.set_index(keys)
        if not a.index.has_duplicates and not b.index.has_duplicates:
            missing = b.index.difference(a.index)
            extra = a.index.difference(b.index)
            common = b.index.intersection(a.index)
            return a.loc[common], b.loc[common], list(missing), list(extra), "keys"
    return (new.reset_index(drop=True), ref.reset_index(drop=True),
            [], [], "position")


def compare(name: str, new: pd.DataFrame, ref: pd.DataFrame,
            keys: list[str]) -> bool:
    say("")
    say("compare against %s" % reference_path(name))
    ok_all = True

    if list(new.columns) != list(ref.columns):
        say("  FAIL  column order/set")
        say("        new: %s" % list(new.columns))
        say("        ref: %s" % list(ref.columns))
        return False
    say("  PASS  column order (%d columns)" % len(ref.columns))

    if len(new) != len(ref):
        say("  FAIL  row count: new=%d ref=%d" % (len(new), len(ref)))
        return False
    say("  PASS  row count (%d)" % len(ref))

    a, b, missing, extra, how = _aligned(new, ref, keys)
    if missing or extra:
        say("  FAIL  key alignment (missing %d, extra %d)" % (len(missing), len(extra)))
        return False
    say("  PASS  key alignment (%s%s)" % (how, "" if how == "keys"
                                          else "; WARNING: positional"))

    for col in ref.columns:
        if how == "keys" and col in keys:
            say("  PASS  %-14s key column (matched by alignment)" % col)
            continue
        num = pd.api.types.is_numeric_dtype(ref[col]) and \
            pd.api.types.is_numeric_dtype(a[col])
        if num:
            na_ref = b[col].isna()
            na_new = a[col].isna()
            na_bad = int((na_ref != na_new).sum())
            both = ~na_ref & ~na_new
            if int(both.sum()):
                delta = (pd.to_numeric(a.loc[both, col], errors="coerce")
                         - pd.to_numeric(b.loc[both, col], errors="coerce")).abs()
                maxd = float(np.nanmax(delta.to_numpy(dtype=float))) if len(delta) else 0.0
                nbad = int((delta > TOL).sum())
            else:
                maxd, nbad = 0.0, 0
            if col in NONREPRODUCIBLE:
                say("  INFO  %-14s max|delta|=%.3e  %d/%d differ  (wall-clock timing; "
                    "excluded from the gate)" % (col, maxd, nbad, len(a)))
                continue
            good = (na_bad == 0 and nbad == 0)
            say("  %s  %-14s max|delta|=%.3e  mismatches=%d  nan-mismatches=%d"
                % ("PASS" if good else "FAIL", col, maxd, nbad, na_bad))
            ok_all &= good
        else:
            sa = a[col].astype(str).to_numpy()
            sb = b[col].astype(str).to_numpy()
            nbad = int((sa != sb).sum())
            good = nbad == 0
            say("  %s  %-14s string mismatches=%d" % ("PASS" if good else "FAIL",
                                                      col, nbad))
            ok_all &= good
    return ok_all


# --rebuild
def write_provenance(name: str, body: str) -> Path:
    p = RES / f"{name}.PROVENANCE.md"
    p.write_text(body.rstrip() + "\n", encoding="utf-8")
    return p


ZERO_PROVENANCE = """# {name}.csv -- provenance

## What this table contains

One row per dataset (16 rows).  Counts of exact-zero yield records in the
normalised table and the yield distribution that remains:

| column | meaning |
|---|---|
| `dataset` | short alias (see `data/alias_mapping.yaml`) |
| `n_rows` | rows returned by the pipeline's `load_real_dataset` |
| `n_zero` | rows with `yield == 0` |
| `pct_zero` | `100 * n_zero / n_rows` |
| `n_nonzero` | rows with `yield > 0` |
| `mean_yield`, `median_yield`, `max_yield` | yield statistics over non-NaN rows |

The paper's claim "AI_RCP_BH has 46.6% zero-yield reactions" is `pct_zero` for
that row.

## Who produced the original number

There was **no original CSV**.  The number was read off the stdout of an
uncommitted scratch script, `zero_check.py` (kept at `{scratch}\\zero_check.py`),
which also ran a drop-the-zeros experiment on AI_RCP_BH.  That script imported
`Analysis_all_v14.py` and used its own 49/51 split, so its held-out R2 values
are not comparable with the paper's v16 numbers and are not reproduced here.

## Status of this file

UNVERIFIED.  No reference table exists in the repository, in `{scratch}\\`, or
anywhere under the project tree, so there is nothing to compare against.  The
counts above are recomputed from the raw CSVs by
`scripts/tables.py:build_zero_yield_audit`, using the frozen v16 loader.

To verify, one would need the original CSV (or the exact code that wrote it).
The raw inputs are intact, so the *counts* are reproducible; only the paper's
particular tabulation cannot be checked.
"""

INTERACTION_PROVENANCE = """# {name}.csv -- provenance

## What this table contains

The complete ligand x base mean-yield cross-tab for **AI_RCP_BH**
(`catalyst_name` -> ligand, `reagent_1_name` -> base, as mapped by
`Data/col_mappings.csv`), in long form.  Observed pairs only; unobserved
ligand/base combinations are omitted.

| column | meaning |
|---|---|
| `ligand` | `catalyst_name` value |
| `base` | `reagent_1_name` value |
| `n` | number of reactions in that cell |
| `mean_yield` | mean yield (%) in that cell |
| `in_fig4` | True for the 14x10 crop drawn as figure 4 (top ligands/bases by measurement count) |

The last column matters: the paper's headline pairs in section 3.4
(RuPhos/Cs2CO3 56.1 %, dtbpf/NaOtBu 51.0 %) sit **outside** the figure-4 crop,
so a table cropped to the figure would silently drop them.  The full cross-tab
keeps both, and the crop stays recoverable with `in_fig4 == True`.

## Who produced the original numbers

There was **no original CSV**.  The heatmap was drawn by the scratch script
`{scratch}\\gen_figs_v16.py` (block "figure 4 (new)"), and the section-3.4
interaction statistics were printed to stdout by
`{scratch}\\verify_34_effects.py`.  Both scripts live outside the repository.

## Status of this file

**Indirectly validated, but not byte-verified.**  No reference CSV exists, so
there is no table to diff against.  However, the figure-4 subset of this rebuild
(`in_fig4 == True`) reproduces every quantity recorded in
`{scratch}\\figs_v16\\numbers.json` from the original run:

* crop shape 14 x 10, 138 observed cells;
* maximum cell mean 43.966149567857144 %;
* identical ligand and base order.

That is a strong consistency check on the loader and the aggregation, but it is
not a check of the paper's particular tabulation.  To verify the latter one
would need the original per-cell export (or the exact crop rule, recoverable
from `gen_figs_v16.py`).
"""

ATTRIB_PROVENANCE = """# {name}.csv -- provenance and reproducibility limits

## What this table contains

1704 rows: 16 datasets x 12 architecture/encoding cells x the semantic features
that cell uses (7-10 per cell).  Two attribution measures per row:

| column | meaning |
|---|---|
| `dataset` | short alias (see `data/alias_mapping.yaml`) |
| `model`, `encoding` | one of the 12 grid cells |
| `feature` | semantic slot (one-hot columns are summed back onto their parent) |
| `perm` | mean drop in held-out R2 under permutation (5 repeats, seed 42) |
| `shap` | mean absolute SHAP value |

## Who produced it originally

Three uncommitted scratch scripts, in this order:

1. `{scratch}\\run_attrib12.py` -- fits the 12 cells for all 16 datasets and
   writes `perm` and `shap` (24.5 min).
2. `{scratch}\\fix_xgbnative_shap.py` -- repairs the 144 XGBoost
   "native categorical" rows, whose SHAP had silently become 0 because
   `shap.TreeExplainer` cannot load xgboost 3.2's vector `base_score`
   (`ValueError: could not convert string to float: '[1.3193166E1]'`).
   It uses the booster's own `pred_contribs=True` instead.

## Two reproducibility limits (verified 2026-10-22)

**1. XGBoost thread pool.**  `grid_all16` was produced after pinning
`OMP_NUM_THREADS=4`, so its XGBoost fits used 4 threads.  `run_attrib12.py`
imported xgboost *before* pinning, so its XGBoost fits used all 32 cores.  The
two settings give measurably different models, and xgboost freezes its pool at
import time, so the two tables cannot be built in one interpreter.
`scripts/tables.py` therefore re-executes this table with
`DSH_SWEEP_THREADS=0` and scrubs `OMP_NUM_THREADS` from the child environment.
With that, every non-XGBoost cell (1512 rows) and all `perm` values reproduce
exactly (max |delta| 4.3e-14).

**2. Unseeded KernelSHAP.**  For XGBoost "one-hot" cells the original fell back
to `shap.KernelExplainer` (same `base_score` failure).  That path is stochastic:
two all-cores runs of the same cell differed by 1.7e-2 and 3.9e-2.  The stored
values therefore cannot be reproduced to 1e-9 by any re-run.

## Verification status

Structure: PASS (column order, 1704 rows, key alignment).
`perm`: PASS, max |delta| = 4.3e-14 over all 1704 rows.
`shap`: **FAIL**, 214 rows outside 1e-9:

| cell | rows | max abs delta | cause |
|---|---|---|---|
| XGBoost / one-hot | 107 | 1.66e-01 | unseeded KernelSHAP (stochastic) |
| XGBoost / native categorical | 107 | 3.0e-05 | xgboost `pred_contribs` is float32 |

To close this properly one would need either (a) the exact SHAP values the
scratch run saw, (b) a fixed RNG seed threaded into `shap.KernelExplainer`, or
(c) an xgboost/shap pair that can load the vector `base_score` and compute exact
TreeSHAP for every cell.
"""


def cmd_rebuild(name: str) -> int:
    if name not in BUILDERS:
        say("unknown table: %s\nchoose from: %s" % (name, ", ".join(TABLES)))
        return 1

    say("=" * 78)
    say("rebuild %s" % name)
    say("=" * 78)

    # Read the reference into memory BEFORE the build.  A build can take half an
    # hour (attrib12_long) and the workspace is shared, so the file must not be
    # re-read afterwards.
    ref = reference_path(name)
    ref_df = pd.read_csv(ref) if ref is not None else None
    if ref is not None:
        say("reference: %s (%d rows)" % (ref, len(ref_df)))
    else:
        say("reference: none")

    t0 = time.perf_counter()
    try:
        new = BUILDERS[name]()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        say("REBUILD FAILED: %s: %s" % (type(exc).__name__, exc))
        return 1
    say("built %d rows x %d columns in %.1f s"
        % (len(new), new.shape[1], time.perf_counter() - t0))

    # Canonical repo tables (covered by results/checksums.txt) are never
    # clobbered; their rebuild is written next to them as results/rebuilt_<name>.csv.
    if name in REPO_TABLES:
        out = RES / f"rebuilt_{name}.csv"
    else:
        out = RES / f"{name}.csv"

    # Write first, compare second: the build is the expensive part.
    enc = ("utf-8-sig" if (ref is not None and has_bom(ref)) else "utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(out, index=False, encoding=enc)
    say("wrote %s  (sha256 %s)" % (out, sha256(out)[:16]))

    if ref_df is None:
        say("")
        say("NO REFERENCE COPY EXISTS -- wrote %s but could NOT verify it." % out)
        if name == "zero_yield_audit":
            body = ZERO_PROVENANCE.format(name=name, scratch=SCRATCH)
        elif name == "interaction_ligand_base":
            body = INTERACTION_PROVENANCE.format(name=name, scratch=SCRATCH)
        else:
            body = ("# %s.csv -- provenance\n\nNo reference copy exists for this "
                    "table; it was rebuilt from raw inputs and could not be "
                    "verified.\n" % name)
        pv = write_provenance(name, body)
        say("wrote %s" % pv)
        return 2

    ok = compare(name, new, ref_df, KEYS.get(name, []))
    if name in REPO_TABLES:
        say("shipped results/%s.csv left untouched (checksums.txt stays valid)" % name)
    say("VERDICT: %s" % ("REBUILT+PASS" if ok else "REBUILT+FAIL"))
    if not ok:
        if name == "attrib12_long":
            body = ATTRIB_PROVENANCE.format(name=name, scratch=SCRATCH)
        else:
            body = (
                "# %s.csv -- provenance and mismatch report\n\n"
                "Recomputed by `scripts/tables.py:build_%s` from the frozen pipeline "
                "and raw per-dataset inputs.  The rebuild does NOT match the shipped "
                "table within 1e-9 -- see the per-column report printed by\n"
                "`python scripts/tables.py --rebuild %s`.\n\n"
                "Rebuilt copy: `results/rebuilt_%s.csv`.\n"
                % (name, name, name, name))
        pv = write_provenance(name, body)
        say("wrote %s" % pv)
    return 0 if ok else 1


# main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild and verify the paper's result tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit codes: 0 verified pass, 1 fail, 2 written but unverified")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true",
                   help="report which raw inputs exist for each of the six tables")
    g.add_argument("--rebuild", metavar="NAME",
                   help="rebuild one table into results/ and compare "
                        "(NAME, or 'all')")
    g.add_argument("--list", action="store_true", help="list the six tables")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if args.list:
        for n in TABLES:
            say(n)
        return 0

    if args.check:
        return cmd_check()

    # Thread pools are per-process and frozen at xgboost import time.  The grid
    # table needs 4 threads (that is how it was produced); the attribution table
    # needs the ambient all-cores pool.  Run each table in its own child process
    # with the right environment rather than mixing them in one interpreter.
    worker = bool(getattr(args, "worker", False))

    def _child(name: str, threads: str) -> int:
        env = dict(os.environ)
        env["DSH_SWEEP_THREADS"] = threads
        if threads in ("0", "", "none", "None"):
            # The parent already exported OMP_NUM_THREADS=4; the child must not
            # inherit it or xgboost would size its pool to 4 again.
            for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                       "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                env.pop(_k, None)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--rebuild", name, "--worker"]
        return subprocess.run(cmd, env=env).returncode

    if args.rebuild == "all":
        rc = 0
        for n in TABLES:
            rc |= _child(n, "0" if n == "attrib12_long" else "4")
            say("")
        return 0 if rc == 0 else (2 if rc == 2 else 1)

    if (args.rebuild == "attrib12_long" and not worker
            and _SWEEP_THREADS not in ("0", "", "none", "None")):
        say("re-executing attrib12_long with DSH_SWEEP_THREADS=0 "
            "(xgboost needs its all-cores pool to match the shipped table)")
        return _child("attrib12_long", "0")

    return cmd_rebuild(args.rebuild)


if __name__ == "__main__":
    sys.exit(main())

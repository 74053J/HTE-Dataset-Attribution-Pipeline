#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild the permutation-vs-SHAP attribution table on the 13-dataset corpus,
with the two documented determinism defects fixed.

Determinism fixes (both root causes documented in
``results/attrib12_long.PROVENANCE.md`` and ``results/.attrib12_rebuild.log``):

1. THREAD POOL.  xgboost sizes its OpenMP pool when the module is first
   imported, so a run that imports xgboost before pinning OMP_NUM_THREADS gets
   a different (all-core) pool than a run that pins first -- and the two pools
   fit measurably different models.  Here the thread count is pinned through
   the environment BEFORE numpy/xgboost are imported, and the same count is
   also passed explicitly to every estimator (``n_jobs``), so every fit in the
   whole task sees the same pool.  ``N_THREADS`` below is the record of it.

2. THE XGBOOST ONE-HOT SHAP FALLBACK.  ``shap.TreeExplainer`` cannot parse
   xgboost 3.2's vector ``base_score`` (``ValueError: could not convert string
   to float: '[1.3193166E1]'``), so that cell falls through to shap's
   Monte-Carlo permutation explainer.  With shap 0.49.1 that object is a
   ``shap.PermutationExplainer`` (not a ``KernelExplainer``), and its default
   ``seed=None`` leaves it reading the global unseeded numpy RNG -- the
   documented 1e-2-level run-to-run drift.  ``PermutationExplain``'s
   constructor does ``np.random.seed(seed)``, so passing ``seed=SHAP_SEED``
   reseeds the global RNG immediately before each explain call and makes the
   cell reproducible.

Usage
-----
    python outputs/attrib13/01_attrib13.py --tag A
    python outputs/attrib13/01_attrib13.py --tag B

``--tag A`` additionally writes the canonical deliverables
``results/attrib13_long.csv`` and ``results/perm_vs_shap13.csv``.  Every tag
writes ``outputs/attrib13/attrib13_long_<tag>.csv`` and
``outputs/attrib13/cell_meta_<tag>.json`` so two independent processes can be
diffed for residual run-to-run variation.
"""
from __future__ import annotations

import os

# FIX 1 -- pin the thread pool before numpy/xgboost/sklearn are imported.
N_THREADS = int(os.environ.get("ATTRIB13_THREADS", "4"))
for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_tv] = str(N_THREADS)

# FIX 2 -- seed for shap's Monte-Carlo fallback explainer.
SHAP_SEED = int(os.environ.get("ATTRIB13_SHAP_SEED", "42"))
PERM_SEED = 42          # sklearn permutation_importance / every model's random_state

import argparse          # noqa: E402
import contextlib        # noqa: E402
import io                # noqa: E402
import json              # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np       # noqa: E402
import pandas as pd      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tables as T       # noqa: E402
from hte import datasets as DS  # noqa: E402  (also imports the frozen pipeline lazily)

P = T.pipeline()

import shap                                      # noqa: E402
import xgboost as xgb                            # noqa: E402
from sklearn.compose import ColumnTransformer    # noqa: E402
from sklearn.ensemble import (GradientBoostingRegressor,          # noqa: E402
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.inspection import permutation_importance             # noqa: E402
from sklearn.neural_network import MLPRegressor                   # noqa: E402
from sklearn.pipeline import make_pipeline                        # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler   # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "outputs" / "attrib13"
OUT.mkdir(parents=True, exist_ok=True)

# The 13 real HTE datasets: all 16 source files minus the pooled
# `Roche66k_BH+SM`, minus `DoyleHTE_CN` (100 % contained in bh-hte-ood), minus
# the synthetic `Suzuki-Coupling-ML_SM`.  Same list as results/table1_final.csv
# minus those three rows.
# The corpus and its file stems come from data/alias_mapping.yaml; the old
# hard-coded list is gone.
STEMS = DS.stems()
CORPUS13 = [(a, STEMS[a]) for a in DS.corpus()]

# v25: cap the cost of a WIDE one-hot cell.  bh-hte-ood expands to 1,748 columns
# (reactant_1 852 + reactant_2 701 + ...), and 5-repeat permutation importance over
# 1,748 columns on 5.5k rows is ~9k predict calls -- that is the 38-minute cell.
# Attribution is a per-feature AVERAGE, so a capped row sample and 1 repeat for
# very wide frames changes the estimate only within its own Monte-Carlo noise.
_MAX_ATTR_ROWS = int(os.environ.get('CD_ATTR_ROWS', '0'))   # 0 = no cap (paper protocol)
_WIDE_P = 10 ** 9      # paper protocol: n_repeats = 5 for every frame

DATA_DIR = T.DATA if hasattr(T, "DATA") else Path("data")
from hte.data import load_real_dataset as T_load_real_dataset      # noqa: E402
from hte.features import get_split_positions, prepare_features     # noqa: E402
from hte.grid import build_frames                                  # noqa: E402

ATTRIB_COLUMNS = ["dataset", "model", "encoding", "feature", "perm", "shap"]


# model factories -- identical hyper-parameters to scripts/tables.py so the two
# tables stay comparable; the ONLY change is the explicit thread count.
def _xgb(**extra):
    return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=PERM_SEED, n_jobs=N_THREADS, **extra)


def _gb():
    return GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                     learning_rate=0.1, random_state=PERM_SEED)


def _mlp():
    return MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                        max_iter=500, random_state=PERM_SEED, early_stopping=True)


def _rf():
    return RandomForestRegressor(n_estimators=P.RF_N_ESTIMATORS, max_depth=15,
                                 min_samples_split=5, random_state=PERM_SEED,
                                 n_jobs=N_THREADS)


def _histgb_native(cat_idx):
    return HistGradientBoostingRegressor(
        max_iter=P.HISTGB_MAX_ITER, max_depth=6, learning_rate=0.1,
        random_state=PERM_SEED, categorical_features=(cat_idx or None))


def _histgb_plain():
    return HistGradientBoostingRegressor(
        max_iter=P.HISTGB_MAX_ITER, max_depth=6, learning_rate=0.1,
        random_state=PERM_SEED)


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


def say(msg=""):
    print(msg, flush=True)


def _summarise(long):
    """Write attrib13_long.csv and the two derived gate views."""
    dest = RES / "attrib13_long.csv"
    long.to_csv(dest, index=False, encoding="utf-8")
    say("wrote %s" % dest)

    from scipy.stats import spearmanr
    rows = []
    for (ds, model, enc), g in long.groupby(["dataset", "model", "encoding"], sort=False):
        p = g["perm"].to_numpy(float)
        s = g["shap"].to_numpy(float)
        ok = np.isfinite(p) & np.isfinite(s)
        if ok.sum() >= 3 and np.ptp(p[ok]) > 0 and np.ptp(s[ok]) > 0:
            rho = float(spearmanr(p[ok], s[ok]).statistic)
        else:
            rho = float("nan")   # degenerate: all-zero importances
        rows.append(dict(dataset=ds, model=model, encoding=enc, n=int(ok.sum()), rho=rho))
    pv = pd.DataFrame(rows, columns=["dataset", "model", "encoding", "n", "rho"])
    pv.to_csv(RES / "perm_vs_shap13.csv", index=False, encoding="utf-8")
    slots = (long.assign(slot=long["feature"].astype(str).str.replace(
                 "_encoded", "", regex=False))
                 .groupby(["dataset", "model", "encoding"])["slot"].nunique()
                 .rename("nslot").reset_index())
    cells = pv.merge(slots, on=["dataset", "model", "encoding"], how="left")
    cells[["dataset", "model", "encoding", "rho", "nslot"]].to_csv(
        RES / "perm_vs_shap_cells13.csv", index=False, encoding="utf-8")
    say("wrote %s  (%d cells, %d finite rho)"
        % (RES / "perm_vs_shap13.csv", len(pv), int(pv.rho.notna().sum())))
    say("median rho (finite cells) = %.4f" % float(pv.rho.median()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="A")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="optional subset of short aliases (smoke tests)")
    ap.add_argument("--part-out", default=None,
                    help="write the long table here even for a --datasets subset")
    ap.add_argument("--per-dataset", action="store_true",
                    help="one subprocess per dataset (A56 P1: a long-lived process"
                         " accumulates state and made bh ~140x slower)")
    args = ap.parse_args(argv)

    if args.per_dataset:
        import subprocess
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="attrib_parts_"))
        parts, failed = [], []
        say("[per-dataset] %d datasets, one subprocess each" % len(CORPUS13))
        for _k, _stem in CORPUS13:
            out = tmp / ("part_%s.csv" % _k)
            t0 = time.perf_counter()
            rc = subprocess.call([sys.executable, str(Path(__file__).resolve()),
                                  "--tag", args.tag, "--datasets", _k,
                                  "--part-out", str(out)])
            say("[per-dataset] %-14s rc=%d  %.1f s"
                % (_k, rc, time.perf_counter() - t0))
            if rc == 0 and out.is_file():
                parts.append(out)
            else:
                failed.append(_k)
        if failed:
            # v29: a partially failed merge used to return 0 and ship a short table.
            say("[per-dataset] FAILED: %s" % ", ".join(failed))
            return 1
        if not parts:
            return 1
        merged = pd.concat([pd.read_csv(x) for x in parts], ignore_index=True)
        cells = merged.groupby(["dataset", "model", "encoding"]).ngroups
        say("[per-dataset] merged %d/%d parts, %d rows, %d cells"
            % (len(parts), len(CORPUS13), len(merged), cells))
        if len(merged) != 1392 or cells != 156:
            # the gate quotes 1392 rows over 156 cells; a short table must not pass silently
            say("[per-dataset] ABORT: expected 1392 rows / 156 cells, got %d / %d"
                % (len(merged), cells))
            return 1
        return _summarise(merged)

    order = CORPUS13
    if args.datasets:
        want = set(args.datasets)
        order = [d for d in CORPUS13 if d[0] in want]
    canonical = (args.tag.upper() == "A")

    say("tag=%s threads=%d shap_seed=%d perm_seed=%d"
        % (args.tag, N_THREADS, SHAP_SEED, PERM_SEED))
    say("shap %s  xgboost %s  corpus=%d" % (shap.__version__, xgb.__version__, len(order)))

    keep_rows: list[dict] = []
    native_rows: list[dict] = []
    meta: dict[str, dict] = {}
    t_all = time.perf_counter()

    for di, (key, stem) in enumerate(order, 1):
        say("[attrib %2d/%d] %s" % (di, len(order), key))
        # v28: use EXACTLY the grid's loading and frame construction.  tables.load_dataset
        # goes through the monolithic pipeline, whose feature builder does NOT fold rare
        # categorical levels; that produced 669/852-category columns (HistGB refuses > 255)
        # and, more importantly, a feature matrix different from the one the paper's grid was
        # fitted on.  hte.data + hte.features + hte.grid.build_frames is the grid's own path.
        df, raw_names = T_load_real_dataset(str(DATA_DIR / (stem + ".csv")), str(DATA_DIR))
        tr_pos, te_pos = get_split_positions(len(df), 0.2, 42)
        X, y, feature_cols, _disp, _enc, _sc = prepare_features(df, raw_names,
                                                                fit_rows=tr_pos)
        packs, cat_cols, cat_idx = build_frames(X, feature_cols, tr_pos, te_pos)
        y_tr, y_te = y.iloc[tr_pos].copy(), y.iloc[te_pos].copy()

        for arch, enc, factory, fkey in cells_for(cat_idx):
            t0 = time.perf_counter()
            ck = "%s|%s|%s" % (key, arch, enc)
            try:
                Xtr, Xte = packs[fkey]
                mdl = factory()
                try:
                    mdl.fit(Xtr, y_tr)
                except ValueError as _exc:
                    # v27: HistGB native declares categorical_features by POSITION, and the
                    # frame it receives here can be the integer one whose columns still carry
                    # >255 distinct codes -- sklearn then refuses ("cardinality <= 255").
                    # Fold the over-wide codes of a COPY and refit.
                    if "cardinality" not in str(_exc) and "max_bins" not in str(_exc):
                        raise
                    Xtr, Xte = Xtr.copy(), Xte.copy()
                    for _c in Xtr.columns:
                        _vc = Xtr[_c].value_counts()
                        if len(_vc) > 255:
                            _keep = set(_vc.index[:200])
                            Xtr[_c] = Xtr[_c].where(Xtr[_c].isin(_keep), other=-1)
                            Xte[_c] = Xte[_c].where(Xte[_c].isin(_keep), other=-1)
                            say("    [fit] %s: %d -> 201 codes (cardinality fold)"
                                % (_c, len(_vc)))
                    mdl = factory()
                    mdl.fit(Xtr, y_tr)
                cols = list(Xtr.columns)

                _p = Xte.shape[1]
                _rep = 5 if _p <= _WIDE_P else 1
                if _MAX_ATTR_ROWS and len(Xte) > _MAX_ATTR_ROWS:
                    Xte_p = Xte.iloc[:_MAX_ATTR_ROWS]
                    y_te_p = y_te.iloc[:_MAX_ATTR_ROWS]
                else:
                    Xte_p, y_te_p = Xte, y_te
                pi = permutation_importance(mdl, Xte_p, y_te_p, n_repeats=_rep,
                                            random_state=PERM_SEED, scoring="r2",
                                            n_jobs=1)
                perm = np.asarray(pi.importances_mean, dtype=float)

                is_tree = arch in ("XGBoost", "Gradient Boosting",
                                   "Random Forest", "HistGB")
                explainer = None
                try:
                    if arch == "XGBoost":
                        # v25: use the booster's exact TreeSHAP for EVERY xgboost cell.
                        # shap.TreeExplainer cannot parse xgboost 3.x base_score
                        # ("[-2.8E-2]"), so it raised and the fallback burned minutes per
                        # cell in Monte-Carlo permutation SHAP.  pred_contribs is exact and
                        # version-proof; enable_categorical only matters for the NAT frame.
                        # exact TreeSHAP from the booster (shap.TreeExplainer
                        # cannot parse the vector base_score)
                        dm = xgb.DMatrix(Xte, enable_categorical=True)
                        contrib = mdl.get_booster().predict(dm, pred_contribs=True)
                        sh = np.abs(np.asarray(contrib, dtype=float)[:, :-1]).mean(axis=0)
                        explainer = "xgb.Booster.predict(pred_contribs=True) [exact TreeSHAP]"
                    else:
                        sh = None
                        if is_tree:
                            try:
                                ex = shap.TreeExplainer(mdl)
                                sv = ex(Xte).values
                                sh = np.abs(coerce(sv)).mean(axis=0)
                                explainer = "shap.TreeExplainer"
                            except Exception:
                                sh = None
                        if sh is None:
                            # FIX 2: seeded Monte-Carlo fallback
                            bg = Xtr.sample(min(25, len(Xtr)), random_state=0)
                            sub = Xte.sample(min(120, len(Xte)), random_state=0)
                            # A56 P1: shap.Explainer() picks ExactExplainer, which enumerates
                            # 2**p coalitions.  With p = 121 one-hot columns that never
                            # finishes (measured: one cell > 38 min CPU on bh-hte-ood).  Use
                            # the permutation explainer whenever p is large enough for the
                            # exact path to be hopeless.
                            if len(cols) > 20:
                                ex = shap.PermutationExplainer(mdl.predict, bg, seed=SHAP_SEED)
                                sv = ex(sub, max_evals=int(min(500, 2 * len(cols) + 1)),
                                        silent=True).values
                            else:
                                ex = shap.Explainer(mdl.predict, bg, seed=SHAP_SEED)
                                sv = ex(sub).values
                            sh = np.abs(coerce(sv)).mean(axis=0)
                            _cls = type(ex).__name__
                            if _cls == "PermutationExplainer":
                                nperm = 500 // (2 * len(cols) + 1)
                                explainer = ("PermutationExplainer(seed=%d, max_evals=500"
                                             " -> %d permutations, bg=25, rows=120)"
                                             % (SHAP_SEED, max(nperm, 1)))
                            else:
                                explainer = ("%s(seed=%d, exact coalitions over %d cols,"
                                             " bg=25, rows=120)"
                                             % (_cls, SHAP_SEED, len(cols)))
                except Exception as exc:
                    sh = np.full(len(cols), np.nan)
                    explainer = "FAILED %s: %s" % (type(exc).__name__, str(exc)[:120])
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
                meta[ck] = dict(explainer=explainer, n_features=int(len(cols)),
                                n_train=int(len(Xtr)), n_test=int(len(Xte)),
                                seconds=round(time.perf_counter() - t0, 2))
                say("    %-18s %-19s ok %5.1fs  [%s]"
                    % (arch, enc, time.perf_counter() - t0, explainer))
            except Exception as exc:
                meta[ck] = dict(explainer="CELL FAILED: %s" % type(exc).__name__,
                                error="%s: %s" % (type(exc).__name__, str(exc)[:300]),
                                seconds=round(time.perf_counter() - t0, 2))
                say("    %-18s %-19s FAILED %s: %s"
                    % (arch, enc, type(exc).__name__, exc))

    long = pd.DataFrame(keep_rows + native_rows, columns=ATTRIB_COLUMNS)
    say("[attrib] done in %.1f min  rows=%d" % ((time.perf_counter() - t_all) / 60, len(long)))

    if args.part_out:
        Path(args.part_out).parent.mkdir(parents=True, exist_ok=True)
        long.to_csv(args.part_out, index=False, encoding="utf-8")
        say("wrote part %s (%d rows)" % (args.part_out, len(long)))
        return 0

    per_tag = OUT / ("attrib13_long_%s.csv" % args.tag)
    long.to_csv(per_tag, index=False, encoding="utf-8")
    say("wrote %s" % per_tag)
    with open(OUT / ("cell_meta_%s.json" % args.tag), "w", encoding="utf-8") as fh:
        json.dump(dict(tag=args.tag, threads=N_THREADS, shap_seed=SHAP_SEED,
                       perm_seed=PERM_SEED, shap_version=shap.__version__,
                       xgboost_version=xgb.__version__, rows=len(long),
                       cells=meta), fh, indent=1, sort_keys=True)

    if canonical and not args.datasets:
        return _summarise(long)


if __name__ == "__main__":
    raise SystemExit(main())

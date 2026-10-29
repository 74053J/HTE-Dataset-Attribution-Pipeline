#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 10b -- the acquisition analysis behind Figures 8/9.

Produces the two vectors the acquisition panel needs:

  outputs/acquisition/acq_mu.npy   (float32)  ensemble mean per virtual condition
  outputs/acquisition/acq_sd.npy   (float32)  ensemble standard deviation
  outputs/acquisition/acq_idx.npy  (int32)    the level index of each condition

Ported from ``_work/v239/acq.py`` (the scratch script that produced the shipped
arrays).  What changed in the port, and why:

* the dataset is not hard-coded -- it comes from ``data/alias_mapping.yaml``
  (``case_study: true``) or ``--dataset``;
* the axis list uses the current slot names.  The scratch script still named the
  two partner slots ``leaving_group`` / ``substrate_type``; they are
  ``reactant_2`` / ``reactant_1`` now, in the same positions;
* the loader and the feature builder come from ``hte`` (the packaged pipeline),
  not from a private ``tables`` module;
* every path is derived from the repository root, and the chunk size, the seed
  and the model panel are arguments with the historical defaults.

The model panel is the paper's five fixed configurations (XGBoost, Gradient
Boosting, Random Forest, HistGB, MLP).  mu/sd are the mean and the standard
deviation **across the five models**, which is what the EI curve uses.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

#: the seven axes the virtual library enumerates, in the historical order
AXES = ["metal", "catalyst", "ligand", "base", "solvent", "reactant_2", "reactant_1"]


def _panel(seed: int, n_jobs: int):
    import xgboost as xgb
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.neural_network import MLPRegressor
    return {
        "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=seed, enable_categorical=True,
                                    tree_method="hist", max_cat_to_onehot=4,
                                    n_jobs=n_jobs),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=seed),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=seed, n_jobs=n_jobs),
        "HistGB": HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, learning_rate=0.1, random_state=seed),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                            max_iter=500, random_state=seed, early_stopping=True),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import protocol as L
    from hte import datasets as DS
    from hte.v16_core import get_split_positions, prepare_features

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=None,
                    help="dataset alias; defaults to the case-study dataset in "
                         "data/alias_mapping.yaml")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "acquisition"))
    ap.add_argument("--chunk-rows", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the search space and stop without fitting")
    args = ap.parse_args()

    alias = args.dataset or DS.case_study()
    print("[acq] dataset %s" % alias)

    with contextlib.redirect_stdout(io.StringIO()):
        df, raw = L.load(alias)
    tr, te = get_split_positions(len(df))
    with contextlib.redirect_stdout(io.StringIO()):
        X, y, fcols, _disp, enc, _sc = prepare_features(df, raw, fit_rows=tr)

    cat_cols = [c for c in fcols if str(c).endswith("_encoded")]
    Xc = X.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].fillna(-1).astype("int64").astype("category")

    # one level list per axis, taken from the encoder that built the features
    levels = {}
    for a in AXES:
        le = enc.get(a) if hasattr(enc, "get") else None
        if le is not None and hasattr(le, "classes_"):
            levels[a] = [str(v) for v in le.classes_]
        elif a in df.columns:
            levels[a] = [str(v) for v in pd.unique(df[a].astype(str))]
        else:
            levels[a] = ["none"]
    # the library fixes the metal to its modal level (historical convention)
    if "metal" in df.columns:
        levels["metal"] = [str(df["metal"].astype(str).value_counts().index[0])]

    n_comb = int(np.prod([len(levels[a]) for a in AXES]))
    print("[acq] axes: %s" % {k: len(v) for k, v in levels.items()})
    print("[acq] combinations: %d" % n_comb)
    if args.dry_run:
        print("[acq] dry run: nothing fitted, nothing written")
        return 0

    panel = _panel(args.seed, args.n_jobs)
    fitted = {}
    for name, model in panel.items():
        frame = Xc if name in ("XGBoost", "HistGB") else X
        if name == "HistGB":
            idx_cat = [i for i, c in enumerate(fcols) if str(c).endswith("_encoded")]
            model.set_params(categorical_features=(idx_cat or None))
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(frame.iloc[tr], y.iloc[tr])
        fitted[name] = frame
        r2 = model.score(frame.iloc[te], y.iloc[te])
        print("[fit] %-18s R2 test %+.4f  %.1fs" % (name, r2, time.perf_counter() - t0))

    # every non-enumerated feature is held at its training median (numeric) or
    # mode (categorical)
    base_row = {}
    for c in fcols:
        col = X[c].iloc[tr]
        if c in cat_cols:
            base_row[c] = int(col.mode().iloc[0]) if len(col.mode()) else -1
        else:
            base_row[c] = float(pd.to_numeric(col, errors="coerce").median())

    idx = np.array(np.meshgrid(*[np.arange(len(levels[a])) for a in AXES],
                               indexing="ij")).reshape(len(AXES), -1).T.astype(np.int32)
    mu_all = np.empty(len(idx), dtype=np.float32)
    sd_all = np.empty(len(idx), dtype=np.float32)
    step = int(np.ceil(len(idx) / max(1, args.chunk_rows)))
    for s in range(step):
        sl = slice(s * args.chunk_rows, min((s + 1) * args.chunk_rows, len(idx)))
        block = idx[sl]
        frame = pd.DataFrame({c: np.repeat(base_row[c], len(block)) for c in fcols})
        for j, a in enumerate(AXES):
            col = a + "_encoded"
            if col in frame.columns:
                frame[col] = block[:, j].astype(np.int64)
        framec = frame.copy()
        for c in cat_cols:
            framec[c] = framec[c].astype("int64").astype("category")
        preds = []
        for name, (model) in panel.items():
            use = framec if name in ("XGBoost", "HistGB") else frame
            preds.append(np.asarray(model.predict(use), dtype=float))
        ps = np.vstack(preds)
        mu_all[sl] = ps.mean(0)
        sd_all[sl] = ps.std(0)
        if s % 4 == 0:
            print("  chunk %d/%d" % (s + 1, step))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "acq_mu.npy", mu_all)
    np.save(out / "acq_sd.npy", sd_all)
    np.save(out / "acq_idx.npy", idx)
    order = np.argsort(-mu_all)
    cum = np.maximum.accumulate(mu_all[order])
    print("[acq] top-1 / top-10 / top-100 predicted yield: %.2f / %.2f / %.2f"
          % (cum[0], cum[9], cum[99]))
    print("[acq] wrote %s (%d conditions)" % (out, len(idx)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

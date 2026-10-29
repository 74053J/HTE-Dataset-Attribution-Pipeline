# -*- coding: utf-8 -*-
"""Benchmark the virtual-library sweep under the four configurations.

    python bench_virtual.py <csv> <data_dir> <label>

Prints one JSON line with: label, n_conditions, seconds, peak_MB, n_pool, top20 (first 5),
mu stats of the pool.  Run it once per configuration with the env vars set.
"""
import importlib.util
import json
import os
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "mono", os.path.join(ROOT, "archive", "single_dataset_full_benchmark.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

csv_name, data_dir, label = sys.argv[1], sys.argv[2], sys.argv[3]
df, raw = m.load_real_dataset(os.path.join(data_dir, csv_name), data_dir)
tr, _te = m.get_split_positions(len(df))
X, y, feat_cols, disp, enc, scaler = m.prepare_features(df, raw, fit_rows=tr)

# three quick members: single fit each, no CV (the sweep does not depend on how they were fit)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
models = {
    "XGBoost": xgb.XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1),
}
for k, est in models.items():
    est.fit(X.iloc[tr], y.iloc[tr])
weights = np.full(len(models), 1.0 / len(models))
fstar, _ = m.resolve_ei_incumbent(df)

tracemalloc.start()
t0 = time.perf_counter()
bo_df, n_lib, preview = m.generate_bo_results_streaming(
    df=df, models=models, feat_cols=feat_cols, label_encoders=enc, scaler=scaler,
    best_observed_yield=fstar, chunk_combos=m.STREAM_CHUNK_COMBOS,
    max_combinations=m.MAX_VIRTUAL_COMBINATIONS, xi=0.01,
    keep_top=m.SURVIVOR_POOL_ROWS, weights=weights, verbose=False)
secs = time.perf_counter() - t0
cur, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

pred = bo_df["Ensemble Predicted Yield (%)"].to_numpy(float) if len(bo_df) else np.array([])
top = bo_df.head(20) if len(bo_df) else bo_df
cols = [c for c in ("catalyst", "ligand", "base", "solvent", "reactant_1", "reactant_2",
                     "catalyst_loading", "temperature", "time") if c in top.columns]
key = top[cols].astype(str).agg("|".join, axis=1).tolist() if len(top) else []
rec = dict(label=label, n_conditions=int(n_lib), seconds=round(secs, 2),
           peak_MB=round(peak / 1e6, 1), n_pool=int(len(bo_df)),
           mu_mean=round(float(pred.mean()), 4) if pred.size else None,
           mu_max=round(float(pred.max()), 4) if pred.size else None,
           top20_first5=key[:5], top20_hash=hash(tuple(key)))
print("RESULT " + json.dumps(rec, ensure_ascii=False))

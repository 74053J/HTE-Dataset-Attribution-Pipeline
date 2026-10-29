#!/usr/bin/env python
"""Condition recommendation -- two independent routes, one entry point.

Route (1): ``rank-real --dataset <ALIAS>``
    Rank the REAL, experimentally measured reaction records by the ensemble's
    predicted yield.  This is the cheap route: it consumes the per-member
    prediction table that the frozen pipeline already wrote
    (``model_predictions_all.csv``) and calls v16's own
    ``generate_figure6_direct_ranking`` to dedupe replicate conditions, apply
    the model-agreement gate and sort.  It never enumerates a virtual library
    and normally finishes in a few seconds.  If no cached prediction table is
    available, ``--fit-compact`` fits the compact member panel first (slower).

Route (2): ``enumerate --dataset <ALIAS> --max-combos N --allow-virtual``
    Enumerate at most ``N`` virtual reagent combinations (default 20,000,000, the whole library),
    score each with the ensemble and compute expected improvement.  This route
    is GATED: without ``--allow-virtual`` it prints an explanation and exits
    with status 3 before touching the data.  ``--max-combos`` is a hard cap --
    the module never falls back to v16's ``MAX_VIRTUAL_COMBINATIONS`` (1e9) or
    to ``build_virtual_library`` (which materialises the whole library).

``--dry-run`` (either subcommand) only loads the raw dataset and reports the
search space, the implied full combination count, the estimated cost and the
exact command that would execute the run.  It enumerates nothing and writes no
files.

The frozen legacy pipeline is no longer loaded; this module
never edits ``archive/Analysis_all_v16.py``.  Raw datasets are not
redistributed with the repository: point ``--data-dir`` at the directory that
holds ``<source>.csv`` and ``col_mappings.csv``.  If either is missing the
script prints the exact paths it needs and exits 0 without fabricating data.

Reuse / deviation notes (see the final report for the measured versions)
--------------------------------------------------------------------------
* ``get_enumeration_spec``, ``get_loading_grid``, ``get_representative_conditions``,
  ``build_virtual_feature_frame``, ``predict_in_batches``,
  ``calculate_expected_improvement``, ``resolve_ei_incumbent`` and
  ``audit_condition_scores`` are used unchanged.
* ``generate_figure6_direct_ranking`` IS usable standalone when the
  ``Predicted_Yield_<member>`` columns are present; its fallback recomputes
  features only when they are absent.  It writes a PNG and two CSVs into
  ``outputs/recommend_real_<alias>/``; the ranked table is then written to
  ``results/recommend_real_<alias>.csv``.
* ``build_virtual_library`` is NOT used: it materialises the full enumeration
  in memory (and ``preview_only`` hard-codes 200,000 rows).
* ``generate_bo_results_streaming`` is NOT used: its ``max_combinations`` only
  sizes the loading axis -- ``stream_virtual_library`` always yields every
  categorical combination -- so it cannot honour an "at most N" cap when
  N < n_categorical.  The capped chunk iterator below replaces it while still
  using v16's feature builder, batched predictor and expected improvement.
* ``train_and_evaluate_models`` is NOT used: it fits all seven members with
  5-fold CV, a GroupKFold and two learning-curve sweeps per model (minutes),
  and its model constructors are local variables.  ``_fit_compact_ensemble``
  rebuilds the same estimator configurations for a five-member panel and keeps
  the best four by held-out R^2, using the pipeline's own weighting rule.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

# Importing the frozen loader must not leave new bytecode files in the repo.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
# The repo root must be importable too: without it ``import hte`` fails and the
# script silently falls back to the legacy v16 column split (catalyst,
# leaving_group and substrate_type then come out 'Unknown' on files whose slots
# only the packaged table binds).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hte.rank_real import generate_figure6_direct_ranking  # noqa: E402
from hte.virtual import (
    get_enumeration_spec,
    get_loading_grid,
    get_representative_conditions,
    build_virtual_feature_frame,
    predict_in_batches,
    calculate_expected_improvement,
    resolve_ei_incumbent,
    audit_condition_scores,
    loading_axis_available,
    N_LOADING_POINTS,
    ENSEMBLE_MIN_WEIGHT_R2,
    ENSEMBLE_WEIGHT_POWER,
    ENSEMBLE_MAX_MODEL_GAP,
    PREDICTION_MIN,
    PREDICTION_MAX,
    PREDICT_BATCH_ROWS,
    RF_N_ESTIMATORS,
    HISTGB_MAX_ITER,
)  # noqa: E402


from hte.v16_core import load_real_dataset as _load_real_dataset  # noqa: E402

# The frozen loader prints Chinese column-mapping lines; make the console
# encoding explicit so a cp936 terminal cannot turn a successful run into a
# UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Force a non-interactive backend *before* v16 imports pyplot: Figure 6 is
# rendered at 300 dpi and must not need a display.
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402


DATA_YAML = ROOT / "data" / "alias_mapping.yaml"
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"

# Raw data are not redistributed; this is the project-local location named by
# the paper's run notes.  Override with --data-dir on any other machine.
DEFAULT_DATA_DIR = ROOT / "data"

# v16 ENUM_AXES: the seven categorical slots that can form the Cartesian
# product.  The remaining three of alias_mapping.yaml's ten slots are process
# variables: temperature/time are FIXED at the representative values and
# catalyst_loading is swept on a grid (or collapsed to one point when the
# dataset has no usable amount axis).
ENUM_AXES = ["metal", "catalyst", "ligand", "base", "solvent",
             "reactant_2", "reactant_1"]
PROCESS_AXES = ["temperature", "time", "catalyst_loading"]

# Members named by v16's hard-coded Figure-6 PRED_COLS map.  HistGB-NativeCat
# and MLP-OneHot exist in the prediction table but cannot enter that fast path.
FIG6_MEMBERS = ["XGBoost", "Gradient Boosting", "Random Forest",
                "Multilayer Perceptron"]

# Compact panel fitted by --fit-compact / enumerate.  HistGB is included
# because it is a v16 member and is often among the top four; it simply cannot
# be consumed by Figure 6's PRED_COLS map.
FIT_MEMBERS = FIG6_MEMBERS + ["HistGB-NativeCat"]

# Order-of-magnitude cost model for --dry-run.  Both constants are measured on
# this machine with the four-member tree panel and are deliberately
# conservative for other datasets.
ASSUMED_SCORING_ROWS_PER_S = 120_000.0
MATERIALISED_BYTES_PER_ROW = 135.0   # ~30 B categorical frame + ~105 B features
FEATURE_BYTES_PER_ROW = 105.0


# small helpers
def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _fmt_secs(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    if seconds < 120:
        return "%.1f s" % seconds
    if seconds < 7200:
        return "%.1f min" % (seconds / 60.0)
    return "%.1f h" % (seconds / 3600.0)


def _peak_working_set_mb() -> float | None:
    """Peak process working set on Windows (None when unavailable)."""
    try:
        import ctypes
        import ctypes.wintypes as wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_mem = psapi.GetProcessMemoryInfo
        get_mem.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        get_mem.restype = wintypes.BOOL
        counters = _PMC()
        counters.cb = ctypes.sizeof(counters)
        if not get_mem(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                       counters.cb):
            return None
        return float(counters.PeakWorkingSetSize) / 1024.0 ** 2
    except Exception:
        return None


def _command(args: argparse.Namespace) -> str:
    """The exact command line that would repeat this run."""
    argv = [sys.executable, str(Path(__file__).resolve())]
    if args.command == "rank-real":
        argv += ["rank-real", "--dataset", args.dataset]
        if args.fit_compact:
            argv.append("--fit-compact")
        if args.predictions:
            argv += ["--predictions", str(args.predictions)]
    else:
        argv += ["enumerate", "--dataset", args.dataset,
                 "--max-combos", str(args.max_combos), "--allow-virtual"]
    if args.data_dir:
        argv += ["--data-dir", str(args.data_dir)]
    if args.command == "rank-real" and args.outputs_dir:
        argv += ["--outputs-dir", str(args.outputs_dir)]
    return subprocess.list2cmdline(argv)


# dataset / alias resolution
def _load_mapping() -> dict:
    if not DATA_YAML.is_file():
        raise SystemExit("alias mapping not found: %s" % DATA_YAML)
    with DATA_YAML.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _dataset_stem(mapping: dict, alias: str) -> str:
    datasets = mapping.get("datasets") or {}
    if alias not in datasets:
        known = ", ".join(sorted(datasets))
        raise SystemExit("unknown dataset alias %r.  Known aliases: %s"
                         % (alias, known))
    return str(datasets[alias]["source"])


def _resolve_data_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    for candidate in (DEFAULT_DATA_DIR, ROOT / "Data", ROOT / "data"):
        if candidate.is_dir():
            return candidate
    return DEFAULT_DATA_DIR


def _check_inputs(data_dir: Path, stem: str) -> bool:
    """Print the exact missing paths and return False (caller exits 0)."""
    needed = [data_dir / (stem + ".csv"), data_dir / "col_mappings.csv"]
    missing = [p for p in needed if not p.is_file()]
    if not missing:
        return True
    print("[missing] raw input(s) not found:", file=sys.stderr)
    for path in needed:
        print("  %s%s" % (path, "" if path.is_file() else "   <-- MISSING"),
              file=sys.stderr)
    print("\nRaw datasets are not redistributed with this repository.  Put the "
          "files above in place (or pass --data-dir) and re-run.  No data were "
          "fabricated and nothing was written.", file=sys.stderr)
    return False


def _load_dataset(ns: dict, data_dir: Path, stem: str):
    return _load_real_dataset(str(data_dir / (stem + ".csv")), str(data_dir))


# enumeration space
_LAST_ENCODERS: dict | None = None


def _feature_cols_from_frame(df: pd.DataFrame) -> list[str]:
    """The feature names ``prepare_features`` would produce for this frame.

    Used by --dry-run so the combination count can be reported without fitting
    anything.  The rule mirrors prepare_features: every categorical slot that
    exists is encoded; a numerical slot enters only when it varies.
    """
    # Ask the packaged builder itself, so that the plan and the fit see exactly the
    # same columns (the local rule diverged: it kept slots the builder drops).
    import contextlib
    import io
    from hte.v16_core import (get_split_positions as _gsp,
                              prepare_features as _pf)
    tr, _ = _gsp(len(df))
    with contextlib.redirect_stdout(io.StringIO()):
        _X, _y, cols, _disp, enc, _sc = _pf(df, None, fit_rows=tr)
    global _LAST_ENCODERS
    _LAST_ENCODERS = enc
    return [str(c) for c in cols]


def _enumeration_plan(ns: dict, df: pd.DataFrame, cap: int) -> dict:
    """Resolve the enumerable space and the exact rows a cap would score."""
    feat_cols = _feature_cols_from_frame(df)
    cat_cols, value_lists, n_cat, inner = get_enumeration_spec(
        df, feat_cols)
    # A level the encoder never saw cannot be scored (the transform raises), so the
    # enumerable space is the intersection with the encoder's classes.  Rare levels
    # have already been folded into OTHER by the packaged feature builder.
    if _LAST_ENCODERS:
        kept_lists = []
        for slot, levels in zip(cat_cols, value_lists):
            le = _LAST_ENCODERS.get(slot) if hasattr(_LAST_ENCODERS, "get") else None
            if le is not None and hasattr(le, "classes_"):
                allowed = {str(x) for x in le.classes_}
                kept = [x for x in levels if str(x) in allowed]
                kept_lists.append(kept if kept else list(levels))
            else:
                kept_lists.append(list(levels))
        value_lists = kept_lists
    n_cat = int(np.prod([len(v) for v in value_lists])) if value_lists else 0
    inner = [int(np.prod([len(v) for v in value_lists[i + 1:]]))
             for i in range(len(value_lists))]

    grid_full = np.asarray(
        get_loading_grid(df, n_points=N_LOADING_POINTS),
        dtype=float)
    if grid_full.size == 0:
        grid_full = np.array([0.0], dtype=float)

    full_rows = int(n_cat) * int(grid_full.size)
    cap = int(cap)
    if cap <= 0:
        raise ValueError("--max-combos must be a positive integer")
    effective_cap = min(cap, full_rows) if full_rows else 0

    if effective_cap >= n_cat:
        # Every categorical combination is covered; spend the remainder on the
        # loading axis, exactly as v16's build_virtual_library / streamer do.
        n_load = max(1, min(int(grid_full.size), effective_cap // max(n_cat, 1)))
        mode = "full-product"
    else:
        # The cap cannot cover the categorical product: take one loading point
        # (the historical median) and sample the product in index space.
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
        "feat_cols": feat_cols,
        "cat_cols": list(cat_cols),
        "value_lists": [list(v) for v in value_lists],
        "inner": [int(v) for v in inner],
        "n_cat": n_cat,
        "grid_full": grid_full,
        "load_values": load_values,
        "n_load": int(load_values.size),
        "full_rows": full_rows,
        "cap": cap,
        "effective_cap": effective_cap,
        "rows": rows,
        "mode": mode,
    }


def _iter_plan_chunks(ns: dict, df: pd.DataFrame, plan: dict,
                      chunk_rows: int):
    """Yield bounded DataFrames covering exactly ``plan['rows']`` combinations."""
    representative = get_representative_conditions(df)
    # The representative row supplies the held-fixed slots; the enumerated axes must
    # be added explicitly, otherwise they are silently dropped from the output
    # (the table used to show ligand/base/solvent only, with no reactant pair).
    out_cols = list(representative.keys()) + [c for c in plan["cat_cols"]
                                              if c not in representative]
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
            load_idx = np.tile(
                np.arange(n_load, dtype=np.int64), stop - start)
            yield _build(combo_idx, load_idx)
    else:
        # Uniform stride through the flat product index: every level of the
        # fast-varying axes and a spread of the slow-varying ones is covered,
        # unlike taking the first N rows of the nested product.
        total = int(plan["rows"])
        combo_idx = np.linspace(0, max(n_cat - 1, 0), total).round().astype(
            np.int64)
        zeros = np.zeros(total, dtype=np.int64)
        for start in range(0, total, chunk_rows):
            stop = min(start + chunk_rows, total)
            yield _build(combo_idx[start:stop], zeros[start:stop])


# compact ensemble (v16 hyperparameters, no CV / learning curves)
def _build_member_models(ns: dict, feat_cols: list[str]) -> dict:
    """Estimator panel with the configurations used by v16's train function.

    ``train_and_evaluate_models`` defines these as local variables inside a
    function that also runs 5-fold CV, GroupKFold and two learning-curve sweeps
    per member, so it cannot be called for a seconds-scale recommendation.
    The literals below are copied from the frozen file's model dictionary.
    """
    import xgboost as xgb
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.neural_network import MLPRegressor

    cat_idx = [i for i, c in enumerate(feat_cols)
               if str(c).endswith("_encoded")]
    n_jobs = min(4, os.cpu_count() or 1)
    return {
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=n_jobs),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=int(RF_N_ESTIMATORS), max_depth=15,
            min_samples_split=5, random_state=42, n_jobs=n_jobs),
        "HistGB-NativeCat": HistGradientBoostingRegressor(
            max_iter=int(HISTGB_MAX_ITER), max_depth=6,
            learning_rate=0.1, random_state=42,
            categorical_features=(cat_idx if cat_idx else None)),
        "Multilayer Perceptron": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation="relu", max_iter=500,
            random_state=42, early_stopping=True),
    }


def _fit_compact_ensemble(ns: dict, df: pd.DataFrame, raw: dict | None,
                          verbose: bool = True) -> dict:
    """Fit the five-member panel and report held-out R^2 per member."""
    from sklearn.metrics import r2_score

    # Use the packaged feature builder, not the monolith's: the latter still reads
    # the retired slot names and silently drops both reactant slots (it fitted at
    # R2 0.48 instead of 0.74 and enumerated reactant levels the model never saw).
    from hte.v16_core import (get_split_positions as _gsp,
                              prepare_features as _pf)
    train_idx, test_idx = _gsp(len(df))
    X, y, feat_cols, _display, encoders, scaler = _pf(
        df, raw, fit_rows=train_idx)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    models = _build_member_models(ns, feat_cols)
    metrics = []
    started = time.perf_counter()
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        try:
            r2 = float(r2_score(y_test, model.predict(X_test)))
        except Exception:
            r2 = float("nan")
        metrics.append({"name": name, "r2": r2, "fit_s": fit_s})
        if verbose:
            print("  [fit] %-24s %5.1f s   held-out R2 = %s"
                  % (name, fit_s,
                     "nan" if not np.isfinite(r2) else "%.4f" % r2))
    return {
        "models": models,
        "metrics": metrics,
        "X": X,
        "y": y,
        "feat_cols": feat_cols,
        "encoders": encoders,
        "scaler": scaler,
        "fit_seconds": time.perf_counter() - started,
    }


def _select_members(ns: dict, r2_map: dict[str, float], top_k: int = 4,
                    allowed: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Rank members by held-out R^2 and apply v16's ensemble-weight rule."""
    floor = float(ENSEMBLE_MIN_WEIGHT_R2)
    power = float(ENSEMBLE_WEIGHT_POWER)
    candidates = list(allowed or r2_map)
    names = [n for n in candidates
             if n in r2_map and np.isfinite(float(r2_map[n]))]
    if names:
        scores = np.array([float(r2_map[n]) for n in names], dtype=float)
    else:
        # Every held-out R^2 is undefined (tiny test split): keep the requested
        # members and give each the floor weight rather than emitting NaN.
        names = [n for n in candidates if n in r2_map]
        if not names:
            raise ValueError("no ensemble member has a usable held-out R^2")
        scores = np.full(len(names), floor, dtype=float)
    order = np.argsort(-scores, kind="stable")
    names = [names[i] for i in order[:max(1, int(top_k))]]
    scores = scores[order[:max(1, int(top_k))]]
    raw_w = np.power(np.maximum(scores, floor), power)
    return names, raw_w / raw_w.sum()


# cached prediction tables (route 1's cheap input)
def _prediction_roots(args, data_dir: Path) -> list[Path]:
    roots = []
    if args.outputs_dir:
        roots.append(Path(args.outputs_dir).expanduser())
    roots += [data_dir.parent / "outputs", OUTPUTS_DIR]
    seen, out = set(), []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        if root.is_dir():
            out.append(root)
    return out


def _find_prediction_table(args, data_dir: Path, stem: str):
    """Latest ``model_predictions_all.csv`` whose run directory names the stem."""
    candidates = []
    for root in _prediction_roots(args, data_dir):
        for pattern in ("*/model_predictions_all.csv",
                        "model_predictions_all.csv"):
            for path in root.glob(pattern):
                run_name = path.parent.name
                if run_name == stem or run_name.endswith(stem) or stem in run_name:
                    candidates.append(path)
    if not candidates:
        return None, None
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    return path, path.parent


def _load_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "model_evaluation_metrics.csv"
    if not path.is_file():
        return {}
    try:
        table = pd.read_csv(path)
        if "Model" not in table.columns or "R² (Test)" not in table.columns:
            return {}
        return {str(r["Model"]): float(r["R² (Test)"]) for _, r in table.iterrows()}
    except Exception:
        return {}


def _read_prediction_columns(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _align_predictions(pred: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Return ``pred`` in ``df``'s row order, or raise a clear error."""
    if len(pred) != len(df):
        raise ValueError("prediction table has %d rows but the dataset has %d"
                         % (len(pred), len(df)))
    if "yield" in pred.columns and "yield" in df.columns:
        a = pd.to_numeric(pred["yield"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(df["yield"], errors="coerce").to_numpy(float)
        if np.allclose(a, b, rtol=0.0, atol=1e-3, equal_nan=True):
            return pred.reset_index(drop=True)

    # Different row order: align on the full condition key when both sides
    # contain exactly the same multiset of conditions.
    key_cols = [c for c in ENUM_AXES + PROCESS_AXES
                if c in pred.columns and c in df.columns]
    if key_cols:
        pkey = pred[key_cols].astype(str).agg("|".join, axis=1)
        dkey = df[key_cols].astype(str).agg("|".join, axis=1)
        if pkey.is_unique and dkey.is_unique and set(pkey) == set(dkey):
            order = dkey.map({k: i for i, k in enumerate(pkey)})
            if not order.isna().any():
                return pred.iloc[order.to_numpy(dtype=int)].reset_index(drop=True)
    raise ValueError("prediction table does not belong to this dataset "
                     "(row count or condition keys differ)")


# route 1: rank real measured records
def _ranked_output_table(ranked: pd.DataFrame, alias: str,
                         members: list[str]) -> pd.DataFrame:
    cols = ["Rank"] + ENUM_AXES + ["temperature", "time", "catalyst_loading"]
    cols += ["Pred_" + n for n in members]
    cols += ["Ensemble Predicted Yield (%)", "Prediction Uncertainty (%)",
             "Model Gap (%)", "Model Min (%)", "Model Max (%)", "Model Preset",
             "yield", "N Runs", "Yield SD (pp)", "Split",
             "Prediction Error (%)", "Data Status", "Support Count"]
    cols = [c for c in cols if c in ranked.columns]
    table = ranked[cols].copy()
    if "yield" in table.columns:
        table = table.rename(columns={"yield": "Measured Yield (%)"})
    table.insert(0, "Dataset", alias)
    return table


def _print_top(table: pd.DataFrame, n: int = 5) -> None:
    wanted = ["Rank", "catalyst", "ligand", "base", "solvent", "leaving_group",
              "substrate_type", "catalyst_loading",
              "Ensemble Predicted Yield (%)", "Measured Yield (%)"]
    cols = [c for c in wanted if c in table.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.head(int(n))[cols].to_string(index=False))


def _route1_dry_run(ns: dict, args, data_dir: Path, stem: str,
                    df: pd.DataFrame) -> int:
    print("[rank-real] dry run -- no ranking is performed")
    print("  dataset      : %s  (source %r)" % (args.dataset, stem))
    print("  data file    : %s" % (data_dir / (stem + ".csv")))
    print("  measured rows: %s" % f"{len(df):,}")
    print("  route        : real measured records; no virtual enumeration")

    if args.predictions:
        pred_path = Path(args.predictions).expanduser()
    elif args.fit_compact:
        pred_path = None
        print("  predictions  : ignored (--fit-compact fits the panel locally)")
    else:
        pred_path, _ = _find_prediction_table(args, data_dir, stem)
    if pred_path is None:
        if not args.fit_compact:
            print("  predictions  : NOT FOUND under %s"
                  % ", ".join(_rel(p) for p in _prediction_roots(args, data_dir)))
        else:
            print("  members      : fitted locally by --fit-compact "
                  "(held-out R^2 weights)")
    elif not pred_path.is_file():
        print("  predictions  : NOT FOUND: %s" % pred_path)
    else:
        print("  predictions  : %s" % _rel(pred_path))
        columns = _read_prediction_columns(pred_path)
        available = [n for n in FIG6_MEMBERS
                     if "Predicted_Yield_" + n in columns]
        metrics = _load_metrics(pred_path.parent)
        if metrics and available:
            try:
                members, weights = _select_members(
                    ns, {n: metrics.get(n, float("nan")) for n in available},
                    top_k=4, allowed=available)
                print("  members      : %s"
                      % ", ".join("%s (w=%.4f)" % (n, w)
                                  for n, w in zip(members, weights)))
            except Exception:
                print("  members      : %s (unweighted)"
                      % ", ".join(available))
        else:
            print("  members      : %s (no metrics -> unweighted)"
                  % ", ".join(available))

    print("  output       : %s"
          % _rel(RESULTS_DIR / ("recommend_real_%s.csv" % args.dataset)))
    print("  command      : %s" % _command(args))
    return 0


def cmd_rank_real(ns: dict, args) -> int:
    mapping = _load_mapping()
    stem = _dataset_stem(mapping, args.dataset)
    data_dir = _resolve_data_dir(args.data_dir)
    if not _check_inputs(data_dir, stem):
        return 0
    df, raw = _load_dataset(ns, data_dir, stem)

    if args.dry_run:
        return _route1_dry_run(ns, args, data_dir, stem, df)

    t_start = time.perf_counter()
    pred_path, run_dir = (None, None)
    if args.predictions:
        pred_path = Path(args.predictions).expanduser()
        run_dir = pred_path.parent
        if not pred_path.is_file():
            print("[rank-real] prediction table not found: %s" % pred_path,
                  file=sys.stderr)
            return 0
    elif args.fit_compact:
        # --fit-compact explicitly asks to fit here, so do not silently
        # substitute a cached prediction table.
        pred_path, run_dir = None, None
    else:
        pred_path, run_dir = _find_prediction_table(args, data_dir, stem)

    panel = None
    if pred_path is not None:
        try:
            pred = _align_predictions(pd.read_csv(pred_path), df)
        except Exception as exc:
            print("[rank-real] cannot use %s: %s"
                  % (_rel(pred_path), exc), file=sys.stderr)
            return 0
        available = [n for n in FIG6_MEMBERS
                     if "Predicted_Yield_" + n in pred.columns]
        if len(available) < 2:
            print("[rank-real] %s has fewer than two usable "
                  "Predicted_Yield_* columns" % _rel(pred_path),
                  file=sys.stderr)
            return 0
        metrics = _load_metrics(run_dir)
        r2_map = {n: metrics.get(n, float("nan")) for n in available}
        if all(not np.isfinite(v) for v in r2_map.values()):
            members, weights = available, None
            print("[rank-real] no model metrics next to the prediction table; "
                  "using an unweighted mean of %s" % ", ".join(members))
        else:
            members, weights = _select_members(ns, r2_map, top_k=4,
                                               allowed=available)
            if len(members) < 2:
                print("[rank-real] only one ensemble member has a usable "
                      "held-out R^2; route (1) needs at least two.",
                      file=sys.stderr)
                return 0
            print("[rank-real] ensemble: %s"
                  % ", ".join("%s (w=%.4f)" % (n, w)
                              for n, w in zip(members, weights)))
        rank_df = df.copy()
        for name in members:
            values = pd.to_numeric(pred["Predicted_Yield_" + name],
                                   errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                print("[rank-real] prediction column %r contains non-finite "
                      "values" % name, file=sys.stderr)
                return 0
            rank_df["Predicted_Yield_" + name] = values
        print("[rank-real] using cached predictions: %s" % _rel(pred_path))
    else:
        if not args.fit_compact:
            print("[rank-real] no cached model_predictions_all.csv for %r."
                  % args.dataset, file=sys.stderr)
            print("[rank-real] route (1) consumes the prediction table written "
                  "by a pipeline run; pass --predictions <csv>, or "
                  "--fit-compact to fit the compact member panel here.",
                  file=sys.stderr)
            return 0
        panel = _fit_compact_ensemble(ns, df, raw, verbose=True)
        r2_map = {m["name"]: m["r2"] for m in panel["metrics"]}
        members, weights = _select_members(ns, r2_map, top_k=4,
                                           allowed=FIG6_MEMBERS)
        if len(members) < 2:
            print("[rank-real] the compact panel produced fewer than two "
                  "usable members; nothing written.", file=sys.stderr)
            return 0
        rank_df = df.copy()
        for name in members:
            values = np.clip(
                np.asarray(panel["models"][name].predict(panel["X"]),
                           dtype=float),
                float(PREDICTION_MIN), float(PREDICTION_MAX))
            rank_df["Predicted_Yield_" + name] = values
        print("[rank-real] compact ensemble: %s"
              % ", ".join("%s (w=%.4f)" % (n, w)
                          for n, w in zip(members, weights)))

    out_dir = OUTPUTS_DIR / ("recommend_real_%s" % args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    gap = (float(ENSEMBLE_MAX_MODEL_GAP)
           if args.max_model_gap is None else float(args.max_model_gap))

    t_rank = time.perf_counter()
    top_n = max(1, int(args.top_n))
    ranked = generate_figure6_direct_ranking(
        df=rank_df,
        output_dir=str(out_dir),
        best_models_dict=None,
        top_n_table=top_n,
        top_csv_n=max(500, top_n),
        max_model_gap=gap,
        min_support_count=0,
        weights=weights,
        member_names=members,
    )
    rank_s = time.perf_counter() - t_rank
    if ranked is None or len(ranked) == 0:
        print("[rank-real] the model-gap gate removed every condition; "
              "nothing written (raise --max-model-gap to keep more).")
        return 0

    # When the compact panel was fitted here, re-score the returned conditions
    # with v16's own audit -- the guard against scores attached to the wrong row.
    audit_note = ""
    if panel is not None:
        audit_models = {n: panel["models"][n] for n in members}
        ranked_a, n_bad, n_total = audit_condition_scores(
            ranked, df, audit_models, panel["feat_cols"],
            panel["encoders"], panel["scaler"], weights=weights,
            tag="rank-real")
        audit_note = "  audit: %d rows re-scored, %d dropped" % (
            n_total, n_bad)
        ranked = ranked_a
        if len(ranked) == 0:
            print("[rank-real] the score audit removed every row; nothing "
                  "written.")
            return 0

    table = _ranked_output_table(ranked, args.dataset, members)
    out_path = RESULTS_DIR / ("recommend_real_%s.csv" % args.dataset)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    elapsed = time.perf_counter() - t_start

    print("[rank-real] %s -> %s" % (args.dataset, _rel(out_path)))
    print("[rank-real] input records: %s; ranked conditions written: %s"
          % (f"{len(df):,}", f"{len(table):,}"))
    print("[rank-real] ensemble members: %s" % ", ".join(members))
    print("[rank-real] ranking step: %.2f s; total wall time: %.2f s%s"
          % (rank_s, elapsed, audit_note))
    print("[rank-real] top-5 preview:")
    _print_top(table, 5)
    return 0


# route 2: capped virtual enumeration
def _route2_dry_run(ns: dict, args, data_dir: Path, stem: str,
                    df: pd.DataFrame, plan: dict) -> int:
    mapping = _load_mapping()
    slots = list(mapping.get("slots") or [])
    print("[enumerate] dry run -- nothing is enumerated, scored or written")
    print("  dataset      : %s  (source %r)" % (args.dataset, stem))
    print("  data file    : %s" % (data_dir / (stem + ".csv")))
    print("  measured rows: %s" % f"{len(df):,}")
    print("  alias slots  : %d total -> %d categorical, %d process"
          % (len(slots),
             len([s for s in slots if (s if s != "metal_center" else "metal")
                  in ENUM_AXES]),
             len([s for s in slots if s in PROCESS_AXES])))
    print("  categorical axes (values per slot):")
    for col, values in zip(plan["cat_cols"], plan["value_lists"]):
        print("    %-16s %6d" % (col, len(values)))
    factors = " x ".join(str(len(v)) for v in plan["value_lists"]) or "1"
    print("  categorical product: %s = %s" % (factors, f"{plan['n_cat']:,}"))
    if loading_axis_available():
        print("  loading grid : %d points (%g .. %g eq.)"
              % (len(plan["grid_full"]),
                 float(plan["grid_full"].min()),
                 float(plan["grid_full"].max())))
    else:
        print("  loading grid : none (no usable amount axis; collapsed to 1)")
    print("  FULL library : %s combinations x %d loading point(s) = %s rows"
          % (f"{plan['n_cat']:,}", len(plan["grid_full"]),
             f"{plan['full_rows']:,}"))

    print("  requested cap: --max-combos %s" % f"{plan['cap']:,}")
    if plan["cap"] > plan["full_rows"]:
        print("    -> capped at the full library (%s rows)" % f"{plan['full_rows']:,}")
    if plan["mode"] == "sampled":
        coverage = 100.0 * plan["rows"] / max(plan["n_cat"], 1)
        print("    -> cap < categorical product: 1 loading point "
              "(historical median %g eq.), %s of %s reagent combinations (%.2f%%)"
              % (float(plan["load_values"][0]), f"{plan['rows']:,}",
                 f"{plan['n_cat']:,}", coverage))
    else:
        print("    -> full categorical coverage with %d loading point(s); "
              "%s rows" % (plan["n_load"], f"{plan['rows']:,}"))

    fit_s = 5.0 + 1e-4 * len(df)
    score_s = plan["rows"] / ASSUMED_SCORING_ROWS_PER_S
    chunk_rows = int(args.chunk_rows)
    # ~350 MB is the measured interpreter + fitted-panel baseline; the chunk
    # adds one float64 feature matrix and the member prediction vectors.
    stream_mb = (chunk_rows * FEATURE_BYTES_PER_ROW / 1024.0 ** 2) + 350.0
    print("  estimated cost for this run: model fit ~%s + scoring ~%s = ~%s"
          % (_fmt_secs(fit_s), _fmt_secs(score_s), _fmt_secs(fit_s + score_s)))
    print("    streaming peak ~%.0f MB with --chunk-rows %s (interpreter + "
          "models + feature matrix + member predictions)"
          % (stream_mb, f"{chunk_rows:,}"))
    full_s = plan["full_rows"] / ASSUMED_SCORING_ROWS_PER_S + fit_s
    full_gib = plan["full_rows"] * MATERIALISED_BYTES_PER_ROW / 1024.0 ** 3
    print("  estimated cost for the FULL library: ~%s single process; "
          "~%.1f GiB if materialised (it is streamed, not materialised)"
          % (_fmt_secs(full_s), full_gib))
    print("  basis        : %.0f scored rows/s measured on this machine with "
          "the four-member tree panel" % ASSUMED_SCORING_ROWS_PER_S)
    print("  output       : %s"
          % _rel(RESULTS_DIR / ("recommend_virtual_%s.csv" % args.dataset)))
    print("  command      : %s" % _command(args))
    return 0


def cmd_enumerate(ns: dict, args) -> int:
    if not args.allow_virtual and not args.dry_run:
        print("[enumerate] REFUSED: route (2) enumerates a virtual library and "
              "is never run implicitly.", file=sys.stderr)
        print("[enumerate] Re-run with --allow-virtual and a bounded "
              "--max-combos (default 20000000).", file=sys.stderr)
        print("[enumerate] Use --dry-run to inspect the search space and cost "
              "without enumerating anything.", file=sys.stderr)
        return 3
    if int(args.max_combos) <= 0:
        print("[enumerate] --max-combos must be a positive integer",
              file=sys.stderr)
        return 2

    mapping = _load_mapping()
    stem = _dataset_stem(mapping, args.dataset)
    data_dir = _resolve_data_dir(args.data_dir)
    if not _check_inputs(data_dir, stem):
        return 0
    df, raw = _load_dataset(ns, data_dir, stem)
    plan = _enumeration_plan(ns, df, int(args.max_combos))

    if args.dry_run:
        return _route2_dry_run(ns, args, data_dir, stem, df, plan)

    t_start = time.perf_counter()
    tracemalloc.start()
    print("[enumerate] %s: %s of %s full-library rows (cap %s)"
          % (args.dataset, f"{plan['rows']:,}", f"{plan['full_rows']:,}",
             f"{plan['effective_cap']:,}"))
    panel = _fit_compact_ensemble(ns, df, raw, verbose=True)
    if list(plan["feat_cols"]) != list(panel["feat_cols"]):
        # The enumeration spec was resolved before fitting; a mismatch would
        # silently change which axes the product spans.
        print("[enumerate] ERROR: feature columns changed between the "
              "enumeration plan and prepare_features:\n  plan: %s\n  fit : %s"
              % (plan["feat_cols"], panel["feat_cols"]), file=sys.stderr)
        return 1
    r2_map = {m["name"]: m["r2"] for m in panel["metrics"]}
    members, weights = _select_members(ns, r2_map, top_k=4)
    models = {n: panel["models"][n] for n in members}
    print("[enumerate] ensemble: %s"
          % ", ".join("%s (w=%.4f)" % (n, w)
                      for n, w in zip(members, weights)))

    incumbent, record = resolve_ei_incumbent(df)
    print("[enumerate] EI incumbent f* = %.2f%% (all-time record %.2f%%)"
          % (float(incumbent), float(record)))
    print("[enumerate] NOTE: temperature/time are FIXED at the representative "
          "values; catalyst_loading is the only process axis swept.  The "
          "library cannot rank across temperature or time (v16 behaviour).")
    gap_thr = float(ENSEMBLE_MAX_MODEL_GAP)
    keep_top = max(1, int(args.keep_top))
    chunk_rows = max(1, int(args.chunk_rows))
    wsumsq = float((weights ** 2).sum())

    t_score = time.perf_counter()
    buffered, n_scored, n_chunks = [], 0, 0
    for chunk in _iter_plan_chunks(ns, df, plan, chunk_rows):
        n = len(chunk)
        feature = build_virtual_feature_frame(
            df, panel["feat_cols"], panel["encoders"], panel["scaler"], chunk)
        total = np.zeros(n, dtype=np.float64)
        total_sq = np.zeros(n, dtype=np.float64)
        vmin = np.full(n, np.inf, dtype=np.float64)
        vmax = np.full(n, -np.inf, dtype=np.float64)
        predictions = {}
        for j, (name, model) in enumerate(models.items()):
            p = np.asarray(predict_in_batches(
                model, feature, batch_size=int(PREDICT_BATCH_ROWS),
                desc="predict " + name, show_progress=False), dtype=np.float64)
            p = np.clip(p, float(PREDICTION_MIN),
                        float(PREDICTION_MAX))
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
            mean, std, float(incumbent), xi=0.01)
        chunk["UCB (%)"] = mean + 1.96 * std
        chunk["Passes Ensemble Gate"] = gap <= gap_thr
        for name in models:
            chunk["Pred_" + name] = predictions[name]

        n_scored += n
        n_chunks += 1
        buffered.append(chunk.nlargest(min(keep_top, n),
                                       "Expected Improvement"))
        if len(buffered) > 8:
            pooled = pd.concat(buffered, ignore_index=True)
            buffered = [pooled.nlargest(keep_top, "Expected Improvement")]
        del feature, predictions, total, total_sq, vmin, vmax, mean, std
    score_s = time.perf_counter() - t_score

    if not buffered:
        print("[enumerate] no rows were scored; nothing written.")
        return 0
    top = pd.concat(buffered, ignore_index=True)
    top = top.nlargest(min(keep_top, len(top)),
                       "Expected Improvement").reset_index(drop=True)

    # v16 guard: re-score the conditions that survived and drop any row whose
    # stored ensemble value does not reproduce.
    top, n_bad, n_audit = audit_condition_scores(
        top, df, models, panel["feat_cols"], panel["encoders"],
        panel["scaler"], weights=weights, tag="enumerate")
    if len(top) == 0:
        print("[enumerate] the score audit removed every row; nothing written.")
        return 0
    top = top.sort_values("Expected Improvement", ascending=False).reset_index(
        drop=True)
    top.insert(0, "Rank", np.arange(1, len(top) + 1))

    out_cols = ["Rank"] + ENUM_AXES + ["temperature", "time",
                                       "catalyst_loading"]
    out_cols += ["Pred_" + n for n in members]
    out_cols += ["Ensemble Predicted Yield (%)", "Prediction Uncertainty (%)",
                 "Model Gap (%)", "Model Min (%)", "Model Max (%)",
                 "Expected Improvement", "UCB (%)", "Passes Ensemble Gate"]
    out_cols = [c for c in out_cols if c in top.columns]
    table = top[out_cols].copy()
    table.insert(0, "Dataset", args.dataset)
    table.insert(2, "EI Incumbent (%)", float(incumbent))

    out_path = RESULTS_DIR / ("recommend_virtual_%s.csv" % args.dataset)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    elapsed = time.perf_counter() - t_start
    traced_peak = tracemalloc.get_traced_memory()[1] / 1024.0 ** 2
    tracemalloc.stop()
    peak_ws = _peak_working_set_mb()

    print("[enumerate] rows scored: %s in %d chunk(s)" % (f"{n_scored:,}", n_chunks))
    print("[enumerate] timing: model fit %.2f s; scoring %.2f s; total %.2f s"
          % (panel["fit_seconds"], score_s, elapsed))
    print("[enumerate] throughput: %.0f rows/s (scoring only)"
          % (n_scored / max(score_s, 1e-9)))
    print("[enumerate] peak memory: %.1f MiB Python heap%s"
          % (traced_peak,
             "" if peak_ws is None
             else "; %.1f MiB process working set" % peak_ws))
    print("[enumerate] audit: %d rows re-scored, %d dropped"
          % (n_audit, n_bad))
    n_pass_gate = (int(table["Passes Ensemble Gate"].sum())
                   if "Passes Ensemble Gate" in table.columns else 0)
    print("[enumerate] ensemble gate (Model Gap <= %g pp): %d of %d written "
          "rows pass; the ranking keeps all rows by expected improvement and "
          "flags the rest" % (gap_thr, n_pass_gate, len(table)))
    print("[enumerate] wrote %s (%d rows, top %d by expected improvement)"
          % (_rel(out_path), len(table), len(table)))
    print("[enumerate] top-5 by expected improvement:")
    show = [c for c in ["Rank", "catalyst", "ligand", "base", "solvent",
                        "reactant_2", "reactant_1", "catalyst_loading",
                        "Ensemble Predicted Yield (%)",
                        "Expected Improvement"] if c in table.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.head(5)[show].to_string(index=False))
    return 0


# CLI
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recommend_virtual.py",
        description="Condition recommendation from the frozen v16 pipeline.  "
                    "rank-real is the cheap route; enumerate is the virtual-"
                    "library route and requires --allow-virtual.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default=None,
                        help="dataset alias from data/alias_mapping.yaml "
                             "(default: the case-study dataset)")
    common.add_argument("--data-dir", default=None,
                        help="directory holding <source>.csv and "
                             "col_mappings.csv (default: the project Data dir)")
    common.add_argument("--dry-run", action="store_true",
                        help="report the search space/cost and the exact "
                             "command; enumerate/score nothing")

    p1 = sub.add_parser("rank-real", parents=[common],
                        help="route (1): rank real measured reactions")
    p1.add_argument("--predictions", default=None,
                    help="model_predictions_all.csv from a pipeline run "
                         "(default: newest matching run under outputs/)")
    p1.add_argument("--outputs-dir", default=None,
                    help="root searched for cached prediction tables")
    p1.add_argument("--fit-compact", action="store_true",
                    help="fit the compact member panel when no cached "
                         "prediction table exists")
    p1.add_argument("--top-n", type=int, default=20,
                    help="rows shown in v16's rendered table (default 20)")
    p1.add_argument("--max-model-gap", type=float, default=None,
                    help="model-agreement gate in pp (default: v16's "
                         "ENSEMBLE_MAX_MODEL_GAP = 20.0)")

    p2 = sub.add_parser("enumerate", parents=[common],
                        help="route (2): capped virtual library (gated)")
    p2.add_argument("--max-combos", type=int, default=20_000_000,
                    help="score at most this many combinations (default 20000000: "
                         "the whole library)")
    p2.add_argument("--allow-virtual", action="store_true",
                    help="required to run route (2); without it the command "
                         "refuses and exits non-zero")
    p2.add_argument("--chunk-rows", type=int, default=250_000,
                    help="rows per scoring chunk (default 250000)")
    p2.add_argument("--keep-top", type=int, default=1_000,
                    help="rows retained and written (default 1000)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "dataset", None) is None:
        from hte.datasets import case_study as _case
        args.dataset = _case()
    # The frozen pipeline is no longer loaded: everything this module needs is
    # imported from hte.virtual / hte.rank_real / hte.v16_core.
    ns = {}
    if args.command == "rank-real":
        return cmd_rank_real(ns, args)
    return cmd_enumerate(ns, args)


if __name__ == "__main__":
    sys.exit(main())

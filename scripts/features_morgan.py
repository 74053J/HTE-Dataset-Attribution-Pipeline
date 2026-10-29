#!/usr/bin/env python
"""Structural baseline for the reaction-yield models: Morgan/ECFP fingerprints.

WHY THIS FILE EXISTS
--------------------
Every feature the paper's pipeline sees is a *reaction condition*: the seven
categorical slots (metal, catalyst, ligand, base, solvent, leaving_group,
substrate_type) as integer codes plus temperature / time / catalyst_loading as
numbers.  Nothing in the matrix describes the *shape* of the molecules, so a
reviewer is entitled to ask "does adding structure help?".  This module is the
controlled experiment that answers it:

    same dataset, same rows, same train/test split, same estimator
        arm A:  condition features                    (the paper's baseline)
        arm B:  condition features + Morgan(ECFP) of the leaving-group
                (electrophile) SMILES

and reports the held-out R2 of both arms plus the delta.  Only ONE thing is
allowed to differ between the arms -- the fingerprint block -- so the delta is
attributable to structure and nothing else.  Slot convention: the project's
`substrate_type` is the NUCLEOPHILE and its `leaving_group` is the ELECTROPHILE
(see the upstream Data/col_mappings.csv), so the fingerprints here are computed
on the electrophile.  The `--slot substrate` name is historical: it selects the
electrophile SMILES, whose column is derived from `data/col_mappings.csv` (the
raw column that feeds the slot) plus a `_name`/`_id` -> `_smiles` sibling rule.

USAGE
-----
    python scripts/features_morgan.py --check
        Self-test on a few SMILES: prints the set-bit count of each vector.
        Touches no repo state and does not load the frozen pipeline.

    python scripts/features_morgan.py --dataset <ALIAS> [--model XGBoost]
                                      [--fraction 0.2] [--slot substrate]
                                      [--smiles-col COLUMN] [--seed 42]
                                      [--data-dir DIR] [--repeats 1]

    python scripts/features_morgan.py --list
        Which aliases have a leaving-group (electrophile) SMILES column this
        module knows how to reach, and which raw file each alias points at.

The alias -> source-file mapping is read from data/alias_mapping.yaml, so an
alias used on the command line is the same identifier the paper uses.  Raw CSV
files live outside the repository; the data directory is resolved from
--data-dir, otherwise the packaged data/ directory.

WHY THE FROZEN PIPELINE IS USED, BUT NOT ITS BENCHMARK LOOP
----------------------------------------------------------
the frozen pipeline is read with its file left byte-identical (its
sha256 is pinned in results/checksums.txt).  This module reuses, verbatim:

    load_real_dataset    column mapping + yield cleaning, exactly the frame the
                         paper's models were fitted on
    prepare_features     label encoding + train-only scaling/imputation
    get_split_positions  the ONE train/test split definition
    estimate_noise_ceiling   the replicate-based R2 ceiling

`train_and_evaluate_models` is deliberately NOT called: it fits SIX estimators
and, per estimator, runs three cross-validations and two learning curves.  The
comparison here needs one estimator on one split, and on the largest
fingerprint-bearing dataset that loop would blow the smoke-test budget by an
order of magnitude.  `build_estimator` below therefore mirrors the exact
constructor arguments of v16's model registry (n_estimators=200, max_depth=6,
learning_rate=0.1, subsample/colsample 0.8; RF 100 trees depth 15; HistGB
HISTGB_MAX_ITER; MLP (128,64,32)), reading the two numeric constants from the
v16 namespace so they cannot drift.

DESIGN NOTES
------------
* Fingerprint: Morgan / ECFP, radius 2, 2048 bits, returned as a numpy uint8
  vector of 0/1.  Invalid, empty, NaN or placeholder ("NoAdditive") SMILES give
  None from `morgan_fingerprint`; in the dataset comparison those rows become an
  all-zero block AND ARE KEPT, so both arms score exactly the same rows.  The
  failure count is reported; nothing crashes.
* The fingerprint block is appended AFTER the condition columns, so the
  categorical-column indices computed by `prepare_features` stay valid for the
  HistGB-native-categorical and MLP-one-hot estimators.
* The fingerprint block is NOT rescaled: the bits are already 0/1 and the three
  tree estimators are scale-free.  The MLP-one-hot pipeline standard-scales the
  whole matrix after one-hot encoding, exactly as it does in v16.
* --fraction draws a deterministic row subsample (seed) BEFORE the split, to
  keep the run fast on the 27k-row datasets.  Both arms see the same rows.
* No leakage is introduced: the fingerprint is a function of the REACTANT
  SMILES only; the split, scaler and imputation are fitted on training rows.

WHAT THIS CANNOT DO
-------------------
A Morgan fingerprint needs a SMILES string.  Datasets whose routed structural
column is only a plain reagent name (Roche 66k, Summit, rxn_yields, the synthetic
benchmark) have no structure to encode; for those the module stops with an
explicit message instead of inventing molecules.  Use --slot / --smiles-col to
point it at a different structural column that does exist.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _p in (SCRIPTS, ROOT):          # ROOT last, so `import hte` works
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
FP_DTYPE = np.uint8

#: Where the raw CSVs live when no override is given.  They are NOT in the repo.
DEFAULT_DATA_DIR = ROOT / "data"

#: Columns that are placeholders, not molecules.  Counted as failures, and the
#: comparison keeps their rows with an all-zero fingerprint block.
_NON_MOLECULE_TOKENS = {"nan", "none", "null", "no", "noadditive", "nosolvent", "-", "unknown"}

_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_N_BITS)

# RDKit prints a parse error per bad SMILES; we count and report every failure
# ourselves, so silence the per-molecule spam.
RDLogger.DisableLog("rdApp.*")


# 1. SMILES -> fingerprint
def _clean_smiles(value) -> str:
    """Normalise a raw cell to a stripped string ('' when it is not a molecule)."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip().strip("'\"")
    if not text or text.lower() in _NON_MOLECULE_TOKENS:
        return ""
    return text


def morgan_fingerprint(smiles) -> np.ndarray | None:
    """SMILES -> Morgan/ECFP fingerprint as a uint8 vector of length MORGAN_N_BITS.

    Radius 2, 2048 bits.  Returns None -- never raises -- for empty / NaN /
    placeholder / unparseable input.
    """
    text = _clean_smiles(smiles)
    if not text:
        return None
    try:
        mol = Chem.MolFromSmiles(text)
    except Exception:                      # RDKit can raise on exotic input
        mol = None
    if mol is None:
        return None
    try:
        bit_string = _GEN.GetFingerprint(mol).ToBitString()
    except Exception:
        return None
    if len(bit_string) != MORGAN_N_BITS:
        return None
    bits = np.frombuffer(bit_string.encode("ascii"), dtype=np.uint8) - ord("0")
    return bits.astype(FP_DTYPE, copy=False)


def fingerprint_matrix(smiles_values) -> tuple[np.ndarray, dict]:
    """Fingerprint every row; return (uint8 matrix [n, MORGAN_N_BITS], report).

    Rows that fail to parse get an all-zero row so the caller can keep the row
    set identical across arms.  Parsing is cached by unique SMILES string.
    """
    cleaned = [_clean_smiles(v) for v in smiles_values]
    unique = list(dict.fromkeys(cleaned))
    cache: dict[str, np.ndarray | None] = {}
    failed: list[str] = []
    for text in unique:
        fp = morgan_fingerprint(text) if text else None
        cache[text] = fp
        if text and fp is None:
            failed.append(text)

    matrix = np.zeros((len(cleaned), MORGAN_N_BITS), dtype=FP_DTYPE)
    n_ok = 0
    for i, text in enumerate(cleaned):
        fp = cache[text]
        if fp is not None:
            matrix[i] = fp
            n_ok += 1

    row_bits = matrix.sum(axis=1)
    patterns = len({matrix[i].tobytes() for i in range(matrix.shape[0])})
    report = {
        "n_rows": int(matrix.shape[0]),
        "n_ok": int(n_ok),
        "n_failed_rows": int(matrix.shape[0] - n_ok),
        "n_unique": int(sum(1 for t in unique if t)),
        "n_failed_unique": int(len(failed)),
        "failed_examples": failed[:5],
        "n_patterns": int(patterns),
        "mean_bits": float(row_bits.mean()) if matrix.shape[0] else 0.0,
        "min_bits": int(row_bits.min()) if matrix.shape[0] else 0,
        "max_bits": int(row_bits.max()) if matrix.shape[0] else 0,
    }
    return matrix, report


# 2. Dataset plumbing: alias -> file -> SMILES column
#: Which raw column carries the structure of a slot.
#:
#: The column is NOT chosen per dataset in code.  It is derived from the two
#: tables: `data/col_mappings.csv` says which raw column feeds the slot, and the
#: derivation below turns that into the sibling column that holds the structure
#: (`..._name` -> `..._smiles`, `..._id` -> `..._smiles`), falling back to the
#: first SMILES column whose name mentions the slot.  `--smiles-col` overrides.
#:
#: Slot convention: `reactant_1` (historically `substrate_type`) is the
#: NUCLEOPHILE and `reactant_2` (historically `leaving_group`) the ELECTROPHILE;
#: the fingerprint used here is the electrophile's, i.e. `substrate`.
SLOT_ALIASES = {
    "substrate": "reactant_1",
    "substrate_type": "reactant_1",
    "leaving_group": "reactant_2",
    "product": None,
}

_SLOT_KEYWORDS = {
    "substrate": ("substrate", "startingmat_1", "reactant_1", "nucleophile", "aryl"),
    "ligand": ("ligand", "catalyst"),
    "base": ("base", "reagent_1"),
    "solvent": ("solvent",),
    "reactant_2": ("reactant_2", "startingmat_2", "amine", "nucleophile",
                  "boronic", "electrophile"),
}


def load_aliases() -> dict:
    import yaml

    path = ROOT / "data" / "alias_mapping.yaml"
    if not path.exists():
        raise SystemExit(f"alias mapping not found: {path}")
    mapping = yaml.safe_load(path.read_text(encoding="utf-8"))
    return mapping["datasets"]


def resolve_data_dir(cli_value: str | None) -> tuple[Path | None, list[str]]:
    """First existing directory (in priority order) that contains CSV files."""
    candidates = [
        cli_value,
        str(ROOT / "data"),
        str(DEFAULT_DATA_DIR),
        str(ROOT / "data"),
    ]
    tried: list[str] = []
    for cand in candidates:
        if not cand:
            continue
        path = Path(cand).expanduser()
        tried.append(str(path))
        if path.is_dir() and next(path.glob("*.csv"), None) is not None:
            return path, tried
    return None, tried


def resolve_smiles_column(df, raw_feature_names, stem, slot, override=None):
    """Return (column or None, how it was chosen).

    The slot is resolved to its canonical name through :data:`SLOT_ALIASES`, the
    raw column that feeds it comes from the mapping table (via
    ``raw_feature_names``), and the structure column is that column's sibling.
    No dataset is named here.
    """
    if override:
        if override not in df.columns:
            return None, f"--smiles-col '{override}' is not a column of this file"
        return override, "explicit --smiles-col"

    canon = SLOT_ALIASES.get(slot, slot)
    raw = raw_feature_names.get(slot) or (raw_feature_names.get(canon) if canon else None)

    # 1) the sibling of the column that feeds the slot
    candidates: list[str] = []
    if raw:
        low = str(raw).lower()
        if low.endswith("_name"):
            candidates.append(str(raw)[:-5] + "_smiles")
        if low.endswith("_id"):
            candidates.append(str(raw)[:-3] + "_smiles")
        candidates.append(str(raw) + "_smiles")

    # 2) any SMILES column whose name mentions the slot
    keywords = _SLOT_KEYWORDS.get(slot, (slot,))
    for col in df.columns:
        if "smiles" not in str(col).lower():
            continue
        if any(k in str(col).lower() for k in keywords):
            candidates.append(col)

    for col in candidates:
        if col in df.columns:
            return col, "derived from the mapped column name"
    return None, "no candidate column"


def load_v16():
    """The names this module used to borrow from the frozen pipeline.

    They now come from the package; only the frozen file's SHA-256 is still read
    (read here directly; the file is never executed).
    """
    from hte import params as P
    from hte.ceiling import estimate_noise_ceiling
    from hte.v16_core import get_split_positions, load_real_dataset
    from hte.virtual import HISTGB_MAX_ITER, RF_N_ESTIMATORS

    ns = {
        "load_real_dataset": load_real_dataset,
        "get_split_positions": get_split_positions,
        "estimate_noise_ceiling": estimate_noise_ceiling,
        "TEST_SIZE": P.TEST_SIZE,
        "RF_N_ESTIMATORS": RF_N_ESTIMATORS,
        "HISTGB_MAX_ITER": HISTGB_MAX_ITER,
    }
    import hashlib
    frozen = Path(__file__).resolve().parent.parent / "archive" / "single_dataset_full_benchmark.py"
    sha = hashlib.sha256(frozen.read_bytes()).hexdigest() if frozen.is_file() else ""
    return ns, sha


# 3. Estimators -- mirrored from v16's model registry
MODEL_NAMES = [
    "XGBoost",
    "Gradient Boosting",
    "Multilayer Perceptron",
    "Random Forest",
    "HistGB-NativeCat",
    "MLP-OneHot",
]

_MODEL_ALIASES = {
    "xgboost": "XGBoost",
    "gradient boosting": "Gradient Boosting",
    "gradientboosting": "Gradient Boosting",
    "multilayer perceptron": "Multilayer Perceptron",
    "mlp": "MLP-OneHot",
    "mlp-onehot": "MLP-OneHot",
    "random forest": "Random Forest",
    "randomforest": "Random Forest",
    "histgb": "HistGB-NativeCat",
    "histgb-nativecat": "HistGB-NativeCat",
}


def normalise_model_name(name: str) -> str:
    key = str(name).strip().lower().replace("_", " ")
    if key in _MODEL_ALIASES:
        return _MODEL_ALIASES[key]
    for canonical in MODEL_NAMES:
        if key == canonical.lower():
            return canonical
    raise SystemExit(
        f"unknown --model '{name}'.  Choose from: " + ", ".join(MODEL_NAMES)
    )


def build_estimator(name: str, cat_idx: list[int], rf_n_estimators: int,
                    histgb_max_iter: int):
    """Same constructor arguments as v16's model registry, one estimator only."""
    from sklearn.ensemble import (
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    import xgboost as xgb

    cat = cat_idx if cat_idx else None
    if name == "XGBoost":
        return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, random_state=42)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                         learning_rate=0.1, random_state=42)
    if name == "Multilayer Perceptron":
        return MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                            max_iter=500, random_state=42, early_stopping=True)
    if name == "Random Forest":
        # v16 sets RF_N_JOBS (= os.cpu_count()) for the main fit only; CV loops
        # are not run here, so the fast setting can stay on.
        return RandomForestRegressor(n_estimators=rf_n_estimators, max_depth=15,
                                     min_samples_split=5, random_state=42,
                                     n_jobs=(os.cpu_count() or 4))
    if name == "HistGB-NativeCat":
        return HistGradientBoostingRegressor(max_iter=histgb_max_iter, max_depth=6,
                                             learning_rate=0.1, random_state=42,
                                             categorical_features=cat)
    if name == "MLP-OneHot":
        return make_pipeline(
            ColumnTransformer(
                [("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
                remainder="passthrough"),
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu",
                         max_iter=500, random_state=42, early_stopping=True),
        )
    raise SystemExit(f"no builder for model '{name}'")


def _fit_and_r2(estimator, X_train, y_train, X_test, y_test):
    from sklearn.metrics import r2_score

    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    pred = estimator.predict(X_test)
    return float(r2_score(y_test, pred)), fit_seconds


# 4. --check
def run_check() -> int:
    cases = [
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("benzene", "c1ccccc1"),
        ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("methane", "C"),
        ("ionic salt", "[Na+].[Cl-]"),
        ("padded SMILES", "  c1ccccc1  "),
        ("invalid SMILES", "not_a_smiles"),
        ("empty string", ""),
        ("None", None),
        ("placeholder token", "NoAdditive"),
    ]
    print("Morgan/ECFP self-test   radius=%d  n_bits=%d  dtype=%s"
          % (MORGAN_RADIUS, MORGAN_N_BITS, np.dtype(FP_DTYPE).name))
    print("-" * 74)
    valid: dict[str, np.ndarray] = {}
    failures = 0
    for label, smiles in cases:
        fp = morgan_fingerprint(smiles)
        if fp is None:
            failures += 1
            print("  %-18s %-34s -> None (not a molecule)" % (label, repr(smiles)[:34]))
            continue
        assert fp.shape == (MORGAN_N_BITS,), "unexpected shape %r" % (fp.shape,)
        assert fp.dtype == np.dtype(FP_DTYPE), "unexpected dtype %r" % (fp.dtype,)
        assert set(np.unique(fp)).issubset({0, 1}), "vector is not binary"
        valid[label] = fp
        print("  %-18s %-34s -> %4d bits set  (shape %s, dtype %s)"
              % (label, repr(smiles)[:34], int(fp.sum()), fp.shape, fp.dtype))

    print("-" * 74)
    # Invariants that would catch a silently broken fingerprint.
    assert valid["methane"].sum() == 1, "methane (one heavy atom) must set exactly one bit"
    assert not np.array_equal(valid["aspirin"], valid["benzene"]), "distinct molecules must differ"
    assert np.isfinite(valid["benzene"].sum())
    again = morgan_fingerprint(cases[0][1])
    assert again is not None and np.array_equal(again, valid["aspirin"]), "not deterministic"
    n_overlap = int(np.logical_and(valid["aspirin"], valid["benzene"]).sum())
    print("  checks: methane == 1 bit, aspirin != benzene, repeat call identical")
    print("  aspirin/benzene share %d of %d bits; aspirin sets %d, benzene sets %d"
          % (n_overlap, MORGAN_N_BITS, int(valid["aspirin"].sum()), int(valid["benzene"].sum())))
    print("  %d/%d parsed, %d rejected cleanly (no exception)"
          % (len(valid), len(cases), failures))
    print("SELF-TEST PASSED")
    return 0


# 5. --list
def smiles_from_tables(path, data_dir, slot="substrate"):
    """The structure column for ``slot``, derived from the two tables only.

    No frame is loaded: ``col_mappings.csv`` says which raw column feeds the
    slot for this file, and the sibling column that holds the structure is
    derived from its name (then from any SMILES column mentioning the slot).
    Returns ``(column or None, how)``.
    """
    import csv

    from hte.mapping import load_mapping, resolve_for_file

    try:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
            header = next(csv.reader(fh))
    except OSError as exc:
        return None, "cannot read the file (%s)" % exc
    mapping = load_mapping(str(data_dir))
    if mapping is None:
        return None, "col_mappings.csv not found in %s" % data_dir
    resolved = resolve_for_file(mapping, Path(path).name, header)
    canon = SLOT_ALIASES.get(slot, slot)
    raw = resolved.get(slot) or (resolved.get(canon) if canon else None)
    cands = []
    if isinstance(raw, str) and raw:
        low = raw.lower()
        if low.endswith("_name"):
            cands.append(str(raw)[:-5] + "_smiles")
        if low.endswith("_id"):
            cands.append(str(raw)[:-3] + "_smiles")
        cands.append(str(raw) + "_smiles")
    keywords = _SLOT_KEYWORDS.get(slot, (slot,))
    for col in header:
        if "smiles" in str(col).lower() and any(k in str(col).lower() for k in keywords):
            cands.append(col)
    for c in cands:
        if c in header:
            return c, "derived from the mapped column name"
    return None, "no candidate column"


def run_list(args) -> int:
    aliases = load_aliases()
    data_dir, tried = resolve_data_dir(args.data_dir)
    print("data directory: %s" % (data_dir if data_dir else "NOT FOUND"))
    if data_dir is None:
        for t in tried:
            print("  tried: %s" % t)
    print()
    print("%-24s %-8s %-52s %s" % ("alias", "file", "source",
                                   "leaving-group (electrophile) SMILES column"))
    print("-" * 130)
    for alias in sorted(aliases):
        source = aliases[alias]["source"]
        path = data_dir / (source + ".csv") if data_dir else None
        exists = "yes" if (path and path.exists()) else "no"
        if path and path.exists():
            sub, why = smiles_from_tables(path, data_dir)
            sub = "%s (%s)" % (sub, why) if sub else "-  (%s)" % why
        else:
            sub = "-"
        print("%-24s %-8s %-52s %s" % (alias, exists, source, sub))
    return 0


# 6. The controlled comparison
def run_comparison(args) -> int:
    started = time.perf_counter()
    if not (0.0 < float(args.fraction) <= 1.0):
        raise SystemExit("--fraction must be in (0, 1]")
    if int(args.repeats) < 1:
        raise SystemExit("--repeats must be >= 1")
    aliases = load_aliases()
    if args.dataset not in aliases:
        raise SystemExit(
            "--dataset '%s' is not in data/alias_mapping.yaml.  Known aliases:\n  %s"
            % (args.dataset, "\n  ".join(sorted(aliases)))
        )
    meta = aliases[args.dataset]
    source = meta["source"]
    stem = source
    data_dir, tried = resolve_data_dir(args.data_dir)
    if data_dir is None:
        print("ERROR: no data directory found.  Tried:")
        for t in tried:
            print("  " + t)
        print("Pass --data-dir <the folder holding the raw CSVs>.")
        return 2
    data_path = data_dir / (source + ".csv")
    if not data_path.exists():
        print("ERROR: raw file for alias '%s' not found." % args.dataset)
        print("  expected: %s" % data_path)
        print("  data dir: %s" % data_dir)
        print("  Nothing was invented; stopping.")
        return 2

    ns, v16_sha = load_v16()
    print("=" * 100)
    print("STRUCTURAL BASELINE  --  condition features vs condition features + Morgan(ECFP)")
    print("  dataset        : %s  (%s)" % (args.dataset, source))
    print("  raw file       : %s" % data_path)
    print("  frozen v16     : sha256 %s" % v16_sha[:16])
    print("  model          : %s" % args.model)
    print("  fingerprint    : Morgan radius %d, %d bits -> uint8"
          % (MORGAN_RADIUS, MORGAN_N_BITS))
    print("=" * 100)

    df, raw_feature_names = ns["load_real_dataset"](str(data_path), str(data_dir))
    n_all = len(df)

    smiles_col, how = resolve_smiles_column(
        df, raw_feature_names, stem, args.slot, args.smiles_col)
    if smiles_col is None:
        print()
        print("ERROR: no usable SMILES column for slot '%s'." % args.slot)
        print("  %s" % how)
        print("  This dataset has no structure to encode in that slot; a Morgan")
        print("  fingerprint cannot be computed from reagent NAMES.  Nothing was")
        print("  invented.  Options:")
        print("    --slot ligand|base|solvent          (other structural slots)")
        print("    --smiles-col '<column name>'        (pick one explicitly)")
        available = [c for c in df.columns if "smiles" in str(c).lower()]
        print("  SMILES-like columns in this file: %s"
              % (", ".join(str(c) for c in available) if available else "none"))
        print("  The column is derived from data/col_mappings.csv (which raw column"
              " feeds the slot)")
        print("  plus a _name -> _smiles / _id -> _smiles sibling rule; pass"
              " --smiles-col to override.")
        return 3
    print("\n[structure] slot '%s' <- raw column '%s'  (%s)" % (args.slot, smiles_col, how))

    ceil = ns["estimate_noise_ceiling"](df)
    ceiling = ceil.get("ceiling", float("nan"))
    # Second ceiling: the SAME replicate estimator with the electrophile identity
    # (the fingerprinted column) added to the grouping key.  Necessary because
    # the published key calls two different electrophiles "the same condition"
    # whenever the file has no name column for them (bh-hte-ood: leaving_group
    # and substrate_type are both 'Unknown'), so its within-group variance
    # contains the electrophile effect and the published ceiling is a LOWER
    # BOUND -- v16 says so explicitly.  A structure-aware model can legitimately
    # beat that bound, and this number shows by how much of the
    # electrophile-corrected ceiling it is really doing it.
    from hte import params as _P
    ceil_struct = ns["estimate_noise_ceiling"](df, list(_P.CONDITION_COLS) + [smiles_col])
    ceiling_struct = ceil_struct.get("ceiling", float("nan"))

    fraction = float(args.fraction)
    if fraction < 1.0:
        k = max(2, int(round(fraction * n_all)))
        idx = np.sort(np.random.RandomState(args.seed).choice(n_all, size=k, replace=False))
        frame = df.iloc[idx].reset_index(drop=True)
    else:
        frame = df.reset_index(drop=True)
    n = len(frame)
    if n < 10:
        print("ERROR: --fraction %.3f leaves only %d rows; need at least 10."
              % (fraction, n))
        return 2

    fp_matrix, fp_report = fingerprint_matrix(frame[smiles_col].to_numpy())
    print("[structure] %d rows, %d unique SMILES, %d failed to parse (%.2f%% of rows), "
          "%d distinct fingerprint patterns"
          % (fp_report["n_rows"], fp_report["n_unique"], fp_report["n_failed_rows"],
             100.0 * fp_report["n_failed_rows"] / max(1, fp_report["n_rows"]),
             fp_report["n_patterns"]))
    if fp_report["n_failed_unique"]:
        print("            failures kept as all-zero vectors so both arms score the same "
              "rows; examples: %s" % (fp_report["failed_examples"] or "(blank cells)"))
    print("[structure] bits set per row: mean %.1f, min %d, max %d"
          % (fp_report["mean_bits"], fp_report["min_bits"], fp_report["max_bits"]))

    constants = {
        "test_size": float(ns["TEST_SIZE"]),
        "rf_n_estimators": int(ns["RF_N_ESTIMATORS"]),
        "histgb_max_iter": int(ns["HISTGB_MAX_ITER"]),
    }

    print("\n  rows used      : %d of %d  (--fraction %.3f, seed %d)"
          % (n, n_all, fraction, args.seed))
    print("  held-out split : test_size=%.2f (v16 TEST_SIZE), get_split_positions()"
          % constants["test_size"])
    print()

    base_scores: list[float] = []
    fp_scores: list[float] = []
    for rep in range(int(args.repeats)):
        split_seed = int(args.seed) + rep
        train_idx, test_idx = ns["get_split_positions"](n, constants["test_size"], split_seed)
        # The packaged builder, so Table 2 sees exactly the columns the rest of the
        # package uses (the monolith is no longer the feature authority).
        from hte.v16_core import prepare_features as _prepare_features
        X_cond_df, y_series, feature_cols, _disp, _enc, _scaler = _prepare_features(
            frame, raw_feature_names, fit_rows=train_idx)
        X_cond = np.asarray(X_cond_df, dtype=np.float64)
        y = y_series.to_numpy(dtype=np.float64)
        cat_idx = [i for i, c in enumerate(feature_cols) if str(c).endswith("_encoded")]

        X_aug = np.empty((n, X_cond.shape[1] + MORGAN_N_BITS), dtype=np.float32)
        X_aug[:, :X_cond.shape[1]] = X_cond
        X_aug[:, X_cond.shape[1]:] = fp_matrix

        y_train, y_test = y[train_idx], y[test_idx]
        r2_cond, t_cond = _fit_and_r2(
            build_estimator(args.model, cat_idx, constants["rf_n_estimators"],
                            constants["histgb_max_iter"]),
            X_cond[train_idx], y_train, X_cond[test_idx], y_test)
        r2_fp, t_fp = _fit_and_r2(
            build_estimator(args.model, cat_idx, constants["rf_n_estimators"],
                            constants["histgb_max_iter"]),
            X_aug[train_idx], y_train, X_aug[test_idx], y_test)
        base_scores.append(r2_cond)
        fp_scores.append(r2_fp)

        print("split %d/%d  (random_state=%d, n_train=%d, n_test=%d)"
              % (rep + 1, int(args.repeats), split_seed, len(train_idx), len(test_idx)))
        print("  conditions only          %2d features  held-out R2 = %+0.4f   [fit %5.2f s]"
              % (X_cond.shape[1], r2_cond, t_cond))
        print("  conditions + Morgan      %2d features  held-out R2 = %+0.4f   [fit %5.2f s]"
              % (X_aug.shape[1], r2_fp, t_fp))
        print("  delta (with - without)                        = %+0.4f" % (r2_fp - r2_cond))
        print()

    base_mean = float(np.mean(base_scores))
    fp_mean = float(np.mean(fp_scores))
    delta_mean = fp_mean - base_mean
    base_sd = float(np.std(base_scores, ddof=1)) if len(base_scores) > 1 else 0.0
    fp_sd = float(np.std(fp_scores, ddof=1)) if len(fp_scores) > 1 else 0.0
    delta_sd = (float(np.std(np.subtract(fp_scores, base_scores), ddof=1))
                if len(base_scores) > 1 else 0.0)

    print("=" * 100)
    print("RESULT  %s / %s / slot=%s(%s) / n=%d / %d split(s)"
          % (args.dataset, args.model, args.slot, smiles_col, n, int(args.repeats)))
    print("  held-out R2, conditions only      : %+0.4f%s"
          % (base_mean, ("  (sd %.4f)" % base_sd) if len(base_scores) > 1 else ""))
    print("  held-out R2, + Morgan fingerprint : %+0.4f%s"
          % (fp_mean, ("  (sd %.4f)" % fp_sd) if len(fp_scores) > 1 else ""))
    print("  DELTA (with - without)            : %+0.4f%s"
          % (delta_mean, ("  (sd %.4f)" % delta_sd) if len(base_scores) > 1 else ""))
    print("  SMILES parse failures             : %d of %d rows (%.2f%%), "
          "%d of %d unique strings"
          % (fp_report["n_failed_rows"], n, 100.0 * fp_report["n_failed_rows"] / max(1, n),
             fp_report["n_failed_unique"], fp_report["n_unique"]))
    if np.isfinite(ceiling):
        print("  ceiling, published key (full n=%d) : %.4f  "
              "(%d repeated conditions, %.1f%% of rows, %d conditions)"
              % (n_all, ceiling, ceil.get("n_rep_groups", 0),
                 100.0 * ceil.get("rep_frac", 0.0), ceil.get("n_cond", 0)))
        for label, value in (("without", base_mean), ("with", fp_mean)):
            print("  capture ratio %-7s              : %.1f%% of the ceiling"
                  % (label, 100.0 * value / ceiling if ceiling else float("nan")))
        print("  headroom left for structure       : %.4f R2"
              % max(0.0, ceiling - base_mean))
    else:
        print("  ceiling                           : UNDEFINED for this dataset "
              "(no condition measured twice), so no capture ratio")
    if np.isfinite(ceiling_struct):
        print("  ceiling, + electrophile identity  : %.4f  "
              "(%d repeated conditions, %.1f%% of rows)"
              % (ceiling_struct, ceil_struct.get("n_rep_groups", 0),
                 100.0 * ceil_struct.get("rep_frac", 0.0)))
        for label, value in (("without", base_mean), ("with", fp_mean)):
            print("  capture vs struct. ceiling %-7s : %.1f%%"
                  % (label, 100.0 * value / ceiling_struct))
        if np.isfinite(ceiling):
            if ceiling_struct > ceiling + 1e-6:
                print("  (both ceilings are replicate-based LOWER BOUNDS, as v16 documents:")
                print("   the published key pools every molecule it cannot see -- here the whole")
                print("   electrophile -- into one 'condition', and the structure-aware key only")
                print("   removes the pooling it fingerprints, so a model can exceed either.)")
            elif abs(ceiling_struct - ceiling) <= 1e-6:
                print("  (adding the electrophile identity does not change the replicate key: the")
                print("   electrophile is already represented in the condition features, so its")
                print("   fingerprint can only add redundancy.)")
    else:
        print("  ceiling, + electrophile identity  : undefined (no replicate pair shares "
              "both conditions and structure)")
    if fraction < 1.0:
        print("  NOTE: R2 above is on a %.0f%% row subsample; the ceiling is for the full "
              "dataset, so the capture ratio is indicative." % (100 * fraction))
    print("  elapsed                           : %.1f s" % (time.perf_counter() - started))
    print("=" * 100)
    return 0


# 7. CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Morgan/ECFP structural baseline for the conditions-only yield models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/features_morgan.py --check\n"
            "  python scripts/features_morgan.py --dataset bh-hte-ood_BH --model XGBoost "
            "--fraction 0.2\n"
            "  python scripts/features_morgan.py --dataset AI_RCP_SM --model XGBoost\n"
        ),
    )
    parser.add_argument("--check", action="store_true",
                        help="self-test the SMILES -> fingerprint function")
    parser.add_argument("--list", action="store_true",
                        help="list aliases, raw files and the routed leaving-group "
                             "(electrophile) SMILES column")
    parser.add_argument("--dataset", help="dataset alias from data/alias_mapping.yaml")
    parser.add_argument("--model", default="XGBoost",
                        help="estimator name (%s)" % ", ".join(MODEL_NAMES))
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of rows to use before splitting (default 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="subsample + split seed; 42 reproduces the frozen split")
    parser.add_argument("--slot", default="substrate",
                        choices=["substrate", "ligand", "base", "solvent"],
                        help="which structural slot to fingerprint (default: substrate, "
                             "which routes to the leaving-group/electrophile column)")
    parser.add_argument("--smiles-col", default=None,
                        help="raw column to fingerprint, overriding the routing table")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat the split with seed+0..n-1 and average (default 1)")
    parser.add_argument("--data-dir", default=None,
                        help="folder holding the raw CSVs (default: project Data/)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check:
        return run_check()
    if args.list:
        return run_list(args)
    if not args.dataset:
        parser.error("one of --check, --list or --dataset is required")
    args.model = normalise_model_name(args.model)
    return run_comparison(args)


if __name__ == "__main__":
    sys.exit(main())

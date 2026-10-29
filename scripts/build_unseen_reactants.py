#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 8 -- §3.6, part 2: reactants never seen at all.

Ported from ``_work/v248/q5_smiles.py`` (2026-10-26 19:52).  The split is by
reactant-1 identity, so every held-out row carries a reactant the model never
saw; the question is how much of that a Morgan fingerprint can recover.

Arms (estimator settings are the paper's):

    global mean
    reactant-2 historical mean
    conditions + reactant-2 identity            (no structure)
    conditions + reactant-2 identity + Morgan   (2048 bits, radius 2, both partners)
    conditions + Morgan only                    (identity columns deleted)

The two reactant slots hold SMILES for this dataset -- that is what
``data/col_mappings.csv`` maps them to -- so the fingerprints are computed from
the slots themselves; no column name is guessed and no dataset is named.  The
dataset is the one flagged ``unseen_reactant: true`` in
``data/alias_mapping.yaml``.

Writes
  results/unseen_reactants.csv
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from hte import params as P                     # noqa: E402
from hte.data import load_real_dataset          # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data"
RES = ROOT / "results"

SEED = 42
TEST_SIZE = 0.25
FP_BITS, FP_RADIUS = 2048, 2
XGB_BASE = dict(max_depth=6, learning_rate=0.05, subsample=0.9, n_jobs=4,
                random_state=SEED, enable_categorical=True, tree_method="hist")


def _dataset() -> str:
    from hte.datasets import _entries
    hits = [a for a, s in _entries(DATA / "alias_mapping.yaml")
            if bool(s.get("unseen_reactant", False))]
    if len(hits) != 1:
        raise SystemExit("alias_mapping.yaml must flag exactly one unseen_reactant "
                         "dataset, found %d: %s" % (len(hits), hits))
    return hits[0]


def fingerprints(values, bits=FP_BITS, radius=FP_RADIUS) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    cache, bad = {}, 0
    for v in pd.unique(values):
        m = Chem.MolFromSmiles(str(v))
        if m is None:
            bad += 1
            cache[v] = np.zeros(bits, dtype=np.float32)
        else:
            cache[v] = np.asarray(gen.GetFingerprintAsNumPy(m), dtype=np.float32)
    print("  fingerprints: %d unique strings, %d unparsable" % (len(cache), bad))
    return np.vstack([cache[v] for v in values])


def _cat(train: pd.DataFrame, test: pd.DataFrame, cols=None):
    """Both frames in the training categories, so XGBoost sees the same levels.

    Only ``cols`` (the condition / identity columns) become categorical; the
    fingerprint block stays float, otherwise XGBoost refuses the frame with
    "Category index from DataFrame has floating point dtype".
    """
    a, b = train.copy(), test.copy()
    for c in (list(cols) if cols is not None else list(a.columns)):
        a[c] = a[c].astype("category")
        b[c] = pd.Categorical(b[c], categories=a[c].cat.categories)
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude-slot", action="append", default=[],
                    help="condition slot to leave out of the arm (repeatable)")
    args = ap.parse_args()

    import xgboost as xgb
    from hte.datasets import stems

    alias = args.dataset or _dataset()
    stem = stems()[alias]
    t0 = time.perf_counter()
    df, _raw = load_real_dataset(str(DATA / (stem + ".csv")), str(DATA))
    df = df.dropna(subset=["reactant_1", "reactant_2", "yield"]).copy()
    df["r1"] = df["reactant_1"].astype(str)
    df["r2"] = df["reactant_2"].astype(str)
    cond = [c for c in P.CATEGORICAL_SLOTS
            if c in df.columns and df[c].astype(str).nunique() > 1
            and c not in ("reactant_1", "reactant_2")]
    cond = [c for c in cond if c not in args.exclude_slot]
    cond += [c for c in ("temperature", "time")
             if c in df.columns and df[c].nunique() > 1]
    print("%s: %d rows | %d distinct reactant-1 | %d distinct reactant-2 | "
          "condition columns: %s"
          % (alias, len(df), df.r1.nunique(), df.r2.nunique(), ", ".join(cond)))

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_i, te_i = next(gss.split(df, groups=df["r1"]))
    tr, te = df.iloc[tr_i], df.iloc[te_i]
    print("  split by reactant-1: train %d rows / %d distinct, test %d rows / "
          "%d distinct (%.1f%% of the test reactants were seen in training)"
          % (len(tr), tr.r1.nunique(), len(te), te.r1.nunique(),
             100 * te.r1.isin(set(tr.r1)).mean()))

    y_tr = tr["yield"].astype(float).values
    y_te = te["yield"].astype(float).values
    gmean = float(y_tr.mean())
    rows = []

    def rec(name, pred, structured: bool):
        r2 = 1 - np.sum((y_te - pred) ** 2) / np.sum((y_te - y_te.mean()) ** 2)
        rows.append(dict(dataset=alias, arm=name, structured=structured,
                         n_train=len(tr), n_test=len(te),
                         n_unseen_reactant1=int((~te.r1.isin(set(tr.r1))).sum()),
                         r2=float(r2), mae=float(np.mean(np.abs(y_te - pred))),
                         spearman=float(stats.spearmanr(y_te, pred).statistic)))

    rec("global mean", np.full(len(te), gmean), False)
    rec("reactant-2 mean",
        te.r2.map(tr.groupby("r2")["yield"].mean()).fillna(gmean).values, False)

    Xc_tr, Xc_te = _cat(tr[["r2"] + cond].astype(str), te[["r2"] + cond].astype(str))
    m1 = xgb.XGBRegressor(n_estimators=700, colsample_bytree=0.9, **XGB_BASE)
    m1.fit(Xc_tr, y_tr)
    rec("conditions + reactant-2 identity", m1.predict(Xc_te), False)

    fp1 = fingerprints(df.r1.values)
    fp2 = fingerprints(df.r2.values)
    m_tr = np.isin(df.index, tr.index)
    m_te = np.isin(df.index, te.index)

    Dtr = pd.concat([Xc_tr.reset_index(drop=True),
                     pd.DataFrame(np.hstack([fp1[m_tr], fp2[m_tr]]))], axis=1)
    Dte = pd.concat([Xc_te.reset_index(drop=True),
                     pd.DataFrame(np.hstack([fp1[m_te], fp2[m_te]]))], axis=1)
    Dte = Dte.reindex(columns=Dtr.columns, fill_value=0)
    a, b = _cat(Dtr, Dte, list(Xc_tr.columns))
    m2 = xgb.XGBRegressor(n_estimators=900, colsample_bytree=0.6, **XGB_BASE)
    m2.fit(a, y_tr)
    rec("conditions + reactant-2 identity + Morgan", m2.predict(b), True)

    Dtr3 = pd.concat([Xc_tr.drop(columns=["r2"]).reset_index(drop=True),
                      pd.DataFrame(np.hstack([fp1[m_tr], fp2[m_tr]]))], axis=1)
    Dte3 = pd.concat([Xc_te.drop(columns=["r2"]).reset_index(drop=True),
                      pd.DataFrame(np.hstack([fp1[m_te], fp2[m_te]]))], axis=1)
    Dte3 = Dte3.reindex(columns=Dtr3.columns, fill_value=0)
    a3, b3 = _cat(Dtr3, Dte3, cond)
    m3 = xgb.XGBRegressor(n_estimators=900, colsample_bytree=0.6, **XGB_BASE)
    m3.fit(a3, y_tr)
    rec("conditions + Morgan only (identity deleted)", m3.predict(b3), True)

    out = pd.DataFrame(rows)
    dst = Path(args.out) if args.out else RES / "unseen_reactants.csv"
    out.to_csv(dst, index=False, encoding="utf-8")
    print("\n%-46s %8s %7s %7s" % ("arm", "R2", "MAE", "rho"))
    for r in rows:
        print("%-46s %+8.3f %7.2f %+7.3f"
              % (r["arm"], r["r2"], r["mae"], r["spearman"]))
    print("\n[unseen] %d arms, %d unseen reactant-1 rows -> %s  (%.0fs)"
          % (len(out), rows[0]["n_unseen_reactant1"], dst,
             time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())

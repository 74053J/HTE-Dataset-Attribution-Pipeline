#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 12b -- turn the raw attribution table into the one the gate reads.

``results/attrib13_long.csv`` (written by ``build_attrib13.py``) has one row per
(dataset, model, encoding, feature) with the permutation importance and the mean
absolute SHAP value.  ``verify_numbers.py`` and ``figures_paper2.py`` instead read
``results/rebuilt_attrib12_long_v7.csv``, which is the same table restricted to
the 13-dataset corpus with the slot name added and rows without a finite SHAP
value dropped.

That derivation used to live in ``_work/v247/compute_new_numbers.py``: it carried
a hard-coded set of three dataset names to drop and absolute paths.  Here the
corpus comes from ``data/alias_mapping.yaml``, so nothing is hard-coded.

Writes
  results/rebuilt_attrib12_long_v7.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

COLUMNS = ["dataset", "model", "encoding", "feature", "perm", "shap", "slot"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from hte import datasets as DS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(ROOT / "results" / "attrib13_long.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "rebuilt_attrib12_long_v7.csv"))
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_file():
        print("[attrib-post] missing input: %s" % src)
        return 1
    df = pd.read_csv(src)
    corpus = set(DS.corpus())
    before = len(df)
    df = df[df["dataset"].astype(str).isin(corpus)].copy()

    if "feature" not in df.columns:
        for cand in ("slot", "column", "name"):
            if cand in df.columns:
                df["feature"] = df[cand]
                break
    df["slot"] = df["feature"].astype(str).str.replace("_encoded", "", regex=False)
    for c in ("perm", "shap"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["shap"])

    out = pd.DataFrame(df, columns=[c for c in COLUMNS if c in df.columns])
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("[attrib-post] %d -> %d rows (%d datasets, %d cells) -> %s"
          % (before, len(out), out.dataset.nunique(),
             out.groupby(["dataset", "model", "encoding"]).ngroups, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

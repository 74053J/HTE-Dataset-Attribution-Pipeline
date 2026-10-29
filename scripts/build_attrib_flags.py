#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 12c -- flag the attribution rows whose slot is constant in that cell.

A categorical slot that never varies inside a dataset cannot contribute to the
model, so its permutation importance and its SHAP value are both exactly zero in
that cell.  The encoding-gap analysis (``attrib_encoding_gap.py``) needs those
rows marked, and read a ``set_aside`` column that nothing in the package wrote
(the flag used to be added by a scratch script outside this repository).

Here the rule is stated once: ``set_aside = perm == 0 and shap == 0``.

Writes
  results/attrib13_long_cosolvent_flagged.csv  (or --out)
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

TOL = 1e-12


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=str(ROOT / "results" / "attrib13_long.csv"))
    ap.add_argument("--out", default=str(ROOT / "results"
                                         / "attrib13_long_cosolvent_flagged.csv"))
    ap.add_argument("--compare", default=None,
                    help="optional reference table; report whether the flag matches")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    perm = pd.to_numeric(df["perm"], errors="coerce").fillna(0.0)
    shap = pd.to_numeric(df["shap"], errors="coerce").fillna(0.0)
    out = df.copy()
    out["set_aside"] = (perm.abs() < TOL) & (shap.abs() < TOL)

    cols = [c for c in ["dataset", "model", "encoding", "feature", "perm", "shap",
                        "set_aside"] if c in out.columns]
    out = out[cols]
    out.to_csv(args.out, index=False, encoding="utf-8")
    print("[attrib-flags] %d rows, %d set aside (%.0f%%) -> %s"
          % (len(out), int(out.set_aside.sum()), 100.0 * out.set_aside.mean(), args.out))

    if args.compare:
        ref = pd.read_csv(args.compare)
        key = ["dataset", "model", "encoding", "feature"]
        if "set_aside" not in ref.columns:
            print("[attrib-flags] reference has no set_aside column")
            return 0
        m = ref.merge(out, on=key, suffixes=("_ref", "_new"))
        same = int((m.set_aside_ref.astype(int) == m.set_aside_new.astype(int)).sum())
        print("[attrib-flags] vs %s : %d / %d rows agree" % (Path(args.compare).name,
                                                             same, len(m)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 11b -- the per-target table behind Figure 5's transfer bar.

``figures.py`` plots the fourth bar of that figure from
``results/transfer_check.csv`` (mean of ``r2_transfer`` over the donor's
targets).  That file used to have no producer in the package: the transfer
matrix was written, the derived view was not.  This script derives it from the
two tables that do have producers:

  outputs/c20_transfer_4donor/transfer_matrix.csv   per (donor, target) metrics
  results/table1_v18_final.csv                      n_rows and mean_yield

No model is fitted here, so the numbers are exactly the matrix's numbers; the
script only reshapes them into the columns ``figures.py`` expects.

Writes
  results/transfer_check.csv   alias, n, target_mean, source_mean, r2_transfer
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


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from hte import datasets as DS

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--donor", default=None,
                    help="donor alias; defaults to the case-study dataset")
    ap.add_argument("--matrix", default=str(ROOT / "outputs" / "c20_transfer_4donor"
                                            / "transfer_matrix.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "transfer_check.csv"))
    args = ap.parse_args()

    donor = args.donor or DS.case_study()
    mat = pd.read_csv(args.matrix, encoding="utf-8-sig")
    t1 = pd.read_csv(ROOT / "results" / "table1_v18_final.csv")

    if "donor" in mat.columns:
        sub = mat[mat["donor"].astype(str) == donor].copy()
    else:                                     # single-donor matrix
        sub = mat.copy()
    if sub.empty:
        print("[transfer_check] donor %r not in %s" % (donor, args.matrix))
        return 1

    tgt_col = "target" if "target" in sub.columns else "alias"
    t1 = t1.set_index("dataset")
    rows = []
    for _, r in sub.iterrows():
        tgt = str(r[tgt_col])
        rows.append(dict(alias=tgt,
                         n=float(t1.loc[tgt, "n_rows"]) if tgt in t1.index else float("nan"),
                         target_mean=float(t1.loc[tgt, "mean_yield"]) if tgt in t1.index else float("nan"),
                         source_mean=float(t1.loc[donor, "mean_yield"]) if donor in t1.index else float("nan"),
                         r2_transfer=float(r["r2_transfer"])))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print("[transfer_check] %s -> %s : %d targets, mean r2 %+0.6f"
          % (donor, args.out, len(out), out.r2_transfer.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

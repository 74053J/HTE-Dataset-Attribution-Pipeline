# -*- coding: utf-8 -*-
"""Producer for the §3.4 signed-SHAP panel (paper Fig. 4a/4b): ``results/panel_signed_shap.csv``.

The paper quotes mean signed SHAP values per reactant level (e.g. 6-chloronicotinonitrile
+27.5 yield points, 6-bromonicotinonitrile +23.3, 6-iodonicotinonitrile −13.3).  Those
numbers used to live only inside the rendered figure; this script re-derives them from the
same code path the figure uses (``scripts/figures.py`` helpers), so the table and the panel
cannot drift apart.

Output columns
    slot            reactant_1 / reactant_2
    level           raw level name as printed in Fig. 4
    n               number of test rows carrying that level
    mean_shap       mean signed SHAP (yield points)  <- the paper's number
    mean_abs_shap   mean |SHAP|
    share           mean_abs_shap as a percentage of the cell's total |SHAP|

usage
    python scripts/build_panel_signed_shap.py [--repo DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    scripts = repo / "scripts"
    sys.path.insert(0, str(scripts))
    sys.path.insert(0, str(repo))
    sys.stdout.reconfigure(encoding="utf-8")

    import figures as F  # noqa: E402  (the figure module is the reference implementation)

    F._DATA_OVERRIDE = repo / "data"
    state = F._load_bh()
    shap = F._shap()
    cols, values = shap["cols"], shap["values"]
    df, te_pos = state["df"], state["te_pos"]

    # same raw-column candidates, same order, as the two Fig. 4 panels
    SLOTS = [
        ("reactant_1", "reactant_1_encoded",
         ("startingmat_1_name", "substrate_name", "aryl_halide_name")),
        ("reactant_2", "reactant_2_encoded",
         ("startingmat_2_name", "leaving_group_name", "amine_name")),
    ]

    out = []
    for slot, col, candidates in SLOTS:
        if col not in cols:
            print("[skip] %s is not a column of the SHAP matrix" % col)
            continue
        j = cols.index(col)
        names = None
        for raw_col in candidates:
            if raw_col in df.columns:
                names = df[raw_col].astype(str).to_numpy()[te_pos]
                break
        if names is None:
            names = state["test"].iloc[:, j].astype(str).to_numpy()
        series = pd.Series(values[:, j])
        total = float(np.abs(values[:, j]).sum())
        grouped = series.groupby(names)
        for level, vals in grouped:
            v = np.asarray(vals, float)
            out.append(dict(
                slot=slot, level=str(level), n=int(v.size),
                mean_shap=float(v.mean()), mean_abs_shap=float(np.abs(v).mean()),
                median_shap=float(np.median(v)),
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                share=float(100.0 * np.abs(v).mean() / total) if total > 0 else float("nan"),
            ))

    frame = pd.DataFrame(out).sort_values(["slot", "mean_shap"], ascending=[True, False])
    dst = Path(args.out) if args.out else repo / "results" / "panel_signed_shap.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dst, index=False, float_format="%.4f")
    print("wrote %s (%d rows)" % (dst, len(frame)))
    for slot in frame.slot.unique():
        top = frame[frame.slot == slot].head(3)
        print("  %s top: %s" % (slot, ", ".join(
            "%s %+.1f" % (r.level, r.mean_shap) for r in top.itertuples())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

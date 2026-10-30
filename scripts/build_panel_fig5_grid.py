# -*- coding: utf-8 -*-
"""Producer for the Fig. 5 ligand x base panel: ``results/panel_fig5_grid.csv``
and ``results/panel_fig5_margins.csv``.

The paper's §3.4 quotes the IHept x KOtBu cell (68.9%) against the IHept row mean
(16.9%), the KOtBu column mean (18.7%) and the grand mean (9.6%), i.e. a coupling of
+42.9 points.  Those numbers existed only inside the rendered heatmap; this script
re-derives them, and reproduces all four:

  * cell               = mean yield of the rows with that ligand and that base
  * row / column means = mean yield over every row of that ligand / base
  * grand mean         = mean of the per-cell means (unweighted, 393 cells)

Outputs
    results/panel_fig5_grid.csv     every observed (ligand, base) cell, with n, mean
                                    yield and an ``in_panel`` flag for the top-14 x
                                    top-10 selection the figure draws
    results/panel_fig5_margins.csv  ligand / base marginals (weighted), grand means

usage
    python scripts/build_panel_fig5_grid.py [--repo DIR] [--outdir DIR] [--n-lig 14] [--n-base 10]
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
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n-lig", type=int, default=14)
    ap.add_argument("--n-base", type=int, default=10)
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo))
    sys.stdout.reconfigure(encoding="utf-8")

    import figures as F  # noqa: E402

    F._DATA_OVERRIDE = repo / "data"
    state = F._load_bh()
    df = state["df"]
    needed = ["catalyst_name", "reagent_1_name", "yield"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit("case-study dataset is missing %s" % ", ".join(missing))

    data = df[needed].dropna().copy()
    counts = data.pivot_table(index="catalyst_name", columns="reagent_1_name",
                              values="yield", aggfunc="size").fillna(0)
    keep_l = list(counts.sum(axis=1).sort_values(ascending=False).head(args.n_lig).index)
    keep_b = list(counts.sum(axis=0).sort_values(ascending=False).head(args.n_base).index)

    cell_n = data.pivot_table(index="catalyst_name", columns="reagent_1_name",
                              values="yield", aggfunc="size")
    cell_m = data.pivot_table(index="catalyst_name", columns="reagent_1_name",
                              values="yield", aggfunc="mean")

    grid = []
    for lig in cell_m.index:
        for base in cell_m.columns:
            v = cell_m.loc[lig, base]
            if pd.isna(v):
                continue
            grid.append(dict(ligand=str(lig), base=str(base),
                             n=int(cell_n.loc[lig, base]), mean_yield=float(v),
                             in_panel=bool(lig in keep_l and base in keep_b)))
    grid_frame = pd.DataFrame(grid).sort_values(["in_panel", "mean_yield"],
                                                ascending=[False, False])

    marg = []
    for kind, series in (("ligand", data.groupby("catalyst_name")["yield"]),
                         ("base", data.groupby("reagent_1_name")["yield"])):
        agg = series.agg(["mean", "size"])
        for name, row in agg.iterrows():
            marg.append(dict(kind=kind, name=str(name), n=int(row["size"]),
                             mean_yield=float(row["mean"])))

    grand_all = float(data["yield"].mean())
    grand_cell_means = float(np.nanmean(cell_m.to_numpy(float)))
    grand_panel_rows = float(data[data.catalyst_name.isin(keep_l) &
                                  data.reagent_1_name.isin(keep_b)]["yield"].mean())
    for kind, val in (("grand_all_rows", grand_all),
                      ("grand_cell_means", grand_cell_means),
                      ("grand_panel_rows", grand_panel_rows)):
        marg.append(dict(kind=kind, name="", n=int(len(data)), mean_yield=val))
    marg_frame = pd.DataFrame(marg)

    outdir = Path(args.outdir) if args.outdir else repo / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    g_dst = outdir / "panel_fig5_grid.csv"
    m_dst = outdir / "panel_fig5_margins.csv"
    grid_frame.to_csv(g_dst, index=False, float_format="%.4f")
    marg_frame.to_csv(m_dst, index=False, float_format="%.4f")
    print("wrote %s (%d cells, %d inside the panel)"
          % (g_dst, len(grid_frame), int(grid_frame.in_panel.sum())))
    print("wrote %s (%d rows)" % (m_dst, len(marg_frame)))

    def pick(kind, name):
        hit = marg_frame[(marg_frame.kind == kind) & (marg_frame.name == name)]
        return float(hit.mean_yield.iloc[0]) if len(hit) else float("nan")

    cell = float(cell_m.loc["IHept", "KOtBu"]) if ("IHept" in cell_m.index and "KOtBu" in cell_m.columns) else float("nan")
    row_m = pick("ligand", "IHept")
    col_m = pick("base", "KOtBu")
    print("  IHept x KOtBu cell %.2f%% (n=%d)" % (cell, int(cell_n.loc["IHept", "KOtBu"])))
    print("  IHept row mean %.2f%% | KOtBu column mean %.2f%% | grand (cell means) %.2f%%"
          % (row_m, col_m, grand_cell_means))
    if np.isfinite([cell, row_m, col_m, grand_cell_means]).all():
        print("  coupling %+.1f points above the grand mean"
              % (cell - row_m - col_m + grand_cell_means))
    print("  grand: all rows %.2f%% | panel rows %.2f%% | cell means %.2f%%"
          % (grand_all, grand_panel_rows, grand_cell_means))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

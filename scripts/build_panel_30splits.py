# -*- coding: utf-8 -*-
"""Producer for the split-stability table: ``results/panel_30splits.csv``.

The manuscript claims that the held-out R² of the *shipped configuration* is stable across
fresh 80/20 splits (previously quoted as "SD 0.011 to 0.048 over 30 splits").  No table
backed that sentence.  This script re-derives it: for each dataset it refits the
validation-selected configuration (from ``results/table1_v18_final.csv``) on N independent
80/20 splits and reports the distribution of the test R².

Output columns
    dataset, model, encoding, split_seed, n_train, n_test, r2_test

A per-dataset summary (n, mean, sd, min, max) is printed and written to
``results/panel_30splits_summary.csv``.

usage
    python scripts/build_panel_30splits.py [--repo DIR] [--splits 30] [--datasets A,B] \
                                           [--outdir DIR] [--seed0 1]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--datasets", default="", help="comma-separated aliases (default: all 13)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo))
    sys.stdout.reconfigure(encoding="utf-8")

    import protocol as L  # noqa: E402
    from sklearn.metrics import r2_score  # noqa: E402
    from hte.features import get_split_positions, prepare_features  # noqa: E402
    from hte.grid import build_frames  # noqa: E402

    table1 = pd.read_csv(repo / "results" / "table1_v18_final.csv")
    wanted = {r.dataset: (r.best_model, r.best_encoding) for r in table1.itertuples()}
    aliases = [a.strip() for a in args.datasets.split(",") if a.strip()] or list(L.ALIASES13)

    rows = []
    t_all = time.perf_counter()
    for alias in aliases:
        if alias not in wanted:
            print("[skip] %s is not in table1_v18_final.csv" % alias)
            continue
        model_name, enc_name = wanted[alias]
        df, raw = L.load(alias)
        y_all = df["yield"].to_numpy(float)
        t0 = time.perf_counter()
        for seed in range(args.seed0, args.seed0 + args.splits):
            tr, te = get_split_positions(len(df), random_state=seed)
            X, y, feature_cols, _disp, _enc, _sc = prepare_features(df, raw, fit_rows=tr)
            packs, _cc, cat_idx = build_frames(X, feature_cols, tr, te)
            cells = L.grid_cells_threaded(cat_idx)
            hit = [c for c in cells
                   if str(c.architecture) == str(model_name) and str(c.encoding) == str(enc_name)]
            if not hit:
                raise SystemExit("%s: no cell matches %s / %s" % (alias, model_name, enc_name))
            cell = hit[0]
            Xtr, Xte = packs[cell.frame]
            yhat = L.fit_predict(cell, Xtr, y.iloc[tr], Xte)
            rows.append(dict(dataset=alias, model=model_name, encoding=enc_name,
                             split_seed=seed, n_train=int(len(tr)), n_test=int(len(te)),
                             r2_test=float(r2_score(y.iloc[te], yhat))))
        sub = [r["r2_test"] for r in rows if r["dataset"] == alias]
        print("[%-18s] %s/%s  n=%d  mean %.4f  sd %.4f  min %.4f  max %.4f  [%.0fs]"
              % (alias, model_name, enc_name, len(sub), float(np.mean(sub)),
                 float(np.std(sub, ddof=1)) if len(sub) > 1 else float("nan"),
                 float(np.min(sub)), float(np.max(sub)), time.perf_counter() - t0), flush=True)

    frame = pd.DataFrame(rows)
    outdir = Path(args.outdir) if args.outdir else repo / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / "panel_30splits.csv"
    frame.to_csv(dst, index=False, float_format="%.6f")
    summary = (frame.groupby(["dataset", "model", "encoding"])["r2_test"]
               .agg(n="size", mean="mean", sd="std", min="min", max="max").reset_index())
    sdst = outdir / "panel_30splits_summary.csv"
    summary.to_csv(sdst, index=False, float_format="%.6f")
    print("\nwrote %s (%d rows) and %s (%d datasets) in %.0fs"
          % (dst, len(frame), sdst, len(summary), time.perf_counter() - t_all))
    if len(summary):
        print("  sd range: %.4f to %.4f  (median %.4f)"
              % (summary.sd.min(), summary.sd.max(), summary.sd.median()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

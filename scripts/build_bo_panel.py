#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 10c -- the four panels of the Bayesian-optimisation figure.

``figures.py`` looks for ``results/bo_panel_data.csv`` (or the older
``Figure9_..._panel_data.csv``) and raises ``FigureInputMissing`` when neither
exists; nothing in the package wrote it.  This script does, in the same long
format the shipped file uses: ``panel, series, x, y, value, note``.

  panel a  cumulative best predicted yield under three orderings (100 steps)
  panel b  ligand x base grid of the ensemble prediction
  panel c  base x solvent grid
  panel d  ensemble mean and +/-1 SD against the catalyst loading (61 points)

Everything is a model prediction -- the shipped table carries no measured BO
trajectory -- so the same fitted panel that ``build_acquisition.py`` uses is
enough; no optimisation experiment is run.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AXES = ["metal", "catalyst", "ligand", "base", "solvent", "reactant_2", "reactant_1"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import protocol as L
    from hte import datasets as DS
    from hte.v16_core import get_split_positions, prepare_features
    from build_acquisition import _panel

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--points", type=int, default=100)
    ap.add_argument("--loading-points", type=int, default=61)
    ap.add_argument("--out", default=str(ROOT / "results" / "bo_panel_data.csv"))
    args = ap.parse_args()
    alias = args.dataset or DS.case_study()

    with contextlib.redirect_stdout(io.StringIO()):
        df, raw = L.load(alias)
    tr, te = get_split_positions(len(df))
    with contextlib.redirect_stdout(io.StringIO()):
        X, y, fcols, _d, enc, _s = prepare_features(df, raw, fit_rows=tr)

    cat_cols = [c for c in fcols if str(c).endswith("_encoded")]
    Xc = X.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].fillna(-1).astype("int64").astype("category")
    models, frames = {}, {}
    for name, m in _panel(42, 4).items():
        frame = Xc if name in ("XGBoost", "HistGB") else X
        if name == "HistGB":
            idx_cat = [i for i, c in enumerate(fcols) if str(c).endswith("_encoded")]
            m.set_params(categorical_features=(idx_cat or None))
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(frame.iloc[tr], y.iloc[tr])
        models[name] = m
        frames[name] = frame
    print("[bo] panel fitted on %s (%d rows)" % (alias, len(df)))

    def predict(frame_rows):
        """mean and sd across the five models for a frame of feature rows"""
        preds = []
        for name, m in models.items():
            use = frame_rows
            if name in ("XGBoost", "HistGB"):
                use = frame_rows.copy()
                for c in cat_cols:
                    use[c] = use[c].astype("int64").astype("category")
            preds.append(np.asarray(m.predict(use), dtype=float))
        p = np.vstack(preds)
        return p.mean(0), p.std(0)

    def frame_for(n):
        base = {}
        for c in fcols:
            col = X[c].iloc[tr]
            base[c] = int(col.mode().iloc[0]) if c in cat_cols else float(
                pd.to_numeric(col, errors="coerce").median())
        return pd.DataFrame({c: np.repeat(base[c], n) for c in fcols}), base

    rows = []

    # ---- panel a: three orderings, cumulative best predicted yield
    levels = {}
    for a in AXES:
        le = enc.get(a) if hasattr(enc, "get") else None
        levels[a] = ([str(v) for v in le.classes_] if le is not None and hasattr(le, "classes_")
                     else (sorted(set(df[a].astype(str))) if a in df.columns else ["none"]))
    if "metal" in df.columns:
        levels["metal"] = [str(df["metal"].astype(str).value_counts().index[0])]
    idx = np.array(np.meshgrid(*[np.arange(len(levels[a])) for a in AXES],
                               indexing="ij")).reshape(len(AXES), -1).T
    fr, _base = frame_for(len(idx))
    for j, a in enumerate(AXES):
        col = a + "_encoded"
        if col in fr.columns:
            fr[col] = idx[:, j].astype("int64")
    mu, sd = predict(fr)
    order_mu = np.argsort(-mu)
    # Expected improvement, not a normalised gap: EI(x) = (mu - best)*Phi(z) + sd*phi(z)
    from scipy.stats import norm
    best = float(mu.max())
    _z = (mu - best) / np.maximum(sd, 1e-9)
    ei = (mu - best) * norm.cdf(_z) + sd * norm.pdf(_z)
    rng = np.random.default_rng(42)
    order_rand = rng.permutation(len(mu))
    for series, order in (("acquisition order (EI)", np.argsort(-ei)),
                          ("predicted-yield order", order_mu),
                          ("random order", order_rand)):
        cum = np.maximum.accumulate(mu[order])[: args.points]
        for k, v in enumerate(cum, 1):
            rows.append(dict(panel="a", series=series, x=k, y=float(v), value=np.nan,
                             note="cumulative best PREDICTED yield (%)"))
    print("[bo] panel a from %d virtual conditions" % len(mu))

    # ---- panels b and c: two-dimensional prediction grids
    for panel, ax_x, ax_y in (("b", "ligand", "base"), ("c", "base", "solvent")):
        lx, ly = levels[ax_x], levels[ax_y]
        n = len(lx) * len(ly)
        g, _b = frame_for(n)
        g[ax_x + "_encoded"] = np.repeat(np.arange(len(lx)), len(ly))
        g[ax_y + "_encoded"] = np.tile(np.arange(len(ly)), len(lx))
        gm, _gs = predict(g)
        for i, vx in enumerate(lx):
            for j, vy in enumerate(ly):
                rows.append(dict(panel=panel,
                                 series=("Ligand x Base" if panel == "b" else "Base x Solvent"),
                                 x=vx, y=vy, value=float(gm[i * len(ly) + j]),
                                 note="ensemble predicted yield (%)"))
        print("[bo] panel %s: %d x %d" % (panel, len(lx), len(ly)))

    # ---- panel d: ensemble mean and +/-1 SD against the ligand loading
    # (the case-study dataset has no catalyst-loading axis: its catalyst slot is
    # disabled because that column holds the ligand, so the dose axis is
    # ligand_eq -- decided with the author on 2026-10-28)
    load_col = next((c for c in fcols if str(c) == "ligand_eq"), None) or \
        next((c for c in fcols if "loading" in str(c)), None)
    if load_col is not None:
        base_fr, _bf = frame_for(1)
        lo = float(pd.to_numeric(X[load_col].iloc[tr], errors="coerce").quantile(0.05))
        hi = float(pd.to_numeric(X[load_col].iloc[tr], errors="coerce").quantile(0.95))
        grid = np.linspace(lo, hi, args.loading_points)
        g = pd.concat([base_fr] * len(grid), ignore_index=True)
        g[load_col] = grid
        gm, gs = predict(g)
        for k in range(len(grid)):
            rows.append(dict(panel="d", series="ensemble mean (weighted)", x=float(grid[k]),
                             y=float(gm[k]), value=float(gm[k]),
                             note="x = ligand loading (eq.); value = predicted yield (%)"))
            rows.append(dict(panel="d", series="ensemble mean + 1 SD", x=float(grid[k]),
                             y=float(gm[k] + gs[k]), value=np.nan, note=""))
            rows.append(dict(panel="d", series="ensemble mean - 1 SD", x=float(grid[k]),
                             y=float(gm[k] - gs[k]), value=np.nan, note=""))
        print("[bo] panel d: %d loading points" % len(grid))
    else:
        print("[bo] panel d skipped: no loading column")

    out = pd.DataFrame(rows, columns=["panel", "series", "x", "y", "value", "note"])
    out.to_csv(args.out, index=False, encoding="utf-8")
    print("[bo] wrote %s (%d rows)" % (args.out, len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

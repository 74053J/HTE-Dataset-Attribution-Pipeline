# -*- coding: utf-8 -*-
"""Producer for the two confidence intervals the manuscript quotes: ``results/rho_ci.csv``.

    --statistic ceiling   ρ(noise ceiling, held-out R²) over the 11 datasets with a ceiling
                          (paper: ρ = +0.709, 95% CI 0.167–0.990)
    --statistic size      ρ(rows, held-out R²) over the same 11 datasets
                          (paper: ρ = −0.100, 95% CI −0.714 to +0.705)

Both are dataset-level percentile bootstraps: resample the datasets with replacement,
recompute Spearman ρ, take the 2.5/97.5 percentiles.  n = 11 is small, so the interval is
seed-sensitive; the script therefore writes the primary run (seed 42, B=1000) **and** three
sensitivity rows (seed 7, seed 999, B=5000) so the manuscript can name the method instead
of relying on a single unseeded draw.

Output columns
    statistic, n, rho, ci_low, ci_high, B, seed, method, datasets, input_sha256,
    script_sha256, role  (role = primary | sensitivity)

usage
    python scripts/build_panel_rho_ci.py --statistic ceiling [--repo DIR] [--out FILE]
    python scripts/build_panel_rho_ci.py --statistic size    [--append]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--statistic", choices=["ceiling", "size"], default="ceiling")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default=None)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append", action="store_true", help="append rows instead of rewriting")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd
    from scipy import stats

    repo = Path(args.repo).resolve()
    table_path = repo / "results" / "table1_v18_final.csv"
    table = pd.read_csv(table_path)
    sub = table[np.isfinite(pd.to_numeric(table.ceiling, errors="coerce"))].copy()
    y = pd.to_numeric(sub.r2_best, errors="coerce").to_numpy(float)
    x = (pd.to_numeric(sub.ceiling, errors="coerce").to_numpy(float)
         if args.statistic == "ceiling" else
         pd.to_numeric(sub.n_rows, errors="coerce").to_numpy(float))
    rho = float(stats.spearmanr(x, y).statistic)

    def ci(seed, B):
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(B):
            idx = rng.integers(0, len(x), len(x))
            xx, yy = x[idx], y[idx]
            if np.ptp(xx) == 0 or np.ptp(yy) == 0:
                continue
            v = stats.spearmanr(xx, yy).statistic
            if np.isfinite(v):
                draws.append(float(v))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return float(lo), float(hi)

    in_sha = _sha256(table_path)
    sc_sha = _sha256(Path(__file__).resolve())
    datasets = ";".join(str(d) for d in sub.dataset)

    rows = []
    lo, hi = ci(args.seed, args.B)
    rows.append(dict(statistic=args.statistic, n=len(x), rho=round(rho, 6),
                     ci_low=round(lo, 6), ci_high=round(hi, 6), B=args.B, seed=args.seed,
                     method="dataset_level_percentile", datasets=datasets,
                     input_sha256=in_sha, script_sha256=sc_sha, role="primary"))
    for seed, B in ((7, 1000), (999, 1000), (args.seed, 5000)):
        lo, hi = ci(seed, B)
        rows.append(dict(statistic=args.statistic, n=len(x), rho=round(rho, 6),
                         ci_low=round(lo, 6), ci_high=round(hi, 6), B=B, seed=seed,
                         method="dataset_level_percentile", datasets=datasets,
                         input_sha256=in_sha, script_sha256=sc_sha, role="sensitivity"))

    out = Path(args.out) if args.out else repo / "results" / "rho_ci.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    mode = "a" if args.append and out.exists() else "w"
    with open(out, mode, encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            w.writeheader()
        w.writerows(rows)

    print("statistic :", args.statistic, "| n =", len(x), "| rho = %.4f" % rho)
    for r in rows:
        print("   %-11s seed %-3d B %-5d -> %+.4f .. %+.4f"
              % (r["role"], r["seed"], r["B"], r["ci_low"], r["ci_high"]))
    print("wrote", out, "(mode %s)" % mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

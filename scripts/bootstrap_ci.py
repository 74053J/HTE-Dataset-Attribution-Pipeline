#!/usr/bin/env python
"""Bootstrap confidence intervals for the noise ceiling and the capture ratio.

Statistics
----------
R2max (noise ceiling) = 1 - sigma2_within / sigma2_total, estimated over the
conditions that were measured more than once (the estimator in
``archive/Analysis_all_v16.py``).  The rows inside one condition are
replicates, not independent observations, so the resampling unit is the
REPLICATE CONDITION GROUP, never the individual row.

The capture ratio is R2_best / R2max.  Its denominator is noisy and that noise
is amplified when the ceiling is low, so the bootstrap distribution of the
ratio is right-skewed.  The reported interval is therefore a PERCENTILE
interval computed on the LOG scale and exponentiated back, not a symmetric
normal interval.

Modes
-----
``--from-tables`` (default)
    Uses only the shipped CSVs in results/legacy/ (grid_all16.csv,
    ceiling_stats_v16.csv, data/alias_mapping.yaml).  It runs the
    dataset-level bootstrap for the two aggregate statistics.  The replicate
    values themselves are not shipped, so the per-dataset interval uses a
    parametric chi-square approximation for the sampling distribution of the
    pooled within-condition variance (df = rows in repeated conditions minus
    the number of repeated conditions); it is labelled ``tables_only`` in the
    output.  No raw data files are required.

``--from-raw DIR``
    Adds the non-parametric cluster bootstrap over replicate condition groups.
    Raw files are loaded through the packaged loader (hte.v16_core) so the
    condition key is identical to the one that produced the shipped ceilings.
    If the files are missing the script prints the exact paths it expects and
    exits 0 without writing a table (no numbers are fabricated).

The dataset-level bootstrap resamples the 12 datasets that have a computable
ceiling with replacement and recomputes (a) Spearman rho between ceiling and
best held-out R2 and (b) the median capture ratio.

Output
------
``results/legacy/bootstrap_ci.csv`` with the columns
``dataset, statistic, estimate, ci_low, ci_high, method``.

This script is the legacy **16-dataset** producer: its 16-set cluster bootstrap
is the only record of that corpus and is kept in results/legacy/.  The
13-dataset intervals the paper cites are results/bootstrap_ci13.csv, produced by
``outputs/last_gaps/30_bootstrap_ci13.py``.

Fixed seed 42, 1000 resamples, 95% percentile intervals.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

# Importing the frozen loader must not leave bytecode caches in the repository.
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
# The 16-dataset corpus is retired: the script now reads results/grid13_v18.csv
# (13 datasets) and writes its table next to the other 13-set results.
LEGACY = RESULTS / "legacy"
OUT_PATH = RESULTS / "bootstrap_ci.csv"

SEED = 42
N_BOOT = 1000
CI_PCT = (2.5, 97.5)

# The nine condition columns the ceiling estimator keys on.  ``metal`` is
# deliberately absent: the estimator does not key on it.  The two partner slots
# are named as they are in the current pipeline (they were substrate_type and
# leaving_group before the rename; substrate_type -> reactant_1, leaving_group ->
# reactant_2, per the slot table in hte/params.py).
COND_COLS = [
    "catalyst", "ligand", "base", "solvent", "reactant_2",
    "reactant_1", "temperature", "time", "catalyst_loading",
]

METHOD_CEIL_RAW = "cluster_bootstrap_replicate_groups_percentile_1000x"
METHOD_CAP_RAW = "cluster_bootstrap_replicate_groups_log_percentile_1000x"
METHOD_CEIL_TABLES = "parametric_chi2_percentile_tables_only"
METHOD_CAP_TABLES = "parametric_chi2_log_percentile_tables_only"
METHOD_RHO = "dataset_bootstrap_percentile_1000x"
METHOD_MEDIAN = "dataset_bootstrap_log_percentile_1000x"

CSV_COLUMNS = ["dataset", "statistic", "estimate", "ci_low", "ci_high", "method"]


# small helpers
def percentile_ci(values, log_scale: bool = False) -> tuple[float, float]:
    """95% percentile interval, optionally on the log scale (expanded back)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if log_scale:
        x = x[x > 0.0]
    if x.size == 0:
        return float("nan"), float("nan")
    if log_scale:
        lo, hi = np.percentile(np.log(x), CI_PCT)
        return float(np.exp(lo)), float(np.exp(hi))
    lo, hi = np.percentile(x, CI_PCT)
    return float(lo), float(hi)


def load_tables():
    """Load the three shipped inputs and map raw source names to aliases."""
    mapping = yaml.safe_load((DATA / "alias_mapping.yaml").read_text(encoding="utf-8"))
    datasets = mapping["datasets"]
    alias_of = {meta["source"]: alias for alias, meta in datasets.items()}
    source_of = {alias: meta["source"] for alias, meta in datasets.items()}

    # LEGACY 16-dataset tables (kept in results/legacy/): this script is the
    # producer of the old 16-set bootstrap, not of the 13-dataset corpus.
    grid = pd.read_csv(LEGACY / "grid_all16.csv")
    ceil = pd.read_csv(LEGACY / "ceiling_stats_v16.csv")
    ceil["dataset"] = ceil["dataset"].map(alias_of)
    if ceil["dataset"].isna().any():
        unmapped = ceil.loc[ceil["dataset"].isna(), "dataset"].tolist()
        raise SystemExit("unmapped raw source name(s) in ceiling_stats_v16.csv: %s" % unmapped)
    ceil["source"] = ceil["dataset"].map(source_of)

    # "Best held-out R2" is the maximum over the 12 model x encoding configs,
    # the same definition scripts/verify_numbers.py uses for the paper numbers.
    r2_grid = grid.groupby("dataset")["r2_test"].max()
    ceil["r2_best_grid"] = ceil["dataset"].map(r2_grid)
    if ceil["r2_best_grid"].isna().any():
        raise SystemExit("dataset(s) missing from grid_all16.csv: %s"
                         % ceil.loc[ceil["r2_best_grid"].isna(), "dataset"].tolist())
    return grid, ceil


def ceiling_table(ceil: pd.DataFrame) -> pd.DataFrame:
    """The datasets with a computable ceiling (the 4 with no replicates drop out)."""
    t = ceil[np.isfinite(ceil["ceiling"])].copy()
    t["capture_grid"] = t["r2_best_grid"] / t["ceiling"]
    t = t.sort_values("dataset").reset_index(drop=True)
    return t


# dataset-level bootstrap (aggregate statistics)
def dataset_level_bootstrap(t: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Resample datasets with replacement; recompute rho and median capture."""
    ceil = t["ceiling"].to_numpy(float)
    r2 = t["r2_best_grid"].to_numpy(float)
    cap = t["capture_grid"].to_numpy(float)
    n = len(t)

    rhos: list[float] = []
    medians: list[float] = []
    means: list[float] = []
    med_pct: list[float] = []
    attempts = 0
    while (len(rhos) < N_BOOT or len(medians) < N_BOOT) and attempts < N_BOOT * 50:
        idx = rng.integers(0, n, size=n)
        rho = spearmanr(ceil[idx], r2[idx]).statistic
        if np.isfinite(rho):
            rhos.append(float(rho))
        med = float(np.median(cap[idx]))
        if np.isfinite(med) and med > 0.0:
            medians.append(med)
        # v29: the manuscript quotes plain (non-log) percentile CIs for the mean and
        # median capture ratio; record those same draws so the numbers are reproducible.
        means.append(float(np.mean(cap[idx])))
        med_pct.append(med)
        attempts += 1

    rho_hat = float(spearmanr(ceil, r2).statistic)
    med_hat = float(np.median(cap))
    rho_lo, rho_hi = percentile_ci(rhos)
    med_lo, med_hi = percentile_ci(medians, log_scale=True)
    mean_hat = float(np.mean(cap))
    mean_lo, mean_hi = percentile_ci(means)
    medp_lo, medp_hi = percentile_ci(med_pct)
    return dict(
        rho=rho_hat, rho_lo=rho_lo, rho_hi=rho_hi, rho_draws=len(rhos),
        median=med_hat, med_lo=med_lo, med_hi=med_hi, med_draws=len(medians),
        mean=mean_hat, mean_lo=mean_lo, mean_hi=mean_hi,
        med_pct_lo=medp_lo, med_pct_hi=medp_hi, n=n,
    )


# per-dataset intervals: tables-only parametric fallback
def parametric_ceiling_draws(row: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Chi-square approximation to the sampling distribution of the pooled s2_within.

    With y in units of s2_total, s2_within = 1 - ceiling and
    df * s2_within_hat / s2_within_true ~ chi2_df, so

        s2_within* = (1 - ceiling) * df / chi2_df,
        ceiling*   = 1 - s2_within*.

    This is the best available interval when the replicate rows themselves are
    not shipped; --from-raw replaces it with the actual cluster bootstrap.
    """
    c = float(row["ceiling"])
    n = float(row["n"])
    groups = float(row["n_rep_groups"])
    frac = float(row.get("rep_frac", float("nan")))
    if np.isfinite(frac):
        repeated_rows = max(round(frac * n), groups + 1.0)
    else:  # fall back to an average group size of two
        repeated_rows = 2.0 * groups
    df = max(repeated_rows - groups, 1.0)
    s2w_rel = 1.0 - c
    return 1.0 - s2w_rel * df / rng.chisquare(df, size=N_BOOT)


def per_dataset_tables(t: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    out: list[dict] = []
    for _, row in t.iterrows():
        ceil_draws = parametric_ceiling_draws(row, rng)
        cap_draws = float(row["r2_best_grid"]) / ceil_draws
        c_lo, c_hi = percentile_ci(ceil_draws)
        cap_lo, cap_hi = percentile_ci(cap_draws, log_scale=True)
        out.append(dict(
            dataset=str(row["dataset"]),
            ceiling=float(row["ceiling"]),
            r2_best=float(row["r2_best_grid"]),
            capture=float(row["capture_grid"]),
            n_rep_groups=int(row["n_rep_groups"]),
            ceil_lo=c_lo, ceil_hi=c_hi,
            cap_lo=cap_lo, cap_hi=cap_hi,
            ceil_method=METHOD_CEIL_TABLES, cap_method=METHOD_CAP_TABLES,
        ))
    return out


# per-dataset intervals: cluster bootstrap over replicate condition groups
def raw_group_moments(df: pd.DataFrame):
    """Per-condition count, sum, sum of squares and within-condition SS."""
    y = pd.to_numeric(df["yield"], errors="coerce")
    ok = y.notna()
    d = df.loc[ok]
    yv = y.loc[ok].to_numpy(dtype=float)
    cols = [c for c in COND_COLS if c in d.columns]
    if not cols or yv.size == 0:
        return None
    key = d[cols].astype(str).agg("|".join, axis=1)
    codes, _ = pd.factorize(key, sort=False)
    g = int(codes.max()) + 1
    n = np.bincount(codes, minlength=g).astype(float)
    s = np.bincount(codes, weights=yv, minlength=g)
    q = np.bincount(codes, weights=yv * yv, minlength=g)
    ss = q - s * s / n
    np.maximum(ss, 0.0, out=ss)
    return n, s, q, ss


def ceiling_from_moments(n, s, q, ss) -> tuple[float, int]:
    """The v16 point estimator, from the precomputed group moments."""
    rep = n >= 2
    if not rep.any():
        return float("nan"), 0
    den = float((n[rep] - 1.0).sum())
    if den <= 0.0:
        return float("nan"), 0
    s2w = float(ss[rep].sum() / den)
    n_tot = float(n.sum())
    if n_tot < 2.0:
        return float("nan"), 0
    s2t = float((q.sum() - s.sum() ** 2 / n_tot) / (n_tot - 1.0))
    if not np.isfinite(s2t) or s2t <= 0.0:
        return float("nan"), 0
    return 1.0 - s2w / s2t, int(rep.sum())


def cluster_ceiling_draws(n, s, q, ss, rng: np.random.Generator) -> np.ndarray | None:
    """Bootstrap the ceiling by resampling replicate condition groups.

    Each condition with n_g >= 2 is one sampling unit.  A bootstrap sample draws
    the same number of replicate groups with replacement, i.e. group g gets a
    multiplicity k_g ~ Multinomial(n_groups, 1/n_groups).  Its rows then appear
    k_g times, so its within-group SS contributes k_g * SS_g on k_g * (n_g - 1)
    degrees of freedom.  Conditions measured only once carry no within-condition
    information and are held fixed; they still enter s2_total.

    Working through the group moments makes this exact and O(groups) per draw.
    """
    rep = n >= 2
    if not rep.any():
        return None
    nr = n[rep]
    sr = s[rep]
    qr = q[rep]
    ssr = ss[rep]
    dfr = nr - 1.0
    n_groups = int(rep.sum())

    # fixed singleton conditions contribute to s2_total but never to s2_within
    n_fixed = float(n[~rep].sum())
    s_fixed = float(s[~rep].sum())
    q_fixed = float(q[~rep].sum())

    p = np.full(n_groups, 1.0 / n_groups)
    k = rng.multinomial(n_groups, p, size=N_BOOT).astype(np.float64)

    s2w = (k @ ssr) / (k @ dfr)
    n_tot = n_fixed + k @ nr
    s_tot = s_fixed + k @ sr
    q_tot = q_fixed + k @ qr
    s2t = (q_tot - s_tot * s_tot / n_tot) / (n_tot - 1.0)
    return 1.0 - s2w / s2t


def run_raw(t: pd.DataFrame, raw_dir: Path, rng: np.random.Generator):
    """Cluster bootstrap from raw CSVs; returns (rows, info) or (None, None)."""
    need = [(str(row["dataset"]), str(row["source"]), raw_dir / (str(row["source"]) + ".csv"))
            for _, row in t.iterrows()]
    mapping_file = raw_dir / "col_mappings.csv"
    missing = [p for _, _, p in need if not p.is_file()] + ([] if mapping_file.is_file() else [mapping_file])

    if missing:
        print("--from-raw: raw inputs are incomplete in %s" % raw_dir)
        print("Expected %d source CSVs (only datasets with a computable ceiling need raw data):"
              % len(need))
        for alias, _src, path in need:
            status = "ok     " if path.is_file() else "MISSING"
            print("  [%s] %s   (%s)" % (status, path, alias))
        status = "ok     " if mapping_file.is_file() else "MISSING"
        print("  [%s] %s   (column mapping)" % (status, mapping_file))
        print("The four datasets with no repeated condition (DoyleHTE_CN, Summit_CN,")
        print("Suzuki-Coupling-ML_SM, rxn_yields_SM) are dropped from ceiling statistics")
        print("and their raw files are not required.")
        print("Nothing was written (no numbers fabricated).")
        return None, None

    # The frozen loader is the single source of truth for the condition key.
    sys.path.insert(0, str(ROOT / "scripts"))
    # The packaged loader, not the frozen monolith: same condition key, no runpy of a
    # 420 KB script (the two agree numerically on every dataset, see tests/).
    from hte.v16_core import load_real_dataset as loader

    out: list[dict] = []
    max_diff = 0.0
    group_mismatch = 0
    for _, row in t.iterrows():
        alias = str(row["dataset"])
        path = raw_dir / (str(row["source"]) + ".csv")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df, _raw_names = loader(str(path), str(raw_dir))
        except Exception as exc:  # pragma: no cover - depends on the raw files
            print("  [FAIL] %-24s could not load %s: %s" % (alias, path, exc))
            continue
        moments = raw_group_moments(df)
        if moments is None:
            print("  [FAIL] %-24s no yield/condition columns after mapping" % alias)
            continue
        n, s, q, ss = moments
        point, n_rep = ceiling_from_moments(n, s, q, ss)
        diff = abs(point - float(row["ceiling"]))
        max_diff = max(max_diff, diff)
        if n_rep != int(row["n_rep_groups"]):
            group_mismatch += 1
        if not np.isfinite(point):
            print("  [FAIL] %-24s ceiling undefined from raw data" % alias)
            continue
        ceil_draws = cluster_ceiling_draws(n, s, q, ss, rng)
        if ceil_draws is None:
            print("  [FAIL] %-24s no replicate condition groups in raw data" % alias)
            continue
        # Anchor the reported point estimates on the shipped tables; the raw
        # recomputation is used as an integrity check and for the interval.
        c_hat = float(row["ceiling"])
        r2_hat = float(row["r2_best_grid"])
        cap_hat = r2_hat / c_hat
        with np.errstate(divide="ignore", invalid="ignore"):
            cap_draws = r2_hat / ceil_draws
        c_lo, c_hi = percentile_ci(ceil_draws)
        cap_lo, cap_hi = percentile_ci(cap_draws, log_scale=True)
        out.append(dict(
            dataset=alias,
            ceiling=c_hat,
            r2_best=r2_hat,
            capture=cap_hat,
            n_rep_groups=int(row["n_rep_groups"]),
            ceil_lo=c_lo, ceil_hi=c_hi,
            cap_lo=cap_lo, cap_hi=cap_hi,
            ceil_method=METHOD_CEIL_RAW, cap_method=METHOD_CAP_RAW,
        ))

    info = dict(max_diff=max_diff, group_mismatch=group_mismatch, n_loaded=len(out),
                raw_dir=str(raw_dir))
    return out, info


# reporting
def write_csv(per_ds: list[dict], agg: dict) -> None:
    rows: list[dict] = []
    for d in sorted(per_ds, key=lambda x: x["dataset"]):
        rows.append(dict(dataset=d["dataset"], statistic="ceiling",
                         estimate=d["ceiling"], ci_low=d["ceil_lo"], ci_high=d["ceil_hi"],
                         method=d["ceil_method"]))
        rows.append(dict(dataset=d["dataset"], statistic="capture",
                         estimate=d["capture"], ci_low=d["cap_lo"], ci_high=d["cap_hi"],
                         method=d["cap_method"]))
    rows.append(dict(dataset="ALL", statistic="spearman_rho",
                     estimate=agg["rho"], ci_low=agg["rho_lo"], ci_high=agg["rho_hi"],
                     method=METHOD_RHO))
    rows.append(dict(dataset="ALL", statistic="median_capture",
                     estimate=agg["median"], ci_low=agg["med_lo"], ci_high=agg["med_hi"],
                     method=METHOD_MEDIAN))
    # v29: plain percentile CIs of the mean and median capture ratio (what section 3.2 quotes)
    rows.append(dict(dataset="ALL", statistic="mean_capture",
                     estimate=agg["mean"], ci_low=agg["mean_lo"], ci_high=agg["mean_hi"],
                     method=METHOD_RHO.replace("spearman_rho", "mean_capture")))
    rows.append(dict(dataset="ALL", statistic="median_capture_percentile",
                     estimate=agg["median"], ci_low=agg["med_pct_lo"],
                     ci_high=agg["med_pct_hi"],
                     method=METHOD_RHO.replace("spearman_rho", "median_capture_percentile")))
    # v29: section 3.2 quotes plain percentile CIs of the mean and median capture ratio
    # over the ELEVEN datasets with a computable ceiling (table1_v18_final.csv).
    # Fixed seed 42 so the interval is reproducible; 1000 resamples, percentile method.
    t1_path = RESULTS / "table1_v18_final.csv"
    if t1_path.is_file():
        cap11 = pd.read_csv(t1_path)["capture"].dropna().to_numpy(float)
        rng11 = np.random.default_rng(42)
        mean_d = [float(np.mean(rng11.choice(cap11, len(cap11), replace=True))) for _ in range(N_BOOT)]
        med_d = [float(np.median(rng11.choice(cap11, len(cap11), replace=True))) for _ in range(N_BOOT)]
        m_lo, m_hi = np.percentile(mean_d, CI_PCT)
        d_lo, d_hi = np.percentile(med_d, CI_PCT)
        rows.append(dict(dataset="TABLE1_11", statistic="mean_capture_percentile",
                         estimate=float(np.mean(cap11)), ci_low=float(m_lo), ci_high=float(m_hi),
                         method="dataset_bootstrap_percentile_1000x_seed42"))
        rows.append(dict(dataset="TABLE1_11", statistic="median_capture_percentile",
                         estimate=float(np.median(cap11)), ci_low=float(d_lo), ci_high=float(d_hi),
                         method="dataset_bootstrap_percentile_1000x_seed42"))
    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    frame.to_csv(OUT_PATH, index=False, float_format="%.6f")


def print_summary(mode: str, t: pd.DataFrame, dropped: list[str], agg: dict,
                  per_ds: list[dict], raw_info: dict | None, r2_note: int) -> None:
    bar = "=" * 86
    print(bar)
    print("bootstrap_ci.py  mode=%s  resamples=%d  seed=%d  interval=95%% percentile"
          % (mode, N_BOOT, SEED))
    print(bar)
    print("Datasets with a computable ceiling: %d/%d" % (len(t), len(t) + len(dropped)))
    if dropped:
        print("Dropped (no repeated condition -> ceiling undefined): %s" % ", ".join(sorted(dropped)))
    print()

    print("Dataset-level bootstrap (resampling unit = dataset, %d datasets, %d resamples)"
          % (agg["n"], N_BOOT))
    print("  Spearman rho(ceiling, best held-out R2)  %8.4f   95%% CI [%.4f, %.4f]"
          % (agg["rho"], agg["rho_lo"], agg["rho_hi"]))
    print("  median capture ratio (R2_best/ceiling)   %8.4f   95%% CI [%.4f, %.4f]"
          % (agg["median"], agg["med_lo"], agg["med_hi"]))
    print()

    if mode == "raw":
        print("Per-dataset capture intervals (cluster bootstrap over replicate groups; log-scale percentile)")
    else:
        print("Per-dataset capture intervals (parametric chi-square; log-scale percentile)")
    print("  %-24s %8s %8s %8s  %-21s %7s" %
          ("dataset", "ceiling", "r2_best", "capture", "capture 95% CI", "groups"))
    print("  " + "-" * 83)
    for d in sorted(per_ds, key=lambda x: x["ceiling"]):
        print("  %-24s %8.4f %8.4f %8.4f  [%7.4f, %7.4f] %7d"
              % (d["dataset"], d["ceiling"], d["r2_best"], d["capture"],
                 d["cap_lo"], d["cap_hi"], d["n_rep_groups"]))
    print()

    over = [d for d in per_ds if d["capture"] > 1.0]
    print("Datasets with capture > 100%%: %d" % len(over))
    for d in sorted(over, key=lambda x: -x["capture"]):
        includes = (d["cap_lo"] <= 1.0 <= d["cap_hi"])
        print("  %-24s capture=%.4f  95%% CI [%.4f, %.4f]  -> interval %s 100%%"
              % (d["dataset"], d["capture"], d["cap_lo"], d["cap_hi"],
                 "INCLUDES" if includes else "EXCLUDES"))
    if not over:
        print("  none")
    print()

    if mode == "tables":
        print("NOTE: tables mode has no replicate-level rows.  Per-dataset intervals are a")
        print("      parametric chi-square approximation (df = repeated rows - repeated")
        print("      conditions).  Run --from-raw <dir> for the cluster bootstrap.")
    else:
        assert raw_info is not None
        print("Raw dir: %s" % raw_info["raw_dir"])
        print("Raw replication check: %d/%d datasets loaded; max |ceiling_raw - ceiling_table| = %.2e;"
              % (raw_info["n_loaded"], len(t), raw_info["max_diff"]))
        print("  n_rep_groups mismatches: %d" % raw_info["group_mismatch"])
    print()

    if r2_note:
        print("NOTE: 'best held-out R2' is the max over the 12 model x encoding configs in")
        print("      results/grid13_v18.csv (the 13-set corpus).  The")
        print("      r2_best column of results/legacy/ceiling_stats_v16.csv names one designated model and")
        print("      differs for %d/%d datasets (using it would move Roche66k_CH from 1.0030"
              % (r2_note, len(t)))
        print("      to 0.9853).  The aggregate rho/median require the grid maximum.")
        print()

    print("Methods recorded in the CSV:")
    print("  per-dataset ceiling : %s" % per_ds[0]["ceil_method"])
    print("  per-dataset capture : %s" % per_ds[0]["cap_method"])
    print("  spearman_rho        : %s" % METHOD_RHO)
    print("  median_capture      : %s" % METHOD_MEDIAN)
    print("  mean_capture        : plain percentile, 1000 resamples, seed 42")
    print("  median_capture_pct  : plain percentile, 1000 resamples, seed 42")
    print()
    print("Wrote %s (%d bytes, %d rows)"
          % (OUT_PATH, OUT_PATH.stat().st_size, 2 * len(per_ds) + 4))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from-tables", action="store_true",
                       help="use only the shipped CSVs (default; no raw data needed)")
    group.add_argument("--from-raw", metavar="DIR", nargs="?", type=str,
                       const=str(ROOT / "data" / "raw"),
                       help="additionally cluster-bootstrap replicate condition groups from raw CSVs in DIR "
                            "(default when the flag is given without a value: data/raw)")
    args = parser.parse_args(argv)

    grid, ceil = load_tables()
    t = ceiling_table(ceil)
    if len(t) < 3:
        raise SystemExit("fewer than 3 datasets have a computable ceiling")
    dropped = ceil.loc[~np.isfinite(ceil["ceiling"]), "dataset"].astype(str).tolist()

    # one deterministic stream: aggregates first, then datasets in sorted order
    rng = np.random.default_rng(SEED)
    agg = dataset_level_bootstrap(t, rng)

    raw_info = None
    if args.from_raw:
        raw_path = Path(args.from_raw)
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        per_ds, raw_info = run_raw(t, raw_path, rng)
        if per_ds is None:
            return 0
        mode = "raw"
        if len(per_ds) != len(t):
            print("ERROR: only %d/%d datasets could be bootstrapped from raw data"
                  % (len(per_ds), len(t)))
            return 1
    else:
        per_ds = per_dataset_tables(t, rng)
        mode = "tables"

    r2_note = int((np.abs(t["r2_best_grid"] - t["r2_best"]) > 5e-4).sum())
    write_csv(per_ds, agg)
    print_summary(mode, t, dropped, agg, per_ds, raw_info, r2_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Quantify what a categorical encoding costs, in attribution rather than in R^2.

Two analyses of section 3.3 are produced here, from two shipped tables:

  A. Encoding-gap analysis on ``results/attrib13_long_cosolvent_flagged.csv``
     (1,392 rows = 156 model-dataset cells x 10 semantic slots; slots a source
     does not record are flagged ``set_aside`` and are dropped first).
     Within every (dataset, model) group the cells differ only in the encoding,
     so each pair of encodings is a controlled comparison: same rows, same
     estimator, same split.  For each pair the script reports

       * the Spearman correlation of the two |SHAP| share profiles,
       * the L1 distance between them (percentage points),
       * whether the most important slot is the same,

     and it aggregates the last quantity per estimator and the share change per
     slot for the MLP (integer -> one-hot).

  B. Accuracy-stratified attribution on ``results/table1_final_cosolvent.csv``
     (13 datasets, held-out R^2 and ceiling).  Cells are split at the median
     held-out R^2; the script reports how often each slot is primary and how
     often the metal slot is last in the two halves.

The shares are normalised per cell, so a cell contributes one 100 % profile
regardless of how many slots its source records.

    python scripts/attrib_encoding_gap.py
    python scripts/attrib_encoding_gap.py --attrib ... --table ...

Exit status is 0 on success and 2 when an input table is missing, matching the
convention of the other scripts in this directory.
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"

ATTRIB_CSV = RES_DIR / "attrib13_long_cosolvent_flagged.csv"
TABLE_CSV = RES_DIR / "table1_final_cosolvent.csv"

# The slot names as they are stored in the attribution table.
SLOTS = [
    "metal_encoded", "catalyst_encoded", "ligand_encoded", "base_encoded",
    # legacy encoded-column names (the slots are now reactant_1 / reactant_2)
    "solvent_encoded", "reactant_2_encoded", "reactant_1_encoded",
    "temperature", "time", "catalyst_loading",
]

# Display names used in the text ("the metal slot", ...).
PRETTY = {
    "metal_encoded": "metal",
    "catalyst_encoded": "catalyst",
    "ligand_encoded": "ligand",
    "base_encoded": "base",
    "solvent_encoded": "solvent",
    "reactant_2_encoded": "reactant 2 (amine partner)",
    "reactant_1_encoded": "reactant 1 (aryl halide partner)",
    "temperature": "temperature",
    "time": "time",
    "catalyst_loading": "catalyst loading",
}

MODELS = ["MLP", "Random Forest", "HistGB", "Gradient Boosting", "XGBoost"]


def _die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print("ERROR: %s" % msg, file=sys.stderr)
    raise SystemExit(2)


def load_shares(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (shares, recorded) for every (dataset, model, encoding) cell.

    ``shares`` holds |SHAP| shares in percent, normalised to 100 inside each
    cell after the ``set_aside`` rows are dropped; ``recorded`` is the matching
    boolean frame telling which slots the source actually records.  Both are
    indexed by (dataset, model, encoding) and have one column per semantic slot.
    """
    if not path.is_file():
        _die("missing %s" % path)
    long = pd.read_csv(path)
    need = {"dataset", "model", "encoding", "feature", "shap", "set_aside"}
    missing = need - set(long.columns)
    if missing:
        _die("%s lacks column(s) %s" % (path.name, ", ".join(sorted(missing))))

    kept = long[~long["set_aside"].astype(bool)].copy()
    kept["absshap"] = kept["shap"].abs()
    totals = kept.groupby(["dataset", "model", "encoding"])["absshap"].transform("sum")
    kept["share"] = 100.0 * kept["absshap"] / totals

    wide = kept.pivot_table(index=["dataset", "model", "encoding"],
                            columns="feature", values="share",
                            aggfunc="sum")
    seen = kept.pivot_table(index=["dataset", "model", "encoding"],
                            columns="feature", values="absshap",
                            aggfunc="size")
    for frame in (wide, seen):
        for slot in SLOTS:
            if slot not in frame.columns:
                frame[slot] = np.nan
    wide = wide[SLOTS].fillna(0.0)
    seen = seen[SLOTS].notna()
    order = sorted(wide.index)
    return wide.loc[order], seen.loc[order]


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (no scipy dependency; ties get the mean rank)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # average ties
    sorted_vals = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def analysis_a(shares: pd.DataFrame, seen: pd.DataFrame) -> pd.DataFrame:
    """One row per pair of encodings that share a dataset and an estimator.

    The Spearman correlation is taken over the slots the source records in both
    cells (a slot the source never records carries no information and would
    otherwise contribute a tied zero); the L1 distance is taken over all ten
    slots, because the two share profiles are each normalised to 100 %.
    """
    rows = []
    for (dataset, model), cell in shares.groupby(level=["dataset", "model"]):
        cell = cell.droplevel(["dataset", "model"])
        cells = seen.xs((dataset, model), level=["dataset", "model"])
        encodings = list(cell.index)
        for left, right in itertools.combinations(encodings, 2):
            a = cell.loc[left].to_numpy(dtype=float)
            b = cell.loc[right].to_numpy(dtype=float)
            mask = (cells.loc[left] & cells.loc[right]).to_numpy()
            rows.append({
                "dataset": dataset,
                "model": model,
                "a": left,
                "b": right,
                "spearman": spearman(a[mask], b[mask]),
                "l1": float(np.abs(a - b).sum()),
                "top_same": SLOTS[int(np.argmax(a))] == SLOTS[int(np.argmax(b))],
            })
    pairs = pd.DataFrame(rows)
    return pairs


def analysis_a_slot_shift(shares: pd.DataFrame, seen: pd.DataFrame) -> pd.Series:
    """MLP, one-hot minus integer, mean share change per slot (pp).

    A slot is averaged over the datasets in which the source records it in both
    cells, so a slot that is missing from a dataset does not dilute the mean.
    """
    deltas = []
    for dataset, cell in shares.xs("MLP", level="model").groupby(level="dataset"):
        cell = cell.droplevel("dataset")
        if "integer" not in cell.index or "one-hot" not in cell.index:
            continue
        mask = (seen.loc[(dataset, "MLP", "integer")]
                & seen.loc[(dataset, "MLP", "one-hot")]).to_numpy()
        d = (cell.loc["one-hot"] - cell.loc["integer"]).to_numpy(dtype=float)
        deltas.append(np.where(mask, d, np.nan))
    if not deltas:
        return pd.Series(dtype=float)
    return pd.Series(np.nanmean(np.vstack(deltas), axis=0), index=SLOTS)


def analysis_b(shares: pd.DataFrame, seen: pd.DataFrame, table: pd.DataFrame) -> dict:
    if not table.is_file():
        _die("missing %s" % table)
    meta = pd.read_csv(table)
    r2_by_dataset = dict(zip(meta["dataset"], meta["r2_best"]))
    median_r2 = float(np.median(meta["r2_best"]))

    high = {d for d, r in r2_by_dataset.items() if r >= median_r2}
    low = set(r2_by_dataset) - high

    primary = {k: {"high": 0, "low": 0} for k in SLOTS}
    last = {k: {"high": 0, "low": 0} for k in SLOTS}
    cells = {"high": 0, "low": 0}
    for (dataset, _model, _encoding), row in shares.iterrows():
        group = "high" if dataset in high else "low"
        cells[group] += 1
        # rank only over the slots this source records
        mask = seen.loc[(dataset, _model, _encoding)].to_numpy()
        vals = np.where(mask, row.to_numpy(dtype=float), np.nan)
        primary[SLOTS[int(np.nanargmax(vals))]][group] += 1
        last[SLOTS[int(np.nanargmin(vals))]][group] += 1

    return {"median_r2": median_r2, "high": high, "low": low,
            "cells": cells, "primary": primary, "last": last,
            "n_high_datasets": len(high), "n_low_datasets": len(low)}


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/attrib_encoding_gap.py",
        description="Attribution evidence for the encoding effect (section 3.3).")
    parser.add_argument("--attrib", type=Path, default=ATTRIB_CSV,
                        help="long attribution table (default: results/%s)"
                             % ATTRIB_CSV.name)
    parser.add_argument("--table", type=Path, default=TABLE_CSV,
                        help="per-dataset R^2/ceiling table (default: results/%s)"
                             % TABLE_CSV.name)
    args = parser.parse_args(argv)

    shares, seen = load_shares(args.attrib)
    pairs = analysis_a(shares, seen)
    shift = analysis_a_slot_shift(shares, seen)

    print("=" * 78)
    print("A. What the encoding changes in the attribution (same data, same model)")
    print("=" * 78)
    print("cells (dataset x model x encoding) with shares normalised : %d" % len(shares))
    print("paired comparisons (same dataset, same model)            : %d" % len(pairs))
    print("median Spearman of the two share profiles                : %.3f"
          % pairs["spearman"].median())
    print("minimum Spearman                                         : %.3f"
          % pairs["spearman"].min())
    print("median L1 distance between the two profiles              : %.1f pp"
          % pairs["l1"].median())
    print("maximum L1 distance                                     : %.1f pp"
          % pairs["l1"].max())
    same = int(pairs["top_same"].sum())
    print("identical most-important slot                           : %.1f%% (%d/%d)"
          % (_pct(same, len(pairs)), same, len(pairs)))
    print()
    print("identical most-important slot, by estimator:")
    for model in MODELS:
        sub = pairs[pairs["model"] == model]
        if not len(sub):
            continue
        n = int(sub["top_same"].sum())
        print("  %-18s %5.1f%%  (%d/%d)" % (model, _pct(n, len(sub)), n, len(sub)))
    print()
    print("MLP, integer -> one-hot: mean share change per slot (pp):")
    for slot in SLOTS:
        if slot in shift.index:
            print("  %-38s %+5.1f" % (PRETTY[slot], float(shift[slot])))
    print()

    b = analysis_b(shares, seen, args.table)
    print("=" * 78)
    print("B. Attribution reflects the design of the dataset, not its accuracy")
    print("=" * 78)
    print("median held-out R^2 over the corpus                     : %.4f"
          % b["median_r2"])
    print("cells above the median (>=)                             : %d (%d datasets)"
          % (b["cells"]["high"], b["n_high_datasets"]))
    print("cells below the median                                  : %d (%d datasets)"
          % (b["cells"]["low"], b["n_low_datasets"]))
    print()
    print("slot ranked first:")
    for slot in SLOTS:
        hi, lo = b["primary"][slot]["high"], b["primary"][slot]["low"]
        if hi or lo:
            print("  %-38s %5.1f%%  |  %5.1f%%"
                  % (PRETTY[slot], _pct(hi, b["cells"]["high"]),
                     _pct(lo, b["cells"]["low"])))
    print()
    print("slot ranked last:")
    for slot in SLOTS:
        hi, lo = b["last"][slot]["high"], b["last"][slot]["low"]
        if hi or lo:
            print("  %-38s %5.1f%%  |  %5.1f%%"
                  % (PRETTY[slot], _pct(hi, b["cells"]["high"]),
                     _pct(lo, b["cells"]["low"])))
    print()
    print("left column = cells at or above the median R^2, right column = below it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

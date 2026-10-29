#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 4 -- the 48 cross-dataset transfer cells.

Protocol recovered on 2026-10-28 from the file that wrote the shipped table
(``_work/v234/c20_transfer_4donor.py``, 2026-10-24 14:42), in that file's own
words:

    source LabelEncoders, unseen level -> len(classes_),
    XGBoost 200x6 lr0.1 subsample0.8 colsample0.8 seed42,
    features built on the source only, target predicted over its full row set

and in its code: the model is fitted on the donor's 80 % TRAIN rows, the
preprocessing (scaler, imputation) is fitted on those same rows, and the
constant baseline is the DONOR's train mean -- not the target's mean.

What this script does NOT do: it does not choose a donor model by its own test
score.  The shipped protocol fixes one learner for every donor.

    python scripts/build_transfer.py --only-donor AI_RCP_BH   # smoke test
    python scripts/build_transfer.py                          # all four donors

Writes outputs/c20_transfer_4donor/transfer_matrix.csv (the shipped copy is only replaced when
this is run as STEP 4 of the pipeline).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hte.data import load_real_dataset                        # noqa: E402
from hte.features import get_split_positions, prepare_features  # noqa: E402

DATA = ROOT / "data"
RES = ROOT / "results"
OUT = ROOT / "outputs" / "c20_transfer_4donor"

# the donor -> target matrix of the paper (§3.5); the four donors are flagged
# ``donor: true`` in data/alias_mapping.yaml, so no name is written here
from protocol import donors as _donors  # noqa: E402

DONORS = _donors()

XGB_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)


def _stems() -> dict:
    """alias -> source stem, from the one table that holds it."""
    from protocol import stems
    return stems()


def _prepare(alias: str, stems: dict):
    """Load one dataset and build its feature matrix (preprocessing on train rows)."""
    df, raw = load_real_dataset(str(DATA / (stems[alias] + ".csv")), str(DATA))
    tr, te = get_split_positions(len(df))
    X, y, cols, _disp, enc, scaler = prepare_features(df, raw, fit_rows=tr)
    return df, X, y, cols, enc, scaler, tr, te


def _apply(tgt: pd.DataFrame, cols: list, enc: dict, scaler) -> pd.DataFrame:
    """Encode a target frame with the DONOR's encoders.

    The shipped convention: a level the donor never saw gets the code
    ``len(classes_)`` -- one integer beyond the donor's vocabulary, which the
    fitted tree model then treats as the highest number.  Dropping the row,
    coding it -1, or refitting on the target are the three alternatives the
    paper explicitly reports as a different (milder) protocol.
    """
    out = pd.DataFrame(index=tgt.index)
    for c in cols:
        base = c[:-len("_encoded")] if c.endswith("_encoded") else c
        if c.endswith("_encoded"):
            le = enc.get(base)
            if le is None or base not in tgt.columns:
                out[c] = 0.0
                continue
            code = {v: i for i, v in enumerate(le.classes_)}
            unseen = len(le.classes_)
            out[c] = [code.get(v, unseen) for v in tgt[base].astype(str)]
        elif c in tgt.columns:
            out[c] = pd.to_numeric(tgt[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0
    num = [c for c in cols if not c.endswith("_encoded")]
    if num and scaler is not None:
        out[num] = scaler.transform(out[num])
    return out[cols]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-donor", default=None, help="smoke test with one donor alias")
    ap.add_argument("--out", default=None, help="override the output CSV path")
    args = ap.parse_args()

    import xgboost as xgb

    stems = _stems()
    targets = list(pd.read_csv(RES / "table1_v18_final.csv").dataset)
    donors = [args.only_donor] if args.only_donor else DONORS
    rows = []
    for donor in donors:
        if donor not in stems:
            print("  donor %s not in alias_mapping.yaml -- skipped" % donor)
            continue
        _df, X, y, cols, enc, scaler, tr, te = _prepare(donor, stems)
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X.iloc[tr], y.iloc[tr])
        own = float(r2_score(y.iloc[te], model.predict(X.iloc[te])))
        src_mean = float(y.iloc[tr].mean())
        print("  donor %-14s %2d feature columns · own held-out R2 %+.4f · "
              "train mean %.3f" % (donor, len(cols), own, src_mean), flush=True)
        for tgt_alias in targets:
            if tgt_alias == donor:
                continue
            tgt, _traw = load_real_dataset(str(DATA / (stems[tgt_alias] + ".csv")),
                                           str(DATA))
            yt = pd.to_numeric(tgt["yield"], errors="coerce").to_numpy(float)
            Xt = _apply(tgt, cols, enc, scaler)
            pred = np.asarray(model.predict(Xt), float).ravel()
            const = np.full_like(yt, src_mean)
            pear = (float(stats.pearsonr(yt, pred).statistic) ** 2
                    if len(np.unique(pred)) > 1 else 0.0)
            rows.append(dict(donor=donor, target=tgt_alias, n=int(len(yt)),
                             r2_transfer=float(r2_score(yt, pred)),
                             r2_constant=float(r2_score(yt, const)),
                             r2_pearson=pear,
                             mae_transfer=float(mean_absolute_error(yt, pred)),
                             mae_constant=float(mean_absolute_error(yt, const)),
                             target_mean=float(yt.mean()), pred_mean=float(pred.mean())))
            print("    %-18s n=%-6d R2=%+8.4f (const %+8.4f)  MAE=%6.2f (const %6.2f)"
                  % (tgt_alias, len(yt), rows[-1]["r2_transfer"],
                     rows[-1]["r2_constant"], rows[-1]["mae_transfer"],
                     rows[-1]["mae_constant"]), flush=True)

    table = pd.DataFrame(rows)
    if table.empty:
        print("no cells produced - nothing written")
        return 1
    # v24: default to the path the downstream check and verify_numbers.py read.
    OUT.mkdir(parents=True, exist_ok=True)
    dst = Path(args.out) if args.out else (OUT / "transfer_matrix.csv")
    table.to_csv(dst, index=False, encoding="utf-8")
    print("\n[transfer] %d cells -> %s" % (len(table), dst))
    print("\nper donor:")
    for donor, g in table.groupby("donor", sort=False):
        print("  %-14s mean R2 %+8.4f   negative %2d/%2d   mean r2_corr %.4f"
              % (donor, g.r2_transfer.mean(), int((g.r2_transfer < 0).sum()),
                 len(g), g.r2_pearson.mean()))
    print("  pooled       mean R2 %+8.4f   range %+.2f .. %+.2f"
          % (table.r2_transfer.mean(), table.r2_transfer.min(),
             table.r2_transfer.max()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

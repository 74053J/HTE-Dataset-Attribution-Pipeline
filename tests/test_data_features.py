#!/usr/bin/env python
"""Equivalence check (a): hte.data + hte.features vs the frozen v16 pipeline.

For three deliberately different datasets (normal catalyst-loading axis, a
substitute loading axis, and no usable amount axis at all) this reruns the SAME
input file through both implementations and asserts:

* the loaded DataFrame is identical (values, dtypes, column order);
* the raw-column mapping dict, the resolved loading axis and the catalyst-slot
  label are identical;
* the train/test positional split is identical;
* the feature matrix, target, feature names, display names, label-encoder
  classes and StandardScaler statistics are identical.

Floats are compared at 1e-12; in practice every numeric check is bit-exact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C


DATASETS = [
    # (label, source stem, why this one)
    ("AI_RCP_BH", "(AI_RCP_pub)bh_data_clean_all_whitelisted",
     "normal catalyst-loading axis"),
    ("Summit_CN", "(Summit)Baumgartner_CN_coupling_96",
     "substitute loading axis (base_equivalents)"),
    ("rxn_yields_SM", "(rxn_yields)aap9112_Data_File_S1",
     "no usable amount axis and no varying numeric feature"),
]


def main() -> int:
    root = C.data_dir()
    if root is None:
        print("SKIP: data/ directory not found next to this repository")
        return 0

    import hte

    rep = C.Report("test_data_features")
    for label, stem, why in DATASETS:
        print("\n[%s] %s  (%s)" % (label, stem, why))
        dv, rv, dh, rh = C.load_pair(stem, root)
        v16 = C.load_v16()

        # The frozen monolith still materialises the slots that the mapping
        # disables (catalyst, catalyst_loading): it fills them with a constant.
        # hte drops them instead.  A constant column cannot influence the model,
        # so the comparison ignores such columns and records the difference.
        dropped = [c for c in dv.columns
                   if c not in dh.columns and dv[c].nunique(dropna=False) <= 1]
        dv_cmp = dv.drop(columns=dropped) if dropped else dv
        if dropped:
            rep.check("%s: monolith-only constant columns ignored" % label, True, 0.0,
                      note=", ".join(map(str, dropped)))
        try:
            pd.testing.assert_frame_equal(dv_cmp, dh, check_dtype=True,
                                          check_exact=True, check_names=True)
            rep.check("%s: DataFrame identical (values+dtypes+order)" % label, True, 0.0)
        except AssertionError as exc:
            rep.check("%s: DataFrame identical (values+dtypes+order)" % label,
                      False, note=str(exc).splitlines()[0][:120])

        rep.equal("%s: raw_feature_names dict" % label, rv, rh)
        rep.equal("%s: LOADING_AXIS dict" % label,
                  dict(v16.LOADING_AXIS), dict(hte.data.LOADING_AXIS))
        rep.equal("%s: CATALYST_SLOT_LABEL" % label,
                  str(v16.CATALYST_SLOT_LABEL), str(hte.data.CATALYST_SLOT_LABEL))

        tr_v, te_v = v16.get_split_positions(len(dv))
        tr_h, te_h = hte.get_split_positions(len(dh))
        rep.equal("%s: train split positions" % label, tr_v.tolist(), tr_h.tolist())
        rep.equal("%s: test split positions" % label, te_v.tolist(), te_h.tolist())

        with C.quiet():
            Xv, yv, fcv, dnv, encv, scv = v16.prepare_features(dv, rv, fit_rows=tr_v)
        with C.quiet():
            Xh, yh, fch, dnh, ench, sch = hte.prepare_features(dh, rh, fit_rows=tr_h)
        rep.diff("%s: X matrix" % label, np.asarray(Xv), np.asarray(Xh))
        rep.diff("%s: y vector" % label, np.asarray(yv), np.asarray(yh))
        rep.equal("%s: feature_cols" % label, list(fcv), list(fch))
        rep.equal("%s: display_feature_names" % label, list(dnv), list(dnh))
        rep.equal("%s: label-encoder classes" % label,
                  {k: list(v.classes_) for k, v in encv.items()},
                  {k: list(v.classes_) for k, v in ench.items()})

        v_fit, h_fit = hasattr(scv, "mean_"), hasattr(sch, "mean_")
        rep.equal("%s: StandardScaler fitted state" % label, v_fit, h_fit)
        if v_fit and h_fit:
            rep.diff("%s: scaler.mean_" % label, scv.mean_, sch.mean_)
            rep.diff("%s: scaler.scale_" % label, scv.scale_, sch.scale_)
            rep.diff("%s: scaler.var_" % label, scv.var_, sch.var_)

    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())

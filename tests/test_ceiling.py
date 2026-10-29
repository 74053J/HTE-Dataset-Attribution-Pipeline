#!/usr/bin/env python
"""Equivalence check (b): hte.ceiling vs v16 and vs the legacy 16-set ceiling table.

For all 16 published datasets this loads the raw file through both
implementations and asserts that ``estimate_noise_ceiling`` agrees bit-exactly,
then checks the same quantities against the superseded 16-dataset table, now
kept at ``results/legacy/ceiling_stats_v16.csv`` (``n``, ``ceiling``, ``n_cond``,
``n_rep_groups``, ``rep_frac``).  The 13-dataset replacement is
``results/table1_v18_final.csv``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C

# order of the legacy 16-dataset results/legacy/ceiling_stats_v16.csv (copied from scripts/tables.py)
CEILING_ORDER = [
    "(AI_RCP_pub)bh_data_clean_all_whitelisted",
    "(AI_RCP_pub)sm_data_clean_all_whitelisted",
    "(bh-hte-ood)BH_HTE_Curated_Dataset_v20260219",
    "(DoyleHTE)CN_coupling_3600",
    "(Roche66k)BH_plus_SM_24137",
    "(Roche66k)Buchwald_Hartwig_14191",
    "(Roche66k)Suzuki_Miyaura_9946",
    "(Roche66k)Amide_coupling_3917",
    "(Roche66k)CN_coupling_3100",
    "(Roche66k)CO_coupling_2112",
    "(Roche66k)Condensation_2095",
    "(Roche66k)CH_activation_1251",
    "(rxn_yields)aap9112_Data_File_S1",
    "(Summit)Reizman_Suzuki_385",
    "(Summit)Baumgartner_CN_coupling_96",
    "(Suzuki-Coupling-ML)suzuki_dataset_raw_origin",
]

FLOAT_KEYS = ["ceiling", "s2_within", "s2_total", "rep_frac"]
INT_KEYS = ["n_cond", "n_rep_groups"]


def main() -> int:
    root = C.data_dir()
    if root is None:
        print("SKIP: data/ directory not found next to this repository")
        return 0
    # The current published table, not the retired 16-dataset one: the legacy copy
    # predates the metal rule and the slot renames, so it measures "how much we
    # changed" rather than "is it still the published number".
    ref_path = C.RESULTS / "table1_v18_final.csv"
    if not ref_path.is_file():
        print("SKIP: missing %s" % ref_path)
        return 0

    import hte

    ref = pd.read_csv(ref_path).set_index("dataset")
    from hte.datasets import stems as _stems
    _alias_of = {v: k for k, v in _stems().items()}   # stem -> alias
    rep = C.Report("test_ceiling")
    n_defined = 0
    worst_v16 = 0.0
    worst_csv = 0.0

    for i, stem in enumerate(CEILING_ORDER, 1):
        print("\n[ceiling %2d/%d] %s" % (i, len(CEILING_ORDER), stem))
        dv, _rv, dh, _rh = C.load_pair(stem, root)
        v16 = C.load_v16()
        info_v = v16.estimate_noise_ceiling(dv)
        info_h = hte.estimate_noise_ceiling(dh)

        diffs = []
        for key in FLOAT_KEYS:
            diffs.append(C.max_abs_diff([info_v[key]], [info_h[key]]))
        ok = max(diffs) <= 1e-12
        worst_v16 = max(worst_v16, max(diffs))
        rep.check("%s: v16 vs hte ceiling stats" % stem, ok, max(diffs))

        ok = (info_v["n_cond"] == info_h["n_cond"]
              and info_v["n_rep_groups"] == info_h["n_rep_groups"])
        rep.check("%s: v16 vs hte integer stats" % stem, ok, 0.0)

        alias = _alias_of.get(stem, stem)
        if alias not in ref.index:
            rep.skip("%s: shipped ceiling row" % stem, "not in %s" % ref_path.name)
            continue
        row = ref.loc[alias]
        if np.isfinite(float(info_h["ceiling"])):
            n_defined += 1
        d_n = abs(len(dh) - int(row["n_rows"] if "n_rows" in row.index else row["n"]))
        d_num = C.max_abs_diff(
            [info_h["ceiling"], info_h["n_cond"], info_h["n_rep_groups"],
             info_h["rep_frac"]],
            [row["ceiling"], row["n_cond"], row["n_rep_groups"], row["rep_frac"]])
        worst_csv = max(worst_csv, d_num, float(d_n))
        rep.check("%s: shipped ceiling_stats_v16.csv" % stem,
                  d_n == 0 and d_num <= 1e-12, max(d_num, float(d_n)))

    print("\n[summary] datasets with a defined ceiling: %d/%d"
          % (n_defined, len(CEILING_ORDER)))
    print("[summary] worst v16-vs-hte max|diff| = %.3e" % worst_v16)
    print("[summary] worst hte-vs-shipped max|diff| = %.3e" % worst_csv)
    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())

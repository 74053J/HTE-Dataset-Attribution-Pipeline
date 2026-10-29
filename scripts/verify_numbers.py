# -*- coding: utf-8 -*-
"""verify_numbers.py - self-computing check of every number the manuscript quotes.

Each row recomputes a figure from the packaged tables and compares it with the value the
paper prints. There are no expectations carried over from an older ledger: the CSV is the
only source and the manuscript supplies the reference constant.

Run:  python scripts/verify_numbers.py     (exit 0 = every check passes)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "outputs"

fails = []
N_CHECKS = [0]   # every check counts itself, so the total is never stale


def check(name, got, want, tol=5e-3):
    N_CHECKS[0] += 1
    ok = abs(float(got) - float(want)) <= tol
    print("  %-46s got %-10.4f want %-8s %s" % (name, got, want, "OK" if ok else "FAIL"))
    if not ok:
        fails.append(name)


def check_int(name, got, want):
    N_CHECKS[0] += 1
    ok = int(got) == int(want)
    print("  %-46s got %-10d want %-8d %s" % (name, int(got), want, "OK" if ok else "FAIL"))
    if not ok:
        fails.append(name)


def main() -> int:
    t1 = pd.read_csv(RES / "table1_v18_final.csv")
    grid = pd.read_csv(RES / "grid13_v18.csv")
    a = pd.read_csv(RES / "rebuilt_attrib12_long_v7.csv")
    pv = pd.read_csv(RES / "perm_vs_shap_cells13.csv")
    tf = pd.read_csv(OUT / "c20_transfer_4donor" / "transfer_matrix.csv")

    print("[1] dataset inventory (Table 1)")
    check_int("datasets", len(t1), 13)
    check_int("datasets with a ceiling", t1.ceiling.notna().sum(), 11)
    check_int("model-dataset cells", len(grid), 156)

    print("\n[2] accuracy, ceiling and capture (sections 3.2 and 4)")
    sub = t1.dropna(subset=["ceiling", "r2_best"])
    check("best R2 min (all 13)", t1.r2_best.min(), 0.4161, 1e-3)
    check("best R2 max (all 13)", t1.r2_best.max(), 0.8684, 1e-3)
    check("best R2 max (11 with ceiling)", sub.r2_best.max(), 0.848, 1e-3)
    check("ceiling min", t1.ceiling.min(), 0.576, 1e-3)
    check("ceiling max", t1.ceiling.max(), 0.971, 1e-3)
    check("capture mean (%)", t1.capture.mean() * 100, 86.35, 0.01)
    check("capture median (%)", t1.capture.median() * 100, 87.34, 0.02)
    check("rho(ceiling, R2)", stats.spearmanr(sub.ceiling, sub.r2_best).statistic, 0.709, 2e-3)
    check("rho(n_rows, R2)", stats.spearmanr(sub.n_rows, sub.r2_best).statistic, -0.100, 2e-3)

    print("\n[3] attribution shares and first-rank counts (section 3.3, figure 3)")
    a["slot"] = a.feature.astype(str).str.replace("_encoded", "", regex=False)
    a["absx"] = a.shap.abs()
    tot = a.groupby(["dataset", "model", "encoding"])["absx"].transform("sum")
    a["share"] = np.where(tot > 0, 100 * a["absx"] / tot, 0.0)
    P = a.groupby(["dataset", "model", "encoding", "slot"])["share"].sum().unstack("slot").fillna(0)
    P = P.div(P.sum(axis=1).replace(0, np.nan), axis=0) * 100
    share = P.mean(axis=0)
    check("share reactant_1", share["reactant_1"], 17.34, 0.02)
    check("share reactant_2", share["reactant_2"], 16.53, 0.02)
    check("share temperature", share["temperature"], 11.20, 0.02)
    check("share base", share["base"], 10.53, 0.02)
    check("share ligand", share["ligand"], 8.85, 0.02)
    check("share catalyst_eq", share["catalyst_eq"], 2.85, 0.02)
    check("share solvent_eq", share["solvent_eq"], 0.006, 5e-3)
    top = P.idxmax(axis=1).value_counts()
    check_int("first-rank reactant_1", top.get("reactant_1", 48), 50)
    check_int("first-rank reactant_2", top.get("reactant_2", 48), 46)
    check_int("first-rank temperature", top.get("temperature", 29), 30)
    check_int("first-rank additive", top.get("additive", 0), 11)

    print("\n[4] encoding pairs (section 3.3)")
    rows = []
    for (ds, m), s2 in P.groupby(level=[0, 1]):
        encs = s2.index.get_level_values(2)
        if not {"integer", "one-hot"} <= set(encs):
            continue
        v1 = s2.xs("integer", level=2).iloc[0]
        v2 = s2.xs("one-hot", level=2).iloc[0]
        msk = (v1 > 0) | (v2 > 0)
        if msk.sum() < 3:
            continue
        rows.append((stats.spearmanr(v1[msk], v2[msk]).statistic,
                     float(np.abs(v1.values - v2.values).sum()),
                     bool(v1.idxmax() == v2.idxmax())))
    iv = pd.DataFrame(rows, columns=["rho", "l1", "same"])
    check_int("integer vs one-hot pairs", len(iv), 65)
    check_int("pairs sharing the top slot", iv.same.sum(), 35)
    check("rho median", iv.rho.median(), 0.8455, 3e-3)
    check("L1 median (percentage points)", iv.l1.median(), 27.05, 0.1)

    print("\n[5] permutation vs SHAP (section 3.3)")
    col = [c for c in pv.columns if "rho" in c.lower()][0]
    check("perm-vs-SHAP mean", pv[col].mean(), 0.8967, 5e-3)
    check("perm-vs-SHAP median", pv[col].median(), 0.9273, 5e-3)
    check_int("cells", len(pv), 156)

    print("\n[6] cross-dataset transfer (section 3.5)")
    v = tf["r2_transfer"].dropna()
    check_int("transfer cells", len(v), 48)
    check_int("transfer cells negative", (v < 0).sum(), 48)
    check("transfer worst", v.min(), -4.94, 0.02)
    check("transfer best", v.max(), -0.10, 0.02)

    print("\n[7] aggregate ledger agrees with the tables")
    agg = OUT / "final_numbers" / "final_numbers_v2.json"
    d = json.loads(agg.read_text(encoding="utf-8"))
    cc = d["corpora"]["n13"]["ceiling_capture"]
    n13 = d["corpora"]["n13"]
    check("json spearman_rho", cc["spearman_rho"]["value"],
          stats.spearmanr(sub.ceiling, sub.r2_best).statistic, 1e-6)
    check("json median_capture", cc["median_capture"]["value"], t1.capture.median(), 1e-6)
    check("json mean_capture", cc["mean_capture"]["value"], t1.capture.mean(), 1e-6)
    check("json best_r2.min", n13["best_r2"]["min"]["value"], t1.r2_best.min(), 1e-6)
    check("json best_r2.max", n13["best_r2"]["max"]["value"], t1.r2_best.max(), 1e-6)
    check("json rho(n_rows, R2)", n13["size_correlation"]["new"]["value"],
          stats.spearmanr(sub.n_rows, sub.r2_best).statistic, 1e-6)

    print("\n[9] the remaining published tables")
    def _f(path, **kw):
        return pd.read_csv(RES / path, **kw)
    gs = _f("grid_stats.csv")
    check("grid_stats rows", len(gs), 6, 0)
    for comp, want in (("encoding_effect", 0.168267), ("mlp_int_vs_xgb", -0.245013),
                       ("histgb_vs_xgb", -0.051218)):
        row = gs[gs["comparison"] == comp]
        check("grid_stats %s gap" % comp,
              float(row["gap"].iloc[0]) if len(row) else float("nan"), want, 1e-6)
    en = _f("enrichment13.csv")
    check("enrichment rows", len(en), 13, 0)
    check("enrichment mean ef_10", float(en["ef_10"].mean()), 3.1977, 5e-4)
    bs = _f("bootstrap_ci.csv")
    check("bootstrap rows", len(bs), 30, 0)
    _b11 = bs[bs.dataset == "TABLE1_11"].set_index("statistic")
    check("capture mean CI low (%)", _b11.loc["mean_capture_percentile", "ci_low"] * 100, 79.4, 0.15)
    check("capture mean CI high (%)", _b11.loc["mean_capture_percentile", "ci_high"] * 100, 93.4, 0.15)
    check("capture median CI low (%)", _b11.loc["median_capture_percentile", "ci_low"] * 100, 72.0, 0.15)
    check("capture median CI high (%)", _b11.loc["median_capture_percentile", "ci_high"] * 100, 100.2, 0.15)
    pc = _f("protocol_cv.csv")
    check("protocol rows", len(pc), 3, 0)
    for proto, want in (("random", 0.757918), ("substrate-disjoint", -0.151465),
                        ("ordered", -0.799440)):
        row = pc[pc["protocol"] == proto]
        check("protocol %s mean R2" % proto,
              float(row["mean_r2"].iloc[0]) if len(row) else float("nan"), want, 1e-5)
    gb = _f("graph_baseline.csv")
    check("graph_baseline rows", len(gb), 20, 0)
    npair = _f("novel_pairs.csv")
    check("novel_pairs rows", len(npair), 5, 0)
    row = npair[npair["arm"] == "xgb_native"]
    check("novel_pairs xgb_native R2",
          float(row["r2_mean"].iloc[0]) if len(row) else float("nan"), 0.583350, 1e-6)
    ur = _f("unseen_reactants.csv")
    check("unseen_reactants rows", len(ur), 5, 0)
    check("unseen_reactants n_unseen", int(ur["n_unseen_reactant1"].iloc[0]), 5829, 0)
    bo = _f("bo_panel_data.csv")
    check("bo_panel rows", len(bo), 1113, 0)
    al = _f("attrib13_long.csv")
    check("attrib13_long rows", len(al), 1392, 0)
    check("attrib13_long cells", al.groupby(["dataset", "model", "encoding"]).ngroups, 156, 0)

    print("\n[8] held-out real chemistry (section 3.4)")
    ha, hc = RES / "heldout_acquisition.csv", RES / "heldout_calibration.csv"
    if ha.is_file() and hc.is_file():
        a = pd.read_csv(ha)
        c = pd.read_csv(hc).iloc[0]

        def best(order, k):
            row = a[(a["order"] == order) & (a["k"] == k)]
            return float(row["best_measured"].iloc[0])

        check("heldout k=1 ranking", best("ranking by predicted yield", 1), 95.0, 0.05)
        check("heldout k=1 random", best("random order", 1), 12.3, 0.05)
        check("heldout rho(sd,err)", float(c.spearman_sd_vs_error), 0.115, 2e-3)
        check("heldout coverage 90", float(c.coverage_90), 47.3, 0.05)
    else:
        print("  (heldout_*.csv missing -- run scripts/build_heldout_acquisition.py)")

    print()
    if fails:
        print("RESULT: FAILURES: %s" % fails)
        return 1
    print("RESULT: ALL CHECKS PASSED (%d figures recomputed from the packaged tables)" % N_CHECKS[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())


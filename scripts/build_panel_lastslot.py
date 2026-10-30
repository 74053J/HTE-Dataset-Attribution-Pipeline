# -*- coding: utf-8 -*-
"""Producer for the last-rank panel: ``results/panel_lastslot.csv`` (+ per-cell detail).

Background: the manuscript used to claim that a slot which is constant within a dataset
"occupies the last rank in most cells" and quoted metal 46 / additive 29 / cosolvent 18.
Constant slots are in fact dropped before fitting, so they never enter the attribution;
and the quoted counts depend entirely on how "present in that cell" is defined.

This script therefore reports the last-ranking slot under three explicit conventions:

    V1  row_present   a slot is a candidate if the cell has at least one row for it
                      (a recorded slot whose contribution is exactly 0 still ranks last).
                      <-- MAIN convention; this is what "present in that cell" means.
    V2  positive      a slot is a candidate only if its summed |SHAP| is > 0.  Zero
                      contributions are silently dropped, which promotes other slots.
                      <-- SENSITIVITY only; reported to document the difference.
    V3  all16_zero    all 16 slots are candidates, absent ones scored 0 (ties broken
                      alphabetically).  Documents the old 46/29/18-style artefact.
                      <-- SENSITIVITY only.

Both the input table and this script are hashed, because the two numbers people quote
(V1 35/33/24 vs V2 36/30/19) differ only by the convention, not by the data.

Outputs
    results/panel_lastslot.csv          one row per (convention, slot)
    results/panel_lastslot_cells.csv    one row per (dataset, model, encoding, slot)
    results/panel_lastslot_meta.csv     input/script sha256, cell counts, convention notes

usage
    python scripts/build_panel_lastslot.py [--repo DIR] [--outdir DIR] [--table FILE]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--table", default=None)
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    src = Path(args.table) if args.table else repo / "results" / "rebuilt_attrib12_long_v7.csv"
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))

    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        try:
            v = abs(float(r["shap"]))
        except Exception:
            continue
        k = (r["dataset"], r["model"], r["encoding"])
        sums[k][r["slot"]] += v
        counts[k][r["slot"]] += 1
    slots = sorted({s for d in sums.values() for s in d})

    def last_by(present_slots, values):
        if not present_slots:
            return ""
        return min(present_slots, key=lambda s: (values.get(s, 0.0), s))

    v1, v2, v3 = Counter(), Counter(), Counter()
    detail = []
    for k in sorted(sums):
        val = sums[k]
        cnt = counts[k]
        p1 = [s for s in slots if cnt.get(s, 0) > 0]          # row present
        p2 = [s for s in slots if val.get(s, 0.0) > 0.0]      # positive sum
        a = last_by(p1, val)
        b = last_by(p2, val)
        c = last_by(slots, val)                               # absent slots read as 0
        v1[a] += 1
        v2[b] += 1
        v3[c] += 1
        total = sum(val.values())
        for s in slots:
            detail.append(dict(dataset=k[0], model=k[1], encoding=k[2], slot=s,
                               n_rows=int(cnt.get(s, 0)),
                               sum_abs_shap=round(val.get(s, 0.0), 6),
                               share=round(val.get(s, 0.0) / total, 6) if total > 0 else 0.0,
                               present_row=int(cnt.get(s, 0) > 0),
                               present_positive=int(val.get(s, 0.0) > 0.0),
                               rank_last_V1=int(s == a),
                               rank_last_V2=int(s == b),
                               rank_last_V3=int(s == c)))

    n_cells = len(sums)
    out_rows = []
    for conv, counter, note in (
        ("V1_row_present", v1, "MAIN: a recorded slot is a candidate even at zero contribution"),
        ("V2_positive_only", v2, "SENSITIVITY: zero-contribution slots dropped"),
        ("V3_all16_zero_filled", v3, "SENSITIVITY: absent slots read as zero (tie artefact)"),
    ):
        for s in slots:
            c = counter.get(s, 0)
            out_rows.append(dict(convention=conv, slot=s, cells_last=c,
                                 share_of_cells=round(c / n_cells, 4), note=note))

    outdir = Path(args.outdir) if args.outdir else repo / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    s_dst = outdir / "panel_lastslot.csv"
    d_dst = outdir / "panel_lastslot_cells.csv"
    m_dst = outdir / "panel_lastslot_meta.csv"
    with open(s_dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["convention", "slot", "cells_last", "share_of_cells", "note"])
        w.writeheader()
        w.writerows(out_rows)
    with open(d_dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "encoding", "slot", "n_rows",
                                          "sum_abs_shap", "share", "present_row",
                                          "present_positive", "rank_last_V1", "rank_last_V2",
                                          "rank_last_V3"])
        w.writeheader()
        w.writerows(detail)
    with open(m_dst, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerow(["input_table", src.name])
        w.writerow(["input_sha256", _sha256(src)])
        w.writerow(["script_sha256", _sha256(Path(__file__).resolve())])
        w.writerow(["cells", n_cells])
        w.writerow(["slots", len(slots)])
        w.writerow(["rows_with_zero_shap", sum(1 for r in rows if r["shap"] and abs(float(r["shap"])) == 0)])
        w.writerow(["main_convention", "V1_row_present"])

    print("input :", src)
    print("sha256:", _sha256(src))
    print("cells : %d  slots: %d  detail rows: %d" % (n_cells, len(slots), len(detail)))
    print("wrote %s, %s, %s" % (s_dst.name, d_dst.name, m_dst.name))
    for conv, counter in (("V1_row_present", v1), ("V2_positive_only", v2), ("V3_all16_zero_filled", v3)):
        print("  %-20s %s" % (conv, counter.most_common(5)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

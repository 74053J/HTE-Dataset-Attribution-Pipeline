#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STEP 0 -- fetch the public datasets this package needs, then split Roche66k.

Nothing here runs unless you pass ``--i-agree``: the script first prints the licence
that applies to every source and what it obliges you to do.  Only the five sources
that permit redistribution are downloaded; the two AAAS datasets are pointers
(obtain them yourself, the loader is file-name driven).

The Roche66k branch is the important one.  The upstream record is a *single* table of
72,142 reactions; the paper uses eight subsets of it, and they are not arbitrary:
rows without a parsed temperature and time are dropped (72,142 -> 51,275) and the
remainder is split by reaction type.  The script reproduces exactly that, checks every
row count against the published one, and prints the rule again when it is done --
because a differently cut table gives different numbers.

Usage
-----
    python scripts/fetch_datasets.py                # show the licence table, do nothing
    python scripts/fetch_datasets.py --i-agree      # download + split into data/
    python scripts/fetch_datasets.py --i-agree --only roche --out-dir /tmp/x

Exit codes: 0 ok, 1 download/verification failure, 2 licence not accepted.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# --------------------------------------------------------------------------------------
# Sources.  Only these are downloaded; each row states the licence that applies and the
# obligation it carries.  The two AAAS datasets are deliberately absent (see POINTERS).
# --------------------------------------------------------------------------------------
SOURCES = [
    dict(
        key="ai_rcp",
        label="AI_RCP_pub (Suzuki-Miyaura / Buchwald-Hartwig)",
        licence="MIT",
        obligation="keep the copyright and permission notice",
        files=[
            ("https://raw.githubusercontent.com/LorenzoTarricone/AI_RCP_pub/main/data/bh_data_clean_all_whitelisted.csv",
             "(AI_RCP_pub)bh_data_clean_all_whitelisted.csv"),
            ("https://raw.githubusercontent.com/LorenzoTarricone/AI_RCP_pub/main/data/sm_data_clean_all_whitelisted.csv",
             "(AI_RCP_pub)sm_data_clean_all_whitelisted.csv"),
        ],
    ),
    dict(
        key="bh",
        label="bh-hte-ood (curated Buchwald-Hartwig collection)",
        licence="Apache-2.0",
        obligation="keep the NOTICE and state that this copy is a subset (27,497 of 357,461 rows)",
        files=[
            ("https://raw.githubusercontent.com/schwallergroup/bh-hte-ood/main/data/BH_HTE_Curated_Dataset_v20260219.csv",
             "(bh-hte-ood)BH_HTE_Curated_Dataset_v20260219.csv"),
        ],
    ),
    dict(
        key="doyle",
        label="DoyleHTE (Ahneman 2018 C-N coupling, authors' data repo)",
        licence="CC BY 4.0",
        obligation="attribution to doyle-lab-ucla/ochem-data",
        files=[
            ("https://raw.githubusercontent.com/doyle-lab-ucla/ochem-data/main/CN/rxns/smiles.csv",
             "_doyle_cn_raw.csv"),
        ],
    ),
    dict(
        key="summit",
        label="Summit (Reizman 2016 / Baumgartner 2019 flow optimisation)",
        licence="MIT",
        obligation="keep the upstream copyright notice",
        files=[
            ("https://raw.githubusercontent.com/sustainable-processes/summit/main/summit/benchmarks/data/reizman_suzuki_case_1.csv", "_summit_case_1.csv"),
            ("https://raw.githubusercontent.com/sustainable-processes/summit/main/summit/benchmarks/data/reizman_suzuki_case_2.csv", "_summit_case_2.csv"),
            ("https://raw.githubusercontent.com/sustainable-processes/summit/main/summit/benchmarks/data/reizman_suzuki_case_3.csv", "_summit_case_3.csv"),
            ("https://raw.githubusercontent.com/sustainable-processes/summit/main/summit/benchmarks/data/reizman_suzuki_case_4.csv", "_summit_case_4.csv"),
            ("https://raw.githubusercontent.com/sustainable-processes/summit/main/summit/benchmarks/data/baumgartner_aniline_cn_crosscoupling.csv", "_summit_baumgartner.csv"),
        ],
    ),
    dict(
        key="roche",
        label="Roche66k (single 72,142-reaction table, split into 8 subsets)",
        licence="CC BY-NC 4.0",
        obligation="attribute the six authors + paper DOI; NON-COMMERCIAL use only; "
                   "state that the CSVs are derived from the published table",
        files=[
            ("https://raw.githubusercontent.com/georg-wuitschik-Roche/zScore-App/main/frontend/public/data/v2.1.parquet",
             "_roche66k_master.parquet"),
        ],
    ),
]

POINTERS = [
    ("rxn_yields (Science 2018, Perera et al., doi 10.1126/science.aap9112)",
     "no licence is registered for this article (Crossref has no licence field) and reuse "
     "goes through a manual CCC decision, so you have to fetch it yourself: download Data "
     "File S1 from the article page and save it as\n"
     "        data/(rxn_yields)aap9112_Data_File_S1.csv"),
]

# The eight subsets, in the exact order and with the exact row counts the paper used.
ROCHE_SUBSETS = [
    ("Amide coupling", "Amide_coupling", 3917),
    ("Buchwald-Hartwig", "Buchwald_Hartwig", 14191),
    ("Suzuki-Miyaura", "Suzuki_Miyaura", 9946),
    ("CN-Coupling", "CN_coupling", 3100),
    ("CO-Coupling", "CO_coupling", 2112),
    ("Condensation", "Condensation", 2095),
    ("CH-Activation", "CH_activation", 1251),
]
ROCHE_UNION = ("BH_plus_SM", ("Buchwald-Hartwig", "Suzuki-Miyaura"), 24137)
ROCHE_MASTER_ROWS = 72142
ROCHE_KEPT_ROWS = 51275

# "3 h at 60 °C, then 17 h at 80 °C" -> the published files keep the FIRST stage,
# i.e. time = 3, temperature = 60.  Strings such as "None" or "unknown time/temp"
# carry no stage at all and are the rows the paper drops.
TEMP_TIME = re.compile(r"([0-9.]+)\s*h\s+at\s+([0-9.]+)\s*\u00b0?C", re.I)


def parse_temp_time(text):
    """First '<time> h at <temperature> °C' stage, or (nan, nan)."""
    m = TEMP_TIME.search(str(text))
    if not m:
        return float("nan"), float("nan")
    return float(m.group(2)), float(m.group(1))


def _glyphs() -> tuple[str, str]:
    """Continuous block bar when the console can show it, ASCII otherwise."""
    try:
        "\u2588\u2591".encode(sys.stdout.encoding or "utf-8")
        return "\u2588", "\u2591"
    except Exception:
        return "#", "-"


def _bar(label: str, done: int, total: int) -> None:
    """One-line text progress bar, rewritten in place."""
    full, empty = _glyphs()
    mb = done / 1e6
    if total:
        frac = min(done / total, 1.0)
        width = 28
        filled = int(width * frac)
        line = "   %-46s %s%s %5.1f/%.1f MB" % (
            label[:46], full * filled, empty * (width - filled), mb, total / 1e6)
    else:
        line = "   %-46s %8.2f MB" % (label[:46], mb)
    print("\r" + line, end="", flush=True)


def _fetch(url: str, dest: Path, label: str | None = None) -> int:
    """Stream the file to disk, showing a text progress bar."""
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_datasets/1.0"})
    name = label or dest.name
    with urllib.request.urlopen(req, timeout=180) as r:
        total = int(r.headers.get("Content-Length") or 0)
        dest.parent.mkdir(parents=True, exist_ok=True)
        done = 0
        with open(dest, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                _bar(name, done, total)
    print()
    return done


def _licence_table() -> None:
    """One table: every source, its licence, what you owe, and how it is obtained."""
    rows = [
        ("AI_RCP_pub", "MIT",
         "keep the copyright and permission notice", "downloaded by this script"),
        ("bh-hte-ood", "Apache-2.0",
         "keep the NOTICE; state that this copy is a subset", "downloaded by this script"),
        ("Summit", "MIT",
         "keep the copyright notice",
         "downloaded by this script, then the 4 cases are stacked (385 rows)"),
        ("Roche66k", "CC BY-NC 4.0",
         "attribute six authors + DOI; NON-COMMERCIAL only; state it is derived",
         "downloaded by this script, then split into 8"),
        ("DoyleHTE", "CC BY 4.0 (authors' data repo)",
         "attribution",
         "downloaded by this script, then cut to the complete 15 x 20 grid"),
        ("rxn_yields", "AAAS Science Journals Default License (no open licence)",
         "reading and non-commercial research use are permitted; REDISTRIBUTION "
         "needs written permission from AAAS",
         "you fetch it for local use; the package ships a pointer, not a copy"),
    ]
    print("=" * 78)
    print("DATASETS AND LICENCES")
    print("=" * 78)
    for name, lic, owe, how in rows:
        print("\n  %s" % name)
        print("    licence : %s" % lic)
        print("    you owe : %s" % owe)
        print("    obtain  : %s" % how)
    print("\n" + "=" * 78)


def _paper_notice() -> None:
    """One short panel: how the files must be prepared, and what changes if not."""
    print("\n" + "=" * 78)
    print("HOW THE FILES MUST BE PREPARED")
    print("=" * 78)
    print("  1. Roche66k : keep rows carrying a \"<time> h at <temperature> \u00b0C\" stage")
    print("                (first stage) -> 51,275 of 72,142; split by reaction type")
    print("                into 3,917 / 14,191 / 9,946 / 3,100 / 2,112 / 2,095 /")
    print("                1,251 / 24,137.")
    print("  2. Summit   : skip the TYPE row, stack the 4 Reizman cases (385 rows),")
    print("                add the substrate_type column.")
    print("  3. DoyleHTE : cut to the complete 15 x 20 grid (done above) -> 3,600 rows.")
    print("  4. rxn_yields: xlsx -> CSV, column names unchanged (5,760 rows).")
    print()
    print("  Prepared this way the pipeline reproduces the paper; the raw files, a")
    print("  different cut, or a missing step changes Table 1, the ceilings and R2.")
    print("  \u53ea\u6709\u6309\u4e0a\u8ff0\u65b9\u5f0f\u5904\u7406\u540e\u653e\u5165\u7ba1\u7ebf\uff0c\u624d\u4f1a\u5f97\u5230\u8bba\u6587\u7684\u6570\u5b57\uff1b\u7528\u539f\u59cb\u6587\u4ef6\u3001")
    print("  \u5c11\u505a\u4e00\u6b65\u6216\u6362\u4e2a\u5207\u6cd5\uff0cTable 1\u3001ceiling \u548c R2 \u90fd\u4f1a\u4e0d\u540c\u3002")
    print("=" * 78)


SUMMIT_LABELS = [
    "3-bromoquinoline + dimethylisoxazole boronate",
    "3-chloropyridine + dimethylisoxazole boronate",
    "3-chloropyridine + benzofuran-2-boronic acid",
    "2-chloropyridine + 1-Boc-2-pyrroleboronic acid",
]


def _prep_summit(out_dir: Path) -> bool:
    """Stack the four Reizman cases (each carries a TYPE row) and add substrate_type,
    then rename the Baumgartner file.  Expected: 385 and 96 rows."""
    try:
        import pandas as pd
    except Exception as exc:
        print("  pandas is required: %s" % exc)
        return False
    frames = []
    for i, label in enumerate(SUMMIT_LABELS, 1):
        f = out_dir / ("_summit_case_%d.csv" % i)
        if not f.is_file():
            print("  missing %s" % f.name)
            return False
        d = pd.read_csv(f, skiprows=[1])          # row 2 is TYPE,DATA,DATA,...
        d = d.drop(columns=[c for c in d.columns if str(c).startswith("Unnamed")], errors="ignore")
        d["substrate_type"] = label
        frames.append(d)
        f.unlink()
    merged = pd.concat(frames, ignore_index=True)
    cols = ["NAME", "catalyst", "t_res", "temperature", "catalyst_loading", "ton",
            "yld", "substrate_type"]
    merged = merged[[c for c in cols if c in merged.columns]]
    dest = out_dir / "(Summit)Reizman_Suzuki_385.csv"
    merged.to_csv(dest, index=False, encoding="utf-8")
    print("  Reizman 4 cases        : %d rows %s" % (len(merged), "ok" if len(merged) == 385 else "MISMATCH (expected 385)"))

    b = out_dir / "_summit_baumgartner.csv"
    ok_b = True
    if b.is_file():
        bd = pd.read_csv(b, skiprows=[1])
        bd = bd.drop(columns=[c for c in bd.columns if str(c).startswith("Unnamed")], errors="ignore")
        bd.to_csv(out_dir / "(Summit)Baumgartner_CN_coupling_96.csv", index=False, encoding="utf-8")
        ok_b = len(bd) == 96
        print("  Baumgartner            : %d rows %s" % (len(bd), "ok" if ok_b else "MISMATCH (expected 96)"))
        b.unlink()
    return len(merged) == 385 and ok_b


def _prep_doyle(raw: Path, out_dir: Path) -> bool:
    """Keep the complete 15 x 20 grid (300 cells x 12 rows = 3,600) and rename the
    substrate column.  The raw file has 4,599 rows over 16 substrates x 24 additives;
    dropping substrate 'none' and the four additives that do not cover every
    condition is what the published file does."""
    try:
        import pandas as pd
    except Exception as exc:
        print("  pandas is required: %s" % exc)
        return False
    df = pd.read_csv(raw, low_memory=False)
    if "aryl_halide_smiles" in df.columns:
        df = df.rename(columns={"aryl_halide_smiles": "substrate_smiles"})
    df = df.drop(columns=[c for c in ("index",) if c in df.columns])
    print("  raw rows               : %d" % len(df))
    # drop the placeholder substrate, then keep the additives that cover the whole
    # 15 x 12 panel (the four extras cover fewer rows)
    df = df[df["substrate_smiles"].astype(str) != "none"]
    per_add = df.groupby("additive_smiles").size()
    full = per_add[per_add == per_add.max()]
    kept = df[df["additive_smiles"].isin(full.index)].reset_index(drop=True)
    # A56/D6: Table 2 needs CONDITION features, and this file only ships SMILES columns.
    # Derive a stable integer id per distinct structure -- one-to-one with the SMILES, so no
    # chemistry is invented, but the categorical slots (ligand / base / additive) can be filled.
    for _col, _id in (("ligand_smiles", "ligand_id"),
                      ("base_smiles", "base_id"),
                      ("additive_smiles", "additive_id"),
                      ("substrate_smiles", "substrate_id")):
        if _col in kept.columns:
            _levels = {v: i for i, v in enumerate(sorted(kept[_col].astype(str).unique()))}
            kept[_id] = kept[_col].astype(str).map(_levels).astype("int32")
    print("  id columns             : "
          + ", ".join("%s=%d" % (c, kept[c].nunique())
                      for c in ("ligand_id", "base_id", "additive_id") if c in kept.columns))
    dest = out_dir / "(DoyleHTE)CN_coupling_3600.csv"
    kept.to_csv(dest, index=False, encoding="utf-8")
    ok = len(kept) == 3600
    print("  complete grid          : %d rows (%d substrates x %d additives) %s"
          % (len(kept), kept.substrate_smiles.nunique(), kept.additive_smiles.nunique(),
             "ok" if ok else "MISMATCH (expected 3,600)"))
    raw.unlink(missing_ok=True)
    return ok


def _split_roche(master: Path, out_dir: Path) -> bool:
    """Filter the 72,142-row table and write the eight subsets.  Returns True on match."""
    try:
        import pandas as pd
    except Exception as exc:                                    # pragma: no cover
        print("  pandas is required to split the Roche66k table: %s" % exc)
        return False

    df = pd.read_parquet(master)
    rename = {
        "Reaction Type": "reaction_type", "Base": "base", "Catalyst": "catalyst",
        "Solvent": "solvent", "Ligand": "ligand", "Additive": "additive",
        "Coupling Reagent": "coupling_reagent", "Secondary Solvent": "solvent_2",
        "Tertiary Solvent": "solvent_3", "FG A": "fg_a", "FG B": "fg_b",
        "ELN_ID": "eln_id", "PLATENUMBER": "plate", "Coordinate": "well",
        "AREA_TOTAL_REDUCED": "yield", "z-Score": "z_score",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "output_column" not in df.columns:
        print("  the table has no output_column column -- cannot recover temperature/time")
        return False

    parsed = [parse_temp_time(s) for s in df["output_column"]]
    df["temperature"] = [t for t, _ in parsed]
    df["time"] = [h for _, h in parsed]

    print("  master rows            : %d" % len(df))
    kept = df[df["temperature"].notna() & df["time"].notna()].copy()
    print("  with temperature+time  : %d   (expected %d)" % (len(kept), ROCHE_KEPT_ROWS))
    if len(kept) != ROCHE_KEPT_ROWS:
        print("  !! the row count does not match the published one -- STOP")
        return False

    cols = ["reaction_type", "catalyst", "ligand", "base", "solvent", "additive",
            "coupling_reagent", "solvent_2", "fg_a", "fg_b", "temperature", "time",
            "yield", "z_score", "eln_id", "plate", "well"]
    order = [c for c in cols if c in kept.columns]
    ok = True
    written = {}
    for reaction, slug, expect in ROCHE_SUBSETS:
        sub = kept[kept["reaction_type"] == reaction][order].copy()
        if slug == "Amide_coupling" and "coupling_reagent" in sub.columns:
            sub = sub.rename(columns={"coupling_reagent": "amide_coupling_reagent"})
        name = "(Roche66k)%s_%d.csv" % (slug, expect)
        sub.to_csv(out_dir / name, index=False, encoding="utf-8")
        written[name] = len(sub)
        flag = "ok" if len(sub) == expect else "MISMATCH (expected %d)" % expect
        if len(sub) != expect:
            ok = False
        print("    %-42s %6d  %s" % (name, len(sub), flag))

    union = kept[kept["reaction_type"].isin(ROCHE_UNION[1])][order].copy()
    name = "(Roche66k)%s_%d.csv" % (ROCHE_UNION[0], ROCHE_UNION[2])
    union.to_csv(out_dir / name, index=False, encoding="utf-8")
    written[name] = len(union)
    flag = "ok" if len(union) == ROCHE_UNION[2] else "MISMATCH (expected %d)" % ROCHE_UNION[2]
    if len(union) != ROCHE_UNION[2]:
        ok = False
    print("    %-42s %6d  %s" % (name, len(union), flag))

    master.unlink(missing_ok=True)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i-agree", action="store_true",
                    help="confirm that you accept the licences printed above")
    ap.add_argument("--allow-unlicensed-rxn", action="store_true",
                    help="also fetch rxn_yields from its third-party mirror (the data "
                         "itself is under the AAAS terms, not the mirror's MIT)")
    ap.add_argument("--interactive", action="store_true",
                    help="ask for confirmation in the terminal instead of using --i-agree")
    ap.add_argument("--only", default="all",
                    help="comma separated: ai_rcp,bh,summit,suzukiml,roche (default all)")
    ap.add_argument("--out-dir", default=str(DATA), help="where to write the files")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    _licence_table()
    if not args.i_agree:
        if not args.interactive:
            return 2
        print("\nType  I AGREE  to accept these licences and start the download")
        print("(anything else quits): ", end="", flush=True)
        answer = sys.stdin.readline().strip()
        if answer.upper() != "I AGREE":
            print("\nNothing downloaded.")
            return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    want = {k.strip() for k in args.only.split(",")} if args.only != "all" else None
    failed = []
    for s in SOURCES:
        if want and s["key"] not in want:
            continue
        print("\n== %s (%s) ==" % (s["label"], s["licence"]))
        for url, name in s["files"]:
            dest = out_dir / name
            try:
                _fetch(url, dest, label=name)
            except Exception as exc:
                print("   %-52s FAILED: %s" % (name, exc))
                failed.append(name)
        if s["key"] == "summit" and not failed:
            print("\n   merging the four Reizman cases ...")
            if not _prep_summit(out_dir):
                failed.append("Summit merge")
        if s["key"] == "doyle" and not failed:
            raw = out_dir / "_doyle_cn_raw.csv"
            if raw.is_file():
                print("\n   preparing the 15 x 20 grid ...")
                if not _prep_doyle(raw, out_dir):
                    failed.append("DoyleHTE grid")
        if s["key"] == "roche" and not failed:
            master = out_dir / "_roche66k_master.parquet"
            if master.is_file():
                print("\n   downloaded: %s (%.2f MB)" % (master.name, master.stat().st_size / 1e6))
                do_split = True
                # rxn_yields: the only dataset whose data carries no licence.  A third-party
                # mirror (rxn4chemistry/rxn_yields, MIT) does redistribute the Science SI file,
                # but that MIT covers the repository's code, not the AAAS data.  It is fetched
                # only on an explicit second confirmation.
                rxn_ok = None
                if not failed:
                    want_rxn = args.allow_unlicensed_rxn
                    if not want_rxn and args.interactive:
                        print("\nType  yes  to fetch rxn_yields for LOCAL, NON-COMMERCIAL research use.")
                        print("The mirror is MIT, but the data is the Science supplementary file:")
                        print("reading and non-commercial research use ARE PERMITTED under the AAAS")
                        print("terms; redistribution is NOT.  Do not commit the file.  [y/N] ",
                              end="", flush=True)
                        want_rxn = sys.stdin.readline().strip().lower() in ("y", "yes")
                    if want_rxn:
                        url = ("https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/"
                               "data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx")
                        tmp = out_dir / "_rxn_yields_si.xlsx"
                        try:
                            _fetch(url, tmp, label="(rxn_yields)aap9112_Data_File_S1.xlsx")
                            import pandas as _pd
                            df = _pd.read_excel(tmp, sheet_name=0)
                            dest = out_dir / "(rxn_yields)aap9112_Data_File_S1.csv"
                            df.to_csv(dest, index=False, encoding="utf-8")
                            print("   -> %s  (%d rows, %d columns, names unchanged)"
                                  % (dest.name, len(df), df.shape[1]))
                            print("      local use only -- do not commit it")
                            if len(df) != 5760:
                                print("   !! expected 5,760 rows -- check the sheet")
                                failed.append("rxn_yields rows")
                            rxn_ok = True
                            tmp.unlink(missing_ok=True)
                        except Exception as exc:
                            print("   rxn_yields failed: %s" % exc)
                            failed.append("rxn_yields")

                _paper_notice()
                if args.interactive:
                    print("\nPress Enter (or type anything) to process the files listed")
                    print("above.  Close this window to stop here without processing.")
                    print("> ", end="", flush=True)
                    sys.stdin.readline()
                if do_split:
                    print("\n   splitting the master table into the eight subsets ...")
                    if not _split_roche(master, out_dir):
                        failed.append("Roche66k split")
                else:
                    print("   kept as %s; nothing was cut." % master.name)


    print("\n" + "=" * 78)
    if failed:
        print("FAILED: %s" % ", ".join(failed))
        return 1
    print("Done.  Files are in %s" % out_dir)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

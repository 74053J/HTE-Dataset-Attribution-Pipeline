# -*- coding: utf-8 -*-
"""Column mapping: global defaults + per-file overrides.

Priority order for every standard column:

    1. per-file override   (dataset == this file)      <- HIGHEST
    2. global alias        (dataset == '') by priority
    3. nothing             -> the slot is filled with a placeholder

``raw_column == 'disable'`` in an override means: this file has a candidate
column but it must NOT be used for that slot -- the slot is dropped from the
frame before the feature matrix is built.  A slot the table does not mention is
left exactly as the alias pass left it.

The file is ``col_mappings.csv`` with columns
``dataset, standard_column, raw_column, priority, note``.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

MAPPING_FILENAME = "col_mappings.csv"

#: slots whose values are numbers, not categories
NUMERIC_SLOTS = ("temperature", "time", "catalyst_loading", "partner_eq", "reagent_eq")

#: placeholder written when a slot has no source column
MISSING = "Unknown"

#: sentinel: the table says this file's column for the slot must not be used
DISABLED = object()

#: placeholders inside the data that mean "this component is absent"
_NONE_TOKENS = {"nosolvent", "noadditive", "none", "nan", "", "no solvent",
                "no additive", "n/a", "-"}


def _find_mapping(data_dir: str) -> str | None:
    """Locate col_mappings.csv.  Only the packaged data/ directory is searched: the
    legacy fallbacks ("Data", "../Data") belonged to the old project layout and made
    the pipeline depend on where it happened to be run from."""
    from pathlib import Path as _P
    _pkg_data = _P(__file__).resolve().parents[1] / "data"
    for d in (data_dir, str(_pkg_data)):
        p = os.path.join(d, MAPPING_FILENAME)
        if os.path.exists(p):
            return p
    return None


def load_mapping(data_dir: str) -> dict | None:
    """Parse ``col_mappings.csv`` into {global: {slot: [raw...]}, override: {file: {slot: raw}}}."""
    path = _find_mapping(data_dir)
    if path is None:
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = {"dataset", "standard_column", "raw_column"}
    if not need.issubset(df.columns):
        return None
    df = df.fillna("")
    mapping = {"global": {}, "override": {}, "path": path}
    for _, r in df.iterrows():
        ds = str(r["dataset"]).strip()
        slot = str(r["standard_column"]).strip()
        raw = str(r["raw_column"]).strip()
        if not slot or not raw:
            continue
        if ds:
            mapping["override"].setdefault(ds, {})[slot] = raw
        else:
            prio = r.get("priority", "")
            try:
                prio = int(float(prio))
            except (TypeError, ValueError):
                prio = 10 ** 6
            mapping["global"].setdefault(slot, []).append((prio, raw))
    for slot, lst in mapping["global"].items():
        mapping["global"][slot] = [raw for _p, raw in sorted(lst)]
    return mapping


def resolve_for_file(mapping: dict, filename: str, header) -> dict:
    """slot -> raw column name, DISABLED, or absent (the table says nothing)."""
    ov = mapping["override"].get(filename, {})
    low = {str(h).strip().lower(): str(h).strip() for h in header}
    out = {}
    for slot, aliases in mapping["global"].items():
        if slot in ov:
            v = ov[slot]
            out[slot] = DISABLED if v == "disable" else v
            continue
        pick = None
        for cand in aliases:
            if cand in header:
                pick = cand
                break
            if cand.lower() in low:
                pick = low[cand.lower()]
                break
        if pick is not None:
            out[slot] = pick
        else:
            # keep the key so the caller can still say "this file has no
            # column for that slot" (the four newer slots need to know)
            out[slot] = None
    for slot, v in ov.items():          # override-only slots
        if slot not in out:
            out[slot] = DISABLED if v == "disable" else v
    return out


# Each pattern must be specific enough not to fire on ordinary chemistry words
# ("cumene" contains "cu", "nitrile" contains "ni", "amine" contains "mi").
_METAL_RULES = [
    ("Cu", r"\bcui?\b|cu2o|cu\(|\bcuo\b|cubr2?|cucl2?|copper"),
    ("Ru", r"\bru\b|\(ru\(|ruthenium|ru\(|rucl"),
    ("Ni", r"\bni\b|nicl2|ni\(|nickel"),
    ("Zn", r"\bzn\b|zncl2|zinc|zn\("),
    ("Pd", r"pd|peppsi"),                 # pd(allyl), pdcl2, xantphospdcl2, PEPPSI-...
]


def parse_metal(*series: pd.Series) -> pd.Series:
    """The metal that appears EARLIEST in the catalyst name / SMILES.

    Taking the leftmost match matters: '(Ru(p-cymene)Cl2)2, Copper' is primarily
    a ruthenium catalyst with a copper additive, and 'CuI' must not be missed
    just because the token has no word boundary after 'Cu'.
    """
    base = series[0]
    out = pd.Series(MISSING, index=base.index, dtype=object)
    for s in series:
        if s is None:
            continue
        txt = s.astype(str).str.lower()
        best_pos = pd.Series(np.inf, index=txt.index)
        best_metal = pd.Series("", index=txt.index, dtype=object)
        for metal, pat in _METAL_RULES:
            rx = re.compile(pat)
            loc = txt.map(lambda v, rx=rx: (rx.search(v).start()
                                            if rx.search(v) else -1))
            hit = (loc >= 0) & (loc < best_pos)
            best_pos = best_pos.where(~hit, loc)
            best_metal = best_metal.where(~hit, metal)
        fill = (out == MISSING) & (best_metal != "")
        out[fill] = best_metal[fill]
    return out


def _normalise_none(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.where(~s.str.lower().isin(_NONE_TOKENS), "none")


def apply_mapping(df: pd.DataFrame, data_path: str, data_dir: str,
                     raw_feature_names: dict | None = None,
                     source_df: pd.DataFrame | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Overwrite the condition slots according to col_mappings.csv.

    ``source_df`` must be the frame as read from disk, BEFORE any alias pass.
    Resolving against ``df`` itself is wrong: by the time this runs, ``df`` has
    derived columns that shadow real raw columns (e.g. Summit_CN has a raw
    column literally named ``catalyst``, which an older per-file routing table
    overwrote).

    Returns the frame (same object, mutated) and records provenance in
    ``raw_feature_names``.
    """
    mapping = load_mapping(data_dir)
    if mapping is None:
        if verbose:
            print("[mapping] col_mappings.csv not found -- keeping the old mapping")
        return df
    src = source_df if source_df is not None else df
    fname = os.path.basename(data_path)
    header = [str(c).strip() for c in src.columns]
    resolved = resolve_for_file(mapping, fname, header)
    raw_feature_names = raw_feature_names if raw_feature_names is not None else {}
    disabled = []

    for slot, raw in resolved.items():
        if slot == "metal":
            continue
        if raw is DISABLED:
            # explicit 'disable': the slot must not reach the feature matrix.
            # The column is dropped at the END of this function, so that the
            # metal parser below can still read `catalyst` when the file does
            # keep it.
            disabled.append(slot)
            continue
        if raw is None or raw not in src.columns:
            # Do not clobber slots this table says nothing about.  For the four
            # NEW slots an absent source is meaningful, so make it explicit.
            if slot in ("cosolvent", "additive"):
                df[slot] = "none"
                raw_feature_names[slot] = ""
            elif slot in ("partner_eq", "reagent_eq"):
                df[slot] = pd.NA
                raw_feature_names[slot] = ""
            continue
        df[slot] = src[raw].values
        raw_feature_names[slot] = raw

    # numeric slots -> numbers
    for slot in NUMERIC_SLOTS:
        if slot in df.columns:
            df[slot] = pd.to_numeric(df[slot], errors="coerce")
            if slot in raw_feature_names and raw_feature_names[slot]:
                pass

    # placeholders that mean "component absent" become the literal 'none'
    for slot in ("cosolvent", "additive"):
        if slot in df.columns:
            df[slot] = _normalise_none(df[slot])

    # metal: parse from the catalyst name, falling back to its SMILES
    cat = df["catalyst"] if "catalyst" in df.columns else None
    cat_smi = None
    for cand in ("catalyst_smiles", "Catalyst SMILES"):
        if cand in df.columns:
            cat_smi = df[cand]
            break
    if cat is not None:
        c = cat.astype(str)
        df["metal"] = parse_metal(c.where(c != MISSING, ""),
                                  cat_smi.astype(str) if cat_smi is not None else None) \
            if cat_smi is not None else parse_metal(c.where(c != MISSING, ""))
        raw_feature_names["metal"] = "(parsed from catalyst)"

    # explicit 'disable' overrides: drop the column, so neither a model nor the
    # condition key can pick it up.  Done last so `metal` could still be parsed
    # from `catalyst` above.  The list is also published on the frame, because a
    # later step of the loader re-fills missing slots with placeholders and the
    # slot must not come back that way.
    for slot in disabled:
        if slot in df.columns:
            df.drop(columns=[slot], inplace=True)
            raw_feature_names.pop(slot, None)
    if disabled:
        df.attrs["disabled_slots"] = sorted(disabled)

    if verbose:
        shown = {k: v for k, v in resolved.items() if v and v is not DISABLED}
        print("[mapping] %s -> %d slots from %d raw columns"
              % (fname, len(shown), len({v for v in shown.values()})))
        if disabled:
            print("  [mapping] disabled by table: %s" % ", ".join(sorted(disabled)))
    return df



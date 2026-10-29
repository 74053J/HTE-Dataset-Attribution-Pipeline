#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read the catalytic metal out of the data instead of assuming one.

Two independent readers, ported from the V1 metal/solvent audit:

  (a) SMILES -- parse the catalyst-bearing SMILES column with RDKit and keep the
      transition-metal atomic numbers that occur anywhere in the molecule;
  (b) NAME   -- match a metal token in the catalyst *name* string after removing
      every ligand stem that occurs in the corpus, and additionally allowing
      formula concatenation, so ``dtbpfPdCl2`` and ``XantPhosPdCl2`` both read Pd.

The SMILES reading wins when both speak and disagree: it is structural evidence,
the name is a label.  A file flagged ``metal_free: true`` in
``data/alias_mapping.yaml`` yields ``none``; a row with neither reading yields
``unknown``.  No dataset name is written in this module -- the columns to read
are derived from ``data/col_mappings.csv``.

Run ``python -m hte.metal_reading`` for the per-dataset audit.
"""
from __future__ import annotations

import io
from collections import Counter
import contextlib
import re
import sys
import unicodedata
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

#: transition metals relevant to cross-coupling catalysis
METALS = ("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W "
          "Os Ir Pt Au Hg").split()
_METAL_SET = frozenset(METALS)
#: a metal token, optionally followed by a formula continuation
_TOK = re.compile(r"([A-Z][a-z]?)")
_SYMB = re.compile(r"(?<![A-Za-z])(%s)(?![a-z])"
                  % "|".join(sorted(METALS, key=len, reverse=True)))
_CONCAT_AFTER = re.compile(r"(%s)(?=[A-Z0-9()\[\]]|$)" % "|".join(sorted(METALS, key=len,
                                                                          reverse=True)))
#: English names written out in reagent labels ("Copper(II) acetate")
NAME_TO_SYM = {"copper": "Cu", "palladium": "Pd", "ruthenium": "Ru", "nickel": "Ni",
               "iron": "Fe", "silver": "Ag", "gold": "Au", "rhodium": "Rh",
               "iridium": "Ir", "platinum": "Pt", "cobalt": "Co", "manganese": "Mn",
               "zinc": "Zn", "chromium": "Cr", "vanadium": "V", "molybdenum": "Mo",
               "tungsten": "W", "zirconium": "Zr"}

_CONCAT_BEFORE = re.compile(r"(?<=[a-z0-9)\]])(%s)(?![a-z])"
                            % "|".join(sorted(METALS, key=len, reverse=True)))

STEMS: list[str] = []
_LIG_CACHE: dict[str, str] = {}


def _s(v) -> str:
    if v is None:
        return ""
    try:
        if v != v:                                    # NaN
            return ""
    except Exception:                                 # noqa: BLE001
        pass
    return str(v).strip()


def normalise(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", " ", t).strip()


def build_stems(values) -> list[str]:
    """Ligand vocabulary, taken from the corpus itself (longest first)."""
    out = set()
    for v in values:
        t = normalise(_s(v))
        if not t:
            continue
        for part in re.split(r"[/,;()\[\]]|\s+", t):
            part = part.strip()
            if len(part) >= 4 and not re.search(r"[0-9]", part):
                out.add(part)
    return sorted(out, key=len, reverse=True)


def strip_stems(t: str) -> str:
    low = t.lower()
    for stem in STEMS:
        low = low.replace(stem.lower(), " ")
    return low


def strip_stems_case(t: str) -> str:
    """Same as strip_stems but keeps the original case (for formula matching)."""
    out = t
    for stem in STEMS:
        out = re.sub(re.escape(stem), " ", out, flags=re.I)
    return out


def metals_from_name(value) -> set[str]:
    """Metal tokens in a catalyst name, after ligand stems are removed."""
    t = normalise(_s(value))
    if not t:
        return set()
    t = re.sub(r"[/.]", " ", t)
    stripped = strip_stems(t)                  # lower case, for the English names
    stripped_case = strip_stems_case(t)        # original case, for element symbols
    hits = set(_SYMB.findall(stripped_case))
    hits |= set(_CONCAT_AFTER.findall(stripped_case))
    hits |= set(_CONCAT_BEFORE.findall(stripped_case))
    low = t.lower()
    for word, sym in NAME_TO_SYM.items():
        if word in low:
            hits.add(sym)
    # lower-case element symbols written inline, e.g. "j-009-pd(oac)2".  The
    # pattern is case-sensitive on the original string, so the "CO" of a
    # carbonate (Na2CO3) is not read as cobalt.
    for sym in re.findall(r"(?<![A-Za-z])(pd|cu|ni|ru|fe|ag|au|rh|ir|pt|mn|zn|cr|mo)"
                         r"(?![A-Za-z])", t):
        hits.add(sym.title())
    return hits


def metals_from_smiles(values) -> set[str]:
    """Transition metals present in any of the given SMILES strings."""
    hits: set[str] = set()
    for v in values:
        smi = _s(v)
        if not smi:
            continue
        if smi in _LIG_CACHE:
            hits |= set(_LIG_CACHE[smi].split())
            continue
        mol = Chem.MolFromSmiles(smi)
        found = set()
        if mol is not None:
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                if sym in _METAL_SET:
                    found.add(sym)
        _LIG_CACHE[smi] = " ".join(sorted(found))
        hits |= found
    return hits


def read_rows(names, smiles, metal_free: bool = False):
    """Per-row metal reading. Returns (metal_final, metal_source).

    Order of evidence:
      1. the SMILES reader (structural) wins, after dropping a ferrocene Fe that
         sits in the ligand scaffold and after a data-driven tie-break when the
         SMILES still names several metals;
      2. a PEPPSI precatalyst is palladium by definition;
      3. a ligand name on its own cannot identify the metal, so the file's
         dominant metal is used only when it accounts for >=95 % of the rows that
         could be read (``inferred``); otherwise the slot stays empty;
      4. otherwise the name reader speaks;
      5. a metal-free file yields ``none``.
    """
    n = len(names)

    def _cell(seq, i):
        return seq.iloc[i] if hasattr(seq, "iloc") else seq[i]

    def _read(i):
        nm = _cell(names, i)
        sm = _cell(smiles, i)
        got_s = metals_from_smiles([sm]) if _s(sm) else set()
        return got_s, metals_from_name(nm)

    raw = [_read(i) for i in range(n)]
    freq: dict[str, int] = {}
    for got_s, got_n in raw:
        for m in (got_s | got_n):
            freq[m] = freq.get(m, 0) + 1
    total = sum(freq.values())

    final, source = [], []
    for i, (got_s, got_n) in enumerate(raw):
        nm = _cell(names, i)
        cand = set(got_s)
        if len(cand) > 1:
            cand.discard("Fe")
        if cand == {"Fe"}:
            cand = set()
        if len(cand) > 1:
            cand = {max(sorted(cand), key=lambda m: freq.get(m, 0))}
        if cand:
            agree = got_n & cand
            metal = sorted(agree)[0] if len(agree) == 1 else sorted(cand)[0]
            src = "smiles"
        elif got_n:
            metal = max(sorted(got_n), key=lambda m: freq.get(m, 0))
            src = "name"
        elif "cx31" in _s(nm).lower().replace(" ", ""):
            # CX31 is a palladium NHC precatalyst (Umicore / Sigma-Aldrich
            # CAT 709042, CAS 884879-23-6), sold for Suzuki and Buchwald-
            # Hartwig couplings: a curated precatalyst family, like PEPPSI.
            metal, src = "Pd", "name"
        elif "peppsi" in _s(nm).lower():
            metal, src = "Pd", "name"
        elif _s(nm) and freq:
            top = max(freq, key=lambda m: freq[m])
            if total and freq[top] / total >= 0.95:
                metal, src = top, "inferred"
            else:
                metal, src = "none", "none"
        elif _s(nm) and not strip_stems_case(_s(nm)).strip() and _s(nm).strip():
            # the cell holds nothing but a known ligand name (the whole string
            # is a learned ligand stem), so the metal is read from the ligand
            # family: those bulky biaryl phosphines are palladium ligands.
            metal, src = "Pd", "ligand_family"
        else:
            metal, src = "none", "none"
        final.append(metal)
        source.append(src)

    # Final pass (author's rule, 2026-10-28): when the slot is blank for less than
    # 5 % of the rows, those blanks are filled with the dominant metal.  Rows that
    # already carry a reading are NEVER overwritten -- a genuine minority metal
    # (27 nickel records in bh-hte-ood) must survive.
    if not metal_free and n:
        blank = [i for i, m in enumerate(final) if m == "none"]
        if blank and len(blank) / n < 0.05:
            known = Counter(m for m in final if m != "none")
            if known:
                top = known.most_common(1)[0][0]
                for i in blank:
                    final[i] = top
                    source[i] = "blank_filled"
    return final, source


def _catalyst_columns(mapping, fname: str, header: list[str]):
    """(name_column, smiles_column) for the catalyst slot, from the mapping."""
    from hte.mapping import resolve_for_file

    resolved = resolve_for_file(mapping, fname, header)
    raw = resolved.get("catalyst")
    if not isinstance(raw, str):
        # the catalyst slot is unset for this file, but a catalyst-bearing column
        # may still be present; fall back to the column names themselves.
        smi = next((c for c in header
                    if "catalyst" in c.lower() and "smiles" in c.lower()), None)
        nm = next((c for c in header
                   if "catalyst" in c.lower() and "name" in c.lower()), None)
        return (nm, smi) if (nm or smi) else (None, None)
    if re.search(r"smiles", raw, re.I):
        return None, raw
    stem = re.sub(r"[\s_\-]*(name|Name|NAME)\s*$", "", raw)
    for cand in header:
        if cand == raw:
            continue
        if re.search(r"smiles", cand, re.I) and stem and stem.lower() in cand.lower():
            return raw, cand
    return raw, None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(ROOT))
    from hte.data import load_real_dataset
    from hte.mapping import load_mapping

    mapping = load_mapping(str(DATA))
    files = sorted(p for p in DATA.glob("*.csv")
                   if not p.name.startswith("col_mappings"))
    frames, lig_values = {}, []
    for p in files:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df, _raw = load_real_dataset(str(p), str(DATA))
        except Exception:
            continue
        frames[p.name] = df
        if "ligand" in df.columns:
            lig_values.extend(df["ligand"].astype(str).tolist())

    global STEMS
    STEMS = build_stems(lig_values)
    print("ligand stems learned from the corpus: %d" % len(STEMS))
    print("%-44s %6s  %s" % ("file", "n", "catalyst metal"))
    for fname, df in frames.items():
        nm_col, smi_col = _catalyst_columns(mapping, fname, list(df.columns))
        if nm_col is None and smi_col is None:
            print("%-44s %6d  (no catalyst column)" % (fname[:44], len(df)))
            continue
        names = df[nm_col] if nm_col else [""] * len(df)
        smiles = df[smi_col] if smi_col else [""] * len(df)
        final, source = read_rows(names, smiles)
        vc = {}
        for m in final:
            vc[m] = vc.get(m, 0) + 1
        top = sorted(vc.items(), key=lambda kv: -kv[1])[:4]
        print("%-44s %6d  %s   [name=%s smiles=%s]"
              % (fname[:44], len(df),
                 ", ".join("%s %.0f%%" % (k, 100.0 * v / len(df)) for k, v in top),
                 nm_col, smi_col))
    return 0


if __name__ == "__main__":
    sys.exit(main())


#: metals that appear as stoichiometric salts / bases rather than as catalysts
#: alkali / alkaline counter-ions: present in almost every row, so they never
#: mark a stoichiometric transition-metal salt (Ag, Zn and Cu do)
SALT_METALS = frozenset("Na K Li Cs Mg Ca Ba".split())


def classify_rows(final, aux_metals):
    """Label each row ``catalytic``, ``salt`` or ``unknown``.

    ``aux_metals`` is the per-row set of metals found in the base / additive /
    reagent columns.  A row whose catalytic metal is Pd and whose only other
    metal sits in an auxiliary column is a palladium reaction with a
    stoichiometric salt (V1's class C), not a bimetallic catalyst.
    """
    out = []
    for m, aux in zip(final, aux_metals):
        aux = set(aux or ())
        if m in ("none", "none"):
            out.append("none")
        elif m == "Pd" and (aux - {"Pd"} - SALT_METALS):
            # only a non-alkali metal in an auxiliary column marks a salt; a
            # potassium or sodium base is a counter-ion, not a stoichiometric
            # transition-metal additive.
            out.append("salt")
        elif m in SALT_METALS:
            out.append("salt")
        else:
            out.append("catalytic")
    return out

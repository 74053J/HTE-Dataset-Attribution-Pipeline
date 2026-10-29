#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sync the rendered panels into the manuscript, and verify it.

The manuscript embeds twelve images in this order (verified against v3.37):

    1  Figure 1 (a)     figures/fig1a.png
    2  Figure 1 (b)     figures/fig1b.png
    3  Figure 2         figures/fig2.png
    4  Figure 3 (a)     figures/fig3a_shares.png
    5  Figure 3 (b)     figures/fig3b_ranks.png
    6  Figure 4 (a)     figures/fig4a_reactant1.png
    7  Figure 4 (b)     figures/fig4b_reactant2.png
    8  Figure 5         figures/fig5_heatmap.png
    9  Figure 6         figures/fig6_enrich.png
   10  Figure 7         figures/fig7_protocols.png
   11  Figure 8 (a)     figures/fig8a_acquisition.png
   12  Figure 8 (b)     figures/fig8b_mu_sd.png

The Figure 8(b) backup (``fig8b_backup_loading.png``) is never embedded.

Each embedded image is replaced in place: its binary blob and its display extent
are updated together, so a panel whose aspect ratio changed is not stretched.
The script then re-opens the result and checks every image's SHA-256 against the
file it came from, so "the figure is in the document" is a verified fact rather
than an assumption.

Usage
    python scripts/sync_figures_to_docx.py --src <paper.docx> --dst <out.docx>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "figures"
ORDER = [
    ("Figure 1 (a)", "fig1a.png"),
    ("Figure 1 (b)", "fig1b.png"),
    ("Figure 2", "fig2.png"),
    ("Figure 3 (a)", "fig3a_shares.png"),
    ("Figure 3 (b)", "fig3b_ranks.png"),
    ("Figure 4 (a)", "fig4a_reactant1.png"),
    ("Figure 4 (b)", "fig4b_reactant2.png"),
    ("Figure 5", "fig5_heatmap.png"),
    ("Figure 6", "fig6_enrich.png"),
    ("Figure 7", "fig7_protocols.png"),
    ("Figure 8 (a)", "fig8a_acquisition.png"),
    ("Figure 8 (b)", "fig8b_mu_sd.png"),
]
NS_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
NS_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"


def _shapes(doc):
    """Inline shapes in document order, each with its rId."""
    out = []
    for shape in doc.inline_shapes:
        blip = shape._inline.findall(".//" + NS_A)
        if blip:
            out.append((shape, blip[0].get(NS_R)))
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from docx import Document
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="the manuscript to update")
    ap.add_argument("--dst", required=True, help="where to write the result")
    ap.add_argument("--fig-dir", default=str(FIG_DIR))
    args = ap.parse_args()

    fig_dir = Path(args.fig_dir)
    missing = [n for _, n in ORDER if not (fig_dir / n).is_file()]
    if missing:
        print("[sync] missing renders: %s" % missing)
        return 1

    doc = Document(args.src)
    shapes = _shapes(doc)
    if len(shapes) != len(ORDER):
        print("[sync] the manuscript has %d images, the map has %d -- refusing to "
              "guess which is which." % (len(shapes), len(ORDER)))
        return 1

    for (label, name), (shape, rid) in zip(ORDER, shapes):
        data = (fig_dir / name).read_bytes()
        doc.part.related_parts[rid]._blob = data
        with Image.open(fig_dir / name) as im:
            w, h = im.size
        width = shape.width                       # keep the layout width
        shape.height = int(round(width * h / w))  # restore the true aspect ratio
        print("  %-14s <- %-24s (%d x %d)" % (label, name, w, h))

    doc.save(args.dst)
    print("[sync] wrote %s" % args.dst)

    # verify every embedded image against the file it came from
    doc2 = Document(args.dst)
    shapes2 = _shapes(doc2)
    bad = 0
    for (label, name), (_shape, rid) in zip(ORDER, shapes2):
        got = hashlib.sha256(doc2.part.related_parts[rid].blob).hexdigest()
        want = hashlib.sha256((fig_dir / name).read_bytes()).hexdigest()
        ok = got == want
        bad += 0 if ok else 1
        print("  verify %-14s %s" % (label, "MATCH" if ok else "MISMATCH"))
    print("[sync] %d / %d images verified" % (len(ORDER) - bad, len(ORDER)))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

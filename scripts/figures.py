#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate the nine figures of the paper from this repository.

The paper carries nine image files: fig1a, fig1b, fig2, fig3a, fig3b, fig4,
fig5, fig6a and fig6b.  They are rendered to ``figures/<name>.png`` at 300 dpi.

Font handling
-------------
The frozen pipeline ``archive/Analysis_all_v16.py`` resets
``font.family`` to ``sans-serif`` when it runs, so the rcParams have to be
applied *after* it is loaded.  ``_paper_rcparams`` reproduces them directly
(``runpy.run_path``), assert that the family really is ``['serif']``, and only
then touch matplotlib further.  The legacy file is never imported or edited.

Two classes of figure
---------------------
``fig1a``, ``fig1b``, ``fig2`` read the shipped tables in ``results/`` and need
nothing else.  ``fig5`` also reads shipped tables -- ``results/protocol_cv.csv``
for the three in-dataset protocol bars and ``results/transfer_check.csv`` for
the cross-dataset bar -- and refits nothing.  ``fig3a``, ``fig3b``, ``fig4``
retrain the native-categorical XGBoost model on the AI_RCP_BH dataset (cheap:
seconds) and derive the SHAP attribution from it; they need the raw dataset,
which is not redistributed
with this repository, so point ``--data-dir`` at a directory holding
``(AI_RCP_pub)bh_data_clean_all_whitelisted.csv`` plus the pipeline's
``col_mappings.csv``.  ``fig6a``/``fig6b`` are the heavy ones: they display
outputs of the virtual-library / Bayesian-optimisation path and are refused
unless ``--allow-heavy`` is given.

Usage
-----
    python scripts/figures.py --list
    python scripts/figures.py --only fig2
    python scripts/figures.py --only fig1a fig1b fig2
    python scripts/figures.py --only fig1a fig1b --grid results/grid13_v18.csv
    python scripts/figures.py --only fig2 --ceiling outputs/final_numbers/ceiling_11_for_fig2.csv
    python scripts/figures.py --only fig3a --data-dir ../Suzuki-Coupling-ML/Data
    python scripts/figures.py --only fig6a --allow-heavy

The virtual-library / Bayesian-optimisation generators in the legacy pipeline
(``build_virtual_library``, ``stream_virtual_library``,
``generate_bo_results_parallel``, ``generate_bo_results_two_stage``,
``generate_bo_results_streaming``, ``generate_loading_curves_pdf``) are never
called from this script.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

# Pin the thread pools BEFORE numpy/xgboost are imported.  The frozen pipeline used to do
# this as a side effect of being loaded; without it XGBoost sizes its pool from the
# machine, and the SHAP figures (3a, 4a, 4b) stop being byte-reproducible.
import os as _os

for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_tv, "4")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
DPI = 450
FS, FL = 7.0, 8.0

# Matplotlib must be headless before the legacy pipeline pulls pyplot in.
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


# Loading the frozen pipeline used to be the way these settings were applied: importing
# it sets sixteen matplotlib parameters that the paper's figures were drawn with.  They
# are reproduced here explicitly, so that the pipeline no longer has to be executed.
def _paper_rcparams():
    import matplotlib
    matplotlib.rcParams.update({
        "backend": "Agg",
        "backend_fallback": False,
        "figure.dpi": 300.0,
        "figure.titlesize": 11.0,
        "font.family": ["serif"],
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.0,
        "legend.fontsize": 7.0,
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.fontset": "custom",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.rm": "Times New Roman",
        "savefig.dpi": 300.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
    })


_paper_rcparams()
NS = {}

assert mpl.rcParams["font.family"] == ["serif"], (
    "font.family is %r after the paper rcParams were applied; "
    "expected ['serif'].  The pipeline forces sans-serif on import, so every "
    "figure would come out in the wrong typeface.  Aborting."
    % (mpl.rcParams["font.family"],)
)

# Apply the paper's typography only now that the pipeline has finished resetting
# rcParams.  Times New Roman for text *and* for every mathtext glyph.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.sf": "Times New Roman",
    "mathtext.tt": "Times New Roman",
    "mathtext.cal": "Times New Roman:italic",
    "axes.unicode_minus": False,
    "font.size": FS,
    "axes.titlesize": FL,
    "axes.labelsize": FL,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "savefig.dpi": DPI,
})

# The second assert is deliberate belt-and-braces: nothing between load() and
# here may have flipped the family back to a sans-serif default.
assert mpl.rcParams["font.family"] == ["serif"], mpl.rcParams["font.family"]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # noqa: E402

# Colours used by the paper.
BLUE, RED, GREY = "#2E5FA3", "#B03A2E", "0.45"

# Tight output and post-crop panel stamps
#
# Every figure is saved with bbox_inches="tight" and pad_inches=10/dpi, so
# the white margin between the outermost artist and the image edge is ten
# pixels at the saved dpi.  The (a)/(b) panel letters are deliberately NOT
# matplotlib artists: they are stamped into the finished bitmap with PIL, so
# they cannot pull the tight bounding box outwards.  The letters keep the
# typeface and point size the figures used before (serif bold, FL points).
SAVE_PAD_IN = 10.0 / DPI                 # 0.0333 in == 10 px at 300 dpi
PANEL_INSET_PX = 4                       # inset of the stamp from the bitmap corner
PANEL_LABEL = {                          # only figures that carry a letter today
    "fig1a": "(a)",
    "fig1b": "(b)",
    # the paper2 layout writes this panel as "fig3a_shares"; the old "fig3a" key
    # never matched, so Figure 3(a) shipped without its letter (fixed 2026-10-30)
    "fig3a_shares": "(a)",
    "fig3a": "(a)",
    "fig3b": "(b)",
    "fig6a": "(a)",
    # Figure 8(a) was missing from this table entirely (fixed 2026-10-30)
    "fig8a_acquisition": "(a)",
    "fig8b_backup_loading": "(b)",
    "fig3b_ranks": "(b)",
    "fig4a_reactant1": "(a)",
    "fig4b_reactant2": "(b)",
    "fig8b_mu_sd": "(b)",
}

# Preferred offset direction for individual dataset labels of figure 2, as
# (dx, dy) in offset points.  The greedy placement in
# ``_place_dataset_labels`` is otherwise direction-blind: it only minimises
# the distance from the marker, so a label hemmed in by its neighbours can
# still settle above its own point.  A label listed here is charged a large
# penalty for candidates in the half-plane opposite to the given direction and
# a small one for candidates that deviate from it, so the search keeps the
# preferred side unless that side is genuinely blocked (an overlap costs two
# to five orders of magnitude more than this preference).
#
# Roche66k_SM sits just below Roche66k_Amide and left of Roche66k_CH; its
# default anchor is the upper right of the marker, which reads as "too high",
# so it is nudged to the lower right.
LABEL_DIRECTION = {
    "R66k_SM": (1.0, -1.0),
}

# Canvas sizes in cm, i.e. the sizes at which Word places the image.
# fig1a/1b/2/5/6a/6b come from the layouts of the original figure scripts;
# fig3a/3b/4 are the extents used for the final (font-correct) rebuild.
SIZE = {
    "fig1a": (8.43, 5.90),
    "fig1b": (6.99, 5.05),
    "fig2": (9.09, 6.78),
    "fig3a": (8.20, 5.60),
    "fig3b": (6.94, 5.79),
    "fig4": (10.05, 5.84),
    "fig5": (10.01, 5.95),
    "fig6a": (7.79, 6.22),
    "fig8b_backup_loading": (7.75, 6.18),
}

# Short labels used on the ceiling figure -- the `label:` field of
# data/alias_mapping.yaml, so the display text lives with the dataset table.
def _short_labels() -> dict:
    import yaml

    mapping = yaml.safe_load(
        (DATA_DIR / "alias_mapping.yaml").read_text(encoding="utf-8"))
    return {k: str(v.get("label") or k) for k, v in mapping["datasets"].items()}


SHORT_LABEL = _short_labels()

# PROVENANCE -- Figure 5: all four bars are read from shipped tables.
#
#   * random-split CV         +0.7687    <- results/protocol_cv.csv
#   * substrate-disjoint CV   -0.1827    <- results/protocol_cv.csv
#   * ordered CV              -1.1743    <- results/protocol_cv.csv
#   * cross-dataset transfer  -1.119212  <- MEAN of results/transfer_check.csv
#
# The first three used to be hardcoded literals (+0.766, -0.183, -1.235)
# carried over from the authoring script `make_figs_v14.py`; the ordered bar
# (-1.235) could not be reproduced from the repository.  They are now the
# recomputed 5-fold values for XGBoost on AI_RCP_BH with integer encoding
# (results/protocol_cv.csv, written by outputs/last_gaps/20_protocol_cv.py).
#
# The fourth bar used to plot the MEDIAN of `r2_transfer` (-1.171 from the
# paper's rounded value, or -1.0699 from the shipped table) although the paper
# text quotes the MEAN.  That was a mistake: the bar now plots the mean over
# the table's 12 targets (-1.119212).  The old TRANSFER_MEDIAN_FALLBACK =
# -1.171 path is DELETED -- a missing table raises FigureInputMissing instead
# of silently plotting a literal.
PROTOCOL_CV_CSV = RES_DIR / "protocol_cv.csv"
TRANSFER_CSV = RES_DIR / "transfer_check.csv"
PROTOCOLS = [
    ("Random-split CV\n(unseen rows)", "random"),
    ("Substrate-disjoint CV\n(new substrates)", "substrate-disjoint"),
    ("Ordered CV\n(next round)", "ordered"),
]


class FigureInputMissing(RuntimeError):
    """Raised when a figure's input table is not shipped with the repository."""


_DATA_OVERRIDE: Path | None = None
# fig1a/fig1b/fig2 each read one shipped table.  Defaults are the 13-dataset
# corpus the paper reports; the overrides let a rebuild point elsewhere.
_GRID_OVERRIDE: Path | None = None
_CEILING_OVERRIDE: Path | None = None


def _grid_path() -> Path:
    # 13-dataset grid (156 cells); the old 16-dataset grid is in results/legacy/.
    return _GRID_OVERRIDE if _GRID_OVERRIDE is not None else RES_DIR / "grid13_v18.csv"


def _ceiling_path() -> Path:
    # 13-dataset Table I (11 computable ceilings); legacy 16-set copy in results/legacy/.
    return (_CEILING_OVERRIDE if _CEILING_OVERRIDE is not None
            else RES_DIR / "table1_v18_final.csv")


def _data_root() -> Path:
    """Directory searched for the raw datasets (repo-relative unless overridden)."""
    return _DATA_OVERRIDE if _DATA_OVERRIDE is not None else DATA_DIR


def _rel(path: Path) -> str:
    """Render a path relative to the repository root, for messages."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _canvas(name: str) -> tuple[float, float]:
    w_cm, h_cm = SIZE[name]
    return w_cm / 2.54, h_cm / 2.54


def _dataset_stem(alias: str) -> str:
    """Map a short dataset alias to its source file stem via alias_mapping.yaml."""
    import yaml

    mapping = yaml.safe_load(
        (DATA_DIR / "alias_mapping.yaml").read_text(encoding="utf-8"))
    return mapping["datasets"][alias]["source"]


def _case_study() -> str:
    """The dataset the case-study figures are drawn on (``case_study: true``)."""
    from hte.datasets import case_study
    return case_study(DATA_DIR / "alias_mapping.yaml")


def _dataset_path(alias: str) -> Path:
    stem = _dataset_stem(alias)
    root = _data_root()
    for base in (root, root / "raw"):
        candidate = base / (stem + ".csv")
        if candidate.is_file():
            return candidate
    raise FigureInputMissing(
        "dataset %r not found.  Looked for:\n"
        "  %s\n"
        "  %s\n"
        "The raw datasets are not redistributed with this repository; pass "
        "--data-dir pointing at a directory that holds the CSV (and the "
        "pipeline's col_mappings.csv)."
        % (stem, _rel(root / (stem + ".csv")), _rel(root / "raw" / (stem + ".csv")))
    )


# Shared layout helpers (ported from the scripts that produced the paper art)
def autofit(fig, margin: float = 0.070, iters: int = 12) -> None:
    """Shrink and centre the axes so the drawing sits inside the canvas.

    Text keeps its point size; only the plotting area contracts.  This is what
    keeps the authored point sizes equal to the printed point sizes.
    """
    w_in, h_in = fig.get_size_inches()
    last = None
    for _ in range(iters):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        sx = min(1.0, (w_in - 2 * margin) / max(bb.width, 1e-6))
        sy = min(1.0, (h_in - 2 * margin) / max(bb.height, 1e-6))
        dx = margin + max(0.0, (w_in - 2 * margin) - bb.width * sx) / 2 - bb.x0 * sx
        dy = margin + max(0.0, (h_in - 2 * margin) - bb.height * sy) / 2 - bb.y0 * sy
        if last and max(abs(sx - last[0]), abs(sy - last[1]),
                        abs(dx - last[2]) / w_in, abs(dy - last[3]) / h_in) < 1e-4:
            break
        last = (sx, sy, dx, dy)
        for ax in fig.axes:
            p = ax.get_position()
            ax.set_position([(p.x0 * w_in * sx + dx) / w_in,
                             (p.y0 * h_in * sy + dy) / h_in,
                             p.width * sx, p.height * sy])
    fig.canvas.draw()


def _margins(path: Path, ignore: tuple[int, int, int, int] | None = None
             ) -> tuple[int, int, int, int] | None:
    """Ink margins of a rendered PNG (top, bottom, left, right), in pixels.

    ``ignore`` is a pixel box (x0, y0, x1, y1) that is masked out before the
    scan -- used to keep the post-crop (a)/(b) stamp (which sits 4 px from the
    corner by construction) out of the white-margin measurement.
    """
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    arr = np.asarray(img)
    dark = (arr < 250).any(axis=2)
    if ignore is not None:
        x0, y0, x1, y1 = ignore
        dark[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return None
    return (int(rows[0]), int(arr.shape[0] - 1 - rows[-1]),
            int(cols[0]), int(arr.shape[1] - 1 - cols[-1]))


def _normalize_border(path: Path, margin_px: int = 10) -> None:
    """Make the visible white border exactly ``margin_px`` on every side.

    ``bbox_inches="tight"`` uses matplotlib's *layout* boxes: text boxes carry
    ascent/descent/leading that extend past the visible glyphs, and spine
    boxes are their centrelines, not their stroke.  At 300 dpi that leaves
    anywhere between 5 and 18 px of visible white around the ink.  The stated
    target is measured from the ink ("scan for the first non-white
    row/column"), so the saved bitmap is cropped to the ink box and re-padded
    with exactly ``margin_px`` white pixels.  No content is ever cut: the crop
    box is the full non-white bounding box.
    """
    with Image.open(path) as src:
        im = src.convert("RGB")
    arr = np.asarray(im)
    dark = (arr < 250).any(axis=2)
    rows = np.where(dark.any(axis=1))[0]
    cols = np.where(dark.any(axis=0))[0]
    if not len(rows) or not len(cols):
        raise RuntimeError("figure %s is blank" % path.name)
    inner = im.crop((int(cols[0]), int(rows[0]),
                     int(cols[-1]) + 1, int(rows[-1]) + 1))
    out = Image.new("RGB", (inner.width + 2 * margin_px,
                            inner.height + 2 * margin_px), (255, 255, 255))
    out.paste(inner, (margin_px, margin_px))
    out.save(path, format="PNG", dpi=(DPI, DPI))


def _panel_font_path() -> str:
    """TTF actually used by the figures for a bold serif letter."""
    prop = FontProperties(family=mpl.rcParams["font.family"], weight="bold")
    return fm.findfont(prop)


def _stamp_panel_label(path: Path, label: str) -> tuple[int, int, int, int]:
    """Draw ``label`` into an already-cropped PNG at its top-left corner.

    The stamp is applied *after* bbox_inches="tight" has fixed the bitmap, so
    it has no influence on the crop.  It uses the same serif family and the
    same point size (FL) as the old matplotlib ``fig.text`` letters, rendered
    at the figure's 300 dpi (4x supersampled, then resampled) for clean edges.
    Returns the ink box (x0, y0, x1, y1) for the margin measurement.
    """
    px_per_pt = DPI / 72.0
    font_px = FL * px_per_pt
    supersample = 4
    with Image.open(path) as src:
        base = src.convert("RGB")
    width, height = base.size
    mask = Image.new("L", (width * supersample, height * supersample), 0)
    draw = ImageDraw.Draw(mask)
    font = ImageFont.truetype(_panel_font_path(),
                              int(round(font_px * supersample)))
    draw.text((PANEL_INSET_PX * supersample, PANEL_INSET_PX * supersample),
              label, font=font, fill=255, anchor="lt")
    # BOX (area average) avoids the negative lobes / halo of LANCZOS, so the
    # ink really does start at PANEL_INSET_PX from the corner.
    mask = mask.resize((width, height), Image.BOX)
    ink = mask.getbbox()
    if ink is None:
        raise RuntimeError("panel label %r produced no ink" % label)
    base.paste(Image.new("RGB", base.size, (0, 0, 0)), (0, 0), mask)
    base.save(path, format="PNG", dpi=(DPI, DPI))
    return ink


def _save(fig, name: str) -> None:
    """Finish a figure, write figures/<name>.png at 300 dpi, and report it."""
    autofit(fig)
    post = getattr(fig, "_post_layout", None)
    if post is not None:
        post(fig)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / (name + ".png")
    fig.savefig(out, dpi=DPI, facecolor="white",
                bbox_inches="tight", pad_inches=SAVE_PAD_IN)
    plt.close(fig)
    _normalize_border(out, margin_px=10)
    label = PANEL_LABEL.get(name)
    stamp_box = _stamp_panel_label(out, label) if label else None
    with Image.open(out) as im:
        w_px, h_px = im.size
    margin = _margins(out, ignore=stamp_box)
    tail = ""
    if margin is not None:
        tail = "  margins T/B/L/R = %d %d %d %d" % margin
    if stamp_box is not None:
        tail += "  stamp=%r at %s" % (label, stamp_box)
    print("  wrote %s  (%d x %d px, %.1f KB)%s"
          % (_rel(out), w_px, h_px, out.stat().st_size / 1024.0, tail))
    print("  fonts: font.family=%r -> %s"
          % (mpl.rcParams["font.family"],
             Path(fm.findfont(fm.FontProperties(family="serif"))).name))


# Figure 1(a): held-out R^2 by model architecture and encoding
def fig1a():
    grid = _grid_path()
    if not grid.is_file():
        raise FigureInputMissing("missing %s" % _rel(grid))
    rows = list(csv.DictReader(open(grid, encoding="utf-8-sig")))
    val = {(r["dataset"], r["model"], r["encoding"]): float(r["r2_test"])
           for r in rows}
    datasets = sorted({r["dataset"] for r in rows})
    models = ["XGBoost", "Gradient Boosting", "Random Forest", "HistGB", "MLP"]
    encodings = ["integer", "one-hot", "native categorical"]
    colours = ["#4472C4", "#ED7D31", "#70AD47"]

    w_in, h_in = _canvas("fig1a")
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    bw = 0.26
    for j, enc in enumerate(encodings):
        xs, ys = [], []
        for i, model in enumerate(models):
            vals = [val[(d, model, enc)] for d in datasets if (d, model, enc) in val]
            if not vals:
                continue
            xs.append(i + (j - 1) * bw)
            ys.append(np.mean(vals))
        ax.bar(xs, ys, width=bw, color=colours[j], edgecolor="black",
               linewidth=0.4, label=enc.replace(" categorical", ""))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=FS, rotation=20, ha="right")
    ax.set_ylabel("Held-out R$^2$", fontsize=FL, labelpad=1.5)
    ax.set_ylim(0.35, 0.74)
    ax.tick_params(axis="both", labelsize=FS, direction="in")
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=FS, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              frameon=False, handlelength=1.2, columnspacing=1.0, borderpad=0.1)
    # The "(a)" is stamped after the tight crop (see PANEL_LABEL / _save).
    fig.subplots_adjust(top=0.80, left=0.185, right=0.985, bottom=0.30)
    return fig


# Figure 1(b): paired differences in held-out R^2, one marker per dataset
def fig1b():
    grid = _grid_path()
    if not grid.is_file():
        raise FigureInputMissing("missing %s" % _rel(grid))
    rows = list(csv.DictReader(open(grid, encoding="utf-8-sig")))
    val = {(r["dataset"], r["model"], r["encoding"]): float(r["r2_test"])
           for r in rows}
    datasets = sorted({r["dataset"] for r in rows})
    pairs = [
        ("MLP one-hot", "MLP", "one-hot", "MLP integer", "MLP", "integer"),
        ("MLP integer", "MLP", "integer", "XGBoost", "XGBoost", "integer"),
        ("Random Forest", "Random Forest", "integer", "XGBoost", "XGBoost", "integer"),
        ("GBoost", "Gradient Boosting", "integer", "XGBoost", "XGBoost", "integer"),
        ("HistGB", "HistGB", "integer", "XGBoost", "XGBoost", "integer"),
    ]

    w_in, h_in = _canvas("fig1b")
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ypos, labels = [], []
    for i, (an, am, ae, bn, bm, be) in enumerate(pairs):
        y = len(pairs) - 1 - i
        ypos.append(y)
        diffs = [val[(d, am, ae)] - val[(d, bm, be)] for d in datasets
                 if (d, am, ae) in val and (d, bm, be) in val]
        mu = float(np.mean(diffs))
        k = sum(1 for x in diffs if x > 0)
        for positive, marker, face in ((True, "o", "#4472C4"), (False, "v", "#C00000")):
            pts = [x for x in diffs if (x > 0) == positive]
            jit = [((j % 5) - 2) * 0.035 for j in range(len(pts))]
            ax.scatter(pts, [y + j for j in jit], s=13, marker=marker, facecolor=face,
                       edgecolor="black", linewidth=0.3, zorder=3)
        ax.plot([mu, mu], [y - 0.27, y + 0.27], color="black", lw=1.5, zorder=4)
        labels.append("%s vs %s\nmean %+.3f, %d/%d > 0"
                      % (an, bn, mu, k, len(diffs)))
    ax.axvline(0, color="black", ls="--", lw=0.8, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=FS)
    for tick in ax.get_yticklabels():
        tick.set_linespacing(1.3)
    ax.set_ylim(-0.45, len(pairs) - 0.55)
    ax.set_xlim(-0.62, 0.62)
    ax.set_xlabel("Difference in held-out R$^2$", fontsize=FL, labelpad=2)
    ax.tick_params(axis="x", labelsize=FS, direction="in")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.scatter([], [], s=13, marker="o", facecolor="#4472C4", edgecolor="black",
               linewidth=0.3, label="first better")
    ax.scatter([], [], s=13, marker="v", facecolor="#C00000", edgecolor="black",
               linewidth=0.3, label="second better")
    # Legend goes OUTSIDE the axes, in the empty band at the bottom-left of the
    # whole figure: inside the axes it covered the HistGB row's markers.
    _h, _l = ax.get_legend_handles_labels()
    fig.legend(_h, _l, loc="lower left", bbox_to_anchor=(0.006, 0.006),
               fontsize=FS * 0.92, frameon=True, framealpha=0.95,
               edgecolor="0.45", facecolor="white", fancybox=False,
               borderpad=0.30, handletextpad=0.40, labelspacing=0.25,
               ncol=1)
    # The "(b)" is stamped after the tight crop (see PANEL_LABEL / _save).
    fig.subplots_adjust(top=0.965, left=0.435, right=0.985, bottom=0.115)
    return fig


# Figure 2: held-out R^2 versus the noise ceiling, one point per dataset
def _short_label(dataset: str) -> str:
    import yaml

    mapping = yaml.safe_load(
        (DATA_DIR / "alias_mapping.yaml").read_text(encoding="utf-8"))
    alias_of = {v["source"]: k for k, v in mapping["datasets"].items()}
    alias = alias_of.get(dataset, dataset)
    return SHORT_LABEL.get(alias, alias)


def fig2():
    table = _ceiling_path()
    if not table.is_file():
        raise FigureInputMissing("missing %s" % _rel(table))
    ce = pd.read_csv(table)
    have = ce[ce["ceiling"].notna() & (ce["ceiling"] > 0)].copy()
    none = ce[~ce["ceiling"].notna() | (ce["ceiling"] <= 0)].copy()
    # Draw and label the triangles in ONE fixed order, otherwise the label
    # anchors and the drawn markers disagree.
    none = none.sort_values("r2_best", ascending=False).reset_index(drop=True)

    w_in, h_in = _canvas("fig2")
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.scatter(have["ceiling"], have["r2_best"], s=26, c=BLUE, edgecolor="k",
               linewidth=0.4, zorder=3,
               label="replicate groups ($n$ = %d)" % len(have))
    if len(none):
        # Only datasets with no repeated condition need the imputed triangle;
        # a de-duplicated ceiling table may legitimately contain none of them.
        ax.scatter([1.0] * len(none), none["r2_best"], s=26, c=RED, marker="^",
                   edgecolor="k", linewidth=0.4, zorder=3,
                   label="no replicates, imputed ($n$ = %d)" % len(none))
    diagonal = np.linspace(0.2, 1.02, 50)
    ax.plot(diagonal, diagonal, "--", color="0.55", lw=0.9, zorder=1,
            label="$R^2 = R^2_{max}$")

    x0, x1 = 0.40, 1.28
    y0, y1 = 0.38, 1.04
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Noise ceiling $R^2_{max}$", fontsize=FL, labelpad=1.5)
    ax.set_ylabel("Validation-selected held-out $R^2$ per dataset", fontsize=FL, labelpad=1.5)
    ax.tick_params(labelsize=FS, direction="in", length=2.2, pad=1.5)
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Validation-selected held-out $R^2$ vs. noise ceiling", fontsize=FL, pad=3)

    # The legend is placed first, then treated as an obstacle for every label.
    leg = ax.legend(fontsize=FS, loc="upper left", framealpha=0.95, borderpad=0.28,
                    labelspacing=0.22, handletextpad=0.35, borderaxespad=0.25)
    left, right, top, bot = 0.155, 0.980, 0.880, 0.145
    fig.subplots_adjust(top=top, left=left, right=right, bottom=bot)

    # The imputed points sit at x = 1.0 (no jitter): the marker that scatter()
    # draws and the point the label is anchored to must be the same point,
    # otherwise the label starts its constant gap from empty space.
    pts = ([(r["ceiling"], r["r2_best"], _short_label(r["dataset"]), "o")
            for _, r in have.iterrows()]
           + [(1.0, r["r2_best"], _short_label(r["dataset"]), "^")
              for _, r in none.iterrows()])
    pts = sorted(pts, key=lambda p: p[0])
    points = [(x, y, t) for x, y, t, _s in pts]
    markers = [(x, y, s) for x, y, _t, s in pts]
    # Placement runs after autofit() (i.e. against the axes geometry that is
    # actually rendered) and is told to fail loudly rather than return an
    # overlapping layout.
    def _post(f):
        _place_dataset_labels(f, ax, points, markers, leg, (x0, x1, y0, y1))

    fig._post_layout = _post
    return fig


def _place_dataset_labels(fig, ax, points, markers, legend, limits) -> None:
    """Figure-2 labels: one constant *ink* gap, angle fan-out, leaders last.

    Every label is placed so that the shortest distance between its rendered
    ink and its own marker's ink is exactly ``GAP`` points, whatever the angle,
    the label length or the marker shape.  The earlier version measured the gap
    from the marker CENTRE and then subtracted a constant "side bearing"; that
    is why the two ``^`` labels (whose ink is not a circle) and the long labels
    ended up visibly farther out than the circle labels.

    ``points`` are ``(x, y, text)``; ``markers`` are ``(x, y, shape)`` with the
    marker ``scatter`` actually drew (``"o"`` or ``"^"``).  Both sides of the
    gap are real ink: the marker's path with its stroke, and the hull of the
    label's rasterised glyph pixels (a text box's near corner is usually empty,
    so a box-based gap is too big at every diagonal angle).  Only the angle
    varies -- 0 deg (straight to the right of the marker) is preferred, then
    the shallower fan.  A leader line is the last resort.
    """
    import matplotlib.markers as mmarkers

    x0, x1, y0, y1 = (float(v) for v in limits)
    # Measure at the dpi the PNG is written at: a 0.2 pt side bearing is 0.8 px
    # at 300 dpi but 0.3 px at the figure's default 100 dpi.
    if abs(float(fig.dpi) - DPI) > 1e-6:
        fig.set_dpi(DPI)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ppp = float(fig.dpi) / 72.0          # pixels per point
    ab = ax.get_window_extent(r)

    GAP, GAP_STEP = 4.2, 6.5             # pt: the constant gap, one allowed step
    LEAD_GAPS = (9.0, 18.0)              # pt: only when the crowd forbids it
    LABEL_PAD = 2.0                      # px kept between two label ink boxes
    MARKER_PAD = 4.0                     # px kept between a label and any marker
    FRAME_PAD = 1.5                      # px kept inside the axes frame
    M_SIZE, M_LW = 26.0, 0.4             # what fig2() hands to scatter()
    # 0 deg first, then the fan in steps of 15 deg (the author's list of
    # candidates -- 0, +-30, +-60, +-90, +-120, +-150, 180 -- is a subset; the
    # half-steps only exist so that a label a few px too wide for 0 deg can
    # lean by 15 deg instead of jumping to 30).
    ANGLES = [a for d in range(0, 181, 15) for a in ((0.0,) if d == 0 else (d, -d))]

    def _cost(a):
        d = abs(a)
        if d <= 15.0:
            return 0.0
        if d <= 60.0:
            return 1.0
        if d <= 110.0:
            return 3.0
        return 8.0

    # scatter() scales a marker's unit path (half-extent 0.5) by sqrt(s) marker
    # points; to_polygons() flattens the Bezier circle to its real outline.
    def _marker_ink(shape):
        ms = mmarkers.MarkerStyle(shape)
        polys = ms.get_path().transformed(ms.get_transform()).to_polygons()
        poly = np.asarray(max(polys, key=len), float)
        return poly * math.sqrt(M_SIZE) * ppp, 0.5 * M_LW * ppp

    inks = {}
    for m in markers:
        inks.setdefault(m[2] if len(m) > 2 else "o", None)
    for shape in list(inks):
        inks[shape] = _marker_ink(shape)
    mk = []
    for m in markers:
        shape = m[2] if len(m) > 2 else "o"
        poly, hw = inks[shape]
        c = np.asarray(ax.transData.transform((float(m[0]), float(m[1]))), float)
        mk.append((float(c[0]), float(c[1]), poly + c, hw))

    probe = [(ax.annotate(t, (x, y), xytext=(0.0, 0.0), textcoords="offset points",
                          fontsize=FS, ha="left", va="bottom", color="0.12"), t)
             for x, y, t in points]
    fig.canvas.draw()
    lbox = {}
    for a, t in probe:
        we = a.get_window_extent(r)
        lbox[t] = (float(we.width), float(we.height))
        a.remove()
    fig.canvas.draw()

    # get_window_extent() is the layout box (advance widths plus ascent and
    # descent), not the ink, so each string is rasterised once: the insets give
    # the real ink box and the convex hull is what the gap is measured to.
    def _hull(pts):
        p = np.unique(np.round(pts, 3), axis=0)
        if len(p) < 3:
            return p
        p = p[np.lexsort((p[:, 1], p[:, 0]))]

        def half(seq):
            out = []
            for q in seq:
                while len(out) >= 2:
                    o, b = out[-2], out[-1]
                    if ((b[0] - o[0]) * (q[1] - o[1])
                            - (b[1] - o[1]) * (q[0] - o[0])) <= 0.0:
                        out.pop()
                    else:
                        break
                out.append(q)
            return out

        lower = half(p)
        upper = half(p[::-1])
        return np.array(lower[:-1] + upper[:-1], float)

    def _ink_shapes(strings):
        out = {}
        sfig = plt.figure(figsize=(1.9, 0.6), dpi=float(fig.dpi))
        try:
            for s in strings:
                sfig.clf()
                sax = sfig.add_axes([0.0, 0.0, 1.0, 1.0])
                sax.set_axis_off()
                st = sax.text(0.08, 0.30, s, fontsize=FS, color="black",
                              ha="left", va="bottom", transform=sax.transAxes)
                sfig.canvas.draw()
                sr = sfig.canvas.get_renderer()
                we = st.get_window_extent(sr)
                buf = np.asarray(sfig.canvas.buffer_rgba())[:, :, :3]
                ys, xs = np.nonzero((buf < 250).any(axis=2))
                if not len(xs):
                    out[s] = ((0.0, 0.0, 0.0, 0.0), np.zeros((4, 2)))
                    continue
                h = buf.shape[0]
                ins = (float(xs.min() - we.x0),                # left
                       float(we.x1 - (xs.max() + 1)),          # right
                       float((h - (ys.max() + 1)) - we.y0),    # bottom
                       float(we.y1 - (h - ys.min())))          # top
                pts = np.column_stack([xs + 0.5 - we.x0,
                                       (h - ys - 0.5) - we.y0])
                out[s] = (ins, _hull(pts))
        finally:
            plt.close(sfig)
        return out

    shapes = _ink_shapes([t for _x, _y, t in points])

    def _pt_poly(pts, poly):
        """Distance (and the closest polygon point) from every point in ``pts``."""
        a = poly
        b = np.roll(poly, -1, axis=0)
        px, py = pts[:, 0][:, None], pts[:, 1][:, None]
        dx, dy = (b[:, 0] - a[:, 0])[None, :], (b[:, 1] - a[:, 1])[None, :]
        t = np.clip(((px - a[None, :, 0]) * dx + (py - a[None, :, 1]) * dy)
                    / np.maximum(dx * dx + dy * dy, 1e-12), 0.0, 1.0)
        qx = a[None, :, 0] + t * dx
        qy = a[None, :, 1] + t * dy
        d = np.hypot(qx - px, qy - py)
        k = np.argmin(d, axis=1)
        rows = np.arange(len(pts))
        return d[rows, k], np.column_stack([qx[rows, k], qy[rows, k]])

    def _convex_overlap(a, b):
        for pts in (a, b):
            e = np.roll(pts, -1, axis=0) - pts
            n = np.column_stack([-e[:, 1], e[:, 0]])
            pa, pb = n @ a.T, n @ b.T
            if np.any(pa.max(axis=1) < pb.min(axis=1)):
                return False
            if np.any(pb.max(axis=1) < pa.min(axis=1)):
                return False
        return True

    def _closest(hh, poly, hw):
        """Ink gap, plus the nearest point on each ink (label, marker)."""
        d1, q1 = _pt_poly(hh, poly)          # label vertices -> marker edges
        d2, q2 = _pt_poly(poly, hh)          # marker vertices -> label edges
        i1, i2 = int(np.argmin(d1)), int(np.argmin(d2))
        if d1[i1] <= d2[i2]:
            return float(d1[i1]) - hw, hh[i1], q1[i1]
        return float(d2[i2]) - hw, q2[i2], poly[i2]

    def _ink_gap(hh, poly, hw):
        """Shortest distance (px) between the label's ink and a marker's ink."""
        if _convex_overlap(hh, poly):
            return 0.0
        return max(0.0, _closest(hh, poly, hw)[0])

    def _align(ang):
        ux, uy = math.cos(ang), math.sin(ang)
        ha = "left" if ux > 1e-9 else "right" if ux < -1e-9 else "center"
        va = "bottom" if uy > 1e-9 else "top" if uy < -1e-9 else "center"
        return ux, uy, ha, va

    def _hull_of(t, ha, va):
        """The label's ink hull relative to the annotate anchor."""
        w, h = lbox[t]
        dx = 0.0 if ha == "left" else -w if ha == "right" else -0.5 * w
        dy = 0.0 if va == "bottom" else -h if va == "top" else -0.5 * h
        return shapes[t][1] + np.array([dx, dy])

    def _place(i, ang, gap, hh):
        """Anchor point for ink hull ``hh`` at exactly ``gap`` pt from marker i.

        The hull's near edge is put on the ray through the marker centre.  The
        first step uses the hull's bounding box, whose near corner is usually
        empty for real glyphs; the second step slides the label back along the
        ray until the hull's own leading support is exactly ``gap`` away.  Both
        sides are convex, so no point of the label can then be nearer than that.
        """
        cx, cy, poly, hw = mk[i]
        ux, uy, ha, va = _align(ang)
        u = np.array([ux, uy])
        e = poly[int(np.argmax(poly @ u))] + hw * u    # marker ink, leading edge
        lo, hi = hh.min(axis=0), hh.max(axis=0)
        near = np.array([lo[0] if ux > 1e-9 else hi[0] if ux < -1e-9 else 0.5 * (lo[0] + hi[0]),
                         lo[1] if uy > 1e-9 else hi[1] if uy < -1e-9 else 0.5 * (lo[1] + hi[1])])
        a = e + gap * ppp * u - near
        a = a + ((float(e @ u) + gap * ppp) - (float(a @ u) + float(np.min(hh @ u)))) * u
        return a, (a[0] + lo[0], a[0] + hi[0], a[1] + lo[1], a[1] + hi[1])

    def _refine(i, gap, hh, anc):
        """Slide the label along the true closest-pair direction onto ``gap``.

        ``_place`` fixes the gap measured along the ray; at shallow angles the
        closest pair can sit slightly off that ray (a label's leading corner
        against the middle of a round marker), which would leave the visible
        gap a few tenths of a point wide.  A few Newton steps on the actual
        closest pair remove the residual.
        """
        poly, hw = mk[i][2], mk[i][3]
        target = gap * ppp
        for _ in range(4):
            d, hp, mp = _closest(hh + anc, poly, hw)
            step = d - target
            if step <= 0.02:
                break
            v = hp - mp
            n = float(np.hypot(v[0], v[1]))
            if n < 1e-6:
                break
            anc = anc - v / n * step
        return anc, (anc[0] + hh[:, 0].min(), anc[0] + hh[:, 0].max(),
                     anc[1] + hh[:, 1].min(), anc[1] + hh[:, 1].max())

    def _pt_rect(px, py, bb):
        """Distance from a point to a box (0 inside)."""
        return float(np.hypot(max(bb[0] - px, 0.0, px - bb[1]),
                              max(bb[2] - py, 0.0, py - bb[3])))

    def _ov(a, b, pad=0.0):
        return (min(a[1], b[1]) - max(a[0], b[0]) > -pad
                and min(a[3], b[3]) - max(a[2], b[2]) > -pad)

    # A single greedy pass is order-dependent: the first label takes the best
    # spot and can push a neighbour round the back of the cluster.  Several
    # fixed orders are planned and the cheapest plan wins, so the result stays
    # deterministic but is no longer an artefact of one sweep direction.
    legend_box = legend.get_window_extent(r)
    legend_obs = (legend_box.x0 - 1.0, legend_box.x1 + 1.0,
                  legend_box.y0 - 1.0, legend_box.y1 + 1.0)
    STAGES = ((GAP, True), (GAP, False),
              (GAP_STEP, True), (GAP_STEP, False),
              (LEAD_GAPS[0], False), (LEAD_GAPS[1], False))

    def _eval(i, hh, ibb, placed):
        """Frame / label / marker clearance, plus the two ownership distances."""
        if (ibb[0] < ab.x0 + FRAME_PAD or ibb[1] > ab.x1 - FRAME_PAD
                or ibb[2] < ab.y0 + FRAME_PAD or ibb[3] > ab.y1 - FRAME_PAD):
            return False, 0.0, 0.0
        if any(_ov(ibb, q, LABEL_PAD) for q in placed):
            return False, 0.0, 0.0
        d_own = _pt_rect(mk[i][0], mk[i][1], ibb)
        d_oth, ink_oth = 1e9, 1e9
        for j, (mx, my, poly, hw) in enumerate(mk):
            near = _pt_rect(mx, my, ibb)
            if near > 34.0:                       # far: its ink cannot reach
                if j != i:
                    d_oth = min(d_oth, near)
                continue
            g = _ink_gap(hh, poly, hw)
            if g < MARKER_PAD:
                return False, 0.0, 0.0
            if j != i:
                d_oth = min(d_oth, near)
                ink_oth = min(ink_oth, g)
        return True, d_oth - d_own, ink_oth

    def _margin(ibb, i):
        """How much nearer the own marker centre is than any other one, in px."""
        d_own = _pt_rect(mk[i][0], mk[i][1], ibb)
        d_oth = min([_pt_rect(mx, my, ibb)
                     for j, (mx, my, _p, _h) in enumerate(mk) if j != i] or [1e9])
        return d_oth - d_own

    def _plan(order):
        placed = [legend_obs]
        out = []
        for i in order:
            t = points[i][2]
            best = None
            for gap, need_near in STAGES:
                cands = []
                for a in ANGLES:
                    rad = math.radians(a)
                    ux, uy, ha, va = _align(rad)
                    hh = _hull_of(t, ha, va)
                    anc, ibb = _place(i, rad, gap, hh)
                    ok, mar, ink_oth = _eval(i, hh + anc, ibb, placed)
                    if ok:
                        anc2, ibb2 = _refine(i, gap, hh, anc)
                        ok2, mar2, ink_oth2 = _eval(i, hh + anc2, ibb2, placed)
                        if ok2:
                            anc, ibb, mar, ink_oth = anc2, ibb2, mar2, ink_oth2
                    if not ok:
                        continue
                    # Own marker must own the label: nearest centre by 2 px and
                    # nearer ink than any other marker (the centre test alone
                    # lets a long label slide under a neighbour's marker).
                    if need_near and (mar < 2.0 or ink_oth < gap * ppp + 2.0):
                        continue
                    # cheapest angle first, then the shallower one, then the
                    # least ambiguous ownership
                    cands.append(((_cost(a), abs(a), -mar), anc, ibb, a, ha, va,
                                  ink_oth))
                if cands:
                    _k, anc, ibb, a, ha, va, ink_oth = min(cands, key=lambda c: c[0])
                    best = (anc, ibb, a, ha, va, gap, gap > GAP_STEP, ink_oth)
                    break
            if best is None:                   # keep the figure readable
                anc, ibb = _place(i, 0.0, GAP, _hull_of(t, "left", "center"))
                best = (anc, ibb, 0.0, "left", "center", GAP, True, 0.0)
            anc, ibb, a, ha, va, gap, lead, ink_oth = best
            placed.append(ibb)
            out.append({"i": i, "label": t, "angle_deg": a, "gap_pt": gap,
                        "leader": bool(lead), "margin_px": _margin(ibb, i),
                        "other_ink_px": ink_oth,
                        "anchor": anc, "box": ibb, "ha": ha, "va": va})
        return out

    def _score(plan):
        return (sum(q["gap_pt"] > GAP for q in plan),
                sum(q["leader"] for q in plan),
                sum(_cost(q["angle_deg"]) for q in plan),
                -sum(min(q["margin_px"], 40.0) for q in plan),
                sum(q["radius_pt"] for q in plan))

    crowd = [sum(1 for j in range(len(points)) if j != i
                 and abs(points[j][0] - points[i][0]) < 0.10
                 and abs(points[j][1] - points[i][1]) < 0.10)
             for i in range(len(points))]
    idx = list(range(len(points)))
    orders = [
        sorted(idx, key=lambda j: (points[j][0], -points[j][1])),
        sorted(idx, key=lambda j: (-points[j][0], -points[j][1])),
        sorted(idx, key=lambda j: (points[j][1], -points[j][0])),
        sorted(idx, key=lambda j: (-points[j][1], -points[j][0])),
        sorted(idx, key=lambda j: (-crowd[j], points[j][0])),
        sorted(idx, key=lambda j: (-crowd[j], -points[j][1])),
    ]
    plans = []
    for order in orders:
        pl = _plan(order)
        for q in pl:
            q["radius_pt"] = float(np.hypot(*(q["anchor"] - np.array(mk[q["i"]][:2])))) / ppp
        plans.append(pl)
    plan = min(plans, key=_score)

    report, drawn = [], []
    for q in plan:
        i, t = q["i"], q["label"]
        x, y, _t = points[i]
        cx, cy = mk[i][0], mk[i][1]
        anc, ha, va = q["anchor"], q["ha"], q["va"]
        ann = ax.annotate(t, (x, y),
                          xytext=((anc[0] - cx) / ppp, (anc[1] - cy) / ppp),
                          textcoords="offset points", fontsize=FS,
                          ha=ha, va=va, color="0.12", zorder=6,
                          arrowprops=(dict(arrowstyle="-", lw=0.35, color="0.5",
                                           shrinkA=1.0, shrinkB=3.0)
                                      if q["leader"] else None))
        drawn.append((ann, q, ha, va))
        report.append({k: v for k, v in q.items()
                       if k not in ("box", "anchor", "i", "ha", "va")})

    fig.canvas.draw()
    seen = []
    for (ann, q, ha, va), rep in zip(drawn, report):
        i = q["i"]
        we = ann.get_window_extent(r)          # the box that was really drawn
        anc = np.array([we.x0 + (0.0 if ha == "left" else we.width if ha == "right"
                                 else 0.5 * we.width),
                        we.y0 + (0.0 if va == "bottom" else we.height if va == "top"
                                 else 0.5 * we.height)])
        hh = _hull_of(q["label"], ha, va) + anc
        bb = (hh[:, 0].min(), hh[:, 0].max(), hh[:, 1].min(), hh[:, 1].max())
        cx, cy, poly, hw = mk[i]
        rep["gap_measured_pt"] = _ink_gap(hh, poly, hw) / ppp
        rep["nearest_own_px"] = _pt_rect(cx, cy, bb)
        rep["nearest_other_px"] = min(
            [_pt_rect(mx, my, bb)
             for j, (mx, my, _p, _h) in enumerate(mk) if j != i] or [1e9])
        seen.append(rep["gap_measured_pt"])
        if (abs(rep["gap_measured_pt"] - rep["gap_pt"]) > 0.2
                or rep["nearest_other_px"] < rep["nearest_own_px"]):
            print("  fig2 WARNING: %r gap %.2f pt (target %.2f), nearest "
                  "other/own marker %.1f/%.1f px"
                  % (rep["label"], rep["gap_measured_pt"], rep["gap_pt"],
                     rep["nearest_other_px"], rep["nearest_own_px"]))

    fig._label_report = {"labels": report}
    far = [q for q in report if q["leader"]]
    step = [q for q in report if not q["leader"] and q["gap_pt"] > GAP]
    print("  fig2: %d labels, %d at the %g pt ink gap, %d stepped to %g pt, "
          "%d with leader lines"
          % (len(report), len(report) - len(far) - len(step), GAP, len(step),
             GAP_STEP, len(far)))
    print("  fig2 gap measured on the canvas: %.2f .. %.2f pt (target %g)"
          % (min(seen), max(seen), GAP))
    print("  fig2 labels: " + " | ".join(
        "%s %ddeg %.1fpt" % (q["label"], round(q["angle_deg"]), q["gap_pt"])
        for q in report))

def _xgb(**extra):
    """XGBoost configuration used for the native-categorical attribution."""
    return xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, **extra)


def _load_bh():
    """Load and featurise the case-study dataset (``case_study: true``) once, cached."""
    if "bh" not in _CACHE:
        path = _dataset_path(_case_study())
        data_dir = path.parent
        # The case-study figures must use the same loader/feature builder as the
        # rest of the package.  Going through the monolith (NS) dropped the two
        # reactant slots and left 8 columns, so the Figure 4 panels could not find
        # `reactant_1_encoded` at all.
        from hte.v16_core import (get_split_positions as _gsp,
                                  load_real_dataset as _lrd,
                                  prepare_features as _pf)
        df, raw = _lrd(str(path), str(data_dir))
        tr_pos, te_pos = _gsp(len(df))
        X, y, feature_cols, _disp, _enc, _sc = _pf(df, raw, fit_rows=tr_pos)
        cat_cols = [str(c) for c in feature_cols if str(c).endswith("_encoded")]
        Xn = X.copy()
        for col in cat_cols:
            Xn[col] = Xn[col].astype("category")
        _CACHE["bh"] = {
            "df": df,
            "raw": raw,
            "tr_pos": tr_pos,
            "te_pos": te_pos,
            "X": Xn,
            "y": y,
            "train": Xn.iloc[tr_pos].copy(),
            "test": Xn.iloc[te_pos].copy(),
        }
    return _CACHE["bh"]


_CACHE: dict = {}


def _shap():
    """Native-categorical XGBoost + SHAP contributions (cached)."""
    if "shap" not in _CACHE:
        state = _load_bh()
        model = _xgb(enable_categorical=True, tree_method="hist", max_cat_to_onehot=4)
        model.fit(state["train"], state["y"].iloc[state["tr_pos"]])
        contribs = np.asarray(model.get_booster().predict(
            xgb.DMatrix(state["test"], enable_categorical=True),
            pred_contribs=True))[:, :-1]
        cols = [str(c) for c in state["train"].columns]
        mean_abs = np.abs(contribs).mean(0)
        total = mean_abs.sum()
        _CACHE["shap"] = {
            "values": contribs,
            "cols": cols,
            "pct": contribs / total * 100.0,
            "share": mean_abs / total * 100.0,
        }
    return _CACHE["shap"]


# Figure 3(a): SHAP value distribution per feature
SHAP_LABELS = {
    "reactant_1_encoded": "Reactant 1",
    "reactant_2_encoded": "Reactant 2",
    "substrate_type_encoded": "Reactant 1",
    "leaving_group_encoded": "Reactant 2",
    "base_encoded": "Base",
    "catalyst_encoded": "Catalyst",
    "ligand_encoded": "Ligand",
    "solvent_encoded": "Solvent",
    "metal_encoded": "Metal centre",
    "temperature": "Temperature",
    "time": "Time",
    "catalyst_loading": "Catalyst loading",
}


def _bees(values, seed):
    """Deterministic beeswarm offsets (ported from the paper's figure script)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    quant = np.round(100 * (values - values.min())
                     / (values.max() - values.min() + 1e-8))
    order = np.argsort(quant + rng.standard_normal(n) * 1e-6)
    layer, last, offsets = 0, -1, np.zeros(n)
    for idx in order:
        if quant[idx] != last:
            layer = 0
        offsets[idx] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last = quant[idx]
    return offsets * 0.9 * (0.4 / max(np.ptp(offsets) + 1, 1e-9))


def fig3a():
    shap = _shap()
    state = _load_bh()
    pct, share, cols = shap["pct"], shap["share"], shap["cols"]
    live = sorted([i for i in range(len(cols)) if share[i] > 1e-9],
                  key=lambda i: share[i])

    w_in, h_in = _canvas("fig3a")
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    cmap = plt.get_cmap("coolwarm")
    n = len(live)
    for row, j in enumerate(live):
        values = pct[:, j].copy()
        feature = state["test"].iloc[:, j].astype(float).to_numpy().copy()
        rng = np.random.default_rng(1000 + j)
        perm = rng.permutation(len(values))
        values, feature = values[perm], feature[perm]
        vmin, vmax = np.nanpercentile(feature, 5), np.nanpercentile(feature, 95)
        if vmin == vmax:
            vmin, vmax = float(np.nanmin(feature)), float(np.nanmax(feature))
        if vmin == vmax:
            vmax = vmin + 1
        ax.scatter(values, row + _bees(values, j), c=np.clip(feature, vmin, vmax),
                   cmap=cmap, vmin=vmin, vmax=vmax, s=7, linewidths=0, zorder=3)
        ax.text(1.015, row, "%.1f%%" % share[j], transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=FS, color="0.12")
    ax.axvline(0, color="#333333", lw=0.8, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([SHAP_LABELS.get(cols[j], cols[j]) for j in live], fontsize=FS)
    ax.set_ylim(-0.9, n - 0.1)
    ax.set_xlim(pct.min() * 1.06, pct.max() * 1.03)
    ax.set_xlabel("SHAP value  (% of total attribution)", fontsize=FL, labelpad=2)
    ax.tick_params(axis="x", labelsize=FS, length=2.2, pad=1.5)
    ax.tick_params(axis="y", labelsize=FS, length=2.6, direction="out", pad=2)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    cax = fig.add_axes([0.945, 0.31, 0.016, 0.40])
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([])
    cax.text(0.5, 1.045, "High", transform=cax.transAxes, ha="center", va="bottom",
             fontsize=FS)
    cax.text(0.5, -0.075, "Low", transform=cax.transAxes, ha="center", va="top",
             fontsize=FS)
    cax.text(2.6, 0.5, "Feature value", transform=cax.transAxes, ha="center",
             va="center", fontsize=FL, rotation=90)
    fig.subplots_adjust(left=0.245, right=0.835, top=0.975, bottom=0.125)
    return fig


# Figure 3(b): signed SHAP contribution of the leaving-group slot
def fig3b():
    shap = _shap()
    state = _load_bh()
    cols, values = shap["cols"], shap["values"]
    j_lg = cols.index("reactant_1_encoded")
    df, te_pos = state["df"], state["te_pos"]
    if "startingmat_1_name" in df.columns:
        raw_lg = df["startingmat_1_name"].astype(str).to_numpy()[te_pos]
    else:
        raw_lg = state["test"].iloc[:, j_lg].astype(str).to_numpy()
    groups: dict[str, list[float]] = {}
    for i, name in enumerate(raw_lg):
        groups.setdefault(name, []).append(values[i, j_lg])
    ranked = sorted(groups, key=lambda k: -np.mean(np.abs(groups[k])))[:12]

    w_in, h_in = _canvas("fig3b")
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.boxplot([groups[k] for k in ranked], widths=0.62, patch_artist=False,
               showfliers=False, medianprops=dict(color="#e8a33d", lw=1.2),
               boxprops=dict(color="black", lw=0.7),
               whiskerprops=dict(color="black", lw=0.7),
               capprops=dict(color="black", lw=0.7))
    ax.axhline(0, ls="--", lw=0.8, color="0.45")
    ax.set_ylabel("SHAP value (yield %)", fontsize=FL, labelpad=2)
    ax.set_xticks(range(1, len(ranked) + 1))
    ax.set_xticklabels([s.strip()[:14] for s in ranked], rotation=38, ha="right",
                       fontsize=FS)
    ax.tick_params(axis="x", length=0, pad=1.5, labelsize=FS)
    ax.tick_params(axis="y", labelsize=FS, length=2.2, pad=1.5)
    ax.grid(axis="y", alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.155, right=0.99, top=0.985, bottom=0.40)
    return fig


# Figure 4: measured mean yield for ligand x base combinations
def fig4():
    state = _load_bh()
    df = state["df"]
    needed = ["catalyst_name", "reagent_1_name", "yield"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise FigureInputMissing(
            "%s is missing column(s) %s needed for the ligand x base "
            "heatmap." % (_case_study(), ", ".join(missing)))
    data = df[needed].dropna()
    counts = data.pivot_table(index="catalyst_name", columns="reagent_1_name",
                              values="yield", aggfunc="size").fillna(0)
    keep_l = counts.sum(axis=1).sort_values(ascending=False).head(14).index
    keep_b = counts.sum(axis=0).sort_values(ascending=False).head(10).index
    matrix = data.pivot_table(index="catalyst_name", columns="reagent_1_name",
                              values="yield", aggfunc="mean").reindex(
                                  index=keep_l, columns=keep_b)

    w_in, h_in = _canvas("fig4")
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=70)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right", fontsize=FS)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=FS)
    ax.set_xlabel("Base", fontsize=FL, labelpad=1.5)
    ax.set_ylabel("Ligand", fontsize=FL, labelpad=1.5)
    ax.tick_params(labelsize=FS, length=2.2, pad=1.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            if np.isfinite(value) and value >= 45:
                ax.text(j, i, "%.0f" % value, ha="center", va="center", fontsize=FS,
                        color="w", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("Mean yield (%)", fontsize=FL, labelpad=2)
    cbar.ax.tick_params(labelsize=FS, length=2, pad=1.5)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", lw=0.7)
    ax.tick_params(which="minor", length=0)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.98, bottom=0.20)
    return fig


# Figure 5: held-out R^2 under the four validation protocols
def _protocol_cv() -> tuple[dict, Path]:
    """Mean held-out R^2 per in-dataset protocol, from results/protocol_cv.csv.

    The table is written by ``outputs/last_gaps/20_protocol_cv.py``: 5-fold
    ``KFold(5, shuffle=True, random_state=42)`` for the random split,
    ``GroupKFold(5)`` on ``substrate_type_encoded`` for the substrate-disjoint
    split, ``KFold(5, shuffle=False)`` for the ordered split, all with XGBoost
    on AI_RCP_BH and integer encoding.  No literal fallback: if the table is
    absent or incomplete this raises.
    """
    if not PROTOCOL_CV_CSV.is_file():
        raise FigureInputMissing(
            "Figure 5 needs the recomputed protocol table:\n  %s\n"
            "It is produced by outputs/last_gaps/20_protocol_cv.py.  The three "
            "in-dataset bars are no longer hardcoded, so the figure cannot be "
            "drawn without it." % _rel(PROTOCOL_CV_CSV))
    table = pd.read_csv(PROTOCOL_CV_CSV)
    values = {str(row.protocol): float(row.mean_r2)
              for row in table.itertuples(index=False)}
    missing = [key for _, key in PROTOCOLS if key not in values]
    if missing:
        raise FigureInputMissing(
            "Figure 5: %s has no row(s) for %s (found %s)."
            % (_rel(PROTOCOL_CV_CSV), ", ".join(missing),
               ", ".join(sorted(values)) or "nothing"))
    return values, PROTOCOL_CV_CSV


def _transfer_mean() -> tuple[float, int, Path]:
    """MEAN cross-dataset transfer R^2 over the targets in results/transfer_check.csv.

    The paper text quotes the mean, not the median.  (The previous bar plotted
    the median by mistake -- -1.0699 from the shipped table, -1.171 from the
    paper's rounded value; both are gone.)  Missing table -> hard error.
    """
    if not TRANSFER_CSV.is_file():
        raise FigureInputMissing(
            "Figure 5 needs the cross-dataset transfer table:\n  %s\n"
            "It carries one r2_transfer per target dataset; the bar plots their "
            "mean." % _rel(TRANSFER_CSV))
    table = pd.read_csv(TRANSFER_CSV)
    values = pd.to_numeric(table["r2_transfer"], errors="coerce").dropna()
    if values.empty:
        raise FigureInputMissing("Figure 5: %s has no usable r2_transfer rows."
                                 % _rel(TRANSFER_CSV))
    return float(values.mean()), int(len(values)), TRANSFER_CSV


def fig5():
    cv, cv_path = _protocol_cv()
    transfer, n_transfer, transfer_path = _transfer_mean()
    labels = [label for label, _ in PROTOCOLS] + ["Cross-dataset transfer\n(new project)"]
    values = [cv[key] for _, key in PROTOCOLS] + [transfer]
    print("  protocols: %s" % ", ".join("%+.4f" % v for v in values))
    print("  in-dataset bars: %s (mean 5-fold R^2)" % _rel(cv_path))
    print("  cross-dataset bar: %s (MEAN over %d targets, %.6f; median not used)"
          % (_rel(transfer_path), n_transfer, transfer))

    w_in, h_in = _canvas("fig5")
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    colours = [BLUE if v > 0 else RED for v in values]
    bars = ax.bar(labels, values, color=colours, edgecolor="k", linewidth=0.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + (0.05 if value > 0 else -0.17),
                "%+.3f" % value, ha="center", fontsize=FS)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_ylabel("Held-out $R^2$", fontsize=FL, labelpad=2)
    ax.set_ylim(-1.60, 1.10)
    ax.tick_params(axis="y", labelsize=FS, length=2.2, pad=1.5)
    ax.tick_params(axis="x", labelsize=FS, length=0, pad=2)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("Held-out $R^2$ under four validation protocols", fontsize=FL, pad=3)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.905, bottom=0.235)
    return fig


# Backup panel (not in the manuscript): the loading sweep
BO_TABLE_CANDIDATES = [
    RES_DIR / "bo_panel_data.csv",
    RES_DIR / "Figure9_Bayesian_Optimization-Ensemble_panel_data.csv",
]


def _bo_table() -> pd.DataFrame:
    for candidate in BO_TABLE_CANDIDATES:
        if candidate.is_file():
            return pd.read_csv(candidate)
    raise FigureInputMissing(
        "the Bayesian-optimisation panel table is not available.  Looked for:\n"
        "  %s\n"
        "  %s\n"
        "That table is produced by the expensive virtual-library / "
        "Bayesian-optimisation path of the frozen pipeline "
        "(build_virtual_library, stream_virtual_library, "
        "generate_bo_results_parallel, generate_bo_results_two_stage, "
        "generate_bo_results_streaming); this script deliberately does not run "
        "it.  Generate the table with the pipeline, copy it into results/, then "
        "re-run with --allow-heavy."
        % (_rel(BO_TABLE_CANDIDATES[0]), _rel(BO_TABLE_CANDIDATES[1]))
    )


def fig6a():
    table = _bo_table()
    w_in, h_in = _canvas("fig6a")
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    panel = table[table["panel"] == "a"].copy()
    panel["x"] = pd.to_numeric(panel["x"], errors="coerce")
    panel["y"] = pd.to_numeric(panel["y"], errors="coerce")
    series = [("acquisition order (EI)", BLUE, "-", 1.4, "EI acquisition"),
              ("random order", "0.35", ":", 1.2, "random order"),
              ("predicted-yield order", RED, "--", 1.2, "predicted-yield order")]
    for key, colour, style, width, label in series:
        group = panel[panel["series"] == key].sort_values("x")
        if len(group):
            ax.plot(group["x"], group["y"], color=colour, ls=style, lw=width,
                    label=label)
    ax.set_xlabel("Number of candidates taken", fontsize=FL, labelpad=1.5)
    ax.set_ylabel("Cumulative best predicted yield (%)", fontsize=FL, labelpad=1.5)
    ax.tick_params(labelsize=FS, length=2.2, pad=1.5)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=FS, loc="lower right", borderpad=0.3, labelspacing=0.25,
              handletextpad=0.4, borderaxespad=0.3)
    ax.set_axisbelow(True)
    # The "(a)" is stamped after the tight crop (see PANEL_LABEL / _save).
    fig.subplots_adjust(left=0.145, right=0.985, top=0.965, bottom=0.135)
    return fig


def fig6b():
    table = _bo_table()
    w_in, h_in = _canvas("fig8b_backup_loading")
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    panel = table[table["panel"] == "d"].copy()
    panel["x"] = pd.to_numeric(panel["x"], errors="coerce")
    panel["y"] = pd.to_numeric(panel["y"], errors="coerce")
    mean = panel[panel["series"] == "ensemble mean (weighted)"].sort_values("x")
    upper = panel[panel["series"] == "ensemble mean + 1 SD"].sort_values("x")
    lower = panel[panel["series"] == "ensemble mean - 1 SD"].sort_values("x")
    if len(upper) and len(lower):
        ax.fill_between(upper["x"], lower["y"], upper["y"], color="#AFC7E8",
                        alpha=0.6, label="$\\pm$1 SD", zorder=1)
    ax.plot(mean["x"], mean["y"], color=BLUE, lw=1.4, label="ensemble mean", zorder=3)
    best = mean["y"].idxmax()
    ax.plot([mean.loc[best, "x"]], [mean.loc[best, "y"]], "o", ms=5, mfc="#D94F9B",
            mec="k", mew=0.5, zorder=4)
    ax.annotate("max %.1f%% @ %.4f eq." % (mean.loc[best, "y"], mean.loc[best, "x"]),
                (mean.loc[best, "x"], mean.loc[best, "y"]), fontsize=FS,
                xytext=(46, -20), textcoords="offset points", alpha=0.60,
                bbox=dict(boxstyle="round,pad=0.25", fc="w", alpha=0.60,
                          ec="0.6", lw=0.6))
    ax.set_xlabel("Catalyst loading (eq.)", fontsize=FL, labelpad=1.5)
    ax.set_ylabel("Predicted yield (%)", fontsize=FL, labelpad=1.5)
    ax.tick_params(labelsize=FS, length=2.2, pad=1.5)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=FS, loc="lower right", borderpad=0.3, labelspacing=0.25,
              handletextpad=0.4, borderaxespad=0.3)
    ax.set_axisbelow(True)
    # The "(b)" is stamped after the tight crop (see PANEL_LABEL / _save).
    fig.subplots_adjust(left=0.145, right=0.985, top=0.965, bottom=0.135)
    return fig


# ---------------------------------------------------------------------------
# LEGACY PANELS (not in FIGURES any more)
#
# fig1a, fig1b, fig2   superseded by the same panels in figures_paper2.py
# fig3a, fig3b         these were Figures 3(a)/(b) under the OLD numbering; they are
#                      now Figure 4(a)/(b) and are rendered by figures_paper2.py
# fig4                 old numbering; now Figure 5 (fig5_heatmap)
# fig5                 old numbering; now Figure 7 (fig7_protocols)
# fig6a                old numbering; now Figure 8(a) (fig8a_acquisition)
# fig6b                kept as the Figure 8(b) BACKUP (fig8b_backup_loading)
#
# They stay callable because figures_paper2.py imports fig4 and the shared helpers.
# ---------------------------------------------------------------------------
# Registry
FIGURES: dict[str, dict] = {
    # Only the backup panel lives here: every figure that appears in the
    # manuscript is rendered by figures_paper2.py.  This one is the alternative
    # second panel of Figure 8 (predicted yield against catalyst loading); the
    # paper itself uses the predicted-yield-versus-sd scatter instead.
    "fig8b_backup_loading": {
        "heavy": True,
        "desc": "Figure 8(b) backup: predicted yield of the best candidate versus "
                "catalyst loading (NOT used in the manuscript)",
        "fn": fig6b,
    },
}


def _print_list() -> None:
    print("Available figures (%d):" % len(FIGURES))
    for name, spec in FIGURES.items():
        print("  %-6s  heavy=%-3s  %s"
              % (name, "yes" if spec["heavy"] else "no", spec["desc"]))
    heavy = [name for name, spec in FIGURES.items() if spec["heavy"]]
    print()
    print("Heavy figures (%s) are gated: pass --allow-heavy to render them."
          % ", ".join(heavy))


def _resolve_names(requested: list[str]) -> list[str]:
    if "all" in requested:
        return list(FIGURES)
    unknown = [name for name in requested if name not in FIGURES]
    if unknown:
        raise SystemExit("unknown figure name(s): %s\nvalid names: %s, all"
                         % (", ".join(unknown), ", ".join(FIGURES)))
    # de-duplicate while keeping the order the user gave
    seen, ordered = set(), []
    for name in requested:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/figures.py",
        description="Regenerate the paper's figures into figures/<name>.png at 300 dpi.")
    parser.add_argument("--list", action="store_true",
                        help="list the available figures and their heavy flag")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="render only the named figure(s); 'all' renders every "
                             "cheap figure and skips the heavy ones")
    parser.add_argument("--allow-heavy", action="store_true",
                        help="allow figures that depend on the virtual-library / "
                             "Bayesian-optimisation path")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="directory holding the raw dataset CSVs and the "
                             "pipeline's col_mappings.csv (default: <repo>/data)")
    parser.add_argument("--grid", type=Path, default=None,
                        help="grid table for fig1a/fig1b (default: "
                             "results/grid13_v18.csv, the 13-dataset corpus)")
    parser.add_argument("--ceiling", type=Path, default=None,
                        help="per-dataset ceiling table for fig2 (default: "
                             "results/table1_v18_final.csv, the 13-dataset corpus)")
    args = parser.parse_args(argv)

    if args.list:
        _print_list()
        return 0
    if not args.only:
        parser.print_help()
        return 0

    global _DATA_OVERRIDE, _GRID_OVERRIDE, _CEILING_OVERRIDE
    if args.data_dir is not None:
        _DATA_OVERRIDE = args.data_dir
    if args.grid is not None:
        _GRID_OVERRIDE = args.grid
    if args.ceiling is not None:
        _CEILING_OVERRIDE = args.ceiling

    print("figure renderer: font.family=%r (serif enforced by _paper_rcparams())"
          % (mpl.rcParams["font.family"],))
    skipped: list[str] = []
    rendered: list[str] = []
    try:
        for name in _resolve_names(args.only):
            spec = FIGURES[name]
            if spec["heavy"] and not args.allow_heavy:
                print("skipping heavy figure %r: it depends on the virtual-library / "
                      "Bayesian-optimisation path of the frozen pipeline." % name)
                print("  Re-run with --allow-heavy to render it.")
                skipped.append(name)
                continue
            print("rendering %s ..." % name)
            _save(spec["fn"](), name)
            rendered.append(name)
    except FigureInputMissing as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    print()
    print("rendered: %s" % (", ".join(rendered) if rendered else "none"))
    if skipped:
        print("skipped (heavy, pass --allow-heavy): %s" % ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())

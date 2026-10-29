# -*- coding: utf-8 -*-
"""Render every panel the paper needs, using the pipeline's own helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm

import os
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))   # the repo root, so that `import hte` works
sys.stdout.reconfigure(encoding="utf-8")
import figures as F

OUT = _HERE.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
F.FIG_DIR = OUT
F._DATA_OVERRIDE = _HERE.parent / "data"

RES = _HERE.parent / "results"
V239 = _HERE.parent / "outputs" / "acquisition"   # acq_mu.npy / acq_sd.npy (in-package)
ATTRIB = RES / "rebuilt_attrib12_long_v7.csv"
ENRICH = RES / "enrichment13.csv"


def _corpus_drop() -> set:
    """Datasets present in data/ but outside the manuscript's corpus.

    Read from ``data/alias_mapping.yaml`` (``corpus: false``); this used to be a
    literal set of three dataset names.
    """
    from hte.datasets import corpus, dataset_order
    return {a for a, _s in dataset_order()} - set(corpus())


DROP = _corpus_drop()
BLUE, RED = "#4472C4", "#C00000"
F._GRID_OVERRIDE = RES / "grid13_v18.csv"
F._CEILING_OVERRIDE = RES / "table1_v18_final.csv"


def canvas(w_cm, h_cm):
    return w_cm / 2.54, h_cm / 2.54


def blend(lo, hi, v):
    """Blend relative to the 2.0 base colour: lighter below it, darker above."""
    import matplotlib.colors as mc
    v = float(v)
    if v >= 2.0:
        f = 0.55 + 0.45 * min((v - 2.0) / 2.0, 1.0)      # base 2.0 -> darkest at 4.0
    else:
        f = 0.55 * (v / 2.0)                              # lightest at 1.0
    f = max(0.10, min(1.0, f))
    return tuple(1 - f * (1 - c) for c in mc.to_rgb(BLUE))


def finish(fig, name):
    F._save(fig, name)
    print("  ->", (OUT / (name + ".png")).name)


def frame(ax, letter=None):
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    if letter:
        pass  # panel letters are stamped by _save


finish(F.fig1a(), "fig1a")
finish(F.fig1b(), "fig1b")

finish(F.fig2(), "fig2")
finish(F.fig5(), "fig7_protocols")

finish(F.fig3a(), "fig3a_shares")

a = pd.read_csv(ATTRIB)
a = a[~a.dataset.isin(DROP)].copy()
a["slot"] = a.feature.astype(str).str.replace("_encoded", "", regex=False)
a["absx"] = a.shap.abs()
tot = a.groupby(["dataset", "model", "encoding"])["absx"].transform("sum")
a["share"] = np.where(tot > 0, 100 * a["absx"] / tot, 0.0)
piv = (a.groupby(["dataset", "model", "encoding", "slot"])["share"].sum()
       .unstack("slot").fillna(0.0))
piv = piv.loc[:, piv.max() > 1e-9]
piv = piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0) * 100
first = piv.idxmax(axis=1).value_counts().reindex(piv.columns).fillna(0)

w, h = canvas(8.6, 7.2)
fig, ax = plt.subplots(figsize=(w, h))
o = first[first > 0].sort_values()
_n = max(len(o) - 1, 1)
_cmap = plt.get_cmap("Blues_r")
_bars = [_cmap(0.85 - 0.65 * i / _n) for i in range(len(o))]
ax.barh(range(len(o)), o.values, color=_bars,
        edgecolor="black", linewidth=0.5, height=0.66)
for i, v in enumerate(o.values):
    ax.text(v + 0.7, i, "%d" % int(v), va="center", fontsize=F.FS)
ax.set_yticks(range(len(o)))
ax.set_yticklabels([F.SHAP_LABELS.get(str(k) + "_encoded", F.SHAP_LABELS.get(str(k), str(k).replace("_", " ").capitalize())) for k in o.index], fontsize=F.FS)
ax.tick_params(axis="y", length=2.6, direction="out")
ax.set_xlabel("Number of the 156 cells in which the slot ranks first",
              fontsize=F.FL, labelpad=2)
ax.tick_params(axis="x", labelsize=F.FS, direction="in", length=2.2)
ax.yaxis.set_tick_params(length=3.0, width=0.8, direction="out", color="black")
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.grid(axis="x", color="#dddddd", lw=0.5)
ax.set_axisbelow(True)
ax.set_xlim(0, o.max() * 1.16)
frame(ax, "(b)")
finish(fig, "fig3b_ranks")

shap = F._shap()
state = F._load_bh()
cols, values, df, te = shap["cols"], shap["values"], state["df"], state["te_pos"]


def slot_boxes(slot, raw_candidates, name, letter, xlabel):
    j = cols.index(slot)
    names = None
    for rc in raw_candidates:
        if rc in df.columns:
            names = df[rc].astype(str).to_numpy()[te]
            break
    if names is None:
        names = state["test"].iloc[:, j].astype(str).to_numpy()
    s = pd.Series(values[:, j]).groupby(names)
    ranked = sorted(s.groups, key=lambda k: -np.mean(np.abs(s.get_group(k))))[:12][::-1]
    w, h = canvas(8.6, 8.4)
    fig, ax = plt.subplots(figsize=(w, h))
    bp = ax.boxplot([s.get_group(k).values for k in ranked], vert=False,
                    widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#E8A33D", lw=1.3))
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)
    for elem in ("whiskers", "caps"):
        for line in bp[elem]:
            line.set_color("black")
            line.set_linewidth(0.7)
    ax.set_yticks(range(1, len(ranked) + 1))
    ax.set_yticklabels(ranked, fontsize=F.FS)
    ax.axvline(0, color="black", ls=(0, (4, 3)), lw=0.7)
    ax.set_xlabel(xlabel, fontsize=F.FL, labelpad=2)
    ax.tick_params(labelsize=F.FS, direction="in", length=2.2)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    ax.margins(x=0.06)
    frame(ax, letter)
    finish(fig, name)


try:
    slot_boxes("reactant_1_encoded",
               ("startingmat_1_name", "substrate_name", "aryl_halide_name"),
               "fig4a_reactant1", "(a)", "SHAP value (yield points)")
    slot_boxes("reactant_2_encoded",
               ("startingmat_2_name", "leaving_group_name", "amine_name"),
               "fig4b_reactant2", "(b)", "SHAP value (yield points)")
except Exception as _exc:                # a missing slot must not block the later panels
    print("[skip] slot boxes: %s" % _exc)

fig = F.fig4()
ax = fig.axes[0]
mat = np.asarray(ax.images[0].get_array(), dtype=float)
finite = mat[np.isfinite(mat)]
thr = np.nanpercentile(finite, 90)
vmax = np.nanmax(finite)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        if not np.isfinite(v) or v < thr:
            continue
        ax.text(j, i, "%.0f" % v, ha="center", va="center_baseline",
                fontsize=F.FS - 0.5,
                color="white" if v > 0.62 * vmax else "black", fontweight="bold")
pass
finish(fig, "fig5_heatmap")

e = pd.read_csv(ENRICH).sort_values("ef_1000")
w, h = canvas(8.6, 7.4)
fig, ax = plt.subplots(figsize=(w, h))
lo, hi = float(e.ef_1000.min()), float(e.ef_1000.max())
ramp = [blend(lo, hi, v) for v in e.ef_1000]
ax.barh(range(len(e)), e.ef_1000, color=ramp,
        edgecolor="black", linewidth=0.5, height=0.66)
for i, v in enumerate(e.ef_1000):
    ax.text(v + 0.07, i, "%.2f" % v, va="center", fontsize=F.FS)
ax.axvline(1.0, color="black", lw=0.9, ls=(0, (4, 3)))
ax.set_yticks(range(len(e)))
ax.set_yticklabels([d.replace("Roche66k_", "Roche_") for d in e.alias], fontsize=F.FS)
ax.set_xlabel("Enrichment at k = 1000\n(predicted-yield ranking vs random)",
              fontsize=F.FL, labelpad=2)
ax.tick_params(axis="x", labelsize=F.FS, direction="out", length=2.5)
ax.tick_params(axis="y", labelsize=F.FS, direction="out", length=2.5)
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
ax.grid(axis="x", color="#dddddd", lw=0.5)
ax.set_axisbelow(True)
ax.set_xlim(0, e.ef_1000.max() * 1.16)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(True)
ax.yaxis.set_minor_locator(mticker.NullLocator())
finish(fig, "fig6_enrich")


mu = np.load(V239 / "acq_mu.npy").astype(float)
sd = np.load(V239 / "acq_sd.npy").astype(float)
N = 100
order = np.argsort(-mu)
rank_curve = np.maximum.accumulate(mu[order])[:N]
avail = np.ones(len(mu), bool)
# Two different "best" values: the EI REFERENCE is the best predicted condition in
# the library (exploration is measured against it), while `best_seen` is the best
# condition the EI order has actually reached so far -- that is what the curve plots.
# Using one variable for both made the EI curve a flat line on top of the ranking curve.
ei_ref = float(mu.max())
best_seen, ei_curve = -np.inf, []
for _ in range(N):
    z = (mu - ei_ref) / np.maximum(sd, 1e-9)
    ei = (mu - ei_ref) * norm.cdf(z) + sd * norm.pdf(z)
    ei[~avail] = -np.inf
    k = int(np.argmax(ei))
    avail[k] = False
    best_seen = max(best_seen, float(mu[k]))
    ei_curve.append(best_seen)
perm = np.random.default_rng(42).permutation(len(mu))
rand_curve = np.maximum.accumulate(mu[perm])[:N]
print("acq: rank %.2f  EI %.2f  random %.2f" % (rank_curve[0], ei_curve[0], rand_curve[0]))

w, h = canvas(8.6, 6.4)
fig, ax = plt.subplots(figsize=(w, h))
ax.plot(range(1, N + 1), rank_curve, color=BLUE, lw=1.3, label="ranking by predicted yield")
ax.plot(range(1, N + 1), ei_curve, color=RED, lw=1.3, label="expected improvement")
ax.plot(range(1, N + 1), rand_curve, color="0.45", lw=1.0, ls="--", label="random order")
ax.set_xlabel("Conditions evaluated", fontsize=F.FL, labelpad=2)
ax.set_ylabel("Cumulative best predicted yield (%)", fontsize=F.FL, labelpad=2)
ax.tick_params(labelsize=F.FS, direction="in", length=2.2)
ax.grid(color="#dddddd", lw=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=F.FS, loc="lower right", frameon=False, borderpad=0.2,
          labelspacing=0.25, handletextpad=0.4, borderaxespad=0.3)
frame(ax, "(a)")
finish(fig, "fig8a_acquisition")

sel = np.random.default_rng(42).choice(len(mu), size=min(20000, len(mu)), replace=False)
i_top = int(np.argmax(mu))
z = (mu - mu[i_top]) / np.maximum(sd, 1e-9)
ei0 = (mu - mu[i_top]) * norm.cdf(z) + sd * norm.pdf(z)
i_ei = int(np.argmax(ei0))
print("sd: top mu %.2f sd %.2f ; EI mu %.2f sd %.2f ; median sd %.2f"
      % (mu[i_top], sd[i_top], mu[i_ei], sd[i_ei], np.median(sd)))
w, h = canvas(8.6, 6.8)
fig, ax = plt.subplots(figsize=(w, h))
ax.scatter(sd[sel], mu[sel], s=1.0, alpha=0.22, color=BLUE, linewidths=0)
ax.scatter([sd[i_top]], [mu[i_top]], s=34, color=BLUE, edgecolor="black",
           linewidth=0.6, zorder=3, label="top predicted yield")
ax.scatter([sd[i_ei]], [mu[i_ei]], s=34, color=RED, edgecolor="black",
           linewidth=0.6, zorder=3, label="first EI pick")
ax.set_xlabel("Predictive standard deviation (yield points)", fontsize=F.FL, labelpad=2)
ax.set_ylabel("Predicted yield (%)", fontsize=F.FL, labelpad=2)
ax.tick_params(labelsize=F.FS, direction="in", length=2.2)
ax.grid(color="#dddddd", lw=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=F.FS, loc="lower right", frameon=False, borderpad=0.2,
          labelspacing=0.25, handletextpad=0.4, borderaxespad=0.3)
frame(ax, "(b)")
finish(fig, "fig8b_mu_sd")

print("\noutput dir:", OUT)
for q in sorted(OUT.glob("*.png")):
    print("  %-24s %7d B" % (q.name, q.stat().st_size))

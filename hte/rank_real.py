# -*- coding: utf-8 -*-
"""Route-1 ranking of MEASURED records, extracted from the frozen pipeline.

``generate_figure6_direct_ranking`` and the helpers/constants it needs used to be
fetched from ``single_dataset_full_benchmark.py`` through a loader that has since
been retired (2026-10-28).  They are
copied here so that the recommendation tool no longer loads the monolith at run time.

One deliberate edit: the nested helper ``_canon`` closed over the outer ``df``; it now
takes ``df`` as a parameter, and its call sites were updated.

Extracted 2026-10-28.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hte.v16_core import get_split_positions, prepare_features

ENSEMBLE_MIN_SUPPORT = 0

STATUS_MATCH_LEVEL = 'reagent'

PREDICTION_MIN = 0.0

PREDICTION_MAX = 100.0

ENSEMBLE_MAX_MODEL_GAP = 20.0

CONDITION_COLS = ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'cosolvent', 'additive',
               'reactant_2', 'reactant_1', 'temperature', 'time',
               'catalyst_loading', 'reactant_2_eq', 'base_eq', 'reactant_1_eq',
               'ligand_eq', 'solvent_eq', 'additive_eq']

COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#CA9161',
    'grey': '#949494'
}

CLIP_MEMBER_PREDICTIONS = True

def _canon(df, cols):
    """Copied from the monolith, where it closed over ``df``; here df is a parameter."""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = df[c].astype(str)
    return out


def generate_figure6_direct_ranking(
    df,
    output_dir,
    best_models_dict=None,
    top_n_table=20,
    top_csv_n=500,
    max_model_gap=ENSEMBLE_MAX_MODEL_GAP,
    min_support_count=0,
    weights=None,
    member_names=None
):
    """
    FIGURE 6 -- rank REAL, experimentally measured reactions by the ENSEMBLE MEAN
    predicted yield, with no acquisition function involved.

    Data source: the historical reaction records themselves.  Every row IS a reaction that
    was actually run and measured, which is why this is the ONLY figure with a
    "Measured Yield" column -- that value is ground truth, not a prediction.  The models
    are used here purely to RANK those real reactions.

    What this is NOT: it is not the virtual library.  The enumerated, largely untested
    reagent combinations live in Figure 9 (Bayesian optimisation) and Figure 10 (top-20
    predicted conditions); those tables have predicted yields and no measured yield.

      Figure 9 / 10 ->  virtual library: millions of enumerated, largely UNTESTED
                        reagent combinations, ranked by predicted yield.
      Figure 6      ->  real reactions bearing measured yields, ranked by the ensemble
                        mean.  Because ground truth exists, the ranking can be audited:
                        the figure reports the measured yield of the predicted top-k.

    v13 consistency fix: this used to average EVERY `Predicted_Yield_*` column present in
    the table, which included the Multilayer Perceptron -- so the pool was a mean of FOUR models
    while the virtual-library figures used a weighted mean of the top THREE.  It now uses
    the same member set and the same R^2-derived weights, so all of them describe one
    ensemble.

    `member_names` are the ensemble members in weight order; when given, exactly those
    `Predicted_Yield_<name>` columns are used and the Multilayer Perceptron is excluded.

    Returns the ranked DataFrame (real reactions, best first).
    """


    PRED_COLS = {
        'XGBoost': 'Predicted_Yield_XGBoost',
        'Random Forest': 'Predicted_Yield_Random Forest',
        'Gradient Boosting': 'Predicted_Yield_Gradient Boosting',
        'Multilayer Perceptron': 'Predicted_Yield_Multilayer Perceptron',
    }
    if member_names:
        ordered = [n for n in member_names if PRED_COLS.get(n) in df.columns]
        dropped = [n for n in PRED_COLS if n not in ordered and PRED_COLS[n] in df.columns]
        available = {n: PRED_COLS[n] for n in ordered}
        if dropped:
            print(f"  excluding from the ensemble: {dropped} "
                  f"(Figure 10's ensemble members are {ordered})")
    else:
        available = {k: v for k, v in PRED_COLS.items() if v in df.columns}

    pool = df.copy()
    if len(available) >= 2:
        usable = list(available.values())
        pool = pool.dropna(subset=usable).copy()
        mat = pool[usable].to_numpy(dtype=float)
        # v13: same member-level clipping as the Figure 10 sweep.  A yield is a normalised UV area
        # percentage, so [0, 100] are hard limits; without this an additive-boosting
        # member can report >100 % (observed: XGBoost 104 % on a real reaction) and the
        # impossible value would be averaged in as if it were evidence.
        if CLIP_MEMBER_PREDICTIONS:
            n_bad = int(np.count_nonzero((mat < PREDICTION_MIN) | (mat > PREDICTION_MAX)))
            mat = np.clip(mat, PREDICTION_MIN, PREDICTION_MAX)
            if n_bad:
                print(f"  [Figure 6] clipped {n_bad:,} member predictions to "
                      f"[{PREDICTION_MIN:g}, {PREDICTION_MAX:g}]")
        if weights is not None and len(weights) == mat.shape[1]:
            wv = np.asarray(weights, dtype=float)
            wv = np.clip(wv, 0.0, None)
            wv = wv / wv.sum()
            mean_v = (mat * wv[None, :]).sum(axis=1)
            var_v = (wv[None, :] * (mat - mean_v[:, None]) ** 2).sum(axis=1) / \
                max(1.0 - float((wv ** 2).sum()), 1e-12)
            print(f"  weighted ensemble mean, weights {dict(zip(available, wv.round(4)))}")
            pool['Ensemble Predicted Yield (%)'] = mean_v
            pool['Prediction Uncertainty (%)'] = np.sqrt(np.maximum(var_v, 0.0))
        else:
            pool['Ensemble Predicted Yield (%)'] = mat.mean(axis=1)
            pool['Prediction Uncertainty (%)'] = mat.std(axis=1, ddof=1)
            print("  (unweighted ensemble mean — no weights supplied)")
        pool['Model Gap (%)'] = mat.max(axis=1) - mat.min(axis=1)
        pool['Model Min (%)'] = mat.min(axis=1)
        pool['Model Max (%)'] = mat.max(axis=1)
        pool['Model Preset'] = ', '.join(available.keys())
        for j, (nm, col) in enumerate(available.items()):
            # the CLIPPED member values, so the table matches what the ensemble averaged
            pool[f'Pred_{nm}'] = mat[:, j]
        print(f"  predictions found for: {list(available)}")
    else:
        # fallback: recompute with the trained models
        if not best_models_dict:
            print("  Figure 6 skipped: no prediction columns and no trained models.")
            return None
        _tr_rows, _ = get_split_positions(len(df))
        X_all, _, feat_cols, _, enc, sc = prepare_features(df, fit_rows=_tr_rows)
        mat = np.vstack([m.predict(X_all) for m in best_models_dict.values()])
        if CLIP_MEMBER_PREDICTIONS:
            mat = np.clip(mat, PREDICTION_MIN, PREDICTION_MAX)
        pool['Ensemble Predicted Yield (%)'] = mat.mean(axis=0)
        pool['Prediction Uncertainty (%)'] = mat.std(axis=0, ddof=1)
        pool['Model Gap (%)'] = mat.max(axis=0) - mat.min(axis=0)
        pool['Model Min (%)'] = mat.min(axis=0)
        pool['Model Max (%)'] = mat.max(axis=0)
        pool['Model Preset'] = ', '.join(best_models_dict.keys())
        for j, nm in enumerate(best_models_dict.keys()):
            pool[f'Pred_{nm}'] = mat[j]


    # Computed per CONDITION from the original rows BEFORE dedupe, because the row-level
    # label cannot survive the groupby (see _condition_split_table).  The rule is the
    # conservative one: 'train' only if EVERY replicate of the condition was fitted.
    try:
        idx_tr, idx_te = get_split_positions(len(df))
        _split_tbl, _key_cols = _condition_split_table(df, idx_tr, CONDITION_COLS)
        if _key_cols:
            # np.asarray strips the index so the merge cannot silently align on labels
            # instead of matching values.
            _pool_key = np.asarray(
                df[_key_cols].astype(str).agg('|'.join, axis=1).reindex(pool.index))
            pool = pool.merge(_split_tbl, left_on=_pool_key, right_on='_ckey', how='left')
            pool = pool.drop(columns=['_ckey'])
            n_unmatched = int(pool['Split'].isna().sum())
            pool['Split'] = pool['Split'].fillna('unknown')
            if n_unmatched:
                print(f"  [Figure 6] WARNING: {n_unmatched:,} rows had no condition-key "
                      f"match; labelled Split='unknown'")

    except Exception as e:
        print(f"  (could not reconstruct the train/test split: {e})")
        pool['Split'] = 'unknown'

    # v13: collapse replicate runs so the ranking is over CONDITIONS, not measurements.
    # Without this the top-20 contained only 9 distinct conditions (every one repeated).
    # 'Split' is constant within a condition key, so the 'first' aggregation that
    # dedupe_conditions applies to it preserves the correct value.
    n_pool_rows = len(pool)          # for the consistency check below
    pool, n_replicates = dedupe_conditions(pool, condition_cols=CONDITION_COLS)

    n_all = len(pool)
    if max_model_gap is not None:
        before = len(pool)
        pool = pool[pool['Model Gap (%)'] <= float(max_model_gap)].copy()
        print(f"  [Gate A] Model Gap <= {max_model_gap:g} pp: kept {len(pool):,} / {before:,} "
              f"conditions ({(len(pool) / before):.1%})")
    if len(pool) == 0:
        print("  Figure 6 skipped: the gate removed every reaction.")
        return None

    historical_stats, group_cols, hist_aux = build_prediction_lookup(df, pool)
    pool = add_validation_status(pool, historical_stats, group_cols, hist_aux)
    pool, _ = apply_support_gate(pool, min_support_count)

    # (the train/held-out label was assigned before dedupe_conditions above)

    # This is the MEASURED reaction pool, so "which model?" has a concrete answer: the
    # predicted-yield columns come from the same trained members as Figure 10, and the
    # ensemble value is the same weighted mean (R^2 weights from this run).  The measured
    # `yield` column is ground truth and never enters the ranking.
    ranked = pool.sort_values('Ensemble Predicted Yield (%)', ascending=False).reset_index(drop=True)
    ranked.insert(0, 'Rank', np.arange(1, len(ranked) + 1))
    ranked['Prediction Error (%)'] = ranked['Ensemble Predicted Yield (%)'] - ranked['yield']
    table_src = ranked.head(top_n_table).copy()
    _member_cols = [f'Pred_{n}' for n in (member_names or []) if f'Pred_{n}' in ranked.columns]


    base_mean = float(df['yield'].mean())
    base_hit = float((df['yield'] >= 50).mean())
    ks = [10, 20, 50, 100, 250, 500, 1000]
    audit = []
    for k in ks:
        t = ranked.head(k)
        if len(t) == 0:
            continue
        hit = float((t['yield'] >= 50).mean())
        row = {'k': k, 'mean_yield': float(t['yield'].mean()),
               'hit_rate': hit,
               'enrichment': (hit / base_hit) if base_hit > 0 else np.nan}
        if 'Split' in t.columns:
            ho = t[t['Split'] == 'held-out']
            fi = t[t['Split'] == 'train']
            row['n_heldout_in_topk'] = len(ho)
            row['mean_yield_heldout'] = float(ho['yield'].mean()) if len(ho) else np.nan
            row['n_fitted_in_topk'] = len(fi)
            row['mean_yield_fitted'] = float(fi['yield'].mean()) if len(fi) else np.nan
        audit.append(row)
    audit_df = pd.DataFrame(audit)


    # Taller than the table strictly needs: the footnote below the table is 5 lines and was
    # being clipped at the figure edge.  `bottom` leaves room for it in figure coordinates
    # so bbox_inches='tight' does not have to guess.
    fig = plt.figure(figsize=(17, 13.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.8], hspace=0.30,
                          top=0.96, bottom=0.115, left=0.05, right=0.98)

    # panel row 1: (a) ranking enrichment, (b) calibration -- both on measured data
    gs_a = gs[0].subgridspec(1, 2, wspace=0.20)
    ax1 = fig.add_subplot(gs_a[0])
    # Panel (a) -- the three curves measure three DIFFERENT populations, so they must be
    # labelled as such.  `all` mixes fitted and held-out conditions; the fitted subset is
    # systematically easier (the model has seen those exact conditions), which is exactly
    # why `all` sits ABOVE the held-out curve.  The held-out curve is the honest
    # generalisation estimate; the fitted curve shows how much of the apparent enrichment
    # is memorisation.
    ax1.plot(audit_df['k'], audit_df['mean_yield'], 'o-', color=COLORS['blue'], lw=2, ms=5,
             label='Predicted top-$k$, all conditions (fitted + held-out)')
    if 'mean_yield_fitted' in audit_df.columns:
        ax1.plot(audit_df['k'], audit_df['mean_yield_fitted'], '^:', color=COLORS['red'],
                 lw=1.8, ms=4.5, label='... of which fitted (seen in training)')
    if 'mean_yield_heldout' in audit_df.columns:
        ax1.plot(audit_df['k'], audit_df['mean_yield_heldout'], 's--', color=COLORS['green'],
                 lw=1.8, ms=4, label='... of which held-out (unseen)')
    ax1.axhline(base_mean, color=COLORS['orange'], lw=1.6, ls='--',
                label=f'All reactions mean ({base_mean:.1f}%)')
    ax1.set_xscale('log')
    ax1.set_xlabel('k  (top-$k$ real conditions, ranked by ensemble mean)', fontweight='bold')
    ax1.set_ylabel('Measured yield (%)', fontweight='bold')
    ax1.set_title('(a) Enrichment — measured yield of the predicted top-$k$',
                  fontsize=9, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=6.5)
    ax1.grid(True, linestyle='--', alpha=0.35)

    # panel (b) v13: CALIBRATION, not parity.
    # The old (b) was a predicted-vs-measured scatter, which duplicated the residual /
    # predicted-vs-actual panel of Figure 3 (Figure 3 restricts it to the test split;
    # this showed all rows).  Ranking quality is already evidenced by (a), so (b) now
    # answers the complementary question the paper could not answer before:
    #     "when the ensemble says 80 %, is the measured yield really about 80 %?"
    # Rows are binned into equal-count buckets of PREDICTED yield; each point is that
    # bucket's mean measured yield with a standard-error bar.  A well-calibrated model
    # lies on y = x.  The overall bias is printed onto the panel so the number is
    # quotable even if the reader only glances at the figure.
    ax2 = fig.add_subplot(gs_a[1])
    lim = [-2, 102]
    ax2.plot(lim, lim, color='black', lw=1.0, ls='--', label='Perfect calibration (y = x)')

    calib_summary = {}
    for tag, sub, colr, mk in (
            ('all rows', ranked, COLORS['blue'], 'o'),
            ('held-out only', ranked[ranked['Split'] == 'held-out'], COLORS['green'], 's')):
        sub = sub.dropna(subset=['Ensemble Predicted Yield (%)', 'yield'])
        if len(sub) < 50:
            continue
        nbin = int(min(10, max(3, len(sub) // 150)))
        edges = np.unique(np.quantile(sub['Ensemble Predicted Yield (%)'],
                                      np.linspace(0, 1, nbin + 1)))
        if len(edges) < 3:
            continue
        idx = np.clip(np.digitize(sub['Ensemble Predicted Yield (%)'], edges[1:-1]), 0, len(edges) - 2)
        g = sub.groupby(idx)
        px = g['Ensemble Predicted Yield (%)'].mean().to_numpy()
        my = g['yield'].mean().to_numpy()
        se = g['yield'].sem().to_numpy()
        cnt = g.size().to_numpy()
        ok = cnt >= 10
        px, my, se = px[ok], my[ok], se[ok]
        if len(px) == 0:
            continue
        ax2.errorbar(px, my, yerr=np.nan_to_num(se), fmt=mk + '-', color=colr,
                     lw=1.8, ms=4.5, capsize=2.5, alpha=0.95,
                     label=f'{tag} (n={len(sub):,})')
        # calibration slope through the origin: measured ~ slope * predicted
        s = float((px * my).sum() / max((px * px).sum(), 1e-12))
        bias = float(np.mean(my - px))
        mace = float(np.mean(np.abs(my - px)))
        calib_summary[tag] = (s, bias, mace, len(sub))

    if calib_summary:
        lines = [f'{t}: slope={v[0]:.2f}, bias={v[1]:+.1f} pp, MACE={v[2]:.1f} pp'
                 for t, v in calib_summary.items()]
        ax2.text(0.03, 0.97, '\n'.join(lines), transform=ax2.transAxes,
                 fontsize=6.5, va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.5', alpha=0.9))

    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel('Mean predicted yield in bin (%)', fontweight='bold')
    ax2.set_ylabel('Mean measured yield in bin (%)', fontweight='bold')
    ax2.set_title('(b) Calibration — mean measured vs mean predicted, by predicted-yield bin',
                  fontsize=9, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=6.5)
    ax2.grid(True, linestyle='--', alpha=0.35)

    for axx in (ax1, ax2):
        for sp in axx.spines.values():
            sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(0.8)

    # panel (c): the Figure-9-style table
    axb = fig.add_subplot(gs[1])
    axb.axis('off')

    # same column treatment as Figure 9's table (all columns still in the CSV)
    # dropped from the figure: metal / temperature / time (constant here),
    # Ens. SD (collinear with Model Gap), Cond. Avg. Yield (exact-condition mean,
    # uninformative for a novel-condition ranking), Match Level (redundant with
    # Data Status + Support)
    display_columns = [
        'Rank', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_1', 'reactant_2',
        'catalyst_loading',
    ]
    # v13: same treatment as Figure 10 -- show each member's own prediction immediately
    # before the ensemble value, so the reader can see where the average comes from.
    _order = member_names if member_names else []
    _mc = [f'Pred_{n}' for n in _order if f'Pred_{n}' in table_src.columns]
    if not _mc:
        _mc = [c for c in table_src.columns if c.startswith('Pred_')]
    display_columns += _mc + [
        'Ensemble Predicted Yield (%)', 'Model Gap (%)',
        'yield', 'Yield SD (pp)', 'N Runs', 'Prediction Error (%)', 'Split',
    ]
    display_columns = [c for c in display_columns if c in table_src.columns]
    table_df = table_src[display_columns].copy()

    # v13: Data Status and Support (exact/reagent) are deliberately NOT rendered.
    # Every reaction here is a real measured reaction, so Data Status is 'Validated' for
    # all of them and carries no information; the exact-condition support is likewise
    # uninformative on this table.  Both remain in the CSV.  Row shading below is driven
    # by the MEASURED yield, not by any of these columns.
    #
    # 'N Runs' and 'Yield SD (pp)' make the replicate collapse explicit: one row per
    # distinct CONDITION, with the measured yield being the mean of its runs and the
    # spread between them reported alongside.

    rename_map = {
        'catalyst': 'Catalyst', 'base': 'Base', 'solvent': 'Solvent',
        'reactant_1': 'Substrate', 'reactant_2': 'Leaving Group',
        'catalyst_loading': 'Loading (eq.)',
        'Ensemble Predicted Yield (%)': 'Predicted Yield (%)',
        'Model Gap (%)': 'Model Gap (pp)',
        'yield': 'Measured Yield (%)',
        'Prediction Error (%)': 'Error (pp)',
        'Split': 'Split',
        'Data Status': 'Data Status',
    }
    table_df = table_df.rename(columns=rename_map)
    table_df = table_df.rename(columns=lambda c: (
        'Pred. ' + c[len('Pred_'):].replace('Gradient Boosting', 'GBDT')
        if c.startswith('Pred_') else c))
    member_disp = [c for c in table_df.columns if c.startswith('Pred. ')]

    for col in ['Loading (eq.)', 'Predicted Yield (%)', 'Model Gap (pp)',
                'Measured Yield (%)', 'Yield SD (pp)', 'Error (pp)'] + member_disp:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors='coerce')

    if 'Loading (eq.)' in table_df.columns:
        table_df['Loading (eq.)'] = table_df['Loading (eq.)'].map(lambda x: f'{x:.4g}' if pd.notna(x) else '—')
    for col in (['Predicted Yield (%)', 'Model Gap (pp)', 'Measured Yield (%)',
                 'Yield SD (pp)', 'Error (pp)'] + member_disp):
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f'{x:.1f}' if pd.notna(x) else '—')
    if 'N Runs' in table_df.columns:
        table_df['N Runs'] = table_df['N Runs'].map(lambda x: f'{int(x)}' if pd.notna(x) else '—')
    if 'Split' in table_df.columns:
        # make the meaning explicit: 'train' here means "every replicate of this condition
        # was in the fitting split", not merely "the reaction appears in the dataset".
        _smap = {'train': 'fitted', 'held-out': 'held-out', 'unknown': '?'}
        table_df['Split'] = table_df['Split'].map(lambda v: _smap.get(str(v), str(v)))

    table = axb.table(cellText=table_df.values, colLabels=table_df.columns,
                      cellLoc='center', colLoc='center',
                      bbox=[0.0, 0.05, 1.0, 0.90])
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    axb.set_xlim(0, 1); axb.set_ylim(0, 1)

    # v12: column widths proportional to the widest cell in each column
    _apply_table_widths(table, list(table_df.columns), table_df.values)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#D9EAF7')
        elif col == 0:
            cell.set_text_props(weight='bold')

    # highlight rows whose measured yield is high (the recommendation worked)
    if 'Measured Yield (%)' in table_df.columns:
        mycol = table_df.columns.get_loc('Measured Yield (%)')
        for row in range(1, len(table_df) + 1):
            try:
                v = float(table_df.iloc[row - 1, mycol])
            except (TypeError, ValueError):
                continue
            if v >= 90:
                for col in range(len(table_df.columns)):
                    table[(row, col)].set_facecolor('#D6F5D6')
            elif v < 20:
                for col in range(len(table_df.columns)):
                    table[(row, col)].set_facecolor('#FFF2CC')

    gate_txt = (f'model gap ≤ {max_model_gap:g} pp'
                if max_model_gap is not None else 'no model-gap gate')
    axb.set_title(f'Top {len(table_df)} measured reactions by direct ensemble ranking — {gate_txt}',
                  fontsize=12, fontweight='bold', pad=14)
    # Guard: the two numbers in the first footnote line must add back up to the INPUT row
    # count.  Measured over 10,138 records the accounting is 5,476 conditions + 4,662
    # merged = 10,138; a printed figure that exceeds the dataset (e.g. 5,476 + 15,712 =
    # 21,188) means the conditioned pool was built from a duplicated/re-concatenated frame.
    #
    # IMPORTANT: this check compares the POOL against `df`.  If both were duplicated by the
    # same factor the arithmetic still balances and nothing is reported -- so a warning here
    # means specifically that the POOL and the raw input DISAGREE.  It also means the
    # input/output must be checked directly for duplicate records.
    if n_all + n_replicates != len(df):
        print(f"  [Figure 6] WARNING: {n_all:,} conditions + {n_replicates:,} merged "
              f"= {n_all + n_replicates:,}, but the raw input has {len(df):,} rows.")
        print(f"  [Figure 6]   conditioned pool held {n_pool_rows:,} rows before dedupe "
              f"(expected <= {len(df):,}).")
        print(f"  [Figure 6]   => the pool and the raw input DISAGREE, so the footnote's "
              f"'merged' count must not be quoted.")
        print(f"  [Figure 6]   Check the input for duplicate records, e.g.:")
        print(f"  [Figure 6]     assert not df.duplicated().any()  "
              f"# and len(df) == len(df.drop_duplicates())")
        if n_pool_rows > len(df):
            print(f"  [Figure 6]   pool/input ratio = {n_pool_rows / max(len(df), 1):.3f}")
        print(f"  [Figure 6]   the RANKING is still valid: dedupe restores the "
              f"{n_all:,} distinct conditions, and the enrichment/calibration numbers are "
              f"computed from those.")
    note_lines = [
        f'Source: {n_all:,} distinct reaction CONDITIONS (replicate runs collapsed; '
        f'{n_replicates:,} duplicate measurements merged from {len(df):,} records), ranked by '
        f'ensemble mean predicted yield only (no expected improvement / UCB). '
        f'Model preset: {pool["Model Preset"].iloc[0]}.',
        "N Runs = how many replicate MEASUREMENTS this exact condition has; Measured Yield is "
        "their mean and Yield SD (pp) their spread.",
        "Split = 'fitted' only if EVERY replicate of the exact CONDITION was in the training "
        "set, else 'held-out' (not fully seen during fitting).",
        'Green = measured yield ≥ 90%; cream = < 20%. '
        'Constant columns and the full set of diagnostics are retained in the CSV.',
    ]
    axb.text(0.5, -0.02, '\n'.join(note_lines),
             ha='center', va='top', transform=axb.transAxes, fontsize=7, linespacing=1.35)

    fig10_path = os.path.join(output_dir, 'Figure6_Direct_Ensemble_Ranking.png')
    plt.savefig(fig10_path, dpi=300, bbox_inches='tight')
    plt.close()

    out_csv = os.path.join(
        output_dir, 'Figure6_Direct_Ensemble_Ranking_top%d.csv' % int(top_csv_n))
    ranked.head(int(top_csv_n)).to_csv(out_csv, index=False, encoding='utf-8-sig')
    audit_df.to_csv(os.path.join(output_dir, 'Figure6_enrichment_audit.csv'),
                    index=False, encoding='utf-8-sig')

    return ranked



def dedupe_conditions(pool, condition_cols, verbose=True):
    """
    Collapse replicate experiments: one row per distinct reaction CONDITION.

    Why this is needed for the Figure-6 table
    -----------------------------------------
    The historical file contains replicate runs of the same condition (46 % of its 10,138
    rows are not the first copy of their condition; 5,476 distinct conditions overall).
    The ensemble predicts from the CONDITION columns only, so every replicate of a
    condition receives an IDENTICAL prediction while carrying a different MEASURED yield.
    Ranked by prediction, the replicates therefore appear as adjacent "duplicate" rows:
    the top-500 contained only 250 distinct conditions and the top-20 only 9.

    Consequences if left alone:
      * the table looks like it has duplicates (it effectively does);
      * a heavily replicated condition occupies several ranks, crowding out other
        conditions;
      * the k-sweep in the audit is driven by replicate multiplicity, not by chemistry.

    How it aggregates
    -----------------
    Grouped on the full condition key.  Because the model is a function of the condition,
    all prediction columns agree within a group, so they are taken as-is (max).  The
    measured columns are AVERAGED -- `mean(yield | condition)` is exactly the quantity the
    regressor estimates -- and the within-condition spread is kept for transparency:
      `N Runs`   how many replicate runs exist
      `Yield SD (pp)`  their standard deviation (NaN when only one run)
    """
    key = [c for c in condition_cols if c in pool.columns]
    if not key or len(pool) == 0:
        return pool, 0

    n_before = len(pool)
    present = set(pool.columns)
    agg = {}
    for c in present:
        if c in key:
            continue
        if c == 'yield':
            agg[c] = 'mean'
        elif pd.api.types.is_numeric_dtype(pool[c]):
            agg[c] = 'max'          # identical within a condition (model output)
        else:
            agg[c] = 'first'        # descriptors etc.

    grouped = pool.groupby(key, dropna=False, as_index=False).agg(agg)

    # replicate bookkeeping
    runs = pool.groupby(key, dropna=False)['yield'].agg(['count', 'std']).reset_index()
    grouped = grouped.merge(runs, on=key, how='left')
    grouped = grouped.rename(columns={'count': 'N Runs', 'std': 'Yield SD (pp)'})
    grouped['N Runs'] = grouped['N Runs'].fillna(1).astype(int)

    n_after = len(grouped)
    if verbose:
        print(f"  [dedupe] {n_before:,} measured runs -> {n_after:,} distinct conditions "
              f"({n_before - n_after:,} replicate rows collapsed; "
              f"{(n_before - n_after) / max(n_before, 1):.0%} were repeats)")


    return grouped, n_before - n_after



def build_prediction_lookup(df, virtual_df):
    """
    Historical support for each virtual candidate.

    v12 fix: the original version merged on the EXACT condition, including
    catalyst_loading rounded to 6 dp.  That join almost never succeeds, because
    the virtual library takes its loading values from a coarse linspace grid while
    the historical data has thousands of distinct measured loadings -- so every
    candidate was reported as 'Discovery' even when its reagent combination had
    been run many times.  We now compute support at three levels: exact condition,
    reagent combination, and substrate.

    The three levels, and what each really means:
      Support Count           the full condition INCLUDING loading -> effectively always 0
                              for the enumerated grid (no grid point equals a measured
                              loading), so treat it as an exact-reproduction check
      Reagent Support Count   catalyst + base + solvent + leaving group (see the v13 note
                              at REAGENT below) -> "has this reagent combination been run"
      Substrate Support Count how often the substrate appears at all
    'Validated' is decided by Reagent Support Count, so that is the definition that
    matters for the figure's Data Status column.
    """
    group_cols = [
        c for c in ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1', 'catalyst_loading']
        if c in df.columns and c in virtual_df.columns
    ]

    hist = df.copy()
    virt = virtual_df.copy()
    loading_used = 'catalyst_loading' in group_cols

    if loading_used:
        hist['_loading_key'] = pd.to_numeric(hist['catalyst_loading'], errors='coerce').round(6)
        # idempotent: drop a stale key and rebuild, so the caller's frame does not
        # have to carry it and repeated calls are safe
        virt = virt.drop(columns=['_loading_key'], errors='ignore')
        virt['_loading_key'] = pd.to_numeric(virt['catalyst_loading'], errors='coerce').round(6)
        group_cols = [('_loading_key' if c == 'catalyst_loading' else c) for c in group_cols]

    if not group_cols:
        return pd.DataFrame(), group_cols, {}

    stats_df = (
        hist.groupby(group_cols, dropna=False)['yield']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'Actual Avg. Yield', 'count': 'Support Count'})
    )

    # v13 FIX: this used to be [catalyst, base, solvent] only, which OMITTED the leaving
    # group.  'Validated' is decided from this count (STATUS_MATCH_LEVEL == 'reagent'), so
    # the figure was labelling a candidate as validated merely because the same
    # catalyst + base + solvent had been used together -- even when the
    # catalyst + base + solvent + leaving-group combination had NEVER been run.
    # Measured on the F9 top-500: 159 of the 160 rows labelled 'Validated' had zero
    # historical runs of their actual reagent combination.  The leaving group is a
    # reagent, not a spectator, so it belongs in the key.
    REAGENT = [c for c in ['catalyst', 'ligand', 'base', 'solvent', 'reactant_2']
               if c in hist.columns]
    reagent_stats = pd.DataFrame()
    if REAGENT:
        reagent_stats = (
            hist.groupby(REAGENT, dropna=False)['yield']
            .agg(['mean', 'count'])
            .reset_index()
            .rename(columns={'mean': 'Reagent Avg. Yield', 'count': 'Reagent Support Count'})
        )

    SUB = [c for c in ['reactant_1'] if c in hist.columns]
    sub_stats = pd.DataFrame()
    if SUB:
        sub_stats = (
            hist.groupby(SUB, dropna=False)['yield']
            .agg(['count'])
            .reset_index()
            .rename(columns={'count': 'Substrate Support Count'})
        )

    aux = {'reagent_cols': REAGENT, 'reagent_stats': reagent_stats,
           'sub_cols': SUB, 'sub_stats': sub_stats, 'loading_used': loading_used}
    return stats_df, group_cols, aux


def apply_support_gate(candidates, min_support_count=ENSEMBLE_MIN_SUPPORT):
    """Gate B: keep only conditions with enough historical precedent (if requested)."""
    if not min_support_count or 'Support Count' not in candidates.columns:
        return candidates, 0
    before = len(candidates)
    out = candidates[candidates['Support Count'] >= int(min_support_count)].reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        print(f"  [Gate B] support count >= {min_support_count}: "
              f"keeps {len(out):,} / {before:,} -> drops {dropped:,}")
    if len(out) == 0:
        print("  WARNING: the support gate removed every candidate — returning the ungated set.")
        return candidates, 0
    return out, dropped


def add_validation_status(df_candidates, historical_stats, group_cols, aux=None):
    out = df_candidates.copy()

    # Idempotent: this function may be called more than once on the same frame (e.g. the
    # caller pre-computed the support and the figure then applied it again).  Merging a
    # column that already exists produces Support Count_x / Support Count_y and a KeyError
    # further down, so drop anything this function is about to add.
    _produced = ['Actual Avg. Yield', 'Support Count', 'Reagent Avg. Yield',
                 'Reagent Support Count', 'Substrate Support Count',
                 'Data Status', 'Match Level']
    out = out.drop(columns=[c for c in _produced if c in out.columns], errors='ignore')

    # ensure the merge key exists on the candidate frame (the lookup builds it on a
    # copy, so callers without a pre-existing '_loading_key' still merge correctly)
    if '_loading_key' in group_cols and '_loading_key' not in out.columns:
        out = out.drop(columns=['_loading_key'], errors='ignore')
        out['_loading_key'] = pd.to_numeric(out['catalyst_loading'], errors='coerce').round(6)

    if historical_stats is None or historical_stats.empty or not group_cols:
        out['Actual Avg. Yield'] = np.nan
        out['Support Count'] = 0
    else:
        out = out.merge(historical_stats, on=list(group_cols), how='left')
        out['Support Count'] = out['Support Count'].fillna(0).astype(int)

    # v12: reagent-level and substrate-level support, so a candidate that reuses a
    # well-tested reagent combination is no longer mislabelled as 'Discovery'.
    if aux:
        if aux.get('reagent_stats') is not None and not aux['reagent_stats'].empty:
            out = out.merge(aux['reagent_stats'], on=aux['reagent_cols'], how='left')
            out['Reagent Support Count'] = out['Reagent Support Count'].fillna(0).astype(int)
        else:
            out['Reagent Support Count'] = 0
            out['Reagent Avg. Yield'] = np.nan

        if aux.get('sub_stats') is not None and not aux['sub_stats'].empty:
            out = out.merge(aux['sub_stats'], on=aux['sub_cols'], how='left')
            out['Substrate Support Count'] = out['Substrate Support Count'].fillna(0).astype(int)
        else:
            out['Substrate Support Count'] = 0

    if STATUS_MATCH_LEVEL == 'exact':
        hit = out['Support Count'] > 0
    elif STATUS_MATCH_LEVEL == 'reagent':
        hit = out['Reagent Support Count'] > 0
    else:
        hit = (out['Reagent Support Count'] > 0) | (out['Substrate Support Count'] > 0)

    out['Data Status'] = np.where(hit, 'Validated', 'Discovery')
    out['Match Level'] = np.where(out['Support Count'] > 0, 'exact condition',
                                  np.where(out['Reagent Support Count'] > 0, 'reagent combo',
                                           'new combination'))
    return out



def _condition_split_table(df, train_idx, condition_cols):
    """
    Per-CONDITION train membership, keyed so it can be merged onto the candidate pool.

    Why this exists
    ---------------
    The row-level split cannot be pushed through `dedupe_conditions`, because that function
    aggregates every non-key column with a scalar rule and a string column falls back to
    'first' -- so the collapsed condition inherited whichever replicate happened to be
    first, and the label FLIPPED with row order.  38.6 % of conditions (2,113 of 5,476)
    have replicates on both sides of the row split, so this mattered a great deal.

    The rule implemented here is the CONSERVATIVE one, which is the only defensible reading:
        a condition is 'train' only if EVERY one of its replicate rows was in the training
        split; if even one replicate was held out, the condition counts as 'held-out'.
    Reverse implication: 'held-out' means the condition was NOT fully seen during fitting.

    Matching uses a canonical string key so the merge cannot silently fail on dtype
    differences (a float 80.0 and the string '80.0' would otherwise not match).
    Returns (table_with_keys, key_cols).

    NOTE: this used to also return an `N exist` column (distinct training conditions per
    reagent compound).  That concept moved to Figure 10, where it is expressed as
    `Match Count` over the 5 reagent axes -- see max_component_match.
    """
    key_cols = [c for c in condition_cols if c in df.columns]
    if not key_cols:
        return pd.DataFrame(columns=['_ckey', 'Split']), key_cols

    def _canon(cols):
        out = pd.DataFrame(index=df.index)
        for c in cols:
            out[c] = df[c].astype(str)
        return out

    is_train = np.zeros(len(df), dtype=bool)
    idx = np.asarray(train_idx, dtype=int)
    idx = idx[(idx >= 0) & (idx < len(df))]
    is_train[idx] = True

    work = _canon(df, key_cols)
    work['_ckey'] = work[key_cols].agg('|'.join, axis=1)
    work['_is_tr'] = is_train
    all_tr = work.groupby('_ckey', sort=False)['_is_tr'].apply(lambda s: bool(s.all()))
    split_tbl = pd.DataFrame({'_ckey': all_tr.index,
                              'Split': np.where(all_tr.to_numpy(), 'train', 'held-out')})
    # one row per DISTINCT condition key (the pool has one row per condition)
    out = pd.DataFrame({'_ckey': np.unique(work['_ckey'].to_numpy())})
    return out.merge(split_tbl, on='_ckey', how='left'), key_cols



def _apply_table_widths(table, labels, values):
    """Set per-column widths on an ax.table and return the width fractions."""
    fracs = _table_widths(labels, values)
    for j, frac in enumerate(fracs):
        for i in range(len(values) + 1):
            table[(i, j)].set_width(frac)
    return fracs



def _table_widths(labels, values, key_chars=5.6):
    """
    Column widths for ax.table, proportional to the widest rendered cell.

    matplotlib's ax.table gives every column 1/ncols by default, which squeezes the
    long columns (catalyst / substrate / leaving group) and wastes space on the short
    ones (Rank, Temp.).  Weighting by the longest string per column fixes both.
    """
    ncol = len(labels)
    nrow = len(values)
    weights = []
    for j in range(ncol):
        longest = len(str(labels[j]))
        for i in range(nrow):
            longest = max(longest, len(str(values[i][j])))
        weights.append(longest + key_chars)
    total = float(sum(weights))
    return [w / total for w in weights]



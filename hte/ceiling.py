"""Noise ceiling: ``R2_max = 1 - sigma2_within / sigma2_total``.

``estimate_noise_ceiling`` is copied verbatim from
``archive/Analysis_all_v16.py`` (v16 line 889).  The body is unchanged;
it reads no module globals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params as P


def estimate_noise_ceiling(df, cond_cols=None):
    """
    R2_max = 1 - sigma2_within / sigma2_total, over conditions that were measured more than once.

    Distinguishes the two very different reasons a model can fail:
        (a) the features carry no signal            -> ceiling HIGH, the model is the problem
        (b) replicate noise dominates the variance  -> ceiling LOW,  the model is at its limit

    Returns a dict with `ceiling` = NaN when NO condition repeats, because the
    quantity is then undefined rather than zero.  `rep_frac` is the share of rows
    in a repeated condition.

    CAVEAT carried over from the paper: the condition key here is the eight MAPPED
    columns, and the mapping does not separate catalyst from ligand, so the estimate
    sits BELOW the data source's own official value.  Treat it as a lower bound on
    the true ceiling.
    """
    COND = list(P.CONDITION_COLS)
    cols = [c for c in (cond_cols or COND) if c in df.columns]
    out = dict(ceiling=float('nan'), s2_within=float('nan'), s2_total=float('nan'),
               n_rep_groups=0, rep_frac=0.0, n_cond=0)
    if not cols or 'yield' not in df.columns:
        return out
    y = pd.to_numeric(df['yield'], errors='coerce')
    ok = y.notna()
    if not ok.any():
        return out
    _d, y = df[ok], y[ok]
    key = _d[cols].astype(str).agg('|'.join, axis=1)
    grp = y.groupby(key)
    sizes = grp.size()
    out['n_cond'] = int(sizes.size)
    rep = sizes[sizes >= 2]
    if len(rep) == 0:
        return out
    var = grp.var(ddof=1)
    num = ((rep - 1) * var.loc[rep.index]).sum()
    den = (rep - 1).sum()
    if den <= 0:
        return out
    s2w = float(num / den)
    s2t = float(y.var(ddof=1))
    out.update(s2_within=s2w, s2_total=s2t, n_rep_groups=int(len(rep)),
               rep_frac=float(rep.sum() / len(y)))
    out['ceiling'] = float(1.0 - s2w / s2t) if s2t > 0 else float('nan')

    # sigma2_within is estimated from (den) degrees of freedom, so the ceiling is
    # itself uncertain.  Its UPPER bound uses the LOWER bound of sigma2_within:
    #     s2w_lo = s2w * den / chi2(1-alpha, den)
    # Reporting capture against this bound means a ratio above one would require
    # the model to beat the noise floor *beyond the uncertainty of the ceiling*.
    try:
        from scipy.stats import chi2
        for alpha, key in ((0.05, 'ceiling_ub95'), (0.01, 'ceiling_ub99')):
            s2w_lo = s2w * den / float(chi2.ppf(1.0 - alpha, den))
            out[key] = float(1.0 - s2w_lo / s2t) if s2t > 0 else float('nan')
    except Exception:
        out['ceiling_ub95'] = out['ceiling_ub99'] = out['ceiling']
    return out

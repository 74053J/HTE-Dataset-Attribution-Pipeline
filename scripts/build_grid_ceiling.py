"""STEP 1 -- the grid, the ceilings and Table 1.

For each of the 13 datasets:
  * the condition key is every column of ``hte.params.CONDITION_COLS`` the mapping
    produced for this file -- no per-dataset choice (see scripts/protocol.py);
  * the noise ceiling on ALL rows and its cluster bootstrap SE over replicate
    condition groups (seed 42, 1000 resamples);
  * the 12-cell grid on the untouched 20 % test split;
  * the inner-validation selection (5 seeds: 42, 1, 2, 3, 4); the reported
    ``r2_best`` is the seed-42 selection's test R2, the five-seed mean is kept
    as ``r2_best_5seed``.

Writes
  results/grid13_v18.csv                    <- the 156-cell grid (12 cells x 13)
  results/table1_v18_final.csv              <- Table 1 (13 rows)
  outputs/grid_ceiling/ceilings_final.csv
  outputs/grid_ceiling/grid13_final.csv
  outputs/grid_ceiling/m3_final.csv         <- the five selection seeds, per dataset
  outputs/grid_ceiling/thread_env.json
  outputs/grid_ceiling/cache/<alias>.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import protocol as L  # noqa: E402  (pins OMP_NUM_THREADS before numpy)
from hte import params as P  # noqa: E402  (protocol puts the repo root on sys.path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def modal_winner(sel_rows):
    """Modal validation winner, tie-broken by its mean test R2."""
    keys = [(r["selected_model"], r["selected_encoding"]) for r in sel_rows]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    means = {k: float(np.mean([r["r2_test_selected"] for r in sel_rows
                               if (r["selected_model"], r["selected_encoding"]) == k]))
             for k in counts}
    best = sorted(counts, key=lambda k: (-counts[k], -means[k]))[0]
    return best[0], best[1], counts[best], means[best]


def finalize(rec):
    """(Re)derive the reported configuration / R2 from the per-seed selection rows.

    When the five seeds pick five different cells there is no mode; the reported
    configuration is then the seed-42 pick, and `winner_stability` says so.  The
    reported R2 is always the mean over the five seeds of the winner's test R2.
    """
    sel_rows = rec["sel_rows"]
    m, e, k, mtest = modal_winner(sel_rows)
    no_mode = k == 1
    if no_mode:
        s42 = [r for r in sel_rows if r["val_seed"] == 42][0]
        m, e = s42["selected_model"], s42["selected_encoding"]
    rec["selected_model"], rec["selected_encoding"] = m, e
    rec["winner_votes"] = int(k)
    rec["winner_stability"] = ("%d/5, no mode" % k) if no_mode else ("%d/5" % k)
    rec["r2_test_5seed"] = float(np.mean([r["r2_test_selected"] for r in sel_rows]))
    rec["r2_test_of_modal_winner"] = float(mtest)
    # The reported held-out R2 is the seed-42 selection's test R2 -- one
    # configuration chosen on one inner validation split, which is what Table 1
    # carries.  The five-seed mean is kept alongside as r2_test_5seed.
    s42 = [r for r in sel_rows if r["val_seed"] == 42][0]
    rec["r2_test"] = float(s42["r2_test_selected"])
    return rec


def _write_with_diff(frame: pd.DataFrame, path: Path, key: str) -> None:
    """Write ``frame`` to ``path``, keeping the previous copy and printing the diff.

    Two reasons this is not a plain ``to_csv``:
      * a re-run must never silently replace a number the manuscript quotes, so
        the previous file is kept next to it as ``<name>.previous.csv``;
      * the drift is printed per numeric column, which is how a thread-count or
        mapping change becomes visible instead of invisible.
    """
    if path.exists():
        prev = pd.read_csv(path, encoding="utf-8-sig")
        bak = path.with_name(path.name.replace(".csv", ".previous.csv"))
        prev.to_csv(bak, index=False, encoding="utf-8")
        common = [c for c in frame.columns
                  if c in prev.columns and pd.api.types.is_numeric_dtype(frame[c])]
        drifts = []
        for c in common:
            try:
                a = prev.set_index(key)[c]
                b = frame.set_index(key)[c]
                d = (b - a).abs().max()
            except Exception:
                continue
            if pd.notna(d) and d > 1e-9:
                drifts.append((c, float(d)))
        if drifts:
            print("  [diff] %s differs from the previous copy:" % path.name)
            for c, d in sorted(drifts, key=lambda t: -t[1]):
                print("           %-18s max|change| = %.3e" % (c, d))
        else:
            print("  [diff] %s is unchanged" % path.name)
    frame.to_csv(path, index=False, encoding="utf-8")


def main(force=False):
    t0 = time.perf_counter()
    L.CACHE.mkdir(parents=True, exist_ok=True)
    out = []

    for i, alias in enumerate(L.ALIASES13, 1):
        cache = L.CACHE / (alias + ".json")
        cached = None
        if cache.exists() and not force:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            # a cache written before the per-cell grid rows were recorded cannot
            # produce results/grid13_v18.csv, so it is treated as stale
            if "cell_rows" not in cached or "mean_yield" not in cached:
                cached = None
        if cached is not None:
            rec = finalize(cached)
            print("[%2d/13] %-18s cache hit (test max %.6f, selected %.6f)"
                  % (i, alias, rec["r2_test_max12"], rec["r2_test"]), flush=True)
        else:
            df, raw = L.load(alias)
            y = df["yield"].to_numpy(float)
            key, key_type, note = L.condition_key(alias, df)
            st = L.ceiling_stats(key, y)
            boot = L.boot_ceiling(key, y)
            # The key above must be the same grouping hte.ceiling groups on,
            # otherwise this table and the packaged definition would drift apart.
            from hte.ceiling import estimate_noise_ceiling
            _ref = estimate_noise_ceiling(df)
            _a, _b = st["ceiling"], _ref["ceiling"]
            _same = (np.isnan(_a) and np.isnan(_b)) or abs(_a - _b) < 1e-9
            if not (_same and st["n_cond"] == _ref["n_cond"]):
                raise SystemExit("ceiling definition mismatch on %s: %.9f/%d vs %.9f/%d"
                                 % (alias, _a, st["n_cond"], _b, _ref["n_cond"]))
            g = L.grid_and_select(df, raw)
            m, e, k, mtest = modal_winner(g["sel_rows"])
            mean_sel = float(np.mean([r["r2_test_selected"] for r in g["sel_rows"]]))
            seed42 = [r for r in g["sel_rows"] if r["val_seed"] == 42][0]
            cell_rows = [dict(cr, dataset=alias) for cr in g["cell_rows"]]
            rec = dict(
                alias=alias, n=int(len(df)), mean_yield=float(y.mean()),
                key_type=key_type, key_note=note,
                ceiling=float(st["ceiling"]), ceiling_se=float(boot["se"]),
                ceiling_lo=float(boot["lo"]), ceiling_hi=float(boot["hi"]),
                s2_within=float(st["s2_within"]), s2_total=float(st["s2_total"]),
                n_cond=int(st["n_cond"]), n_rep_groups=int(st["n_rep_groups"]),
                n_rep_rows=int(st["n_rep_rows"]), rep_frac=float(st["rep_frac"]),
                df_within=float(st["df_within"]),
                n_test=int(len(g["te"])), n_train=int(len(g["tr"])),
                selected_model=m, selected_encoding=e, winner_votes=int(k),
                winner_stability="%d/5" % k,
                r2_test=mean_sel, r2_test_of_modal_winner=float(mtest),
                r2_test_seed42=float(seed42["r2_test_selected"]),
                sel42_model=seed42["selected_model"],
                sel42_encoding=seed42["selected_encoding"],
                r2_test_max12=float(g["r2_test"][g["best"]]),
                max12_model=g["cells"][g["best"]].architecture,
                max12_encoding=g["cells"][g["best"]].encoding,
                sel_rows=g["sel_rows"],
                cell_rows=cell_rows,
                r2_test_by_cell=[float(v) for v in g["r2_test"]],
                cell_names=list(g["names"]),
            )
            rec = finalize(rec)
            cache.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            print("[%2d/13] %-18s key=%-15s C=%.6f (se %.4f)  sel=%s/%s %s  "
                  "r2=%.6f (seed42 %.6f, max12 %.6f)  [%.0fs]"
                  % (i, alias, key_type, rec["ceiling"], rec["ceiling_se"],
                     rec["selected_model"], rec["selected_encoding"],
                     rec["winner_stability"], rec["r2_test"], rec["r2_test_seed42"],
                     rec["r2_test_max12"], time.perf_counter() - t0), flush=True)
        rec["capture"] = (rec["r2_test"] / rec["ceiling"]
                          if np.isfinite(rec["ceiling"]) and rec["ceiling"] > 0 else np.nan)
        rec["capture_old"] = np.nan
        out.append(rec)

    ce = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("sel_rows", "cell_rows", "r2_test_by_cell",
                                     "cell_names")}
                       for r in out])
    ce.to_csv(L.OUT / "ceilings_final.csv", index=False, encoding="utf-8")

    # ------------------------------------------------------------------
    # results/table1_v18_final.csv -- the schema the rest of the package reads
    # (verify_numbers.py, figures.py, figures_paper2.py, build_enrichment.py):
    #   dataset, n_rows, mean_yield, best_model, best_encoding, best_config,
    #   r2_best, ceiling, ceiling_se, capture, n_cond, n_rep_groups, rep_frac
    # The diagnostic columns come after, so a plain read of the first thirteen
    # reproduces the shipped table.
    # ------------------------------------------------------------------
    rows = []
    for r in out:
        rows.append(dict(
            dataset=r["alias"], n_rows=r["n"], mean_yield=r["mean_yield"],
            best_model=r["selected_model"], best_encoding=r["selected_encoding"],
            best_config="%s / %s" % (r["selected_model"], r["selected_encoding"]),
            r2_best=r["r2_test"], ceiling=r["ceiling"], ceiling_se=r["ceiling_se"],
            capture=r["capture"], n_cond=r["n_cond"], n_rep_groups=r["n_rep_groups"],
            rep_frac=r["rep_frac"],
            key_type=r["key_type"], ceiling_lo=r["ceiling_lo"], ceiling_hi=r["ceiling_hi"],
            r2_best_5seed=r["r2_test_5seed"], r2_best_seed42=r["r2_test_seed42"],
            selected_seed42_model=r["sel42_model"],
            selected_seed42_encoding=r["sel42_encoding"],
            winner_stability=r["winner_stability"],
            r2_test_max12=r["r2_test_max12"],
            max12_model=r["max12_model"], max12_encoding=r["max12_encoding"],
        ))
    tab = pd.DataFrame(rows)
    _write_with_diff(tab, L.RESULTS / "table1_v18_final.csv", "dataset")

    grid_rows, m3_rows = [], []
    for r in out:
        for cr in r["cell_rows"]:
            grid_rows.append(cr)
        for s in r["sel_rows"]:
            m3_rows.append(dict(dataset=r["alias"], **s))
    grid = pd.DataFrame(grid_rows)
    grid_cols = ["dataset", "n_rows", "model", "encoding", "n_features",
                 "r2_train", "r2_test", "rmse_train", "rmse_test",
                 "fit_seconds", "status", "threads"]
    grid = grid[[c for c in grid_cols if c in grid.columns]]
    _write_with_diff(grid, L.RESULTS / "grid13_v18.csv", ["dataset", "model", "encoding"])
    grid.to_csv(L.OUT / "grid13_final.csv", index=False, encoding="utf-8")
    pd.DataFrame(m3_rows).to_csv(L.OUT / "m3_final.csv", index=False, encoding="utf-8")

    # results/dataset_index.csv -- alias / source stem / row count, i.e. the
    # table every other script needs to find a file.  Regenerated from the same
    # pass, so it can never drift from what was actually loaded.
    _stems = L.stems()
    idx = pd.DataFrame([dict(dataset=r["alias"], source_stem=_stems[r["alias"]],
                             n_rows=r["n"]) for r in out])
    idx.to_csv(L.RESULTS / "dataset_index.csv", index=False, encoding="utf-8")

    import matplotlib
    import scipy
    import sklearn
    import xgboost
    env = dict(
        # what actually governed the fits -- the pools, not the request
        effective_pools={k: os.environ.get(k) for k in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                          "DSH_SWEEP_THREADS")},
        threads_requested=L.N_THREADS,
        pinned_by_cd_threads=L.PINNED_THREADS,
        cd_threads=os.environ.get("CD_THREADS"),
        rf_n_jobs=(int(P.RF_N_JOBS) if str(P.RF_N_JOBS).lstrip('-').isdigit()
                   else P.RF_N_JOBS),
        cpu_logical=os.cpu_count(),
        numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
        xgboost=xgboost.__version__, scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,
        python=sys.version.split()[0],
        val_seeds=L.VAL_SEEDS, val_test_size=L.VAL_TEST_SIZE,
        boot_resamples=L.N_BOOT, boot_seed=L.SEED_BOOT,
        elapsed_seconds=round(time.perf_counter() - t0, 1),
    )
    (L.OUT / "thread_env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("\n[step1] %d datasets in %.0fs at %s threads (pools: OMP=%s)"
          % (len(out), time.perf_counter() - t0,
             env["threads_requested"] if env["pinned_by_cd_threads"] else "unpinned",
             os.environ.get("OMP_NUM_THREADS", "unset")))
    print(tab[["dataset", "mean_yield", "ceiling", "ceiling_se", "r2_best",
               "capture", "n_cond", "key_type"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(force="--force" in sys.argv))

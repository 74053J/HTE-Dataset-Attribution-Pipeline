# Yield prediction and condition recommendation on HTE datasets

**[中文说明](README.zh-CN.md)** · English

Code and result tables for the manuscript submitted to *Chemometrics and Intelligent
Laboratory Systems*.

## What is here

| path | content |
|---|---|
| `hte/` | core package: dataset loading, slot mapping, feature construction, models |
| `archive/single_dataset_full_benchmark.py` | the second pipeline path: single-dataset, end-to-end (see the last section) |
| `scripts/figures.py` | renders every manuscript figure (450 dpi, Times New Roman) |
| `scripts/figures_paper2.py`, `scripts/tables.py` | composite panels and result tables |
| `results/` | **generated** by `run_all.bat` (Step 1-11); the shipped package does not carry it |
| `outputs/attrib13/` | **generated** by Step 9 (13 datasets x 12 cells) |
| `outputs/c20_transfer_4donor/` | **generated** by Step 4 (48 cross-dataset transfer cells) |
| `scripts/verify_numbers.py` | recomputes every number quoted in the manuscript from the shipped tables |
| `docs/`, `data/` | method notes and the slot-mapping CSV |
| `tests/` | unit tests |
| `archive/single_dataset_full_benchmark.py` | the single-dataset pipeline path -- independent of the manuscript, documented at the end |
| `Single_dataset_full_benchmark.bat` | one-click launcher for that pipeline |

## Reproduce

```bash
pip install -r requirements.txt
python scripts/figures.py            # all manuscript figures
python scripts/build_attrib13.py --tag A --per-dataset   # attribution table
python scripts/tables.py             # result tables
python scripts/verify_numbers.py     # check every manuscript number
python tests/run_equivalence.py      # package-vs-pipeline equivalence tests
```

## Data

Raw datasets are not redistributed: `fetch_datasets.bat` downloads each one from its
official record and prepares it (see `LICENSE-DATA`). Two tables in `data/` carry every per-file
decision, so no script names a dataset:

* `col_mappings.csv` — raw column → slot, plus per-file overrides (`(none)`
  switches a slot off for one file);
* `alias_mapping.yaml` — alias → file stem, plus `corpus:`, `ceiling:` and
  `donor:` flags.

To add a dataset: drop the CSV in `data/`, add its alias to
`alias_mapping.yaml`, add its columns (or overrides) to `col_mappings.csv`.

## Licence

Code: MIT. Our result tables and the two mapping tables in `data/` are CC BY 4.0.
No third-party data is redistributed: run `fetch_datasets.bat` to obtain it.
Each dataset keeps the licence of its original record; `LICENSE-DATA` carries the
per-source table, the evidence for every entry and the preparation each one needs.

## Scope of the R2 range

`0.416-0.848` covers the eleven datasets for which a noise ceiling can be computed.
Over all thirteen datasets the range is `0.416-0.868`: the two extra files
(`rxn_yields_SM`, `Summit_CN`) have no ceiling and therefore do not enter the capture ratio.

## Verifying the numbers

`python scripts/verify_numbers.py` recomputes sixty-seven figures from the tables in `results/`
and compares them with the values printed in the manuscript; it exits non-zero on any
mismatch. Tables from an earlier pipeline revision are deliberately not shipped: keeping
two number sets side by side would let a reader reproduce values the paper does not quote.

## Reproducing the published tables

`python scripts/build_grid_ceiling.py` rebuilds `results/grid13_v18.csv` (156 cells),
`results/table1_v18_final.csv` (Table 1) and `results/dataset_index.csv` from `data/*.csv`.
Measured against the copies shipped with the manuscript (2026-10-27 run, 4 threads):

| quantity | reproduced |
|---|---|
| `n_features` (156 cells) | 156 / 156 exact |
| `n_rows`, `mean_yield` (13 datasets) | 13 / 13 exact |
| noise ceiling `R2max` (11 datasets) | 11 / 11 exact (< 1e-15) |
| `n_cond`, `n_rep_groups`, `rep_frac` | 13 / 13 exact |
| `r2_test` (156 cells) | 120 bit-identical, 149 within 1e-3 |
| `r2_best` (13 datasets) | 11 / 13 exact |

**Threads.** Everything here is a four-thread result: the frozen pipeline pins four
threads in its header, and four is what reproduces the shipped grid (120 of 156 cells
bit-identical; at eight threads only 105, and XGBoost 0 of 39). `scripts/protocol.py`
pins four by default (`CD_THREADS=N` to change, `CD_THREADS=0` to leave the pools alone).
The one exception is `results/protocol_cv.csv`, which is a **32-thread** table: the file
records `n_jobs = 32` on all three protocol rows, and `scripts/build_protocols.py` pins 32
itself. The header of the frozen pipeline claims the thread count changes no reported
number; that is **not true for XGBoost and the MLP**, as the 4-vs-8 row above shows.

**The residual is the feature matrix, not the machine.** All 18 non-identical cells sit
on the two AI_RCP files, and they include the thread-*invariant* models (Random Forest
4.9e-4, Gradient Boosting 3.0e-4), so the thread count cannot explain them; sweeping
`n_jobs` from 1 to 32 never reproduces that dataset's XGBoost cell either. The grid was
written on 2026-10-25 16:11, before the equivalents/slot rework of 19:03-20:53, so it was
fitted on a feature matrix that no longer exists in the repository; today's 13-column set
and the 9-column pre-equivalents set were both tried and neither reproduces it. That is
also why `results/protocol_cv.csv` (2026-10-22), `results/enrichment13.csv` (2026-10-24)
and `outputs/c20_transfer_4donor/transfer_matrix.csv` (2026-10-24) reproduce their
protocols but not their numbers, while the attribution table written at 21:13 reproduces
exactly. `docs/NUMBERS_PROVENANCE.md` lists every artifact with its date and verdict.

`scripts/build_enrichment.py`, `scripts/build_transfer.py` and
`scripts/build_protocols.py` print their own numbers; the shipped tables keep the
manuscript values and `scripts/verify_numbers.py` checks those.

---

## Single-dataset audit pipeline (separate from the manuscript)

`archive/single_dataset_full_benchmark.py` is a second, **self-contained pipeline path**. It is
independent of the paper's tables: it takes ONE dataset and produces its own evidence pack
(figures + CSV, ~60 files per run) under `output_singledataset/<timestamp>_<dataset>/`.
Nothing here is required to reproduce the manuscript, and nothing it writes is quoted in it.

```bat
Single_dataset_full_benchmark.bat "(Summit)Reizman_Suzuki_385.csv"
```

What one run gives you:

* **12-cell benchmark** -- every (architecture x encoding) cell trained and ranked by held-out
  R2 on this dataset only, printed as a numbered table;
* **interactive selection** (seven questions, every one defaulting on Enter) deciding which
  cells are drawn and which members form the ensemble -- the selection drives Figures 2-4 and
  7-8 *and* the Bayesian ensemble, so the two cannot disagree;
* **attribution** -- permutation importance (train vs held-out), SHAP dependence and
  interaction panels, per member;
* **virtual-library ranking** -- exhaustive enumeration over the reagent axes, with every
  dose axis expanded to `DOSE_POINTS` (default 10) points, ranked by expected improvement;
* **loading-response curves** as a PDF, one curve per top candidate;
* **self-auditing output** -- noise ceiling, calibration slope, held-out coverage and an
  explicit warning when the condition audit itself is untrustworthy.

It shares the `hte/` package (slots, grid, models, features) with the manuscript pipeline but
has its own entry point, its own outputs and no effect on the published numbers.

### What the run writes

One run creates **about 60 files** in `output_singledataset/<YYYYMMDD_HHMMSS>_<dataset>/`.
Naming is uniform -- `Figure<N>_<name>[-<Arch (encoding)>].<png|csv|pdf>` -- so the file name
alone tells you which figure it is and which (model x encoding) cell produced it.

| file pattern | count | content |
|---|---|---|
| `Figure1_Model_Comparison.png` | 1 | R2 (with the noise-ceiling line), RMSE, cumulative error, 5-fold CV |
| `Figure2_ActiveLearning_ResponseSurface- <cell>.png` | one per selected cell | learning / response surface for that cell |
| `Figure3_Model_Diagnostics_LearningCurve_Residuals- <cell>.png` | same | learning curve and residual diagnostics |
| `Figure4_Feature_Importance_Attribution- <cell>.png` + `_panel_data.csv` | same | permutation importance, sensitivity, train vs held-out, predicted-vs-measured |
| `Figure5_Comprehensive_Substrate_and_Solvent_Analysis.png` + `_panel_data.csv` | 1 | violin distributions, two interaction heatmaps, SMILES PCA |
| `Figure6_Direct_Ensemble_Ranking.png` + `_top500.csv` + `_enrichment_audit.csv` | 3 | ensemble ranking of the real reactions, Top-20 table, enrichment audit |
| `Figure7_Deep_SHAP_Attribution- <cell>.png` + `_panel_data.csv` | one per selected cell | SHAP dependence, SHAP interaction, force analysis, global attribution |
| `Figure8_Partial_Dependence_Plots- <cell>.png` + `_panel_data.csv` | same | three partial-dependence curves |
| `Figure9_Bayesian_Optimization-Ensemble.png` + `_panel_data.csv` | 2 | convergence, predicted heatmap, second interaction, loading optimisation |
| `Figure10_Top20_Predicted_Conditions-Ensemble.png/.csv` | 2 | Top-20 by predicted yield (member predictions and CI included) |
| `Figure10b_Top20_ByExpectedImprovement-Ensemble.png/.csv` | 2 | Top-20 by **expected improvement** (a different question from 10a) |
| `Figure11_Loading_Curves_Top<N>.pdf/.csv` | 2 | one loading-response curve per top candidate (vector PDF) |
| `grid12_metrics.csv` | 1 | **all twelve cells**: model, encoding, frame, n_features, train/test R2, RMSE, fit time, status |
| `model_evaluation_metrics.csv` | 1 | CV / learning-curve / permutation summary for the selected members |
| `model_predictions_all.csv` | 1 | row-level predictions (fitted and held-out) for further analysis |
| `timings.csv` | 1 | per-stage timings (fit, SHAP, PDP, sweep, ...) |
| `choices.json` | 1 | **the seven answers of this run** -- replays the same figure set |

> Two notes: (a) every `_panel_data.csv` holds the numbers the panel was drawn from, so you can
> check a value without reading it off the pixels; (b) `choices.json` replays the run -- point
> `CD_CHOICES` at it and you get the same figures again.

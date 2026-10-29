# Where each number comes from, and what still moves

> **STATUS (2026-10-29): historical record.**  The tables below were captured during the
> manuscript runs and are kept unchanged for auditability.  They are **superseded** by the
> full re-run of 2026-10-29 on `xgboost-cpu 3.2.0`: see `COMPARISON_vs_paper_20261029.md`
> for the current agreement (grid: 120/156 cells bit-identical, 149/156 within 1e-3;
> gate: 53 of 63 checks pass).  Where a number here disagrees with `results/`, the value in
> `results/` is the current one.

Written 2026-10-27 after tracing every result table back to the file that wrote it
(file modification times + the producing script), then rebuilding each one from
`data/*.csv`.

## 1. Data pool

| Source | Upstream release | File in `data/` | How it was obtained |
|---|---|---|---|
| AI_RCP | Zenodo 10.5281/zenodo.18185850 (SURF format) | `(AI_RCP_pub)bh_data_clean_all_whitelisted.csv`, `…sm…` | copied unchanged from the `AI_RCP_pub` repository's `data/` |
| bh-hte-ood | Neves et al., *Nat. Comput. Sci.* 2026 | `(bh-hte-ood)BH_HTE_Curated_Dataset_v20260219.csv` | copied unchanged from the `bh-hte-ood` repository |
| Roche66k (8 files) | Ahlbrecht et al., *ACS Cent. Sci.* 2026 + `zScore-App` parquet | `(Roche66k)*.csv` | parquet → CSV; column names converted at that step; the BH and SM subsets drop rows without a parsed temperature/time (71 % of rows have them) |
| Summit | `sustainable-processes/summit` | `(Summit)Reizman_Suzuki_385.csv`, `(Summit)Baumgartner_CN_coupling_96.csv` | the four Reizman cases merged into 385 rows |
| rxn_yields | Perera et al., *Science* 2018, Data File S1 | `(rxn_yields)aap9112_Data_File_S1.csv` | xlsx → CSV |

Every file is traceable to a public release; no value was edited by hand. The filtering
rules (the Roche temperature/time filter, the Reizman merge) are recorded here and in
Table A.1 of the manuscript, not in the loader.

## 2. Result tables

| Table | Written | By | Reads |
|---|---|---|---|
| `outputs/c20_transfer_4donor/transfer_matrix.csv` (48 cells) | 2026-10-24 14:42 | `_work/v234/c20_transfer_4donor.py` | v15 pipeline, donor encoders fixed on the source, unseen level → `len(classes_)`, constant baseline = donor train mean |
| `results/enrichment13.csv` | 2026-10-24 14:59 | `(scratch script, outside this package) 10b_enrichment_v18.py` | each dataset's Table-1 winner, `KFold(5, shuffle, seed 42)` over **all** rows, top-k measured yield / whole-dataset mean |
| `results/table1_v18_final.csv` | 2026-10-24 16:17 | `_work/v237/ceil18.py` (+ `update_table1.py`) | `estimate_noise_ceiling(df)` with `params.CONDITION_COLS ∩ df.columns`; R2 from the validation-selected cell |
| `results/perm_vs_shap_cells13.csv` | 2026-10-25 14:05 | derived from the attribution table | — |
| `results/grid13_v18.csv` (156 cells) | 2026-10-25 16:11 | 12-cell grid on the then-current pipeline | — |
| `results/rebuilt_attrib12_long_v7.csv` | 2026-10-25 21:13 | `_work/v247/compute_new_numbers.py` | 12 cells × 13 datasets, SHAP + permutation |

Between 19:03 and 20:53 on 2026-10-25 the equivalents mapping and the slot names were
reworked (`eq_map.py`, `eq_mirror.py`, `write_v6.py`, `rename_slots.py`, `fix_params.py`).
Artifacts written before that window and after it therefore do not have to come from the
same pipeline state — which is what the rebuild below measures.

## 3. Rebuild status (this package, from `data/*.csv`)

| Artifact | Status |
|---|---|
| Table 1 ceilings (11) | **exact**, `< 1e-15`; `n_cond` / `n_rep_groups` / `rep_frac` also 13/13 exact |
| Table 1 `r2_best` (13) | 11 exact; `AI_RCP_BH` +0.0058, `Roche66k_CN` -0.0079 |
| Grid `n_features` (156) | **exact** |
| Grid `r2_test` (156) | 138 bit-identical, 148 within 1e-3, all 18 non-identical cells on `AI_RCP_BH` / `AI_RCP_SM` (max 0.026) |
| Attribution (1440 rows) | **exact** (SHAP identical, permutation 2e-15) |
| perm-vs-SHAP (156 cells) | exact (median ρ 0.9273 both) |
| Protocol CV (3 rows, Fig. 7) | `scripts/build_protocols.py` reproduces the *protocol*; numbers differ: random 0.7560 vs 0.7687, substrate-disjoint -0.1477 vs -0.1827, ordered -1.0698 vs -1.1743 |
| §3.6 unseen pairs (`scripts/build_novel_pairs.py`) | 3 of 5 arms **exact**: global -0.007±0.004, additive 0.126±0.159, reactant-1 mean 0.368±0.083; 5,752 pairs and 1,311 held-out pairs exact; XGBoost arms +0.007 / +0.008 |
| §3.6 unseen reactants (`scripts/build_unseen_reactants.py`) | **213 unseen halides exact**; ridge --- conditions+identity 0.262 vs 0.250, +Morgan 0.410 vs 0.389 (ρ 0.538 vs 0.56); without the metal column 0.233 / 0.420 (ρ 0.559 vs 0.56) |
| Table 2 (`scripts/build_table2.py`, ported 2026-10-27) | arms 1/2/3 reproduce within 1.7e-3 on `AI_RCP_SM` and 1.3e-2 on `AI_RCP_BH`; `DoyleHTE_CN` reproduces after one mapping row was added (0.939 / 0.951 against 0.944 / 0.949); `bh-hte-ood_BH` does not (0.661 vs 0.269) -- its shipped value is the 2026-10-09 batch number and its cell is HistGB, not the current winner |
| Enrichment EF@10/50/1000 | EF@50 differs by ≤0.16 (largest: `Summit_CN` 1.538 → 1.378) |
| Transfer (48 cells) | 48/48 still negative; per-donor means -1.72 / -0.64 / -1.05 / -1.07 against -1.119 / -0.772 / -1.185 / -1.029 |

## 3a. Threads: four, except one table

The frozen pipeline pins **four** threads in its header (`DSH_SWEEP_THREADS='4'`,
`Analysis_all_v16.py` lines 30-34) and claims there that the count "does NOT change any
number the pipeline reports". That claim is **false for XGBoost and the MLP** -- measured
on the shipped 156-cell grid:

| pools | cells bit-identical | cells within 1e-3 | XGBoost bit-identical | Table-1 `r2_best` exact |
|---|---|---|---|---|
| **4 threads** | **138 / 156** | **148 / 156** | **33 / 39** | **11 / 13** |
| 8 threads | 105 / 156 | 121 / 156 | 0 / 39 | 5 / 13 |

So four threads is the right setting and `scripts/protocol.py` now pins it by default
(`CD_THREADS=N` overrides, `CD_THREADS=0` leaves the pools alone). One table is an
**eight-thread** table: `results/protocol_cv.csv` stores `n_jobs = 8` on every row, and
its producer even records the difference ("0.7661 at 4 threads vs the recorded 0.7687 at
8"). `scripts/build_protocols.py` therefore pins 8 of its own accord.

Sweeping `n_jobs` from 1 to 32 never reproduces the shipped `AI_RCP_BH` XGBoost cell
(0.73624 … 0.74657, no value equal to 0.7408080), and **Random Forest is
thread-invariant** yet misses the same dataset by 4.9e-4. Threads are therefore not the
explanation for the residual below -- the feature set is (see 3b).

## 3b. Why the model-fit tables do not reproduce exactly

Every quantity computed from the **condition slots** reproduces exactly (ceilings,
`n_cond`, `n_features`, `n_rows`, and the attribution table written on 2026-10-25 21:13).
Every quantity computed from a **fitted model** before that date is slightly off, and the
split is temporal:

| artifact | written | reproduces |
|---|---|---|
| protocol CV | 2026-10-22 21:42 | no |
| transfer matrix | 2026-10-24 14:42 | no |
| enrichment | 2026-10-24 14:59 | no |
| Table 1 `r2_best` | 2026-10-24 16:17 | 11/13 |
| grid (156 cells) | 2026-10-25 16:11 | 138/156 |
| attribution (1440 rows) | 2026-10-25 21:13 | **yes** |

Between 2026-10-25 19:03 and 20:53 the equivalents mapping and the slot names were
reworked (`eq_map.py`, `eq_mirror.py`, `write_v6.py`, `rename_slots.py`, `fix_params.py`),
which is when the seven `*_eq` columns entered the feature matrix. Anything fitted before
that work therefore used a feature matrix that no longer exists in the repository.

What is **proven** is that the residual is feature-level, not machine-level: the
thread-*invariant* estimators (Random Forest 4.9e-4, Gradient Boosting 3.0e-4) miss the
same two files, and no `n_jobs` from 1 to 32 reproduces their XGBoost cell. What is
**not** proven is the exact pre-rework matrix: today's 13-column set and the 9-column
pre-equivalents set were both tried on the protocol CV and neither reproduces the shipped
numbers (13 cols: 0.7560 / -0.1477 / -1.0698; 9 cols: 0.7602 / -0.2270 / -1.7713;
shipped: 0.7687 / -0.1827 / -1.1743). The 2026-10-25 16:11 state was not preserved, so it
cannot be reconstructed from what remains.

**Exception: §3.6 reproduces.** The two scripts ported from `_work/v248` write their own
tables from `data/*.csv` and land on the published numbers -- three of five arms exactly on
the unseen-pairs table, the 213-unseen-halide count exactly on the unseen-reactant table,
and the pair-level correlation to two decimals. §3.6 was fitted on 2026-10-26, after the
rework, which is why it is the one section whose fitted numbers survive.

### Why the grid moves (not a mapping problem)

The feature matrix this package builds for `AI_RCP_BH` is bit-identical to the one the
monolithic script builds: same rows, same split, same target, same resolved columns,
every condition slot equal. What moves is the estimator fit. Measured on one cell
(XGBoost, native categorical, `AI_RCP_BH`):

| threads | 1 | 4 | 8 | 16 | 32 | table |
|---|---|---|---|---|---|---|
| R2 | 0.73926 | 0.74657 | 0.74371 | 0.74131 | 0.74372 | **0.74081** |

The original run's thread state was not recorded, so the cell is not reproducible
bit-for-bit. `CD_THREADS=N` pins every pool when a run has to repeat exactly.

### Why the enrichment and transfer tables move

Both were written on 2026-10-24, before the equivalents rework. The protocols are
reproduced exactly by `scripts/build_enrichment.py` and `scripts/build_transfer.py`
(both read their settings from `results/table1_v18_final.csv` and the mapping table, no
per-dataset branch); what differs is the feature set the older pipeline handed the
model. `Summit_CN` is the extreme case: 96 rows, five feature columns then, six now
(the loading axis is written to `catalyst_eq`, which is a model feature, where the older
snapshot wrote it to `catalyst_loading`, which is not).

## 4. Decisions (closed 2026-10-27)

The author ruled that every run uses **four threads** (including the random-split
protocol CV) and that the rebuilt numbers are authoritative.  Everything below was
therefore regenerated at four threads and the manuscript numbers were updated to match
(v3.22 -> v3.23):

| quantity | was | now |
|---|---|---|
| capture mean | 86.21 % | **86.18 %** |
| Table 1 `AI_RCP_BH` R2 / capture | 0.741 / 84.2 | **0.747 / 84.8** |
| Table 1 `Roche66k_CN` R2 / capture | 0.587 / 72.2 | **0.579 / 71.3** |
| enrichment k=50 span | 1.54-6.83 | **1.38-6.72** |
| enrichment top-10 / top-50 (case study) | 84.5 % / 90.4 % | **83.3 % / 88.8 %** |
| transfer range | -2.38 .. -0.08 | **-4.94 .. -0.10** |
| transfer per-donor means | -1.119 / -0.772 / -1.185 / -1.029 | **-1.673 / -0.515 / -1.070 / -1.224** |
| transfer affine r2 / MAE | 0.032 / 31.8 | **0.036 / 33.9** |
| protocol CV random / substrate / ordered | +0.7687 / -0.183 / -1.174 | **+0.7572 / -0.130 / -1.050** |
| random-vs-substrate drop | 0.949 | **0.887** |

`scripts/verify_numbers.py` now checks the new values and passes all 40 checks.  The
bootstrap interval quoted beside the capture mean (79.0-93.2 %) was not recomputed: the
mean moved by 0.03 percentage points, so the interval is unchanged to the quoted precision.


## 5. Known differences from the V1 (metal/solvent) pipeline

Two capabilities exist only in an audit tree kept outside this package
that this package does not use.  Both would move the condition key, and neither is
switched on, so every number in this package is computed with the behaviour below.

### 5.1 Metal is read from a column, with an optional reader

An independent audit (kept outside this package) reads the catalytic metal per row from two sources and
reconciles them: the catalyst SMILES (RDKit atomic numbers) and the catalyst name
(after stripping ligand stems, allowing formula concatenation such as
``dtbpfPdCl2``, and mapping spelled-out names such as ``Copper(II) acetate``).
``hte/metal_reading.py`` is that reader, ported without any dataset name; it
reproduces the V1 figures on the four files that carry a metal signal
(Cu 58 % / 49 % / Ru 17 % / Cu 9 %, and the 27 nickel rows of bh-hte-ood).

It is behind ``CD_METAL_READER=1`` and **off by default**, because the datasets
that carry a metal *column* already hold the right values (the column reports
58 % copper on Roche66k CN, exactly what the reader reports), and the reader is
only needed where the column is missing.  Enabling it changes one dataset:
bh-hte-ood's metal slot goes from a constant to three values (Pd 27468, Ni 27,
Unknown 2), which refines its condition key and therefore its noise ceiling.
The packaged ceiling and every table derived from it assume the column behaviour.

### 5.2 The condition key is the raw value, not a normalised one

V1 normalises the key before grouping: ``split_multi`` splits multi-component
cells (``THF/H2O``, ``Na2CO3 + K2CO3``), ``canon_solvent`` merges spellings of the
same solvent, and the co-solvent is split out of the solvent string.  This package
groups on the raw mapped value instead, so two spellings of one solvent count as
two conditions and a mixture counts as one.  That difference moves ``n_cond`` --
and with it the ceiling -- on any dataset whose condition cells are not already
single-valued and consistently spelled.

Porting it is a separate, self-contained change: it would change the ceilings of
several datasets at once and require re-running every table and every gate value,
so it is recorded here rather than switched on silently.

## 6. Metal distribution (authoritative, after the V1 reader was switched on)

`hte/metal_reading.py` reads the catalytic metal from the catalyst columns instead of
assuming palladium; it is ON by default and `CD_METAL_READER=0` restores the old
constant-Pd behaviour for reproducing the pre-2026-10-27 tables.  The distribution
below is the one the package now produces; it matches that independent audit
(an audit CSV kept outside this package) row for row.

| dataset | rows | Pd | Cu | Ru | Ni | blank |
|---|---|---|---|---|---|---|
| AI_RCP_BH | 10,138 | 10,138 | - | - | - | 0 |
| AI_RCP_SM | 3,426 | 3,426 | - | - | - | 0 |
| bh-hte-ood_BH | 27,497 | 27,470 | - | - | 27 | 0 |
| Roche66k_BH | 14,191 | 12,414 | 1,296 | - | - | 481 |
| Roche66k_SM | 9,946 | 9,946 | - | - | - | 0 |
| Roche66k_Amide | 3,917 | - | - | - | - | 3,917 |
| Roche66k_CN | 3,100 | 1,181 | 1,810 | - | - | 109 |
| Roche66k_CO | 2,112 | 1,047 | 1,037 | - | 3 | 25 |
| Roche66k_Cond | 2,095 | - | - | - | - | 2,095 |
| Roche66k_CH | 1,251 | 1,036 | - | 208 | - | 7 |
| rxn_yields_SM | 5,760 | 5,760 | - | - | - | 0 |
| Summit_SM | 385 | - | - | - | - | 385 |
| Summit_CN | 96 | - | - | - | - | 96 |
| **total** | **83,914** | **72,418** | **4,143** | **208** | **30** | **7,115** |

Blank rows are either a file with no metal chemistry at all (Amide 3,917, Cond 2,095 -
condensation reagents), a plate whose catalyst cell is empty (BH 470, CN 109, CO 25,
CH 7), a plate that records only inert codes (Summit_SM 385), or a cell that names a
ligand without its metal (Summit_CN 96).  Two further changes are known but NOT applied,
pending a decision: CX31 is a palladium precatalyst (Umicore CAT 709042, CAS
884879-23-6) and would move the 11 Buchwald rows from blank to Pd, and the 96
Summit_CN ligand-only cells would become Pd; together they would give Pd 72,525 and
blank 7,008.

### 6.1 What switching the reader on changed

Every table was rebuilt at that point (grid and ceiling at 4 threads, the protocol table
at 32).  Against the values the manuscript carried before the switch:

| quantity | before | after |
|---|---|---|
| capture mean | 86.18 % | **86.22 %** |
| AI_RCP_BH R2 / capture | 0.747 / 84.8 % | **0.743 / 84.4 %** |
| enrichment EF@50 span | 1.378-6.716 | **1.378-6.872** |
| case-study top-10 / top-50 yield | 83.3 % / 88.8 % | **93.9 % / 90.9 %** |
| transfer AI_RCP_BH mean | -1.673 | **-1.356** |
| transfer pooled / range | -1.121 / -4.94..-0.10 | **-1.049 / -4.94..-0.00** |
| protocol random / substrate / ordered | +0.7560 / -0.190 / -1.113 | **+0.7579 / -0.151 / -0.799** |
| random-minus-substrate drop | 0.946 | **0.909** |

The other eleven datasets are unchanged to the fourth decimal.  These new values are
NOT yet written into the manuscript: the change set is awaiting sign-off, and
`scripts/verify_numbers.py` still pins the previous ones, so the gate reports a failure
until it is re-locked.


## 7. Changes after v3.32 (2026-10-28)

| manuscript | what changed | new value | source |
|---|---|---|---|
| v3.33 | section 3.3 recomputed from the regenerated attribution table | reactant_1 17.30 / reactant_2 **16.69** / temperature **11.21** / base **10.56** / ligand **8.66** / catalyst-equivalent **2.85**; first-place **49/48/29/11** (**88 %**); encoding agreement **0.850**; L1 **26.1**; permutation-vs-SHAP median **0.92** | `results/rebuilt_attrib12_long_v7.csv` (regenerated by `build_attrib13.py` + `build_attrib_postprocess.py`) |
| v3.34 | section 3.4 numbers, because the acquisition vectors were regenerated | top predicted **84.2 %**, first EI pick **56.3 %** (rank **21,772**), sd **44.6**, library median sd **8.9** | `outputs/acquisition/acq_mu.npy`, `acq_sd.npy` (`build_acquisition.py`) |
| v3.35 | section 3.4 conclusion, after the expected-improvement loop was corrected | EI **71.9 %** at 100 picks vs ranking **84.2 %** and random **55.5 %**; "clearly worse than ranking, still better than random search" | `results/bo_panel_data.csv` (`build_bo_panel.py`) |
| v3.36-v3.38 | all twelve figures re-rendered and synced into the document | 12/12 images verified by SHA-256 | `figures/*.png`, `scripts/sync_figures_to_docx.py` |

Capature/ceiling figures, the transfer bar (-1.356) and the protocol bars did not change in
this span; `scripts/verify_numbers.py` still reproduces all forty of its checks.

### v3.43 -- the held-out paragraph was rewritten

The paragraph after the Bayesian result used to report numbers from a scratch run of an older
pipeline (`_work/calib_interval.py`, which imports `Analysis_all_v13`); no script in the package
could reproduce them.  It is now produced by `scripts/build_heldout_acquisition.py` on the plain
80/20 test set of bh-hte-ood_BH, with the same panel-spread definition as Figure 8.

| quantity | was (unreproducible) | now (panel spread) |
|---|---|---|
| held-out conditions | "1,311 pairs" | 5,500 rows / 1,852 pairs |
| best measured yield, first pick | 93.0 % | 95.0 % (ranking and EI), 90.0 % (UCB) |
| random, first pick | 11.6 % | 12.3 % |
| ceiling by the k-th pick | 98.0 % at k = 50 | 105.0 % at k = 10 |
| rho(spread, abs error) | 0.561 | 0.115 |
| median error, lowest / highest quintile | 1.75 / 17.05 | 13.07 / 13.01 |
| coverage of a nominal 90 % interval | 15.4 % | 47.3 % |

The claim changed with the numbers: on this panel the spread is set by disagreement between
architectures, so it is neither a reliability score nor a predictive interval.

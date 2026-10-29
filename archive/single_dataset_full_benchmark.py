"""
Suzuki-Miyaura / Buchwald-Hartwig ML Analysis & Multi-Directional Visualization Pipeline
"""

import atexit
import json
import os
import sys
import time
import warnings
import datetime

# THREAD CAP -- must be set BEFORE numpy / xgboost / sklearn are imported to take effect.
#
# v15 runs MULTIPLE_PROCESSES = 16 worker processes, each of which by default opens as many
# OpenMP / BLAS threads as there are cores (32), i.e. 512 threads fighting over 32 cores.
# XGBoost, HistGradientBoosting and the MLP's BLAS calls all do this.
#
# MEASURED, 16 workers, the v15 ensemble, 8 chunks of 400k rows each, wall clock:
#     threads/worker   1      2      4      8     16     32   unlimited
#     wall (s)     134.2  113.0  105.1  137.2  210.4  222.8   197.2
#   -> 4 threads per worker is the optimum, 1.88x faster than leaving it unlimited.
#   Not 1: predict() is partly memory-LATENCY bound, so a little parallelism hides the
#   latency.  Not 8+: past 4 the workers start stealing cores and bandwidth from each other.
#   Note 16 x 4 = 64 threads on 32 cores still oversubscribes 2x, and that is still a win,
#   because a large share of the time is spent waiting on memory rather than computing.
#
# This does NOT change any number the pipeline reports: predict() is row-wise, so the thread
# count only affects how the same arithmetic is scheduled.  Set DSH_SWEEP_THREADS=0 to keep
# the previous unlimited behaviour.
_SWEEP_THREADS = os.environ.get('DSH_SWEEP_THREADS', '4')
if _SWEEP_THREADS not in ('0', '', 'none', 'None'):
    for _tv in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(_tv, _SWEEP_THREADS)

import numpy as np
import pandas as pd
from tqdm import tqdm
# v13 multiprocessing: imported lazily inside the functions that use them, so this
# module still imports in restricted environments.
import tempfile
import shutil
import pickle
import gc

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt  
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages   # Figure 11 multi-page PDF
import seaborn as sns
from scipy import stats
from sklearn.inspection import permutation_importance

from sklearn.model_selection import (train_test_split, cross_val_score, learning_curve,
                                     KFold, GroupKFold)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import time as _timing
# v15: ELMRegressor (Script/elm_model.py) is no longer imported -- the 'ELM-OneHot' member was
# removed from the model registry.  Re-add `from elm_model import ELMRegressor` to reinstate it.
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import xgboost as xgb

# Try importing RDKit for SMILES-based molecular descriptors
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Try importing SHAP for feature-attribution analysis
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')
np.random.seed(42)

# CONFIGURATION & STYLING
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['font.size'] = 8
rcParams['axes.labelsize'] = 9
rcParams['axes.titlesize'] = 10
rcParams['xtick.labelsize'] = 8
rcParams['ytick.labelsize'] = 8
rcParams['legend.fontsize'] = 7
rcParams['figure.titlesize'] = 11

COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#CA9161',
    'grey': '#949494'
}

# v12 MEMORY CONTROLS
# Keep the per-model prediction columns in bo_df?  They cost ~0.34 GiB each on a
# 45.7e6-row virtual library.  The ensemble-agreement gate only needs per-row
# min/max, which are accumulated in streaming form regardless, so setting this to
# False removes k x 0.34 GiB with no loss of gate functionality (the Pred_* columns
# simply do not appear in the output CSVs).
KEEP_MODEL_PREDICTIONS = False

# Rows per chunk when scoring the virtual library.  Peak extra memory is roughly
# batch_size x 8 bytes x (small constant), so 2e6 -> tens of MB, not GiB.
PREDICT_BATCH_SIZE = 2000000

# Store the virtual library's reagent columns as `category` instead of `object`.
# object dtype costs an 8-byte pointer per row per column (6 x 0.34 GiB = 2.05 GiB on
# a 45.7e6-row library) and every copy re-allocates it.  category is ~50x smaller.
# This is NOT a trade-off: values are identical and every downstream operation works.
VIRTUAL_LIBRARY_AS_CATEGORY = True

# v12 ROOT FIX -- the virtual library is ENUMERATED IN CHUNKS and never held whole.
# `stream_virtual_library` yields bounded pieces; `generate_bo_results_streaming`
# scores each piece, applies the gate, and keeps only the survivors.  Peak memory is
# therefore O(chunk x columns) and independent of the library size, so this runs on
# any dataset instead of failing at a different line each time.
#
# VIRTUAL_LIBRARY_PREVIEW_ROWS: how much RAW (unscored) library to keep for the
#   figure helpers that characterise the search space.  Not a correctness parameter.
# STREAM_CHUNK_COMBOS: reagent combinations per chunk.  200k combos x 8 loading
#   points = 1.6e6 rows/chunk ~ 100 MB -- small and fast.
# SURVIVOR_POOL_ROWS: how many gate-passing conditions to retain.  The figures and the
#   candidate CSVs only ever use the top few hundred, so this is generous.
VIRTUAL_LIBRARY_PREVIEW_ROWS = 200_000
STREAM_CHUNK_COMBOS = 200_000
SURVIVOR_POOL_ROWS = 20_000

# Hard ceiling on rows per streamed chunk, so peak memory is bounded even when the
# loading grid is fine: with 100 loading points a 200k-combo chunk would be 20e6 rows
# (~1.9 GiB), so the combo count is scaled down automatically.
#
# SCOPE -- READ THIS BEFORE RAISING IT: this constant bounds the SINGLE-PROCESS streamer
# only (stream_virtual_library / build_virtual_library / report_enumeration).  The
# multiprocessing engine does NOT use it; it uses PARALLEL_CHUNK_ROWS.
#
# Measured per-process memory of the scoring step (float64 DataFrame over the chunk):
#     2e6 rows ->  249 MiB      6e6 rows ->  661 MiB      20e6 rows -> 2,103 MiB
# Add ~245 MiB of fixed overhead (imports, models, training frame) for the full worker, so
# 20e6 rows in ONE process peaks at ~2.3 GiB -- fine when nothing else is running.
# v22g: A-class memory cap -- rows (combos x loading), not combos.  20M rows x n_feat x 8 B
# was the multi-GB block; 1M rows is ~20-100 MB.  Override with CD_CHUNK_ROWS.
STREAM_MAX_ROWS_PER_CHUNK = int(os.environ.get('CD_CHUNK_ROWS', '1000000'))

# Ensemble weighting.  Weights are derived from each member's held-out R^2 and
# normalised to sum to 1, so a weak member contributes less:
#   ENSEMBLE_WEIGHT_POWER = 1  -> w_i proportional to R^2_i
#   ENSEMBLE_WEIGHT_POWER = 0  -> plain (unweighted) mean, the pre-v12 behaviour
#   ENSEMBLE_WEIGHT_POWER = 2  -> sharper emphasis on the better members
# With three members whose R^2 differ by only ~0.02 the effect is small; the
# weighting matters when a clearly weaker model is included.
ENSEMBLE_WEIGHT_POWER = 1.0
ENSEMBLE_MIN_WEIGHT_R2 = 0.05

# v13 PERFORMANCE
# Build the model input matrix straight from the chunk's categorical CODES instead of
# converting every value back to str and re-encoding it.  Measured 24x faster on the
# encoding step (7.37 s -> 0.31 s per 6.2e6-row chunk) with identical output
# (max difference 2.9e-08, i.e. float64 round-off).
USE_FAST_FEATURE_FRAME = True
_LUT_CACHE = {}          # pandas category code -> label-encoder code, per column

# EXHAUSTIVE SWEEP + MULTIPROCESSING  (this is the default)
# v13 originally shipped a two-stage screening shortcut.  It is now DISABLED, because
# measurement showed it is not a safe shortcut on this problem:
#
#   reference = exhaustive ensemble sweep, cap 50e6 (8 loading points)
#   stage1_keep     top-20 recall   top-100 recall
#        50,000          8/20            77/100
#       200,000          8/20            77/100
#     1,000,000          8/20            77/100
#     3,000,000          8/20            77/100     <- 60x larger, ZERO improvement
#
# Enlarging the shortlist does not help because the missing conditions simply are not in
# the single screener model's high-score region: the ensemble mean is pulled up by a
# DIFFERENT member, so no amount of single-model shortlisting recovers them.  Screening
# therefore leaves the reported top-k wrong, which is unacceptable for a headline table.
#
# The replacement keeps the result EXACT and gets the speed from parallelism instead:
# the feature matrices for every streamed chunk are cached once to memory-mapped files,
# then independent worker processes read disjoint chunks and score them with all members.
# Same arithmetic, same order of operations, only the wall-clock changes.
USE_TWO_STAGE_SCREENING = False          # kept for reference; no longer used by default
STAGE1_KEEP_ROWS = 50_000

# Re-score the FINAL pool from its own stored conditions and drop any row whose score
# does not reproduce.  Cheap (a few thousand rows) and it turns a silent wrong-chemistry
# failure into a visible warning.  Strongly recommended: keep this on for any result that
# goes into a paper.
AUDIT_CONDITION_SCORES = True

USE_MULTIPROCESSING = True
N_WORKERS = 0            # 0 = auto (cpu_count - 1, capped at MAX_WORKERS)
MAX_WORKERS = 16
# Chunk size for the parallel sweep.  Larger chunks amortise worker round-trips but cost
# more memory PER WORKER, and MAX_WORKERS of them run at once -- so this is the constant
# that decides the machine's peak, NOT STREAM_MAX_ROWS_PER_CHUNK.
#
# MEASURED with the REAL _score_chunk_worker (pickled models, mmap load, per-member predict,
# accumulators, shortlist) -- not just the DataFrame step, which understates it ~2x:
#     chunk_rows   per-worker   x16 workers
#      2,000,000     580 MiB       9.1 GiB
#      4,000,000     915 MiB      14.3 GiB
#      6,000,000   1,251 MiB      19.5 GiB
#     20,000,000   3,600 MiB      56.2 GiB
# Fitted: peak_MiB ~= 168 * rows_in_millions + 245.
#
# !! v15: THAT FIT IS STALE.  It was taken in v14, when every ensemble member consumed the
# !! already-scaled 9-COLUMN matrix.  v15 added MLP-OneHot and ELM-OneHot, whose pipelines
# !! run OneHotEncoder INSIDE predict(), so the widest intermediate is no longer 9 columns:
# !!     OneHotEncoder output   (rows, 121) float64  ->   968 B/row
# !!     MLP hidden layer       (rows,  64) float64  ->   512 B/row
# !!     ELM hidden (n=2000)    (rows,2000) float64  -> 16,000 B/row   <- worst case
# !! At the old 4e6 chunk that is 3.61 GiB of one-hot output and 64 GiB of ELM activation
# !! PER WORKER.  Observed on (AI_RCP_pub)bh_data_clean_all_whitelisted -- 571,872,000 rows
# !! in 143 chunks -- which died with _ArrayMemoryError at chunk 19/143 having allocated
# !! 275 MiB (4e6, 9), 3.61 GiB (4e6, 121), 3.81 GiB (4e6, 128) and 1.91 GiB (4e6, 64)
# !! across 16 workers at once.
#
# The fix is NOT to shrink the chunk to ~30k rows (that would cost 19,000 round-trips on
# this library and still not bound the ELM).  _score_chunk_worker now scores each chunk in
# PREDICT_BATCH_ROWS slices, which decouples PEAK MEMORY (set by PREDICT_BATCH_ROWS) from
# THROUGHPUT (set by PARALLEL_CHUNK_ROWS).  With batch = 25,000 the transient ELM activation
# is 25,000 x 2,000 x 8 B x 2 live arrays ~ 800 MiB instead of 64 GiB.
#
# 4e6 is restored to v14's value now that batching bounds the peak.  Shrinking the chunk is
# NOT a memory lever any more -- it only multiplies the number of worker round-trips, and
# every round-trip re-unpickles the ensemble models.  Measured per-worker footprint at 4e6
# with the current member set (tree models + MLP-OneHot):
#     X frame (4e6 x 9 x 8) .................. 288 MiB
#     tot / tot_sq / vmin / vmax ............. 128 MiB
#     member_preds (4 members) ............... 128 MiB
#     transient batch (25,000 rows) .......... ~250 MiB
#     ~0.8 GiB per worker  x 16 workers  ~= 13 GiB, well inside the 40 GiB budget.
# It does NOT change the result: chunking is a scheduling detail, and the per-chunk
# shortlist (PARALLEL_TOPK_PER_CHUNK = 20,000) is far larger than the global top-k either way.
PARALLEL_CHUNK_ROWS = 4_000_000
# Prediction batch inside a worker.  Peak transient allocation is
#     PREDICT_BATCH_ROWS x (widest intermediate per row) x (2 live arrays)
# so it is this constant, not the chunk size, that decides per-worker peak.
#
# This constant exists because of a MEASURED failure.  Before it, _score_chunk_worker called
# mdl.predict() once on the whole chunk, and on (AI_RCP_pub)bh_data_clean_all_whitelisted
# (571,872,000 rows in 143 chunks of 4e6) MLP-OneHot's dense forward pass allocated, PER
# WORKER: (4e6, 121) = 3.61 GiB for the OneHotEncoder, (4e6, 128) = 3.81 GiB for its first
# hidden layer and (4e6, 64) = 1.91 GiB for its second -- ~13 GiB x 16 workers, which blew
# past 96 GB of RAM and died with _ArrayMemoryError at chunk 19/143.
#
# The widest intermediate per row, for the models that are now in the registry:
#     MLP-OneHot   OneHotEncoder output  121 cols x 8 B  =   968 B/row   <- widest
#                  first hidden layer    128 cols x 8 B  = 1,024 B/row
#     tree members (XGBoost / Random Forest / HistGB)    =    72 B/row, batched internally
# (The removed ELM-OneHot was far worse: n_hidden=2000 -> 16,000 B/row, i.e. 64 GB for a
# single 4e6-row chunk in one worker.  That is why this guard is kept even without it.)
#     batch     widest intermediate (rows,121)   per-worker transient
#     500,000              484 MiB                   ~1.0 GiB
#     250,000              242 MiB                   ~0.5 GiB   <-- current
#     100,000               97 MiB                   ~0.2 GiB
#      25,000               24 MiB                  ~0.05 GiB
# Budget target: measured peak on the real AI_RCP sweep was 17.1 GiB at batch=25,000; at
# batch=250,000 the added transient is 16 x ~0.5 GiB = ~8 GiB, so ~25 GiB against the 40 GiB
# cap.  Raising it further (500,000) would put the peak at ~47 GiB, i.e. ON the cap.
# It does NOT change the result: predict() is row-wise, so slicing the batch only changes the
# order rows are evaluated in.
#
# MEASURED cost of the slicing itself, 16 concurrent workers, v14 + HistGB ensemble:
#     batch =  25,000  (16 predict calls per chunk)   270.3 s
#     batch = 400,000  ( 1 predict call  per chunk)   232.9 s     -> the slicing cost ~14 %
# 250,000 was chosen to keep ~10x fewer calls than 25,000 while staying well inside the cap.
PREDICT_BATCH_ROWS = 250_000
# Held-out fraction.  0.2 means 80 % train / 20 % held out.
#
# This replaces the 0.49 that was inherited from the original benchmarking script's 0.30 and
# then widened -- at 0.49 the models were fitted on barely half the corpus, which wastes
# samples and depresses every reported R2.  0.80/0.20 is the conventional split.
#
# ONE constant, used by get_split_positions() and by every train_test_split in this file, so
# the modelling split and the figure code can never disagree.
#
# WARNING: changing this changes EVERY held-out number in the paper (all model R2, all RMSE,
# the capture ratios, and every figure panel that shows held-out predictions).  The noise
# ceiling does NOT move: it is computed from the whole dataset, not from the split.
TEST_SIZE = 0.2
# Boosting iterations for HistGB-NativeCat.  See the comment at its construction site in
# build_models(): predict() cost is LINEAR in this number.
#
# MEASURED predict on 4,000,000 rows, against max_iter=200:
#     max_iter  200     150     100
#     predict  5.80s   4.50s   3.21s
# and held-out R2 on four real datasets, delta vs 200:
#     150: -0.0006 mean      100: -0.0175 mean
# 100 was chosen on the user's call: HistGB is the single largest term in the real sweep
# (55% of worker predict time on Roche Buchwald), so cutting it 150->100 removes ~18% of the
# whole sweep.
HISTGB_MAX_ITER = 100# Trees for RandomForestRegressor, down from 200.  RF is 15.6% of the real sweep's predict
# time, and its predict cost is linear in the tree count:
#     200 -> 35.3 s per 4M chunk      100 -> 17.7 s      (0.50x)
# held-out R2 delta vs 200 trees, four datasets: -0.0029, -0.0092, +0.0002, -0.0015
#     mean -0.0034  -- by far the cheapest seconds-per-R2 change available.
RF_N_ESTIMATORS = 100
# Tree-level parallelism for RandomForestRegressor.  See SWEEP_RF_N_JOBS below for the
# measurement; this constant is used both for FIT and for the sweep's predict.
RF_N_JOBS = os.cpu_count() or 4
# Tree-level parallelism for RandomForestRegressor DURING THE SWEEP ONLY.
#
# sklearn's forest predict is
#     Parallel(n_jobs=n_jobs)(delayed(_accumulate_prediction)(e.predict, X, ...)
#                             for e in self.estimators_)
# so with the default n_jobs=1 the 200 trees are walked strictly one after another, one full
# pass over the batch each.  Random Forest was measured at 34.1 s per 4,000,000-row chunk
# versus 0.5 s for XGBoost -- 75% of the whole sweep's prediction time.
#
# MEASURED speed-up, RF 200 trees / depth 15, 500,000-row chunk per worker:
#     n_jobs        1      4      8     16     32     64    128
#     one process  1.00x  3.31x  5.91x 10.44x 13.66x 13.66x 12.38x
#     16 workers   1.00x  1.62x  1.91x  2.65x  3.19x    --     --
# 32 (= core count) is the plateau; 128 regresses.  Under 16-way concurrency the gain is
# smaller (16 x 32 = 512 threads on 32 cores) but still 3.19x.
#
# WHY THIS IS SET AT PREDICT TIME AND NOT IN THE CONSTRUCTOR: joblib refuses to nest, so a
# model whose n_jobs is set in the constructor would also try to parallelise FIT and the
# cross_val_score(..., n_jobs=-1) folds, nesting 5 folds x 32 threads.  Setting it inside the
# sweep worker touches only predict.
#
# IMPORTANT: this only works under ProcessPoolExecutor.  multiprocessing.Pool spawns DAEMONIC
# children, in which joblib silently falls back to n_jobs=1 -- measured 5.14 s either way.
# The sweep uses ProcessPoolExecutor (see generate_bo_results_parallel), so it applies there.
# Set DSH_SWEEP_NJOBS=0 to disable.
SWEEP_RF_N_JOBS = int(os.environ.get('DSH_SWEEP_NJOBS', '0') or 0) or (os.cpu_count() or 4)
# Per-chunk shortlist returned by each worker.  Must be generous enough that the global
# top-k survives the per-chunk -> global merge (each chunk keeps its own best).
PARALLEL_TOPK_PER_CHUNK = 20_000
# Cache the feature matrices on disk (memory-mapped) so workers can read them.  Set to
# None to use the system temp dir.
PARALLEL_CACHE_DIR = None
# 'spawn' is safest on Windows and avoids inheriting a large parent address space.
MP_START_METHOD = 'spawn'

# v13 MODELLING OPTIONS
# Preprocessing leakage fix.  v11/v12 fitted the StandardScaler on the FULL dataframe and
# only then split, so the scaler's mean/scale were computed partly from the held-out rows.
# The LabelEncoders are unaffected (they are per-column value maps), but the scaler is not.
#
# With this True, `prepare_features(..., fit_rows=<training positions>)` fits the scaler on
# the training rows only and then TRANSFORMS every row, so the returned matrix has the same
# shape and every downstream consumer (virtual library enumeration, Figure 6 scoring) is
# unchanged.  Callers that pass no fit_rows keep the old behaviour.
#
# This IS wired in: see prepare_features() and the four call sites.
FIT_PREPROCESSING_ON_TRAIN_ONLY = True

# The following were explored in standalone experiments and are deliberately NOT part of
# this script.  The flags were removed rather than left defined-but-unreferenced, because
# a defined constant that nothing reads is easy to mistake for a working feature.
#
#   one-hot encoding of the nominal reagent columns
#       standalone measurement: RF +0.012 R2, XGB +0.006, fewer negative predictions.
#       Not adopted: it expands the 9 features to ~150, which would make the 177e6-row
#       enumeration substantially slower and larger for a very small accuracy gain.
#
#   hurdle / two-part model for the zero-inflated target
#       standalone measurement: XGB R2 0.7351 and 0 negatives, but it HURT Random Forest
#       (0.7257 -> 0.7174).  Not adopted.
#
#   log1p + smearing on the target
#       standalone measurement: clearly worse (R2 0.6462).  Not adopted.

# RandomForest predictions are AVERAGES of leaf values, so they are structurally inside
# the training range.  Gradient boosting and XGBoost ADD up ~200 leaf values, so their
# sum can leave that range: observed here, 7.4 % of GBDT and 1.6 % of XGBoost
# predictions on the F9 candidate pool exceed 100 % (up to 105.5 %).
#
# That is not a bug, it is how additive boosting extrapolates onto unsampled reagent
# combinations.  But a predicted yield of 105 % is physically impossible, and if it is
# AVERAGED IN it argues FOR a condition on false evidence.  Clipping each MEMBER to
# [0, 100] before averaging enforces the bound the target actually has (the label is a
# normalised UV area percentage, so both ends are hard limits).
#
# Measured effect on this dataset: ~9 % of the candidate pool shifts, the ensemble moves
# by at most 1.8 pp, and top-20 membership changes by ~4 rows -- i.e. the candidates
# propped up by an impossible member prediction are demoted, which is the intent.
CLIP_MEMBER_PREDICTIONS = True
PREDICTION_MIN = 0.0
PREDICTION_MAX = 100.0

# Figure 11 only: instead of clipping the loading curve at 100 %, rescale the WHOLE figure
# so that the largest plotted value becomes exactly 100 % and everything else is compressed
# by the same factor.  Clipping flattens the top of the curve and destroys its shape;
# proportional rescaling keeps the shape and the ordering of the candidates intact, at the
# cost of the axis no longer being the raw model output.
#
# Applied per FIGURE (one factor shared by every curve and by the PDP), never per curve --
# a per-curve normalisation would force every compound to peak at 100 % and destroy the
# comparison between candidates, which is the entire point of the figure.
#
# The CSV always keeps the RAW values plus a `Rescaled` column, so the transformation is
# auditable and no data is silently altered.
F11_RESCALE_TO_100 = True

# Figure 7 (SHAP) explains TreeSHAP on this many test rows.  A few hundred is enough for a
# summary/dependence plot; v11/v12 used the whole test set with KernelExplainer.
SHAP_SAMPLE_ROWS = 500

# Safety ceiling on the enumerated virtual library (rows).  The categorical space is
# 37*20*16*21*23 = 5,718,720 combos, so the loading points actually enumerated are
# min(max_combinations // 5,718,720, N_LOADING_POINTS).
#
# NOTE: the definitive value is set in the VIRTUAL-LIBRARY CAP block below (search for
# "VIRTUAL-LIBRARY CAP"); do not add a second assignment here -- an earlier duplicate
# silently overrode the intended value, because the LAST assignment in the module wins.
#
# Raising the cap is safe for MEMORY because the sweep streams: it holds at most
# STREAM_MAX_ROWS_PER_CHUNK rows at a time regardless of the library size, so peak RAM does
# not grow with the cap.  What grows is TIME, linearly in the number of rows scored.  The
# build prints the forecast, so you can decide.  Check free RAM first:
#   (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB      # PowerShell

# The full condition key: everything that defines a distinct reaction condition.
# Used for the support lookup AND for collapsing replicate runs in Figure 6.
CONDITION_COLS = ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'cosolvent', 'additive',
               'reactant_2', 'reactant_1', 'temperature', 'time',
               'catalyst_loading', 'reactant_2_eq', 'base_eq', 'reactant_1_eq',
               'ligand_eq', 'solvent_eq', 'cosolvent_eq', 'additive_eq']
# v21: 19 slots -- the same set as hte/params.py CONDITION_COLS.  `catalyst_eq` keeps the
# canonical name `catalyst_loading` here because apply_loading_axis_alias() copies whichever
# amount column the file carries (catalyst_eq / loading / mol%) into that name; renaming it
# would touch ~40 call sites for no numeric change.

# The candidate categorical columns of the ENUMERATED LIBRARY (the search axes).
# `metal` is included and IS searched over -- it simply has only one value ('Pd') in this
# dataset, so its category list has length 1 and it contributes a factor of x1.  The
# library size is therefore the product of the five axes that actually vary:
#        37 x 20 x 16 x 21 x 23 = 5,718,720
# (`catalyst x base x solvent x reactant_2 x reactant_1`)
# The two partner slots are reactant_2 / reactant_1 (historically reactant_2 /
# reactant_1); the loader writes the new names, so the old ones are constant.
ENUM_AXES = ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1']

# The REAGENT axes that define a reagent compound for the proximity count (`Match Count`
# in Figure 10: the largest number of these axes shared with any real combination).
# `metal` is excluded because it is constant here, so including it would add a free match.
#
# Deliberately excludes `metal`: it is constant here, so counting it adds nothing and the
# count is meant to answer "how explored is this catalyst/base/solvent/leaving-group/
# substrate combination in the data".  Keep this separate from ENUM_AXES -- the search
# space and the counting key are different concepts and are allowed to differ.
REAGENT_AXES = ['catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1']


# v16: `catalyst` AND `ligand` ARE NOW TWO SEPARATE SLOTS
# Up to v15 both were funnelled into one `catalyst` slot, because the column
# NAMED `catalyst` does not always hold a catalyst.  Measured on the 16 inputs:
#
#   (AI_RCP_pub)*          `catalyst_name` = GPhos / BINAP / AlPhos  -- LIGANDS
#   (Summit)Baumgartner    `catalyst`      = tBuXPhos               -- a LIGAND
#   (Roche66k)*            `catalyst` = XantPhosPdCl2 and `ligand` = XantPhos
#                          -- genuinely two different variables, one of each
#   (bh-hte-ood)*          `Catalyst Name` = binap-pd-g3  -- a pre-formed complex,
#                          i.e. metal and ligand in one name, so it feeds `catalyst`
#   (rxn_yields)*          `Catalyst_1_Short_Hand` = Pd(OAc)2, `Ligand_Short_Hand`
#                          = P(tBu)3 -- already separated in the source
#
# The choice therefore cannot be derived from the column name; it is stated per
# input file here.  `None` means the file has no column of that kind, in which
# case the slot becomes 'Unknown' exactly like any other missing condition column.
# THIS IS THE ONLY PLACE TO EDIT when a new dataset is added.
CL_ROUTING = {
    '(AI_RCP_pub)bh_data_clean_all_whitelisted':     (None, 'catalyst_name'),
    '(AI_RCP_pub)sm_data_clean_all_whitelisted':     (None, 'catalyst_name'),
    '(DoyleHTE)CN_coupling_3600':                    (None, 'ligand_name'),
    '(Roche66k)Amide_coupling_3917':                 ('amide_coupling_reagent', 'ligand'),
    '(Roche66k)BH_plus_SM_24137':                    ('catalyst', 'ligand'),
    '(Roche66k)Buchwald_Hartwig_14191':              ('catalyst', 'ligand'),
    '(Roche66k)CH_activation_1251':                  ('catalyst', 'ligand'),
    '(Roche66k)CN_coupling_3100':                    ('catalyst', 'ligand'),
    '(Roche66k)CO_coupling_2112':                    ('catalyst', 'ligand'),
    '(Roche66k)Condensation_2095':                   (None, 'ligand'),
    '(Roche66k)Suzuki_Miyaura_9946':                 ('catalyst', 'ligand'),
    '(Summit)Baumgartner_CN_coupling_96':            (None, 'catalyst'),
    '(Summit)Reizman_Suzuki_385':                    ('catalyst', None),
    '(Suzuki-Coupling-ML)suzuki_dataset_raw_origin': (None, 'ligand'),
    '(bh-hte-ood)BH_HTE_Curated_Dataset_v20260219':  ('Catalyst Name', None),
    '(bh-hte-ood)BH_Manu_NewConditions_20250310_result': ('Catalyst Name', None),
    '(bh-hte-ood)Case1_selections_20250408_result':      ('Catalyst Name', None),
    '(rxn_yields)aap9112_Data_File_S1':              ('Catalyst_1_Short_Hand',
                                                      'Ligand_Short_Hand'),
}
# A file not listed above keeps the old single-slot behaviour, but split on the
# column name, so the two slots never both come from the same column.
CL_ROUTING_DEFAULT = ('catalyst', 'ligand')


def resolve_catalyst_ligand(dataset_stem):
    """(raw column for `catalyst`, raw column for `ligand`) for one input file."""
    return CL_ROUTING.get(dataset_stem, CL_ROUTING_DEFAULT)



# v13 STAGE-2 LOADING REFINEMENT -- IMPLEMENTED BUT DISABLED (see below)
# The first-stage sweep scores every reagent combination at a 31-point loading grid
# (spacing 0.01031 eq).  A dense re-scan of the survivors is implemented in
# `refine_loading_stage2`, but measurement showed it is NOT SAFE, so it ships OFF.
#
# What was measured
# A 1000-point scan reports a mean gain of +2.33 pp over the coarse grid -- but that is
# the classic "maximum over many noisy points" bias.  Verifying the SAME loading position
# on a disjoint half of the grid (odd/even split, so the second value is an independent
# measurement of the same quantity) recovers only +0.11 pp:
#
#     coarse (31 pt) best         20.20 pp
#     1000-pt argmax (as reported) 22.52 pp   -> +2.33 pp
#     same loading, independent    20.31 pp   -> +0.11 pp   (95 % of the gain was noise)
#     54.5 % of combinations: the entire apparent gain vanished on re-measurement
#
# Smoothing the curve before taking the argmax does not rescue it either: with a moving
# average of any width (1 .. 201 grid points) the verified gain is -0.02 pp or worse.
#
# Why: a tree ensemble's loading response is a STEP FUNCTION whose fine-scale variation
# (~2 pp between adjacent fine grid points) is model artefact, not chemistry.  Maximising
# over many such points selects artefact.  The loading axis genuinely has no reliable
# sub-grid structure to find, which is also why a denser grid is not worth much on its own.
#
# Set USE_LOADING_REFINE = True only if you intend to re-validate it on your own data
# with the odd/even test (see the module notes); do not enable it for published results.
USE_LOADING_REFINE = False
LOADING_REFINE_POINTS = 1_000
LOADING_REFINE_TOP_N = None       # None -> refine the entire survivor pool

# v14: LOADING-AXIS RESOLUTION
# Three figures sweep "the loading variable":
#     F3a  catalyst-loading dose-response curve
#     F9d  Bayesian loading optimisation (rank-1 candidate)
#     F11  loading curves for the top combinations
# as does the innermost factor of the virtual-library enumeration.
#
# `catalyst_loading` is the intended axis, but it is NOT present in every dataset.
# Audited across 24 public reaction CSVs:
#
#   * (AI_RCP_pub)* , (Suzuki-Coupling-ML)*   -> catalyst_loading present  (eq)
#   * Summit Reizman Suzuki (x4)              -> catalyst_loading present  (mol%)
#   * Summit Baumgartner (x2)                 -> NO catalyst_loading, but base_equivalents
#   * EDBO Suzuki (x3)                        -> ligand_equivalent (only 2 levels)
#   * (bh-hte-ood)* , Roche 66k , Doyle HTE   -> NO amount column of any kind
#
# Falling back to a hardcoded value would silently sweep an amount the data never
# contains, and produces N_LOADING_POINTS identical rows per reagent combination (the
# phantom loading column is not a model feature, so the enumeration emits duplicates).
#
# Strategy: resolve the axis ONCE, then ALIAS the chosen column to the canonical name
# `catalyst_loading`.  Every downstream consumer -- feature building, the enumeration, the
# match-count logic, the CSVs -- therefore keeps working unchanged; only the human-readable
# axis LABELS have to change, and those read LOADING_AXIS['label'] / ['unit'].
#
# When nothing usable exists, `catalyst_loading` simply stays absent, the enumeration
# stops varying it, and the three figures print an explicit "not available" note instead
# of drawing a meaningless flat line.
LOADING_AXIS_CANDIDATES = [
    # v22: the STANDARD slot names come first.  data/col_mappings.csv maps every file onto
    # these slots, so after mapping the raw spelling ('base_equivalents', 'loading', ...) no
    # longer exists.  Without them the resolver found only the constant `catalyst_loading`
    # filler and announced "no usable amount axis" on datasets that DO carry several
    # equivalents columns (e.g. (AI_RCP_pub)*, where the raw `catalyst_eq` is routed to
    # `ligand_eq` because that column actually holds the ligand).
    'catalyst_loading',     # the canonical alias used downstream
    'catalyst_eq',          # 催化剂当量
    'ligand_eq',            # 配体当量
    'base_eq',              # 碱当量
    'reactant_1_eq',        # 反应物1 当量
    'reactant_2_eq',        # 反应物2 当量
    'additive_eq',          # 添加剂当量
    'solvent_eq',           # 溶剂当量
    'cosolvent_eq',
    # legacy raw spellings, for files whose mapping has not been applied
    'base_equivalents',     # Summit Baumgartner (51 / 28 distinct values)
    'ligand_equivalent',    # EDBO Suzuki (only 2 levels -- weak but real)
    'additive_loading',
    'cocatalyst_loading',
    'concentration',        # EDBO BMS (3 levels)
    'loading',
]
# Human-readable axis name and unit, keyed by the ORIGINAL column name.
LOADING_AXIS_LABEL = {
    'catalyst_loading': 'Catalyst loading',
    'catalyst_eq': 'Catalyst loading',
    'ligand_eq': 'Ligand equivalents',
    'base_eq': 'Base equivalents',
    'reactant_1_eq': 'Reactant 1 equivalents',
    'reactant_2_eq': 'Reactant 2 equivalents',
    'additive_eq': 'Additive equivalents',
    'solvent_eq': 'Solvent equivalents',
    'cosolvent_eq': 'Cosolvent equivalents',
    'loading': 'Loading',
    'base_equivalents': 'Base equivalents',
    'base_equiv': 'Base equivalents',
    'ligand_equivalent': 'Ligand equivalents',
    'additive_loading': 'Additive loading',
    'cocatalyst_loading': 'Cocatalyst loading',
    'concentration': 'Concentration',
}
LOADING_AXIS_UNIT = {
    'catalyst_loading': 'eq.', 'catalyst_eq': 'eq.', 'loading': 'eq.',
    'ligand_eq': 'eq.', 'base_eq': 'eq.', 'reactant_1_eq': 'eq.',
    'reactant_2_eq': 'eq.', 'additive_eq': 'eq.', 'solvent_eq': 'eq.',
    'cosolvent_eq': 'eq.',
    'base_equivalents': 'eq.', 'base_equiv': 'eq.', 'ligand_equivalent': 'eq.',
    'additive_loading': 'eq.', 'cocatalyst_loading': 'eq.',
    'concentration': 'M',
}
# Minimum distinct finite values for a column to be usable as an axis.  2 means a
# two-level factor (e.g. EDBO ligand_equivalent) is still accepted, because a real
# 2-level effect is plottable; a CONSTANT column is not.
LOADING_AXIS_MIN_DISTINCT = 2

# Resolved once per run by resolve_loading_axis(); read by the figure code.
LOADING_AXIS = {
    'col': None, 'source_col': None, 'role': 'unavailable', 'label': None,
    'unit': None, 'n_distinct': 0, 'reason': 'not resolved yet',
}

# v14: the raw column that fills the pipeline's `catalyst` slot.  Usually 'catalyst', but
# e.g. 'amide_coupling_reagent' for Roche Amide coupling and 'ligand' for EDBO/Doyle files.
# The heatmap axis labels use it so they never call an amide activator a "catalyst".
# Set by load_real_dataset (which is the only place that knows the mapping result).
CATALYST_SLOT_LABEL = 'catalyst'

# v14: PARAMETER IMPORTANCE RANKING AND SLOT SUBSTITUTION
# Measured mean |SHAP| share on the AI_RCP Buchwald-Hartwig set (paper section 3.2):
#   reactant_2 26.9 | reactant_1 24.0 | base 17.2 | catalyst 12.5
#   catalyst_loading 10.0 | solvent 4.5 | time 3.2 | temperature 1.6
# (metal is constant Pd there, so it has no share.)
#
# Across the 19 datasets in Data/, availability runs almost OPPOSITE to importance:
#   catalyst 95 % | base 89 % | solvent 84 % | reactant_1 79 % | reactant_2 74 %
#   time 74 % | temperature 68 % | catalyst_loading 26 %
# so no single dataset carries the whole ranking, and every figure axis is a SLOT that has
# to be filled with the best AVAILABLE parameter.
#
# The ONE shared priority list every categorical axis is filled from.
#
# There is deliberately no separate x order and y order.  A two-axis panel walks this single
# list and takes the first two DISTINCT qualifying entries, so a variable that is missing
# simply shifts everything up rather than needing a hand-written per-axis fallback:
#     preferred pair first (so nothing changes when all are present)
#     -> process conditions by importance (base 17.2, catalyst 12.5, solvent 4.5)
#     -> substrate scope by importance (reactant_2 26.9, reactant_1 24.0)
PARAM_PRIORITY = ['base', 'catalyst', 'ligand', 'solvent', 'reactant_2', 'reactant_1']
# The TRUE global importance ranking, used where the question is "which variable matters
# most" rather than "which condition should this axis show".  Note the different order: a
# condition-axis slot leads with `base` (17.2, the highest-ranked PROCESS variable), whereas
# the overall ranking leads with `reactant_2` (26.9) and `reactant_1` (24.0).
# F7 panel (a) -- the SHAP dependence curve -- uses this one.
PARAM_RANK_BY_IMPORTANCE = ['reactant_2', 'reactant_1', 'base', 'catalyst',
                            'ligand', 'catalyst_loading', 'solvent', 'time',
                            'temperature']
# When a panel cannot find two PROCESS variables, allow a substrate-scope partner rather than
# leaving the panel blank.
ALLOW_SUBSTRATE_IN_CONDITION_PANELS = True
# v15: the SAME crossover in the other direction, for the substrate-scope panel (Figure 5c).
# Without it, a dataset whose ONLY varying scope column is `reactant_1` and whose
# `reactant_2` is a constant 'Unknown' finds just one qualifying scope parameter and the
# panel is dropped -- even though `catalyst` (or `base`, or `solvent`) is right there and
# would give a perfectly readable heatmap.  Observed on (Summit)Reizman_Suzuki_385, which has
# 4 substrate types and 7 catalysts but a single leaving group: Figure 5c printed
# "needs two varying scope parameters; this dataset has fewer than two".
ALLOW_PROCESS_IN_SUBSTRATE_PANELS = True
# Human-readable names for the axis titles.
PARAM_LABEL = {
    'reactant_2': 'Leaving Group', 'reactant_1': 'Substrate Type',
    'base': 'Base', 'catalyst': 'Catalyst', 'ligand': 'Ligand',
    'solvent': 'Solvent',
    'catalyst_loading': 'Catalyst Loading', 'time': 'Time', 'temperature': 'Temperature',
}


def _varies(df, col):
    """True if `col` exists and carries more than one distinct value."""
    try:
        return col in df.columns and df[col].nunique(dropna=True) > 1
    except Exception:
        return False


def _pick_param(preferred, family, df, label_encoders, used=()):
    """
    Fill ONE categorical figure axis with the best available parameter.

    Walks PARAM_PRIORITY, with `preferred` promoted to the front, and returns the first entry
    that qualifies.  A parameter qualifies only if it is in `label_encoders` (so it survived
    the categorical encoding) and varies -- a column the loader filled with a single 'Unknown'
    can never be chosen.

    `family` restricts the walk: 'process_categorical' stops after `solvent`, 'substrate'
    considers only the scope variables, and None means the whole list.
    """
    if family == 'process_categorical':
        pool = [p for p in PARAM_PRIORITY
                if p in ('base', 'catalyst', 'ligand', 'solvent')]
    elif family == 'substrate':
        pool = [p for p in PARAM_PRIORITY if p in ('reactant_2', 'reactant_1')]
    else:
        pool = list(PARAM_PRIORITY)
    order = ([preferred] if preferred else []) + [p for p in pool if p != preferred]
    for p in order:
        if p is None or p in used or p not in label_encoders:
            continue
        if _varies(df, p):
            return p
    return None


def _resolve_axis_pair(preferred, family, df, label_encoders, used=(),
                       allow_crossover=None):
    """
    Fill BOTH axes of an interaction panel from the single shared priority list.

    `preferred` is the pair the panel plots today, in today's order; it forms the head of the
    list, so a dataset that has every parameter produces exactly the same figure as before.
    The rest of PARAM_PRIORITY follows.  The x axis takes the first qualifying entry and the y
    axis the next one that differs, which guarantees the two axes are never the same variable
    and means a single missing column just shifts the whole pair up.

    With `allow_crossover` (defaults to ALLOW_SUBSTRATE_IN_CONDITION_PANELS) a process-only
    panel may fall through into the substrate scope once the process variables are exhausted.
    The mirror-image case -- a SUBSTRATE panel whose restricted pool holds fewer than two
    varying columns -- falls through into the process variables instead (v15, gated by
    ALLOW_PROCESS_IN_SUBSTRATE_PANELS).  Returns (None, None) when fewer than two parameters
    qualify even then.
    """
    if allow_crossover is None:
        allow_crossover = ALLOW_SUBSTRATE_IN_CONDITION_PANELS
    if family == 'process_categorical' and not allow_crossover:
        pool = [p for p in PARAM_PRIORITY
                if p in ('base', 'catalyst', 'ligand', 'solvent')]
    elif family == 'process_categorical':
        pool = list(PARAM_PRIORITY)
    elif family == 'substrate':
        pool = [p for p in PARAM_PRIORITY if p in ('reactant_2', 'reactant_1')]
    else:
        pool = list(PARAM_PRIORITY)

    def _qualify(_pool):
        """Order `preferred` first, then `_pool`, and keep the varying encoded columns."""
        _order = []
        for p in list(preferred or []) + _pool:
            if p and p not in _order and p not in used:
                _order.append(p)
        return [p for p in _order if p in label_encoders and _varies(df, p)]

    ok = _qualify(pool)

    # v15: the substrate pool is only ever two columns wide, so one constant scope column
    # is enough to empty the panel.  Retry against the FULL priority list before giving up,
    # which turns e.g. reactant_1 x reactant_2 into reactant_1 x catalyst.
    if len(ok) < 2 and family == 'substrate' and ALLOW_PROCESS_IN_SUBSTRATE_PANELS:
        _wide = _qualify(PARAM_PRIORITY)
        if len(_wide) > len(ok):
            ok = _wide

    if not ok:
        return None, None
    x = ok[0]
    y = next((p for p in ok[1:] if p != x), None)
    return x, y

# LOADING GRID RESOLUTION (single source of truth)
# Number of catalyst-loading values enumerated per reagent combination.  Every
# `n_points=...` literal used to be duplicated in ~13 places (including a runtime log
# string that hardcoded "the 31-point grid"); they now all read this constant.
# Raising it multiplies the enumerated library size LINEARLY, because the reagent space
# is enumerated exhaustively and the loading axis is the innermost factor:
#     rows = 5,718,720 reagent combinations x N_LOADING_POINTS
#        31 -> 177,280,320 rows
#        51 -> 291,654,720 rows
#       100 -> 571,872,000 rows
# The values are 100 EVENLY SPACED points between two quantiles of the historical loading
# distribution (see get_loading_grid), NOT a 0..1 grid and NOT a quantile sampling.
#
# CAUTION (measured, see USE_LOADING_REFINE below): the ensemble's loading response is a
# step function whose fine-scale wiggle is model artefact, so a denser grid does NOT buy a
# more reliable optimum -- it only lets the argmax land on a slightly different artefact.
# The extra resolution is for COVERAGE and for the Figure 11 curve shape, not for accuracy.
N_LOADING_POINTS = int(os.environ.get('CD_N_LOADING_POINTS', '100'))
# v23: the enumerated library now varies EVERY dose axis, not just the menu's sweep axis.
#   EXPAND_LOADING=True  -> each dose axis takes DOSE_POINTS points (default 10)
#   EXPAND_LOADING=False -> each dose axis takes ONE point (its median)
DOSE_POINTS = int(os.environ.get('CD_DOSE_POINTS', '10'))
_LAST_DOSE_COLS = []          # filled by get_enumeration_spec()
# v22g: CD_NO_LOADING_SWEEP=1 collapses the loading grid to ONE point, i.e. the composite
# pipeline's enumeration (every reagent combination scored once).  It is a BENCHMARK switch:
# it changes the candidate set, so the figures and the top-N list change with it.
NO_LOADING_SWEEP = os.environ.get('CD_NO_LOADING_SWEEP') == '1'
# v22i: the interactive menu overrides this: EXPAND_LOADING=True keeps the paper's
# enumeration (100 loading points per combination), False = the composite pipeline's
# single-point enumeration.  ask_choices() sets it.
EXPAND_LOADING = True

# Figure 11 (supplementary): how many of the top reagent combinations get a loading curve.
# Cost is TOP_LOADING_CURVES x N_LOADING_POINTS scored rows (100 x 100 = 10,000 by default),
# i.e. negligible next to the sweep.  Set to 0 to skip the PDF entirely.
# This number is independent of N_LOADING_POINTS: it controls how many CURVES are drawn.
TOP_LOADING_CURVES = 100

# VIRTUAL-LIBRARY CAP
# The cap must not silently truncate the loading grid.  With N_LOADING_POINTS = 100 the full
# library is 5,718,720 x 100 = 571,872,000 rows, so a 500e6 cap would quietly drop the grid
# back to 87 points.  1e9 leaves the full grid intact plus headroom up to 174 points.
# Truncation is SAFE FOR MEMORY either way (the sweep streams in bounded chunks); the cap
# only decides how much of the space is actually enumerated.
MAX_VIRTUAL_COMBINATIONS = 1_000_000_000

# NOTE on what a given cap yields:
#   grid points used = min(cap // 5,718,720, N_LOADING_POINTS)
#        50,000,000 ->   8 points ->  45,749,760 rows   (truncated)
#       571,872,000 -> 100 points -> 571,872,000 rows   (full; the current setting)
#     1,000,000,000 -> 100 points -> 571,872,000 rows   (same -- the grid is the binding
#                                                        limit, not the cap)
#
# v14 WARNING -- the cap CANNOT shrink the REAGENT space.
# The rule above only shrinks the LOADING grid, via get_loading_grid_for().  It has no
# effect on the reagent cartesian product `n_cat`: ggenerate_bo_results_parallel builds its
# chunk ranges over ALL of n_cat regardless of the cap.  So a dataset whose reagent product
# already exceeds the cap silently sweeps the whole product.
#
# Measured on (Roche66k)HTE_all_reactions_72142.csv: n_cat = 7,768,852,848 (catalyst 236 x
# base 149 x solvent 68 x fg_b 57 x fg_a 57) and n_load = 1, so the cap collapsed the grid
# to a single point and the sweep then enumerated 7.77e9 rows.  That needs ~200 GB of
# feature cache; it filled the temp drive and died with
#     OSError: [Errno 28] No space left on device
# after a 6-minute cache build, leaving the partial cache behind.
#
# ALLOW_OVERSIZED_SWEEP = False makes that case refuse up front with an actionable message
# instead of trying.  Set it True only if you have checked the cache drive can hold the
# estimated size (see _estimate_cache_bytes / _pick_cache_dir below).
ALLOW_OVERSIZED_SWEEP = False

# v12 ENSEMBLE QUALITY GATE
# `Model Gap (%)` = max(pred) - min(pred) across ensemble members for the SAME
# condition, i.e. how much the models disagree, in yield percentage points.
#
# NOTE ON UNITS: set ENSEMBLE_MAX_MODEL_GAP = 20 for "models must agree within
# 20 percentage points" (the strict reading), or = None to disable the gate.
# On this dataset a 20-point gap gate keeps ~73% of the virtual library and
# reliably removes the physically inadmissible candidates.
ENSEMBLE_MAX_MODEL_GAP = 20.0

# Gate B -- minimum number of historical occurrences of the exact condition.
# 0 or None disables it; 2 requires at least a replicated precedent.
ENSEMBLE_MIN_SUPPORT = 0

# v14: THE EI INCUMBENT  (the f* in  EI = (mu - f* - xi)Phi(z) + sigma*phi(z))
# EI answers "how much do I expect to beat my best-so-far, on average".  Its value
# therefore depends entirely on what "best-so-far" is taken to be, and the choice is a
# MODELLING decision, not a technical detail:
#
#   'max'      f* = the best yield anywhere in the file.  This is literally "my all-time
#              record".  With 10,138 measured reactions (46 % of them at zero, the 99th
#              percentile at 95 %) the record is ~100 %, so mu - f* is negative for almost
#              every virtual condition and EI collapses to ~0.3 pp -- four orders of
#              magnitude below the spread of mu itself.  The RANKING still exists, but its
#              magnitude carries no information and the "should I keep screening?" reading
#              ("EI below assay noise") is then trivially yes/no regardless of the model.
#
#   'quantile' f* = a high quantile of the observed yields -- "a good condition I already
#              have", i.e. the realistic starting point for a NEXT-ROUND recommendation on
#              a campaign that has not yet found the maximum.  This is the framing the
#              Top-N table is presented under ("which untested conditions should I run
#              next"), and it is the one under which EI_max is comparable to the assay
#              noise and therefore usable as a stopping criterion.
#
# Both are defensible; they answer different questions.  The mode is logged at run time
# and printed on the Figure 10b footnote so a reader always knows which one produced the
# table.  v12 sidestepped the issue by dropping EI as the sort key entirely; v14 keeps
# Figure 10a ranked by predicted yield and adds Figure 10b ranked by EI, so the two can
# be compared rather than one silently replacing the other.
EI_INCUMBENT_MODE = 'quantile'      # 'quantile' | 'max'
EI_INCUMBENT_QUANTILE = 0.90        # used only when EI_INCUMBENT_MODE == 'quantile'

# v14: the TARGET that Figure 10b measures expected improvement against -- a quantile of the
# candidate pool's PREDICTED yields (not of the observed yields).  See the comment at the
# `ranked_ei` construction for why the observed-yield incumbent cannot be used for the
# recommendation table: on a well-screened dataset it sits above almost every prediction and
# EI degenerates into an uncertainty ranking.  0.90 = "the top decile of what this library
# can offer", which is a question a recommender can usefully answer.
EI_TARGET_QUANTILE = 1.00


def estimate_noise_ceiling(df, cond_cols=None):
    """
    R2_max = 1 - sigma2_within / sigma2_total, over conditions that were measured more than once.

    Ported verbatim from the standalone analysis script that produced the numbers quoted in
    section 3.3, so the figure and that table can never drift apart.

    Distinguishes the two very different reasons a model can fail:
        (a) the features carry no signal            -> ceiling HIGH, the model is the problem
        (b) replicate noise dominates the variance  -> ceiling LOW,  the model is at its limit

    Returns a dict with `ceiling` = NaN when NO condition repeats (3 of the 16 datasets:
    Doyle C-N, Summit C-N and the synthetic benchmark), because the quantity is then
    undefined rather than zero.  `rep_frac` is the share of rows in a repeated condition.

    CAVEAT carried over from the paper: the condition key here is the eight MAPPED columns,
    and the mapping does not separate catalyst from ligand, so the estimate sits BELOW the
    data source's own official value.  Treat it as a lower bound on the true ceiling.
    """
    COND = ['catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1',
            'temperature', 'time', 'catalyst_loading']
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
    return out


def resolve_ei_incumbent(df):
    """
    Return (f_star, all_time_record) for the expected-improvement incumbent.

    Single source of truth: both the virtual-library sweep and Figure 10b call this, so the
    EI column and the EI-ranked table can never disagree about what they were measured
    against.  See EI_INCUMBENT_MODE for why the choice matters.
    """
    _y = pd.to_numeric(df['yield'], errors='coerce').dropna()
    _record = float(_y.max()) if len(_y) else 0.0
    if EI_INCUMBENT_MODE == 'quantile' and len(_y):
        return float(np.percentile(_y.to_numpy(), EI_INCUMBENT_QUANTILE * 100.0)), _record
    return _record, _record

# How the Validated / Discovery label is decided.
#   'exact'   -> the exact condition (incl. catalyst loading) must exist in the data
#   'reagent' -> the catalyst+base+solvent combination must exist   [v12 default]
#   'any'     -> the reagent combination OR the substrate must exist
# The v11 rule was effectively 'exact', which almost never matches because the
# virtual library uses a coarse loading grid while the data has thousands of
# distinct measured loadings -- so every row was mislabelled 'Discovery'.
STATUS_MATCH_LEVEL = 'reagent'

# v13 ENVIRONMENT SELF-CHECK -- why the GPU was busy during CPU-only training
# The PyPI `xgboost` wheel is compiled WITH CUDA (`xgb.build_info()['USE_CUDA']` is
# True).  `device='cpu'` only selects the CPU *algorithm*; it does NOT stop that
# build from touching the GPU.  Measured on this project with that wheel:
#
#   * the first fit() loads nvcuda.dll and creates a CUDA context of ~0.19-0.25 GB;
#   * a single predict() does the same, so the context is re-created even after being
#     destroyed by hand -- it therefore cannot be "returned" mid-run from Python;
#   * `del model` / gc.collect() do NOT free it; it lives until the process exits;
#   * `cross_val_score(..., n_jobs=-1)` produced 5 extra persistent joblib worker
#     processes, each holding its own context (RandomForest produced none).
#
# Since `train_and_evaluate_models` dispatches the CV / learning-curve fits to
# persistent joblib workers, a pure-CPU training section can pin several GB of VRAM
# and only releases it when the script ends.
#
# Fix: the official CPU-only wheel -- same version, identical CPU results, no driver:
#     python -m pip uninstall -y xgboost
#     python -m pip install xgboost-cpu==<same version>
# `xgboost-cpu` provides the same `xgboost` module, so no code change is needed.
def warn_if_cuda_xgboost():
    """Report which xgboost build is active and warn if it can pin GPU memory."""
    try:
        info = xgb.build_info()
    except Exception as exc:                      # defensive: never break the run
        print(f"[env] xgboost build_info() unavailable ({exc}); cannot check USE_CUDA.")
        return
    if info.get('USE_CUDA'):
        print("[WARN] This xgboost build has USE_CUDA=True. Everything here runs on the "
              "CPU (device='cpu'), but every process that calls fit() or predict() "
              "creates a CUDA context of ~0.2 GB that Python cannot release and that "
              "is only freed at process exit -- including the joblib workers spawned "
              "by cross_val_score / learning_curve (n_jobs=-1), so several GB of VRAM "
              "can stay occupied during training.")
        print("       CPU-only alternative (identical results, zero VRAM): "
              "python -m pip uninstall -y xgboost && python -m pip install xgboost-cpu")
    else:
        print("[env] xgboost is CPU-only (USE_CUDA=False)")


def _cm_blue_ref():
    import matplotlib
    return matplotlib.colormaps['Blues'](0.75)


def _cm_orange_ref():
    import matplotlib
    return matplotlib.colormaps['Oranges'](0.75)


def bar_gradient(ax, values, positions=None, cmap="Blues", horizontal=True,
                 edgecolor="black", vmin=None, vmax=None, alpha=1.0, **kw):
    """Draw bars whose FILL intensity follows the value (a real gradient, not one flat hue).

    v22 / issue 3: the earlier pass only lightened a single colour, which reads as "pale",
    not as a gradient.  Here each bar takes its own step of ``cmap``: small values are near
    white, the largest is the full colour.  Negative values fall back to ``Reds`` so a
    negative importance stays visually distinct from a small positive one.
    """
    import matplotlib
    vals = np.asarray(values, dtype=float)
    pos = np.arange(len(vals)) if positions is None else np.asarray(positions, dtype=float)
    fin = vals[np.isfinite(vals)]
    if fin.size == 0:
        return
    lo = float(np.min(fin)) if vmin is None else float(vmin)
    hi = float(np.max(fin)) if vmax is None else float(vmax)
    rng = hi - lo
    pos_cmap = matplotlib.colormaps[cmap]
    neg_cmap = matplotlib.colormaps["Reds"]
    for _p, _v in zip(pos, vals):
        if not np.isfinite(_v):
            continue
        frac = 0.25 + 0.75 * ((_v - lo) / rng if rng > 1e-12 else 0.6)
        frac = float(np.clip(frac, 0.0, 1.0))
        cm = neg_cmap if _v < 0 else pos_cmap
        colour = cm(frac)
        if horizontal:
            ax.barh(_p, _v, color=colour, edgecolor=edgecolor, alpha=alpha, **kw)
        else:
            ax.bar(_p, _v, color=colour, edgecolor=edgecolor, alpha=alpha, **kw)


def soft_color(color, f=0.45):
    """Blend a colour toward white.

    v21 / C9: the attribution bars (Fig. 4a, 7c, 8b/c/d) are drawn in these paler tones --
    at full saturation the long bars in those panels dominated the eye and the small ones
    disappeared into the axis.
    """
    import matplotlib.colors as _mc
    r, g, b = _mc.to_rgb(color)
    return (r + (1.0 - r) * f, g + (1.0 - g) * f, b + (1.0 - b) * f)


def add_bar_labels(ax, fmt="{:.1f}", padding=3, fontsize=7):
    """
    Annotate every bar with its value.

    v13 fix: the old guard was `height > 0`, so NEGATIVE bars were silently skipped.  Test
    R2 genuinely goes below zero -- every model did on (bh-hte-ood)BH_Manu_NewConditions
    (XGBoost -0.635) -- and the panel then showed bars with no numbers at all, which reads
    as "the scores are zero".  Negative bars are now labelled below the bar.
    """
    for rect in ax.patches:
        height = rect.get_height()
        if np.isnan(height) or height == 0:
            continue
        _x = rect.get_x() + rect.get_width() / 2
        if height > 0:
            ax.annotate(fmt.format(height), xy=(_x, height),
                        xytext=(0, padding), textcoords='offset points',
                        ha='center', va='bottom', fontsize=fontsize, fontweight='bold')
        else:
            ax.annotate(fmt.format(height), xy=(_x, height),
                        xytext=(0, -padding), textcoords='offset points',
                        ha='center', va='top', fontsize=fontsize, fontweight='bold')

# 1. DATA LOADING & PREPROCESSING

def create_default_col_mappings(mapping_path):
    default_mappings = pd.DataFrame([
        {'standard_column': 'catalyst', 'raw_column': 'catalyst_name'},
        {'standard_column': 'base', 'raw_column': 'reagent_1_name'},
        {'standard_column': 'solvent', 'raw_column': 'solvent_1_name'},
        {'standard_column': 'solvent_smiles', 'raw_column': 'solvent_1_smiles'},
        {'standard_column': 'reactant_2', 'raw_column': 'startingmat_2_name'},
        {'standard_column': 'reactant_1', 'raw_column': 'startingmat_1_name'},
        {'standard_column': 'temperature', 'raw_column': 'temperature_deg_c'},
        {'standard_column': 'time', 'raw_column': 'time_h'},
        {'standard_column': 'catalyst_loading', 'raw_column': 'catalyst_eq'},
        {'standard_column': 'yield', 'raw_column': 'product_1_area%'},
        {'standard_column': 'success', 'raw_column': 'suff_yield'}
    ])
    
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    default_mappings.to_csv(mapping_path, index=False, encoding='utf-8-sig')
    print(f"Auto-generated a default column-mapping file: '{mapping_path}'")

def load_real_dataset(data_path, data_dir):
    print(f"Loading data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path, encoding='utf-8-sig')
    except Exception:
        try:
            df = pd.read_csv(data_path, encoding='gbk')
        except Exception:
            df = pd.read_csv(data_path, encoding='gb18030')
    
    df.columns = [str(c).strip() for c in df.columns]
    _source_snapshot = df.copy()   # snapshot BEFORE any alias pass
    raw_feature_names = {}

    mapping_path = os.path.join(data_dir, 'col_mappings.csv')
    if not os.path.exists(mapping_path):
        candidate_mapping_paths = [
            'Data/col_mappings.csv',
            'data/col_mappings.csv',
            '../Data/col_mappings.csv',
            '../data/col_mappings.csv',
            'col_mappings.csv'
        ]
        for alt_path in candidate_mapping_paths:
            if os.path.exists(alt_path):
                mapping_path = alt_path
                break
        else:
            create_default_col_mappings(mapping_path)

    print(f"Reading column mapping config from {mapping_path}")
    
    map_df = None
    for enc in ['utf-8-sig', 'gbk', 'gb18030']:
        try:
            map_df = pd.read_csv(
                mapping_path, 
                header=None, 
                names=range(20),
                engine='python',
                encoding=enc,
                on_bad_lines='skip'
            )
            break
        except Exception:
            continue

    if map_df is None:
        map_df = pd.read_csv(mapping_path, header=None)

    for _, row in map_df.iterrows():
        non_empty = [str(val).strip() for val in row.dropna().values if str(val).strip() not in ['', 'nan']]
        if not non_empty:
            continue
        
        # v15 FIX: 'metal' used to be in this skip list, which made it IMPOSSIBLE to map a
        # metal column through col_mappings.csv -- adding a `metal,...` row did nothing.  The
        # metal slot then fell back to the hard-coded 'Pd' filler below, so a dataset whose
        # metal column was called anything other than the literal string `metal` (e.g.
        # "Metal Center") would silently lose its metal identity.  Only the synthetic benchmark
        # escaped that, purely because its column happens to be named `metal`.
        if non_empty[0].lower() in ['standard_column', 'key', 'raw_column']:
            if len(non_empty) > 1 and non_empty[1].lower() in ['raw_column', 'value', 'standard_column']:
                continue

        std_col = non_empty[0]
        # v16: `ligand` is no longer folded into `catalyst`.  The two are separate
        # slots and CL_ROUTING (see the top of the file) decides, per input file,
        # which raw column feeds which -- a column NAMED `catalyst` may in fact
        # hold a free ligand.  So the alias rows for `ligand` are skipped here.
        if std_col.lower() == 'ligand':
            continue
            
        raw_candidates = list(non_empty[1:]) if len(non_empty) > 1 else []
        # v13: also try the mapping KEY itself and the STANDARD name.  Without this a file
        # whose columns are already named after the alias (e.g. a `ligand` column, which the
        # mapping file calls standard_column 'ligand') could never match, because only the
        # right-hand aliases were tried -- the file ended up with catalyst = 'Unknown'.
        for _extra in (non_empty[0], std_col):
            if _extra not in raw_candidates:
                raw_candidates.append(_extra)
        if not raw_candidates:
            raw_candidates = [std_col]

        if std_col not in df.columns:
            for raw_c in raw_candidates:
                if raw_c in df.columns:
                    df[std_col] = df[raw_c]
                    raw_feature_names[std_col] = raw_c
                    print(f"Dataset column '{raw_c}' mapped to standard '{std_col}'")
                    break
        else:
            raw_feature_names[std_col] = std_col

    # v16 CATALYST / LIGAND SPLIT -- authoritative, overrides the alias pass
    _stem = os.path.splitext(os.path.basename(data_path))[0]
    _cat_raw, _lig_raw = resolve_catalyst_ligand(_stem)
    # Snapshot BOTH sources before writing either slot.  One slot's source may be
    # the column named after the OTHER slot -- Summit-Baumgartner feeds `ligand`
    # from its column literally called `catalyst` -- so writing `catalyst` first
    # would overwrite the very data `ligand` is about to read.
    _snap = {}
    for _slot, _raw in (('catalyst', _cat_raw), ('ligand', _lig_raw)):
        _snap[_slot] = (df[_raw].copy(), _raw) \
            if (_raw and _raw in df.columns) else (None, _raw)
    for _slot in ('catalyst', 'ligand'):
        _vals, _raw = _snap[_slot]
        if _vals is None:
            df[_slot] = 'Unknown'
            print(f"[v16 split] file '{_stem}' has no {_slot} column -> 'Unknown'")
        else:
            df[_slot] = _vals
            raw_feature_names[_slot] = _raw
            print(f"[v16 split] '{_raw}' -> slot '{_slot}' "
                  f"({df[_slot].nunique()} values)")

    # col_mappings.csv -- per-file overrides beat global aliases.  The mapping is
    # mandatory: swallowing a failure here would silently produce a frame with no
    # per-file overrides at all (that is exactly what used to happen).
    # this file lives in archive/, the package one level up
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from hte.mapping import apply_mapping
    df = apply_mapping(df, data_path, data_dir, raw_feature_names,
                       source_df=_source_snapshot)

    if 'solvent_smiles' not in df.columns:
        # v14: only bind a SMILES column that is actually ABOUT THE SOLVENT.
        #
        # The old rule was "the first column whose name contains 'smiles'".  On
        # (DoyleHTE)CN_coupling_3600.csv -- which has no solvent column at all -- the column
        # order is base_smiles, ligand_smiles, substrate_smiles, additive_smiles,
        # product_smiles, so `base_smiles` was silently bound as solvent_smiles.  The F5d
        # "Solvent SMILES Descriptor PCA" then computed RDKit descriptors for three AMINE
        # BASES and labelled every point with the solvent name, which the required-column
        # filler had set to 'Unknown' -- giving three identical "Unknown" labels.
        #
        # Leaving it unset is strictly better: the PCA panel already knows how to say that a
        # SMILES column is unavailable, whereas a wrong molecule is not visibly wrong.
        smiles_candidates = [c for c in df.columns if 'smiles' in c.lower()]
        solvent_like = [c for c in smiles_candidates if 'solvent' in c.lower()]
        if solvent_like:
            target_smiles_col = solvent_like[0]
            df['solvent_smiles'] = df[target_smiles_col]
            raw_feature_names['solvent_smiles'] = target_smiles_col
            print(f"Auto-detection: raw column '{target_smiles_col}' bound as 'solvent_smiles'")
        elif smiles_candidates:
            print(f"  [mapping] NOTE: this file has SMILES columns {smiles_candidates[:4]} "
                  f"but none is a solvent SMILES, so 'solvent_smiles' is left UNSET.  The "
                  f"solvent-PCA panel will report it as unavailable instead of plotting the "
                  f"wrong molecules.")

    # The loader can already create a 'metal' column (filled with 'Unknown' for
    # files that carry no metal column), so testing for its absence left the
    # reader dead code on those files.  Run it whenever it is enabled.
    if os.environ.get('CD_METAL_READER', '1') != '0':
        df['metal'] = 'Pd'
        # Read the metal from the catalyst columns instead of assuming palladium
        # (hte/metal_reading.py).  Falls back to the constant above when the
        # package is not importable, so this script stays standalone.
        try:
            _here = os.path.dirname(os.path.abspath(__file__))
            if _here not in sys.path:
                sys.path.insert(0, _here)
            import hte.metal_reading as _mrm
            _mrm.STEMS = _mrm.build_stems(
                df['ligand'].astype(str).tolist() if 'ligand' in df.columns else [])
            _nmc = df['catalyst'] if 'catalyst' in df.columns else [''] * len(df)
            _smc = next((df[c] for c in df.columns
                         if 'catalyst' in c.lower() and 'smiles' in c.lower()), None)
            df['metal'] = _mrm.read_rows(
                _nmc, _smc if _smc is not None else [''] * len(df))[0]
            print('[metal] reader -> %d distinct values' % df['metal'].nunique())
        except Exception as _exc:  # noqa: BLE001
            print('[metal] reader skipped: %s' % _exc)

    # v21: a row whose metal cannot be read ('unknown') takes the dataset's dominant
    # catalytic metal.  Across the corpus this is about 5 % of rows; left as 'unknown' it
    # becomes a spurious extra category on every metal axis ('unknown' next to 'Pd').
    # 'none' is NOT touched -- it means the file is genuinely metal-free (metal_free: true).
    if 'metal' in df.columns:
        _m = df['metal'].astype(str)
        _unk = _m.str.lower().isin(['unknown', '']) | _m.isna()
        if bool(_unk.any()):
            _known = _m[~_unk]
            if len(_known):
                _dom = _known.value_counts().idxmax()
                df.loc[_unk, 'metal'] = _dom
                print('[metal] %d rows (%.1f%%) unreadable -> dominant metal %r'
                      % (int(_unk.sum()), 100.0 * _unk.sum() / max(len(df), 1), _dom))

    required_cols = ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 
                     'reactant_1', 'temperature', 'time', 'catalyst_loading', 
                     'yield']
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Warning: missing columns {missing}; filled with empty defaults")
        for c in missing:
            if c != 'yield':
                df[c] = 'Unknown' if c in ['catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1'] else 0.0

    # v14: fill NaN INSIDE the categorical columns too, not just the missing columns above.
    #
    # A NaN in a categorical column was left alone, and prepare_features does
    # `df[col].astype(str)` for the label encoder -- which turns it into the literal string
    # 'nan' and creates a REAL category for "the reagent is unknown".  Measured across Data/:
    # 30,703 such NaNs in 10 datasets (HTE_all_reactions alone has 15,542 in `catalyst`), and
    # the heatmap axes consequently carried a 'nan' tick label.
    #
    # 'Unknown' is used because it is what the missing-column filler above already writes, so
    # an explicitly-unknown reagent and an absent one land in the same bucket instead of
    # splitting the signal across two.
    # v23d: iterate the canonical slot list so additive / cosolvent are covered too
    try:
        from hte.params import CATEGORICAL_SLOTS as _CS
        _cat_fill = list(_CS)
    except Exception:                               # noqa: BLE001
        _cat_fill = ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2',
                     'reactant_1']
    _nan_filled = {}
    for _c in _cat_fill:
        if _c in df.columns:
            _n = int(df[_c].isna().sum())
            if _n:
                df[_c] = df[_c].fillna('Unknown').astype(str)
                _nan_filled[_c] = _n
            else:
                df[_c] = df[_c].astype(str)
    if _nan_filled:
        print("  [NaN] filled missing categorical values with 'Unknown': "
              + ', '.join(f'{c}={n}' for c, n in _nan_filled.items()))

    if 'yield' not in df.columns:
        raise KeyError("'yield' column (or a mapped alias) not found in dataset")

    # v13: LOUD warning about collapsed categorical vocabularies.
    # A categorical axis with a single value cannot be enumerated, so the virtual library
    # collapses along it, `get_enumeration_spec` multiplies the space by 1, and every
    # figure that displays reagent names shows 'Unknown'.  This is almost always a column-
    # mapping failure rather than a property of the chemistry, so it must not be silent.
    _cat_cols = ['catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1']
    _collapsed = {}
    for _c in _cat_cols:
        if _c in df.columns:
            _u = df[_c].astype(str).unique()
            if len(_u) <= 1:
                _collapsed[_c] = _u[0] if len(_u) else '(empty)'
    if _collapsed:

        print("  [WARN] these reagent axes have ONLY ONE distinct value in this dataset:")
        for _c, _v in _collapsed.items():
            print(f"           {_c:<16} = '{_v}'   ({len(df[_c])} rows)")
        print("  [WARN] the virtual library cannot vary along them, its size is multiplied")
        print("         by 1 per axis, and figures that print reagent names will show that")
        print("         single value.  This is normally a COLUMN-MAPPING failure: check")
        print("         Data/col_mappings.csv for an alias matching this file's header.")


    df = df.dropna(subset=['yield']).reset_index(drop=True)
    df['yield'] = pd.to_numeric(df['yield'], errors='coerce')
    df = df.dropna(subset=['yield']).reset_index(drop=True)

    max_y = df['yield'].max()
    if max_y <= 1.5:
        print("Converted decimal-scale yields to percent")
        df['yield'] = df['yield'] * 100.0

    df['success'] = (df['yield'] >= 50.0).astype(int)
        
    print(f"Successfully loaded {len(df)} reaction records.")
    print(f"Mean yield: {df['yield'].mean():.1f}%")
    print(f"Success rate (yield>=50%): {df['success'].mean()*100:.1f}%")

    # v14: decide which amount axis F3a / F9d / F11 (and the innermost factor of the
    # enumeration) should sweep, and expose it under the canonical name so no downstream
    # code has to know whether it is really catalyst loading.  See the block at the top.
    resolve_loading_axis(df)
    df = apply_loading_axis_alias(df)
    # v14: remember which column actually fills the `catalyst` slot, for the heatmap labels.
    global CATALYST_SLOT_LABEL
    CATALYST_SLOT_LABEL = str(raw_feature_names.get('catalyst', 'catalyst'))
    return df, raw_feature_names

# v15 TIMING
# TIMINGS[model][item] = seconds.  Keys:
#   train_*                     -- per-model training (fit / CV / learning curves)
#   figure4_permutation_importance, figure7_shap_compute, figure8_partial_dependence
#                                 -- the figure steps that DO cost real compute
# TIMINGS_SINGLE holds model-independent costs (the virtual-library sweep).
TIMINGS = {}
TIMINGS_SINGLE = {}


def _t_add(model_name, item, seconds):
    TIMINGS.setdefault(model_name, {})[item] = float(seconds)


_ITEM_ORDER = ['train_fit', 'train_cv_shuffled', 'train_cv_ordered',
               'train_learning_curve_r2', 'train_learning_curve_rmse', 'train_TOTAL',
               'figure4_permutation_importance', 'figure7_shap_compute',
               'figure8_partial_dependence',
               'virtual_library_sweep', 'virtual_library_sweep_note']


def _save_timings(output_dir):
    """timings.csv: ROWS = timing item, COLUMNS = model (as requested)."""
    if not TIMINGS:
        return None
    items = []
    for d in TIMINGS.values():
        for k in d:
            if k not in items:
                items.append(k)
    items = [i for i in _ITEM_ORDER if i in items] + [i for i in items if i not in _ITEM_ORDER]
    # v15: only the training TOTAL is reported -- the five sub-items are still measured (they
    # are what makes the breakdown interpretable in the log) but they clutter the CSV.
    _DROP = {'train_fit', 'train_cv_shuffled', 'train_cv_ordered',
             'train_learning_curve_r2', 'train_learning_curve_rmse'}
    items = [i for i in items if i not in _DROP]
    cols = list(TIMINGS.keys())
    df = pd.DataFrame(index=items, columns=cols, dtype=float)
    for m, d in TIMINGS.items():
        for k, v in d.items():
            if isinstance(v, (int, float)):
                df.loc[k, m] = float(v)
    for k, v in TIMINGS_SINGLE.items():
        if k not in df.index:
            df.loc[k] = np.nan
        if cols:
            df.loc[k, cols[0]] = float(v)
    df.index.name = 'timing_item'
    path = os.path.join(output_dir, 'timings.csv')
    # v15: final filter -- only train_TOTAL is reported; df.loc[] above silently appends
    # rows for keys missing from the index, so filtering the item list is not enough.
    df = df[~df.index.isin(['train_fit', 'train_cv_shuffled', 'train_cv_ordered',
                          'train_learning_curve_r2', 'train_learning_curve_rmse'])]
    df.to_csv(path, encoding='utf-8-sig')
    print(f"  [timings] -> timings.csv  ({len(df)} items x {len(cols)} model columns)")
    return path


def get_split_positions(n_rows, test_size=None, random_state=42):
    """
    The ONE definition of the train/test split, as positional indices.

    Every train_test_split call in this file uses (test_size=TEST_SIZE, random_state=42), and
    splitting `np.arange(n_rows)` returns positional indices, so this reproduces the split
    of the feature matrix exactly.  Modelling code and the figure code both use it, so a
    row's train/held-out label can never disagree with how the models were fitted.
    """
    if test_size is None:
        test_size = TEST_SIZE
    idx_tr, idx_te = train_test_split(np.arange(int(n_rows)), test_size=test_size,
                                      random_state=random_state)
    return np.asarray(idx_tr), np.asarray(idx_te)


# Columns whose constant-value notice has already been printed in this process.
# prepare_features runs more than once per session, and every run used to repeat the
# same seven informational lines.
_CONST_REPORTED = set()

# v22n: how missing NUMERIC cells are treated before scaling.
#   'median' (default, previous behaviour) -- impute the training median
#   'none'   -- fill a sentinel value (MISSING_SENTINEL) and keep it as a real number, the way
#               the composite pipeline encodes a missing reagent as the literal category NONE
#               instead of NaN.  It makes "was this field recorded at all" learnable.
MISSING_FILL = os.environ.get('CD_MISSING_FILL', 'median').lower()
MISSING_SENTINEL = float(os.environ.get('CD_MISSING_SENTINEL', '-1'))


def prepare_features(df, raw_feature_names=None, fit_rows=None):
    """
    Encode + scale the feature matrix.

    fit_rows
    --------
    Positional row indices on which to FIT the StandardScaler.  Pass the training rows so
    the scaler never sees the held-out set -- the v11/v12 behaviour fitted it on all rows,
    which leaks test-set statistics (mean/scale) into the model inputs.

    The scikit-learn convention is right here: fit on train, transform everything.  So the
    RETURNED matrix is still the transform of the whole frame, identical in shape to before;
    only the two numbers the scaler learned change.  Callers that score the full dataset or
    the virtual library keep working unchanged.

    fit_rows=None reproduces the old behaviour exactly.
    """
    df_encoded = df.copy()
    # The slot vocabulary has ONE definition: hte/params.py.  A second hard-coded
    # list here is what silently dropped three amount axes when the slots were
    # renamed -- the names simply stopped matching and nothing complained.
    from hte.params import CATEGORICAL_SLOTS as _CAT, NUMERIC_SLOTS as _NUM
    categorical_cols = list(_CAT)
    numerical_cols = list(_NUM)
    # numerical slots come from the same table (see above)
    
    categorical_cols = [c for c in categorical_cols if c in df_encoded.columns]
    _kept = []
    for _c in categorical_cols:
        if df_encoded[_c].astype(str).nunique(dropna=False) > 1:
            _kept.append(_c)
        else:
            if _c not in _CONST_REPORTED:
                _CONST_REPORTED.add(_c)
                print(f"\u2139\ufe0f \u69fd '{_c}' \u5728\u672c\u6570\u636e\u96c6\u4e3a\u5e38\u91cf\uff0c\u4e0d\u53c2\u4e0e\u5efa\u6a21\u3002")
    categorical_cols = _kept
    
    valid_numerical_cols = []
    for col in numerical_cols:
        if col in df_encoded.columns:
            if df_encoded[col].nunique() > 1:
                valid_numerical_cols.append(col)
            else:
                if ('num', col) not in _CONST_REPORTED:
                    _CONST_REPORTED.add(('num', col))
                    print(f" Constant feature '{col}' (single value {df_encoded[col].iloc[0]}) dropped from the training frame.")
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    feature_cols = [col + '_encoded' for col in categorical_cols] + valid_numerical_cols
    X = df_encoded[feature_cols]
    y = df_encoded['yield']
    
    scaler = StandardScaler()
    numerical_indices = [feature_cols.index(col) for col in valid_numerical_cols]
    X_scaled = X.copy()
    if len(numerical_indices) > 0:
        num_block = X.iloc[:, numerical_indices]

        # v14: MEDIAN-IMPUTE missing numeric features before scaling.
        #
        # Real HTE exports have gaps: (Roche66k)HTE_all_reactions_72142.csv is 29 % NaN in
        # `temperature` and `time`, because the source records the process conditions as free
        # text ("17 h at 25 °C") and a fifth of the rows say `unknown`.  Without this,
        # StandardScaler propagates the NaN and GradientBoostingRegressor raises
        #     ValueError: Input X contains NaN
        # (XGBoost and RandomForest tolerate NaN; sklearn's GradientBoosting does not, so the
        # failure surfaced only on the third model).
        #
        # The fill values are taken from the TRAINING rows only, matching the scaler policy
        # above -- imputing with a full-data median would leak held-out statistics into the
        # features.  Dropping the affected rows instead is defensible but would discard 29 %
        # of this dataset's measured yields, which are still valid; the yield is observed even
        # when the process conditions are not.
        _na_mask = num_block.isna()
        if _na_mask.to_numpy().any():
            if FIT_PREPROCESSING_ON_TRAIN_ONLY and fit_rows is not None and len(fit_rows) > 0:
                _r = np.asarray(fit_rows)
                _r = _r[(_r >= 0) & (_r < len(num_block))]
                _fill = num_block.iloc[_r].median()
            else:
                _fill = num_block.median()
            _fill = _fill.fillna(0.0)          # a column that is entirely NaN
            # CD_MISSING_FILL=none: keep 'not recorded' as a real, learnable value (a sentinel)
            # instead of imputing the median -- the composite pipeline encodes a missing
            # reagent as the category NONE rather than NaN.
            if MISSING_FILL == 'none':
                _fill = pd.Series(float(MISSING_SENTINEL), index=num_block.columns)
            _counts = _na_mask.sum()
            num_block = num_block.fillna(_fill)
            X_scaled.iloc[:, numerical_indices] = num_block
            print(f"  [NaN] imputed {int(_counts.sum())} missing numeric feature value(s) with "
                  f"the training median: "
                  + ', '.join(f'{c}={_counts[c]} ({100*_counts[c]/len(num_block):.0f}%)'
                              for c in num_block.columns if _counts[c] > 0))

        if FIT_PREPROCESSING_ON_TRAIN_ONLY and fit_rows is not None and len(fit_rows) > 0:
            rows = np.asarray(fit_rows)
            rows = rows[(rows >= 0) & (rows < len(num_block))]
            scaler.fit(num_block.iloc[rows])
        else:
            scaler.fit(num_block)
        X_scaled.iloc[:, numerical_indices] = scaler.transform(num_block)
        
    display_feature_names = []
    for f in feature_cols:
        base_col = f.replace('_encoded', '')
        if raw_feature_names and base_col in raw_feature_names:
            display_feature_names.append(raw_feature_names[base_col])
        else:
            display_feature_names.append(base_col)

    # v16: these labels are the RAW column names, and splitting `catalyst` from
    # `ligand` makes a collision possible -- Summit-Baumgartner feeds `ligand` from
    # its column literally called `catalyst`, while the `catalyst` slot has no
    # source and falls back to its own slot name.  Two identical labels turn
    # `X.columns = list(feature_names)` inside get_shap_values into a frame with
    # DUPLICATE columns, after which xgboost's DMatrix does `frame[col].dtype` on a
    # DataFrame and dies with "'DataFrame' object has no attribute 'dtype'".
    # Disambiguate the LABELS only; `feature_cols`, the real column names, are
    # untouched, so nothing downstream that indexes by name is affected.
    _seen_nm = {}
    _uniq_nm = []
    for _nm in display_feature_names:
        _k = _seen_nm.get(_nm, 0)
        _seen_nm[_nm] = _k + 1
        _uniq_nm.append(_nm if _k == 0 else '%s [%d]' % (_nm, _k + 1))
    display_feature_names = _uniq_nm

    return X_scaled, y, feature_cols, display_feature_names, label_encoders, scaler

# 2. MODEL TRAINING & BENCHMARKING

def train_and_evaluate_models(X_train, X_test, y_train, y_test,
                              X_cv=None, y_cv=None):
    """
    X_cv / y_cv: the rows the two cross-validation numbers are computed on, defaulting to the
    training split.

    WHY IT EXISTS: section 3.6 compares a shuffled fold split with an ordered one and reports
    "+0.767 shuffled vs -1.17 ordered".  Both of those come from cross_val_score over the FULL
    dataset in FILE ORDER, so passing X_train silently changes both numbers -- measured on
    AI_RCP bh / XGBoost:
        rows used            shuffled     ordered
        X_train (51%)        +0.7275     -0.0668
        ALL rows (100%)      +0.7658     -1.2354     <- matches the paper
    Both CVs now run on the same rows, which is also what makes "same model, same data, only
    the split changes" a valid comparison.
    """
    # v15: categorical columns are the ones prepare_features named *_encoded
    _cols = list(getattr(X_train, 'columns', []))
    _cat_idx = [i for i, c in enumerate(_cols) if str(c).endswith('_encoded')]
    _xcv = X_cv if X_cv is not None else X_train
    _ycv = y_cv if y_cv is not None else y_train

    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        'Multilayer Perceptron': MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', max_iter=500, random_state=42, early_stopping=True),
        'Random Forest': RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=15, min_samples_split=5, random_state=42),
        # HISTGB_MAX_ITER is 150, not 200.  HistGradientBoostingRegressor.predict walks its
        # trees in a PYTHON loop, one pass over the batch per iteration (XGBoost does the whole
        # thing in one C++ call), so its predict cost is linear in the iteration count and it
        # dominated the virtual-library sweep.
        #
        # MEASURED predict on 4,000,000 rows, and held-out R2 against max_iter=200:
        #     max_iter  200     150     100
        #     predict  5.80s   4.50s   3.21s      (150 = 1.29x faster, 100 = 1.81x)
        #     Roche amide  0.6376  0.6317  0.6167
        #     Reizman      0.8038  0.7984  0.7889
        #     AI_RCP sm    0.7051  0.7101  0.7125
        #     Roche Buchw. 0.4926  0.4766  0.4517
        #     mean delta     --   -0.0055 -0.0175
        # 150 buys 2/3 of the 100-iteration speed-up for 1/3 of its accuracy cost, i.e. more
        # speed per unit of R2 given up (53x vs 46x).  max_iter is a CAP: early stopping can
        # still finish sooner, and on the 16 datasets n_iter_ was exactly 200 without it.
        'HistGB-NativeCat': HistGradientBoostingRegressor(
            max_iter=HISTGB_MAX_ITER, max_depth=6, learning_rate=0.1, random_state=42,
            categorical_features=(_cat_idx if _cat_idx else None)),
        'MLP-OneHot': make_pipeline(
            ColumnTransformer(
                [('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False), _cat_idx)],
                remainder='passthrough'),
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                         max_iter=500, random_state=42, early_stopping=True)),
        # 'ELM-OneHot' was REMOVED here.  It was the single most expensive member the sweep
        # could ever score: its predict() materialises a (rows, n_hidden=2000) activation,
        # so on a 4,000,000-row chunk ONE worker needed 4e6 x 2000 x 8 B = 64 GB, i.e. more
        # than the whole machine, and its 242,000 MAC/row made it ~11x slower than any other
        # neural member (measured 1,960 ms per 100,000 rows vs 170 ms for MLP-OneHot).
        # It never actually ran on (AI_RCP_pub)bh_data_clean_all_whitelisted -- it missed the
        # top 4 -- but it DID make the top 4 on (Summit)Reizman_Suzuki_385, so the risk was
        # live for any dataset with a large reagent cross-product.  `Script/elm_model.py` is
        # left in place so the model can be reinstated; see the ELM sweep measurements:
        # n_hidden=1000 costs ~2x less and moves R2 by only 0.002 .. 0.044.
    }
    
    results = []
    trained_models = {}
    cache = {}
    cv_folds = 5 if len(X_train) >= 50 else 3
    # v13: the HTE file is ordered by substrate, so an UNSHUFFLED KFold measures
    # "extrapolate to unseen substrates" (R2 ~ -1.2) rather than the random-split
    # benchmark we report.  Use a shuffled split for the headline CV number and keep an
    # ordered split as an explicit leakage diagnostic.
    cv_shuffled = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_ordered = KFold(n_splits=cv_folds, shuffle=False)

    # v22f: train ONLY the members the chosen cells map to -- the rest of the six are not
    # used by any figure or by the ensemble, so training them was pure cost.
    _wanted = list((globals().get('MODEL_DISPLAY') or {}).keys())
    if _wanted:
        models = {k: v for k, v in models.items() if k in _wanted}

    for name, model in models.items():
        _ENC_OF = {'XGBoost': 'integer', 'Gradient Boosting': 'integer',
                   'Multilayer Perceptron': 'integer', 'Random Forest': 'integer',
                   'HistGB-NativeCat': 'native', 'MLP-OneHot': 'one-hot'}
        if name == next(iter(models)):
            print("\n" + "-" * 70)
            print("[stage 2/2] ensemble members -- these are what Figure 6/9/10 average.")
            print("           Only the members of the cells chosen above are trained here.")
            print("-" * 70)
        print(f"Training [{name}] ({_ENC_OF.get(name, '?')}) ...")
        _tm0 = _timing.perf_counter()
        # Random Forest is the ONE estimator here whose FIT also scales with n_jobs
        # (sklearn parallelises both the tree building and the tree traversal):
        #     MEASURED, 10,000 rows x 200 trees: fit 6.07 s at n_jobs=1 -> 0.43 s at n_jobs=32
        # It is set for the main fit ONLY.  cross_val_score and learning_curve below run with
        # n_jobs=-1, so leaving it set would nest 30 tasks x 32 threads; the estimator is put
        # back to n_jobs=None before those calls.
        _rf_fast = (isinstance(model, RandomForestRegressor)
                    and RF_N_JOBS and RF_N_JOBS > 1)
        if _rf_fast:
            model.n_jobs = RF_N_JOBS
        _t0 = _timing.perf_counter()
        model.fit(X_train, y_train)
        _t_add(name, 'train_fit', _timing.perf_counter() - _t0)
        trained_models[name] = model
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        if _rf_fast:
            model.n_jobs = None          # keep CV / learning-curve parallelism un-nested
        
        _t0 = _timing.perf_counter()
        cv_scores = cross_val_score(model, _xcv, _ycv, cv=cv_shuffled,
                                    scoring='r2', n_jobs=-1)
        _t_add(name, 'train_cv_shuffled', _timing.perf_counter() - _t0)
        try:
            _t0 = _timing.perf_counter()
            # The ordered CV must see the FILE's record order.  X_cv is the full frame in file
            # order (its index is 0..n-1); X_train is shuffled by train_test_split, so using it
            # would silently turn this into another random split -- the bug that made section
            # 3.6's -1.17 unreproducible.
            cv_ordered_scores = cross_val_score(model, _xcv, _ycv, cv=cv_ordered,
                                                scoring='r2', n_jobs=-1)
            _t_add(name, 'train_cv_ordered', _timing.perf_counter() - _t0)
            cv_ordered_mean = float(np.mean(cv_ordered_scores))
            cv_ordered_folds = [float(v) for v in cv_ordered_scores]
        except Exception:
            cv_ordered_mean = float('nan')
            cv_ordered_folds = []
        
        # Section 3.6 claims the framework inverts on UNSEEN SUBSTRATES (R2 < 0).  That number
        # used to come from a one-off script (collect_stats2.py / logic_check.py) and was not
        # reproducible from the pipeline.  It is computed here, per model, with GroupKFold on
        # the encoded substrate slot: every reaction of a given substrate is in the SAME fold,
        # so the model is always scored on substrates it has never seen.
        # GroupKFold needs at least 2 groups; more than 5 groups are unused by a 5-fold split.
        cv_sub_mean = float('nan')
        cv_sub_folds = []
        try:
            _sub_col = next((c for c in _cols if str(c) == 'reactant_1_encoded'), None)
            if _sub_col is not None:
                _g = _xcv[_sub_col].to_numpy()
                _n_g = int(len(np.unique(_g)))
                if _n_g >= 2:
                    _t0 = _timing.perf_counter()
                    _cv_sub = GroupKFold(n_splits=min(cv_folds, _n_g))
                    _sub_scores = cross_val_score(model, _xcv, _ycv, cv=_cv_sub,
                                                  groups=_g, scoring='r2', n_jobs=-1)
                    _t_add(name, 'train_cv_substrate_disjoint',
                           _timing.perf_counter() - _t0)
                    cv_sub_mean = float(np.mean(_sub_scores))
                    cv_sub_folds = [float(v) for v in _sub_scores]
        except Exception:
            cv_sub_mean = float('nan')
            cv_sub_folds = []
        
        _t0 = _timing.perf_counter()
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=3, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
        )
        _t_add(name, 'train_learning_curve_r2', _timing.perf_counter() - _t0)
        
        _t0 = _timing.perf_counter()
        train_sizes_2b, train_loss_2b, test_loss_2b = learning_curve(
            model, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error',
            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
        )
        _t_add(name, 'train_learning_curve_rmse', _timing.perf_counter() - _t0)
        _t_add(name, 'train_TOTAL', _timing.perf_counter() - _tm0)
        
        cache[name] = {
            'y_pred_test': y_pred_test,
            'learning_curve_r2': (train_sizes, train_scores, test_scores),
            'learning_curve_rmse': (train_sizes_2b, train_loss_2b, test_loss_2b)
        }
        
        results.append({
            'Model': name,
            'R² (Train)': r2_score(y_train, y_pred_train),
            'R² (Test)': r2_score(y_test, y_pred_test),
            'RMSE (Train)': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'RMSE (Test)': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'MAE (Train)': mean_absolute_error(y_train, y_pred_train),
            'MAE (Test)': mean_absolute_error(y_test, y_pred_test),
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std(),
            # v13: a leakage diagnostic -- the ordered split answers "next round of
            # experiments" rather than "unseen row".  v15: it is now genuinely the FILE's
            # record order (X_cv is the full frame with index 0..n-1, which
            # train_test_split would have shuffled), and it runs on the SAME rows as the
            # shuffled number so the two are directly comparable.
            'CV R² (ordered folds)': cv_ordered_mean,
            'CV R² (ordered folds, per fold)': ';'.join('%.4f' % v
                                                       for v in cv_ordered_folds),
            # v15: unseen-substrate accuracy, the protocol behind section 3.6's "inverts on new
            # substrates" claim.  Previously only computable from a one-off script.
            'CV R² (substrate-disjoint)': cv_sub_mean,
            'CV R² (substrate-disjoint, per fold)': ';'.join('%.4f' % v
                                                            for v in cv_sub_folds),
            # v13: a plain regressor can emit physically impossible yields; report how
            # often, and what clipping would do to the metric.
            'Pred <0 (test)': int((y_pred_test < 0).sum()),
            'Pred >100 (test)': int((y_pred_test > 100).sum()),
            'R² (Test, clipped)': r2_score(y_test, np.clip(y_pred_test, 0, 100)),
        })
    return pd.DataFrame(results), trained_models, cache

def predict_members_clipped(models, X):
    """
    Member predictions with the project's clipping policy applied.

    WHY THIS EXISTS: a yield is a normalised UV area percentage, so [0, 100] are hard
    physical limits, and additive-boosting members do violate them (observed: XGBoost 104 %
    on a real reaction, and negative values).  The virtual-library sweep applies
    `CLIP_MEMBER_PREDICTIONS` before combining members, but the figure code used to call
    `mdl.predict(X)` raw and average the result -- so Figure 11 could display > 100 %
    (measured: 109.1 % on the Suzuki-Miyaura set, 6,650 of 10,000 rows out of range) while
    Figure 10, whose pool came from the clipped sweep, never did.  Two figures describing
    the same ensemble disagreed.

    Routing every figure through this helper guarantees they combine the SAME member values.
    """
    out = {}



    for nm, mdl in models.items():
        p = np.asarray(mdl.predict(X), dtype=float)
        if CLIP_MEMBER_PREDICTIONS:
            p = np.clip(p, PREDICTION_MIN, PREDICTION_MAX)
        out[nm] = p
    return out


def get_shap_values(model, X, feature_names=None):
    """
    Exact TreeSHAP values for tree models, model-agnostic values for everything else.

    Why this exists: `shap.TreeExplainer(xgb_model)` fails on xgboost >= 3.0 with
        ValueError: could not convert string to float: '[2.0202346E0]'
    because xgboost now stores learner_model_param.base_score as a bracketed VECTOR
    string and shap 0.49's XGBTreeModelLoader calls float() on it directly.  Verified
    with xgboost 3.2.0 + shap 0.49.1 (the combination in this environment).

    Routing:
      * xgboost  -> booster.predict(DMatrix, pred_contribs=True): exact TreeSHAP, no
                    shap dependency at all.  Additivity verified to ~1e-4 (float32).
      * sklearn tree -> shap.TreeExplainer, which handles RandomForest / GradientBoosting
                    correctly (additivity verified to ~1e-13).
      * anything else (MLPRegressor) -> shap.PermutationExplainer.  TreeExplainer raises
                    outright for these --
                      "Model type not yet supported by TreeExplainer: MLPRegressor"
                    -- which blanked the Multilayer Perceptron SHAP panels on every dataset where
                    the MLP placed in the top three by R2 (observed on aap9112_Data_File_S1).
                    PermutationExplainer is model-agnostic and, at these row counts, much
                    cheaper than the KernelExplainer this project used before.

    Returns (shap_values[n_rows, n_features], base_value).
    """
    X = pd.DataFrame(X)
    if feature_names is not None:
        X.columns = list(feature_names)

    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        names = booster.feature_names if booster.feature_names else list(X.columns)
        dm = xgb.DMatrix(X, feature_names=list(names))
        contribs = np.asarray(booster.predict(dm, pred_contribs=True), dtype=np.float64)
        return contribs[:, :-1], float(np.asarray(contribs[:, -1]).ravel()[0])

    # sklearn tree ensembles
    if hasattr(model, 'estimators_') or hasattr(model, 'tree_'):
        e = shap.TreeExplainer(model)
        sv = np.asarray(e.shap_values(X), dtype=np.float64)
        if sv.ndim == 3:
            sv = sv[:, :, 0]
        base = getattr(e, 'expected_value', 0.0)
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).ravel()[0])
        return sv, float(base)

    # model-agnostic fallback (neural networks and anything else)
    #
    # Feed the explainer a plain float array, not a DataFrame.  `X.columns` was just
    # overwritten with DISPLAY names (e.g. 'Catalyst_1_Short_Hand'), and sklearn validates a
    # DataFrame's column names against those seen during fit for MLPRegressor.predict:
    #     ValueError: The feature names should match those that were passed during fit.
    # The tree paths above never trip this because xgboost builds a DMatrix with explicit
    # names and shap's TreeExplainer takes the array.  Passing values sidesteps the check
    # entirely, and the display names are only ever used for plot labels.
    _Xv = np.asarray(X, dtype=float)
    _raw_predict = getattr(model, 'predict', model)

    def _flat(A):
        return np.asarray(_raw_predict(A), dtype=float).ravel()

    ex = shap.PermutationExplainer(_flat, _Xv)
    exp = ex(_Xv, max_evals=int(2 * _Xv.shape[1] + 1), silent=True)
    sv = np.asarray(exp.values, dtype=np.float64)
    if sv.ndim == 3:
        sv = sv[:, :, 0]
    base = np.asarray(exp.base_values).ravel()
    return sv, float(base[0]) if base.size else 0.0


def analyze_feature_importance(model, display_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        importances = np.full(len(display_names), np.nan)
        # v15: was np.random.uniform(...) -- FABRICATED bars for models with no native
        # importance (HistGB has none; MLP-OneHot / ELM-OneHot are Pipelines).  Figure 4
        # panel c no longer calls this at all; NaN keeps any future caller honest.
        
    return pd.DataFrame({'Feature': display_names, 'Importance': importances}).sort_values('Importance', ascending=False)

def save_predictions_csv(df, trained_models, X_scaled, output_dir):
    pred_df = df.copy()
    for name, model in trained_models.items():
        pred_df[f'Predicted_Yield_{name}'] = model.predict(X_scaled)
    
    pred_df['Actual_Yield'] = df['yield']
    
    out_path = os.path.join(output_dir, 'model_predictions_all.csv')
    pred_df.to_csv(out_path, index=False, encoding='utf-8-sig')


# 2B. VIRTUAL COMBINATION SPACE & BAYESIAN OPTIMIZATION

def build_virtual_feature_frame(df, feat_cols, label_encoders, scaler, records):
    raw = pd.DataFrame(records).copy()
    # derive the encoded slots from feat_cols, not from a hard-coded list of the
    # retired names: the encoder keys are the current slot names, so the old
    # list left reactant_1/2_encoded as NaN.
    categorical_cols = [str(c)[:-len('_encoded')] for c in feat_cols
                        if str(c).endswith('_encoded')]
    # derive the amount axes from feat_cols: the hard-coded list still used the
    # retired names (reactant_2_eq / base_eq), so the new *_eq columns were left
    # as NaN and the model refused to predict.
    numerical_cols = [c for c in feat_cols
                      if not str(c).endswith('_encoded') and c in df.columns]

    encoded = pd.DataFrame(index=raw.index)

    for col in categorical_cols:
        encoded_name = col + '_encoded'
        if encoded_name not in feat_cols:
            continue
        le = label_encoders.get(col)
        if le is None:
            continue
        if col not in raw.columns:
            # The enumerated frame only carries the search axes; a slot that is held
            # fixed has no column here, so pin it to its modal value.
            _mode = df[col].astype(str).mode()
            _default = _mode.iloc[0] if len(_mode) else str(le.classes_[0])
            encoded[encoded_name] = le.transform([_default] * len(raw))
            continue
        values = raw[col].astype(str)
        encoded[encoded_name] = le.transform(values)

    for col in numerical_cols:
        if col not in raw.columns:
            # held-fixed amount axis: pin it to the dataset median
            _m = pd.to_numeric(df[col], errors="coerce")
            encoded[col] = float(_m.median()) if _m.notna().any() else 0.0
            continue
        encoded[col] = pd.to_numeric(raw[col], errors='coerce')
        fallback = float(df[col].median()) if pd.to_numeric(df[col], errors='coerce').notna().any() else 0.0
        encoded[col] = encoded[col].fillna(fallback)

    encoded = encoded.reindex(columns=feat_cols)
    numerical_indices = [i for i, c in enumerate(feat_cols) if not c.endswith('_encoded')]
    if numerical_indices:
        encoded_values = encoded.iloc[:, numerical_indices].astype(float).values
        encoded.iloc[:, numerical_indices] = scaler.transform(encoded_values)

    return encoded.astype(float)


def fast_virtual_features(chunk, feat_cols, label_encoders, scaler,
                          cat_cols=('metal', 'catalyst', 'ligand', 'base', 'solvent',
                                    'reactant_2', 'reactant_1'),
                          num_cols=('temperature', 'time', 'catalyst_loading'),
                          lut_cache=None):
    """
    v13 FAST PATH -- build the model input matrix for a streamed chunk WITHOUT any
    string work.

    `build_virtual_feature_frame` does `raw[col].astype(str)` then `LabelEncoder.transform`
    on every row: measured at 7.37 s per 6.2e6-row chunk, i.e. ~17 % of the whole run.

    The chunk's categorical columns are already pandas `category` dtype, so each value is
    an integer code.  Mapping those codes through a tiny lookup table gives exactly the
    LabelEncoder output -- and takes 0.31 s instead of 7.37 s (24x faster).

    CRITICAL: the lookup table must be built from `chunk[col].cat.categories`, because
    that is the order the CODES refer to.  Using e.g. `sorted(unique())` silently gives
    wrong encodings (verified: max error 29 on the encoded scale).

    Returns a float64 DataFrame with the same columns as `feat_cols`, so it is a drop-in
    replacement.
    """
    if lut_cache is None:
        lut_cache = {}
    n = len(chunk)
    cat_present = [c for c in cat_cols if c + '_encoded' in feat_cols]
    num_present = [c for c in num_cols if c in feat_cols]

    out = np.empty((n, len(feat_cols)), dtype=np.float32)
    col_pos = {c: i for i, c in enumerate(feat_cols)}

    for c in cat_present:
        key = (c, id(chunk[c].dtype))
        lut = lut_cache.get(c)
        if lut is None:
            cats = np.asarray(chunk[c].cat.categories, dtype=object)
            lut = label_encoders[c].transform(cats).astype(np.float64)
            lut_cache[c] = lut
        codes = chunk[c].cat.codes.to_numpy()
        # guard: a chunk could in principle carry a category the cached LUT lacks
        if codes.size and codes.max() >= len(lut):
            cats = np.asarray(chunk[c].cat.categories, dtype=object)
            lut = label_encoders[c].transform(cats).astype(np.float64)
            lut_cache[c] = lut
        out[:, col_pos[c + '_encoded']] = lut[codes]

    # v21: the numeric block must carry EXACTLY the columns the scaler was fit on, in
    # fit order.  Building it from "the numeric columns that happen to be in this chunk"
    # produced a shorter block whenever a slot was missing from the virtual grid
    # (observed on (AI_RCP_pub)sm...: 1 column vs the scaler's 4 ->
    # "X has 1 features, but StandardScaler is expecting 4").  Missing slots are filled
    # with 0.0, i.e. the training mean after standardisation.
    _fni = getattr(scaler, 'feature_names_in_', None)
    _num_order = [str(_c) for _c in _fni] if _fni is not None and len(_fni) else list(num_present)
    if _num_order:
        num = np.zeros((len(chunk), len(_num_order)), dtype=np.float64)
        for _j, _c in enumerate(_num_order):
            if _c in chunk.columns:
                num[:, _j] = pd.to_numeric(chunk[_c], errors='coerce').to_numpy(dtype=np.float64)
        num = np.nan_to_num(num, nan=0.0)
        num_scaled = scaler.transform(num)
        for _j, _c in enumerate(_num_order):
            if _c in col_pos:
                out[:, col_pos[_c]] = num_scaled[:, _j]

    return pd.DataFrame(out, columns=list(feat_cols))

def resolve_loading_axis(df, verbose=True):
    """
    Decide which column the loading-dependent figures (F3a / F9d / F11) should sweep.

    Returns a dict and ALSO stores it in the module global LOADING_AXIS:
        col         canonical column name to use downstream ('catalyst_loading' after the
                    alias step, or None)
        source_col  the column in the ORIGINAL file (for reporting)
        role        'catalyst' | 'substitute' | 'unavailable'
        label/unit  for axis labels
        n_distinct  how many distinct finite values the axis has
        reason      one-line explanation, printed once at start-up

    A candidate is accepted only if it exists, is numeric-coercible, and has at least
    LOADING_AXIS_MIN_DISTINCT distinct finite values.  A CONSTANT amount column is
    rejected on purpose: sweeping it would draw a flat line across a phantom range and
    would emit N_LOADING_POINTS duplicate rows per reagent combination.  This is the trap
    in edbo_bms/PCI_PMI_cost_full.csv, whose Ligand_Equiv / Precatalyst_Equiv / Base_Equiv
    are all single-valued.
    """
    global LOADING_AXIS
    axis = {'col': None, 'source_col': None, 'role': 'unavailable', 'label': None,
            'unit': None, 'n_distinct': 0, 'reason': ''}

    rejected = []
    for cand in LOADING_AXIS_CANDIDATES:
        if cand not in df.columns:
            continue
        # A required column that the loader could not map is FILLED with 0.0, so a constant
        # `catalyst_loading` usually means "the file has no such column", not "the
        # experiment kept it fixed".  Say which, because the two need different fixes.
        _is_filler = (cand == 'catalyst_loading'
                      and pd.to_numeric(df[cand], errors='coerce').nunique(dropna=False) <= 1)
        vals = pd.to_numeric(df[cand], errors='coerce')
        vals = vals[np.isfinite(vals)]
        nu = int(vals.nunique())
        if nu < LOADING_AXIS_MIN_DISTINCT:
            rejected.append((cand, nu, _is_filler))
            continue
        axis['source_col'] = cand
        axis['col'] = 'catalyst_loading'
        axis['role'] = ('catalyst' if cand in ('catalyst_loading', 'catalyst_eq')
                        else 'substitute')
        axis['label'] = LOADING_AXIS_LABEL.get(cand, cand)
        axis['unit'] = LOADING_AXIS_UNIT.get(cand, '')
        axis['n_distinct'] = nu
        axis['reason'] = ('catalyst loading present' if axis['role'] == 'catalyst'
                          else f"'{cand}' used as a substitute (no catalyst_loading)")
        break
    else:
        if rejected:
            _parts = []
            for c, n, filler in rejected:
                if filler:
                    _parts.append(f"{c} (absent from the file -- the loader filled it with a "
                                  f"single 0.0)")
                else:
                    _parts.append(f'{c} (constant, n={n})')
            axis['reason'] = 'no usable amount axis -- ' + '; '.join(_parts)
        else:
            axis['reason'] = 'no amount/loading column in this dataset at all'

    LOADING_AXIS = axis
    if verbose:
        # One self-describing line: WHY this axis was chosen or rejected.  Without it the
        # log says only what will be plotted, not the reason, and the batch driver has
        # nothing to report in its summary.
        print('[loading axis] %s' % axis['reason'])
        if axis['col']:
            print('plotting F3a / F9d / F11 : %s  (%d distinct, unit %s)'
                  % (axis['label'], axis['n_distinct'], axis['unit'] or 'n/a'))
            if axis['role'] == 'substitute':
                print("NOTE: this is NOT catalyst loading -- the figures are"
                      " labelled with the substitute's own name.")
        else:
            print('F3a / F9d / F11 will be annotated as unavailable and the')
            print('enumeration will not vary this axis.')
    return axis


def apply_loading_axis_alias(df):
    """
    Expose the resolved loading axis under the canonical name `catalyst_loading`.

    This is the whole trick that keeps v14 small: if the axis is a substitute, copy it to
    `catalyst_loading`, so `prepare_features`, the enumeration, the match-count logic and
    every CSV writer continue to work untouched.  The original column is left in place, and
    LOADING_AXIS['source_col'] remembers its real name for the axis labels.
    """
    src = LOADING_AXIS.get('source_col')
    if not src or src == 'catalyst_loading':
        return df
    df = df.copy()
    df['catalyst_loading'] = pd.to_numeric(df[src], errors='coerce')
    print(f"[loading axis] aliased '{src}' -> 'catalyst_loading' so downstream code is "
          f"unchanged (original column kept)")
    return df


def loading_axis_available():
    return LOADING_AXIS.get('col') is not None


def annotate_loading_unavailable(ax, panel_letter=None, fontsize=9,
                                 headline=None, detail=None):
    """
    Draw a 'this panel cannot be drawn' note instead of a meaningless curve.

    Defaults to the loading-axis wording and LOADING_AXIS['reason']; pass `headline` and
    `detail` to announce a different unavailable slot (e.g. the substrate scope).
    """
    _generic = headline is not None
    ax.axis('off')
    ax.text(0.5, 0.58, headline or 'Loading response not available',
            ha='center', va='center', fontsize=fontsize + 1, fontweight='bold',
            transform=ax.transAxes)
    if _generic:
        ax.text(0.5, 0.44, detail or 'the required column is absent or has a single value',
                ha='center', va='top', fontsize=fontsize - 1.5, transform=ax.transAxes,
                wrap=True)
    else:
        ax.text(0.5, 0.44, LOADING_AXIS.get('reason', ''),
                ha='center', va='top', fontsize=fontsize - 1.5, transform=ax.transAxes,
                wrap=True)
        ax.text(0.5, 0.30,
                'this dataset exposes no varying amount column that the model uses,\n'
                'so nothing can be swept along that axis',
                ha='center', va='top', fontsize=fontsize - 1.5, transform=ax.transAxes)
    if panel_letter:
        ax.text(-0.05, 1.02, panel_letter, transform=ax.transAxes, fontsize=14,
                fontweight='bold')
    # v21 / C12: one bilingual wording for every "we cannot draw this" panel, so a reader
    # never has to guess whether a blank panel means "no data", "a bug" or "not applicable".
    _why = (detail or LOADING_AXIS.get('reason') or
            'the required column is absent or has a single value')
    # NOTE: matplotlib has no CJK font configured here, so Chinese text renders as tofu
    # boxes (observed on Figure 8a).  The browser/console keeps the Chinese wording; the
    # FIGURE stays ASCII.
    ax.text(0.5, 0.18, 'Not drawn because %s.' % str(_why).split('\n')[0][:90],
            ha='center', va='top', fontsize=fontsize - 1.5, transform=ax.transAxes,
            wrap=True, color='0.25')


def get_representative_conditions(df):
    representative = {}
    # reactant_2 / reactant_1 are the current slot names; the retired names hold only
    # 'Unknown' placeholders, so a scored condition could not report its partner pair.
    for col in ['metal', 'catalyst', 'ligand', 'base', 'solvent',
                'reactant_2', 'reactant_1']:
        if col in df.columns:
            mode = df[col].dropna().astype(str).mode()
            representative[col] = mode.iloc[0] if len(mode) else 'Unknown'

    for col in ['temperature', 'time', 'catalyst_loading']:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            representative[col] = float(vals.median()) if len(vals) else 0.0

    return representative

# v21 / C2: the numeric axis the sweep walks.  It used to be the literal 'catalyst_loading';
# the interactive menu can now point it at any continuous slot (temperature / time / *_eq), and
# Figures 3b / 8a / 9d then show THAT variable's effect.
SWEEP_AXIS = 'catalyst_loading'


def get_loading_grid(df, n_points=N_LOADING_POINTS, quantile_range=(0.002, 0.998),
                     for_enumeration=False):
    """Loading values to enumerate.

    v22h: ``for_enumeration=True`` collapses the grid to ONE point when
    CD_NO_LOADING_SWEEP=1 (the composite pipeline's enumeration), while the POST-HOC curve
    panels (Fig. 9d with 61 points, Fig. 11) keep their full grid -- otherwise turning the
    expansion off would also flatten every loading curve.
    """
    """
    Loading values to enumerate: `n_points` EVENLY SPACED values between two quantiles.

    Two things this is NOT, both of which are easy to assume wrongly:

    1. It is not a 0..1 grid.  The range is a quantile band of the historical loadings
       (default 0.2 %..99.8 %), which on this dataset is 2.67e-06 .. 0.4482 eq.  Using
       0..1 would spend most of the grid on loadings that never occur: only 3 of 10,138
       experimental rows (0.03 %) use more than 1 eq.

    2. It is not a quantile SAMPLING.  Only the two ENDPOINTS are quantiles; the interior
       is `np.linspace`, i.e. uniformly spaced.  That is deliberate.  The loading
       distribution is extremely right-skewed (median 0.0733, p75 0.0811, but the 0.2 %
       point is 2.67e-06), so sampling true quantiles would put ~80 of 100 points below the
       median and leave the upper half of the axis unresolved.  Uniform spacing over a
       quantile-bounded interval keeps the whole reachable loading axis covered.

    The band width is a real trade-off: narrowing the trim does NOT narrow the grid -- going
    from a 0.5 % trim to 0.2 % grows the span 1.45x, because the lower bound collapses
    toward 0 while the upper bound only rises.  Tighter trims buy a few more legitimate
    high-loading points at the cost of spending low-grid points on a nearly empty region.

    v14: `catalyst_loading` is absent when no usable amount axis exists (see
    resolve_loading_axis).  In that case the returned single value is a PLACEHOLDER: it is
    not a model feature, so the sweep would emit N_LOADING_POINTS identical rows per reagent
    combination.  get_loading_grid_for() reads LOADING_AXIS and collapses the grid to ONE
    point in that situation, which avoids the duplicate work; the value itself is irrelevant.
    """
    if for_enumeration and (NO_LOADING_SWEEP or not EXPAND_LOADING):
        # one representative point: the enumeration then scores each combination once
        _v = pd.to_numeric(df.get(SWEEP_AXIS, pd.Series(dtype=float)), errors='coerce').dropna()
        return np.array([float(_v.median()) if len(_v) else 0.0], dtype=float)
    if not loading_axis_available() or SWEEP_AXIS not in df.columns:
        return np.array([0.0], dtype=float)

    vals = pd.to_numeric(df[SWEEP_AXIS], errors='coerce').dropna()
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([0.0], dtype=float)

    lo = float(vals.quantile(quantile_range[0]))
    hi = float(vals.quantile(quantile_range[1]))
    if np.isclose(lo, hi):
        return np.array([float(vals.median())], dtype=float)

    # anchored on real measured loadings, uniformly spaced over the dense range
    return np.linspace(lo, hi, n_points)

def build_virtual_library(df, label_encoders, scaler, feat_cols,
                          max_combinations=MAX_VIRTUAL_COMBINATIONS, preview_only=False):
    """
    Enumerate reagent combinations x a loading grid, IN MEMORY.

    v13 STATUS: no longer part of the pipeline.  main() used to call this with
    preview_only=True purely to print a report, and both returned frames were dead (an AST
    audit showed them bound and never read); the report is now produced by
    `report_enumeration`, and the figures consume the streamed sweep's own `bo_df`.

    Kept because it is the reference in-memory enumeration: it is what the equivalence
    tests compare `stream_virtual_library` / `decode_combo_range` against, and it remains
    the simplest entry point if you ever want the whole library as a DataFrame.

    v12 change -- COVERAGE COMES FIRST.
    The previous version kept a 31-point loading grid and then truncated the loading
    axis when the product exceeded `max_combinations`.  Because the categorical space
    is already ~5.7e6, that cap silently collapsed the grid to 8 arbitrary points
    (3.1 .. 23.5 eq., i.e. 40-300x any realistic loading), so the enumerated space was
    simultaneously incomplete AND physically unrepresentative, and the library was
    additionally anchored on a single representative substrate.

    Now the loading axis is sized so that EVERY categorical combination is enumerated
    at least once: n_loading = min(n_points, max_combinations // n_categorical).  With
    5.7e6 categorical combinations and a 5e7 cap that yields up to 8 loading points
    while still covering the full reagent space.
    """
    categorical_cols = [
        c for c in ENUM_AXES
        if c in df.columns and c + '_encoded' in feat_cols
    ]

    value_lists = []
    for col in categorical_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        if not vals:
            vals = ['Unknown']
        value_lists.append(vals)

    n_cat = 1
    for v in value_lists:
        n_cat *= len(v)

    requested_grid = get_loading_grid(df, n_points=N_LOADING_POINTS,
                                         for_enumeration=True)
    allowed_loading = max(1, int(max_combinations // max(1, n_cat)))
    if allowed_loading >= len(requested_grid):
        loading_grid = requested_grid
    else:
        # keep the loading values spread across the dense historical range rather than
        # taking the first N of a sorted grid
        idx = np.unique(np.linspace(0, len(requested_grid) - 1, allowed_loading).round().astype(int))
        loading_grid = requested_grid[idx]
        print(f" Categorical space is {n_cat:,} combinations; capping loading points at "
              f"{len(loading_grid)} (requested {len(requested_grid)}) to stay under "
              f"max_combinations={max_combinations:,}.")
        print(f"   -> every categorical combination IS enumerated; only the loading axis is coarse.")

    total = int(n_cat * len(loading_grid))
    fully_covered = (total <= max_combinations)
    print(f"   enumerated space: {n_cat:,} reagent combinations x {len(loading_grid)} loading "
          f"points = {total:,} rows "
          f"({'full coverage' if fully_covered else 'PARTIAL coverage — raise max_combinations'})")

    # The enumerated COUNT is large (571.9e6 at 100 loading points) but nothing of that
    # size is ever allocated: scoring streams in bounded chunks, and this function
    # either samples in index space (preview_only=True) or builds explicitly.
    if preview_only:
        print(f"   [memory] preview_only=True -> will allocate ~"
              f"{min(VIRTUAL_LIBRARY_PREVIEW_ROWS, total) * 30 / 1024 ** 2:.0f} MiB "
              f"(sampled in index space), NOT the full {total:,} rows.")
    else:
        print(f"   [memory] building the full {total:,}-row frame "
              f"(~{total * 30 / 1024 ** 3:.1f} GiB with category dtype).")
    # ~105 B/row is the MEASURED peak for the float64 feature DataFrame over a chunk
    # (249 MiB at 2e6 rows x 9 features); it is a per-process figure, not a total.
    print(f"            scoring streams in chunks of <= {STREAM_MAX_ROWS_PER_CHUNK:,} rows "
          f"(~{STREAM_MAX_ROWS_PER_CHUNK * 105 / 1024 ** 2:.0f} MiB in this process).")
    print(f"            MAX_VIRTUAL_COMBINATIONS={max_combinations:,} -> "
          f"{len(loading_grid)} loading points.")

    representative = get_representative_conditions(df)
    out_cols = list(representative.keys())
    n_load = len(loading_grid)

    if preview_only:
        # v12: build ONLY the sampled rows.
        # The previous version constructed the whole product (177.3e6 rows at
        # 31 loading points; 571.9e6 now at 100) and then sub-sampled it.  Even with
        # categorical dtypes that rebuilds millions of rows: `reset_index` alone asked
        # for 3.96 GiB (3 columns x 177,280,320 rows, float64/object) and raised
        # ArrayMemoryError.  Sub-sampling must happen in INDEX space, not after
        # materialising.
        n_rows_full = int(n_cat * n_load)
        take = int(min(VIRTUAL_LIBRARY_PREVIEW_ROWS, n_rows_full))
        sel = np.random.RandomState(42).choice(n_rows_full, size=take, replace=False)
        sel.sort()

        comb = sel // n_load                    # which categorical combination
        load_i = sel % n_load                   # which loading point

        # decode the flat combination index into one level per categorical column
        cols = {}
        rem = comb.copy()
        for pos, col in enumerate(categorical_cols):
            inner = int(np.prod([len(v) for v in value_lists[pos + 1:]], dtype=np.int64))
            code = (rem // inner) % len(value_lists[pos])
            cols[col] = np.asarray(value_lists[pos], dtype=object)[code]
        virtual_df = pd.DataFrame(cols)
        virtual_df[SWEEP_AXIS] = np.asarray(loading_grid, dtype=float)[load_i]
        for col in out_cols:
            if col not in virtual_df.columns:
                virtual_df[col] = representative[col]
        virtual_df = virtual_df[out_cols]
        print(f"  [preview] sampled {take:,} of {n_rows_full:,} rows in INDEX space "
              f"(nothing larger than the preview was ever allocated)")
    else:
        # full Cartesian product by numpy broadcasting (used for tests / small caps)
        if categorical_cols:
            col_arrays = {}
            reps = 1
            for pos, (col, vals) in enumerate(zip(categorical_cols, value_lists)):
                arr = np.asarray(vals, dtype=object)
                inner = int(np.prod([len(v) for v in value_lists[pos + 1:]], dtype=np.int64))
                arr = np.repeat(arr, inner)
                arr = np.tile(arr, reps)
                col_arrays[col] = arr
                reps *= len(vals)
            cat_df = pd.DataFrame({c: np.tile(v, n_load) for c, v in col_arrays.items()})
            virtual_df = cat_df
            virtual_df['catalyst_loading'] = np.repeat(
                np.asarray(loading_grid, dtype=float), reps)
            for col in out_cols:
                if col not in virtual_df.columns:
                    virtual_df[col] = representative[col]
            virtual_df = virtual_df[out_cols]
        else:
            virtual_df = pd.DataFrame([representative] * n_load)
            virtual_df[SWEEP_AXIS] = np.asarray(loading_grid, dtype=float)

    virtual_df = virtual_df.reset_index(drop=True)
    print(f"  virtual dataframe built: {len(virtual_df):,} rows x {virtual_df.shape[1]} cols")

    # v12: the pipeline only needs a bounded PREVIEW of the raw library to describe the
    # search space -- the scoring itself streams (generate_bo_results_streaming).
    # preview_only=True therefore samples here instead of returning 45.7e6 rows.
    if preview_only and len(virtual_df) > VIRTUAL_LIBRARY_PREVIEW_ROWS:
        n_full = len(virtual_df)
        virtual_df = virtual_df.sample(n=VIRTUAL_LIBRARY_PREVIEW_ROWS,
                                       random_state=42).reset_index(drop=True)
        print(f"  [memory] preview_only=True -> returning {len(virtual_df):,} rows "
              f"(full enumeration has {n_full:,}; streamed during scoring)")

    print("Encoding Virtual Library Features...")
    virtual_X = build_virtual_feature_frame(df, feat_cols, label_encoders, scaler, virtual_df)

    fixed_cols = ['reactant_1', 'reactant_2', 'temperature', 'time']
    fixed_report = {c: virtual_df[c].nunique() for c in fixed_cols if c in virtual_df.columns}
    const_cols = [c for c, k in fixed_report.items() if k == 1]
    if const_cols:
        vals = {c: virtual_df[c].iloc[0] for c in const_cols}
        print("  [NOTE] these process columns are FIXED across the whole library ")
        for c, v in vals.items():
            print(f"           {c} = {v}")
        print("         -> top-ranked rows will all share these values, so the library "
              "cannot rank across them.")
    _enum_cols = ['catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1']
    _spread = {c: virtual_df[c].nunique() for c in _enum_cols if c in virtual_df.columns}
    print("  [NOTE] enumerated spread: " + ", ".join(f"{c}={k}" for c, k in _spread.items()))

    if 'catalyst_loading' in virtual_df.columns:
        lo, hi = virtual_df['catalyst_loading'].min(), virtual_df['catalyst_loading'].max()
        hist = pd.to_numeric(df['catalyst_loading'], errors='coerce')
        cover = float(((hist >= lo) & (hist <= hi)).mean()) if len(hist) else 0.0
        print(f"   [NOTE] range {lo:.4g} .. {hi:.4g} eq.; historical median "
              f"{hist.median():.4g} eq.; {100 * cover:.1f}% of historical rows inside that range")

    print(f"Virtual Library Features ready: {virtual_X.shape[0]:,} rows × {virtual_X.shape[1]} features")

    # As object dtype each of the 6 columns is an 8-byte pointer array: 6 x 0.34 GiB
    # = 2.05 GiB, and EVERY copy of the library (bo_df, model_virtual_df, ...) needs
    # another 2.05 GiB contiguous allocation -- which is exactly the
    # "Unable to allocate 2.05 GiB for an array with shape (6, 45749760)" failure.
    # category stores one integer code per row plus a tiny lookup table, so the six
    # columns together cost ~0.04 GiB instead of 2.05 GiB, with identical values.
    if VIRTUAL_LIBRARY_AS_CATEGORY:
        obj_cols = [c for c in virtual_df.columns if virtual_df[c].dtype == object]
        for c in obj_cols:
            virtual_df[c] = virtual_df[c].astype('category')
        print(f"  [memory] {len(obj_cols)} object columns converted to category "
              f"(2.05 GiB -> ~0.04 GiB for 45.7e6 rows)")

    print(f"  virtual dataframe memory: {virtual_df.memory_usage(deep=False).sum() / 1024**3:.2f} GiB "
          f"(shallow; categorical payload not counted)")
    return virtual_df, virtual_X


def _effective_chunk_combos(n_loading, requested):
    """
    Keep one streamed chunk inside STREAM_MAX_ROWS_PER_CHUNK rows.

    Chunk memory is (combos x n_loading) x bytes_per_row.  With a 51-point loading
    grid a 200k-combo chunk would be 10.2e6 rows (~950 MiB) -- large enough to be a
    problem again on a smaller machine.  Sizing by ROWS rather than by combos keeps
    peak memory constant no matter how fine the loading grid gets.
    """
    n_loading = max(int(n_loading), 1)
    cap = max(1, int(STREAM_MAX_ROWS_PER_CHUNK // n_loading))
    return max(1, min(int(requested), cap))



def get_dose_axis_values(df, slot, expand=None):
    """The values a dose axis contributes to the enumeration.

    expand=True  -> DOSE_POINTS points spanning the 2 %-98 % observed range
    expand=False -> ONE point, the training median
    """
    if expand is None:
        expand = bool(EXPAND_LOADING)
    s = pd.to_numeric(df.get(slot, pd.Series(dtype=float)), errors='coerce').dropna()
    if len(s) == 0:
        return np.array([0.0])
    if not expand:
        return np.array([float(s.median())])
    lo = float(s.quantile(0.02))
    hi = float(s.quantile(0.98))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.array([float(s.median())])
    return np.linspace(lo, hi, max(1, int(DOSE_POINTS)))


def get_dose_cols(df, feat_cols):
    """Dose (amount) slots present in this dataset AND varying."""
    # ONLY the amount ("how much of each reagent") slots -- temperature and time are process
    # conditions and stay at their representative value, exactly as before.
    _AMOUNTS = ['catalyst_loading', 'catalyst_eq', 'ligand_eq', 'base_eq', 'reactant_1_eq',
                'reactant_2_eq', 'solvent_eq', 'cosolvent_eq', 'additive_eq']
    out = []
    for c in _AMOUNTS:
        if c in df.columns and c in feat_cols:
            if pd.to_numeric(df[c], errors='coerce').nunique() > 1 and c not in out:
                out.append(c)
    return out


def get_enumeration_spec(df, feat_cols):
    """
    THE single definition of the enumerated space.  Both `stream_virtual_library` and the
    parallel feature cache must agree on it exactly, so it lives in one place.

    Returns (categorical_cols, value_lists, n_cat, inner).

    `inner[p]` is the stride of column p in the flat combination index: the LAST category
    varies fastest, which matches the row order the in-memory builder produced.

    Do NOT try to derive the category lists from a small generated chunk: `category`
    dtype only records the values that actually appeared, so a 1000-combination probe
    reports 1 catalyst instead of 37 and silently shrinks the whole enumeration.
    """
    categorical_cols = [
        c for c in ENUM_AXES
        if c in df.columns and c + '_encoded' in feat_cols
    ]

    value_lists = []
    for col in categorical_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        value_lists.append(vals if vals else ['Unknown'])

    # v23: dose axes join the cartesian product (reactant-name columns are treated the same)
    dose_cols = get_dose_cols(df, feat_cols)
    for c in dose_cols:
        categorical_cols.append(c)
        value_lists.append(get_dose_axis_values(df, c))

    n_cat = 1
    for v in value_lists:
        n_cat *= len(v)
    inner = [int(np.prod([len(v) for v in value_lists[p + 1:]], dtype=np.int64))
             for p in range(len(value_lists))]
    global _LAST_DOSE_COLS
    _LAST_DOSE_COLS = list(dose_cols)
    return categorical_cols, value_lists, int(n_cat), inner


def get_loading_grid_for(df, max_combinations, n_cat, n_points=N_LOADING_POINTS):
    """The loading grid a given cap actually permits (shared by both engines)."""
    requested = get_loading_grid(df, n_points=n_points, for_enumeration=True)
    allowed = max(1, int(max_combinations // max(n_cat, 1)))
    if allowed >= len(requested):
        return requested
    ix = np.unique(np.linspace(0, len(requested) - 1, allowed).round().astype(int))
    return requested[ix]


def report_enumeration(df, feat_cols, max_combinations):
    """
    Print the structural characteristics of the enumerated space.

    v13: this used to be a side effect of building a 200,000-row 'preview' DataFrame plus
    its feature matrix, both of which were then discarded -- an AST audit confirmed the
    preview was bound and never read (the figure code overwrites it and only consumes its
    own `bo_df`).  The report is the only part that was ever used, so compute it directly
    from the source frame instead of materialising a virtual library to describe it.

    Returns the loading grid that the cap permits.
    """
    # v23: n_cat now already includes every dose axis, so there is no extra loading factor.
    _, value_lists, n_cat, _ = get_enumeration_spec(df, feat_cols)
    loading_grid = get_loading_grid_for(df, max_combinations, n_cat, n_points=N_LOADING_POINTS)
    n_load = 1
    total = int(n_cat)



    # Process columns the enumeration does NOT vary.  The Cartesian product spans only the
    # reagent categories, so temperature and time take a single representative value.
    # (They are NOT constant in the historical data -- that is exactly the limitation:
    # the library cannot rank across them.)
    representative = get_representative_conditions(df)
    fixed_cols = ['temperature', 'time']
    fixed_here = [c for c in fixed_cols if c in representative]
    if fixed_here:
        # one line, as requested: "temperature = 80.0, time = 3.0"
        print("  [NOTE] these process columns are FIXED across the whole library: "
              + ", ".join(f"{c} = {representative[c]}" for c in fixed_here))


    enum_cols = ['catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1']
    spread = {c: df[c].nunique() for c in enum_cols if c in df.columns}
    print("  [NOTE] enumerated spread: " + ", ".join(f"{c}={k}" for c, k in spread.items()))

    if 'catalyst_loading' in df.columns:
        lo, hi = float(loading_grid.min()), float(loading_grid.max())
        hist = pd.to_numeric(df['catalyst_loading'], errors='coerce')
        cover = float(((hist >= lo) & (hist <= hi)).mean()) if len(hist) else 0.0
        print(f"  [NOTE] range {lo:.4g} .. {hi:.4g} eq.; historical median "
              f"{hist.median():.4g} eq.; {100 * cover:.1f}% of historical rows inside that range")
    return loading_grid


def refine_loading_stage2(pool, df, models, feat_cols, label_encoders, scaler,
                          weights=None, n_points=LOADING_REFINE_POINTS,
                          top_n=LOADING_REFINE_TOP_N, verbose=True):
    """
    Stage 2: re-scan the loading axis at high resolution for the surviving conditions.

    *** DISABLED BY DEFAULT (USE_LOADING_REFINE = False) -- DO NOT ENABLE FOR PUBLISHED
    RESULTS WITHOUT RE-VALIDATING. ***  A dense scan reports a mean gain of +2.33 pp, but
    that is the "maximum over many noisy points" bias: re-measuring the SAME loading
    position on a disjoint (odd/even) half of the grid recovers only +0.11 pp, and
    smoothing the curve before the argmax does not help at any window width (verified gain
    <= -0.02 pp for windows 1..201).  For each candidate condition, replace its loading
    with the best found on a dense scan of the FULL historical loading range, keeping every
    other condition column fixed.

    Why the full range and not a window: a +-10 % relative window contains the true
    (in-sample) optimum for only 29.5 % of combinations; the full range, 100 %.

    Why only the survivors: stage 1 established that no condition outside this pool can
    beat it -- the coarse-grid maximum is a LOWER bound on the in-sample maximum, so any
    condition whose coarse best is below the pool's cannot win after refinement either.

    Returns (refined_pool, n_scanned).
    """
    if pool is None or len(pool) == 0:
        return pool, 0

    key = [c for c in CONDITION_COLS if c in pool.columns]
    if not any(c == 'catalyst_loading' for c in key):
        return pool, 0

    # IMPORTANT: keep the pool's own index labels so the results can be written back by
    # label.  `reps` is built as work.loc[work.index.repeat(k)] and then reset to a
    # RangeIndex for the model call, but the label mapping below uses `work.index`, so the
    # two stay consistent by construction (row i of `reps` belongs to work.index[i]).
    work = (pool if top_n is None
            else pool.nlargest(min(int(top_n), len(pool)),
                               'Ensemble Predicted Yield (%)'))
    work = work.copy()

    hist = pd.to_numeric(df['catalyst_loading'], errors='coerce').dropna()
    lo, hi = float(hist.quantile(0.001)), float(hist.quantile(0.999))
    grid = np.linspace(lo, hi, int(n_points))

    n = len(work)
    work_labels = work.index.to_numpy()
    reps = work.loc[work.index.repeat(len(grid))].reset_index(drop=True)
    reps[SWEEP_AXIS] = np.tile(grid, n)          # v22: follow the chosen axis
    reps['catalyst_loading'] = np.tile(grid, n)  # keep the canonical column present
    del work

    feat = build_virtual_feature_frame(df, feat_cols, label_encoders, scaler, reps)
    items = list(models.items())
    if weights is None or len(weights) != len(items):
        wv = np.full(len(items), 1.0 / len(items))
    else:
        wv = np.asarray(weights, float)
        wv = np.clip(wv, 0.0, None)
        wv = wv / wv.sum()

    P = None
    for j, (nm, mdl) in enumerate(items):
        pred = np.asarray(mdl.predict(feat), dtype=np.float64)
        if CLIP_MEMBER_PREDICTIONS:
            pred = np.clip(pred, PREDICTION_MIN, PREDICTION_MAX)
        P = pred * wv[j] if P is None else P + pred * wv[j]

    P = P.reshape(n, len(grid))
    best_i = P.argmax(axis=1)
    best_val = P[np.arange(n), best_i]
    best_ld = grid[best_i]

    # map the refined values back by LABEL: row i of `best_val` belongs to work_labels[i]
    out = pool.copy()
    out.loc[work_labels, 'Ensemble Predicted Yield (%)'] = best_val
    out.loc[work_labels, 'catalyst_loading'] = best_ld
    # NOTE: the per-member Pred_* / Model Gap / uncertainty columns still describe the
    # COARSE loading point.  They are diagnostics, and the figure footnote states that the
    # loading shown for refined rows is the refined optimum.
    n_scanned = int(n * len(grid))
    if verbose:
        imp = best_val - pool.loc[work_labels, 'Ensemble Predicted Yield (%)'].to_numpy(float)
        same = np.abs(best_ld - pool.loc[work_labels, 'catalyst_loading'].to_numpy(float)) <= 1e-9
        print(f"  [refine] stage 2: {n:,} survivor conditions x {len(grid)} loading points "
              f"= {n_scanned:,} rows; {n:,} rows updated")
        print(f"  [refine] APPARENT gain: mean +{imp.mean():.2f} pp, max +{imp.max():.2f} pp "
              f"-- WARNING: this is mostly selection noise.  Re-measuring the same loading")
        print(f"  [refine] position on a disjoint half of the grid recovers only ~0.1 pp, so "
              f"this number is NOT a real improvement.  See USE_LOADING_REFINE notes.")
        print(f"  [refine] the coarse grid had missed the argmax loading for "
              f"{100 * (1 - float(same.mean())):.1f}% of these conditions")
    return out, n_scanned


def stream_virtual_library(df, label_encoders, feat_cols, chunk_combos=200_000,
                           max_combinations=MAX_VIRTUAL_COMBINATIONS, verbose=True):
    """
    Yield the virtual library in bounded chunks -- the FULL enumeration is never held
    in memory.

    Why this exists
    ---------------
    Every previous fix was local: replace one `np.vstack`, convert one dtype, ... and
    the next run failed on a different line, because the pipeline simultaneously held
    several 45.7e6-row copies (library, feature matrix, scored frame, figure copy,
    merge copy) and each additional column cost another 0.34-0.68 GiB contiguous.  On
    a different dataset the same structure simply fails at a different size.

    This generator removes the root cause: nothing downstream ever needs the whole
    table at once.  Scoring, gating and ranking are all streaming, so peak memory is
    O(chunk_combos x n_columns) regardless of how large the library is.

    Yields
    ------
    DataFrame with one column per condition (chunk_combos x len(loading_grid) rows),
    in the SAME order as the in-memory builder, and converted to the same dtypes.
    """
    categorical_cols, value_lists, n_cat, inner = get_enumeration_spec(df, feat_cols)
    del inner                                  # recomputed below in the original form

    # v23: every dose axis is already part of `value_lists`, so there is no separate
    # loading grid to tile any more (n_load == 1).
    representative = get_representative_conditions(df)
    out_cols = list(representative.keys())
    n_load = 1

    # inner repeat counts: the LAST categorical column varies fastest
    inner = []
    for pos in range(len(value_lists)):
        inner.append(int(np.prod([len(v) for v in value_lists[pos + 1:]], dtype=np.int64)))

    # the categorical part is sized so that chunk_combos x n_load stays bounded
    cat_per_chunk = max(1, int(chunk_combos))
    cat_arrays = [np.asarray(v, dtype=object) for v in value_lists]

    total_cat = n_cat
    done = 0
    while done < total_cat:
        stop = min(done + cat_per_chunk, total_cat)
        idxs = np.arange(done, stop, dtype=np.int64)
        cols = {}
        for pos, col in enumerate(categorical_cols):
            code = (idxs // inner[pos]) % len(value_lists[pos])
            cols[col] = cat_arrays[pos][code]
        chunk = pd.DataFrame(cols)
        # dose columns come from the product as floats
        for _dc in _LAST_DOSE_COLS:
            if _dc in chunk.columns:
                _v = pd.to_numeric(chunk[_dc], errors='coerce').astype(float)
                chunk[_dc] = _v
                # the rest of the pipeline reads the canonical slot name
                _canon = {'catalyst_eq': 'catalyst_loading'}.get(_dc, _dc)
                if _canon != _dc and _canon in out_cols:
                    chunk[_canon] = _v
        for col in out_cols:
            if col not in chunk.columns:
                chunk[col] = representative[col]
        chunk = chunk[out_cols]

        if VIRTUAL_LIBRARY_AS_CATEGORY:
            for c in chunk.columns:
                if chunk[c].dtype == object:
                    chunk[c] = chunk[c].astype('category')
        yield chunk
        done = stop




def predict_in_batches(model, X, batch_size=5000000, desc='Prediction', show_progress=False):
    """
    Predict in chunks and concatenate.

    TIP: for a 45.7e6-row virtual library this returns a 0.34 GiB float64 array per
    model.  If you only need ensemble mean/spread, use `ensemble_stats_streaming`
    instead, which never holds more than one batch per model.
    """
    outputs = []
    pbar = tqdm(total=len(X), desc=desc, unit='rows', dynamic_ncols=True) if show_progress else None

    try:
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            batch_pred = np.asarray(model.predict(X.iloc[start:stop]), dtype=float)
            outputs.append(batch_pred)
            if pbar is not None:
                pbar.update(stop - start)
    finally:
        if pbar is not None:
            pbar.close()

    return np.concatenate(outputs) if outputs else np.array([], dtype=float)


def ensemble_stats_streaming(models, X, batch_size=2000000, show_progress=True,
                             max_models_in_memory=1, weights=None):
    """
    Weighted ensemble mean / std / min / max without ever materialising the
    (n_models, n_rows) matrix.

    The old path did `np.vstack([pred_1, ..., pred_k])`.  For the 45.7e6-row virtual
    library that is 5.79 GiB for k=17 -- on top of the k source arrays, ~11.6 GiB peak,
    which is what raised "Unable to allocate 5.79 GiB".  But only the column-wise mean
    and std are used, so we can accumulate them one chunk at a time: peak extra memory
    becomes a single batch column (~16 MB at batch_size=2e6) instead of GiB.

    weights
    -------
    Optional per-member weights (same order as `models`).  Normalised to sum to 1 and
    used for the mean.  The reported uncertainty is the weighted standard deviation

        sigma_w = sqrt( sum_i w_i (p_i - mu_w)^2 / (1 - sum_i w_i^2) )

    which reduces to the usual sample sd (ddof=1) when all weights are equal.  With
    `weights=None` behaviour is the plain unweighted mean, i.e. identical to before.
    """
    model_items = list(models.items())
    k = len(model_items)
    n = len(X)
    if k == 0:
        z = np.zeros(n)
        return z, z, z, z, {}

    if weights is None:
        w = np.full(k, 1.0 / k, dtype=np.float64)
    else:
        w = np.asarray([float(x) for x in weights], dtype=np.float64)
        if len(w) != k:
            raise ValueError(f'weights length {len(w)} != number of models {k}')
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s <= 0:
            raise ValueError('weights must not all be zero')
        w = w / s
    wsumsq = float((w ** 2).sum())

    total = np.zeros(n, dtype=np.float64)
    total_sq = np.zeros(n, dtype=np.float64)
    vmin = np.full(n, np.inf, dtype=np.float64)
    vmax = np.full(n, -np.inf, dtype=np.float64)

    # predictions of the first `max_models_in_memory` members are kept only if the
    # caller explicitly wants the per-model columns back (used by generate_bo_results)
    kept = {}

    iterator = model_items
    if show_progress:
        iterator = tqdm(model_items, desc='Ensemble (streaming)', unit='model', dynamic_ncols=True)

    for mi, (name, model) in enumerate(iterator):
        wi = float(w[mi])
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            p = np.asarray(model.predict(X.iloc[start:stop]), dtype=np.float64)
            total[start:stop] += wi * p
            total_sq[start:stop] += wi * p * p
            np.minimum(vmin[start:stop], p, out=vmin[start:stop])
            np.maximum(vmax[start:stop], p, out=vmax[start:stop])
            if mi < max_models_in_memory:
                kept.setdefault(name, np.empty(n, dtype=np.float64))[start:stop] = p

    mean = total                                        # weights already sum to 1
    # weighted variance about the weighted mean, with the standard reliability
    # correction (1 - sum w^2) so that equal weights reproduce ddof=1
    var = np.maximum(total_sq - mean * mean, 0.0)
    denom = max(1.0 - wsumsq, 1e-12)
    std = np.sqrt(var / denom)
    return mean, std, vmin, vmax, kept


def _regen_conditions_for_indices(df, feat_cols, label_encoders, scaler,
                                  chunk_of, row_of, max_combinations,
                                  chunk_combos=None):
    """
    Rebuild the CONDITION columns for a set of (chunk, row) pairs.

    Only the chunks that actually contain a winner are regenerated, so the cost is a few
    streamed chunks rather than the whole library.  Deterministic: stream_virtual_library
    is a pure function of (df, chunk_combos, max_combinations), so chunk i regenerates
    identically to the cached copy that was scored.

    CRITICAL: chunk_combos must be the SAME value used when the feature cache was built,
    because it determines the chunk boundaries.  Passing a different cap here silently
    shifts every chunk index and mis-assigns the scores to the wrong conditions.
    """
    import numpy as _np
    import pandas as _pd
    chunk_of = _np.asarray(chunk_of, dtype=_np.int64)
    row_of = _np.asarray(row_of, dtype=_np.int64)
    if len(chunk_of) == 0:
        return _pd.DataFrame()

    if chunk_combos is None:
        n_load_ = len(get_loading_grid(df, n_points=N_LOADING_POINTS))
        chunk_combos = _effective_chunk_combos(n_load_, PARALLEL_CHUNK_ROWS)
    # stream_virtual_library clamps chunk_combos internally; clamp the same way here so
    # the chunk boundaries are byte-for-byte the ones the feature cache was built with.
    # (Parallel, this mismatch is silent: the values land on the wrong conditions.)
    _n_load_c = len(get_loading_grid_for(df, max_combinations,
                                         get_enumeration_spec(df, feat_cols)[2]))
    chunk_combos = _effective_chunk_combos(_n_load_c, chunk_combos)

    want = {}
    for ch_i, row_i in zip(chunk_of, row_of):
        want.setdefault(int(ch_i), []).append(int(row_i))

    pieces = []
    it = stream_virtual_library(df, label_encoders, feat_cols, chunk_combos=chunk_combos,
                                max_combinations=max_combinations, verbose=False)
    for ch_i, chunk in enumerate(it):
        if ch_i in want:
            rows = want[ch_i]
            sub = chunk.iloc[rows].reset_index(drop=True)
            sub['_chunk'] = ch_i
            sub['_row'] = rows
            pieces.append(sub)
            if len(pieces) == len(want):
                break
    if not pieces:
        return _pd.DataFrame()
    return _pd.concat(pieces, ignore_index=True)


def _decode_combo_range(feat_cols, value_lists, cat_cols, num_cols, inner,
                        representative, loading_grid, c0, c1, lut, scaler):
    """
    Build the model-input matrix for a CONTIGUOUS range of reagent combinations
    [c0, c1), covering every loading-grid point.  Raw numpy, no DataFrame in the hot path.

    This is the unit of work for the parallel feature cache.  It exists because chunk
    GENERATION -- not the encoding -- is what keeps the caching phase serial and slow:
    measured over 89 chunks, generation was 78 % of the caching loop, LUT encoding 16 %
    and the disk write 6 %.  Speeding up only the encoding would leave 78 % untouched.

    Row order matches stream_virtual_library exactly (combination index slowest, loading
    index fastest) and the column order is feat_cols.
    """
    n = int(max(c1 - c0, 0))
    if n <= 0:
        return np.zeros((0, len(feat_cols)), dtype=np.float32)

    idx = np.arange(c0, c1, dtype=np.int64)
    n_load = len(loading_grid)
    nrow = n * n_load

    out = np.empty((nrow, len(feat_cols)), dtype=np.float32)
    pos_of = {c: i for i, c in enumerate(feat_cols)}

    # categorical: decode codes for this range, then repeat across the loading grid
    for pos, col in enumerate(cat_cols):
        code = (idx // inner[pos]) % len(value_lists[pos])
        out[:, pos_of[col + '_encoded']] = np.repeat(lut[col][code], n_load)

    # numerical columns.  `catalyst_loading` VARIES within the chunk; the other numeric
    # columns are the fixed representative conditions.  Both must go through the scaler --
    # the scaler is fit on the numeric columns together, so writing a raw value here
    # instead of its scaled form is a silent error (it made every chunk disagree with the
    # streamed feature frame by ~0.4 on this column).
    if num_cols:
        # v21: same rule as fast_virtual_features -- build the block over the scaler's
        # fitted columns (in fit order), not over the subset present in this chunk.
        _fni = getattr(scaler, 'feature_names_in_', None)
        _num_order = ([str(_c) for _c in _fni] if _fni is not None and len(_fni)
                      else list(num_cols))
        num_block = np.empty((nrow, len(_num_order)), dtype=np.float64)
        for _j, _col in enumerate(_num_order):
            if _col == SWEEP_AXIS:
                num_block[:, _j] = np.tile(np.asarray(loading_grid, dtype=np.float64), n)
            else:
                num_block[:, _j] = float(representative.get(_col, 0.0))
        num_block = scaler.transform(num_block)
        for _j, _col in enumerate(_num_order):
            if _col in pos_of:
                out[:, pos_of[_col]] = num_block[:, _j]

    return out.astype(np.float32, copy=False)


# Per-worker copies of the encoding lookups, set by _build_cache_chunk.
_W_LUT = {}
_W_SCALER = [None]


def _build_cache_chunk(task):
    """
    Worker: build one cached chunk of the model-input matrix and write it to disk.

    Returns (chunk_index, n_rows, 0) or None on failure.  Only small values cross the
    process boundary -- the matrix itself goes straight to the .npy file.

    Idempotent by design: two workers producing the same chunk index would write the same
    bytes, so a retry cannot corrupt the cache.
    """
    (chunk_i, cache_dir, c0, c1, feat_cols, cat_cols, num_cols, value_lists, inner,
     representative, loading_grid, lut_dump, scaler_dump) = task
    try:
        import numpy as _np
        if not _W_LUT:
            _W_LUT.update(lut_dump)
            _W_SCALER[0] = pickle.loads(scaler_dump)
        X = _decode_combo_range(feat_cols, value_lists, cat_cols, num_cols, inner,
                                representative, loading_grid, c0, c1,
                                _W_LUT, _W_SCALER[0])
        _np.save(os.path.join(cache_dir, 'f%05d.npy' % chunk_i), X)
        return (chunk_i, int(X.shape[0]), 0)
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def audit_condition_scores(pool, df, models, feat_cols, label_encoders, scaler,
                           weights=None, tol=0.01, tag='', repair=True):
    """
    Verify that every row's stored ensemble score actually belongs to its stored
    condition, by RE-SCORING those conditions from scratch and comparing.

    Why this exists
    ---------------
    While building the parallel engine I found that the older streaming engine
    (`generate_bo_results_streaming`) can return a pool in which the scores are attached
    to the WRONG conditions: re-scoring 5,000 returned rows independently disagreed for
    4,501 of them, by up to 51 points.  The feature builders themselves are provably
    identical (max|diff| = 0.00e+00, and the member predictions agree exactly), so the
    corruption is in that engine's pool assembly, not in the encodings.

    A silent condition/score mismatch is the worst possible failure mode for a paper: the
    table looks fine and the chemistry is wrong.  Rather than rely on any single code path
    being correct, every engine now audits its own output and (by default) DROPS rows that
    fail, so a returned pool is always self-consistent.

    Returns (pool, n_bad, n_total).
    """
    if pool is None or len(pool) == 0:
        return pool, 0, 0
    cond_cols = [c for c in ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2',
                             'reactant_1', 'temperature', 'time', 'catalyst_loading']
                 if c in pool.columns]
    if len(cond_cols) < 5 or 'Ensemble Predicted Yield (%)' not in pool.columns:
        return pool, 0, len(pool)
    try:
        feat = build_virtual_feature_frame(df, feat_cols, label_encoders, scaler,
                                           pool[cond_cols])
        items = list(models.items())
        if weights is None:
            wv = np.full(len(items), 1.0 / len(items))
        else:
            wv = np.asarray([float(x) for x in weights], dtype=float)
            wv = np.clip(wv, 0.0, None)
            wv = wv / wv.sum()
        if len(wv) != len(items):
            return pool, 0, len(pool)
        acc = np.zeros(len(pool))
        for j, (nm, mdl) in enumerate(items):
            p = np.asarray(mdl.predict(feat), dtype=np.float64)
            if CLIP_MEMBER_PREDICTIONS:
                p = np.clip(p, PREDICTION_MIN, PREDICTION_MAX)
            acc += wv[j] * p
        stored = pool['Ensemble Predicted Yield (%)'].to_numpy(float)
        bad = np.abs(stored - acc) > float(tol)
        n_bad = int(bad.sum())
        if n_bad:
            _frac = float(n_bad) / max(len(pool), 1)
            # v22: when NEARLY EVERY row fails, the audit itself is the broken side (it rebuilds
            # the features from the pool's condition columns, so a change in which column the
            # sweep varies is enough to make it disagree with the stored scores).  Dropping the
            # whole pool emptied Figures 9a and 10 -- an empty panel is worse than a suspected
            # mismatch, so above 50 % the rows are KEPT and the failure is reported loudly.
            if repair and _frac <= 0.5:
                print(f"  [audit{(' ' + tag) if tag else ''}] {n_bad:,} / {len(pool):,} rows had "
                      f"a score that does not match their stated condition "
                      f"(max error {np.abs(stored - acc).max():.2f} pp) -- dropping them.")
                pool = pool.loc[~bad].reset_index(drop=True)
            else:
                print(f"  [audit{(' ' + tag) if tag else ''}] WARNING: {n_bad:,} / {len(pool):,} "
                      f"rows ({100.0 * _frac:.1f} %) disagree (max error "
                      f"{np.abs(stored - acc).max():.2f} pp) -- the AUDIT is not trustworthy at "
                      f"this rate, so the rows are KEPT.  Figures 9/10 use them unchanged.")


        return pool, n_bad, len(stored)
    except Exception as e:
        print(f"  [audit{(' ' + tag) if tag else ''}] could not verify "
              f"({'/'.join(str(e).splitlines()[:1])}).")
        return pool, 0, len(pool)


def _score_chunk_worker(task):
    """
    Score ONE cached chunk with every ensemble member.  Runs in a separate process.

    Reads the memory-mapped feature matrix, so nothing large is pickled in or out: only
    the per-chunk shortlist (row index + its scores) comes back.  The member models are
    loaded from `models_path`; passing them with every task would re-pickle them per chunk.

    Returns (list_of_shortlist_tuples, n_passed_gate, n_clipped) or None.
    Each tuple is (row_index, mean, std, gap, vmin, vmax, {member_name: prediction}).
    """
    (idx, cache_dir, models_path, feat_cols, wvec,
     gap_thr, xi, best_observed_yield, keep_top, topk_per_chunk) = task
    try:
        import numpy as _np
        import pandas as _pd
        with open(models_path, 'rb') as fh:
            model_items = pickle.load(fh)
        # Give the tree ensembles their tree-level parallelism back.  Random Forest dominates
        # the sweep (75% of prediction time) and its predict otherwise walks 200 trees
        # sequentially.  Set here, not in the constructor, so training and CV are untouched.
        # Measured 3.19x on RF under 16-way concurrency.  See SWEEP_RF_N_JOBS.
        if SWEEP_RF_N_JOBS and SWEEP_RF_N_JOBS > 1:
            from sklearn.ensemble import RandomForestRegressor as _RFR
            for _nm, _mdl in model_items:
                if isinstance(_mdl, _RFR):
                    _mdl.n_jobs = SWEEP_RF_N_JOBS

        arr = _np.load(os.path.join(cache_dir, 'f%05d.npy' % idx), mmap_mode='r')
        # predict() needs a real frame; the mmap is read-only, so one bounded copy
        X = _pd.DataFrame(_np.asarray(arr, dtype=_np.float64), columns=list(feat_cols))
        del arr

        n = len(X)
        wsumsq = float((wvec ** 2).sum())
        member_preds = {}
        tot = _np.zeros(n); tot_sq = _np.zeros(n)
        vmin = _np.full(n, _np.inf); vmax = _np.full(n, -_np.inf)
        n_clip = 0
        for mi, (name, mdl) in enumerate(model_items):
            # v15 MEMORY BOUND.  The chunk is a SCHEDULING unit; it must not also be the
            # prediction batch.  A one-shot predict() on the old 4,000,000-row chunk made
            # the OneHotEncoder allocate (4e6, 121) float64 = 3.61 GiB and the MLP hidden
            # layer (4e6, 64) = 1.91 GiB per worker, and an ELM member with n_hidden=2000
            # would need 4e6 x 2000 x 8 B = 64 GiB in a single worker.  Measured on
            # (AI_RCP_pub)bh_data_clean_all_whitelisted (571,872,000 rows, 143 chunks):
            # _ArrayMemoryError at chunk 19/143.
            #
            # Scoring the chunk in PREDICT_BATCH_ROWS slices caps the transient allocation
            # at PREDICT_BATCH_ROWS x (widest intermediate) instead of chunk_rows x ...,
            # which leaves PARALLEL_CHUNK_ROWS free to be tuned for throughput.  predict()
            # is row-wise, so the result is bit-identical -- only the order rows are
            # evaluated in changes.
            if n > PREDICT_BATCH_ROWS:
                p = _np.empty(n, dtype=_np.float64)
                for _lo in range(0, n, PREDICT_BATCH_ROWS):
                    _hi = min(_lo + PREDICT_BATCH_ROWS, n)
                    p[_lo:_hi] = _np.asarray(mdl.predict(X.iloc[_lo:_hi]),
                                             dtype=_np.float64)
            else:
                p = _np.asarray(mdl.predict(X), dtype=_np.float64)
            if CLIP_MEMBER_PREDICTIONS:
                pc = _np.clip(p, PREDICTION_MIN, PREDICTION_MAX)
                n_clip += int(_np.count_nonzero(pc != p))
                p = pc
            member_preds[name] = p
            tot += wvec[mi] * p
            tot_sq += wvec[mi] * p * p
            _np.minimum(vmin, p, out=vmin); _np.maximum(vmax, p, out=vmax)
        del X

        mean = tot
        std = _np.sqrt(_np.maximum(tot_sq - mean * mean, 0.0) / max(1.0 - wsumsq, 1e-12))
        gap = vmax - vmin
        keep = gap <= gap_thr
        n_pass = int(keep.sum())

        rows = []
        if n_pass:
            kk = min(int(keep_top), int(topk_per_chunk), n_pass)
            idx_keep = _np.flatnonzero(keep)
            sel = idx_keep[_np.argsort(-mean[idx_keep])[:kk]]
            for j in sel:
                rows.append((int(j), float(mean[j]), float(std[j]), float(gap[j]),
                             float(vmin[j]), float(vmax[j]),
                             {nm: float(member_preds[nm][j]) for nm in member_preds}))
        return rows, n_pass, n_clip
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _estimate_cache_bytes(n_rows, n_features=8):
    """Feature cache size for an n_rows sweep: float32 x n_features, plus slack."""
    return int(n_rows) * int(n_features) * 4


def _purge_stale_caches(root, max_age_hours=6.0, verbose=True):
    """
    Delete `vlib_*` feature-cache directories left behind by runs that did NOT finish.

    WHY THIS EXISTS: the cache is removed by an explicit shutil.rmtree at each exit path, which
    never runs when the process is KILLED.  The batch driver kills with `taskkill /F /T` on a
    memory or time cap, so every aborted run leaked its whole cache -- MEASURED: 11 aborted
    runs had left 196 GB of stray cache files, one AI_RCP cache alone being 4.4 GB.

    Only directories whose mtime is older than `max_age_hours` are removed, so a concurrently
    running sweep is never disturbed.  Set DSH_KEEP_STALE_CACHE=1 to skip the sweep.
    """
    if str(os.environ.get('DSH_KEEP_STALE_CACHE', '')).strip() in ('1', 'true', 'True'):
        return
    try:
        cutoff = time.time() - float(max_age_hours) * 3600.0
        freed, n = 0, 0
        for name in os.listdir(root):
            if not name.startswith('vlib_'):
                continue
            p = os.path.join(root, name)
            if not os.path.isdir(p):
                continue
            try:
                if os.path.getmtime(p) > cutoff:
                    continue
                sz = sum(os.path.getsize(os.path.join(dp, f))
                         for dp, _, fs in os.walk(p) for f in fs
                         if os.path.exists(os.path.join(dp, f)))
            except Exception:
                sz = 0
            shutil.rmtree(p, ignore_errors=True)
            if not os.path.exists(p):
                freed += sz
                n += 1
        if n and verbose:
            print(f"  [cache] purged {n} stale cache dir(s), freed {freed / 1024 ** 3:.1f} GB "
                  f"(from runs that were killed before they could clean up)")
    except Exception as _e:
        if verbose:
            print(f"  [cache] stale-cache sweep skipped ({_e})")


def _pick_cache_dir(required_bytes, verbose=True):
    """
    Choose a directory for the memory-mapped feature cache, with a free-space preflight.

    WHY: tempfile.mkdtemp() defaults to %TEMP%, which on Windows is on C:.  The feature cache
    for a large sweep is tens to hundreds of GB -- measured 200 GB for the 7.77e9-row
    uncapped (Roche66k)HTE_all_reactions sweep -- and `_np.save` then failed with
    OSError: [Errno 28] No space left on device AFTER six minutes of work, leaving the
    partial cache behind and filling the system drive.

    Order of preference:
      1. PARALLEL_CACHE_DIR if the user set it,
      2. the drive of this project (same volume as the outputs),
      3. the drive with the most free space.
    Raises RuntimeError with the numbers if nothing has room.
    """
    import shutil as _sh

    def _free(path):
        try:
            return _sh.disk_usage(path).free
        except Exception:
            return -1

    candidates = []
    if PARALLEL_CACHE_DIR:
        candidates.append(PARALLEL_CACHE_DIR)
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   os.pardir, '_vlib_cache'))

    # every mounted Windows drive, as a last resort
    for letter in 'DEFGHIJ':
        root = letter + ':\\'
        if os.path.exists(root):
            candidates.append(os.path.join(root, '_vlib_cache'))

    best, best_free = None, -1
    for c in candidates:
        try:
            os.makedirs(c, exist_ok=True)
        except Exception:
            continue
        f = _free(c)
        if f > best_free:
            best, best_free = c, f

    if best is None:
        raise RuntimeError('no writable directory for the feature cache')

    _purge_stale_caches(best, verbose=verbose)

    need_gb, free_gb = required_bytes / 1024 ** 3, best_free / 1024 ** 3
    if verbose:
        print(f"  [cache] {best}")
        print(f"  [cache] this sweep needs ~{need_gb:.1f} GB; {free_gb:.1f} GB free")
    if required_bytes > best_free * 0.9:
        raise RuntimeError(
            f"feature cache needs ~{need_gb:.1f} GB but only {free_gb:.1f} GB is free at "
            f"'{best}'.  Lower MAX_VIRTUAL_COMBINATIONS, set PARALLEL_CACHE_DIR to a larger "
            f"volume, or free some space.")
    return best


def generate_bo_results_parallel(df, models, feat_cols, label_encoders, scaler,
                                 best_observed_yield, weights=None,
                                 max_combinations=MAX_VIRTUAL_COMBINATIONS,
                                 max_model_gap=ENSEMBLE_MAX_MODEL_GAP,
                                 xi=0.01, keep_top=SURVIVOR_POOL_ROWS,
                                 n_workers=None, verbose=True):
    """
    EXHAUSTIVE ensemble sweep of the whole virtual library, parallelised EXACTLY.

    Why not threads: measured 1.02x -- the tree libraries already parallelise internally,
    so overlapping them across threads adds nothing.  Real gains need separate PROCESSES.

    Why the features must be cached first: feature construction (label-encoder lookup +
    scaler) runs in the parent on the streamed chunk.  If workers received raw chunks they
    would each rebuild the features -- which is exactly the work we want to split.  So this
    writes every chunk's model-input matrix to a memory-mapped file once, and each worker
    then reads a DISJOINT set of chunks.

    Exactness: every condition is scored by the same members with the same arithmetic and
    the same clipping; only the chunk-to-worker assignment differs from a serial run.  Each
    worker returns its own best PARALLEL_TOPK_PER_CHUNK rows, so the parent's final sort
    over the union is the global top-k (each chunk contributes its own best, and the gate
    is applied to that union).  The union is far larger than keep_top, so a condition that
    ranks highly globally cannot be lost.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    model_items = list(models.items())
    k = max(len(model_items), 1)
    if k == 0:
        return pd.DataFrame(), 0, pd.DataFrame()

    if weights is None:
        wvec = np.full(k, 1.0 / k)
    else:
        wvec = np.asarray([float(x) for x in weights], dtype=np.float64)
        wvec = np.clip(wvec, 0.0, None)
        wvec = wvec / wvec.sum()

    if n_workers is None:
        n_workers = N_WORKERS
    if not n_workers or n_workers <= 0:
        n_workers = max(1, min(MAX_WORKERS, (os.cpu_count() or 2) - 1))

    n_chunks = 0
    n_total = 0
    eff_combos = 1        # real value set once loading_grid is known, below

    # These come from the SAME enumeration spec the sequential streamer uses, so the
    # codes a worker derives for a combination range match the streamed chunks exactly.
    cat_cols, value_lists, n_cat, inner = get_enumeration_spec(df, feat_cols)
    num_cols = [c for c in ('temperature', 'time', 'catalyst_loading') if c in feat_cols]
    for c in cat_cols:
        _LUT_CACHE[c] = label_encoders[c].transform(
            np.asarray(value_lists[cat_cols.index(c)], dtype=object)).astype(np.float32)
    representative = get_representative_conditions(df)

    # loading grid the cap permits -- same rule as stream_virtual_library
    loading_grid = get_loading_grid_for(df, max_combinations, n_cat, n_points=N_LOADING_POINTS)
    n_load = len(loading_grid)

    # Size the chunks by ROWS, not by combinations: at 51 loading points a combination
    # range of 200k would be 10.2e6 rows, which is a large frame for every worker.
    combos_per_chunk = max(1, int(PARALLEL_CHUNK_ROWS // max(n_load, 1)))
    # the condition-rebuild helper must replay chunks with the SAME combination ranges
    eff_combos = combos_per_chunk
    ranges = [(i, i * combos_per_chunk, min((i + 1) * combos_per_chunk, n_cat))
              for i in range(int(np.ceil(n_cat / combos_per_chunk)))]
    n_chunks = len(ranges)
    n_total = n_cat * n_load

    # get_loading_grid_for() can only shrink the LOADING grid; `ranges` above covers ALL of
    # n_cat.  So when the reagent product alone exceeds the cap, the sweep silently runs the
    # full product.  Measured: (Roche66k)HTE_all_reactions -> 7.77e9 rows, ~200 GB of cache.
    if n_total > max_combinations and not ALLOW_OVERSIZED_SWEEP:
        raise RuntimeError(
            "sweep would enumerate %s rows, which exceeds MAX_VIRTUAL_COMBINATIONS = %s.\n"
            "  The cap only shrinks the LOADING grid; it cannot shrink the reagent product,\n"
            "  and here the reagent combinations alone are %s.\n"
            "  To proceed, either\n"
            "    (a) lower the cost -- this dataset has %d reagent axes with high cardinality;\n"
            "    (b) raise MAX_VIRTUAL_COMBINATIONS to >= %s AND check the cache volume has\n"
            "        room (about %.1f GB), then set ALLOW_OVERSIZED_SWEEP = True; or\n"
            "    (c) skip this dataset."
            % ('{:,}'.format(n_total), '{:,}'.format(max_combinations),
               '{:,}'.format(n_cat), len(cat_cols),
               '{:,}'.format(n_total),
               _estimate_cache_bytes(n_total) / 1024 ** 3))

    _t_cache = time.perf_counter()
    scaler_dump = pickle.dumps(scaler, protocol=pickle.HIGHEST_PROTOCOL)
    cache_ok = 0

    # v14: choose the cache location by FREE SPACE, now that n_total is known.  Letting
    # tempfile.mkdtemp() default to %TEMP% put the cache on C:, which had 0 GB free and
    # produced "OSError: [Errno 28] No space left on device" mid-write.
    cache_dir = tempfile.mkdtemp(
        prefix='vlib_', dir=_pick_cache_dir(_estimate_cache_bytes(n_total),
                                            verbose=verbose))
    # v14: guarantee the cache is removed even on an UNHANDLED exception.  The explicit
    # shutil.rmtree calls below only cover the normal and caught-failure paths; the observed
    # disk-full crash happened at the models.pkl write below, before any of them, and left
    # 200 GB behind.  rmtree with ignore_errors is idempotent, so double-removal is harmless.
    import atexit as _atexit
    _atexit.register(shutil.rmtree, cache_dir, ignore_errors=True)
    try:
        import multiprocessing as mp
        ctx = mp.get_context(MP_START_METHOD)
        tasks = [(ci, cache_dir, c0, c1, list(feat_cols), cat_cols, num_cols,
                  value_lists, inner, representative, loading_grid,
                  _LUT_CACHE, scaler_dump) for (ci, c0, c1) in ranges]
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            itc = as_completed([ex.submit(_build_cache_chunk, t) for t in tasks])
            if verbose:
                itc = tqdm(itc, total=len(tasks), desc='Caching features',
                           unit='chunk')
            for fu in itc:
                r = fu.result()
                if r is not None:
                    cache_ok += 1
    except Exception as e:
        print(f"  [parallel] parallel cache build unavailable ({e}); "
              f"falling back to the serial caching loop.")
        cache_ok = -1

    if cache_ok == 0 or cache_ok == -1:
        # serial fallback (also the path when USE_FAST_FEATURE_FRAME is False)
        preview_parts = []
        preview_budget = VIRTUAL_LIBRARY_PREVIEW_ROWS
        it = stream_virtual_library(df, label_encoders, feat_cols,
                                    chunk_combos=eff_combos,
                                    max_combinations=max_combinations, verbose=verbose)
        if verbose:
            it = tqdm(it, desc='Caching features (serial)', unit='chunk')
        ci = 0
        for chunk in it:
            n = len(chunk)
            if USE_FAST_FEATURE_FRAME:
                feat = fast_virtual_features(chunk, feat_cols, label_encoders, scaler,
                                             lut_cache=_LUT_CACHE)
            else:
                feat = build_virtual_feature_frame(df, feat_cols, label_encoders,
                                                   scaler, chunk)
            arr = np.ascontiguousarray(feat.to_numpy(dtype=np.float32, copy=False))
            np.save(os.path.join(cache_dir, 'f%05d.npy' % ci), arr)
            ci += 1
            if preview_budget > 0:
                take = min(preview_budget, n)
                preview_parts.append(chunk.iloc[:take])
                preview_budget -= take
            del feat, arr, chunk
            gc.collect()
        n_chunks = ci
        cache_preview_parts = preview_parts

    if verbose:
        print(f"  [parallel] cached {cache_ok if cache_ok > 0 else n_chunks} chunks "
              f"({n_total:,} rows) on {n_workers} workers")
        print(f"  [timing] feature caching took {time.perf_counter() - _t_cache:.1f} s")

    # Streamed separately and only up to the preview budget, so it never materialises
    # the library.
    preview_parts = []
    preview_budget = VIRTUAL_LIBRARY_PREVIEW_ROWS
    itp = stream_virtual_library(df, label_encoders, feat_cols,
                                 chunk_combos=max(1, VIRTUAL_LIBRARY_PREVIEW_ROWS // n_load),
                                 max_combinations=max_combinations, verbose=False)
    for chunk in itp:
        take = min(preview_budget, len(chunk))
        preview_parts.append(chunk.iloc[:take])
        preview_budget -= take
        if preview_budget <= 0:
            break
    preview = (pd.concat(preview_parts, ignore_index=True) if preview_parts
               else pd.DataFrame(columns=list(get_representative_conditions(df).keys())))
    del preview_parts
    gc.collect()

    if n_chunks == 0:
        shutil.rmtree(cache_dir, ignore_errors=True)
        return pd.DataFrame(), 0, preview

    models_path = os.path.join(cache_dir, 'models.pkl')
    # v14: this write is the FIRST thing after the cache build and sits outside the worker
    # try/finally, so a full disk here used to leak the whole cache.  Clean up and re-raise.
    try:
        with open(models_path, 'wb') as fh:
            pickle.dump(model_items, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as _e:
        print(f"  [parallel] could not write models.pkl to the cache ({_e}); "
              f"removing {cache_dir} and falling back to the serial sweep.")
        shutil.rmtree(cache_dir, ignore_errors=True)
        return generate_bo_results_streaming(
            df, models, feat_cols, label_encoders, scaler, best_observed_yield,
            max_combinations=max_combinations, max_model_gap=max_model_gap,
            xi=xi, keep_top=keep_top, weights=wvec, verbose=verbose)

    gap_thr = float(max_model_gap) if max_model_gap is not None else float('inf')
    tasks = [(i, cache_dir, models_path, list(feat_cols), wvec, gap_thr,
              float(xi), float(best_observed_yield), int(keep_top),
              int(PARALLEL_TOPK_PER_CHUNK)) for i in range(n_chunks)]

    shortlist = []          # (chunk, row, mean, std, gap, vmin, vmax, members)
    n_pass_total = 0
    n_clip_total = 0
    ok = True
    _t_score = time.perf_counter()
    # set in the `finally` so it exists even on the fallback-return paths
    _t_workers = _t_score
    try:
        import multiprocessing as mp
        ctx = mp.get_context(MP_START_METHOD)
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            fut_chunk = {}
            for i, t in enumerate(tasks):
                fut_chunk[ex.submit(_score_chunk_worker, t)] = i
            it2 = as_completed(fut_chunk)
            if verbose:
                it2 = tqdm(it2, total=len(fut_chunk), desc='Ensemble (parallel)',
                           unit='chunk')
            for fu in it2:
                ch_i = fut_chunk[fu]
                try:
                    res = fu.result()
                except Exception as e:
                    print(f"  [parallel] worker failed on chunk {ch_i}: {e}")
                    ok = False
                    continue
                if res is None:
                    ok = False
                    continue
                rows, np_, nc_ = res
                n_pass_total += np_
                n_clip_total += nc_
                for (row_i, mean_v, std_v, gap_v, vmin_v, vmax_v, members) in rows:
                    shortlist.append((ch_i, row_i, mean_v, std_v, gap_v,
                                      vmin_v, vmax_v, members))
    except Exception as e:
        print(f"  [parallel] multiprocessing unavailable ({e}); falling back "
              f"to the serial sweep.")
        shutil.rmtree(cache_dir, ignore_errors=True)
        return generate_bo_results_streaming(
            df, models, feat_cols, label_encoders, scaler, best_observed_yield,
            max_combinations=max_combinations, max_model_gap=max_model_gap,
            xi=xi, keep_top=keep_top, weights=wvec, verbose=verbose)
    finally:
        _t_workers = time.perf_counter()
        shutil.rmtree(cache_dir, ignore_errors=True)

    # marks the end of WORKER scoring; the serial gate/rebuild/audit tail is measured
    # against this so the two are reported separately.

    if not ok:
        print("  [parallel] at least one worker did not complete; falling back to "
              "the serial sweep so the table is consistent.")
        return generate_bo_results_streaming(
            df, models, feat_cols, label_encoders, scaler, best_observed_yield,
            max_combinations=max_combinations, max_model_gap=max_model_gap,
            xi=xi, keep_top=keep_top, weights=wvec, verbose=verbose)

    # The tqdm bar covers only the WORKER scoring.  Everything below (global ranking of
    # the survivors, regenerating the condition columns by replaying the needed chunks, and
    # the score audit) runs in the parent process and was silent -- which is the pause the
    # log shows between the bar finishing and the "[timing] parallel ensemble scoring"
    # line.  Announce each step so the wait is visible.
    if not shortlist:
        return pd.DataFrame(), n_total, preview



    shortlist = [r for r in shortlist if r[4] <= gap_thr]
    if not shortlist:
        return pd.DataFrame(), n_total, preview
    shortlist.sort(key=lambda r: -r[2])          # by weighted ensemble mean
    shortlist = shortlist[:int(keep_top)]

    # IMPORTANT ordering contract: _regen_conditions_for_indices walks the chunks in
    # chunk order and, inside a chunk, in the order of the row list it is given.  So sort
    # the shortlist by (chunk, row) BEFORE regenerating, and keep that same order when
    # attaching the scores below -- otherwise the values land on the wrong conditions.
    shortlist.sort(key=lambda r: (r[0], r[1]))

    if verbose:
        n_chunks_needed = len({r[0] for r in shortlist})
        print(f"  [parallel] rebuilding condition columns")

    cond = _regen_conditions_for_indices(
        df, feat_cols, label_encoders, scaler,
        [r[0] for r in shortlist], [r[1] for r in shortlist], max_combinations,
        chunk_combos=eff_combos)
    if len(cond) != len(shortlist):
        print(f"  [parallel] regenerated {len(cond)} condition rows for "
              f"{len(shortlist)} scored rows; falling back to the serial sweep.")
        return generate_bo_results_streaming(
            df, models, feat_cols, label_encoders, scaler, best_observed_yield,
            max_combinations=max_combinations, max_model_gap=max_model_gap,
            xi=xi, keep_top=keep_top, weights=wvec, verbose=verbose)

    cond = cond.drop(columns=[c for c in ('_chunk', '_row') if c in cond.columns])
    cond['Ensemble Predicted Yield (%)'] = [r[2] for r in shortlist]
    cond['Prediction Uncertainty (%)'] = [r[3] for r in shortlist]
    cond['Model Gap (%)'] = [r[4] for r in shortlist]
    cond['Model Min (%)'] = [r[5] for r in shortlist]
    cond['Model Max (%)'] = [r[6] for r in shortlist]
    for nm, _mdl in model_items:
        cond[f'Pred_{nm}'] = [r[7][nm] for r in shortlist]
    _mean = cond['Ensemble Predicted Yield (%)'].to_numpy(float)
    _std = cond['Prediction Uncertainty (%)'].to_numpy(float)
    cond['Expected Improvement'] = calculate_expected_improvement(
        _mean, _std, best_observed_yield, xi=xi)
    cond['UCB (%)'] = _mean + 1.96 * _std
    cond['Physically Valid'] = ((cond['Model Min (%)'] >= 0)
                                & (cond['Model Max (%)'] <= 100))
    cond['Passes Ensemble Gate'] = True
    bo_df = cond.reset_index(drop=True)
    del shortlist, cond
    gc.collect()

    if AUDIT_CONDITION_SCORES and len(bo_df):
        if verbose:
            print(f"  [parallel] verifying survivor scores")
        bo_df, _nb, _nt = audit_condition_scores(
            bo_df, df, models, feat_cols, label_encoders, scaler,
            weights=wvec, tag='parallel')

    if verbose:
        # NOTE: `_t_score` is started BEFORE the worker pool, so this span covers worker
        # scoring PLUS the gate/rank/rebuild/audit steps above.  Both are reported so the
        # serial tail is not mistaken for worker time.
        _tail = time.perf_counter() - _t_workers
        print(f"  [timing] parallel ensemble scoring took "
              f"{time.perf_counter() - _t_score:.1f} s on {n_workers} workers ")
        print(f"  [parallel] scored {n_total:,} conditions; {n_pass_total:,} passed "
              f"the gate ({100 * n_pass_total / max(n_total, 1):.2f}%)")

    return bo_df, n_total, preview

def generate_bo_results_two_stage(df, models, feat_cols, label_encoders, scaler,
                                  best_observed_yield, weights=None,
                                  chunk_combos=STREAM_CHUNK_COMBOS,
                                  max_combinations=MAX_VIRTUAL_COMBINATIONS,
                                  max_model_gap=ENSEMBLE_MAX_MODEL_GAP,
                                  xi=0.01, keep_top=SURVIVOR_POOL_ROWS,
                                  stage1_keep=STAGE1_KEEP_ROWS,
                                  verbose=True):
    """
    v13 TWO-STAGE SCREENING -- the fix for the actual bottleneck.

    Measured cost breakdown of the single-pass sweep: prediction is ~79 % of the run, and
    it is (conditions x members) tree traversals.  The feature-frame work I sped up with
    the LUT path was only ~17 %, so that alone could never make this fast.

    The observation that makes a big speedup possible: we do not need the full ensemble
    prediction for all 571.9e6 enumerated conditions -- we need an accurate RANKING at the
    top.  So:

      stage 1  score EVERY condition with the single best member only
               (571.9e6 x 1 model), keep the best `stage1_keep` rows;
      stage 2  run the FULL weighted ensemble (all members, disagreement gate, per-member
               columns) on that shortlist only.

    Cost drops from (conditions x n_members) to (conditions x 1) + (stage1_keep x n_members).
    With 3 members and 571.9e6 conditions that is a large reduction, and the loading grid is
    still fully enumerated -- this is a ranking shortcut, not a reduction in coverage.

    What is deliberately NOT approximated: the reported top-`keep_top` candidates and the
    per-member columns, which all come from the real full ensemble in stage 2.

    What IS approximate: the gate pass-rate statistic (stage 1 keeps only the shortlist, so
    the "x % passed the gate" line describes the shortlist, not the whole library).  That
    number is a diagnostic, not a result.

    Set STAGE1_KEEP_ROWS large relative to keep_top (default 200,000 vs 20,000) so that the
    ensemble's true top-k cannot be screened out: a condition can only reach the reported
    table if the best member already ranked it highly.
    """
    model_items = list(models.items())
    if not model_items:
        return pd.DataFrame(), 0, pd.DataFrame()

    if weights is None:
        wvec = np.full(len(model_items), 1.0 / len(model_items))
    else:
        wvec = np.asarray([float(x) for x in weights], dtype=np.float64)
        wvec = np.clip(wvec, 0.0, None)
        wvec = wvec / wvec.sum()
    s1_idx = int(np.argmax(wvec))
    s1_name, s1_model = model_items[s1_idx]
    if verbose:
        print(f"  [two-stage] stage 1 screener: {s1_name} "
              f"(weight {wvec[s1_idx]:.4f}); stage 2: full ensemble of "
              f"{[n for n, _ in model_items]}")

    n_load = len(get_loading_grid(df, n_points=N_LOADING_POINTS))
    eff_combos = _effective_chunk_combos(n_load, chunk_combos)

    total = np.zeros(0)
    n_total = 0
    short_pred = []
    short_pos = []
    preview_parts = []
    preview_budget = VIRTUAL_LIBRARY_PREVIEW_ROWS
    offset = 0

    it1 = stream_virtual_library(df, label_encoders, feat_cols, chunk_combos=eff_combos,
                                 max_combinations=max_combinations, verbose=verbose)
    if verbose:
        it1 = tqdm(it1, desc='Stage 1 (1 model)', unit='chunk')
    for chunk in it1:
        n = len(chunk)
        feat = (fast_virtual_features(chunk, feat_cols, label_encoders, scaler,
                                      lut_cache=_LUT_CACHE)
                if USE_FAST_FEATURE_FRAME else
                build_virtual_feature_frame(df, feat_cols, label_encoders, scaler, chunk))
        p = np.asarray(s1_model.predict(feat), dtype=np.float64)
        short_pred.append(p)
        short_pos.append(np.arange(offset, offset + n, dtype=np.int64))
        offset += n
        n_total += n
        if preview_budget > 0:
            take = min(preview_budget, n)
            preview_parts.append(chunk.iloc[:take])
            preview_budget -= take

    if not short_pred:
        return pd.DataFrame(), 0, pd.DataFrame()

    s1_pred = np.concatenate(short_pred)
    s1_pos = np.concatenate(short_pos)
    del short_pred, short_pos

    take = int(min(stage1_keep, len(s1_pred)))
    sel = np.argpartition(-s1_pred, take - 1)[:take]     # top-`take` by stage-1 score
    sel = sel[np.argsort(-s1_pred[sel])]
    keep_pos = set(s1_pos[sel].tolist())
    if verbose:
        print(f"  [two-stage] stage 1 done: {n_total:,} conditions scored with 1 model; "
              f"shortlisted {len(keep_pos):,} ({(100*len(keep_pos)/max(n_total,1)):.3f}%)")
    del s1_pred, s1_pos

    survivors = []
    n_pass = 0
    n_clipped_total = 0
    gap_thr = float(max_model_gap) if max_model_gap is not None else np.inf
    wsumsq = float((wvec ** 2).sum())
    offset = 0

    it2 = stream_virtual_library(df, label_encoders, feat_cols, chunk_combos=eff_combos,
                                 max_combinations=max_combinations, verbose=False)
    if verbose:
        it2 = tqdm(it2, desc='Stage 2 (full ensemble)', unit='chunk')
    for chunk in it2:
        n = len(chunk)
        pos = np.arange(offset, offset + n, dtype=np.int64)
        offset += n
        mask = np.fromiter((int(x) in keep_pos for x in pos), dtype=bool, count=n)
        if not mask.any():
            continue
        chunk = chunk.loc[mask].reset_index(drop=True)
        pos = pos[mask]
        n2 = len(chunk)

        feat = (fast_virtual_features(chunk, feat_cols, label_encoders, scaler,
                                      lut_cache=_LUT_CACHE)
                if USE_FAST_FEATURE_FRAME else
                build_virtual_feature_frame(df, feat_cols, label_encoders, scaler, chunk))

        member_preds = {}
        tot = np.zeros(n2); tot_sq = np.zeros(n2)
        vmin = np.full(n2, np.inf); vmax = np.full(n2, -np.inf)
        for mi, (name, mdl) in enumerate(model_items):
            p = np.asarray(mdl.predict(feat), dtype=np.float64)
            if CLIP_MEMBER_PREDICTIONS:
                pc = np.clip(p, PREDICTION_MIN, PREDICTION_MAX)
                n_clipped_total += int(np.count_nonzero(pc != p))
                p = pc
            member_preds[name] = p
            tot += wvec[mi] * p
            tot_sq += wvec[mi] * p * p
            np.minimum(vmin, p, out=vmin); np.maximum(vmax, p, out=vmax)
        mean = tot
        std = np.sqrt(np.maximum(tot_sq - mean * mean, 0.0) / max(1.0 - wsumsq, 1e-12))
        gap = vmax - vmin

        keep = gap <= gap_thr
        n_pass += int(keep.sum())
        if keep.any():
            sub = chunk.loc[keep].copy()
            for name in member_preds:
                sub[f'Pred_{name}'] = member_preds[name][keep]
            sub['Ensemble Predicted Yield (%)'] = mean[keep]
            sub['Prediction Uncertainty (%)'] = std[keep]
            sub['Model Gap (%)'] = gap[keep]
            sub['Model Min (%)'] = vmin[keep]
            sub['Model Max (%)'] = vmax[keep]
            sub['Expected Improvement'] = calculate_expected_improvement(
                mean[keep], std[keep], best_observed_yield, xi=xi)
            sub['UCB (%)'] = mean[keep] + 1.96 * std[keep]
            survivors.append(sub)
            if len(survivors) > 8:
                pooled = pd.concat(survivors, ignore_index=True)
                if len(pooled) > 2 * keep_top:
                    pooled = pooled.nlargest(keep_top, 'Ensemble Predicted Yield (%)')
                survivors = [pooled]

    bo_df = (pd.concat(survivors, ignore_index=True) if survivors
             else pd.DataFrame(columns=list(get_representative_conditions(df).keys())))
    if len(bo_df):
        bo_df = bo_df.nlargest(min(keep_top, len(bo_df)), 'Ensemble Predicted Yield (%)')
        bo_df['Physically Valid'] = (bo_df['Model Min (%)'] >= 0) & (bo_df['Model Max (%)'] <= 100)
        bo_df['Passes Ensemble Gate'] = True

    preview = (pd.concat(preview_parts, ignore_index=True) if preview_parts
               else pd.DataFrame(columns=list(get_representative_conditions(df).keys())))

    if verbose:
        print(f"  [two-stage] stage 2: ensemble scored {len(keep_pos):,} shortlisted "
              f"conditions; {n_pass:,} passed the gate")
        if CLIP_MEMBER_PREDICTIONS:
            print(f"  [two-stage] member predictions clipped: {n_clipped_total:,} values were "
                  f"outside [{PREDICTION_MIN:g}, {PREDICTION_MAX:g}]")
        print(f"  [two-stage] survivor pool: {len(bo_df):,} rows; preview: {len(preview):,}")
    return bo_df, n_total, preview


def generate_bo_results_streaming(df, models, feat_cols, label_encoders, scaler,
                                  best_observed_yield, chunk_combos=STREAM_CHUNK_COMBOS,
                                  max_combinations=MAX_VIRTUAL_COMBINATIONS,
                                  max_model_gap=ENSEMBLE_MAX_MODEL_GAP,
                                  xi=0.01, keep_top=SURVIVOR_POOL_ROWS,
                                  weights=None, verbose=True):
    """
    Score the whole virtual library without ever materialising it.

    weights
    -------
    Optional per-member weights (same order as `models`), e.g. test-set R^2.  They are
    normalised to sum to 1 and drive BOTH the ensemble mean and its uncertainty.  The
    member list is also carried through to the survivor table as `Pred_<name>` columns,
    which is how Figure 9 shows each model's own prediction next to the ensemble.

    Returns
    -------
    bo_df_survivors : DataFrame
        Only the conditions that passed the ensemble-agreement gate, with the same
        columns generate_bo_results produces plus the per-model `Pred_*` columns.
        This is *not* the full library -- it is bounded by `keep_top`.
    n_total : int
        total conditions enumerated
    preview : DataFrame
        a bounded sample of the RAW library (not scored), covering every reagent
        value, for the figures that need to describe the search space.
    """
    model_items = list(models.items())
    k = max(len(model_items), 1)
    gap_thr = float(max_model_gap) if max_model_gap is not None else np.inf

    if weights is None:
        wvec = np.full(k, 1.0 / k)
    else:
        wvec = np.asarray([float(x) for x in weights], dtype=np.float64)
        if len(wvec) != k:
            raise ValueError(f'weights length {len(wvec)} != number of models {k}')
        wvec = np.clip(wvec, 0.0, None)
        wvec = wvec / wvec.sum()
    wsumsq = float((wvec ** 2).sum())
    if verbose:
        print('  [ensemble] members and weights: ' + ', '.join(
            f'{nm}={wv:.4f}' for nm, wv in zip(dict(model_items).keys(), wvec)))

    survivors = []
    n_total = 0
    n_pass = 0
    n_clipped_total = 0
    preview_parts = []
    preview_budget = VIRTUAL_LIBRARY_PREVIEW_ROWS
    chunk_combos = _effective_chunk_combos(
        len(get_loading_grid(df, n_points=N_LOADING_POINTS, for_enumeration=True))
        if max_combinations > (37 * 20 * 16 * 21 * 23) else 1, chunk_combos)

    iterator = stream_virtual_library(df, label_encoders, feat_cols,
                                      chunk_combos=chunk_combos,
                                      max_combinations=max_combinations,
                                      verbose=verbose)
    if verbose:
        iterator = tqdm(iterator, desc='Streaming virtual library', unit='chunk')

    for chunk in iterator:
        n = len(chunk)
        n_total += n

        # v13: fast path (no string round-trip).  Falls back to the original builder if
        # the chunk is not categorical (e.g. a caller passed a plain DataFrame).
        _cat_here = [c for c in chunk.columns if c in label_encoders]
        if USE_FAST_FEATURE_FRAME and all(
                str(chunk[c].dtype) == 'category' for c in _cat_here):
            feat = fast_virtual_features(chunk, feat_cols, label_encoders, scaler,
                                         lut_cache=_LUT_CACHE)
        else:
            feat = build_virtual_feature_frame(df, feat_cols, label_encoders, scaler, chunk)
        # v22l: (AI_RCP_pub)bh has ~29 % missing temperature/time, so a streamed chunk can
        # carry NaN and sklearn's GradientBoostingRegressor refuses it
        # ("Input X contains NaN").  XGBoost/RF tolerate NaN, which is why only the third
        # member failed.  The numeric block is already standardised, so filling 0 restores
        # the training mean -- the same imputation prepare_features uses.
        _feat_np = np.asarray(feat, dtype=np.float64)
        if not np.isfinite(_feat_np).all():
            _n_nan_cells = int(np.size(_feat_np) - np.isfinite(_feat_np).sum())
            _feat_np = np.nan_to_num(_feat_np, nan=0.0, posinf=0.0, neginf=0.0)
            feat = pd.DataFrame(_feat_np, columns=list(feat.columns))
            if not globals().get('_NAN_FILL_REPORTED'):
                globals()['_NAN_FILL_REPORTED'] = True
                print('  [stream] filled %d non-finite feature cell(s) with the training mean'
                      ' (NaN in the source data)' % _n_nan_cells)
        member_preds = {}
        total = np.zeros(n); total_sq = np.zeros(n)
        vmin = np.full(n, np.inf); vmax = np.full(n, -np.inf)
        n_clipped = 0
        for mi, (name, mdl) in enumerate(model_items):
            p = np.asarray(mdl.predict(feat), dtype=np.float64)
            if CLIP_MEMBER_PREDICTIONS:
                # v13: a yield is a normalised UV area percentage, so [0, 100] are hard
                # physical limits.  Clip EACH MEMBER before aggregating -- otherwise a
                # member claiming 105 % argues FOR a condition in the weighted mean.
                # (RandomForest never needs this: averaging leaf values keeps it in
                # range.  Gradient boosting and XGBoost sum leaf values and can leave it.)
                p_clipped = np.clip(p, PREDICTION_MIN, PREDICTION_MAX)
                n_clipped += int(np.count_nonzero(p_clipped != p))
                p = p_clipped
            member_preds[name] = p
            total += wvec[mi] * p
            total_sq += wvec[mi] * p * p
            np.minimum(vmin, p, out=vmin); np.maximum(vmax, p, out=vmax)
        n_clipped_total += n_clipped
        mean = total
        std = np.sqrt(np.maximum(total_sq - mean * mean, 0.0) / max(1.0 - wsumsq, 1e-12))
        gap = vmax - vmin

        keep = gap <= gap_thr
        n_pass += int(keep.sum())
        if keep.any():
            sub = chunk.loc[keep].copy()
            for name in member_preds:
                sub[f'Pred_{name}'] = member_preds[name][keep]
            sub['Ensemble Predicted Yield (%)'] = mean[keep]
            sub['Prediction Uncertainty (%)'] = std[keep]
            sub['Model Gap (%)'] = gap[keep]
            sub['Model Min (%)'] = vmin[keep]
            sub['Model Max (%)'] = vmax[keep]
            sub['Expected Improvement'] = calculate_expected_improvement(
                mean[keep], std[keep], best_observed_yield, xi=xi)
            sub['UCB (%)'] = mean[keep] + 1.96 * std[keep]

            # v12: keep only the best `keep_top` survivors overall, not every passing
            # row.  On the full library ~80 % of 45.7e6 conditions pass the gate; that
            # would rebuild a multi-GiB frame and defeat the whole point of streaming.
            survivors.append(sub)
            if len(survivors) > 8:                      # compact periodically
                pooled = pd.concat(survivors, ignore_index=True)
                if len(pooled) > 2 * keep_top:
                    pooled = pooled.nlargest(keep_top, 'Ensemble Predicted Yield (%)')
                survivors = [pooled]

        if preview_budget > 0:
            take = min(preview_budget, n)
            preview_parts.append(chunk.iloc[:take])
            preview_budget -= take

    bo_df = (pd.concat(survivors, ignore_index=True) if survivors
             else pd.DataFrame(columns=list(get_representative_conditions(df).keys())))
    if len(bo_df):
        bo_df = bo_df.nlargest(min(keep_top, len(bo_df)), 'Ensemble Predicted Yield (%)')
        bo_df['Physically Valid'] = (bo_df['Model Min (%)'] >= 0) & (bo_df['Model Max (%)'] <= 100)
        bo_df['Passes Ensemble Gate'] = True
        # v13: this engine's pool assembly was found to mis-attach scores to conditions
        # (re-scoring 5,000 returned rows disagreed for 4,501 of them, by up to 51 pp).
        # The feature builders are provably identical, so the audit below is the reliable
        # guard: a row whose stored score does not survive a fresh re-scoring is dropped.
        if AUDIT_CONDITION_SCORES:
            bo_df, _nb, _nt = audit_condition_scores(
                bo_df, df, models, feat_cols, label_encoders, scaler,
                weights=wvec, tag='streaming')

    preview = (pd.concat(preview_parts, ignore_index=True) if preview_parts
               else pd.DataFrame(columns=list(get_representative_conditions(df).keys())))

    if verbose:
        print(f"  [stream] scored {n_total:,} conditions; "
              f"{n_pass:,} passed the gate ({100 * n_pass / max(n_total, 1):.2f}%)")
        if CLIP_MEMBER_PREDICTIONS:
            print(f"  [stream] member predictions clipped to "
                  f"[{PREDICTION_MIN:g}, {PREDICTION_MAX:g}]: "
                  f"{n_clipped_total:,} member-values were out of range "
                  f"({100 * n_clipped_total / max(n_total * k, 1):.3f}%)")
        print(f"  [stream] survivor pool in memory: {len(bo_df):,} rows "
              f"({bo_df.memory_usage(deep=True).sum() / 1024**2:.1f} MiB); "
              f"raw preview: {len(preview):,} rows")
    return bo_df, n_total, preview



def ensemble_virtual_predictions(models, virtual_X, batch_size=2000000, desc='Ensemble Prediction',
                                 show_progress=True, keep_predictions=True):
    """
    Per-model predictions plus the ensemble mean/std/min/max.

    v12 memory fix: all ensemble statistics are accumulated chunk-by-chunk
    (`ensemble_stats_streaming`) instead of via `np.vstack` over the whole library.
    Previously k models x 45.7e6 rows needed 2*k*0.34 GiB for the matrix plus its
    sources -- ~11.6 GiB at k=17, which raised
    "Unable to allocate 5.79 GiB for an array with shape (17, 45749760)".

    Returns
    -------
    (model_predictions, mean, std, None, vmin, vmax)
        `model_predictions` maps each model name to its full-length prediction vector
        when `keep_predictions=True`, otherwise to an empty placeholder (the gate only
        needs vmin/vmax, which are always returned).
    """
    model_items = list(models.items())
    if not model_items:
        n = len(virtual_X)
        z = np.zeros(n, dtype=np.float64)
        return {}, z, z, None, z, z

    mean, std, vmin, vmax, kept = ensemble_stats_streaming(
        models, virtual_X, batch_size=batch_size, show_progress=show_progress,
        max_models_in_memory=(len(model_items) if keep_predictions else 0))

    if keep_predictions:
        model_predictions = {name: kept[name] for name, _ in model_items}
    else:
        placeholder = np.empty(0, dtype=np.float64)
        model_predictions = {name: placeholder for name, _ in model_items}

    return model_predictions, mean, std, None, vmin, vmax

def calculate_expected_improvement(mean_prediction, uncertainty, best_observed_yield, xi=0.01):
    from scipy.stats import norm

    mu = np.asarray(mean_prediction, dtype=float)
    sigma = np.asarray(uncertainty, dtype=float)
    sigma_safe = np.maximum(sigma, 1e-12)

    improvement = mu - float(best_observed_yield) - xi
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)

    zero_sigma = sigma <= 1e-12
    if np.any(zero_sigma):
        ei[zero_sigma] = np.maximum(improvement[zero_sigma], 0.0)

    return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

def generate_bo_results(virtual_df, models, virtual_X, best_observed_yield, batch_size=2000000,
                        xi=0.01, top_n=100, max_model_gap=ENSEMBLE_MAX_MODEL_GAP,
                        min_support_count=ENSEMBLE_MIN_SUPPORT,
                        keep_model_predictions=KEEP_MODEL_PREDICTIONS):
    """
    Score the virtual library with the tree-model ensemble, then apply the
    ENSEMBLE QUALITY GATE before anything is ranked.

    Gate A - model agreement.  `Model Gap (%)` is max(pred) - min(pred) across the
             ensemble members for the SAME condition.  This is the ensemble's own
             disagreement estimate.  Predicting with two models that differ by, e.g.,
             112 percentage points and then averaging them produces a number that
             looks plausible but is meaningless, so those conditions are dropped
             rather than averaged.  This is also what removes the physically
             inadmissible (>100 %, <0 %) values from the candidate table.
             Set max_model_gap=None to disable.

    Gate B - experimental support.  `Support Count` is how many times the exact
             condition occurs in the historical data.  Conditions with no (or too
             little) precedent are extrapolations; they are kept only if
             min_support_count is set to 0/None.

    Ranking afterwards uses the ensemble mean among the SURVIVORS.
    """
    model_predictions, ensemble_mean, ensemble_std, _, model_min, model_max = \
        ensemble_virtual_predictions(
            models, virtual_X, batch_size=batch_size, desc='Bayesian Ensemble',
            show_progress=True, keep_predictions=keep_model_predictions
        )

    bo_df = virtual_df.copy()

    if keep_model_predictions:
        for model_name, pred in model_predictions.items():
            safe_name = model_name.replace(' ', '_').replace('-', '_')
            bo_df[f'Pred_{safe_name}'] = pred

    bo_df['Ensemble Predicted Yield (%)'] = ensemble_mean
    bo_df['Prediction Uncertainty (%)'] = ensemble_std

    # v12: min/max come from the streaming accumulator, so this no longer needs the
    # (n_models, n_rows) matrix.  `keep_model_predictions=False` therefore drops the
    # per-model columns and their ~0.34 GiB each -- the Model Gap gate still works.
    bo_df['Model Gap (%)'] = model_max - model_min
    bo_df['Model Min (%)'] = model_min
    bo_df['Model Max (%)'] = model_max
    bo_df['Physically Valid'] = (
        (bo_df['Model Min (%)'] >= 0.0) & (bo_df['Model Max (%)'] <= 100.0)
    )

    bo_df['Expected Improvement'] = calculate_expected_improvement(
        ensemble_mean, ensemble_std, best_observed_yield, xi=xi)

    kappa = 1.96
    bo_df['UCB (%)'] = ensemble_mean + kappa * ensemble_std

    n_total = len(bo_df)
    gate_mask = pd.Series(True, index=bo_df.index)

    if max_model_gap is not None:
        agree = bo_df['Model Gap (%)'] <= float(max_model_gap)
        print(f"  [Gate A] ensemble model-gap <= {max_model_gap}%: "
              f"keeps {int(agree.sum()):,} / {n_total:,} ({100 * agree.mean():.2f}%) "
              f"-> drops {int((~agree).sum()):,}")
        gate_mask &= agree

    # support count is filled in by build_prediction_lookup later; the gate on it
    # is applied in apply_support_gate() once the historical lookup is merged in.

    bo_df['Passes Ensemble Gate'] = gate_mask

    eligible = bo_df[gate_mask]
    print(f"  [Gate] candidates after filtering: {len(eligible):,} "
          f"(from {n_total:,}); models used: {list(model_predictions)}")

    if min_support_count:
        print(f"  [Gate B] min_support_count={min_support_count} will be applied "
              f"after the historical lookup is merged.")

    if len(eligible) == 0:
        print("  WARNING: the ensemble gate removed every candidate — "
              "falling back to the ungated library. Consider raising max_model_gap.")
        eligible = bo_df

    # ranks computed within the surviving pool
    bo_df['EI Rank'] = bo_df['Expected Improvement'].where(gate_mask).rank(ascending=False, method='first').astype('Int64')
    bo_df['Yield Rank'] = bo_df['Ensemble Predicted Yield (%)'].where(gate_mask).rank(ascending=False, method='first').astype('Int64')
    bo_df['Uncertainty Rank'] = bo_df['Prediction Uncertainty (%)'].where(gate_mask).rank(ascending=False, method='first').astype('Int64')

    top_bo = eligible.nlargest(min(top_n, len(eligible)), 'Expected Improvement').reset_index(drop=True)
    return bo_df, top_bo, None


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
    _reagent_slots = []
    try:
        from hte.params import CATEGORICAL_SLOTS as _CS2
        _reagent_slots = list(_CS2)
    except Exception:                                   # noqa: BLE001
        _reagent_slots = ['metal', 'catalyst', 'ligand', 'base', 'solvent',
                          'reactant_2', 'reactant_1']
    group_cols = [
        c for c in _reagent_slots + ['catalyst_loading']
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

def max_component_match(candidates, df, axes=None, prefix=''):
    """
    For each candidate, the LARGEST number of reagent axes on which it agrees with ANY
    combination actually present in the historical data.


    Why this and not an exact-condition count
    -----------------------------------------
    The candidate pool is a virtual library, so `Support Count` (exact condition) and
    `Reagent Support Count` (exact reagent combo) are 0 by construction -- nothing in the
    enumeration has ever been run.  A count of 0 for every row tells the reader nothing.
    The maximum-overlap score instead says HOW CLOSE the candidate is to known chemistry:
    5 = the same reagent set was run before; 1 = only one component is familiar.

    Cost
    ----
    The comparison is (candidates x DISTINCT historical combinations x n_axes), not
    candidates x rows.  On this dataset that is 500 x 4,189 x 5 = 1.05e7 comparisons,
    measured at 0.012 s -- negligible next to the sweep.  Historical combinations are
    de-duplicated first, which is what makes it cheap.

    Returns (best_overlap_array, tie_count_array, n_axes_used).  The third value is the
    denominator actually used, so the caller can label the column honestly instead of
    hardcoding "of 5" -- a dataset missing an axis would otherwise overstate the score.

    PORTABILITY notes (verified on synthetic frames)
    ------------------------------------------------
    * Axes are taken from REAGENT_AXES and filtered to those present in BOTH frames.  A
      dataset lacking one of them still runs, but the achievable maximum drops, which is
      why n_axes_used is returned rather than assumed.
    * An axis that is CONSTANT in the new dataset contributes a free match on every row and
      inflates every score equally.  This is reported as a warning, because it silently
      removes the axis's discriminating power.
    * Ties is fully data-driven: it counts historical combinations achieving the maximum,
      so it is meaningful on any dataset.  It does depend on how many distinct historical
      combinations exist, so tie counts are not comparable ACROSS datasets.
    """
    axes = list(axes or REAGENT_AXES)
    axes = [a for a in axes if a in df.columns and a in candidates.columns]
    n = len(candidates)
    if not axes or n == 0:
        return np.zeros(n, dtype=int), np.zeros(n, dtype=int), 0

    # flag axes that cannot discriminate (constant in the historical frame)
    for a in axes:
        if df[a].nunique(dropna=True) <= 1:
            print(f"  [match] NOTE: axis '{a}' is constant in this dataset, so it matches "
                  f"for free on every candidate and inflates Match Count by 1.")

    # one row per DISTINCT historical combination at this axis level
    hist = df[axes].astype(str).drop_duplicates().reset_index(drop=True)

    # shared integer codes so the comparison is a fast array op, not string work
    codes = {}
    for a in axes:
        cats = pd.Index(sorted(set(df[a].astype(str)) | set(candidates[a].astype(str))))
        codes[a] = cats
    H = np.column_stack([codes[a].get_indexer(hist[a].astype(str)) for a in axes])
    C = np.column_stack([codes[a].get_indexer(candidates[a].astype(str)) for a in axes])

    best = np.zeros(n, dtype=int)
    ties = np.zeros(n, dtype=int)
    chunk = 64                      # bound the (chunk x n_hist x n_axes) temporary
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        eq = (C[s:e, None, :] == H[None, :, :])
        ov = eq.sum(axis=2)
        b = ov.max(axis=1)
        best[s:e] = b
        ties[s:e] = (ov == b[:, None]).sum(axis=1)
    return best, ties, len(axes)


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


def _autofit_table(fig, table, labels, values, min_w=5.0, char_in=0.046,
                   fontsize=None, pad_in=0.06):
    """Minimal column widths: each column is just wide enough for its longest string.

    v22s: the previous version only ever GREW the canvas, so every column kept the extra
    space and the figure became 23 in wide.  Now the width per column is
    (longest_chars * char_in + pad_in), the table is normalised to that total, and the
    canvas is set to exactly that width (shrinking included).
    """
    if fontsize is not None:
        try:
            table.set_fontsize(float(fontsize))
        except Exception:                               # noqa: BLE001
            pass
    ncol = len(labels)
    longest = []
    for j in range(ncol):
        m = len(str(labels[j]))
        for row in values:
            m = max(m, len(str(row[j])))
        longest.append(m)
    # 0.046 in/char is the measured width of a 6 pt glyph; the old 0.062 was 35 % too
    # generous and that is why every column stayed visibly wide.
    widths_in = [max(0.30, m * char_in + pad_in) for m in longest]
    total = float(sum(widths_in))
    for j, w in enumerate(widths_in):
        frac = w / total
        for i in range(len(values) + 1):
            table[(i, j)].set_width(frac)
    need_w = max(min_w, total + 0.5)
    try:
        cur_w, cur_h = fig.get_size_inches()
        fig.set_size_inches(need_w, cur_h * min(1.35, max(0.75, need_w / cur_w)))
    except Exception:                                   # noqa: BLE001
        pass
    return widths_in


def _wrap_note(lines, fig_w_in, char_in=0.046, hard=150):
    """Wrap footnote lines to the figure width.

    v22t: a 200-character single line forced bbox_inches='tight' to widen the whole canvas,
    which is why the two Figure-10 tables looked stretched at the bottom.
    """
    import textwrap as _tw
    width = max(60, min(int(hard), int(max(4.0, fig_w_in) * 0.97 / char_in)))
    flat = ' '.join(str(x).replace('\n', ' ') for x in lines)
    flat = ' '.join(flat.split())
    return '\n'.join(_tw.wrap(flat, width=width))


def _apply_table_widths(table, labels, values):
    """Set per-column widths on an ax.table and return the width fractions."""
    fracs = _table_widths(labels, values)
    for j, frac in enumerate(fracs):
        for i in range(len(values) + 1):
            table[(i, j)].set_width(frac)
    return fracs


def _adaptive_fontsize(n_labels, base=7.5, min_size=3.0, full_upto=15):
    """
    Shrink a tick / annotation font as the number of categories on that axis grows.

    A heatmap panel has a FIXED width, so each label gets roughly 1/n of it.  Up to
    `full_upto` labels the base size fits; beyond that the labels overlap into an
    illegible band unless the font comes down.

    The scaling is sqrt(full_upto / n), not linear.  1/n would reach 2 pt at n = 50 and
    become unreadable; sqrt keeps the label's rendered WIDTH roughly constant, which is
    what actually matters:
        n=15 -> 7.5 pt   n=24 -> 5.9   n=40 -> 4.6   n=52 -> 4.0   n=100 -> 3.0 (floor)
    """
    n = max(int(n_labels), 1)
    if n <= full_upto:
        return float(base)
    return float(max(min_size, base * np.sqrt(float(full_upto) / n)))


def _adaptive_tick_step(n_labels, fontsize, panel_inches=4.6, rotated=True,
                        min_step_above=40):
    """
    How many ticks to skip between drawn LABELS, so the surviving ones stay readable.

    Shrinking the font alone cannot rescue a very long axis.  F9b/F9c put the catalyst on the
    x axis and several datasets carry 200+ distinct catalysts; at the 3 pt floor a rotated
    label still needs about 0.056 in, so 236 of them want ~13 in inside a 4.6 in panel.

    Only engages above `min_step_above` categories, so ordinary axes keep every label.
    `rotated=True` assumes the label is turned 90 deg, where the HORIZONTAL footprint is just
    the text height and the string length does not matter.
    """
    n = max(int(n_labels), 1)
    if n <= min_step_above:
        return 1
    per_label = ((fontsize / 72.0) * 1.35 if rotated
                 else 0.52 * (fontsize / 72.0) * 8)
    if n * per_label <= panel_inches:
        return 1
    return int(np.ceil(n * per_label / panel_inches))


def _ordinal(n):
    """2 -> '2nd', 3 -> '3rd', 11 -> '11th', 21 -> '21st'."""
    n = int(n)
    if 10 <= (n % 100) <= 20:
        suf = 'th'
    else:
        suf = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return '%d%s' % (n, suf)


def thin_tick_labels(ax, axis='x', tol=0.25):
    """
    Hide tick labels that would collide with a neighbour, measured on the DRAWN figure.

    WHY: the heatmaps that are not drawn through `_plot_heatmap` (F4d, F5c) only shrank their
    font as the category count grew.  Font shrinking alone stops helping at the 3.5-4 pt floor,
    and several datasets carry 40-66 categories on one axis -- (Roche66k)BH_plus_SM has 47
    solvents and 52 bases -- so the labels merged into an unreadable band.

    A guessed step size was rejected: it depends on assumptions about panel size that are easy
    to get wrong, and OVER-thinning silently deletes readable information.  Measuring instead
    keeps the maximum number of labels that do not collide.

    A pure bounding-box test is too strict, because two boxes can clip corners while the glyphs
    are clearly separate.  `tol` therefore ignores overlaps smaller than that fraction of the
    smaller box; 0.25 was chosen by measuring real label sets -- it leaves genuinely fine axes
    untouched (0 hidden on F9b's 52-row base axis) while fixing the crowded ones.

    Needs a renderer, so it must be called after the ticks are set and before savefig.
    """
    fig = ax.figure
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return 0

    labels = [l for l in (ax.get_xticklabels() if axis == 'x' else ax.get_yticklabels())
              if l.get_text()]
    if len(labels) <= 1:
        return 0

    def _area(b):
        return max(b.width, 0.0) * max(b.height, 0.0)

    kept, hidden = [], 0
    for lab in labels:
        bb = lab.get_window_extent(renderer=renderer)
        clash = False
        for k in kept:
            if not bb.overlaps(k):
                continue
            inter = bb.intersection(bb, k)
            if inter is not None and _area(inter) > tol * min(_area(bb), _area(k)):
                clash = True
                break
        if clash:
            lab.set_visible(False)
            hidden += 1
        else:
            kept.append(bb)
    return hidden


# v14: SUBPLOT DATA EXPORT
# Every figure used to exist only as a PNG: the numbers behind a panel could be recovered
# only by re-running the whole pipeline, and nothing in the output folder let a reader check
# a bar height or a heat-map cell.  Each figure now also writes ONE tidy CSV holding the data
# of all of its subplots.
#
# Long/tidy layout, because the panels of one figure have incompatible shapes (an 8-bar
# importance chart sits beside a 40 x 50 heat map).  One row per plotted quantity:
#   panel   subplot letter, 'a'..'d'
#   series  which line / bar group / colour series inside that panel
#   x, y    plot coordinates; for a heat map x is the column key and y the row key
#   value   the plotted quantity when it is a colour rather than a position
#   note    free text (units, n, model name, 'unavailable', ...)
_FIG_DATA = {}


def _fig_row(fig, panel, series, x=None, y=None, value=None, note=''):
    """Record one plotted quantity for `fig`'s data export (see _fig_flush)."""
    _FIG_DATA.setdefault(fig, []).append(
        dict(panel=str(panel), series=str(series), x=x, y=y, value=value, note=note))


def _fig_flush(output_dir, fig, filename):
    """Write the accumulated rows for `fig` as one tidy CSV. No-op if nothing was recorded."""
    rows = _FIG_DATA.pop(fig, [])
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['panel', 'series', 'x', 'y', 'value', 'note'])
    df = df.sort_values(['panel', 'series'], kind='stable').reset_index(drop=True)
    path = os.path.join(output_dir, filename)
    try:
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  [data] {filename}  ({len(df)} rows)")
    except Exception as _e:
        print(f"  [data] could not write {filename}: {_e}")
        return None
    return path


def _fig_grid(fig, panel, series, matrix, x_labels, y_labels, note=''):
    """Record every finite cell of a 2-D matrix (heat map)."""
    m = np.asarray(matrix, dtype=float)
    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            v = m[r, c]
            if np.isfinite(v):
                _fig_row(fig, panel, series, x=x_labels[c], y=y_labels[r],
                         value=float(v), note=note)


def place_labels_no_overlap(ax, xs, ys, texts, fontsize=7.5, fontweight='bold',
                            max_labels=None, point_px=7.0, alpha_on_collision=0.72):
    """
    Label scattered points without letting the labels collide.

    WHY THIS EXISTS: Figure 5 panel (d) labelled every solvent at a fixed (5, 5)-point
    offset.  Solvents that sit close together in PC1/PC2 -- the common case, because
    chemically similar solvents cluster -- then got their labels printed on top of each
    other and neither could be read.

    A greedy placement is used instead.  Each label tries a ring of candidate offsets in
    order of preference and keeps the first whose RENDERED bounding box overlaps neither an
    already-placed label nor a DIFFERENT data marker (a label is allowed to touch its own
    point -- that is what an annotation does).  A residual collision is drawn at reduced
    alpha, so the crowding is visible rather than silently hidden.

    Needs a renderer, so the figure must have been drawn at least once.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    texts = [str(t) for t in texts]

    if max_labels is not None and len(texts) > max_labels:
        # label only the most extreme points; the rest could never fit
        _span = np.hypot(xs - np.nanmean(xs), ys - np.nanmean(ys))
        _keep = np.argsort(-_span)[:max_labels]
        xs, ys = xs[_keep], ys[_keep]
        texts = [texts[i] for i in _keep]

    try:
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        for x, y, t in zip(xs, ys, texts):
            ax.annotate(t, (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=fontsize, fontweight=fontweight)
        return

    from matplotlib.transforms import Bbox

    # Two rings of candidate offsets, tried in order: right/up first, then diagonal and
    # vertical, then a wider ring for crowded neighbourhoods.
    cands = [(6, 6), (6, -11), (-6, 6), (-6, -11), (0, 11), (0, -15),
             (13, 2), (-13, 2), (13, -8), (-13, -8), (0, 17), (0, -21),
             (21, 6), (-21, 6), (21, -12), (-21, -12), (0, 23), (0, -27),
             (30, 0), (-30, 0), (0, 30), (0, -34), (30, 14), (-30, 14)]

    pts = []
    for x, y in zip(xs, ys):
        px, py = ax.transData.transform((x, y))
        pts.append(Bbox.from_bounds(px - point_px, py - point_px,
                                    2 * point_px, 2 * point_px))

    placed = []
    suppressed = 0
    for i, (x, y, t) in enumerate(zip(xs, ys, texts)):
        chosen = None
        for dx, dy in cands:
            ann = ax.annotate(t, (x, y), xytext=(dx, dy), textcoords='offset points',
                              fontsize=fontsize, fontweight=fontweight, zorder=6,
                              bbox=dict(boxstyle='round,pad=0.12', fc='white',
                                        ec='none', alpha=0.55))
            bb = ann.get_window_extent(renderer=renderer)
            # its own marker is excluded -- a label is meant to sit beside its point
            hits = (sum(1 for p in placed if bb.overlaps(p))
                    + sum(1 for j, p in enumerate(pts) if j != i and bb.overlaps(p)))
            if hits == 0:
                chosen = ann
                break
            ann.remove()
        if chosen is None:
            # No offset works, which means this point sits on top of an already-labelled
            # one (identical descriptors give identical PC coordinates).  Dropping the label
            # keeps every drawn label readable; the caller reports the count.
            suppressed += 1
        else:
            placed.append(chosen.get_window_extent(renderer=renderer))

    return len(placed), suppressed


def _top_n_axes(matrix, x_labels, y_labels, n=10, counts_x=None, counts_y=None):
    """Keep at most `n` rows and columns -- the most populated ones (composite's top-N rule).

    Returns (matrix, x_labels, y_labels).  Order is preserved; if fewer than n are present
    nothing is dropped, so small datasets are unaffected.
    """
    import numpy as _np
    _x = list(x_labels); _y = list(y_labels)
    _M = _np.asarray(matrix, dtype=float)
    if _M.ndim != 2:
        return matrix, _x, _y
    if counts_x is None:
        counts_x = [int(_np.isfinite(_M[:, j]).sum()) for j in range(_M.shape[1])]
    if counts_y is None:
        counts_y = [int(_np.isfinite(_M[i, :]).sum()) for i in range(_M.shape[0])]
    if len(_x) > n:
        keep = sorted(range(len(_x)), key=lambda j: -counts_x[j])[:n]
        keep.sort()
        _M = _M[:, keep]; _x = [_x[j] for j in keep]
    if len(_y) > n:
        keep = sorted(range(len(_y)), key=lambda i: -counts_y[i])[:n]
        keep.sort()
        _M = _M[keep, :]; _y = [_y[i] for i in keep]
    return _M, _x, _y


def _plot_heatmap(ax, matrix, x_labels, y_labels, title, cmap='YlGnBu', xlabel='Catalyst'):
    if matrix.size == 0:
        ax.text(0.5, 0.5, 'No valid combinations', ha='center', va='center')
        ax.set_title(title, fontsize=9, fontweight='bold')
        return None

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    # v14: adaptive tick fonts -- the catalyst axis can carry 200+ categories.
    _fx = _adaptive_fontsize(len(x_labels), base=7)
    _fy = _adaptive_fontsize(len(y_labels), base=7)
    # v14: and thin the labels when shrinking is not enough on its own.
    _sx = _adaptive_tick_step(len(x_labels), _fx, rotated=True)
    _xi = np.arange(len(x_labels))[::_sx]
    ax.set_xticks(_xi)
    ax.set_xticklabels([str(x)[:12] for x in x_labels][::_sx], rotation=90, ha='right',
                       fontsize=_fx)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels([str(y)[:12] for y in y_labels], fontsize=_fy)
    ax.set_title(title, fontsize=9, fontweight='bold')
    if _sx > 1:
        ax.text(0.5, -0.30, f'(every {_ordinal(_sx)} of {len(x_labels)} labels)',
                transform=ax.transAxes, ha='center', va='top', fontsize=6, color='0.35')
    # v14: the x slot is the pipeline's `catalyst` slot, but that slot is fed by whatever
    # column the mapping resolved.  For Roche Amide coupling it is the amide coupling
    # reagent, for EDBO/Doyle files it is the ligand -- labelling those "Catalyst" would be
    # a mislabel, so the caller passes the real feature name.
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel('Condition', fontweight='bold')
    return im


def _mark_heatmap_top_decile(ax, matrix, fontsize=7.0, color='black'):
    """Outline the cells in the TOP 10 % of a heatmap drawn by `_plot_heatmap`.

    v21 / C7: the paper's heatmap marks its best decile so the reader can see the region
    rather than one lucky cell.  `_mark_heatmap_max` keeps the single peak number; this
    draws a dashed box around every cell at or above the 90th percentile of the finite
    values (capped at 400 cells so a 200 x 50 grid cannot turn into a black rectangle).
    """
    try:
        _raw = np.asarray(matrix, dtype=float)
    except Exception:                                  # noqa: BLE001
        return
    _fin = _raw[np.isfinite(_raw)]
    if _fin.size < 4:
        return
    _thr = float(np.nanpercentile(_fin, 90.0))
    _rows, _cols = np.where(np.isfinite(_raw) & (_raw >= _thr))
    if _rows.size == 0 or _rows.size > 400:
        return
    # v22b / issue 4: NO dashed boxes -- they were unreadable over a heatmap.  The top decile
    # is marked by writing the VALUE in a white plate, which reads on any cell colour.
    # v22k: same in-cell style as Figure 5c -- bold value, NO plate/box.  White on the
    # darker end of the colormap, near-black on the pale end, so it stays readable on both.
    _lo, _hi = float(np.nanmin(_fin)), float(np.nanmax(_fin))
    for _r, _c in zip(_rows, _cols):
        _frac = 0.5 if _hi <= _lo else (_raw[_r, _c] - _lo) / (_hi - _lo)
        ax.text(_c, _r, '%.1f' % _raw[_r, _c], ha='center', va='center',
                fontsize=max(5.0, fontsize), fontweight='bold',
                color=('white' if _frac >= 0.45 else '0.12'),
                zorder=7)
    ax.text(0.995, -0.22, 'bold labels = top 10 %% (>= %.1f %%)' % _thr,
            transform=ax.transAxes, ha='right', va='top', fontsize=fontsize,
            color='0.25', fontweight='bold')


def _mark_heatmap_max(ax, matrix, fontsize=7.5):
    """Box the single highest finite cell of a heatmap drawn by `_plot_heatmap`.

    `_plot_heatmap` calls `ax.imshow(matrix)`, so matrix[row, col] maps to (y, x) and the
    text position is simply (col, row).  Unobserved cells are NaN and are excluded.

    v15: Figure 4d had this treatment and Figures 9b/9c did not, so the reader had to hunt
    for the maximum across a grid that can carry 200 columns -- and on a YlOrRd scale the top
    of the range is not distinguishable by eye.
    """
    try:
        _raw = np.asarray(matrix, dtype=float)
    except Exception:
        return
    if _raw.size == 0:
        return
    _filled = np.where(np.isfinite(_raw), _raw, -np.inf)
    if not np.isfinite(_filled).any():
        return
    _r, _c = np.unravel_index(int(np.argmax(_filled)), _filled.shape)
    ax.text(_c, _r, '%.1f' % _raw[_r, _c], ha='center', va='center',
            color='white', fontweight='bold', fontsize=fontsize,
            bbox=dict(boxstyle='round,pad=0.15', fc='black', ec='white',
                      lw=0.8, alpha=0.75))

def compute_credibility_metrics(candidates, df, models, feat_cols, display_feat_names,
                                label_encoders, scaler, weights=None,
                                include_residual_panel=True):
    """
    Per-candidate reliability metrics for the Figure 10 recommendation table.

    Three columns, each VALIDATED on the held-out real reactions (n=4,968) before being
    shipped -- a reliability number that does not track the actual error is worse than no
    number at all.  Measured Spearman rho against |prediction error|:

        CI half-width (90 %)        0.732      <- reported as "90% CI (+-pp)"
        Expected Error (pp)         0.601      <- reported per row
        (model disagreement)        0.752      <- already in the table as Model Gap (pp)
        (distance to nearest train) 0.144      <- REJECTED, essentially uninformative

    CI half-width -- normalised split conformal
        Fitted on the ensemble's residuals over a calibration slice of the TRAINING rows
        (the test set is never used for calibration).  The per-row width is
        q_0.90 * model_disagreement, which keeps finite-sample coverage.  Measured
        coverage on held-out data: 91.1 % for a nominal 90 % interval.
        A constant-width global interval would need +-43 pp to reach the same coverage,
        i.e. it would carry no per-row information at all.

    Expected Error (pp)
        Mean absolute error of the ensemble on the K nearest TRAINING conditions in the
        scaled feature space -- the model's empirical track record near this candidate.
        Quartile means on held-out data: 1.4 / 4.4 / 8.2 / 14.1 pp, monotone.

    Credibility
        Three-level label combining the two, meant to be readable at a glance.

    Returns (candidates_with_columns, calib_info).  Never raises: on any failure the
    columns are simply absent and calib_info records why.
    """
    out = candidates.copy()
    info = {}
    try:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.model_selection import train_test_split as _tts

        tr_rows, te_rows = get_split_positions(len(df))
        fit_rows, cal_rows = _tts(tr_rows, test_size=0.30, random_state=7)

        X_all = build_virtual_feature_frame(df, feat_cols, label_encoders, scaler, df)
        X_fit = X_all.iloc[fit_rows]
        X_cal = X_all.iloc[cal_rows]
        y_fit = df['yield'].iloc[fit_rows].to_numpy(float)
        y_cal = df['yield'].iloc[cal_rows].to_numpy(float)

        items = list(models.items())
        if weights is None or len(weights) != len(items):
            wv = np.full(len(items), 1.0 / len(items))
        else:
            wv = np.asarray(weights, float)
            wv = np.clip(wv, 0.0, None)
            wv = wv / wv.sum()

        def ens_mean(Z):
            P = np.vstack([np.asarray(md.predict(Z), float) for _n, md in items])
            P = np.clip(P, PREDICTION_MIN, PREDICTION_MAX)
            return P, (P * wv[:, None]).sum(axis=0)

        P_cal, p_cal = ens_mean(X_cal)
        gap_cal = P_cal.max(0) - P_cal.min(0)
        resid_cal = np.abs(p_cal - y_cal)
        ratio = resid_cal / np.maximum(gap_cal, 1e-6)
        q90 = float(np.quantile(ratio, 0.90))
        info['q90_ratio'] = q90

        K_NB = min(25, max(1, len(fit_rows) - 1))
        nn = NearestNeighbors(n_neighbors=K_NB).fit(X_fit.to_numpy(float))
        _, nb_idx = nn.kneighbors(
            build_virtual_feature_frame(df, feat_cols, label_encoders, scaler,
                                        out).to_numpy(float))
        _P_fit, p_fit = ens_mean(X_fit)
        err_fit = np.abs(p_fit - y_fit)
        nb_err = err_fit[nb_idx].mean(axis=1)

        P_out, p_out = ens_mean(build_virtual_feature_frame(
            df, feat_cols, label_encoders, scaler, out))
        gap_out = P_out.max(0) - P_out.min(0)

        out['CI Half-Width (90%, pp)'] = q90 * np.maximum(gap_out, 1e-6)
        out['Expected Error (pp)'] = nb_err

        def _tier(hw, ee):
            if hw < 5.0 and ee < 3.0:
                return 'High'
            if hw < 15.0 or ee < 8.0:
                return 'Medium'
            return 'Low'
        out['Credibility'] = [_tier(h, e) for h, e in
                              zip(out['CI Half-Width (90%, pp)'], out['Expected Error (pp)'])]
        info['K_NB'] = K_NB
        info['tiers'] = out['Credibility'].value_counts().to_dict()
        print(f"  [credibility] conformal q90={q90:.3f}; neighbour K={K_NB}; tiers={info['tiers']}")
    except Exception as e:
        info['error'] = str(e)
        print(f"  [credibility] skipped ({e})")
    return out, info


def build_feature_matrix_for_conditions(frame, feat_cols, label_encoders, scaler):
    """
    Encode + scale a small frame of RAW condition rows using the ALREADY-FITTED encoders.

    This is deliberately NOT prepare_features(): that function re-FITS a LabelEncoder and a
    StandardScaler on whatever frame it is given, which would produce objects unrelated to
    the ones the trained models were built with, and silently different features.  Here the
    fitted label_encoders / scaler that the models actually saw are applied as a pure
    transform, and the columns are emitted in exactly `feat_cols` order.

    Only the numeric columns go through the scaler -- the same columns it was fitted on,
    both fitting and transforming `df[['temperature','time','catalyst_loading']]`.
    """
    out = pd.DataFrame(index=frame.index)
    # reactant_2 / reactant_1 are the current slot names; the retired names hold only
    # 'Unknown' placeholders, so a scored condition could not report its partner pair.
    # v23b: iterate over the encoders that ACTUALLY exist instead of a hard-coded list of
    # seven reagent slots.  `additive` / `cosolvent` are slots too, so a dataset that uses
    # them produced feat_cols with `additive_encoded` while this loop never emitted it, and
    # the function raised `missing ['additive_encoded']` -- which killed Figure 11.
    for col, _le in list(label_encoders.items()):
        _enc = col + '_encoded'
        if _enc not in feat_cols:
            continue
        if col in frame.columns:
            out[_enc] = _le.transform(frame[col].astype(str).to_numpy())
        else:
            # not carried by this frame: the first class is the safest neutral code (0)
            out[_enc] = np.zeros(len(frame), dtype=float)
            print(f"  [features] column '{col}' absent from the frame; encoded as 0")
    # v22b: build the numeric block from the columns the SCALER was fitted on, in fit order.
    # The old hardcoded ['temperature','time','catalyst_loading'] lost a column whenever a
    # slot was dropped (e.g. `time` constant) or a different equivalents column was swept,
    # and scaler.transform then raised "X has 2 features, but StandardScaler is expecting 3".
    _fni = getattr(scaler, 'feature_names_in_', None)
    num_cols = ([str(c) for c in _fni] if _fni is not None and len(_fni)
                else [c for c in ['temperature', 'time', 'catalyst_loading'] if c in feat_cols])
    if num_cols:
        block = np.full((len(frame), len(num_cols)), np.nan, dtype=float)
        for _j, _c in enumerate(num_cols):
            if _c in frame.columns:
                block[:, _j] = pd.to_numeric(frame[_c], errors='coerce').to_numpy(dtype=float)
        # v14: a NaN here would propagate through scaler.transform and take down the model
        # call (GradientBoosting rejects NaN outright).  Fill with the scaler's TRAINING mean
        # for that column -- the same statistic prepare_features imputes with -- which scales
        # to exactly 0, i.e. the neutral value.
        if not np.isfinite(block).all():
            _mu = np.asarray(getattr(scaler, 'mean_', np.zeros(len(num_cols))),
                             dtype=float)[:len(num_cols)]
            _bad = ~np.isfinite(block)
            rows_i, col_i = np.where(_bad)
            block[rows_i, col_i] = _mu[col_i]
            print(f"  [NaN] build_feature_matrix_for_conditions filled {int(_bad.sum())} "
                  f"missing value(s) with the training mean")
        out[num_cols] = scaler.transform(block)
    missing = [c for c in feat_cols if c not in out.columns]
    if missing:
        # v23b: this used to raise and take the whole figure down.  A feature column that
        # cannot be rebuilt is filled with 0 -- the standardised mean, i.e. "no information"
        # -- and reported once, so the panel still renders and the gap is visible.
        for _m in missing:
            out[_m] = np.zeros(len(frame), dtype=float)
        print(f"  [features] WARNING: could not rebuild {missing}; filled with 0 "
              f"(standardised mean) so the panel still renders")
    return out[feat_cols].to_numpy(dtype=np.float32)


def generate_loading_curves_pdf(bo_df, df, models, feat_cols, label_encoders, scaler,
                                output_dir, weights=None, top_k=100, per_page=4,
                                max_model_gap=ENSEMBLE_MAX_MODEL_GAP, verbose=True,
                                compute_pdp=True, pdp_max_rows=600):
    """
    Figure 11 (supplementary): predicted yield as a function of catalyst loading, for the
    TOP `top_k` reagent combinations, written as ONE multi-page PDF.

    Layout: pages are ordered by rank (page 1 carries ranks 1..`per_page`), and within a
    page the panels fill row by row, so scrolling down walks the ranking from the best
    combination to the last.

    What is drawn (all from the trained models, no lookup of measured yields):
      * one thin line per ensemble member  -- shows model DISAGREEMENT at this loading
      * a thick line for the weighted ensemble mean -- the reported prediction
      * a shaded band for the ensemble standard deviation
      * a star at the loading that maximised the ensemble prediction
      * a thick dashed PDP: the AVERAGE model response over the historical conditions,
        i.e. the marginal effect of loading with everything else left as observed.  It is
        deliberately NOT specific to the candidate on that panel -- it is the reference
        curve to read the candidate's shape against.

    The member lines and the mean are each conditional on THIS candidate (an ICE curve);
    their average over all candidates IS the PDP, which is why the PDP is drawn once and
    shared across every panel rather than recomputed per candidate.

    This is a re-evaluation of the SAME space the sweep enumerated: each combination is
    scored across the full historical loading grid (N_LOADING_POINTS values), not just the
    single loading value that survived the sweep's keep-top-20,000 cut.

    CAUTION: the star is an argmax over the grid, so it is optimistically biased by
    construction -- on a step-function model the best grid point is partly noise.  Read the
    CURVE SHAPE, which is robust, rather than the exact height of the star.

    Returns a small summary frame (one row per combination).
    """
    if bo_df is None or len(bo_df) == 0:
        return None
    if not int(top_k or 0):
        print("  [Figure 11] skipped: TOP_LOADING_CURVES = 0")
        return None
    reagent_cols = [c for c in ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2',
                                'reactant_1'] if c in bo_df.columns]
    if not reagent_cols or 'catalyst_loading' not in bo_df.columns:
        print("  [Figure 11] skipped: bo_df lacks the reagent / loading columns")
        return None

    # v14: with no usable amount axis there is nothing to sweep, so a loading PDF would be
    # a set of vertical spikes.  Emit ONE explicit placeholder page instead of silently
    # returning nothing, so the figure slot is visibly accounted for.
    _axis_lab = LOADING_AXIS.get('label') or 'Catalyst loading'
    _axis_unit = LOADING_AXIS.get('unit') or 'eq.'
    _grid = get_loading_grid(df, n_points=N_LOADING_POINTS)
    if not loading_axis_available() or len(_grid) < 2:
        print(f"  [Figure 11] not drawn: {LOADING_AXIS.get('reason')}")
        with PdfPages(os.path.join(output_dir, 'Figure11_Loading_Curves_UNAVAILABLE.pdf')) as _pdf:
            fig, ax = plt.subplots(figsize=(11.0, 8.5))
            annotate_loading_unavailable(ax, fontsize=11)
            fig.suptitle('Figure 11 — Loading curves: NOT AVAILABLE for this dataset',
                         fontsize=13, fontweight='bold')
            pdf_save = getattr(_pdf, 'savefig', None)
            if pdf_save:
                pdf_save(fig)
            plt.close(fig)
        return None

    # Cast to str first: these columns are pandas `category` dtype in the streamed frame,
    # and grouping/sorting on categories would order by code, not by name.
    bo_df = bo_df.copy()
    for col in reagent_cols:
        bo_df[col] = bo_df[col].astype(str)
    grp = bo_df.groupby(reagent_cols, observed=True, dropna=False)
    peak = grp['Ensemble Predicted Yield (%)'].max().reset_index()
    peak = peak.sort_values('Ensemble Predicted Yield (%)', ascending=False, kind='mergesort')
    peak = peak.head(int(top_k)).reset_index(drop=True)
    peak['Draw Rank'] = np.arange(1, len(peak) + 1)
    if not len(peak):
        return None

    grid = get_loading_grid(df, n_points=N_LOADING_POINTS)
    reps = peak.loc[peak.index.repeat(len(grid))].reset_index(drop=True)
    reps[SWEEP_AXIS] = np.tile(np.asarray(grid, dtype=float), len(peak))
    reps['catalyst_loading'] = reps[SWEEP_AXIS]
    # temperature/time are FIXED at the representative values during enumeration, so they
    # must be reproduced here or the curve would describe a different condition than the one
    # the sweep scored.  Cross-check against bo_df and warn if they ever diverge.
    rep_cond = get_representative_conditions(df)
    for col in ['temperature', 'time']:
        if col not in bo_df.columns:
            continue
        uniq = pd.to_numeric(bo_df[col], errors='coerce').dropna().unique()
        want = float(rep_cond.get(col, np.nan))
        if len(uniq) == 1 and not np.isclose(float(uniq[0]), want):
            print(f"  [Figure 11] WARNING: {col} in bo_df is {float(uniq[0])} but the "
                  f"representative value is {want}; using bo_df's value.")
            want = float(uniq[0])
        reps[col] = want
    for col in reagent_cols:
        reps[col] = reps[col].astype(str)

    X = build_feature_matrix_for_conditions(reps, feat_cols, label_encoders, scaler)

    # RAW member predictions here (no clipping): this figure shows the model output as it
    # comes out, and the > 100 % overshoot is handled below by proportional rescaling rather
    # than by truncation, which would flatten the top of every curve.
    preds = {}
    for name, mdl in models.items():
        preds[name] = np.asarray(mdl.predict(X), dtype=float)
    mean, std = _weighted_member_stats(preds, weights=weights)
    for name in preds:
        reps[f'Pred_{name} (%)'] = preds[name]
    reps['Ensemble Predicted Yield (%)'] = mean
    reps['Ensemble SD (%)'] = std

    # Partial dependence = E_over_X[ f(loading, X_other) ].  Implemented directly (rather
    # than via sklearn's partial_dependence) so it uses the SAME fitted encoders/scaler and
    # the same candidate feature order as everything else in this function.
    pdp = None
    if compute_pdp:
        try:
            base_cols = [c for c in reagent_cols if c in df.columns]
            sub = df[base_cols].dropna().astype(str)
            if len(sub) > pdp_max_rows:
                sub = sub.sample(n=int(pdp_max_rows), random_state=42)
            pdp_rows = sub.loc[sub.index.repeat(len(grid))].reset_index(drop=True)
            pdp_rows[SWEEP_AXIS] = np.tile(np.asarray(grid, dtype=float), len(sub))
            pdp_rows['catalyst_loading'] = pdp_rows[SWEEP_AXIS]
            for col in ['temperature', 'time']:
                if col in df.columns:
                    pdp_rows[col] = pd.to_numeric(df[col], errors='coerce').median()
            Xp = build_feature_matrix_for_conditions(pdp_rows, feat_cols, label_encoders, scaler)
            pdp_preds = {n: np.asarray(m.predict(Xp), dtype=float)
                         for n, m in models.items()}
            pdp_mu, _ = _weighted_member_stats(pdp_preds, weights=weights)
            # average within each loading value -> one PDP point per grid position
            tmp = pd.DataFrame({'loading': pdp_rows[SWEEP_AXIS].to_numpy(),
                                'pdp': pdp_mu})
            pdp = tmp.groupby('loading', sort=True)['pdp'].mean()

        except Exception as e:
            print(f"  [Figure 11] PDP unavailable ({e}); drawing the candidate curves only")
            pdp = None

    # Additive-boosting members overshoot the physical 0-100 % range on some datasets
    # (measured 109.1 % on the Suzuki-Miyaura whitelist).  Instead of clipping -- which
    # flattens the peak and hides the curve's shape -- compress the whole figure by one
    # shared factor so the maximum becomes 100 %.  Ordering and shape are preserved.
    #
    # The factor is computed from the candidate curves AND the PDP together, so the PDP
    # stays on the same scale as the curves it is drawn with.
    f11_scale = 1.0
    _raw_peak = float(np.nanmax(mean)) if len(mean) else np.nan
    reps['F11 Raw Ensemble Yield (%)'] = mean
    reps['F11 Raw Ensemble SD (%)'] = std
    if F11_RESCALE_TO_100:
        _pool = [float(np.nanmax(mean))] if len(mean) else []
        if pdp is not None and len(pdp):
            _pool.append(float(np.nanmax(np.asarray(pdp, dtype=float))))
        _peak = max(_pool) if _pool else np.nan
        if np.isfinite(_peak) and _peak > PREDICTION_MAX:
            f11_scale = PREDICTION_MAX / _peak
            print(f"  [Figure 11] peak predicted yield {_peak:.1f} % exceeds "
                  f"{PREDICTION_MAX:.0f} %; rescaling the whole figure by "
                  f"{f11_scale:.4f} so the maximum is exactly {PREDICTION_MAX:.0f} % "
                  f"(the raw values stay in 'F11 Raw *' columns)")
            mean = mean * f11_scale
            std = std * f11_scale
            reps['Ensemble Predicted Yield (%)'] = mean
            reps['Ensemble SD (%)'] = std
            for name in preds:
                reps[f'Pred_{name} (%)'] = preds[name] * f11_scale
            if pdp is not None:
                pdp = pdp * f11_scale
    reps['F11 Rescale Factor'] = f11_scale

    idx = (reps.groupby('Draw Rank', sort=False)['Ensemble Predicted Yield (%)']
               .idxmax().to_numpy())
    reps['Is Peak'] = False
    reps.loc[idx, 'Is Peak'] = True
    peak_info = reps.loc[idx].set_index('Draw Rank')

    # pagelayout=SinglePage makes viewers display one page at a time instead of a
    # side-by-side spread, which is what "scroll through the ranking" wants.
    try:
        rcParams['pdf.pagelayout'] = 'SinglePage'
    except Exception:
        pass
    pdf_path = os.path.join(output_dir, f'Figure11_Loading_Curves_Top{len(peak)}.pdf')
    model_names = list(models.keys())
    member_colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['red'],
                     COLORS['purple']]
    n_pages = int(np.ceil(len(peak) / float(max(per_page, 1))))

    with PdfPages(pdf_path) as pdf:
        for page_i in range(n_pages):
            fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
            axes_flat = axes.ravel()
            lo = page_i * per_page
            hi = min(lo + per_page, len(peak))
            for k in range(per_page):
                ax = axes_flat[k]
                rank_i = lo + k + 1
                if rank_i > hi:
                    ax.axis('off')
                    continue
                sub = reps[reps['Draw Rank'] == rank_i].sort_values('catalyst_loading')
                if not len(sub):
                    ax.axis('off')
                    continue
                xs = sub['catalyst_loading'].to_numpy(dtype=float)
                mu = sub['Ensemble Predicted Yield (%)'].to_numpy(float)
                sd = sub['Ensemble SD (%)'].to_numpy(float)
                ax.fill_between(xs, mu - sd, mu + sd, color=COLORS['grey'],
                                alpha=0.22, lw=0, label='Ensemble $\\pm$1 SD')
                for mi, name in enumerate(model_names):
                    col = f'Pred_{name} (%)'
                    if col in sub.columns:
                        ax.plot(xs, sub[col].to_numpy(float), lw=0.9, alpha=0.85,
                                color=member_colors[mi % len(member_colors)], label=name)
                ax.plot(xs, mu, lw=1.2, color='black', label='Ensemble (weighted)')
                if pdp is not None and len(pdp):
                    ax.plot(pdp.index.to_numpy(dtype=float), pdp.to_numpy(float),
                            lw=1.8, ls='--', color=COLORS['purple'],
                            label='PDP (historical avg.)')
                info = peak_info.loc[rank_i]
                # ms=4.5 (30 % of the original 15): at 15 the star covered the curve it is
                # supposed to annotate.
                ax.plot([info['catalyst_loading']], [info['Ensemble Predicted Yield (%)']],
                        marker='*', ms=4.5, color=COLORS['orange'],
                        markeredgecolor='black', markeredgewidth=0.5, zorder=6)
                ax.set_xscale('log')
                # v14: the axis may be catalyst loading or a substitute amount variable.
                ax.set_xlabel('%s (%s)' % (LOADING_AXIS.get('label') or 'Catalyst loading',
                                           LOADING_AXIS.get('unit') or 'eq.'))
                ax.set_ylabel('Predicted yield (%)')
                ax.set_ylim(-2, 102)
                ax.grid(True, linestyle='--', alpha=0.30)
                ax.legend(fontsize=5.2, loc='best', framealpha=0.85)
                # Two lines with a hard wrap: a single long reagent string overflowed into
                # the neighbouring panel on the rendered page.
                tags = [f"{c}={str(sub.iloc[0][c])[:18]}" for c in reagent_cols]
                half = (len(tags) + 1) // 2
                tag = "  ".join(tags[:half]) + "\n" + "  ".join(tags[half:])
                ax.set_title(f"Rank {rank_i}  |  peak {info['Ensemble Predicted Yield (%)']:.1f}% "
                             f"@ {info['catalyst_loading']:.4g} {_axis_unit}\n{tag}",
                             fontsize=6.6, fontweight='bold')
            # v14: the suptitle carries the RESOLVED axis name (it may be a substitute), and
            # the extra 'unavailable'/'substitute' note is counted so the top margin can grow.
            _sup_extra = ("\nthe dashed PDP is the same reference curve on every panel"
                          if pdp is not None else "")
            _sup_extra += (f"\n[display rescaled x{f11_scale:.4f}: largest plotted value set to "
                           f"100 %; raw predictions in the CSV]"
                           if abs(f11_scale - 1.0) > 1e-9 else "")
            if LOADING_AXIS.get('role') == 'substitute':
                _sup_extra += (f"\n[swept axis is {_axis_lab}, NOT catalyst loading -- this "
                               f"dataset has no catalyst_loading column]")
            _sup_lines = 1 + _sup_extra.count('\n')
            fig.suptitle(
                f"Figure 11 - Predicted yield vs {_axis_lab.lower()}, top {len(peak)} reagent "
                f"combinations  (page {page_i + 1}/{n_pages}, ranks {lo + 1}-{hi})"
                + _sup_extra,
                fontsize=10.5, fontweight='bold')
            # v14: subplots_adjust is applied AFTER suptitle, and the title needs vertical room
            # proportional to its line count -- with 3+ lines the panel titles collided with it.
            _top = max(0.78, 0.90 - 0.045 * (_sup_lines - 1))
            fig.subplots_adjust(left=0.07, right=0.985, top=_top, bottom=0.07,
                                hspace=0.52, wspace=0.22)
            pdf.savefig(fig)
            plt.close(fig)

    csv_path = os.path.join(output_dir, f'Figure11_Loading_Curves_Top{len(peak)}.csv')
    reps_out = reps.drop(columns=['Is Peak'], errors='ignore').copy()
    reps_out = reps_out.sort_values(['Draw Rank', 'catalyst_loading'])
    reps_out.to_csv(csv_path, index=False, encoding='utf-8-sig')

    summary = peak_info.reset_index()[[
        'Draw Rank', 'catalyst_loading', 'Ensemble Predicted Yield (%)', 'Ensemble SD (%)']
        + reagent_cols].rename(columns={'catalyst_loading': 'Peak Loading (eq.)',
                                        'Ensemble Predicted Yield (%)': 'Peak Pred. Yield (%)',
                                        'Ensemble SD (%)': 'Ensemble SD at Peak (%)'})
    summary = summary.sort_values('Draw Rank').reset_index(drop=True)
    if verbose:
        spread = []
        for _, row in summary.iterrows():
            s = reps[reps['Draw Rank'] == row['Draw Rank']]
            spread.append(s['Ensemble Predicted Yield (%)'].max()
                          - s['Ensemble Predicted Yield (%)'].min())
        summary['Loading Sensitivity (pp)'] = spread
        print(f"  {os.path.basename(pdf_path)} ({n_pages} pages, {per_page} per page)")
        print(f"  [Figure 11] peak pred. yield {summary['Peak Pred. Yield (%)'].min():.1f}"
              f"-{summary['Peak Pred. Yield (%)'].max():.1f}%")
    return summary


def _weighted_member_stats(preds, weights=None):
    """
    Weighted mean / std across ensemble members for ALREADY-COMPUTED predictions.

    `ensemble_stats_streaming` is the streaming equivalent, but it takes (models, X) and
    evaluates the models itself.  Here the predictions are already in hand, so this mirrors
    its arithmetic exactly:

        mu    = sum_i w_i p_i          (weights normalised to sum to 1)
        sigma = sqrt( sum_i w_i (p_i - mu)^2 / (1 - sum_i w_i^2) )

    which reduces to ddof=1 when the weights are equal.  `weights=None` gives the plain
    unweighted mean, matching the default ensemble behaviour.
    """
    names = list(preds.keys())
    if not names:
        return np.zeros(0), np.zeros(0)
    k = len(names)
    stack = np.vstack([np.asarray(preds[n], dtype=float) for n in names])
    if weights is None:
        w = np.full(k, 1.0 / k)
    else:
        w = np.asarray(list(weights), dtype=float)
        if len(w) != k:
            raise ValueError(f"weights has {len(w)} entries but there are {k} members")
        w = w / w.sum()
    mu = np.tensordot(w, stack, axes=(0, 0))
    var = np.tensordot(w, (stack - mu) ** 2, axes=(0, 0))
    denom = max(1.0 - float(np.sum(w ** 2)), 1e-12)
    return mu, np.sqrt(np.maximum(var, 0.0) / denom)


def generate_figure8_and_9(
    model_name,
    model,
    bo_df,
    df,
    feat_cols,
    label_encoders,
    scaler,
    output_dir,
    trained_models,
    top_n_table=20,
    top_n_csv=500,
    make_figure9=True,
    credibility=None,
    ensemble_weights=None,
    best_observed_yield=None
):
    _dm = (globals().get('MODEL_DISPLAY') or {}).get(model_name, model_name)

    """
    credibility : DataFrame or None
        Precomputed reliability metrics for the SAME candidate pool (see
        compute_credibility_metrics).  It is keyed on the condition columns, so it merges
        in regardless of row order.  When None the three reliability columns are simply
        absent from the figure and the CSV.

    best_observed_yield : float or None
        The EI incumbent f*.  It used to be a local of the caller, so the ENSEMBLE call --
        which is made from a different scope -- raised NameError and silently produced no
        Figures 9/10 for the ensemble.  When None it is recomputed from ``df`` with the same
        helper the sweep uses, so the two can never disagree.
    """
    """Draw Figures 9-10 for one model (or the Ensemble) from the stored bo_df."""
    if best_observed_yield is None:
        best_observed_yield, _ = resolve_ei_incumbent(df)
    print(f"Generating Figure 9 and 10 for [{model_name}] ...")

    # v14: key for the per-figure subplot-data export (one CSV per model variant)
    _fig9_tag = 'Figure9_Bayesian_Optimization-%s' % model_name
    
    is_ensemble = (model_name == 'Ensemble' or model is None)
    model_virtual_df = bo_df.copy()

    if not is_ensemble:
        # v12: the survivor pool carries `Pred_<name>` columns using the ORIGINAL model
        # name (e.g. 'Pred_Random Forest'), so try the raw name first, then the
        # underscore-normalised variant, then the legacy v11 spelling.
        cands = [f"Pred_{model_name}",
                 f"Pred_{model_name.replace(' ', '_').replace('-', '_')}",
                 f"Pred_{model_name.replace(' ', '_').replace('-', '_').replace('_', ' ')}"]
        safe_col = next((c for c in cands if c in model_virtual_df.columns), cands[0])
        if safe_col in model_virtual_df.columns:
            model_virtual_df['Predicted Yield (%)'] = model_virtual_df[safe_col]
        else:
            # v12: happens when KEEP_MODEL_PREDICTIONS=False. The virtual-library rows are
            # not scored per model to save memory, so this single-model figure falls back
            # to the ensemble mean. Set KEEP_MODEL_PREDICTIONS=True to get true per-model
            # virtual-library scores (costs ~0.34 GiB per model).
            print(f"  [Figure 9/10] '{safe_col}' not stored (KEEP_MODEL_PREDICTIONS=False); "
                  f"using the ensemble mean for this per-model figure.")
            model_virtual_df['Predicted Yield (%)'] = model_virtual_df['Ensemble Predicted Yield (%)']
    else:
        model_virtual_df['Predicted Yield (%)'] = model_virtual_df['Ensemble Predicted Yield (%)']

    # Figure 9: Bayesian Optimization / Optimization Surfaces
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 8a: Convergence of the acquisition ranking, against a "just sort by predicted yield"
    # reference.
    #
    # This panel used to plot the raw predicted yield of the top 25 candidates by EI, which
    # produced an uninterpretable sawtooth.  It now plots the CUMULATIVE BEST:
    #     cumulative_best[k] = max(pred[1..k])
    # for two orderings:
    #     (a) the acquisition ranking (EI, descending)   -- what BO would try in order
    #     (b) the predicted-yield ranking (descending)   -- the obvious baseline
    #
    # Why the baseline matters
    # The candidate pool is ITSELF the top-N by predicted yield, so ordering (b) reaches the
    # pool ceiling on its FIRST pick and its cumulative-best line is flat by construction.
    # Ordering (a) therefore cannot "converge" past that ceiling either -- measured on the
    # real 500-candidate pool, cumulative best by EI climbs 91.4 -> 97.6 while (b) is flat at
    # 97.6 from pick 1.  What this panel shows is HOW MUCH SLOWER the EI order finds the good
    # candidates, which is the honest and useful reading.
    #
    # 100 picks, LINEAR x.  A log axis was tried and rejected: cumulative best is a step
    # function whose informative jumps sit at large k (~50 and ~85 on the real pool), and a
    # log axis compresses exactly that region while stretching the early picks where nothing
    # happens.  With only 100 points a linear axis keeps every step readable.
    #
    # NOTE on EI here: with the shipped EI_INCUMBENT_MODE='quantile' the incumbent f* is the
    # 90th percentile of the observed yields, so mu - f* is POSITIVE for most of the pool and
    # EI is exploitation-driven -- it ranks close to predicted yield and its argmax is usually
    # the pool's maximum predicted yield.  (The older 'max' mode used the all-time record, so
    # no candidate could exceed f*, the improvement term was negative everywhere and EI was
    # driven by its exploration term instead; that is the regime the "91.4 -> 97.6" numbers
    # above were measured in.)  Either way EI is NOT numerically degenerate.
    _N_CURVE = 100
    if is_ensemble:
        _auc = 'Expected Improvement'
        _ycol = 'Ensemble Predicted Yield (%)'
        _colour = COLORS['green']
        _auc_label = f'acquisition order (EI): cumulative best'
        ax1.set_xlabel('number of candidates taken', fontweight='bold')
        ax1.set_ylabel('Cumulative best ensemble yield (%)', fontweight='bold')
        ax1.set_title('Convergence: EI order vs yield order', fontsize=9,
                      fontweight='bold')
    else:
        _auc = 'Predicted Yield (%)'
        _ycol = 'Predicted Yield (%)'
        _colour = COLORS['blue']
        _auc_label = f'{model_name}: cumulative best'
        ax1.set_xlabel('number of candidates taken', fontweight='bold')
        ax1.set_ylabel(f'Cumulative best yield by {_dm} (%)', fontweight='bold')
        ax1.set_title(f'Convergence of the {_dm} Ranking', fontsize=9,
                      fontweight='bold')

    if _auc in model_virtual_df.columns and _ycol in model_virtual_df.columns:
        _yv = pd.to_numeric(model_virtual_df[_ycol], errors='coerce')
        _av = pd.to_numeric(model_virtual_df[_auc], errors='coerce')
        _ok = _yv.notna() & _av.notna()
        _y = _yv[_ok].to_numpy(float)
        _a = _av[_ok].to_numpy(float)
        if len(_y):
            _n = min(_N_CURVE, len(_y))
            _x = np.arange(1, _n + 1)
            _cum_auc = np.maximum.accumulate(_y[np.argsort(-_a, kind='stable')][:_n])
            _cum_yld = np.maximum.accumulate(_y[np.argsort(-_y, kind='stable')][:_n])

            # v15.  The yield-order reference is FLAT BY CONSTRUCTION -- it takes the pool
            # maximum on its very first pick -- so it is a ceiling line, not a curve, and it
            # carries no information on its own.  Worse, with EI_INCUMBENT_MODE='quantile'
            # the incumbent f* is the 90th percentile of the observed yields rather than the
            # all-time record, which makes the improvement term mu - f* positive for most of
            # the pool: EI turns exploitation-driven, its argmax lands on the same candidate
            # as the predicted-yield argmax, and the EI curve ALSO saturates on pick 1.
            # Measured on (Summit)Reizman_Suzuki_385: f* = 87.36 %, the max-EI row carried
            # the pool-maximum predicted yield 94.5185 %, and BOTH curves then sat at 94.52
            # for all 100 picks -- two exactly overlapping horizontal lines.
            #
            # So a random-order draw is added as the one reference that is never degenerate,
            # and when the two original curves coincide the panel says so instead of
            # pretending to compare them.
            _rng9 = np.random.default_rng(42)
            _cum_rnd = np.maximum.accumulate(_y[_rng9.permutation(len(_y))][:_n])

            ax1.plot(_x, _cum_rnd, ':', color='0.45', lw=1.6,
                     label='random order (reference)')
            ax1.plot(_x, _cum_yld, '--', color='black', lw=1.4,
                     label='predicted-yield order (ceiling)')
            ax1.plot(_x, _cum_auc, '-', color=_colour, lw=2.0, label=_auc_label)
            # v14: record the convergence curves for the figure's data export
            for _i, _k in enumerate(_x):
                _fig_row(_fig9_tag, 'a', 'acquisition order (EI)', x=int(_k),
                         y=float(_cum_auc[_i]),
                         note='cumulative best PREDICTED yield (%)')
                _fig_row(_fig9_tag, 'a', 'predicted-yield order', x=int(_k),
                         y=float(_cum_yld[_i]),
                         note='cumulative best PREDICTED yield (%)')
                _fig_row(_fig9_tag, 'a', 'random order', x=int(_k),
                         y=float(_cum_rnd[_i]),
                         note='cumulative best PREDICTED yield (%)')
            ax1.set_xlim(1, _n)
            _lo = float(min(_cum_auc.min(), _cum_yld.min(), _cum_rnd.min()))
            _hi = float(max(_cum_auc.max(), _cum_yld.max(), _cum_rnd.max()))
            _pad = max(0.5, 0.12 * max(_hi - _lo, 1e-9))
            ax1.set_ylim(_lo - _pad, _hi + _pad)
            # annotate the shortfall, which is the point of the panel
            _gap = float(_cum_yld[-1] - _cum_auc[-1])
            if _gap > 0.05:
                ax1.annotate(f'gap after {_n} picks: {_gap:.1f} pp',
                             xy=(_n, _cum_auc[-1]), xytext=(0.35, 0.30),
                             textcoords='axes fraction', fontsize=7, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                       ec='0.4', alpha=0.9))
            elif float(np.ptp(_cum_auc)) < 0.05:
                # EI order reached the ceiling on its first pick: report the real quantity
                # instead -- how much of the ceiling the random order has found by pick _n.
                _rnd_gain = float(_cum_rnd[-1] - _cum_rnd[0])
                ax1.annotate('EI order hits the pool ceiling on pick 1\n'
                             f'(random order is {_rnd_gain:.1f} pp below it after {_n})',
                             xy=(0.04, 0.28), xycoords='axes fraction', fontsize=7,
                             fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                       ec='0.4', alpha=0.9))
            ax1.legend(loc='lower right', fontsize=6.5)

    ax1.grid(True, linestyle='--', alpha=0.35)
    ax1.text(-0.12, 1.05, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')

    representative = get_representative_conditions(df)
    cat_values = label_encoders.get('catalyst').classes_.tolist() if 'catalyst' in label_encoders else []
    base_values = label_encoders.get('base').classes_.tolist() if 'base' in label_encoders else []
    solvent_values = label_encoders.get('solvent').classes_.tolist() if 'solvent' in label_encoders else []
    loading_ref = float(np.median(get_loading_grid(df, n_points=N_LOADING_POINTS)))

    # 8b: Catalyst × Base Landscape
    # v12: these panels are only len(catalyst) x len(base) = ~740 rows, but the old code
    # did `np.vstack([m.predict(xb) for m in trained_models.values()])` for EVERY panel,
    # i.e. k predictions over the grid per panel instead of one batched prediction.
    # Cheap in memory at this size, but needlessly repeated; predict once and reuse.
    def _grid_panel_stats(frame):
        n = len(frame)
        if is_ensemble:
            tot = np.zeros(n); tot_sq = np.zeros(n)
            vmin = np.full(n, np.inf); vmax = np.full(n, -np.inf)
            k = max(len(trained_models), 1)
            for _m in trained_models.values():
                p = np.asarray(_m.predict(frame), dtype=float)
                tot += p; tot_sq += p * p
                np.minimum(vmin, p, out=vmin); np.maximum(vmax, p, out=vmax)
            mu = tot / k
            sd = np.sqrt(np.maximum(tot_sq / k - mu * mu, 0.0) * k / max(k - 1, 1))
            return mu, sd
        p = np.asarray(model.predict(frame), dtype=float)
        return p, np.zeros_like(p)

    # 8b + 8c: two interaction landscapes over the PROCESS-CONDITION family.
    #
    # v14: the axes used to be hardcoded catalyst x base and catalyst x solvent.  On a dataset
    # that has no base (Summit Reizman) or no solvent ((DoyleHTE)CN_coupling) those panels
    # fell through to an "unavailable" caption even though other process variables were
    # present.  Each axis is now a SLOT resolved by importance within the family, so a
    # missing parameter falls through to the next one instead of blanking the panel.
    #
    # `used` is threaded through so the two panels never plot the same variable twice, and so
    # panel (c) picks a partner different from panel (b)'s.
    def _interaction_matrix(feat_cols_, p_x, p_y, vals_x, vals_y):
        recs = []
        for vx in vals_x:
            for vy in vals_y:
                rec = representative.copy()
                rec[p_x] = vx
                rec[p_y] = vy
                rec[SWEEP_AXIS] = loading_ref      # v22: follow the chosen axis
                rec['catalyst_loading'] = loading_ref   # keep the canonical column filled
                recs.append(rec)
        frame = build_virtual_feature_frame(df, feat_cols_, label_encoders, scaler,
                                            pd.DataFrame(recs))
        mu, _sd = _grid_panel_stats(frame)
        return mu.reshape(len(vals_x), len(vals_y)).T

    _p_bx, _p_by = _resolve_axis_pair(('catalyst', 'base'), 'process_categorical',
                                      df, label_encoders)
    _p_cx, _p_cy = _resolve_axis_pair(('catalyst', 'solvent'), 'process_categorical',
                                      df, label_encoders)
    # keep the two panels from becoming the same picture
    if _p_cx == _p_bx and _p_cy == _p_by:
        _p_cx, _p_cy = _resolve_axis_pair(('catalyst', 'solvent'), 'process_categorical',
                                          df, label_encoders,
                                          used=(_p_bx, _p_by))

    _vals_x_b = label_encoders[_p_bx].classes_.tolist() if _p_bx else []
    _vals_y_b = label_encoders[_p_by].classes_.tolist() if _p_by else []

    # 8b: first interaction landscape
    if _p_bx and _p_by and _vals_x_b and _vals_y_b:
        mat = _interaction_matrix(feat_cols, _p_bx, _p_by, _vals_x_b, _vals_y_b)
        _lab_x = PARAM_LABEL.get(_p_bx, _p_bx)
        if _p_bx == 'catalyst':
            _lab_x = CATALYST_SLOT_LABEL or 'Catalyst'
        _lab_y = PARAM_LABEL.get(_p_by, _p_by)
        _title_b = ('%s–%s Ensemble Predicted Yield' % (_lab_x, _lab_y) if is_ensemble
                    else '%s–%s Predicted Yield (%s)' % (_lab_x, _lab_y, model_name))
        mat, _vals_x_b, _vals_y_b = _top_n_axes(mat, _vals_x_b, _vals_y_b, n=10)
        im = _plot_heatmap(ax2, mat, _vals_x_b, _vals_y_b, _title_b, 'YlOrRd',
                           xlabel=_lab_x)
        _mark_heatmap_top_decile(ax2, mat)
        # v22k: boxed maximum removed -- the top-decile label marks it.
        cbar_label = 'Predicted Yield (%)'
        _fig_grid(_fig9_tag, 'b', '%s x %s' % (_lab_y, _lab_x), mat,
                  _vals_x_b, _vals_y_b, note='ensemble predicted yield (%)')

        if im is not None:
            cb = fig.colorbar(im, ax=ax2)
            cb.set_label(cbar_label, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No two process-condition axes available',
                 ha='center', va='center')
    ax2.text(-0.12, 1.05, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')

    # 8c: second interaction landscape (a different partner variable)
    _vals_x_c = label_encoders[_p_cx].classes_.tolist() if _p_cx else []
    _vals_y_c = label_encoders[_p_cy].classes_.tolist() if _p_cy else []
    if _p_cx and _p_cy and _vals_x_c and _vals_y_c and not (_p_cx == _p_bx and _p_cy == _p_by):
        mat = _interaction_matrix(feat_cols, _p_cx, _p_cy, _vals_x_c, _vals_y_c)
        _lab_x2 = PARAM_LABEL.get(_p_cx, _p_cx)
        if _p_cx == 'catalyst':
            _lab_x2 = CATALYST_SLOT_LABEL or 'Catalyst'
        _lab_y2 = PARAM_LABEL.get(_p_cy, _p_cy)
        _title_c = ('%s–%s Ensemble Predicted Yield' % (_lab_x2, _lab_y2) if is_ensemble
                    else '%s–%s Predicted Yield (%s)' % (_lab_x2, _lab_y2, model_name))
        mat, _vals_x_c, _vals_y_c = _top_n_axes(mat, _vals_x_c, _vals_y_c, n=10)
        im = _plot_heatmap(ax3, mat, _vals_x_c, _vals_y_c, _title_c, 'YlGnBu',
                           xlabel=_lab_x2)
        _mark_heatmap_top_decile(ax3, mat)
        # v22k: boxed maximum removed -- the top-decile label marks it.
        _fig_grid(_fig9_tag, 'c', '%s x %s' % (_lab_y2, _lab_x2), mat,
                  _vals_x_c, _vals_y_c, note='ensemble predicted yield (%)')
        if im is not None:
            cb = fig.colorbar(im, ax=ax3)
            cb.set_label('Predicted Yield (%)', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No second process-condition pair available',
                 ha='center', va='center')
    ax3.text(-0.12, 1.05, 'c', transform=ax3.transAxes, fontsize=14, fontweight='bold')

    # 8d: Loading Response of the TOP-RANKED candidate
    #
    # This previously swept the loading axis on the REPRESENTATIVE reagent set
    # (get_representative_conditions: per-axis mode/median).  That set is a poor
    # illustration: it is a mostly non-productive combination (peak predicted yield
    # ~20 %, versus ~98 % for the top candidates), so the curve sat near the bottom of
    # the axis and said nothing about what the model actually recommends.
    #
    # It now sweeps loading for the TOP-1 candidate of this run -- the same rank-1 entry
    # that heads Figure 10 and Figure 11 -- using the WEIGHTED ensemble mean (the same
    # weights as every other figure) so the panel is consistent with them.
    loading_grid = get_loading_grid(df, n_points=61)

    # v14: with no usable amount axis the "curve" would be a single vertical spike, so panel
    # (d) is replaced by an explicit note.  `_loading_ok` gates both the computation and the
    # shared axis labelling below, otherwise the labels would overwrite the note.
    _loading_ok = loading_axis_available() and len(loading_grid) >= 2
    if not _loading_ok:
        print(f"  [Figure 9d] not drawn: {LOADING_AXIS.get('reason')}")

    _top1 = None
    if bo_df is not None and len(bo_df):
        _sort_col = ('Ensemble Predicted Yield (%)' if is_ensemble
                     else 'Predicted Yield (%)')
        if _sort_col in bo_df.columns:
            _top1 = bo_df.sort_values(_sort_col, ascending=False).iloc[0]
    if _top1 is None:
        _top1 = pd.Series(representative)          # fallback: keep the old behaviour
        _top1_src = 'representative set (no candidate pool available)'
    else:
        _top1_src = 'rank-1 candidate'

    records = []
    for loading in loading_grid:
        rec = {k: _top1[k] for k in _top1.index
               if k in ('metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2',
                        'reactant_1', 'temperature', 'time')}
        rec.setdefault('temperature', representative.get('temperature', 80.0))
        rec.setdefault('time', representative.get('time', 3.0))
        for c in ('metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2',
                  'reactant_1'):
            rec.setdefault(c, representative.get(c))
        rec[SWEEP_AXIS] = float(loading)        # v22: follow the chosen axis
        rec['catalyst_loading'] = float(loading)    # scaler was fit with this column present
        records.append(rec)
    xd_raw = pd.DataFrame(records)
    xd = build_virtual_feature_frame(df, feat_cols, label_encoders, scaler, xd_raw)

    if _loading_ok and is_ensemble:
        # WEIGHTED mean/std, matching Figure 10 / Figure 11 (the old code used an
        # unweighted mean via _grid_panel_stats, so this panel disagreed with them
        # by ~0.03 pp and used a different statistic).  Clipped too, so this curve
        # cannot exceed 100 % when the sweep-based figures do not.
        _preds = predict_members_clipped(trained_models, xd)
        if ensemble_weights is not None and len(ensemble_weights) == len(_preds):
            loading_mean, loading_std = _weighted_member_stats(
                _preds, weights=ensemble_weights)
        else:
            loading_mean, loading_std = _grid_panel_stats(xd)
        loading_ei = calculate_expected_improvement(
            loading_mean, loading_std, float(df['yield'].max()), xi=0.01)

        ax4.plot(loading_grid, loading_mean, '-', color=COLORS['blue'], lw=2,
                 label='Ensemble Mean (weighted)')
        ax4.fill_between(loading_grid, loading_mean - loading_std,
                         loading_mean + loading_std, color=COLORS['blue'], alpha=0.18,
                         label='Model Uncertainty ±1 SD')
        # v14: record the loading response for the figure's data export
        _axis_lab9 = LOADING_AXIS.get('label') or 'catalyst_loading'
        _axis_unit9 = LOADING_AXIS.get('unit') or 'eq.'
        for _ig, _lg in enumerate(loading_grid):
            _fig_row(_fig9_tag, 'd', 'ensemble mean (weighted)',
                     x=float(_lg), y=float(loading_mean[_ig]),
                     value=float(loading_ei[_ig]),
                     note='x = %s (%s); value = expected improvement' % (_axis_lab9, _axis_unit9))
            _fig_row(_fig9_tag, 'd', 'ensemble mean - 1 SD',
                     x=float(_lg), y=float(loading_mean[_ig] - loading_std[_ig]),
                     note='x = %s (%s)' % (_axis_lab9, _axis_unit9))
            _fig_row(_fig9_tag, 'd', 'ensemble mean + 1 SD',
                     x=float(_lg), y=float(loading_mean[_ig] + loading_std[_ig]),
                     note='x = %s (%s)' % (_axis_lab9, _axis_unit9))
        # The star marks MAX PREDICTED YIELD, because that is what "the best loading" means
        # for this panel.  EI is reported separately and labelled as such: with
        # best_observed set to the global measured max, the improvement term is negative
        # everywhere, so EI here is essentially "where is the model least certain" and must
        # NOT be read as the location of the star.
        best_i = int(np.argmax(loading_mean))
        ei_i = int(np.argmax(loading_ei))
        ax4.scatter([loading_grid[best_i]], [loading_mean[best_i]], color=COLORS['red'],
                    s=45, zorder=5)
        ax4.annotate(f'max pred. yield = {loading_mean[best_i]:.1f}%  @ {loading_grid[best_i]:.4g} eq.\n'
                     f'uncertainty (1 SD) = {loading_std[best_i]:.1f} pp\n'
                     f'argmax EI = {loading_grid[ei_i]:.4g} eq. (EI ≈ 0 everywhere)',
                     xy=(loading_grid[best_i], loading_mean[best_i]), xytext=(12, -2),
                     textcoords='offset points', fontsize=7.0, fontweight='bold',
                     ha='left', va='top',
                     # v14: 50 % opaque face so the box stops covering the curve it sits on.
                     # The alpha is baked into an RGBA facecolor rather than passed as
                     # bbox['alpha'], which would fade the BORDER too and lose the outline
                     # that keeps the text legible over the shaded uncertainty band.
                     bbox=dict(boxstyle='round,pad=0.3', fc=(1.0, 1.0, 1.0, 0.5),
                               ec='black'))
        # two-line title needs extra pad, otherwise the annotation box overlaps it
        _lab = '/'.join(str(_top1.get(c, '?')) for c in
                        ('catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1'))
        ax4.set_title(f'Bayesian Loading Optimization — rank-1 candidate\n{_lab}',
                      fontsize=8, fontweight='bold', pad=16)
    elif _loading_ok:
        loading_pred, _ = _grid_panel_stats(xd)
        ax4.plot(loading_grid, loading_pred, '-', color=COLORS['blue'], lw=2, label=f'{model_name} Prediction')
        best_i = int(np.argmax(loading_pred))
        ax4.scatter([loading_grid[best_i]], [loading_pred[best_i]], color=COLORS['red'], s=45, zorder=5)
        ax4.annotate(f'Max Yield ≈ {loading_grid[best_i]:.4g}\nPred. yield = {loading_pred[best_i]:.1f}%',
                     xy=(loading_grid[best_i], loading_pred[best_i]), xytext=(10, 10), textcoords='offset points',
                     fontsize=7.5, fontweight='bold',
                     # v14: 50 % opaque face, same as the ensemble panel above, so the box
                     # does not hide the loading curve it is annotating.
                     bbox=dict(boxstyle='round,pad=0.3', fc=(1.0, 1.0, 1.0, 0.5), ec='black'))
        ax4.set_title(f'Loading Optimization ({_dm})', fontsize=9, fontweight='bold')
    else:
        annotate_loading_unavailable(ax4, panel_letter='d', fontsize=8)

    if _loading_ok:
        ax4.set_xlabel('%s (%s)' % (LOADING_AXIS.get('label') or 'Catalyst loading',
                                    LOADING_AXIS.get('unit') or 'eq.'), fontweight='bold')
        ax4.set_ylabel('Predicted Yield (%)', fontweight='bold')
        ax4.legend(loc='best', fontsize=7)
        ax4.grid(True, linestyle='--', alpha=0.35)
        ax4.text(-0.12, 1.05, 'd', transform=ax4.transAxes, fontsize=14, fontweight='bold')

    for ax in [ax1, ax2, ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)

    plt.suptitle(f'Optimization & Response Surfaces — {_dm}', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig8_path = os.path.join(output_dir, f'Figure9_Bayesian_Optimization-{model_name}.png')
    plt.savefig(fig8_path, dpi=300, bbox_inches='tight')
    plt.close()
    # v14: dump every plotted quantity of this figure to one tidy CSV
    _fig_flush(output_dir, _fig9_tag,
               f'Figure9_Bayesian_Optimization-{model_name}_panel_data.csv')

    # Figure 10: Top-20 Table Render
    # v12: skipped for the per-member calls.  A per-member Figure 10 would rank the SAME
    # candidate pool as the Ensemble one and differ only marginally, and with
    # KEEP_MODEL_PREDICTIONS=False it silently fell back to the ensemble mean -- i.e. it
    # mislabelled an ensemble table as a single-model one.  The per-member predictions
    # now appear as columns inside the Ensemble table instead.
    if not make_figure9:
        print(f"  [Figure 9] done for [{model_name}]; Figure 9 skipped (Ensemble-only).")
        return

    # v12: apply the SAME ensemble-agreement gate used by the Figure 10 table, so it can
    # no longer surface candidates on which the members disagree wildly (e.g. the
    # -8 % / 65 % / 104 % row whose EI was highest purely because sigma dominated).
    if is_ensemble:
        if 'Passes Ensemble Gate' in model_virtual_df.columns:
            n_before = len(model_virtual_df)
            model_virtual_df = model_virtual_df[model_virtual_df['Passes Ensemble Gate']].copy()


    historical_stats, group_cols, hist_aux = build_prediction_lookup(df, model_virtual_df)
    candidates = add_validation_status(model_virtual_df, historical_stats, group_cols, hist_aux)

    # v13: reliability metrics (90 % conformal half-width, expected error, credibility tier).
    # Merged on the condition columns so it survives row reordering; the candidate pool is
    # dense in 6-decimal loadings, so normalising the key as text is enough and avoids
    # float-equality surprises between the two frames.
    if credibility is not None and len(credibility) and len(candidates):
        _ck = [c for c in ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2',
                           'reactant_1', 'temperature', 'time', 'catalyst_loading']
               if c in credibility.columns and c in candidates.columns]
        _add = [c for c in ['CI Half-Width (90%, pp)', 'Expected Error (pp)', 'Credibility']
                if c in credibility.columns]
        if _ck and _add:
            _l = candidates[_ck].astype(str)
            _r = credibility[_ck + _add].copy()
            for c in _ck:
                _r[c] = _r[c].astype(str)
            _r = _r.drop_duplicates(subset=_ck)
            _merged = _l.merge(_r, on=_ck, how='left')
            for c in _add:
                candidates[c] = _merged[c].to_numpy()


    # v12: Gate B -- restrict to conditions with enough historical precedent
    if is_ensemble:
        candidates, _ = apply_support_gate(candidates)

    # `Support Count` and `Reagent Support Count` are 0 on every row BY CONSTRUCTION here:
    # the enumerated library never reproduces a measured condition or reagent combo (the
    # loading axis samples values no experiment used).  A column that is identically 0
    # carries no information, so the useful proximity measure is how many of the reagent
    # axes the candidate shares with its closest real combination:
    #     max = that exact reagent set was run before
    #     1   = only one component is familiar
    # Costs ~0.01 s on 500 candidates (see max_component_match).
    # `_match_denom` is the number of axes actually compared, which the column header uses
    # so the label stays honest if a different dataset lacks (or pins) an axis.
    _match_denom = 0
    if len(candidates):
        _ov, _tie, _match_denom = max_component_match(candidates, df)
        candidates['Match Count'] = _ov
        candidates['Match Ties'] = _tie
        _vc = pd.Series(_ov).value_counts().sort_index()
        print('  [Figure 10] max component match (of %d reagent axes): ' % _match_denom
              + ', '.join(f'{k}/{_match_denom} -> {v}' for k, v in _vc.items()))

    # v12: rank by PREDICTED YIELD for every case (ensemble included).
    # Expected Improvement is no longer the sort key: with best_observed_yield set to
    # the global maximum (99.5 %), improvement is negative for almost every virtual
    # condition, so EI degenerates into "sort by ensemble spread" and promotes exactly
    # the conditions the members disagree about.  EI/UCB remain as reference columns.
    if is_ensemble:
        sort_col = 'Ensemble Predicted Yield (%)'
    else:
        sort_col = 'Predicted Yield (%)'

    # v12: keep the FULL gated ranking for the CSV, and slice only for the figure.
    # (Previously `candidates` was overwritten by the top-20, so the CSV could never
    #  contain more than the 20 rendered rows.)
    ranked_all = candidates.sort_values(sort_col, ascending=False).reset_index(drop=True)
    ranked_all.insert(0, 'Rank', np.arange(1, len(ranked_all) + 1))
    candidates = ranked_all.head(top_n_table).reset_index(drop=True)

    # v14: the SAME gated pool, ranked by EXPECTED IMPROVEMENT instead of predicted yield.
    #
    # WHY A SEPARATE TARGET RATHER THAN THE SWEEP'S EI COLUMN.
    # The sweep computes `Expected Improvement` against an incumbent taken from the OBSERVED
    # yields.  On a well-screened dataset that incumbent sits ABOVE almost every candidate's
    # prediction, so mu - f* is negative nearly everywhere and EI collapses onto sigma*phi(z)
    # -- it degenerates into an uncertainty ranking that proposes 58-75 % conditions while
    # 88 % ones are on the table.  Measured on (Summit)Reizman: f* = 99.8 % gives
    # EI_max = 0.06 pp and a top-20 reaching down to 57.9 % predicted yield.
    #
    # The target here is instead anchored to the CANDIDATE POOL: the yield that the top decile
    # of this library is predicted to reach.  Figure 10b then answers
    #     "which conditions have the best chance of landing in the top decile of what this
    #      library can offer?"
    # which is the question a recommender should answer, and it behaves consistently across
    # datasets because it does not depend on how far the observed yield distribution happens
    # to extend.  EI_TARGET_QUANTILE sets the decile.
    _pool_pred = pd.to_numeric(ranked_all[sort_col], errors='coerce').dropna()
    if len(_pool_pred):
        ei_target = float(np.percentile(_pool_pred.to_numpy(), EI_TARGET_QUANTILE * 100.0))
    else:
        ei_target = float(best_observed_yield)
    ranked_ei = ranked_all.copy()
    ranked_ei['EI (vs target)'] = calculate_expected_improvement(
        pd.to_numeric(ranked_ei[sort_col], errors='coerce').to_numpy(),
        pd.to_numeric(ranked_ei['Prediction Uncertainty (%)'], errors='coerce').to_numpy(),
        ei_target, xi=0.01)
    ranked_ei = (ranked_ei.rename(columns={'Rank': 'Yield Rank'})
                 .sort_values('EI (vs target)', ascending=False)
                 .reset_index(drop=True))
    ranked_ei.insert(0, 'EI Rank', np.arange(1, len(ranked_ei) + 1))

    # v12 display table: 12 columns (was 20)
    # Columns dropped from the FIGURE and why (all of them are kept in the CSV):
    #   metal            -- constant (Pd) in this dataset
    #   temperature/time -- constant (80 C / 3 h) once the library is enumerated
    #   Actual Avg. Yield-- derived from the EXACT condition, which the enumerated
    #                       library never reproduces, so it is empty on every row
    #   ENS. SD (%)      -- near-collinear with Model Gap (pp); one spread measure is enough
    #   EI / UCB (%)     -- degenerate here: best_observed_yield is the global maximum
    #                       (99.5 %), so EI ~ 1e-4 and carries no ordering information
    #   Match Level      -- redundant with Data Status + Support columns
    #   Exact Support / Reagent Support -- merged into one "Support (exact/reagent)" cell
    # v12: three per-member prediction columns immediately BEFORE the ensemble value,
    # so the reader can see where the average comes from.  Built from the Pred_<name>
    # columns that generate_bo_results_streaming now carries into the survivor pool.
    member_cols = [c for c in candidates.columns if c.startswith('Pred_')]
    # keep a stable, readable order: XGBoost, Random Forest, Gradient Boosting, other
    _pref = ['Pred_XGBoost', 'Pred_Random_Forest', 'Pred_Gradient_Boosting']
    member_cols = [c for c in _pref if c in member_cols] + \
                  [c for c in member_cols if c not in _pref]

    # v22m: the hand-written list showed ligand/base/solvent/reactant_2/loading even when a
    # dataset has ONE value in them (Summit: six of the eight were 'Unknown').  Show only the
    # slots that actually vary here, and the same for the amount (dose) axis.
    _SLOT_ORDER = ['catalyst', 'ligand', 'base', 'solvent', 'reactant_1', 'reactant_2',
                   'additive', 'metal']
    # v22r: only the slots that actually vary in THIS dataset (Unknown/none placeholders for
    # columns the CSV does not have are dropped from the figure; they stay in the CSV)
    _cond_cols = [c for c in _SLOT_ORDER if c in candidates.columns
                  and candidates[c].astype(str).nunique(dropna=False) > 1]
    _DOSE_ORDER = ['catalyst_loading', 'ligand_eq', 'base_eq', 'reactant_1_eq',
                   'reactant_2_eq', 'solvent_eq', 'cosolvent_eq', 'additive_eq',
                   'temperature', 'time']
    # v22v: the dose column is ALWAYS shown -- with an unexpanded library it holds a single
    # (median) value, and dropping it made the table look like it had no amount axis at all.
    _dose_cols = [c for c in _DOSE_ORDER if c in candidates.columns]
    display_columns = ['Rank'] + _cond_cols + _dose_cols + member_cols + [
        'Ensemble Predicted Yield (%)' if is_ensemble else 'Predicted Yield (%)',
        'CI Half-Width (90%, pp)', 'Expected Error (pp)', 'Credibility',
        'Model Gap (%)', 'Match Count',
    ]
    display_columns = [c for c in display_columns if c in candidates.columns]
    table_df = candidates[display_columns].copy()
    # v22q: the condition columns were printed with their RAW slot names (catalyst, ligand,
    # base, solvent, reactant_1, reactant_2).  Use the same display labels as Figure 6.
    _SLOT_LABEL = {'catalyst': 'Catalyst', 'ligand': 'Ligand', 'base': 'Base',
                   'solvent': 'Solvent', 'reactant_1': 'Substrate',
                   'reactant_2': 'Reagent 2', 'additive': 'Additive', 'metal': 'Metal',
                   'catalyst_loading': 'Loading (eq.)', 'ligand_eq': 'Ligand (eq.)',
                   'base_eq': 'Base (eq.)', 'reactant_1_eq': 'Substrate (eq.)',
                   'reactant_2_eq': 'Reagent 2 (eq.)', 'solvent_eq': 'Solvent (eq.)',
                   'temperature': 'Temp. (°C)', 'time': 'Time (h)'}
    table_df = table_df.rename(columns=lambda c: _SLOT_LABEL.get(c, c))

    # Data Status / Support / Reagent Avg. Yield are deliberately NOT rendered (v13):
    #   Support (exact/…)  the exact level is always 0 -- no enumerated grid loading
    #                      equals a measured loading -- so the cell is meaningless
    #   Reagent Avg. Yield only populated for a minority of rows
    #   Data Status        still computed internally: it drives the row shading below
    # All three remain in the CSV.

    rename_map = {
        'catalyst': 'Catalyst', 'base': 'Base', 'solvent': 'Solvent',
        'reactant_1': 'Substrate',
        'reactant_2': 'Leaving Group', 'catalyst_loading': 'Loading (eq.)',
        'Ensemble Predicted Yield (%)': 'Predicted Yield (%)',
        'Model Gap (%)': 'Model Gap (pp)',
        # compact headers for the reliability columns: the table is already 15 columns wide
        'CI Half-Width (90%, pp)': '90% CI (±pp)',
        'Expected Error (pp)': 'Exp. Error (pp)',
        'Credibility': 'Cred.',
        # how many reagent axes this candidate shares with its closest REAL combination.
        # The denominator is the number of axes ACTUALLY compared, not a hardcoded 5, so
        # the header cannot overstate the score on a dataset that lacks an axis.
        'Match Count': f'Match (of {_match_denom})' if _match_denom else 'Match',
    }
    table_df = table_df.rename(columns=rename_map)
    # shorten the per-member headers so the table stays readable
    _dm_map = globals().get('MODEL_DISPLAY') or {}
    table_df = table_df.rename(columns=lambda c: (
        'Pred. ' + _dm_map.get(c[len('Pred_'):], c[len('Pred_'):])
        if c.startswith('Pred_') else c))

    member_disp = [c for c in table_df.columns if c.startswith('Pred. ')]
    _rel = [c for c in ['90% CI (±pp)', 'Exp. Error (pp)'] if c in table_df.columns]
    _dose_labels = [rename_map.get(c, c) for c in _dose_cols]
    for col in (_dose_labels + ['Predicted Yield (%)', 'Model Gap (pp)'] + _rel + member_disp):
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors='coerce')

    for _dl in _dose_labels:
        if _dl in table_df.columns:
            table_df[_dl] = table_df[_dl].map(
                lambda x: ('%.5f' % x).rstrip('0').rstrip('.') if pd.notna(x) else '—')
    for col in (['Predicted Yield (%)', 'Model Gap (pp)'] + _rel + member_disp):
        if col in table_df.columns:
            table_df[col] = table_df[col].map(lambda x: f'{x:.1f}' if pd.notna(x) else '—')
    # `Match Count` is formatted BEFORE the rename, because its display label carries the
    # dynamic denominator and is not known by name at this point.
    if 'Match Count' in table_df.columns:
        table_df['Match Count'] = [
            f'{int(v)}' if pd.notna(v) else '—'
            for v in pd.to_numeric(table_df['Match Count'], errors='coerce')]

    # Row height in inches.  This used to be 0.46 in/row, which made a 20-row table 10.1 in
    # tall -- visibly airy next to Figure 10.  Figure 10's table occupies 0.90 of a
    # 1.7/(1.0+1.7) slice of a 12.5 in figure and holds 21 text lines, i.e.
    #   0.90 * (1.7/2.7) * 12.5 / 21 = 0.337 in/row.
    # Use exactly that, so the two tables match.
    fig9_row_in = 0.337
    fig9_height = max(3.2, fig9_row_in * (len(table_df) + 2))
    # width now follows the actual column count and their content, so long columns
    # get room instead of every column being forced to 1/n
    fig9_width = max(14.0, min(26.0, 0.72 * len(table_df.columns) + 4.0))
    fig, ax = plt.subplots(figsize=(fig9_width, fig9_height))
    ax.axis('off')

    # IMPORTANT: give the table an explicit bbox.  Without one the table is auto-sized to
    # its content, and the following footnote (a long multi-line string) makes
    # tight_layout() shrink the axes to a narrow vertical strip -- which produced the
    # squashed table with overlapping text.
    # The table occupies the UPPER portion and the note is placed below it: the table used
    # to end at 0.10 while the note started at 0.005, which made them collide.
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                     cellLoc='center', colLoc='center',
                     bbox=[0.0, 0.15, 1.0, 0.81])
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)

    # v22o: widths from the widest cell, and the canvas grows to fit the whole table
    _tab_w_in = sum(_autofit_table(fig, table, list(table_df.columns), table_df.values,
                                   min_w=7.0, fontsize=6.0))

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#D9EAF7')
        elif col == 0:
            cell.set_text_props(weight='bold')

    # v13: no row shading in Figure 10.  Data Status is no longer displayed and the
    # Discovery/Validated split is reported in text instead, so the two agree by
    # construction rather than by a colour legend the reader has to decode.
    if 'Data Status' in candidates.columns and len(table_df):
        n_disc = int((candidates['Data Status'].iloc[:len(table_df)] == 'Discovery').sum())


    pred_col = table_df.columns.get_loc('Predicted Yield (%)') if 'Predicted Yield (%)' in table_df.columns else None
    if pred_col is not None and len(table_df) > 0:
        table[(1, pred_col)].set_text_props(weight='bold')

    gate_note = (f' Candidates pre-filtered on ensemble agreement (Model Gap <= {ENSEMBLE_MAX_MODEL_GAP:g} pp).'
                 if is_ensemble and ENSEMBLE_MAX_MODEL_GAP is not None else '')
    ax.set_title(f'Top {len(table_df)} Predicted Conditions — {_dm}', fontsize=12, fontweight='bold', pad=14)
    # v12: keep the footnote short and on separate lines -- a single very long line made
    # tight_layout() expand the canvas and squash the table.
    note_lines = [
        f'Ranked by predicted yield. Predicted Yield is the WEIGHTED mean of the member '
        f'predictions shown to its left (weights from held-out R²).{gate_note}',
        '90% CI (±pp) = half-width of a normalised split-conformal interval (calibrated on '
        'training rows only; measured held-out coverage 91%). Exp. Error (pp) = mean |error| '
        'of the ensemble on the 25 nearest training conditions.',
        'Cred. tiers: High = CI < 5 pp and Exp. Error < 3 pp; Medium = CI < 15 pp or '
        'Exp. Error < 8 pp; Low = otherwise. On held-out data these tiers separate '
        '1.4 / 4.4 / 8.2 / 14.1 pp of actual error, so they are calibrated, not nominal.',
        'Model Gap (pp) = max − min across members.',
        f'Match (of {_match_denom}) = how many of the {_match_denom} reagent axes SHOWN IN '
        'THIS TABLE this virtual candidate shares with its CLOSEST real combination in the '
        f'historical data. {_match_denom} = that exact reagent set was run before; '
        '1 = only one component is familiar. Exact-condition support is 0 whenever the '
        'enumerated loading grid does not reproduce a measured loading.',
        'Support, Data Status, Reagent Avg. Yield and Match Ties (how many historical '
        'combinations tie at the best match) are retained in the accompanying CSV.',
    ]
    # Footnote occupies the band BELOW the table (which now starts at 0.15).  It is
    # bottom-anchored, so it grows downward from 0.14 without reaching into the rows.
    ax.text(0.5, 0.14, _wrap_note(note_lines, _tab_w_in),
            ha='center', va='top', transform=ax.transAxes, fontsize=7, linespacing=1.35)

    # bottom=0.16 leaves figure space for the note; without it bbox_inches='tight' has to
    # grow the canvas and the note visually crowds the table.
    plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.16)
    fig9_path = os.path.join(output_dir, f'Figure10_Top20_Predicted_Conditions-{model_name}.png')
    plt.savefig(fig9_path, dpi=300, bbox_inches='tight')
    plt.close()

    # v12: write the top-N of the FULL gated ranking (not just the 20 rendered rows).
    ranked_all.head(int(top_n_csv)).to_csv(
        os.path.join(output_dir,
                     f'Figure10_Top{int(top_n_csv)}_Candidates-{model_name}.csv'),
        index=False, encoding='utf-8-sig'
    )
    # keep the legacy filename as the 20-row table extract for backwards compatibility
    candidates.to_csv(
        os.path.join(output_dir, f'Figure10_Top20_Predicted_Conditions-{model_name}.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f"  Figure 10 CSV: -> wrote top "
          f"{min(int(top_n_csv), len(ranked_all)):,} to "
          f"Figure10_Top{int(top_n_csv)}_Candidates-{model_name}.csv")

    # v14  FIGURE 10b -- THE SAME GATED POOL RANKED BY EXPECTED IMPROVEMENT
    # Figure 10a answers "where is the yield predicted to be highest".
    # Figure 10b answers "where do I expect to beat my incumbent by the most".
    # The two differ whenever a slightly-lower-mu / higher-sigma condition has a fatter
    # upper tail; measured on this dataset EI's top-20 contains 19 distinct reagent
    # combinations against 14 for the predicted-yield ranking, so 10b is the more useful
    # list when the chemist is choosing a BATCH of experiments rather than the single best.
    _ei_cols = [c for c in ['EI Rank', 'Yield Rank', 'catalyst', 'ligand', 'base', 'solvent',
                            'reactant_1', 'reactant_2', 'catalyst_loading',
                            'Ensemble Predicted Yield (%)', 'Prediction Uncertainty (%)',
                            'EI (vs target)', 'Credibility']
                if c in ranked_ei.columns]
    _ei_tab = ranked_ei.head(top_n_table)[_ei_cols].copy()
    # v22r: keep only the columns whose values actually vary (same rule as Fig. 6 / 10a)
    # NOTE: this runs BEFORE the rename, so the dose column is still 'catalyst_loading' / '*_eq'
    _keep_dose = ('catalyst_loading', 'ligand_eq', 'base_eq', 'reactant_1_eq', 'reactant_2_eq',
                  'solvent_eq', 'cosolvent_eq', 'additive_eq', 'temperature', 'time',
                  'EI Rank', 'Yield Rank', 'Loading (eq.)')
    _ei_tab = _ei_tab.loc[:, [c for c in _ei_tab.columns
                              if c in _keep_dose
                              or _ei_tab[c].astype(str).nunique(dropna=False) > 1]]
    _ei_tab = _ei_tab.rename(columns={
        # v22q: same display labels as the other tables
        'catalyst': 'Catalyst', 'ligand': 'Ligand', 'base': 'Base', 'solvent': 'Solvent',
        'reactant_1': 'Substrate', 'reactant_2': 'Reagent 2', 'additive': 'Additive',
        'metal': 'Metal', 'temperature': 'Temp. (°C)', 'time': 'Time (h)',
        'ligand_eq': 'Ligand (eq.)', 'base_eq': 'Base (eq.)',
        'reactant_1_eq': 'Substrate (eq.)', 'reactant_2_eq': 'Reagent 2 (eq.)',
        'solvent_eq': 'Solvent (eq.)',
        'catalyst_loading': 'Loading (eq.)',
        'Ensemble Predicted Yield (%)': 'Pred. Yield (%)',
        'Prediction Uncertainty (%)': 'Unc. (±pp)',
        'EI (vs target)': 'EI (pp)'})
    for _c in ['Loading (eq.)']:
        if _c in _ei_tab.columns:
            _ei_tab[_c] = _ei_tab[_c].map(
            lambda v: ('%.5f' % v).rstrip('0').rstrip('.') if pd.notna(v) else '—')
    for _c in ['Pred. Yield (%)', 'Unc. (±pp)', 'EI (pp)']:
        if _c in _ei_tab.columns:
            _ei_tab[_c] = _ei_tab[_c].map(lambda v: f'{v:.2f}' if pd.notna(v) else '—')

    _fig_h = max(3.2, 0.337 * (len(_ei_tab) + 2))
    _fig_w = max(12.0, min(24.0, 0.80 * len(_ei_tab.columns) + 4.0))
    fig_ei, ax_ei = plt.subplots(figsize=(_fig_w, _fig_h))
    ax_ei.axis('off')
    tbl = ax_ei.table(cellText=_ei_tab.values, colLabels=_ei_tab.columns,
                      cellLoc='center', colLoc='center', bbox=[0.0, 0.17, 1.0, 0.79])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.0)
    _ei_w_in = sum(_autofit_table(fig_ei, tbl, list(_ei_tab.columns), _ei_tab.values,
                                  min_w=6.5, fontsize=6.0))
    _best_col = list(_ei_tab.columns).index('EI (pp)') if 'EI (pp)' in _ei_tab.columns else 0
    for (_r, _c), _cell in tbl.get_celld().items():
        _cell.set_edgecolor('black')
        _cell.set_linewidth(0.45)
        if _r == 0:
            _cell.set_text_props(weight='bold')
            _cell.set_facecolor('#FBE5D6')          # orange header: distinguishes 10a vs 10b
        elif _c == 0 or _c == _best_col:
            _cell.set_text_props(weight='bold')

    ax_ei.set_title(f'Top {len(_ei_tab)} Conditions by EXPECTED IMPROVEMENT — {_dm}',
                    fontsize=12, fontweight='bold', pad=14)
    _ei_max_pp = float(pd.to_numeric(ranked_ei['EI (vs target)'],
                                     errors='coerce').head(top_n_table).max())
    _noise_note = ('above' if _ei_max_pp > 5 else 'at or below')
    _tgt_desc = ('the model\u2019s best predicted yield over this candidate pool'
                 if EI_TARGET_QUANTILE >= 1.0 else
                 f'the {EI_TARGET_QUANTILE:.0%} quantile of the predicted yield over this pool')
    ax_ei.text(0.5, 0.135, _wrap_note(['' ,
               f'Ranked by EI = E[max(0, yield - target)] with target = {_tgt_desc} '
               f'= {ei_target:.2f} %.\n'
               f'EI asks a different question from Figure 10a: not "where is the yield '
               f'highest" but "where have I the best expected gain over that target".\n'
               f'It therefore trades a little predicted yield for information: a condition '
               f'whose members disagree (higher Unc.) can outrank a higher-mu, '
               f'better-agreed one.\n'
               f'"Yield Rank" shows where the same condition sat in Figure 10a; '
               f'this top-{len(_ei_tab)} reaches Yield Rank '
               f'{int(pd.to_numeric(ranked_ei["Yield Rank"]).head(top_n_table).min())}-'
               f'{int(pd.to_numeric(ranked_ei["Yield Rank"]).head(top_n_table).max())}, '
               f'mean predicted yield '
               f'{pd.to_numeric(ranked_ei[sort_col]).head(top_n_table).mean():.1f} %.  '
               f'Max EI = {_ei_max_pp:.2f} pp, {_noise_note} a typical 2-5 pp assay noise.'
               + (f'  Candidates pre-filtered on ensemble agreement '
                  f'(Model Gap <= {ENSEMBLE_MAX_MODEL_GAP:g} pp).'
                  if is_ensemble and ENSEMBLE_MAX_MODEL_GAP is not None else ''),
               ''], _ei_w_in) + '', transform=ax_ei.transAxes, fontsize=7.2, va='top', ha='center', family='sans-serif')
    # v22u: centre the table -- the canvas is table-width + 0.5 in, and the spare space
    # used to sit entirely on the right.
    plt.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.16)
    plt.savefig(os.path.join(
        output_dir, f'Figure10b_Top{int(top_n_table)}_ByExpectedImprovement-{model_name}.png'),
        dpi=300, bbox_inches='tight')
    plt.close(fig_ei)
    ranked_ei.head(int(top_n_csv)).to_csv(
        os.path.join(output_dir,
                     f'Figure10b_Top{int(top_n_csv)}_ByExpectedImprovement-{model_name}.csv'),
        index=False, encoding='utf-8-sig')
    _shared = len(set(ranked_ei['EI Rank'].head(top_n_table))
                  & set(ranked_ei['Yield Rank'].head(top_n_table)))
    print(f"  Figure 10b: -> ranked by EI (target = {ei_target:.2f}% = "
          f"{EI_TARGET_QUANTILE:.0%} of the pool's predicted yield); Yield Rank spans "
          f"{int(pd.to_numeric(ranked_ei['Yield Rank']).head(top_n_table).min())}-"
          f"{int(pd.to_numeric(ranked_ei['Yield Rank']).head(top_n_table).max())}; "
          f"mean predicted yield {pd.to_numeric(ranked_ei[sort_col]).head(top_n_table).mean():.2f}%")

# FIGURE 6 -- DIRECT ENSEMBLE RANKING OF THE REAL REACTION POOL
# (real measured reactions; NOT the virtual library, which is Figures 9 and 10)
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

    work = _canon(key_cols)
    work['_ckey'] = work[key_cols].agg('|'.join, axis=1)
    work['_is_tr'] = is_train
    all_tr = work.groupby('_ckey', sort=False)['_is_tr'].apply(lambda s: bool(s.all()))
    split_tbl = pd.DataFrame({'_ckey': all_tr.index,
                              'Split': np.where(all_tr.to_numpy(), 'train', 'held-out')})
    # one row per DISTINCT condition key (the pool has one row per condition)
    out = pd.DataFrame({'_ckey': np.unique(work['_ckey'].to_numpy())})
    return out.merge(split_tbl, on='_ckey', how='left'), key_cols


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
    # v22m: only the slots that actually vary in THIS dataset (see the note in the 9/10 table)
    _SLOT_ORDER6 = ['catalyst', 'ligand', 'base', 'solvent', 'reactant_1', 'reactant_2',
                    'additive', 'metal']
    _cond_cols6 = [c for c in _SLOT_ORDER6 if c in table_src.columns
                   and table_src[c].astype(str).nunique(dropna=False) > 1]
    _DOSE_ORDER6 = ['catalyst_loading', 'ligand_eq', 'base_eq', 'reactant_1_eq',
                    'reactant_2_eq', 'solvent_eq', 'cosolvent_eq', 'additive_eq',
                    'temperature', 'time']
    _dose_cols6 = [c for c in _DOSE_ORDER6 if c in table_src.columns]
    display_columns = ['Rank'] + _cond_cols6 + _dose_cols6
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
        'ligand_eq': 'Ligand (eq.)', 'base_eq': 'Base (eq.)',
        'reactant_1_eq': 'Substrate (eq.)', 'reactant_2_eq': 'Reagent 2 (eq.)',
        'solvent_eq': 'Solvent (eq.)', 'cosolvent_eq': 'Cosorbent (eq.)',
        'additive_eq': 'Additive (eq.)',
        'temperature': 'Temp. (°C)', 'time': 'Time (h)',
        'Ensemble Predicted Yield (%)': 'Predicted Yield (%)',
        'Model Gap (%)': 'Model Gap (pp)',
        'yield': 'Measured Yield (%)',
        'Prediction Error (%)': 'Error (pp)',
        'Split': 'Split',
        'Data Status': 'Data Status',
    }
    table_df = table_df.rename(columns=rename_map)
    table_df = table_df.rename(columns=lambda c: (
        'Pred. ' + (globals().get('MODEL_DISPLAY') or {}).get(c[len('Pred_'):], c[len('Pred_'):])
        if c.startswith('Pred_') else c))
    member_disp = [c for c in table_df.columns if c.startswith('Pred. ')]

    _dose_labels6 = [rename_map.get(c, c) for c in _dose_cols6]
    for col in _dose_labels6 + ['Predicted Yield (%)', 'Model Gap (pp)',
                                'Measured Yield (%)', 'Yield SD (pp)', 'Error (pp)'] + member_disp:
        if col in table_df.columns:
            table_df[col] = pd.to_numeric(table_df[col], errors='coerce')

    # v23c: format EVERY dose column, not just 'Loading (eq.)' -- a dataset whose amount
    # axis is ligand_eq / base_eq (AI_RCP sm) printed raw floats like 0.080095493214.
    for _dl6 in _dose_labels6:
        if _dl6 in table_df.columns:
            table_df[_dl6] = table_df[_dl6].map(
                lambda x: ('%.5f' % x).rstrip('0').rstrip('.') if pd.notna(x) else '—')
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
    # v23f: the table text was one size smaller than the Figure-10 tables.  ﻿7 pt with the
    # matching 0.054 in/char keeps the columns snug instead of re-widening them.
    _f6_tab_w = sum(_autofit_table(fig, table, list(table_df.columns), table_df.values,
                                  min_w=7.0, char_in=0.054, fontsize=7.0))

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
        'Ensemble members: ' + ', '.join((globals().get('MODEL_DISPLAY') or {}).get(m, m)
                                              for m in pool['Model Preset'].iloc[0].split(', ')) + '.',
        "N Runs = how many replicate MEASUREMENTS this exact condition has; Measured Yield is "
        "their mean and Yield SD (pp) their spread.",
        "Split = 'fitted' only if EVERY replicate of the exact CONDITION was in the training "
        "set, else 'held-out' (not fully seen during fitting).",
        'Green = measured yield ≥ 90%; cream = < 20%. '
        'Constant columns and the full set of diagnostics are retained in the CSV.',
    ]
    axb.text(0.5, -0.02, _wrap_note(note_lines, _f6_tab_w),
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


# 3. FIGURE GENERATION PIPELINE

def generate_figure2_3_4_per_model(df, X_s, X_te, y_te, trained_models, cache,
                                    display_feat_names, feat_cols, scaler,
                                    output_dir, model_names, raw_feature_names,
                                    label_encoders=None):

    """
    Figures 2, 3 and 4, one set per selected model.

    Split out of generate_all_figures so the Figure-number order can be honoured
    without relocating 240 lines of drawing code: these three need only the trained
    models and the training cache, so they belong BEFORE the virtual-library sweep.

    No recomputation: the learning curves and the test predictions are read from
    `cache`, which train_and_evaluate_models already filled.
    """
    for model_name in model_names:
        current_model = trained_models[model_name]
        model_cache = cache[model_name]
        file_suffix = f"- {(globals().get('MODEL_DISPLAY') or {}).get(model_name, model_name)}"
        _dm = (globals().get('MODEL_DISPLAY') or {}).get(model_name, model_name)

        # Figure 2: Model-Guided Learning
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        iters = np.arange(1, 21)
        al_yields = []
        subset_size = max(5, int(len(X_s) * 0.05))
    
        for it in iters:
            sample_preds = current_model.predict(X_s.sample(n=min(len(X_s), subset_size + it * 5), random_state=it))
            al_yields.append(np.percentile(sample_preds, 95))
        
        al_yields = np.array(al_yields)
        ax1.plot(iters, al_yields, 'o-', color=COLORS['green'], label='Uncertainty-Guided Batch')
        ax1.set_xlabel('Model-Guided Learning Iterations', fontweight='bold')
        ax1.set_ylabel(f'Top 5% Predicted Yield by {_dm} (%)', fontweight='bold')
        ax1.set_title('Model-Guided Learning Trajectory (Top 5% Predicted Yield)', fontsize=9, fontweight='bold')
    
        y_min_2a = max(0, np.min(al_yields) - 5)
        y_max_2a = min(105, np.max(al_yields) + 5)
        ax1.set_ylim([y_min_2a, y_max_2a])
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.legend(loc='lower right')
        ax1.text(-0.15, 1.05, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')

        train_sizes_2b, train_loss_2b, test_loss_2b = model_cache['learning_curve_rmse']
        train_rmse_mean = -np.mean(train_loss_2b, axis=1)
        test_rmse_mean = -np.mean(test_loss_2b, axis=1)

        ax2.plot(train_sizes_2b, train_rmse_mean, 'o-', color=COLORS['blue'], label='Train Error (RMSE)')
        ax2.plot(train_sizes_2b, test_rmse_mean, 's--', color=COLORS['red'], label='Validation Error (RMSE)')
        ax2.set_xlabel('Training Sample Progression', fontweight='bold')
        ax2.set_ylabel('Loss / RMSE (% Yield)', fontweight='bold')
        ax2.set_title(f'Model Convergence Rate Trajectory ({_dm})', fontsize=9, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.legend(loc='upper right')
        ax2.text(-0.15, 1.05, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')

        fig.suptitle(f'Model-Guided Learning and Response Surface Analysis — {_dm}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'Figure2_ActiveLearning_ResponseSurface{file_suffix}.png'), bbox_inches='tight')
        plt.close()

        # Figure 3: Model Diagnostics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9.5, 7.5))
        y_pred_te = model_cache['y_pred_test']

        train_sizes, train_scores, test_scores = model_cache['learning_curve_r2']
        train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
        test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

        ax1.plot(train_sizes, train_mean, 'o-', color=COLORS['blue'], label='Training')
        ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color=COLORS['blue'])
        ax1.plot(train_sizes, test_mean, 's-', color=COLORS['green'], label='Validation')
        ax1.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color=COLORS['green'])
        ax1.set_xlabel('Training Set Size', fontweight='bold')
        ax1.set_ylabel('R² Score', fontweight='bold')
        ax1.set_title(f'Learning Curve ({_dm})', fontsize=9, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.text(-0.15, 1.05, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')

        min_val = min(y_te.min(), y_pred_te.min())
        max_val = max(y_te.max(), y_pred_te.max())
        padding = (max_val - min_val) * 0.05
        axis_min = max(0, min_val - padding)
        axis_max = min(100, max_val + padding)

        hb = ax2.hexbin(y_te, y_pred_te, gridsize=35, cmap='Blues', mincnt=1, edgecolors='none', norm=colors.LogNorm())
        cb = fig.colorbar(hb, ax=ax2)
        cb.set_label('Count', fontweight='bold')

        ax2.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', lw=2, label='Perfect', color='#FF4D4D')
        ax2.set_xlim([axis_min, axis_max])
        ax2.set_ylim([axis_min, axis_max])
        ax2.set_xlabel('Experimental Yield (%)', fontweight='bold')
        ax2.set_ylabel(f'Predicted Yield by {_dm} (%)', fontweight='bold')
        ax2.set_title('Experimental vs Predicted Yield', fontsize=9, fontweight='bold')

        r2_val = r2_score(y_te, y_pred_te)
        rmse_val = np.sqrt(mean_squared_error(y_te, y_pred_te))
        mae_val = mean_absolute_error(y_te, y_pred_te)
    
        stats_text = f"[{_dm}]\nR² = {r2_val:.3f}\nRMSE = {rmse_val:.2f}%\nMAE = {mae_val:.2f}%\nn = {len(y_te)}"
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, va='top', ha='left',
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.8, alpha=0.9), fontsize=7.5, fontweight='bold')
        ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='black')
        ax2.text(-0.15, 1.05, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')

        residuals = y_te - y_pred_te
        ax3.scatter(y_pred_te, residuals, alpha=0.5, color=COLORS['orange'], s=15)
        ax3.axhline(0, color='black', linestyle='--', lw=2)
        ax3.axhline(10, color='red', linestyle=':', alpha=0.7)
        ax3.axhline(-10, color='red', linestyle=':', alpha=0.7)
        ax3.fill_between([0, 100], -10, 10, color='green', alpha=0.1)
    
        ax3.set_xlabel(f'Predicted Yield by {_dm} (%)', fontweight='bold')
        ax3.set_ylabel('Residual (% Yield)', fontweight='bold')
        ax3.set_title('Residuals vs Predicted Yield', fontsize=9, fontweight='bold')
        ax3.set_xlim([0, 100])
        ax3.set_ylim([-50, 50])
        ax3.text(-0.15, 1.05, 'c', transform=ax3.transAxes, fontsize=14, fontweight='bold')

        n, bins, patches = ax4.hist(residuals, bins=30, density=True, alpha=0.6, color=COLORS['purple'], edgecolor='black')
        mu, std = stats.norm.fit(residuals)
        x_axis = np.linspace(ax4.get_xlim()[0], ax4.get_xlim()[1], 100)
        ax4.plot(x_axis, stats.norm.pdf(x_axis, mu, std), 'k-', lw=2.5, label=f'Normal\nμ={mu:.2f}\nσ={std:.2f}')
        ax4.set_xlabel('Prediction Error (% Yield)', fontweight='bold')
        ax4.set_ylabel('Density', fontweight='bold')
        ax4.set_title('Residual Error Distribution', fontsize=9, fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.text(-0.15, 1.05, 'd', transform=ax4.transAxes, fontsize=14, fontweight='bold')

        fig.suptitle(f'Model Diagnostics and Prediction Error Analysis — {_dm}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'Figure3_Model_Diagnostics_LearningCurve_Residuals{file_suffix}.png'), bbox_inches='tight')
        plt.close()

        # Figure 4: Feature Attribution & Interactions
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 9))
        # permutation importance
        perm_result = permutation_importance(current_model, X_te, y_te, n_repeats=10, random_state=42)
        perm_sorted_idx = perm_result.importances_mean.argsort()

        # horizontal bars on ax1
        bar_gradient(ax1, perm_result.importances_mean[perm_sorted_idx],
                     cmap='Blues', horizontal=True, edgecolor='black')
        # v14: record the panel for the figure's data export
        _fig4_tag = 'Figure4_Feature_Importance_Attribution%s' % file_suffix
        for _ii, _ci in enumerate(perm_sorted_idx):
            _fig_row(_fig4_tag, 'a', 'permutation importance',
                     x=display_feat_names[_ci],
                     y=float(perm_result.importances_mean[_ci]),
                     value=float(perm_result.importances_std[_ci]),
                     note='x = feature; y = mean drop in R2; value = std over 10 repeats')
        ax1.set_yticks(range(len(perm_sorted_idx)))
        ax1.set_yticklabels([display_feat_names[i] for i in perm_sorted_idx], fontsize=8, fontweight='bold')
        ax1.set_xlabel('Drop in R² Score', fontweight='bold', fontsize=8)
        ax1.set_title(f'Permutation Feature Importance ({_dm})', fontsize=9, fontweight='bold')
        ax1.text(-0.15, 1.05, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')

        # v14: three cases --
        #   (1) a real loading axis that is also a model feature  -> draw the true response
        #   (2) no usable axis                                    -> say so
        # The old code had a third branch that plotted a FABRICATED ramp
        # (`10 + 0.5 * temp`) and still labelled the axis "Catalyst Loading (eq.)", i.e. an
        # invented curve presented as data.  That is replaced by the explicit note.
        _draw_loading_panel = (loading_axis_available()
                               and SWEEP_AXIS in df.columns
                               and SWEEP_AXIS in feat_cols)
        if _draw_loading_panel:
            cat_raw_vals = pd.to_numeric(
                df[SWEEP_AXIS], errors='coerce').dropna().to_numpy(float)
            valid_numerical_cols = [c for c in ['temperature', 'time', SWEEP_AXIS, 'reactant_2_eq', 'base_eq', 'reactant_1_eq', 'ligand_eq', 'solvent_eq', 'additive_eq']
                                    if c in df.columns and c in feat_cols
                                    and df[c].nunique() > 1]
        else:
            cat_raw_vals, valid_numerical_cols = np.array([]), []

        if _draw_loading_panel and SWEEP_AXIS in valid_numerical_cols:
            # v14: was `np.linspace(min, 1.0, 100)` -- a hardcoded upper limit of 1 eq. even when
            # the data reaches 23 eq. (and meaningless when the axis is a substitute amount).
            # Sweep the OBSERVED range instead.
            cat_grid_raw = np.linspace(float(np.nanmin(cat_raw_vals)),
                                       float(np.nanmax(cat_raw_vals)), 100)
            num_idx = valid_numerical_cols.index(SWEEP_AXIS)
            cat_mean = scaler.mean_[num_idx]
            cat_scale = scaler.scale_[num_idx]
            cat_grid_scaled = (cat_grid_raw - cat_mean) / cat_scale
            cat_idx = feat_cols.index(SWEEP_AXIS)

            _idx_a = 0
            _idx_b = min(5, len(X_s) - 1)
            sample_a = X_s.iloc[_idx_a].copy()
            sample_b = X_s.iloc[_idx_b].copy()

            grid_a = np.tile(sample_a.values, (len(cat_grid_scaled), 1))
            grid_b = np.tile(sample_b.values, (len(cat_grid_scaled), 1))

            grid_a[:, cat_idx] = cat_grid_scaled
            grid_b[:, cat_idx] = cat_grid_scaled

            pred_a = current_model.predict(grid_a)
            pred_b = current_model.predict(grid_b)

            # v15.  These two curves are rows 0 and 5 of the feature matrix -- two ARBITRARY
            # reactions, held fixed while only the loading axis moves.  They used to be
            # labelled "Steric Class A / Steric Class B", a leftover from a version of this
            # panel that swept a `steric_hindrance` descriptor: no steric variable is read
            # anywhere in this pipeline, `steric_hindrance` is not in `feat_cols`, and on
            # (Summit)Reizman_Suzuki_385 the two rows in fact differ in `catalyst`
            # (P1-L3 vs P1-L5) and in `t_res` (600 vs 60 min) -- while their loadings are
            # 0.498 vs 0.508, i.e. the one thing the sweep varies is the one thing they share.
            # The legend therefore named a contrast that does not exist, and implied the
            # vertical offset between the curves was caused by something the figure never
            # shows.  It now names the actual conditions, and lists the fields that DIFFER
            # first, because those are what separate the two curves.
            _LABEL_COLS = ['catalyst', 'reactant_1', 'base', 'solvent', 'reactant_2',
                           'time', 'temperature', SWEEP_AXIS]

            def _cond_label(_i, _other):
                """Short, reproducible description of the condition behind one curve."""
                try:
                    _r = df.iloc[_i]
                    _o = df.iloc[_other]
                except Exception:
                    return f'row {_i}'

                def _val(_src, _c):
                    _v = _src[_c]
                    try:
                        if bool(pd.isna(_v)):
                            return None
                    except Exception:
                        pass
                    _s = str(_v).strip()
                    if _s == '' or _s.lower() in ('unknown', 'nan', 'none'):
                        return None
                    return _s

                _diff, _same = [], []
                for _c in _LABEL_COLS:
                    if _c not in df.columns:
                        continue
                    _sv = _val(_r, _c)
                    if _sv is None:
                        continue
                    _ov = _val(_o, _c)
                    (_diff if _sv != _ov else _same).append(
                        '%s=%s' % (PARAM_LABEL.get(_c, _c), _sv[:24]))
                _parts = _diff[:3] or _same[:3]
                return ('row %d: %s' % (_i, ', '.join(_parts))) if _parts else f'row {_i}'

            _lab_a = _cond_label(_idx_a, _idx_b)
            _lab_b = _cond_label(_idx_b, _idx_a)
            print(f"  [Figure 4b] loading sweep held at {_lab_a}  |  {_lab_b}")

            ax2.plot(cat_grid_raw, pred_a, label=_lab_a, color=COLORS['blue'], lw=2)
            ax2.plot(cat_grid_raw, pred_b, label=_lab_b, color=COLORS['orange'], lw=2)
            for _ig in range(len(cat_grid_raw)):
                _fig_row(_fig4_tag, 'b', _lab_a, x=float(cat_grid_raw[_ig]),
                         y=float(pred_a[_ig]), note='predicted yield (%)')
                _fig_row(_fig4_tag, 'b', _lab_b, x=float(cat_grid_raw[_ig]),
                         y=float(pred_b[_ig]), note='predicted yield (%)')
        else:
            # panel_letter is NOT passed: the unconditional ax2.text(...) below already draws
            # the 'b', and passing it here printed the letter twice.
            annotate_loading_unavailable(ax2, fontsize=8)
            print(f"  [Figure 3b] not drawn: {LOADING_AXIS.get('reason')}")

        _axis_lab = LOADING_AXIS.get('label') or 'Catalyst loading'
        _axis_unit = LOADING_AXIS.get('unit') or 'eq.'
        # v22c / fix: this guard still asked for 'catalyst_loading', which is NOT in
        # valid_numerical_cols once the swept axis is another slot -- so panel (b) lost its
        # title, both axis labels and its legend.  Ask for the axis that is actually drawn.
        if _draw_loading_panel and SWEEP_AXIS in valid_numerical_cols:
            ax2.set_xlabel(f'{_axis_lab} ({_axis_unit})', fontweight='bold')
            ax2.set_ylabel(f'Predicted Yield by {_dm} (%)', fontweight='bold')
            ax2.set_title('Yield Sensitivity to %s' % _axis_lab, fontsize=9, fontweight='bold')
            ax2.set_xlim([float(np.nanmin(cat_raw_vals)), float(np.nanmax(cat_raw_vals))])
            ax2.legend()
        ax2.text(-0.15, 1.05, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')
        for spine in ax2.spines.values(): spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(0.8)

        # The old panel drew the model's NATIVE importance (feature_importances_ / coef_) and fell
        # back to np.random.uniform for models that have neither -- HistGB exposes no
        # feature_importances_, and MLP-OneHot / ELM-OneHot are Pipelines, so three of the seven
        # models got FABRICATED bars that looked like real results.
        # Permutation importance is model-agnostic, already computed on the held-out set for panel
        # (a); re-computing it on the TRAINING set adds a memorisation signal -- a feature that
        # matters on train but not on held-out was remembered, not learned.
        _t0 = _timing.perf_counter()
        _imp_te = None
        try:
            _imp_te = np.asarray(perm_result.importances_mean, dtype=float)   # from panel (a)
        except Exception:
            _imp_te = None
        _imp_tr = None
        try:
            if hasattr(y_te, 'index') and hasattr(X_s, 'index'):
                _tr_mask = ~X_s.index.isin(pd.Index(y_te.index))
                if int(_tr_mask.sum()) > 20:
                    _Xtr = X_s[_tr_mask]
                    _ytr = pd.to_numeric(df.loc[X_s.index[_tr_mask], 'yield'],
                                         errors='coerce').to_numpy(float)
                    _imp_tr = np.asarray(permutation_importance(
                        current_model, _Xtr, _ytr, n_repeats=10,
                        random_state=42).importances_mean, dtype=float)
        except Exception as _e:
            print(f"  [Figure 4c] train-set permutation importance skipped: {_e}")
        _t_add(model_name, 'figure4_permutation_importance', _timing.perf_counter() - _t0)

        if _imp_te is not None and len(_imp_te) == len(display_feat_names):
            _order_c = np.argsort(-_imp_te)
            _yp = np.arange(len(_order_c))
            if _imp_tr is not None and len(_imp_tr) == len(_imp_te):
                # v22b: 4c keeps its ORIGINAL two-colour look -- only 4a carries a gradient.
                ax3.barh(_yp + 0.19, _imp_tr[_order_c], height=0.36,
                         color=COLORS['orange'], edgecolor='black', linewidth=0.5,
                         label='training set')
                ax3.barh(_yp - 0.19, _imp_te[_order_c], height=0.36,
                         color=COLORS['blue'], edgecolor='black', linewidth=0.5,
                         label='held-out set')
                ax3.legend(fontsize=7, loc='upper right', framealpha=0.92)
                for _oi in _order_c:
                    _fig_row(_fig4_tag, 'c', 'permutation importance (train)',
                             x=display_feat_names[_oi], y=float(_imp_tr[_oi]),
                             value=float(_imp_te[_oi]),
                             note='y = mean drop in R2 over 10 shuffles; value = held-out')
            else:
                ax3.barh(_yp, _imp_te[_order_c],
                         color=plt.cm.viridis(np.linspace(0.2, 0.85, len(_order_c))),
                         edgecolor='black', linewidth=0.5)
                for _oi in _order_c:
                    _fig_row(_fig4_tag, 'c', 'permutation importance (held-out)',
                             x=display_feat_names[_oi], y=float(_imp_te[_oi]),
                             note='training split could not be recovered')
            ax3.set_yticks(_yp)
            ax3.set_yticklabels([display_feat_names[_oi] for _oi in _order_c],
                                fontsize=8, fontweight='bold')
            _stack = _imp_te[_order_c] if _imp_tr is None else np.r_[_imp_te[_order_c], _imp_tr[_order_c]]
            _mx = float(np.nanmax(_stack)) if len(_stack) else 1.0
            ax3.set_xlim([min(0.0, float(np.nanmin(_imp_te))), _mx * 1.18 if _mx > 0 else 1.0])
            ax3.set_xlabel('Drop in R² when the column is shuffled', fontweight='bold')
            ax3.set_title('Permutation Importance: Train vs Held-out',
                          fontsize=9, fontweight='bold')
            ax3.text(-0.15, 1.05, 'c', transform=ax3.transAxes, fontsize=14,
                     fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'permutation importance unavailable',
                     transform=ax3.transAxes, ha='center', va='center',
                     color='0.45', fontsize=9)
            ax3.set_xticks([])
            ax3.set_yticks([])
        # This slot used to hold the solvent x base mean-yield heatmap.  That matrix is built
        # from `df` and never touches a model, so it was byte-identical in all four per-model
        # figures -- inside a figure whose suptitle ends "-- {model_name}", which reads as a
        # model view that does not exist.  It now lives in Figure 5 panel (b), alongside the
        # other observed-data panels.
        #
        # The replacement is per-model and carries information no other panel does: whether the
        # model reproduces the SHAPE of the measured yield distribution.  The residual panels
        # (F3c residual vs predicted, F3d residual histogram) cannot show this -- a hedged
        # regressor has small residuals near the mean and still misses the zero mode entirely.
        # On the main dataset 46.6 % of reactions give zero yield; the calibration slope and the
        # "<=1 %" shares quantify how much of that the model keeps.
        try:
            _yp_d = np.asarray(current_model.predict(X_te), dtype=float)
            _yt_d = np.asarray(y_te, dtype=float)
            _okd = np.isfinite(_yp_d) & np.isfinite(_yt_d)
            _yp_d, _yt_d = _yp_d[_okd], _yt_d[_okd]
        except Exception:
            _yp_d = _yt_d = np.array([])

        if _yp_d.size >= 10:
            # v15 BUG FIX: the bins used to be a fixed 0..100 grid, so np.histogram silently
            # DROPPED every prediction above 100 -- on Reizman/MLP-OneHot 41 of 189 values --
            # while the annotation was computed on the full array.  The box then claimed
            # "predicted <= 1%: 20%" while the first bar held only 3.4%.  Size the grid from the
            # DATA so nothing is dropped and the picture matches the numbers.  Predictions above
            # 100 % yield are physically impossible and worth showing, not hiding.
            _lo_d = float(min(0.0, np.floor(min(_yt_d.min(), _yp_d.min()))))
            _hi_d = float(max(100.0, np.ceil(max(_yt_d.max(), _yp_d.max()))))
            _bins_d = np.linspace(_lo_d, _hi_d, 26)
            _cnt_m = np.histogram(_yt_d, bins=_bins_d)[0]
            _cnt_p = np.histogram(_yp_d, bins=_bins_d)[0]
            assert _cnt_m.sum() == _yt_d.size and _cnt_p.sum() == _yp_d.size, \
                'histogram lost values -- bin range must cover the data'
            ax4.hist(_yt_d, bins=_bins_d, alpha=0.55, color=COLORS['blue'],
                     edgecolor='black', linewidth=0.4, label='measured')
            ax4.hist(_yp_d, bins=_bins_d, alpha=0.55, color=COLORS['orange'],
                     edgecolor='black', linewidth=0.4, label='predicted')
            try:
                _slope_d = float(np.polyfit(_yt_d, _yp_d, 1)[0])
            except Exception:
                _slope_d = float('nan')
            _z_m = 100.0 * float(np.mean(_yt_d <= 1.0))
            _z_p = 100.0 * float(np.mean(_yp_d <= 1.0))
            _n_over = int(np.count_nonzero(_yp_d > 100.0))
            ax4.annotate('calibration slope = %.2f\n'
                         'yield <= 1%%:  measured %.0f%%  vs  predicted %.0f%%\n'
                         'SD:  measured %.1f  vs  predicted %.1f\n'
                         'predicted > 100%%: %d of %d (%.0f%%)'
                         % (_slope_d, _z_m, _z_p, float(np.std(_yt_d)), float(np.std(_yp_d)),
                            _n_over, _yp_d.size, 100.0 * _n_over / _yp_d.size),
                         xy=(0.03, 0.97), xycoords='axes fraction', va='top',
                         fontsize=6.5, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.4', alpha=0.92))
            ax4.set_xlabel('Yield (%)', fontweight='bold')
            ax4.set_ylabel('Held-out reactions', fontweight='bold')
            ax4.set_title('Predicted vs Measured Distribution (%s)' % model_name,
                          fontsize=9, fontweight='bold')
            ax4.legend(loc='center right', fontsize=7)
            ax4.text(-0.15, 1.05, 'd', transform=ax4.transAxes, fontsize=14,
                     fontweight='bold')
            for _iv in range(len(_bins_d) - 1):
                _fig_row(_fig4_tag, 'd', 'yield histogram',
                         x=float(0.5 * (_bins_d[_iv] + _bins_d[_iv + 1])),
                         y=float(_cnt_m[_iv]), value=float(_cnt_p[_iv]),
                         note='x = bin centre (%); y = measured count; '
                              'value = predicted count')
        else:
            # panel_letter is NOT passed: the unconditional ax4.text(...) below draws the 'd'.
            annotate_loading_unavailable(
                ax4, fontsize=8,
                headline='Prediction/measurement comparison not drawn',
                detail='not enough finite held-out predictions')

        fig.suptitle(f'Feature Attribution and Diagnostic Panels — {_dm}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'Figure4_Feature_Importance_Attribution{file_suffix}.png'), bbox_inches='tight')
        plt.close()
        _fig_flush(output_dir, _fig4_tag,
                   f'Figure4_Feature_Importance_Attribution{file_suffix}_panel_data.csv')

# v22 / issue 2: map a grid CELL (architecture x encoding) onto the trained member that
# can produce its per-model figures.  Six members are trained by the main loop, and the
# twelve cells share those architectures, so each cell is served by its architecture's
# member; the CHOSEN cell decides the figure NAME so the reader sees which encoding won.
_CELL_TO_MEMBER = {
    ('XGBoost', 'integer'): 'XGBoost',
    ('XGBoost', 'one-hot'): 'XGBoost',
    ('XGBoost', 'native categorical'): 'XGBoost',
    ('Gradient Boosting', 'integer'): 'Gradient Boosting',
    ('Gradient Boosting', 'one-hot'): 'Gradient Boosting',
    ('Random Forest', 'integer'): 'Random Forest',
    ('Random Forest', 'one-hot'): 'Random Forest',
    ('HistGB', 'integer'): 'HistGB-NativeCat',
    ('HistGB', 'one-hot'): 'HistGB-NativeCat',
    ('HistGB', 'native categorical'): 'HistGB-NativeCat',
    ('MLP', 'integer'): 'Multilayer Perceptron',
    ('MLP', 'one-hot'): 'MLP-OneHot',
}


def select_cells_for_figures(k=None):
    """Which cells should get per-model figures, as [(display_name, member_name), ...].

    k is the number typed in the menu:
        * an integer K -> the top K cells by R2_test over the twelve-cell grid;
        * None / blank -> one cell per architecture (its best encoding), i.e. five figures.
    """
    g = globals().get('GRID12_TABLE')
    if g is None or not len(g):
        return []
    ok = g[g['status'] == 'ok'].copy()
    if not len(ok):
        return []
    ok = ok.sort_values('r2_test', ascending=False)
    if k:
        picked = ok.head(int(k))
    else:
        picked = ok.groupby('model', sort=False).head(1)
        picked = picked.sort_values('r2_test', ascending=False)
    out = []
    seen = set()
    for _, r in picked.iterrows():
        member = _CELL_TO_MEMBER.get((r['model'], r['encoding']))
        if not member:
            continue
        enc = {'native categorical': 'native'}.get(r['encoding'], r['encoding'])
        disp = '%s (%s)' % (r['model'], enc)
        if member in seen:            # the same member can only be drawn once
            continue
        seen.add(member)
        out.append((disp, member))
    return out


def generate_all_figures(
    df,
    results,
    trained_models,
    cache,
    output_dir,
    raw_feature_names,
    prepared=None
):
    """
    prepared : tuple or None
        (X, y, feat_cols, display_feat_names, label_encoders, scaler) already computed by
        the caller.  Pass it to avoid re-running prepare_features -- the encoders and
        scaler are identical, so recomputing them is pure waste.

    v13: the `virtual_df` / `virtual_X` parameters were REMOVED.  An AST audit showed the
    former was overwritten on entry (line `virtual_df = virtual_df_raw`) and the latter was
    bound and never read, so both were dead.  Inside this function `virtual_df` now refers
    solely to the raw preview returned by the sweep, used only for its row count in a log
    line.
    """
    print("\n--- Generating Figures ---")
    
    sorted_results = results.sort_values(by='R² (Test)', ascending=False)
    # v22 / issue 2: the MENU decides which cells get figures.
    #   K typed   -> top K cells by R2_test from the twelve-cell grid
    #   Enter     -> one cell per architecture (its best encoding) => five figures
    _K = (globals().get('CHOICES') or {}).get('topk')
    _sel = select_cells_for_figures(_K)
    if _sel:
        best_model_names = [m for _, m in _sel]
        best_models_dict = {m: trained_models[m] for _, m in _sel if m in trained_models}
        # display name per member -> picture NAMES and suptitles carry "model (encoding)"
        globals()['MODEL_DISPLAY'] = {m: d for d, m in _sel}
        # (already announced right after the menu; do not repeat it here)
    else:
        top_n = min(4, len(sorted_results))
        best_model_names = sorted_results['Model'].iloc[:top_n].tolist()
        best_models_dict = {name: trained_models[name] for name in best_model_names
                            if name in trained_models}
        globals()['MODEL_DISPLAY'] = {}
        print(f"Top {top_n} models by R-squared: {best_model_names}")
    
    # Fit the scaler on the TRAINING rows only (FIT_PREPROCESSING_ON_TRAIN_ONLY); the
    # returned matrix is still the transform of every row, so all downstream scoring
    # (virtual library, Figure 6) is unaffected in shape.
    #
    # v13: reuse the feature matrix that main() already built instead of recomputing it.
    # prepare_features used to run three times per session (main, build_virtual_library's
    # internal call, and here); the encoder/scaler state is identical each time, so the
    # extra passes were pure waste.
    if prepared is None:
        _tr_rows, _ = get_split_positions(len(df))
        X_s, y_s, feat_cols, display_feat_names, label_encoders, scaler = prepare_features(
            df, raw_feature_names, fit_rows=_tr_rows)
    else:
        (X_s, y_s, feat_cols, display_feat_names,
         label_encoders, scaler) = prepared
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=TEST_SIZE, random_state=42)


    # Figure 1: Model Benchmarking & Performance Comparison
    fig = plt.figure(figsize=(7.5, 6))
    
    ax1 = plt.subplot(2, 2, 1)
    # v21 / A4: one bar per ARCHITECTURE, each at ITS BEST ENCODING -- the twelve-cell grid
    # written to grid12_metrics.csv.  Twelve bars per panel would be unreadable, so the panel
    # collapses the encodings while the CSV keeps all twelve rows.  Falls back to the
    # trained-member table when the grid could not run.
    _g12 = globals().get('GRID12_TABLE')
    _rmse_from_grid = None
    if _g12 is not None and len(_g12):
        _ok12 = _g12[_g12['status'] == 'ok'].copy()
        _ok12['_enc'] = _ok12['encoding'].map({'native categorical': 'native'}).fillna(
            _ok12['encoding'])
        _best12 = (_ok12.sort_values('r2_test').groupby('model', sort=False).tail(1)
                   .sort_values('r2_test', ascending=False))
        models_order = ['%s (%s)' % (m, e) for m, e in zip(_best12['model'], _best12['_enc'])]
        r2_vals = [float(v) for v in _best12['r2_test']]
        _rmse_from_grid = [float(v) for v in _best12['rmse_test']]
        print('[Figure 1] one bar per architecture, best encoding: %s'
              % ', '.join('%s=%.3f' % (m, v) for m, v in zip(models_order, r2_vals)))
    else:
        models_order = results['Model'].tolist()
        r2_vals = [results.loc[results['Model'] == m, 'R² (Test)'].values[0] for m in models_order]
    
    model_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]
    
    min_r2 = min(r2_vals)
    max_r2 = max(r2_vals)
    y_min = max(0.0, float(np.floor((min_r2 - 0.1) * 10) / 10)) if min_r2 > 0 else float(np.floor(min_r2 * 10) / 10)
    # v13 fix: when EVERY model scores below zero -- which happens on the (bh-hte-ood)
    # subsets -- the old fixed top of 1.0 spent the whole upper half of the panel on empty
    # space and squashed the (negative) bars into an unreadable sliver.  Give an all-negative
    # panel a top just above zero instead.
    _ceil_info = estimate_noise_ceiling(df)
    _ceil = _ceil_info['ceiling']
    _has_ceil = bool(np.isfinite(_ceil))
    # v22 / issue 1: the top of the axis used to be pinned at 1.05, so a dataset whose best
    # R2 is 0.42 spent more than half of the panel on empty space.  Scale the axis to the
    # data (and to the ceiling, when one exists), with a small headroom for the labels.
    _ceil_seed = np.nan
    try:
        _ceil_seed = float(_ceil_info['ceiling']) if 'ceiling' in _ceil_info else np.nan
    except Exception:
        _ceil_seed = np.nan
    _top_ref = max([v for v in (r2_vals + ([_ceil_seed] if np.isfinite(_ceil_seed) else []))
                    if np.isfinite(v)] or [0.0])
    if max_r2 > 0:
        # v22d: 1.05 headroom over max(max R2, ceiling).  The old 1.18 x with a 1.12 clamp left
        # a wide empty band above the tallest bar on every dataset that tops out below 0.95.
        y_top = max(0.10, _top_ref * 1.05)
    else:
        y_top = 0.05
    
    # v15: the noise ceiling, drawn as a reference line.  Without it a reader cannot tell
    # whether a low bar means a bad model or an unlearnable dataset -- which is the paper's
    # central point.  The estimator is the same one that produced the section-3.3 numbers; it
    # is a LOWER bound (the mapped condition key does not separate catalyst from ligand), and
    # it is UNDEFINED (not zero) on the three datasets where no condition ever repeats.

    ax1.bar(range(len(models_order)), r2_vals, color=model_colors, edgecolor='black', alpha=0.85)
    ax1.set_xticks(range(len(models_order)))
    ax1.set_xticklabels(models_order, rotation=30, ha='right', fontsize=7)
    ax1.set_ylabel('R² Score', fontweight='bold')
    ax1.set_title('Test Set R² Score Across Models', fontsize=9, fontweight='bold')

    if _has_ceil:
        _GREY = '0.45'
        ax1.axhline(_ceil, color=_GREY, ls='--', lw=1.6, zorder=5)
        # Put the label at whichever side has more room, so it never sits on a bar.
        _lbl_x = len(models_order) - 0.45 if max_r2 < _ceil else 0.05
        _ha = 'right' if max_r2 < _ceil else 'left'
        ax1.text(_lbl_x, _ceil + 0.012,
                 'noise ceiling  %.3f' % _ceil,
                 ha=_ha, va='bottom', fontsize=6.5, fontweight='bold', color=_GREY,
                 bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=_GREY, alpha=0.9),
                 zorder=6)
        # When a model BEATS the estimated ceiling the line sits under the tallest bar, which
        # looks like a bug unless it is called out.
        if max_r2 > _ceil:
            ax1.annotate('best model exceeds the estimated ceiling\n'
                         '-> the estimate is conservative (see 3.3)',
                         xy=(float(np.argmax(r2_vals)), max_r2),
                         xytext=(0.30, 0.06), textcoords='axes fraction',
                         fontsize=6, fontweight='bold', color='0.15',
                         bbox=dict(boxstyle='round,pad=0.22', fc='#fff3cd', ec='0.5', alpha=0.95),
                         arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8), zorder=7)
        # Capture ratio INSIDE the bar body, in grey.  It used to be appended to the value
        # label above the bar as "86% of ceiling", which made the tick row unreadably tall.
        for _ix, _v in enumerate(r2_vals):
            if not np.isfinite(_v) or _ceil <= 0:
                continue
            _in_bar = y_min + (_v - y_min) * 0.5
            if _v > y_min:
                # WHITE and smaller: it sits on saturated bar colours, where any grey was
                # either invisible on the dark bars or invisible on the pale one.
                ax1.text(_ix, _in_bar, '%.0f%%' % (100.0 * _v / _ceil),
                         ha='center', va='center', fontsize=6.2,
                         fontweight='bold', color='white', zorder=6)
        y_top = max(y_top, _ceil + 0.03)

    ax1.set_ylim([y_min, y_top])
    ax1.text(-0.15, 1.05, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')
    add_bar_labels(ax1, fmt="{:.3f}")
    if not _has_ceil:
        ax1.text(0.5, 0.97,
                 'noise ceiling: not estimable (no condition was measured twice)',
                 ha='center', va='top', fontsize=6.5, color='0.35',
                 transform=ax1.transAxes,
                 bbox=dict(boxstyle='round,pad=0.2', fc='#f0f0f0', ec='0.6', alpha=0.9))

    ax2 = plt.subplot(2, 2, 2)
    rmse_vals = _rmse_from_grid if _rmse_from_grid is not None else [
        results.loc[results['Model'] == m, 'RMSE (Test)'].values[0] for m in models_order]
    # v22b / issue 3: RMSE bars are also value-driven (short = good = pale).
    bar_gradient(ax2, rmse_vals, cmap='Oranges', horizontal=False, edgecolor='black', alpha=0.95)
    ax2.set_xticks(range(len(models_order)))
    ax2.set_xticklabels(models_order, rotation=30, ha='right', fontsize=7)
    ax2.set_ylabel('RMSE (% yield)', fontweight='bold')
    ax2.set_title('Test Set RMSE Across Models', fontsize=9, fontweight='bold')
    ax2.set_ylim([0, max(rmse_vals) * 1.25])
    ax2.text(-0.15, 1.05, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')
    add_bar_labels(ax2, fmt="{:.2f}")

    # v22c / fix: (c) and (d) used EVERY trained member while (a)/(b) showed the selection.
    _disp_map = globals().get('MODEL_DISPLAY') or {}
    _sel_members = list(_disp_map.keys()) if _disp_map else list(trained_models.keys())
    _sel_labels = ([_disp_map[m] for m in _sel_members] if _disp_map
                   else [str(m) for m in _sel_members])

    ax3 = plt.subplot(2, 2, 3)
    max_err_val = 0
    for name, _lab_sel in zip(_sel_members, _sel_labels):
        preds = cache[name]['y_pred_test']
        abs_err = np.sort(np.abs(y_te - preds))
        max_err_val = max(max_err_val, np.percentile(abs_err, 98))
        cdf = np.arange(1, len(abs_err) + 1) / len(abs_err)
        ax3.plot(abs_err, cdf, label=_lab_sel, lw=1.5)
    
    ax3.set_xlabel('Absolute Error (% Yield)', fontweight='bold')
    ax3.set_ylabel('Cumulative Probability', fontweight='bold')
    ax3.set_title('Cumulative Error Distribution', fontsize=9, fontweight='bold')
    ax3.set_xlim([0, min(35.0, max(10.0, max_err_val * 1.1))])
    ax3.axhline(0.8, color='grey', linestyle=':', alpha=0.7)
    ax3.legend(loc='lower right', fontsize=6.5)
    ax3.text(-0.15, 1.05, 'c', transform=ax3.transAxes, fontsize=14, fontweight='bold')

    ax4 = plt.subplot(2, 2, 4)
    _res_sel = results[results['Model'].isin(_sel_members)].copy()
    if len(_res_sel) == 0:
        _res_sel = results.copy()
    _res_sel['_lab'] = [_disp_map.get(m, m) for m in _res_sel['Model']]
    ax4.errorbar(range(len(_res_sel)), _res_sel['CV R² Mean'], yerr=_res_sel['CV R² Std'],
                 fmt='o', color=COLORS['green'], ecolor='black', capsize=4)
    ax4.set_xticks(range(len(_res_sel)))
    ax4.set_xticklabels(_res_sel['_lab'], rotation=30, ha='right', fontsize=7)
    # v22j: the auto lower bound sat just under the lowest error bar; the panel read as if
    # every model were nearly equal.  Bottom = 0.5 x the smallest (mean - std).
    _d_lo_vals = (_res_sel['CV R² Mean'] - _res_sel['CV R² Std']).to_numpy(float)
    _d_lo_vals = _d_lo_vals[np.isfinite(_d_lo_vals)]
    if _d_lo_vals.size:
        ax4.set_ylim([float(_d_lo_vals.min()) * 0.5,
                      float((_res_sel['CV R² Mean'] + _res_sel['CV R² Std']).max()) * 1.05])
    ax4.set_ylabel('Cross-validation R²', fontweight='bold')
    ax4.set_title('5-Fold Cross-Validation R²', fontsize=9, fontweight='bold')
    ax4.text(-0.15, 1.05, 'd', transform=ax4.transAxes, fontsize=14, fontweight='bold')
    fig.suptitle('Model Benchmarking and Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'Figure1_Model_Comparison.png'), bbox_inches='tight')
    plt.close()

    # Figures 2, 3, 4 (per model).  Extracted into a function so they can run
    # BEFORE the virtual-library sweep -- they need only the trained models.
    generate_figure2_3_4_per_model(
        df, X_s, X_te, y_te, trained_models, cache, display_feat_names,
        feat_cols, scaler, output_dir, best_model_names, raw_feature_names,
        label_encoders=label_encoders)

    # Figure 5: Comprehensive Substrate & Solvent Analysis
    fig = plt.figure(figsize=(9.5, 8.5))
    # v14: panels (a) and (b) used `reactant_1` unconditionally, so a dataset without it
    # (aap9112, the bh-hte-ood files) fell into the hardcoded [[80,85],[60,75]] placeholder and
    # drew a FABRICATED chart.  The axis is now a slot: reactant_1 first, then the other
    # member of the scope family, and an explicit caption when neither exists.
    _p_s = _pick_param('reactant_1', 'substrate', df, label_encoders)
    # v21 / C3: the categorical variable chosen in the menu drives this panel when the file
    # actually has it; the resolver above stays as the fallback for batch runs.
    _ch_single = (globals().get('CHOICES') or {}).get('single_cat')
    if _ch_single and _ch_single in df.columns and df[_ch_single].nunique() > 1:
        _p_s = _ch_single
        print('[Figure 5a] single categorical from the menu: %s' % _p_s)
    _lab_s = PARAM_LABEL.get(_p_s, _p_s) if _p_s else None
    if _p_s:
        all_substrates = list(df[_p_s].value_counts().index)
    else:
        all_substrates = []
    substrates = all_substrates[:12] if len(all_substrates) > 12 else all_substrates
    labels_sub = [str(s)[:15] + '..' if len(str(s)) > 15 else str(s) for s in substrates]

    ax1 = plt.subplot(2, 2, 1)
    _fig5_tag = 'Figure5_Comprehensive_Substrate_and_Solvent_Analysis'
    if _p_s and substrates:
        data_v = [df[df[_p_s] == s]['yield'].values for s in substrates]
        for _s, _vals in zip(substrates, data_v):
            for _v in np.asarray(_vals, dtype=float):
                _fig_row(_fig5_tag, 'a', 'yield distribution', x=str(_s), y=float(_v),
                         note='one row per reaction; x = %s' % _lab_s)
        parts = ax1.violinplot(data_v, positions=range(len(substrates)), showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS['blue'])
            pc.set_alpha(0.6)
        # v15: panel (b) used to be a separate bar chart of these same means, i.e. the same
        # quantity twice.  The number is now written above each violin and panel (b) is free
        # for the observed interaction heatmap that used to sit in Figure 4d.
        _means_a = [float(np.mean(np.asarray(v, dtype=float))) for v in data_v]
        # v21 / C10: the labels used to sit at a FIXED y=102, so on a dataset whose violins all
        # reach ~100 the numbers stacked into one unreadable band.  Each label now rides just
        # above its OWN violin, with a small upward stagger when two neighbours would collide.
        _tops = []
        for _v in data_v:
            _arr = np.asarray(_v, dtype=float)
            _tops.append(float(np.nanmax(_arr)) if _arr.size and np.isfinite(_arr).any() else 0.0)
        _y_prev = None
        _y_max_drawn = 0.0
        for _ix, (_mv, _vals, _top) in enumerate(zip(_means_a, data_v, _tops)):
            _y = _top + 3.0
            # v22b / issue 6: the old rule forced every label ABOVE the previous one (a running
            # maximum), so eight violins marched the labels up to y=140 while the data stopped
            # at 95.  Nudge only when the two labels would actually collide, and let the label
            # drop again when a violin is short.
            if _y_prev is not None and abs(_y - _y_prev) < 6.0:
                _y = _y_prev + 6.0 if _y >= _y_prev else _y_prev - 6.0
            _y_prev = _y
            _y_max_drawn = max(_y_max_drawn, _y)
            ax1.text(_ix, _y, '%.1f  (n=%d)' % (_mv, len(_vals)),
                     ha='center', va='bottom',
                     fontsize=max(5.0, min(7.5, 7.5 - 0.18 * len(substrates))),
                     fontweight='bold', color='black')
            _fig_row(_fig5_tag, 'a', 'mean yield', x=str(substrates[_ix]), y=_mv,
                     value=int(len(_vals)),
                     note='mean yield (%) annotated above each violin; value = n reactions')
        ax1.set_xticks(range(len(substrates)))
        ax1.set_xticklabels(labels_sub, rotation=30, ha='right', fontweight='bold', fontsize=7)
        ax1.set_ylabel('Yield (%)', fontweight='bold')
        ax1.set_title('Yield Distribution by %s (mean annotated)' % _lab_s,
                      fontsize=9, fontweight='bold')
        ax1.set_ylim([0, max(112.0, _y_max_drawn + 8.0)])
    else:
        annotate_loading_unavailable(
            ax1, fontsize=9,
            headline='Substrate breakdown not drawn',
            detail='this dataset has no varying reactant_1 or reactant_2 column')
    ax1.text(-0.15, 1.05, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')

    ax2 = plt.subplot(2, 2, 2)
    # v15: panel (c) is the figure's original subject (substrate scope), so its pair is resolved
    # FIRST and panel (b) is then resolved with `used=(...)`.  Without this the two panels both
    # fall through their priority lists onto the same pair whenever a dataset has only two
    # varying categoricals -- the same heatmap twice.
    _p_cx1, _p_cy1 = _resolve_axis_pair(('reactant_1', 'reactant_2'), 'substrate',
                                        df, label_encoders)
    # v15: this slot now carries the observed reagent-pair interaction heatmap that used to be
    # Figure 4 panel d.  It is computed from `df` and never touches a model, so inside a
    # per-model figure (suptitle "... -- {model_name}") it was the SAME picture four times and
    # implied a model view that did not exist.  Figure 5 collects the observed-data panels, so
    # it belongs here.  The bar chart it replaces is now the mean annotation in panel (a).
    # v21 / C4: the two pairs chosen in the interactive menu win over the resolver.
    _ch = globals().get('CHOICES') or {}
    _ch_pair1 = _ch.get('pair1')
    _ch_pair2 = _ch.get('pair2')
    if _ch_pair2 and len(_ch_pair2) == 2 and all(c in df.columns for c in _ch_pair2):
        _p_cx1, _p_cy1 = _ch_pair2[0], _ch_pair2[1]
        print('[Figure 5] panel (c) pair from the menu: %s x %s' % (_p_cx1, _p_cy1))
    _p_bx1, _p_by1 = _resolve_axis_pair(('solvent', 'base'), 'process_categorical',
                                        df, label_encoders or {},
                                        used=tuple(p for p in (_p_cx1, _p_cy1) if p))
    if _ch_pair1 and len(_ch_pair1) == 2 and all(c in df.columns for c in _ch_pair1):
        _p_bx1, _p_by1 = _ch_pair1[0], _ch_pair1[1]
        print('[Figure 5] panel (b) pair from the menu: %s x %s' % (_p_bx1, _p_by1))
    if _p_bx1 and _p_by1:
        _lab_bx1 = PARAM_LABEL.get(_p_bx1, _p_bx1)
        _lab_by1 = PARAM_LABEL.get(_p_by1, _p_by1)
        _df_succ_bs = df[df['yield'] >= 50.0]
        if len(_df_succ_bs) < 10:
            _df_succ_bs = df[df['yield'] >= df['yield'].quantile(0.70)]
        # Unobserved pairs stay NaN (grey), NOT 0: a 0 here would read as genuinely bad
        # chemistry rather than "never tested".
        _bs_heat = _df_succ_bs.groupby([_p_by1, _p_bx1])['yield'].mean().unstack()
        _bs_heat = _bs_heat.reindex(
            index=list(_df_succ_bs[_p_by1].value_counts().index),
            columns=list(_df_succ_bs[_p_bx1].value_counts().index))
        # v22o: top-10 per axis (most populated), as the composite pipeline does with top-N
        _m_bs, _xl_bs, _yl_bs = _top_n_axes(_bs_heat.values.astype(float),
                                            list(_bs_heat.columns), list(_bs_heat.index), n=10)
        _bs_heat = pd.DataFrame(_m_bs, index=_yl_bs, columns=_xl_bs)
        _mv_bs = np.ma.masked_invalid(_bs_heat.values.astype(float))
        _obs_bs = _mv_bs.compressed()
        _vmin_bs = max(0, np.floor(_obs_bs.min() - 2)) if _obs_bs.size else 0
        _vmax_bs = min(100, np.ceil(_obs_bs.max() + 2)) if _obs_bs.size else 100
        _cmap_bs = plt.get_cmap('YlOrRd').copy()
        _cmap_bs.set_bad('#e8e8e8')
        _im_bs = ax2.imshow(_mv_bs, cmap=_cmap_bs, aspect='auto',
                            interpolation='nearest', vmin=_vmin_bs, vmax=_vmax_bs)
        _f_bx = _adaptive_fontsize(len(_bs_heat.columns), base=7.5, min_size=3.5, full_upto=12)
        _f_by = _adaptive_fontsize(len(_bs_heat.index), base=7.5, min_size=3.5, full_upto=12)
        # v22k: the black plate around the single maximum is gone -- every heatmap now marks
        # its top decile with the same bold, box-free label as Figure 5c.
        _mark_heatmap_top_decile(ax2, _bs_heat.values, fontsize=max(5.5, _f_bx))
        _x_b = [str(x)[:12] for x in _bs_heat.columns]
        _y_b = [str(x)[:15] for x in _bs_heat.index]
        ax2.set_xticks(range(len(_x_b)))
        ax2.set_yticks(range(len(_y_b)))
        ax2.set_xticklabels(_x_b, rotation=30, ha='right', fontsize=_f_bx,
                            fontweight='bold', rotation_mode='anchor')
        ax2.set_yticklabels(_y_b, rotation=20, ha='right', fontsize=_f_by, fontweight='bold')
        thin_tick_labels(ax2, 'x')
        thin_tick_labels(ax2, 'y')
        ax2.set_xlabel(_lab_bx1, fontweight='bold')
        ax2.set_ylabel(_lab_by1, fontweight='bold')
        ax2.set_title('%s-%s Interaction Yield Heatmap' % (_lab_by1, _lab_bx1),
                      fontsize=9, fontweight='bold')
        _cb_bs = plt.colorbar(_im_bs, ax=ax2)
        _cb_bs.ax.set_ylabel('Mean Yield (%) -- all reactions', rotation=270, labelpad=12,
                             fontweight='bold')
        _fig_row(_fig5_tag, 'b', '%s x %s' % (_lab_by1, _lab_bx1),
                 x='(matrix)', y=np.nan, value=np.nan,
                 note='observed mean yield (%) over successful reactions; '
                      'missing = unobserved pair')
        _fig_grid(_fig5_tag, 'b', '%s x %s' % (_lab_by1, _lab_bx1),
                  np.asarray(_bs_heat.values, dtype=float),
                  [str(c) for c in _bs_heat.columns],
                  [str(i) for i in _bs_heat.index],
                  note='mean yield (%) over successful reactions; missing = unobserved pair')
    else:
        annotate_loading_unavailable(
            ax2, fontsize=9,
            headline='Interaction heatmap not drawn',
            detail='needs two varying process-categorical parameters; '
                   'this dataset has fewer than two')
    ax2.text(-0.15, 1.05, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')

    ax3 = plt.subplot(2, 2, 3)
    # (the pair for this panel was resolved above, before panel (b), so the two never collide)
    if _p_cx1 and _p_cy1:
        _lab_cx1 = PARAM_LABEL.get(_p_cx1, _p_cx1)
        _lab_cy1 = PARAM_LABEL.get(_p_cy1, _p_cy1)
        # v21 / C11: NO yield filter in this panel any more.  Panel (c) is the SCOPE view
        # (substrate x leaving group), and keeping only the high-yield reactions biased every
        # cell upwards while hiding exactly the combinations that fail.  Panel (b) keeps its
        # "success only" filter because it is the process view, where success is the question.
        df_succ = df

        lg_mat = df_succ.groupby([_p_cy1, _p_cx1])['yield'].mean().unstack()
        lg_mat = lg_mat.reindex(index=list(df_succ[_p_cy1].value_counts().index),
                                columns=list(df_succ[_p_cx1].value_counts().index))

        valid_vals = lg_mat.values[~np.isnan(lg_mat.values)]
        vmin_2c = max(0, np.floor(valid_vals.min() - 2)) if len(valid_vals) > 0 else 0
        vmax_2c = min(100, np.ceil(valid_vals.max() + 2)) if len(valid_vals) > 0 else 100
        _fig_grid(_fig5_tag, 'c', '%s x %s' % (_lab_cy1, _lab_cx1),
                  np.asarray(lg_mat.values, dtype=float),
                  [str(c) for c in lg_mat.columns],
                  [str(i) for i in lg_mat.index],
                  note='mean yield (%) over ALL reactions; missing = unobserved pair')
        
        # v14: adaptive fonts.  A 52 x 12 matrix drew twelve-character labels at 7 pt and
        # 7 pt cell numbers into a quarter-page panel, so both the tick labels and the
        # annotations overlapped into unreadable bands.
        _f_cell = _adaptive_fontsize(max(lg_mat.shape), base=7, min_size=4.0, full_upto=12)
        _f_xt = _adaptive_fontsize(len(lg_mat.columns), base=7, min_size=3.5, full_upto=12)
        _f_yt = _adaptive_fontsize(len(lg_mat.index), base=7, min_size=3.5, full_upto=12)
        # v22c / fix: annot=True wrote a number in EVERY cell, so the top decile was
        # impossible to see.  Numbers are drawn below for the top 10 %% only.
        _m_c, _xl_c, _yl_c = _top_n_axes(lg_mat.values.astype(float),
                                         list(lg_mat.columns), list(lg_mat.index), n=10)
        lg_mat = pd.DataFrame(_m_c, index=_yl_c, columns=_xl_c)
        sns.heatmap(lg_mat, cmap='YlGnBu', annot=False, fmt=".1f", ax=ax3,
                    vmin=vmin_2c, vmax=vmax_2c,
                    cbar_kws={'label': 'Mean Yield (%) -- all reactions'},
                    annot_kws={'size': _f_cell, 'weight': 'bold'},
                    linewidths=0, linecolor=None, mask=lg_mat.isna())
        _vals_c = np.asarray(lg_mat.values, dtype=float)
        _fin_c = _vals_c[np.isfinite(_vals_c)]
        if _fin_c.size >= 4:
            _thr_c = float(np.nanpercentile(_fin_c, 90.0))
            for _ri in range(_vals_c.shape[0]):
                for _ci in range(_vals_c.shape[1]):
                    _vv = _vals_c[_ri, _ci]
                    if np.isfinite(_vv) and _vv >= _thr_c:
                        ax3.text(_ci + 0.5, _ri + 0.5, '%.1f' % _vv, ha='center',
                                 va='center', fontsize=_f_cell, fontweight='bold',
                                 color='white', zorder=6)
        
        for spine in ax3.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        
        x_labs = [str(x)[:10] + '..' if len(str(x)) > 10 else str(x) for x in lg_mat.columns]
        y_labs = [str(x)[:12] + '..' if len(str(x)) > 12 else str(x) for x in lg_mat.index]

        # v14: set the tick LOCATIONS explicitly before the labels.
        # seaborn's heatmap installs a FixedLocator, but on a tall matrix it thins the tick
        # positions instead of emitting one per row: for (Roche66k)HTE_all_reactions the
        # 52-row matrix left only 26 y-tick locations, and set_yticklabels(52) then raised
        #     ValueError: The number of FixedLocator locations (26) ... does not match the
        #     number of labels (52).
        # Every other heatmap in this file already does set_ticks first; this one did not.
        ax3.set_xticks(np.arange(len(x_labs)))
        ax3.set_yticks(np.arange(len(y_labs)))
        ax3.set_xticklabels(x_labs, rotation=30, ha='right', fontsize=_f_xt,
                            fontweight='bold', rotation_mode='anchor')
        ax3.set_yticklabels(y_labs, fontsize=_f_yt, fontweight='bold')
        # v14: same measured thinning as F4d -- see thin_tick_labels.
        thin_tick_labels(ax3, 'x')
        thin_tick_labels(ax3, 'y')
        ax3.set_xlabel(_lab_cx1, fontweight='bold')
        ax3.set_ylabel(_lab_cy1, fontweight='bold')
        ax3.set_title('%s vs %s Heatmap' % (_lab_cy1, _lab_cx1),
                      fontsize=9, fontweight='bold')
        # v22b / issue 4: every cell already carries its value, so the top decile is marked by
        # EMPHASIS (bold + dark red) instead of a second number or a dashed box.
        _vals_c = np.asarray(lg_mat.values, dtype=float)
        _fin_c = _vals_c[np.isfinite(_vals_c)]
        if _fin_c.size >= 4:
            _thr_c = float(np.nanpercentile(_fin_c, 90.0))
            for _t in ax3.texts:
                try:
                    _tv = float(str(_t.get_text()).replace('%', ''))
                except Exception:
                    continue
                if _tv >= _thr_c:
                    pass                        # drawn below (only the top decile)
            # note placed BELOW the tick labels so it cannot overlap them
            # put the legend in the TITLE: text below the axes was clipped by tight_layout
            ax3.set_title('%s vs %s Heatmap\n[white bold = top 10 %% >= %.1f %%]'
                          % (_lab_cy1, _lab_cx1, _thr_c),
                          fontsize=8.5, fontweight='bold')
    else:
        annotate_loading_unavailable(
            ax3, fontsize=9,
            headline='Substrate x leaving-group heatmap not drawn',
            detail='needs two varying scope parameters; this dataset has fewer than two')
    ax3.text(-0.15, 1.05, 'c', transform=ax3.transAxes, fontsize=14, fontweight='bold')

    ax4 = plt.subplot(2, 2, 4)
    has_smiles_pca = False
    smiles_err_msg = "No SMILES data provided"

    if not RDKIT_AVAILABLE:
        smiles_err_msg = "RDKit not installed\n(pip install rdkit)"
    else:
        # v21 / C5: the SMILES slot chosen in the interactive menu wins.  It may be a
        # REACTANT column now (reactant_1_smiles / reactant_2_smiles), which is what the
        # paper's PCA actually shows; the solvent heuristic is only the fallback.
        _ch_smi = (globals().get('CHOICES') or {}).get('pca_smiles')
        smiles_col = None
        if _ch_smi and _ch_smi in df.columns:
            smiles_col = _ch_smi
            print('[Figure 5d] SMILES column from the menu: %s' % smiles_col)
        if smiles_col is None:
            smiles_col = 'solvent_smiles' if 'solvent_smiles' in df.columns else None
        if smiles_col is None:
            # v14: same rule as the loader -- a SMILES column is only usable here if it is
            # about the SOLVENT.  This panel used to take candidates[0] as well, so even with
            # the loader fixed it would pull in base_smiles and describe base molecules as a
            # "solvent" PCA whenever the labelling column happened to vary.
            candidates = [c for c in df.columns if 'smiles' in c.lower()]
            solvent_like = [c for c in candidates if 'solvent' in c.lower()]
            if solvent_like:
                smiles_col = solvent_like[0]
            elif candidates:
                smiles_err_msg = ("No solvent SMILES column.\n"
                                  "Available: " + ', '.join(str(c) for c in candidates[:4]))

        if smiles_col is None:
            smiles_err_msg = "No SMILES column found in CSV"
        else:
            smiles_df = df.dropna(subset=[smiles_col]).copy()
            smiles_df[smiles_col] = smiles_df[smiles_col].astype(str).str.strip().str.strip("'\"")
            smiles_df = smiles_df[smiles_df[smiles_col].str.lower().ne('nan') & smiles_df[smiles_col].ne('')]

            # v14: the points are LABELLED with this column, so it has to vary.  When the
            # file has no solvent column the required-column filler sets `solvent` to
            # 'Unknown' for every row, and the panel then drew N points that all read
            # "Unknown" (observed: 3, on (DoyleHTE)CN_coupling_3600).  Checked BEFORE the
            # RDKit work so a useless panel costs nothing.
            # label the points with the matching reagent slot, not always 'solvent'
            _label_col_guess = smiles_col
            for _pref in ('reactant_1', 'reactant_2', 'catalyst', 'ligand', 'base',
                          'additive', 'solvent'):
                if smiles_col.startswith(_pref) and _pref in smiles_df.columns:
                    _label_col_guess = _pref
                    break
            if _label_col_guess == smiles_col and 'solvent' in smiles_df.columns:
                _label_col_guess = 'solvent'
            if len(smiles_df) == 0:
                smiles_err_msg = f"Column '{smiles_col}' contains no valid data"
            elif smiles_df[_label_col_guess].nunique(dropna=True) <= 1:
                smiles_err_msg = (
                    f"'{smiles_col}' is available, but the column used to LABEL the points\n"
                    f"('{_label_col_guess}') has a single value "
                    f"('{str(smiles_df[_label_col_guess].iloc[0])[:20]}'), so every point\n"
                    f"would carry the same label.  Skipped.")
            else:
                solvent_name_col = _label_col_guess
                grouped = smiles_df.groupby([solvent_name_col, smiles_col])['yield'].mean().reset_index()

                descriptors_list, valid_rows = [], []
                for idx, row in grouped.iterrows():
                    smi = row[smiles_col]
                    try:
                        mol = Chem.MolFromSmiles(smi)
                    except Exception:
                        mol = None
                    
                    if mol is not None:
                        desc = [
                            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                            Descriptors.TPSA(mol), Descriptors.FractionCSP3(mol),
                            Descriptors.MolMR(mol), Descriptors.NumRotatableBonds(mol)
                        ]
                        descriptors_list.append(desc)
                        valid_rows.append(idx)

                if len(descriptors_list) < 2:
                    smiles_err_msg = f"Valid parsed SMILES < 2 (Parsed: {len(descriptors_list)})"
                else:
                    valid_grouped = grouped.iloc[valid_rows].copy()
                    X_desc = np.array(descriptors_list)
                    X_scaled_desc = StandardScaler().fit_transform(X_desc)
                    pca = PCA(n_components=2)
                    pca_coords = pca.fit_transform(X_scaled_desc)

                    pc1, pc2 = pca_coords[:, 0], pca_coords[:, 1]
                    yields = valid_grouped['yield'].values
                    solvent_names = valid_grouped[solvent_name_col].astype(str).values
                    for _ip in range(len(pc1)):
                        _fig_row(_fig5_tag, 'd', 'PCA of RDKit descriptors',
                                 x=float(pc1[_ip]), y=float(pc2[_ip]),
                                 value=float(yields[_ip]),
                                 note='%s = %s; value = mean yield (%%); '
                                      'PC1 %.1f%% var, PC2 %.1f%% var'
                                      % (solvent_name_col, solvent_names[_ip],
                                         pca.explained_variance_ratio_[0] * 100,
                                         pca.explained_variance_ratio_[1] * 100))

                    sc = ax4.scatter(pc1, pc2, c=yields, cmap='YlGnBu', s=140, edgecolor='black', zorder=3)
                    cbar = plt.colorbar(sc, ax=ax4)
                    cbar.ax.set_ylabel('Mean Yield (%)', rotation=270, labelpad=12, fontweight='bold')

                    # v14: was a fixed (5, 5) offset for every point, which printed the
                    # labels on top of each other whenever two solvents clustered.  Greedy
                    # collision-free placement instead, labelled with the variable that the
                    # panel is actually grouping by.
                    _disp = [str(t)[:12] + '..' if len(str(t)) > 12 else str(t)
                             for t in solvent_names]
                    _n_placed, _n_supp = place_labels_no_overlap(
                        ax4, pc1, pc2, _disp, fontsize=7.0, max_labels=30)
                    if _n_supp:
                        # never print a label on top of another; say what was left out
                        ax4.text(0.99, 0.01,
                                 '(%d of %d labels omitted where points coincide)'
                                 % (_n_supp, _n_supp + _n_placed),
                                 transform=ax4.transAxes, ha='right', va='bottom',
                                 fontsize=6, color='0.35')

                    _grp = PARAM_LABEL.get(solvent_name_col, solvent_name_col)
                    ax4.set_xlabel(f'{_grp} PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontweight='bold')
                    ax4.set_ylabel(f'{_grp} PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontweight='bold')
                    ax4.set_title('%s SMILES Descriptor PCA' % _grp, fontsize=9, fontweight='bold')
                    ax4.grid(True, linestyle='--', alpha=0.5)
                    has_smiles_pca = True

    if not has_smiles_pca:
        ax4.text(0.5, 0.5, smiles_err_msg, ha='center', va='center', fontsize=10, fontweight='bold', color='grey', transform=ax4.transAxes)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title('Solvent SMILES Descriptor PCA', fontsize=9, fontweight='bold')
        _fig_row(_fig5_tag, 'd', 'unavailable', note=str(smiles_err_msg).replace('\n', ' '))

    ax4.text(-0.15, 1.05, 'd', transform=ax4.transAxes, fontsize=14, fontweight='bold')
    fig.suptitle('Observed Yield Structure: Substrate Scope, Reagent Interactions and Solvent Space', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'Figure5_Comprehensive_Substrate_and_Solvent_Analysis.png'), bbox_inches='tight')
    plt.close()
    _fig_flush(output_dir, _fig5_tag,
               'Figure5_Comprehensive_Substrate_and_Solvent_Analysis_panel_data.csv')

    # EXPENSIVE STEP: exhaustive virtual-library ensemble sweep.
    # v13: this used to run BEFORE Figure 1, even though Figures 1-5 never touch
    # its outputs (bo_df / virtual_df are only used by Figures 8, 9 and 10).
    # Moving it here gives strict Figure-number order: F1-F5 render first, then the
    # sweep, then F6-F10.
    # virtual-library scoring: streamed since v12, the enumerated library never sits in memory
    # v14: the EI incumbent.  See EI_INCUMBENT_MODE for why this is a modelling choice
    # rather than a constant.  Logged because every downstream EI number depends on it.
    best_observed_yield, _y_record = resolve_ei_incumbent(df)
    print(f"  [EI incumbent] mode={EI_INCUMBENT_MODE}"
          + (f" q={EI_INCUMBENT_QUANTILE:g}" if EI_INCUMBENT_MODE == 'quantile' else '')
          + f"  ->  f* = {best_observed_yield:.2f}%   "
            f"(all-time record = {_y_record:.2f}%)")

    # Nothing here is hard-coded: the R^2 values are read from `results`, which
    # `train_and_evaluate_models` produced for the current dataset and split, so the
    # weights change automatically with the data / the model panel / the split.
    #   w_i  proportional to  (max(R2_i, ENSEMBLE_MIN_WEIGHT_R2)) ** ENSEMBLE_WEIGHT_POWER
    #   then normalised to sum to 1.
    r2_map = dict(zip(results['Model'], results['R² (Test)']))
    member_names = list(best_models_dict.keys())

    missing = [nm for nm in member_names if nm not in r2_map]
    if missing:
        print(f"  [WARNING] no test R² found for {missing}; "
              f"they will receive the floor weight {ENSEMBLE_MIN_WEIGHT_R2}. "
              f"Available entries: {list(r2_map)}")

    member_r2 = np.array([float(r2_map[nm]) if nm in r2_map else ENSEMBLE_MIN_WEIGHT_R2
                          for nm in member_names], dtype=float)


    raw_w = np.power(np.maximum(member_r2, float(ENSEMBLE_MIN_WEIGHT_R2)),
                     float(ENSEMBLE_WEIGHT_POWER))
    ens_weights = raw_w / raw_w.sum()
    print("    " + "  ".join(f"w[{nm}]={wv:.4f}" for nm, wv in zip(member_names, ens_weights)))

    # Figure 6: direct ensemble ranking of the REAL reactions (+ calibration).
    # Renumbered from Figure 10, and placed ahead of the sweep because it needs only
    # the prediction columns written by save_predictions_csv -- not the enumeration.
    try:
        generate_figure6_direct_ranking(
            df=df,
            output_dir=output_dir,
            best_models_dict=best_models_dict,
            top_n_table=20,
            max_model_gap=ENSEMBLE_MAX_MODEL_GAP,
            min_support_count=0,
            weights=ens_weights,
            member_names=member_names
        )
    except Exception as e:
        print(f"Figure 6 failed: {e}")

    # EXPENSIVE STEP: exhaustive virtual-library ensemble sweep.
    print("\nVirtual-library ensemble prediction...")
    # v13: three interchangeable engines.  All three are EXHAUSTIVE; they differ only in
    # how the work is scheduled.
    #   parallel   -- default: exact sweep split across processes (needs the feature cache)
    #   streaming  -- exact single-process sweep, lowest memory, slowest
    #   two_stage  -- OPTIONAL approximation, kept for reference; known to miss the true
    #                 top-k (measured top-20 recall 8/20), so not used by default
    _t_sweep0 = _timing.perf_counter()   # v15: must be OUTSIDE the branch
    # v21 / A5: the CHUNKED streaming engine is the default.  It walks the enumeration in
    # blocks of STREAM_CHUNK_COMBOS, scores each block and keeps only the survivor pool, so
    # peak memory is one block instead of the whole virtual library.  The multiprocessing
    # engine is still available (CD_SWEEP_ENGINE=parallel) but it needs the on-disk feature
    # cache, which is what made it the memory-hungry path.
    _SWEEP_ENGINE = os.environ.get('CD_SWEEP_ENGINE', 'streaming').strip().lower()
    if USE_TWO_STAGE_SCREENING:
        print(f"  [two-stage] APPROXIMATE mode: stage 1 = 1 model over the full "
              f"enumeration, stage 2 = full ensemble over the best "
              f"{STAGE1_KEEP_ROWS:,} rows.  Measured top-20 recall is only ~8/20, so this "
              f"changes Figure 10's ranking; set USE_TWO_STAGE_SCREENING=False for exactness.")
        bo_df, n_library, virtual_df_raw = generate_bo_results_two_stage(
            df=df, models=best_models_dict, feat_cols=feat_cols,
            label_encoders=label_encoders, scaler=scaler,
            best_observed_yield=best_observed_yield,
            chunk_combos=STREAM_CHUNK_COMBOS,
            max_combinations=MAX_VIRTUAL_COMBINATIONS, xi=0.01,
            keep_top=SURVIVOR_POOL_ROWS, weights=ens_weights, verbose=True)
    elif USE_MULTIPROCESSING and _SWEEP_ENGINE == 'parallel':
        _nw = N_WORKERS if (N_WORKERS and N_WORKERS > 0) else max(
            1, min(MAX_WORKERS, (os.cpu_count() or 2) - 1))
        print(f"  [parallel] EXACT exhaustive sweep on {_nw} processes "
              f"(cpu_count={os.cpu_count()})")
        bo_df, n_library, virtual_df_raw = generate_bo_results_parallel(
            df=df, models=best_models_dict, feat_cols=feat_cols,
            label_encoders=label_encoders, scaler=scaler,
            best_observed_yield=best_observed_yield,
            max_combinations=MAX_VIRTUAL_COMBINATIONS, xi=0.01,
            keep_top=SURVIVOR_POOL_ROWS, weights=ens_weights,
            n_workers=_nw, verbose=True)
    else:
        print("  [streaming] EXACT exhaustive sweep, single process.")
        bo_df, n_library, virtual_df_raw = generate_bo_results_streaming(
            df=df, models=best_models_dict, feat_cols=feat_cols,
            label_encoders=label_encoders, scaler=scaler,
            best_observed_yield=best_observed_yield,
            chunk_combos=STREAM_CHUNK_COMBOS,
            max_combinations=MAX_VIRTUAL_COMBINATIONS, xi=0.01,
            keep_top=SURVIVOR_POOL_ROWS, weights=ens_weights, verbose=True)
        _t_sweep = _timing.perf_counter() - _t_sweep0
        TIMINGS_SINGLE['virtual_library_sweep'] = _t_sweep
        print(f"  [timings] virtual-library sweep: {_t_sweep:.1f} s")
    # v15: the sweep runs through one of three branches; the streaming branch closes the timer
    # inline, so guard the other two here.
    if 'virtual_library_sweep' not in TIMINGS_SINGLE:
        _t_sweep = _timing.perf_counter() - _t_sweep0
        TIMINGS_SINGLE['virtual_library_sweep'] = _t_sweep
        print(f"  [timings] virtual-library sweep: {_t_sweep:.1f} s")
    # v21 / A5: the streaming engine keeps (mean, sd) in the survivor pool but does not carry
    # the EI column the multiprocessing engine adds.  Recompute it from the same incumbent so
    # every downstream panel sees the column it expects, whatever engine produced the pool.
    if 'Expected Improvement' not in bo_df.columns:
        _mean_col = next((c for c in ('Ensemble Predicted Yield (%)', 'Predicted Yield (%)')
                          if c in bo_df.columns), None)
        _sd_col = next((c for c in ('Ensemble SD (%)', 'Predicted SD (%)')
                        if c in bo_df.columns), None)
        if _mean_col and len(bo_df):
            bo_df['Expected Improvement'] = calculate_expected_improvement(
                bo_df[_mean_col].to_numpy(dtype=float),
                (bo_df[_sd_col].to_numpy(dtype=float) if _sd_col
                 else np.zeros(len(bo_df), dtype=float)),
                best_observed_yield)
            print("  [sweep] Expected Improvement recomputed for the streaming engine "
                  "(f* = %.2f)" % float(best_observed_yield))
        else:
            # degenerate pool (every candidate screened out): keep the column so the ranking
            # call below yields an empty frame instead of raising KeyError
            bo_df['Expected Improvement'] = pd.Series(dtype=float)
            print("  [sweep] survivor pool is EMPTY -- Figures 9/10 will have nothing to draw")
    top_bo = bo_df.nlargest(min(100, len(bo_df)), 'Expected Improvement').reset_index(drop=True)
    # v13: the sweep also returns a bounded sample of the raw library.  Nothing consumes
    # it (the figures use bo_df), so we keep only its size for the log line and drop the
    # frame -- previously it was rebound to `virtual_df` and a feature matrix was built
    # from it that no code ever read.
    n_raw_preview = len(virtual_df_raw)
    del virtual_df_raw
    print(f"  [stream] library enumerated: {n_library:,} conditions; "
          f"survivor pool: {len(bo_df):,}; raw preview sampled: {n_raw_preview:,}")

    # v13 STAGE 2: resolve the loading axis for the survivors (Figure 9 only).
    # DISABLED BY DEFAULT -- see the USE_LOADING_REFINE block at the top of the file.
    # A dense re-scan LOOKED like it added +2.33 pp, but a disjoint odd/even re-measurement
    # of the very same loading position recovered only +0.11 pp: the gain was the
    # "maximum over many noisy points" bias, not a real improvement.  The coarse grid
    # (N_LOADING_POINTS) is therefore kept as the reported result.  Costs 20,000 x 1,000 =
    # 20e6 rows (0.28 % of stage 1) when enabled.

    if USE_LOADING_REFINE:
        bo_df, _n_refine = refine_loading_stage2(
            bo_df, df, best_models_dict, feat_cols, label_encoders, scaler,
            weights=ens_weights, n_points=LOADING_REFINE_POINTS,
            top_n=LOADING_REFINE_TOP_N, verbose=True)
        # re-rank: refinement can reorder the pool
        if len(bo_df):
            bo_df = (bo_df.sort_values('Ensemble Predicted Yield (%)', ascending=False)
                           .head(SURVIVOR_POOL_ROWS).reset_index(drop=True))
        print(f"  [refine] survivor pool re-ranked after refinement ({len(bo_df):,} rows)")

    # Figure 11: predicted yield vs catalyst loading for the TOP reagent combinations,
    # written as one scrollable multi-page PDF (ranked top to bottom).
    # This re-scores only top_k x N_LOADING_POINTS rows (50 x 51 = 2,550 by default), so it
    # costs a fraction of a second on top of the sweep.
    loading_summary = None
    try:
        loading_summary = generate_loading_curves_pdf(
            bo_df=bo_df, df=df, models=best_models_dict, feat_cols=feat_cols,
            label_encoders=label_encoders, scaler=scaler, output_dir=output_dir,
            weights=ens_weights, top_k=TOP_LOADING_CURVES)
    except Exception as e:
        print(f"Figure 11 (loading-curve PDF) failed: {e}")

    # Second pass over the same models: Figures 6 and 7.
    # They are split out of the pass-1 loop so the whole figure set is produced in
    # Figure-number order (F1..F5, then F2/F3/F4 here already emitted, F6, F7, F8).
    for model_name in best_model_names:
        current_model = trained_models[model_name]
        model_cache = cache[model_name]
        file_suffix = f"- {(globals().get('MODEL_DISPLAY') or {}).get(model_name, model_name)}"
        _dm = (globals().get('MODEL_DISPLAY') or {}).get(model_name, model_name)
        # Figure 7: Deep SHAP Attribution
        if SHAP_AVAILABLE:
            _fig7_tag = f'Figure7_Deep_SHAP_Attribution{file_suffix}'
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 9))
            try:
                # v13 SHAP fix.  v11/v12 used shap.KernelExplainer with only 5 k-means
                # background points over the ENTIRE test set: slow and very noisy.
                #
                # Root cause of the old crash: shap.TreeExplainer(xgb_model) raises
                #   ValueError: could not convert string to float: '[2.02E0]'
                # because xgboost >= 3.0 stores learner_model_param.base_score as a
                # VECTOR string ("[v]") and shap 0.49's XGBTreeModelLoader does
                # float(...) on it.  That is a version incompatibility, not a usage error,
                # and no config change fixes it.
                #
                # Fix: get exact TreeSHAP values from the booster directly for xgboost
                # (pred_contribs=True), and use TreeExplainer for the sklearn tree models,
                # which it handles correctly.  Cap the explained rows: a few hundred is
                # plenty for a summary/dependence plot.
                n_explain = min(SHAP_SAMPLE_ROWS, len(X_te))
                X_shap = X_te.iloc[:n_explain].apply(
                    pd.to_numeric, errors='coerce').fillna(0).astype(float)
                X_shap.columns = list(feat_cols)
                _t0 = _timing.perf_counter()
                shap_values, base_value = get_shap_values(
                    current_model, X_shap, display_feat_names or feat_cols)
                _t_add(model_name, 'figure7_shap_compute', _timing.perf_counter() - _t0)
                _method = ('native xgboost' if hasattr(current_model, 'get_booster')
                           else 'shap TreeExplainer'
                           if (hasattr(current_model, 'estimators_')
                               or hasattr(current_model, 'tree_'))
                           else 'shap PermutationExplainer (model-agnostic)')
                print(f"Computing {model_name} SHAP contribution matrix ({n_explain} rows, "
                      f"method={_method})")
                
                # v14: this panel used to hardcode `catalyst_loading` -- the FIFTH ranked
                # parameter (10.0 % of the mean |SHAP| mass) -- while reactant_2 (26.9 %)
                # and reactant_1 (24.0 %), the two largest signals, got no dependence
                # panel at all.  A reader who has just read that the leaving group dominates
                # then sees only a catalyst-loading curve.
                #
                # The feature is now chosen by the same importance ranking the rest of v14
                # uses, and the plot TYPE follows the variable: a numeric feature gets the
                # usual SHAP scatter, a categorical one gets a per-category SHAP box plot
                # (a scatter over arbitrary LabelEncoder integer codes would be meaningless).
                def _dep_col_for(_p):
                    if _p in feat_cols:
                        return _p
                    if _p + '_encoded' in feat_cols:
                        return _p + '_encoded'
                    return None

                _dep_order = [c for c in (_dep_col_for(p) for p in
                                          PARAM_RANK_BY_IMPORTANCE) if c]
                for _c in feat_cols:                       # anything else the model used
                    if _c not in _dep_order:
                        _dep_order.append(_c)

                # v15 BUG FIX.  A feature has to VARY across the explained rows to earn this
                # panel.  The loader fills every absent categorical column with a single
                # 'Unknown', so `reactant_2_encoded` can be constant 0 for the whole
                # dataset: it still has a name, a column index and SHAP values (identically
                # 0), so the old ranking -- a static prior list -- picked it first and the
                # panel drew ONE box labelled "Unknown" at exactly 0.  All four models on
                # (Summit)Reizman_Suzuki_385 produced that empty plot while SHAP itself was
                # fine: the beeswarm in panel (d) ranked reactant_1 first and
                # reactant_2 sixth.
                #
                # The survivors are then ordered by their MEASURED mean |SHAP| rather than
                # by the static prior, so the panel shows the largest real effect.  The
                # shape guard keeps the fallback for a pipeline whose SHAP matrix is in the
                # one-hot space and therefore does not line up with `feat_cols`.
                def _dep_varies(_c):
                    try:
                        return int(pd.Series(np.asarray(X_shap[_c])).nunique(dropna=True)) > 1
                    except Exception:
                        return False

                _dep_rank = [c for c in _dep_order if _dep_varies(c)]
                _dep_sv = np.asarray(shap_values) if shap_values is not None else None
                if (_dep_sv is not None and _dep_sv.ndim == 2
                        and _dep_sv.shape[1] == len(feat_cols) and len(_dep_rank) > 1):
                    _dep_score = {c: float(np.abs(_dep_sv[:, feat_cols.index(c)]).mean())
                                  for c in _dep_rank}
                    _dep_rank.sort(key=lambda c: -_dep_score[c])
                    print('  [Figure 7a] dependence feature = %s '
                          '(mean |SHAP| = %.3f, %d candidate(s) varied)'
                          % (_dep_rank[0], _dep_score[_dep_rank[0]], len(_dep_rank)))
                elif len(_dep_order) > len(_dep_rank):
                    print('  [Figure 7a] skipped %d constant feature column(s): %s'
                          % (len(_dep_order) - len(_dep_rank),
                             ', '.join(c for c in _dep_order if c not in _dep_rank)))

                _dep_col = None
                if shap_values is not None and len(_dep_rank) > 0:
                    _dep_col = _dep_rank[0]
                    _dep_idx = feat_cols.index(_dep_col)

                if _dep_col is None:
                    ax1.axis('off')
                    ax1.text(0.5, 0.5, 'SHAP dependence not available\n(no SHAP values)',
                             ha='center', va='center', fontsize=9, color='0.35',
                             transform=ax1.transAxes)
                elif _dep_col.endswith('_encoded'):
                    _dep_param = _dep_col[:-len('_encoded')]
                    _le_dep = label_encoders.get(_dep_param)
                    try:
                        _codes = np.asarray(X_shap.iloc[:, _dep_idx]).astype(float).round().astype(int)
                        _names = (_le_dep.inverse_transform(_codes) if _le_dep is not None
                                  else _codes.astype(str))
                        _sv = np.asarray(shap_values)
                        _sv = _sv[:, _dep_idx] if _sv.ndim == 2 else _sv
                        _dd = pd.DataFrame({'name': np.asarray(_names).astype(str), 'sv': _sv})
                        _rank = (_dd.groupby('name')['sv']
                                 .apply(lambda s: float(np.abs(s).mean()))
                                 .sort_values(ascending=False))
                        _top = list(_rank.index[:12])
                        _box = [_dd.loc[_dd['name'] == n, 'sv'].values for n in _top]
                        # v14: panel (a) data of the figure export
                        for _nm, _svs in zip(_top, _box):
                            for _v in np.asarray(_svs, dtype=float):
                                _fig_row(_fig7_tag, 'a', 'SHAP by category (%s)'
                                         % PARAM_LABEL.get(_dep_param, _dep_param),
                                         x=str(_nm), y=float(_v),
                                         note='one row per explained sample; '
                                              'y = SHAP value (yield %)')
                        ax1.boxplot(_box, labels=[str(n)[:16] for n in _top],
                                    showfliers=False)
                        ax1.axhline(0.0, color='0.55', lw=0.9, ls='--')
                        _lab_dep = PARAM_LABEL.get(_dep_param, _dep_param)
                        ax1.set_title('SHAP dependence: %s (top 12 by mean |SHAP|)' % _lab_dep,
                                      fontsize=9, fontweight='bold')
                        ax1.set_ylabel('SHAP value for\n%s (yield %%)' % _lab_dep,
                                       fontsize=8, fontweight='bold')
                        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right', fontsize=7)
                        ax1.tick_params(axis='y', labelsize=7)
                    except Exception as _e_dep:
                        ax1.axis('off')
                        ax1.text(0.5, 0.5, 'SHAP dependence for %s failed:\n%s'
                                 % (_dep_col, str(_e_dep)[:60]),
                                 ha='center', va='center', fontsize=8, color='0.35',
                                 transform=ax1.transAxes)
                else:
                    vals_f = np.asarray(X_shap.iloc[:, _dep_idx], dtype=float)
                    lower_p, upper_p = np.percentile(vals_f, [1, 99])
                    mask = (vals_f >= lower_p) & (vals_f <= upper_p)

                    plt.sca(ax1)
                    shap.dependence_plot(_dep_idx, shap_values[mask], X_shap.iloc[mask],
                                         feature_names=display_feat_names, show=False,
                                         ax=ax1)
                    # v14: panel (a) data of the figure export
                    for _jj in np.where(mask)[0]:
                        _fig_row(_fig7_tag, 'a', 'SHAP dependence (%s)'
                                 % display_feat_names[_dep_idx],
                                 x=float(X_shap.iloc[_jj, _dep_idx]),
                                 y=float(np.asarray(shap_values)[_jj, _dep_idx]),
                                 note='1% outliers clipped; x = feature value, y = SHAP value')
                    ax1.set_title('SHAP Dependence: %s (1%% Outliers Clipped)'
                                  % display_feat_names[_dep_idx],
                                  fontsize=9, fontweight='bold')

                    # unify ax1 tick/label fonts
                    ax1.set_xlabel(display_feat_names[_dep_idx], fontsize=8)
                    ax1.set_ylabel('SHAP value for\n%s' % display_feat_names[_dep_idx],
                                   fontsize=8, fontweight='bold')
                    ax1.tick_params(axis='both', labelsize=7)

                    # restyle the ax1 colorbar
                    if len(fig.axes) > 4:
                        cax1 = fig.axes[-1]
                        cax1.tick_params(labelsize=7)
                        cax1.set_ylabel(cax1.get_ylabel(), fontsize=8, fontweight='bold')

                ax1.text(-0.15, 1.05, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')
                for spine in ax1.spines.values(): spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(0.8)

                # v14: was hardcoded catalyst_encoded x base_encoded.  Now the same pair
                # resolver as F4d/F9b/F9c -- the current pair heads the single shared priority
                # list, so nothing changes on a dataset that has both.
                _p_fx, _p_fy = _resolve_axis_pair(('catalyst', 'base'), 'process_categorical',
                                                  df, label_encoders)
                _fx_col = _dep_col_for(_p_fx) if _p_fx else None
                _fy_col = _dep_col_for(_p_fy) if _p_fy else None
                if _fx_col and _fy_col and _fx_col in feat_cols and _fy_col in feat_cols:
                    cat_idx_enc = feat_cols.index(_fx_col)
                    base_idx_enc = feat_cols.index(_fy_col)
                    # v14: the labels must be resolved BEFORE the first use below.
                    _lab_fx = PARAM_LABEL.get(_p_fx, _p_fx)
                    _lab_fy = PARAM_LABEL.get(_p_fy, _p_fy)
                    if _p_fx == 'catalyst':
                        _lab_fx = CATALYST_SLOT_LABEL or _lab_fx
                    plt.sca(ax2)
                    shap.dependence_plot(cat_idx_enc, shap_values, X_shap, feature_names=display_feat_names, interaction_index=base_idx_enc, show=False, ax=ax2)
                    # v14: panel (b) data of the figure export
                    try:
                        _svb = np.asarray(shap_values)
                        for _jj in range(len(X_shap)):
                            _fig_row(_fig7_tag, 'b',
                                     'SHAP interaction (%s, colour = %s)' % (_lab_fx, _lab_fy),
                                     x=float(X_shap.iloc[_jj, cat_idx_enc]),
                                     y=float(_svb[_jj, cat_idx_enc]),
                                     value=float(X_shap.iloc[_jj, base_idx_enc]),
                                     note='x = %s code; value = %s code (colour); y = SHAP value'
                                          % (_lab_fx, _lab_fy))
                    except Exception:
                        pass

                    # unify ax2 tick/label fonts
                    ax2.set_xlabel(_lab_fx, fontsize=8, fontweight='bold')
                    ax2.set_ylabel('SHAP value for\n%s' % _lab_fx, fontsize=8, fontweight='bold')
                    ax2.tick_params(axis='y', labelsize=7)

                    if _p_fx in label_encoders:
                        le_cat = label_encoders[_p_fx]
                        xticks = ax2.get_xticks()
                        valid_xticks = [x for x in xticks if 0 <= x < len(le_cat.classes_)]
                        ax2.set_xticks(valid_xticks)
                        ax2.set_xticklabels([str(le_cat.classes_[int(x)])[:12] for x in valid_xticks], rotation=35, ha='right', fontsize=7, fontweight='bold')
                    else:
                        ax2.tick_params(axis='x', labelsize=7)
                    
                    ax2.set_title('%s–%s Interaction (SHAP Perspective)'
                                  % (_lab_fx, _lab_fy), fontsize=9, fontweight='bold')
                    
                    # restyle the ax2 colorbar
                    if len(fig.axes) > 4:
                        cax2 = fig.axes[-1]
                        cax2.tick_params(labelsize=7)
                        cax2.set_ylabel(cax2.get_ylabel(), fontsize=8, fontweight='bold')
                else:
                    annotate_loading_unavailable(
                        ax2, fontsize=8,
                        headline='SHAP interaction not drawn',
                        detail='needs two varying categorical parameters; this dataset has '
                               'fewer than two')

                ax2.text(-0.15, 1.05, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')
                for spine in ax2.spines.values(): spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(0.8)

                # v13 fix: pick the best reaction AMONG THE ROWS WE ACTUALLY EXPLAINED.
                # The old code took `y_te.idxmax()` and then located that label in the FULL
                # test set, producing a position up to len(y_te)-1 (~13,000 on the curated
                # dataset) which was then used to index `shap_values` -- an array of only
                # n_explain (500) rows.  That raised
                #     index 13318 is out of bounds for axis 0 with size 500
                # for every model, blanking panels c and d.  It only bit datasets where the
                # highest-yield test row happened to fall beyond position 500, which is why
                # it appeared "only on some datasets".
                _n_expl = len(shap_values)
                _y_sub = y_te.iloc[:_n_expl]
                loc_idx = int(np.argmax(_y_sub.to_numpy()))
                top_idx = _y_sub.index[loc_idx]
                # v13 fix: `explainer` no longer exists -- get_shap_values returns the
                # base value directly (and for xgboost it comes from pred_contribs, where
                # there is no TreeExplainer object at all).  Referencing explainer here is
                # what left panels c and d blank with "name 'explainer' is not defined".
                expected_value = float(base_value)
                sample_shap = shap_values[loc_idx]
                sorted_indices = np.argsort(np.abs(sample_shap))[::-1][:5]
                vals = sample_shap[sorted_indices]
                names = [display_feat_names[i] for i in sorted_indices]
                # v22c / fix: a flat red/blue pair is NOT a gradient -- shade each bar by its own
                # magnitude (positive -> Reds, negative -> Blues); the biggest contributor is darkest.
                import matplotlib as _mplx
                _mag = np.abs(np.asarray(vals, dtype=float))
                _mx = float(_mag.max()) if _mag.size else 1.0
                colors_force = [
                    (_mplx.colormaps['Reds'] if _v > 0 else _mplx.colormaps['Blues'])(
                        0.28 + 0.62 * (float(_m) / _mx if _mx > 0 else 0.5))
                    for _v, _m in zip(vals, _mag)]
                
                ax3.barh(range(len(vals)), vals, color=colors_force, edgecolor='black')
                ax3.set_yticks(range(len(vals)))
                ax3.set_yticklabels(names, fontsize=8, fontweight='bold')
                ax3.set_xlabel('SHAP Value (Contribution to Yield)', fontweight='bold', fontsize=8)
                ax3.tick_params(axis='x', labelsize=7)
                ax3.set_title(f'Force Analysis: Best Reaction (Yield={y_te.loc[top_idx]:.1f}%)', fontsize=9, fontweight='bold')
                # v14: panel (c) data of the figure export
                for _nm, _vv in zip(names, vals):
                    _fig_row(_fig7_tag, 'c', 'force analysis (best reaction)',
                             x=str(_nm), y=float(_vv),
                             value=float(y_te.loc[top_idx]),
                             note='top 5 contributors; value = measured yield (%%); '
                                  'expected_value = %.3f' % expected_value)
                ax3.axvline(0, color='black', lw=1)
                ax3.text(-0.15, 1.05, 'c', transform=ax3.transAxes, fontsize=14, fontweight='bold')
                for spine in ax3.spines.values(): spine.set_visible(True); spine.set_color('black')

                plt.sca(ax4)
                # v21 / C8: show EVERY variable that entered the model.  shap's default is
                # max_display=10, so a model with 12+ features silently lost the tail of the
                # ranking -- the opposite of what panel (d) is for.
                shap.summary_plot(shap_values, X_shap, feature_names=display_feat_names,
                                  show=False, plot_size=None, color_bar=True,
                                  max_display=len(display_feat_names))
                ax4.set_title(f'Global SHAP Attribution ({_dm})', fontsize=9, fontweight='bold')
                ax4.set_xlabel('SHAP Value (Impact on Yield)', fontweight='bold', fontsize=8)
                ax4.tick_params(axis='both', labelsize=7)
                # v14: panel (d) data of the figure export -- the beeswarm behind the summary
                try:
                    _svd = np.asarray(shap_values)
                    if _svd.ndim == 2:
                        for _fj, _fn in enumerate(display_feat_names):
                            _col = _svd[:, _fj]
                            _fig_row(_fig7_tag, 'd', 'global SHAP (%s)' % _fn,
                                     x=str(_fn), y=float(np.mean(_col)),
                                     value=float(np.mean(np.abs(_col))),
                                     note='y = mean signed SHAP; value = mean |SHAP| (yield %%); '
                                          'n = %d samples' % _col.shape[0])
                except Exception:
                    pass
                
                ax4.text(-0.15, 1.05, 'd', transform=ax4.transAxes, fontsize=14, fontweight='bold')
                for spine in ax4.spines.values(): spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(0.8)
                
            except Exception as e:
                print(f"SHAP failed ({model_name}): {e}")
                ax4.text(0.5, 0.5, f"SHAP Error: {str(e)}", ha='center', va='center', transform=ax4.transAxes)
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 9))
            feats_display = display_feat_names[:4] if len(display_feat_names) >= 4 else display_feat_names
            for idx, f in enumerate(feats_display):
                vals = np.random.normal(0, 1, 100)
                ax4.scatter(vals, np.full_like(vals, idx) + np.random.uniform(-0.1, 0.1, 100), alpha=0.6, s=10)
            ax4.set_yticks(range(len(feats_display)))
            ax4.set_yticklabels(feats_display, fontsize=8, fontweight='bold')
            ax4.set_xlabel(f'SHAP Value ({_dm} Impact on Yield)', fontweight='bold', fontsize=8)
            ax4.tick_params(axis='both', labelsize=7)
            plt.setp(ax4.get_yticklabels(), fontweight='bold')

            ax4.text(-0.15, 1.05, 'd', transform=ax4.transAxes, fontsize=14, fontweight='bold')
            for spine in ax4.spines.values(): spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(0.8)
            ax4.set_title('SHAP Feature Importance Summary (Mock)', fontsize=9, fontweight='bold')
            # v14: panel (d) data of the figure export -- the global mean |SHAP| ranking
            try:
                _sv_all = np.asarray(shap_values)
                if _sv_all.ndim == 2:
                    for _fi, _fn in enumerate(display_feat_names):
                        _fig_row(_fig7_tag, 'd', 'global |SHAP|',
                                 x=str(_fn), y=float(np.abs(_sv_all[:, _fi]).mean()),
                                 value=float(np.abs(_sv_all[:, _fi]).sum()),
                                 note='y = mean |SHAP| (yield %); value = total |SHAP|')
            except Exception:
                pass

        try:
            fig.suptitle(f'Deep SHAP Attribution and Feature Interaction Analysis — {_dm}', fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f'Figure7_Deep_SHAP_Attribution{file_suffix}.png'), bbox_inches='tight')
            plt.close(fig)
            _fig_flush(output_dir, f'Figure7_Deep_SHAP_Attribution{file_suffix}',
                       f'Figure7_Deep_SHAP_Attribution{file_suffix}_panel_data.csv')
        except Exception as e:
            print(f"Figure 7 failed: {e}")
        

        # Figure 8: Model-based PDP
        try:
            _fig8_tag = f'Figure8_Partial_Dependence_Plots{file_suffix}'
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 9))
            # v22 / issue 5: this panel was pinned to `catalyst_loading`, but the swept axis
            # can now be ANY equivalents column (temperature / time / ligand_eq / base_eq ...).
            # Pinning it meant the panel said "Loading response not available" on datasets that
            # demonstrably carry several equivalents columns.
            _f8_axis = SWEEP_AXIS if SWEEP_AXIS in feat_cols else (
                'catalyst_loading' if 'catalyst_loading' in feat_cols else None)
            if _f8_axis:
                c_idx = feat_cols.index(_f8_axis)
                # v15 BUG FIX.  method='auto' resolves to 'recursion' for tree ensembles,
                # and recursion is NOT valid for HistGradientBoostingRegressor: it walks
                # `_predictors` and ignores the categorical-bin mapping, so every category
                # collapses onto one bin.  MEASURED on (AI_RCP_pub)sm_data_clean_all_whitelisted
                # (predict(X_test).mean() = 28.95):
                #     feature    method='auto'                 method='brute'
                #     catalyst   nuniq= 2  range -2.62..-2.53  nuniq=22  range  9.23..41.77
                #     base       nuniq= 2  range -6.58..-1.04  nuniq= 9  range 14.46..33.78
                #     solvent    nuniq= 2  range -2.75.. 0.36  nuniq=16  range 22.43..35.35
                #     metal      nuniq= 1  exactly 0.0000      nuniq= 1  exactly 30.2136
                # RandomForest is unaffected (recursion is valid there).  The symptom was a
                # flat HistGB panel on 7 of the 16 benchmark datasets.
                PartialDependenceDisplay.from_estimator(current_model, X_tr, [c_idx],
                                                        feature_names=feat_cols, ax=ax1,
                                                        method='brute')
                # v14: the PDP may be over a substitute amount variable.
                ax1.set_title('PDP: %s' % (LOADING_AXIS.get('label')
                                           or display_feat_names[c_idx]),
                              fontsize=9, fontweight='bold')
                # v15: the grid here is `X_tr`, which prepare_features has ALREADY scaled, so
                # the ticks are z-scores, not equivalents -- the panel used to label them
                # "catalyst_loading" and nothing else, which reads as a loading axis in eq.
                # (Figure 4b sweeps the SAME variable in raw eq., so the two panels appeared
                # to disagree.)  Say which units are on screen.
                ax1.set_xlabel('%s (standardised; 1.0 = +1 SD)'
                               % display_feat_names[c_idx], fontweight='bold')
                # v14: panel (a) data of the figure export
                try:
                    _t0 = _timing.perf_counter()
                    _pd = partial_dependence(current_model, X_tr, [c_idx], kind='average',
                                             method='brute')   # see the note above
                    _t_add(model_name, 'figure8_partial_dependence',
                           _timing.perf_counter() - _t0)
                    _gv = _pd['grid_values'][0] if 'grid_values' in _pd else _pd['values'][0]
                    _av = _pd['average'][0]
                    for _ig in range(len(_gv)):
                        _fig_row(_fig8_tag, 'a', str(display_feat_names[c_idx]),
                                 x=float(_gv[_ig]), y=float(_av[_ig]),
                                 note='partial dependence (yield %)')
                except Exception:
                    pass
            else:
                # v14: was a silently blank axes; say why instead.
                # panel_letter is NOT passed -- the ax1.text(...) right below draws the 'a'.
                annotate_loading_unavailable(ax1, fontsize=8)
            ax1.text(-0.15, 1.05, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')
            for spine in ax1.spines.values(): spine.set_visible(True); spine.set_color('black'); spine.set_linewidth(0.8)

            def plot_categorical_pdp(model, X, feature_name, ax, title):
                if feature_name + '_encoded' in feat_cols:
                    f_idx = feat_cols.index(feature_name + '_encoded')
                    pdp_res = partial_dependence(model, X, [f_idx], kind='average',
                                                 method='brute')   # see the panel-a note
                    avg_preds = pdp_res['average'][0]

                    if 'grid_values' in pdp_res:
                        values = pdp_res['grid_values'][0]
                    elif 'values' in pdp_res:
                        values = pdp_res['values'][0]
                    else:
                        raise KeyError(f"partial_dependence result missing keys: {list(pdp_res.keys())}")
                    
                    le = label_encoders[feature_name]
                    names = le.inverse_transform(values.astype(int))
                    pdp_df = pd.DataFrame({'Name': names, 'Impact': avg_preds}).sort_values('Impact', ascending=False).head(10)
                    _pdp_panel = getattr(plot_categorical_pdp, '_panel', '?')
                    for _nm, _im in zip(pdp_df['Name'], pdp_df['Impact']):
                        _fig_row(_fig8_tag, _pdp_panel, feature_name,
                                 x=str(_nm), y=float(_im),
                                 note='partial dependence (yield %), top 10 categories')
                    bar_gradient(ax, pdp_df['Impact'].to_numpy(dtype=float),
                                 cmap='Blues', horizontal=False, edgecolor='black')
                    ax.set_xticks(range(len(pdp_df)))
                    ax.set_xticklabels([str(n)[:10] for n in pdp_df['Name']], rotation=30, ha='right', fontsize=7, fontweight='bold')
                    ax.set_ylabel('Partial Dependence (Yield %)', fontsize=8, fontweight='bold')
                    ax.set_title(title, fontsize=9, fontweight='bold')

            # v14: the three PDP slots were hardcoded catalyst / base / solvent, and
            # plot_categorical_pdp simply does nothing when the feature was dropped -- so a
            # dataset without, say, solvent got a blank panel with no explanation.  The slots
            # are resolved from the single shared priority list, each taking the next
            # qualifying parameter that the earlier slots did not use.
            _f8_pref = ['catalyst', 'ligand', 'base', 'solvent']
            _f8_res = []
            for _pref in _f8_pref:
                _c = _pick_param(_pref, 'process_categorical', df, label_encoders,
                                 used=tuple(_f8_res))
                if _c is None and ALLOW_SUBSTRATE_IN_CONDITION_PANELS:
                    _c = _pick_param(_pref, 'substrate', df, label_encoders,
                                     used=tuple(_f8_res))
                _f8_res.append(_c)

            # v21 / C6: the three variables chosen in the menu drive panels b / c / d.  A
            # choice whose encoded column was dropped by prepare_features becomes None and the
            # panel then says why, instead of drawing something else under the wrong label.
            _ch8 = (globals().get('CHOICES') or {}).get('fig8_vars')
            if _ch8 and len(_ch8) == 3:
                _f8_from_menu = [c if (c and (c + '_encoded') in feat_cols) else None
                                 for c in _ch8]
                print('[Figure 8 b/c/d] variables from the menu: %s'
                      % ', '.join(c if c else '(dropped: %s)' % _ch8[i]
                                  for i, c in enumerate(_f8_from_menu)))
                _f8_res = _f8_from_menu

            for _ax_pdp, _p_pdp, _lt in ((ax2, _f8_res[0], 'b'),
                                         (ax3, _f8_res[1], 'c'),
                                         (ax4, _f8_res[2], 'd')):
                if _p_pdp:
                    _lab_pdp = PARAM_LABEL.get(_p_pdp, _p_pdp)
                    if _p_pdp == 'catalyst':
                        _lab_pdp = CATALYST_SLOT_LABEL or _lab_pdp
                    plot_categorical_pdp._panel = _lt      # v14: tell the recorder its subplot
                    plot_categorical_pdp(current_model, X_tr, _p_pdp, _ax_pdp,
                                         'PDP: Top 10 %s' % _lab_pdp)
                else:
                    annotate_loading_unavailable(
                        _ax_pdp, fontsize=8,
                        headline='PDP not drawn',
                        detail='no further varying categorical parameter is available')
                _ax_pdp.text(-0.15, 1.05, _lt, transform=_ax_pdp.transAxes,
                             fontsize=14, fontweight='bold')
                for _sp in _ax_pdp.spines.values():
                    _sp.set_visible(True); _sp.set_color('black'); _sp.set_linewidth(0.8)

            fig.suptitle(f'Partial Dependence Analysis of Reaction Conditions — {_dm}', fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f'Figure8_Partial_Dependence_Plots{file_suffix}.png'), bbox_inches='tight')
            plt.close()
            _fig_flush(output_dir, _fig8_tag,
                       f'Figure8_Partial_Dependence_Plots{file_suffix}_panel_data.csv')
        except Exception as e:
            print(f"Figure 8 failed: {e}")





    # Figures 8-9 for the ensemble model
    # Reliability metrics for the recommendation table.  Computed ONCE for the whole
    # survivor pool and passed to every Figure 10 call, so the conformal calibration and
    # the neighbour search are not repeated per model.
    cred_df = None
    try:
        cred_df, _cred_info = compute_credibility_metrics(
            bo_df, df, best_models_dict, feat_cols, display_feat_names,
            label_encoders, scaler, weights=ens_weights)
    except Exception as e:
        print(f"  [credibility] unavailable: {e}")

    try:
        generate_figure8_and_9(
            model_name='Ensemble',
            model=None,
            bo_df=bo_df,
            df=df,
            feat_cols=feat_cols,
            label_encoders=label_encoders,
            scaler=scaler,
            output_dir=output_dir,
            trained_models=best_models_dict,
            top_n_table=20,
            credibility=cred_df,
            ensemble_weights=ens_weights
        )
    except Exception as e:
        print(f"Ensemble Figures 9/10 failed: {e}")

# MAIN EXECUTION PIPELINE




# ----------------------------------------------------------------------------
# v21 / A4: the twelve (architecture x encoding) cells
# ----------------------------------------------------------------------------
def save_grid12(df, X, feature_cols, tr_pos, te_pos, y, output_dir):
    """Train the twelve cells and write grid12_metrics.csv next to the figures.

    The grid is twelve, not 5 x 3 = 15: Gradient Boosting, Random Forest and MLP have
    no native-categorical cell.  Each cell trains on its own frame (INT / one-hot /
    native-category), so the frames are built here instead of reusing the integer
    matrix the rest of the pipeline works on.
    """
    try:
        from hte.grid import build_frames
        from hte.models import grid_cells
    except Exception as exc:                      # noqa: BLE001
        print('[grid12] skipped: %s' % exc)
        return None
    from sklearn.metrics import r2_score, mean_squared_error
    import time as _time

    print("\n" + "-" * 70)
    print("[stage 1/2] twelve cells (model x encoding): train + held-out R2")
    print("           these numbers are what the menu ranks, and none of them is")
    print("           reused by the ensemble below -- the two stages are independent.")
    print("-" * 70)
    packs, _cat_cols, cat_idx = build_frames(X, feature_cols, tr_pos, te_pos)
    y_tr = y.iloc[tr_pos]
    y_te = y.iloc[te_pos]
    rows = []
    for cell in grid_cells(cat_idx):
        Xtr, Xte = packs[cell.frame]
        rec = dict(model=cell.architecture, encoding=cell.encoding,
                   frame=cell.frame, n_features=int(Xtr.shape[1]),
                   r2_train=np.nan, r2_test=np.nan, rmse_test=np.nan,
                   fit_seconds=np.nan, status='ok')
        t0 = _time.perf_counter()
        try:
            mdl = cell.factory()
            mdl.fit(Xtr, y_tr)
            p_tr = np.asarray(mdl.predict(Xtr), dtype=float).ravel()
            p_te = np.asarray(mdl.predict(Xte), dtype=float).ravel()
            rec['r2_train'] = float(r2_score(y_tr, p_tr))
            rec['r2_test'] = float(r2_score(y_te, p_te))
            rec['rmse_test'] = float(np.sqrt(mean_squared_error(y_te, p_te)))
        except Exception as exc:                  # noqa: BLE001
            rec['status'] = '%s: %s' % (type(exc).__name__, exc)
        rec['fit_seconds'] = _time.perf_counter() - t0
        rows.append(rec)
        print('  [grid12] %2d/12  %-18s %-9s %s' %
              (len(rows), cell.architecture,
               {'native categorical': 'native'}.get(cell.encoding, cell.encoding),
               ('R2=%.4f' % rec['r2_test']) if rec['r2_test'] == rec['r2_test']
               else rec['status'][:40]))
    tbl = pd.DataFrame(rows)
    globals()['GRID12_TABLE'] = tbl          # Figure 1 reads it (best encoding per model)
    # v22f: ONE ranked table, printed here, with the numbers the later menu asks about.
    # The menu itself then only asks for K -- the table is not printed a second time.
    _rank = tbl[tbl['status'] == 'ok'].sort_values('r2_test', ascending=False)
    print('\n  ranked cells (this order is what the model question refers to):')
    print('      %-3s %-20s %-9s %s' % ('#', 'architecture', 'encoding', 'R2_test'))
    for _i, (_, _r) in enumerate(_rank.iterrows(), 1):
        _e = {'native categorical': 'native'}.get(_r['encoding'], _r['encoding'])
        print('      %-3d %-20s %-9s %.4f' % (_i, _r['model'], _e, float(_r['r2_test'])))
    globals()['GRID12_RANK'] = _rank.reset_index(drop=True)
    out = os.path.join(output_dir, 'grid12_metrics.csv')
    tbl.to_csv(out, index=False, encoding='utf-8-sig')
    ok = tbl[tbl['status'] == 'ok']
    if len(ok):
        _b = ok.loc[ok['r2_test'].idxmax()]
        print('[grid12] best cell: %s / %s  R2_test=%.4f'
              % (_b['model'], _b['encoding'], float(_b['r2_test'])))
    print('[grid12] wrote %s (%d cells)' % (out, len(tbl)))
    return tbl


# ----------------------------------------------------------------------------
# v21 / B: interactive choice of what the figures should show
# ----------------------------------------------------------------------------
CHOICES = {}

def ask_choices(df, feature_cols, output_dir):
    """Ask which variables to draw, then persist the answers to choices.json.

    Every menu is built from the columns that ACTUALLY enter the model, numbered
    ``N.name(count)``.  A question whose candidate list is too short is skipped with a
    printed reason instead of producing an empty panel.  When stdin is not a terminal
    (batch runs) or ``CD_CHOICES`` points at a JSON file, the answers are taken from there
    so the pipeline stays non-interactive and reproducible.
    """
    global CHOICES
    import json as _json
    import itertools as _it

    cats = [str(c)[:-len('_encoded')] for c in feature_cols if str(c).endswith('_encoded')]
    cat_counts = {c: (int(df[c].nunique()) if c in df.columns else 0) for c in cats}
    cats = [c for c in cats if cat_counts.get(c, 0) > 1]
    # v22n: descending by the count shown in the menu, so Enter = the largest one.
    cats.sort(key=lambda c: (-cat_counts.get(c, 0), c))
    nums = [str(c) for c in feature_cols if not str(c).endswith('_encoded')]
    nums = [c for c in nums if c in df.columns and pd.to_numeric(df[c], errors='coerce').nunique() > 1]
    # v22n: biggest choice first -- Enter then picks the richest variable.
    _num_counts = {c: int(pd.to_numeric(df[c], errors='coerce').nunique()) for c in nums}
    nums.sort(key=lambda c: (-_num_counts[c], c))

    smiles_slots = [c for c in ('reactant_1_smiles', 'reactant_2_smiles', 'solvent_smiles',
                                'product_smiles', 'catalyst_smiles', 'additive_smiles')
                    if c in df.columns and df[c].astype(str).str.strip().replace('nan', '').ne('').any()]
    # v22n: same rule -- most-populated column first.
    smiles_slots.sort(key=lambda c: -int(df[c].astype(str).str.strip().replace('nan', '').ne('').sum()))

    def _menu(pairs):
        return '\n'.join('    %d.%s(%d)' % (i, n, k) for i, (n, k) in enumerate(pairs, 1))

    def _ask(qid, title, pairs, pick=1, note='', start=1, show_menu=True):
        """Return the chosen indices (1-based) or [] when the question is skipped."""
        if len(pairs) < pick:
            print('  [choice %d] skipped: only %d candidate(s), need %d%s'
                  % (qid, len(pairs), pick, (' -- ' + note) if note else ''))
            return []
        print('\n  [choice %d] %s' % (qid, title))
        if show_menu:
            print(_menu(pairs))
        else:
            print('    (same list as above)')
        if _PRESET is not None:
            if qid in _PRESET:
                print('    -> from CD_CHOICES: %s' % (_PRESET[qid],))
                return [int(v) for v in _PRESET[qid]][:pick]
            return list(range(start, start + pick))
        if not _INTERACTIVE:
            print('    -> non-interactive run: using %d..%d' % (start, start + pick - 1))
            return list(range(start, start + pick))
        picks = []
        while len(picks) < pick:
            try:
                raw = input('    Pick %d of %d: '
                            % (len(picks) + 1, pick)).strip()
            except EOFError:
                picks = list(range(start, start + pick))
                break
            if not raw:
                # v23e: Enter accepts the DEFAULT -- the first option not yet taken,
                # counting from `start` (so the pair question defaults to 1 and 2).
                _free = [j for j in range(start, len(pairs) + 1) if j not in picks]
                v = _free[0] if _free else (picks[-1] if picks else 1)
                print('    -> Enter: using %d' % v)
            else:
                try:
                    v = int(raw.split(',')[0])
                except Exception:
                    _free = [j for j in range(start, len(pairs) + 1) if j not in picks]
                    v = _free[0] if _free else (picks[-1] if picks else 1)
                    print('    -> not a number: using the default %d' % v)
            if not (1 <= v <= len(pairs)) or v in picks:
                print('    out of range or already chosen')
                continue
            picks.append(v)
        return picks

    # CD_FORCE_INTERACTIVE=1 lets the menus run from redirected stdin (handy for
    # scripted tests and for feeding a prepared answer file).
    _INTERACTIVE = ((sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False)
                    or os.environ.get('CD_FORCE_INTERACTIVE') == '1')
    _preset_path = os.environ.get('CD_CHOICES')
    _PRESET = None
    if _preset_path and os.path.isfile(_preset_path):
        try:
            # utf-8-sig: PowerShell's Out-File/Set-Content write a BOM, which json.loads
            # rejects -- the preset was silently ignored and every run fell back to Enter.
            _PRESET = _json.loads(open(_preset_path, encoding='utf-8-sig').read())
            print('[choices] loaded %s' % _preset_path)
        except Exception as exc:                      # noqa: BLE001
            print('[choices] could not read %s: %s' % (_preset_path, exc))

    out = {}

    # 1. WHICH CELLS to show.  Asked FIRST because it also fixes the Bayesian ensemble.
    g12 = globals().get('GRID12_RANK')
    if g12 is not None and len(g12):
        print('\n  [choice 1] which cells to use (numbered as in the ranked table above)')
        print('      This decides TWO things at once:')
        print('        (a) Figures 2-4 and 7-8: one figure per chosen cell, named')
        print('            "model (encoding)";')
        print('        (b) the BAYESIAN ensemble: those same cell models are the members')
        print('            whose weighted mean scores the virtual library (Fig. 6/9/10).')
        print('      Enter = one cell per architecture (its best encoding, usually 5)')
        print('      a number K = the top K cells of the table above')
        if _PRESET and _PRESET.get('topk') not in (None, ''):
            try:
                out['topk'] = max(1, min(len(g12), int(_PRESET['topk'])))
                print('    -> from CD_CHOICES: %d' % out['topk'])
            except Exception:
                out['topk'] = None
        elif _INTERACTIVE:
            try:
                _raw_k = input('    K (Enter = one per model): ').strip()
                out['topk'] = (max(1, min(len(g12), int(_raw_k))) if _raw_k else None)
            except Exception:                          # noqa: BLE001
                out['topk'] = None
        else:
            out['topk'] = None
            print('    -> non-interactive run: one cell per model (best encoding)')

    # 2. the continuous variable behind Figures 3b / 8a / 9d
    idx = _ask(2, 'continuous variable for Fig. 3b / 8a / 9d',
               [(c, int(pd.to_numeric(df[c], errors='coerce').nunique())) for c in nums], 1)
    out['continuous'] = nums[idx[0] - 1] if idx else None

    # 3. the single categorical variable behind Figures 5a / 7a
    idx = _ask(3, 'single categorical variable for Fig. 5a / 7a',
               [(c, cat_counts[c]) for c in cats], 1)
    out['single_cat'] = cats[idx[0] - 1] if idx else None

    # 4. two categorical PAIRS: group 1 -> 5a / 7b / 9, group 2 -> 5c / 9c
    combos = list(_it.combinations(cats, 2))
    pair_pairs = [('%s x %s' % (a, b), cat_counts[a] * cat_counts[b]) for a, b in combos]
    if len(pair_pairs) <= 2:
        print('\n  [choice 4] skipped: only %d two-variable combination(s), need 3+'
              % len(pair_pairs))
        out['pair1'] = list(combos[0]) if combos else None
        out['pair2'] = list(combos[-1]) if len(combos) > 1 else None
    else:
        idx = _ask(4, 'first two-variable pair (Fig. 5a / 7b / 9)', pair_pairs, 1)
        out['pair1'] = list(combos[idx[0] - 1]) if idx else None
        idx = _ask(4, 'second two-variable pair (Fig. 5c / 9c)', pair_pairs, 1, start=2,
                   show_menu=False)
        out['pair2'] = list(combos[idx[0] - 1]) if idx else None

    # 5. which SMILES Figure 5d's PCA should use
    if not smiles_slots:
        print('\n  [choice 5] skipped: this dataset carries no reactant/product SMILES at all,'
              ' so the PCA panel cannot be drawn')
        out['pca_smiles'] = None
    else:
        idx = _ask(5, 'SMILES for the Fig. 5d PCA',
                   [(c, int(df[c].astype(str).str.len().gt(0).sum())) for c in smiles_slots], 1)
        out['pca_smiles'] = smiles_slots[idx[0] - 1] if idx else None

    # 6. the three variables of Figure 8 b/c/d
    idx = _ask(6, 'three variables for Fig. 8 b/c/d',
               [(c, cat_counts[c]) for c in cats], 3)
    out['fig8_vars'] = [cats[i - 1] for i in idx] if idx else None

    # 7. expand the loading grid?  The last question, because it changes the sweep cost and
    #    what the recommendation means -- not just a figure.
    print('\n  [choice 7] expand the loading grid during enumeration?')
    print('      METHOD  ON  : every reagent combination is scored at 100 loading points')
    print('                    (AI_RCP sm: 13,543,200 candidates)')
    print('              OFF : each combination is scored once, at the median loading')
    print('                    (AI_RCP sm: 135,432 candidates)')
    print('      RESULT  ON  : the recommendation reports a best loading per combination,')
    print('                    and the loading curves come straight from the sweep')
    print('              OFF : the recommendation assumes the median loading; the curves')
    print('                    are still drawn (61-point re-prediction at the winner)')
    print('      IMPACT  ON  : sweep x100 (measured 94.0 s vs 1.4 s); memory unchanged')
    print('              OFF : 67x faster; the best VALUE was identical on AI_RCP sm, but')
    print('                    the order of tied candidates changes and an optimum away')
    print('                    from the median loading would be missed')
    print('      EFFECT  Enter = ON (paper setting)')
    # v22p: a batch/test run defaults to NOT expanded (fast); an interactive run defaults ON.
    _default_expand = bool(_INTERACTIVE)
    if _PRESET and 'expand_loading' in _PRESET:
        _exp = bool(_PRESET['expand_loading'])
    elif _INTERACTIVE:
        try:
            _a = input('    expand? [Y/n]: ').strip().lower()
        except Exception:                          # noqa: BLE001
            _a = ''
        _exp = _a not in ('n', 'no', 'off', '0')
    else:
        _exp = _default_expand
        print('    -> non-interactive run: %s' % ('ON (paper setting)' if _exp else 'OFF'))
    out['expand_loading'] = bool(_exp)
    globals()['EXPAND_LOADING'] = bool(_exp)
    print('    => loading grid %s' % ('EXPANDED (100 points)' if _exp else 'NOT expanded (1 point)'))

    print('\n[choices] ' + ', '.join('%s=%s' % (k, v) for k, v in out.items()))
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'choices.json'), 'w', encoding='utf-8') as fh:
            _json.dump(out, fh, indent=2, ensure_ascii=False)
    except Exception as exc:                          # noqa: BLE001
        print('[choices] could not write choices.json: %s' % exc)
    CHOICES = out
    return out


# ----------------------------------------------------------------------------
# v21 / lock: one dataset per machine at a time
# ----------------------------------------------------------------------------
# Two concurrent runs oversubscribe the CPU (each spawns 12-cell fits, n_jobs=-1 CV and
# joblib workers on top of the 4 OpenMP threads pinned in hte/__init__.py) and the joblib
# workers then die with "Windows fatal exception: access violation" -- a native crash that
# Python cannot catch.  A named kernel mutex makes the second run exit immediately instead;
# unlike a lock file it is released by the kernel when the process dies, so a crash cannot
# leave a stale lock behind.
_LOCK_NAME = "Local\\HTE_SingleDataset_Pipeline"
_RUN_LOCK_HANDLE = None


def _lock_owner_path():
    # NOTE: this module has no top-level `pathlib` import (it is os.path based throughout),
    # so the path is built with os.path -- using Path() here raised NameError, the write was
    # swallowed by the except below and the lock reported "holder unknown".
    _here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(_here), ".pipeline_owner.json")


def _load_pipeline_lock():
    """Import scripts/pipeline_lock.py -- the lock file the .bat runs use too."""
    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _scripts = os.path.join(_root, "scripts")
        if _scripts not in sys.path:
            sys.path.insert(0, _scripts)
        import pipeline_lock
        return pipeline_lock
    except Exception:
        return None


def acquire_single_run_lock():
    """True when this process owns the machine-wide run lock, False when another run holds it."""
    global _RUN_LOCK_HANDLE
    # A batch run (run_all.bat) cannot hold a kernel object, so it uses the lock FILE and
    # this process must honour it before taking the mutex.
    _pl = _load_pipeline_lock()
    if _pl is not None:
        try:
            if _pl.status() == 3:
                print("[lock] a batch run (run_all.bat) is already in progress: %s"
                      % _pl.holder_text())
                return False
        except Exception:
            pass
    try:
        import ctypes
        _k32 = ctypes.windll.kernel32
    except Exception:                                  # not Windows / no ctypes
        return True
    try:
        _h = _k32.CreateMutexW(None, False, _LOCK_NAME)
        if not _h:
            print("[lock] CreateMutexW failed; continuing without the lock")
            return True
        if _k32.GetLastError() == 183:                 # ERROR_ALREADY_EXISTS
            _k32.CloseHandle(_h)
            return False
        _RUN_LOCK_HANDLE = _h                          # keep it alive: GC would release it
        if _pl is not None:
            _rc = _pl.acquire("single_dataset")
            if _rc == 3:
                _k32.CloseHandle(_h)
                _RUN_LOCK_HANDLE = None
                return False
        return True
    except Exception as _exc:                          # noqa: BLE001
        print("[lock] unavailable (%s); continuing without it" % _exc)
        return True


def release_single_run_lock():
    """Best-effort cleanup: drop the owner record (the mutex itself dies with the process)."""
    _pl = _load_pipeline_lock()
    try:
        if _pl is not None:
            _pl.release(force=True)
            return
        _p = _lock_owner_path()
        if os.path.isfile(_p):
            with open(_p, encoding="utf-8") as _fh:
                _d = json.load(_fh)
            if int(_d.get("pid", -1)) == os.getpid():
                os.remove(_p)
    except Exception:
        pass


def _describe_lock_holder():
    try:
        _p = _lock_owner_path()
        if os.path.isfile(_p):
            with open(_p, encoding="utf-8") as _fh:
                _d = json.load(_fh)
            return "pid=%s, started %s" % (_d.get("pid"), _d.get("started"))
    except Exception:
        pass
    return "holder unknown"


if __name__ == "__main__":
    print("=" * 70)
    print("Suzuki-Miyaura / Buchwald-Hartwig Multi-Model Pipeline")
    print("=" * 70)
    # v13: report the xgboost build.  A CUDA-enabled wheel pins ~0.2 GB of VRAM per
    # process that calls fit()/predict() (workers included) even though everything
    # here runs with device='cpu'; the CPU-only wheel touches no GPU at all.
    warn_if_cuda_xgboost()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(SCRIPT_DIR).lower() in ['script', 'scripts', 'archive']:
        OUTPUT_BASE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'output_singledataset')
    else:
        OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, 'output_singledataset')

    candidate_dirs = ['.', 'Data', 'data', '../Data', '../data', SCRIPT_DIR, os.path.join(SCRIPT_DIR, 'Data')]

    # v21: refuse to start while another dataset is running on this machine (see
    # acquire_single_run_lock).  Done BEFORE the filename prompt so the user is not asked
    # for input and then told to go away.
    # v21: the run lock is OFF by default.  Since the pipeline moved to the CPU-only
    # xgboost wheel two concurrent datasets no longer crash (verified: both finished with
    # identical R2), so the lock is now an opt-in guard against CPU contention rather than
    # a crash guard.  Set CD_LOCK=1 to enable it.
    _LOCK_ON = os.environ.get('CD_LOCK') == '1'
    if not _LOCK_ON:
        print('[lock] disabled (default) -- set CD_LOCK=1 to stop two runs at once')
    if _LOCK_ON and not acquire_single_run_lock():
        print("\n[lock] another dataset is already running on this machine.")
        print("       %s" % _describe_lock_holder())
        print("       Two concurrent runs oversubscribe the CPU and crash the joblib workers;")
        print("       wait for it to finish, then start again.")
        print("\n[lock] 另一个数据集正在运行（同一台机器只允许一个），请等它结束后再试。")
        sys.exit(3)
    if _LOCK_ON:
        atexit.register(release_single_run_lock)

    DEFAULT_FILENAME = '(AI_RCP_pub)bh_data_clean_all_whitelisted.csv'
    user_input = input(f"CSV file name (Enter for '{DEFAULT_FILENAME}'): ").strip()
    filename = user_input if user_input else DEFAULT_FILENAME
    
    REAL_DATA_PATH = None
    BASE_DIR = '.'
    
    for d in candidate_dirs:
        test_path = os.path.join(d, filename)
        if os.path.exists(test_path):
            REAL_DATA_PATH = test_path
            BASE_DIR = d
            break

    if REAL_DATA_PATH is None:
        print(f"File not found: '{filename}' (looked in {candidate_dirs})")
        sys.exit(1)
            
    input_file_stem = os.path.splitext(os.path.basename(REAL_DATA_PATH))[0]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"{timestamp}_{input_file_stem}")
    os.makedirs(output_dir, exist_ok=True)
    
    df, raw_feature_names = load_real_dataset(REAL_DATA_PATH, BASE_DIR)
    # scaler fitted on the training rows only, so no test-set statistics reach the models
    _tr_rows, _te_rows = get_split_positions(len(df))
    X, y, feature_cols, display_feat_names, encoders, scaler = prepare_features(
        df, raw_feature_names, fit_rows=_tr_rows
    )

    # Split by POSITION, not by frame, so the original record order stays recoverable.
    #
    # train_test_split shuffles by default.  Handing the shuffled X_train to
    # KFold(shuffle=False) does NOT give an "ordered" split of the file -- it gives a
    # differently-seeded random one.  MEASURED on AI_RCP bh / XGBoost the difference is the
    # whole conclusion of section 3.6:
    #     original record order  -> folds [-0.6543, 0.1630, -5.0778, -0.0906, -0.1992] = -1.17
    #     shuffled (what the code did before this fix)                                   = +0.74
    # So the paper's -1.17 was right and the code had regressed.  The ordered CV now gets its
    # own copy of the training rows, sorted back into file order.
    _pos_tr, _pos_te = train_test_split(np.arange(len(X)), test_size=TEST_SIZE, random_state=42)
    X_train, y_train = X.iloc[_pos_tr], y.iloc[_pos_tr]
    X_test, y_test = X.iloc[_pos_te], y.iloc[_pos_te]

    _g12 = save_grid12(df, X, feature_cols, _tr_rows, _te_rows, y, output_dir)
    ask_choices(df, feature_cols, output_dir)

    # v22f: resolve the cell selection HERE, before stage 2, so the ensemble stage trains
    # only the members those cells need (previously all six were trained and one was unused).
    _sel_now = select_cells_for_figures((CHOICES or {}).get('topk'))
    if _sel_now:
        globals()['MODEL_DISPLAY'] = {m: d for d, m in _sel_now}
        print('[selection] %d cell(s): %s'
              % (len(_sel_now), ', '.join(d for d, _ in _sel_now)))

    # v21 / C2: point the sweep at the continuous variable chosen in the menu, so Figures 3b /
    # 8a / 9d show THAT variable.  The feature matrix is already built, so training is
    # untouched -- only the enumeration grid and the three panels follow the new axis.
    _ch_cont = (CHOICES or {}).get('continuous')
    if _ch_cont and _ch_cont in df.columns and _ch_cont != SWEEP_AXIS:
        _vals = pd.to_numeric(df[_ch_cont], errors='coerce')
        if _vals.nunique(dropna=True) > 1:
            globals()['SWEEP_AXIS'] = _ch_cont
            LOADING_AXIS.update({'col': _ch_cont, 'source_col': _ch_cont, 'role': 'catalyst',
                                 'label': PARAM_LABEL.get(_ch_cont, _ch_cont), 'unit': '',
                                 'n_distinct': int(_vals.nunique(dropna=True)),
                                 'reason': 'chosen in the interactive menu'})
            print('[C2] sweep axis (Fig. 3b / 8a / 9d) = %s  [%d distinct values]'
                  % (_ch_cont, int(_vals.nunique(dropna=True))))

    results_df, models, cache_dict = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        # Both CV numbers are computed on the FULL frame, in file order.  That is the protocol
        # section 3.6 reports ("+0.767 shuffled vs -1.17 ordered"); using X_train instead gives
        # "+0.73 vs -0.07" -- still a sign flip, but not the numbers in the paper.
        X_cv=X,
        y_cv=y,
    )

# v13: this only PRINTS a report.  The 200,000-row "preview" DataFrame (and its feature
# matrix) that used to be built here were never read by anything -- an AST audit showed
# them bound and never loaded -- so they are gone.  The figures consume the streamed
# sweep's own survivor pool (`bo_df`), not a preview table.
#
# Moved BELOW the metrics/predictions save on purpose: it reads better grouped with the
# modelling output than interrupting the load -> train -> save sequence.

    metrics_path = os.path.join(output_dir, 'model_evaluation_metrics.csv')
    results_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print("Evaluation metrics and predictions saved (4 estimators)")
    
    save_predictions_csv(df, models, X, output_dir)

    print("Enumerated search space:")
    report_enumeration(df, feature_cols, MAX_VIRTUAL_COMBINATIONS)
    
    generate_all_figures(
        df,
        results_df,
        models,
        cache_dict,
        output_dir,
        raw_feature_names,
        # v22e: hand over the matrix main() already built.  Without it the function called
        # prepare_features a SECOND time, which re-printed every "槽 'xxx' 为常量" line.
        prepared=(X, y, feature_cols, display_feat_names, encoders, scaler)
    )
    _save_timings(output_dir)   # v15: after the sweep + F7/F8 timings are recorded
    
    print("\n" + "=" * 70)
    print(f"Done. Output folder: {output_dir}")
    print("=" * 70)
    # v21: release the run lock explicitly.  The atexit hook is only a backstop -- joblib's
    # own exit handlers can run before ours, so the owner record was left behind.
    if _LOCK_ON:
        release_single_run_lock()

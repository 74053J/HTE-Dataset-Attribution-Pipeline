"""htc.v16_core - pipeline core copied from the analysis script that produced the paper.

Verbatim copies of the loader, column mapping, feature construction, split, training and
SHAP functions.  Nothing imports or executes the original script at run time.

Source: Script/Analysis_all_v16.py (the copy whose prepare_features includes reactant_1 and
reactant_2, i.e. the thirteen-dataset version).
"""
from __future__ import annotations

import gc
import os
import pickle
import shutil
import sys
import tempfile
import time
import time as _timing
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, partial_dependence, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, learning_curve, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tqdm import tqdm

import xgboost as xgb

# ---------------------------------------------------------------- constants
FIT_PREPROCESSING_ON_TRAIN_ONLY = True
HISTGB_MAX_ITER = 100
LOADING_AXIS = {
    'col': None, 'source_col': None, 'role': 'unavailable', 'label': None,
    'unit': None, 'n_distinct': 0, 'reason': 'not resolved yet',
}
LOADING_AXIS_CANDIDATES = [
    'catalyst_eq',     # the intended axis
    'base_equivalents',     # Summit Baumgartner (51 / 28 distinct values)
    'ligand_equivalent',    # EDBO Suzuki (only 2 levels -- weak but real)
    'additive_loading',
    'cocatalyst_loading',
    'concentration',        # EDBO BMS (3 levels)
    'catalyst_eq',
    'loading',
]
LOADING_AXIS_LABEL = {
    'catalyst_eq': 'Catalyst loading',
    'catalyst_eq': 'Catalyst loading',
    'loading': 'Loading',
    'base_equivalents': 'Base equivalents',
    'base_equiv': 'Base equivalents',
    'ligand_equivalent': 'Ligand equivalents',
    'additive_loading': 'Additive loading',
    'cocatalyst_loading': 'Cocatalyst loading',
    'concentration': 'Concentration',
}
LOADING_AXIS_MIN_DISTINCT = 2
LOADING_AXIS_UNIT = {
    'catalyst_eq': 'eq.', 'catalyst_eq': 'eq.', 'loading': 'eq.',
    'base_equivalents': 'eq.', 'base_equiv': 'eq.', 'ligand_equivalent': 'eq.',
    'additive_loading': 'eq.', 'cocatalyst_loading': 'eq.',
    'concentration': 'M',
}
RF_N_ESTIMATORS = 100
RF_N_JOBS = os.cpu_count() or 4
TEST_SIZE = 0.2
TIMINGS = {}

# ---------------------------------------------------------------- functions
def _t_add(model_name, item, seconds):
    TIMINGS.setdefault(model_name, {})[item] = float(seconds)

def apply_loading_axis_alias(df):
    """
    Expose the resolved loading axis under the canonical name `catalyst_loading`.

    This is the whole trick that keeps v14 small: if the axis is a substitute, copy it to
    `catalyst_loading`, so `prepare_features`, the enumeration, the match-count logic and
    every CSV writer continue to work untouched.  The original column is left in place, and
    LOADING_AXIS['source_col'] remembers its real name for the axis labels.
    """
    src = LOADING_AXIS.get('source_col')
    if not src or src == 'catalyst_eq':
        return df
    df = df.copy()
    df['catalyst_eq'] = pd.to_numeric(df[src], errors='coerce')
    print(f"[loading axis] aliased '{src}' -> 'catalyst_eq' so downstream code is "
          f"unchanged (original column kept)")
    return df

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
        {'standard_column': 'catalyst_eq', 'raw_column': 'catalyst_eq'},
        {'standard_column': 'yield', 'raw_column': 'product_1_area%'},
        {'standard_column': 'success', 'raw_column': 'suff_yield'}
    ])
    
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    default_mappings.to_csv(mapping_path, index=False, encoding='utf-8-sig')
    print(f"已自动生成默认的列名映射配置文件: '{mapping_path}'")

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

    print(f"从 {mapping_path} 读取列名映射配置文件")
    
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
        # slots: col_mappings.csv decides, per file, which raw column feeds which
        # -- a column NAMED `catalyst` may in fact hold a free ligand.  The alias
        # rows for `ligand` are skipped here and applied by the table instead.
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
                    print(f"数据集列 '{raw_c}' 映射为标准 '{std_col}'")
                    break
        else:
            raw_feature_names[std_col] = std_col

    # ------------------------------------------------------------------
    # col_mappings.csv -- per-file overrides beat global aliases.
    # ------------------------------------------------------------------
    # The mapping is mandatory: a failure here means the columns are wrong, so it must
    # NOT be swallowed (it used to print "mapping skipped" and carry on with a frame
    # that had no per-file overrides at all).
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
            print(f"自动补全检索：已将原始列 '{target_smiles_col}' 识别并绑定为 'solvent_smiles'")
        elif smiles_candidates:
            print(f"  [mapping] NOTE: this file has SMILES columns {smiles_candidates[:4]} "
                  f"but none is a solvent SMILES, so 'solvent_smiles' is left UNSET.  The "
                  f"solvent-PCA panel will report it as unavailable instead of plotting the "
                  f"wrong molecules.")

    if os.environ.get('CD_METAL_READER', '1') == '0':
        # legacy reproduction only: keep the constant 'Pd' (or whatever the
        # slot machinery already put in the metal column)
        if 'metal' not in df.columns:
            df['metal'] = 'Pd'
    else:
        # Read the metal from the catalyst columns instead of assuming palladium
        # (hte/metal_reading.py, ported from the V1 metal audit).  ON by default:
        # the constant 'Pd' mis-states the catalyst metal on every Cu/Ru/Ni row.
        # This OVERWRITES the slot-level parse, which is a simple token match
        # without the ligand-stem, ferrocene, PEPPSI/CX31 and dominance rules.
        # A non-constant metal is a real condition: the ceiling moves with it.
        try:
            import hte.metal_reading as _mr
            _mr.STEMS = _mr.build_stems(
                df['ligand'].astype(str).tolist() if 'ligand' in df.columns else [])
            _nm = df['catalyst'] if 'catalyst' in df.columns else [''] * len(df)
            _sm = next((df[c] for c in df.columns
                        if 'catalyst' in c.lower() and 'smiles' in c.lower()), None)
            _fin, _src = _mr.read_rows(_nm, _sm if _sm is not None else [''] * len(df))
            df['metal'] = _fin
            print('[metal] reader -> %d distinct values: %s'
                  % (df['metal'].nunique(), dict(df['metal'].value_counts())))
        except Exception as exc:                          # noqa: BLE001
            print('[metal] reader skipped: %s' % exc)
            if 'metal' not in df.columns:
                df['metal'] = 'Pd'

    required_cols = ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 
                     'reactant_1', 'temperature', 'time', 'catalyst_eq', 
                     'yield']
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"警告: 缺失列: {missing}，已补充默认空值")
        for c in missing:
            if c != 'yield':
                df[c] = 'none' if c in ['catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1'] else 0.0

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
    _cat_fill = ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1']
    _nan_filled = {}
    for _c in _cat_fill:
        if _c in df.columns:
            _n = int(df[_c].isna().sum())
            if _n:
                df[_c] = df[_c].fillna('none').astype(str)
                _nan_filled[_c] = _n
            else:
                df[_c] = df[_c].astype(str)
    if _nan_filled:
        print("  [NaN] filled missing categorical values with 'none': "
              + ', '.join(f'{c}={n}' for c, n in _nan_filled.items()))

    if 'yield' not in df.columns:
        raise KeyError("无法在数据集中找到 'yield' 及其映射列！")

    # ------------------------------------------------------------------
    # v13: LOUD warning about collapsed categorical vocabularies.
    # A categorical axis with a single value cannot be enumerated, so the virtual library
    # collapses along it, `get_enumeration_spec` multiplies the space by 1, and every
    # figure that displays reagent names shows 'Unknown'.  This is almost always a column-
    # mapping failure rather than a property of the chemistry, so it must not be silent.
    # ------------------------------------------------------------------
    _cat_cols = ['catalyst', 'ligand', 'base', 'solvent', 'reactant_2', 'reactant_1']
    _collapsed = {}
    for _c in _cat_cols:
        if _c in df.columns:
            _u = df[_c].astype(str).unique()
            if len(_u) <= 1:
                _collapsed[_c] = _u[0] if len(_u) else '(empty)'
    if _collapsed:

        print("  [WARN] these reagent axes have ONLY ONE distinct value in this dataset: "
              "(leaving_group/substrate_type are legacy names, merged into reactant_1/2)")
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
        print(f"将小数格式Yield转换为百分比")
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
    # v20: a slot the mapping table disabled with 'none' must not survive the
    # placeholder filler above.  The list comes from col_mappings.csv (via
    # hte.mapping), never from a dataset name.
    for _slot in list(df.attrs.get("disabled_slots", [])):
        if _slot in df.columns:
            df = df.drop(columns=[_slot])
        raw_feature_names.pop(_slot, None)
    # v14: remember which column actually fills the `catalyst` slot, for the heatmap labels.
    global CATALYST_SLOT_LABEL
    CATALYST_SLOT_LABEL = str(raw_feature_names.get('catalyst', 'catalyst'))
    return df, raw_feature_names

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
    categorical_cols = ['metal', 'catalyst', 'ligand', 'base', 'solvent', 'cosolvent', 'additive', 'reactant_2', 'reactant_1']
    numerical_cols = ['temperature', 'time', 'catalyst_eq', 'reactant_2_eq', 'base_eq', 'reactant_1_eq', 'ligand_eq', 'solvent_eq', 'additive_eq']
    
    categorical_cols = [c for c in categorical_cols if c in df_encoded.columns]
    _kept = []
    for _c in categorical_cols:
        if df_encoded[_c].astype(str).nunique(dropna=False) > 1:
            _kept.append(_c)
        else:
            print(f"\u2139\ufe0f \u69fd '{_c}' \u5728\u672c\u6570\u636e\u96c6\u4e3a\u5e38\u91cf\uff0c\u4e0d\u53c2\u4e0e\u5efa\u6a21\u3002")
    categorical_cols = _kept

    # v18: rare categorical levels are folded into 'OTHER'.
    # bh-hte-ood's partner slots hold 701 / 852 distinct SMILES strings; one-hot
    # over those is both statistically meaningless (about 30 rows per level) and
    # computationally explosive (1,746 columns).  A level with fewer than
    # RARE_LEVEL_MIN_COUNT rows is folded, uniformly for every dataset.
    RARE_LEVEL_MIN_COUNT = 50
    for _c in categorical_cols:
        _v = df_encoded[_c].astype(str)
        _cnt = _v.value_counts()
        _rare = _cnt[_cnt < RARE_LEVEL_MIN_COUNT].index
        if len(_rare):
            df_encoded[_c] = _v.where(~_v.isin(_rare), 'OTHER')
            print(f"  [rare] '{_c}': {len(_rare)} levels < {RARE_LEVEL_MIN_COUNT} rows -> 'OTHER'")
    
    valid_numerical_cols = []
    for col in numerical_cols:
        if col in df_encoded.columns:
            if df_encoded[col].nunique() > 1:
                valid_numerical_cols.append(col)
            else:
                print(f"ℹ️ 特征 '{col}' 为常量（唯一值为 {df_encoded[col].iloc[0]}），已自动从模型训练集中剔除。")
    
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


def resolve_loading_axis(df, verbose=True):
    """
    Decide which column the loading-dependent figures (F3a / F9d / F11) should sweep.

    Returns a dict and ALSO stores it in the module global LOADING_AXIS:
        col         canonical column name to use downstream ('catalyst_eq' after the
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
        _is_filler = (cand == 'catalyst_eq'
                      and pd.to_numeric(df[cand], errors='coerce').nunique(dropna=False) <= 1)
        vals = pd.to_numeric(df[cand], errors='coerce')
        vals = vals[np.isfinite(vals)]
        nu = int(vals.nunique())
        if nu < LOADING_AXIS_MIN_DISTINCT:
            rejected.append((cand, nu, _is_filler))
            continue
        axis['source_col'] = cand
        axis['col'] = 'catalyst_eq'
        axis['role'] = 'catalyst' if cand in ('catalyst_eq', 'catalyst_eq') else 'substitute'
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
        # ---- v15 additions: same budget, different CATEGORICAL HANDLING --------------
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

    for name, model in models.items():
        print(f"Training [{name}] ...")
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
        
        # ---- SUBSTRATE-DISJOINT CV ----------------------------------------------------
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

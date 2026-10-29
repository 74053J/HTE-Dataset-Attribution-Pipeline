"""Parameter registry for the standalone ``hte`` package.

Single source of truth
----------------------
``params.json`` in this package already exports the hyper-parameters that
produced the paper's numbers.  Those values are loaded from it at import time and
OVERRIDE the defaults below, so editing ``params.json`` still changes the
package.  The remaining constants are copied verbatim from the frozen
``archive/Analysis_all_v16.py`` because ``params.json`` does not carry
them; each one is annotated with its v16 line.

``params.json`` is never written by this module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS_JSON = Path(__file__).resolve().parent / "params.json"

# v16 constants that params.json already carries.  The values here are only a
# fallback for a checkout without params.json; the file wins when present.
DEFAULTS: dict[str, object] = {
    "VIRTUAL_LIBRARY_PREVIEW_ROWS": 200_000,      # v16:146
    "STREAM_CHUNK_COMBOS": 200_000,               # v16:147
    "SURVIVOR_POOL_ROWS": 20_000,                 # v16:148
    "ENSEMBLE_WEIGHT_POWER": 1.0,                 # v16:171
    "ENSEMBLE_MIN_WEIGHT_R2": 0.05,               # v16:172
    "USE_TWO_STAGE_SCREENING": False,             # v16:206
    "STAGE1_KEEP_ROWS": 50_000,
    "AUDIT_CONDITION_SCORES": True,               # v16:213
    "USE_MULTIPROCESSING": True,                  # v16:215
    "N_WORKERS": 0,                               # v16:216
    "MAX_WORKERS": 16,                            # v16:217
    "PREDICT_BATCH_ROWS": 250_000,                # v16:294
    "TEST_SIZE": 0.2,                             # v16:308
    "HISTGB_MAX_ITER": 100,                       # v16:320
    "RF_N_ESTIMATORS": 100,                       # v16:325
    "PARALLEL_TOPK_PER_CHUNK": 20_000,            # v16:357
    "FIT_PREPROCESSING_ON_TRAIN_ONLY": True,      # v16:377
    "CLIP_MEMBER_PREDICTIONS": True,              # v16:411
    "ENSEMBLE_MAX_MODEL_GAP": 20.0,               # v16:844
    # Constants the package needs that params.json does not export.
    "PREDICTION_MIN": 0.0,                        # v16:412
    "PREDICTION_MAX": 100.0,                      # v16:413
    "SHAP_SAMPLE_ROWS": 500,                      # v16:431
    "KEEP_MODEL_PREDICTIONS": False,              # v16:122
    "PREDICT_BATCH_SIZE": 2_000_000,              # v16:126
    "N_LOADING_POINTS": 100,                      # v16:790
    "MAX_VIRTUAL_COMBINATIONS": 1_000_000_000,    # v16:806
    "EI_INCUMBENT_MODE": "quantile",              # v16:877
    "EI_INCUMBENT_QUANTILE": 0.90,                # v16:878
    "EI_TARGET_QUANTILE": 1.00,                   # v16:886
    "ENSEMBLE_MIN_SUPPORT": 0,                    # v16:848
    "STATUS_MATCH_LEVEL": "reagent",              # v16:961
    "RF_N_JOBS": os.cpu_count() or 4,             # v16:328
    "CATEGORICAL_SLOTS": [                        # v17: 9 categorical condition slots
        "metal", "catalyst", "ligand", "base", "solvent", "cosolvent",
        "additive", "reactant_2", "reactant_1",
    ],
    "NUMERIC_SLOTS": [                            # v18: equivalents slots, one per substance
        "temperature", "time", "catalyst_eq", "reactant_1_eq", "reactant_2_eq",
        "ligand_eq", "base_eq", "solvent_eq", "cosolvent_eq", "additive_eq",
    ],
    "CONDITION_COLS": [                           # v18: 9 categorical + 10 equivalents
        "metal", "catalyst", "ligand", "base", "solvent", "cosolvent",
        "additive", "reactant_2", "reactant_1",
        "temperature", "time", "catalyst_eq", "reactant_1_eq", "reactant_2_eq",
        "ligand_eq", "base_eq", "solvent_eq", "cosolvent_eq", "additive_eq",
    ],
    "ENUM_AXES": [                                # v16:458 -- enumeration axes, kept
        "metal", "catalyst", "ligand", "base", "solvent", "reactant_2",
        "reactant_1",
    ],
    "REAGENT_AXES": [                             # v16:468
        "catalyst", "ligand", "base", "solvent", "reactant_2",
        "reactant_1",
    ],
}


def load_params(path: str | os.PathLike | None = None) -> dict:
    """Return the merged parameter dict (``params.json`` overrides DEFAULTS)."""
    merged = dict(DEFAULTS)
    p = Path(path) if path is not None else PARAMS_JSON
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        for key, value in raw.items():
            if key.startswith("_"):
                continue
            merged[key] = value
        merged["_source"] = raw.get("_source")
        merged["_raw"] = raw
    else:
        merged["_source"] = None
        merged["_raw"] = {}
    merged["_path"] = str(p)
    return merged


PARAMS: dict = load_params()


def get(name: str, default=None):
    """Look up one parameter, falling back to ``default``."""
    return PARAMS.get(name, default)


def as_dict() -> dict:
    """A shallow copy of the merged parameters (for provenance/JSON output)."""
    return {k: v for k, v in PARAMS.items() if not k.startswith("_")}


def __getattr__(name: str):
    """``params.TEST_SIZE`` style access, for modules that import this module."""
    try:
        return PARAMS[name]
    except KeyError as exc:  # pragma: no cover - attribute protocol
        raise AttributeError(name) from exc


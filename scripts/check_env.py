"""Preflight environment check -- run this BEFORE run_all.bat.

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --strict     # also fail on optional packages

Exit code 0 = every REQUIRED import is present; 1 = something is missing (the exact
``pip install`` command is printed).

Why this exists: the pipeline needs five packages that are easy to miss
(``seaborn``, ``tqdm``, ``Pillow``, ``openpyxl`` and the CPU build of ``torch``), and a
missing one shows up only after several minutes of computation, as a bare
``ModuleNotFoundError`` in the middle of a step.
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

# (import name, pip name, required, where it is used)
PACKAGES = [
    ("numpy",      "numpy",         True,  "everywhere"),
    ("pandas",     "pandas",        True,  "everywhere"),
    ("sklearn",    "scikit-learn",  True,  "hte/features.py, scripts/build_grid_ceiling.py"),
    ("xgboost",    "xgboost-cpu",   True,  "hte/models.py, hte/attribution.py"),
    ("matplotlib", "matplotlib",    True,  "scripts/figures*.py"),
    ("shap",       "shap",          True,  "hte/attribution.py, scripts/build_attrib13.py"),
    ("scipy",      "scipy",         True,  "hte/ceiling.py"),
    ("rdkit",      "rdkit",         True,  "hte/metal_reading.py, scripts/build_table2.py"),
    ("yaml",       "PyYAML",        True,  "hte/datasets.py"),
    ("pyarrow",    "pyarrow",       True,  "Roche66k parquet input"),
    ("docx",       "python-docx",   False, "scripts/sync_figures_to_docx.py"),
    ("seaborn",    "seaborn",       True,  "archive/single_dataset_full_benchmark.py (Fig. 5)"),
    ("tqdm",       "tqdm",          True,  "hte/recommend.py, hte/v16_core.py"),
    ("PIL",        "Pillow",        True,  "scripts/figures.py"),
    ("openpyxl",   "openpyxl",      True,  "scripts/fetch_datasets.py (rxn_yields xlsx)"),
    ("torch",      "torch",         True,  "scripts/build_table2.py (GNN arm only)"),
]

REQUIRED_DIRS = ("results", "outputs", "figures")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="treat optional packages as required")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    print("=" * 72)
    print(" environment check")
    print("=" * 72)
    print("python   %s" % sys.version.split()[0])
    print("exe      %s" % sys.executable)
    print("platform %s %s" % (platform.system(), platform.release()))

    missing_req, missing_opt = [], []
    print("\n%-13s %-14s %-9s %s" % ("import", "version", "status", "used by"))
    print("-" * 72)
    for mod, pip_name, required, used_by in PACKAGES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", None) or getattr(m, "version", "?")
            if not isinstance(ver, str):
                ver = "?"
            print("%-13s %-14s %-9s %s" % (mod, ver, "ok",
                                           used_by if len(used_by) < 34 else used_by[:31] + "..."))
        except Exception as exc:                      # noqa: BLE001
            need = required or args.strict
            status = "MISSING" if need else "warn"
            print("%-13s %-14s %-9s %s" % (mod, "-", status, pip_name))
            (missing_req if need else missing_opt).append(pip_name)

    # xgboost-cpu note: the import name is `xgboost`, so also show the distribution
    try:
        from importlib.metadata import version as dist_version
        for dist in ("xgboost-cpu", "xgboost"):
            try:
                print("\n[dist] %-12s %s" % (dist, dist_version(dist)))
            except Exception:                          # noqa: BLE001
                pass
    except Exception:                                  # noqa: BLE001
        pass

    # A6: xgboost must be the CPU distribution (the CUDA wheel creates a context per process)
    print("\nxgboost distribution:")
    try:
        from importlib.metadata import version as _dv, PackageNotFoundError as _PNF
        def _has(name):
            try:
                _dv(name); return True
            except _PNF:
                return False
        cpu, cuda = _has("xgboost-cpu"), _has("xgboost")
        if cpu:
            print("  xgboost-cpu %s  ok (CPU build)" % _dv("xgboost-cpu"))
        elif cuda:
            print("  WARNING: distribution 'xgboost' (CUDA build) is installed, 'xgboost-cpu' is not.")
            print("           Two concurrent runs can die with an access violation.  Replace it:")
            print("             python -m pip uninstall -y xgboost")
            print("             python -m pip install --no-cache-dir xgboost-cpu")
        else:
            print("  neither 'xgboost' nor 'xgboost-cpu' is installed")
    except Exception:                                  # noqa: BLE001
        pass

    # A5: dataset inventory -- run_all needs the 13 CSVs, not just the data/ folder
    print("\ndata inventory:")
    mapping = repo / "data" / "alias_mapping.yaml"
    expected = []
    if mapping.is_file():
        try:
            import yaml
            data = yaml.safe_load(mapping.read_text(encoding="utf-8")) or {}
            entries = ((data.get("datasets") or data.get("aliases") or {})
               if isinstance(data, dict) else {})
            for _alias, meta in (entries.items() if isinstance(entries, dict) else []):
                if isinstance(meta, dict) and meta.get("source"):
                    # corpus:false = the synthetic benchmark; not needed by run_all
                    if meta.get("corpus") is False:
                        continue
                    expected.append(str(meta["source"]) + ".csv")
        except Exception:                              # noqa: BLE001
            pass
    present = {p.name for p in (repo / "data").glob("*.csv")} if (repo / "data").is_dir() else set()
    missing_ds = [e for e in expected if e not in present]
    print("  %d/%d dataset CSVs present" % (len(expected) - len(missing_ds), len(expected)))
    for m in missing_ds:
        print("    missing: %s  -> run fetch_datasets.bat" % m)

    # directories the writers need (they create them, but report for clarity)
    print("\ndirectories:")
    for d in REQUIRED_DIRS:
        p = repo / d
        print("  %-10s %s" % (d, "exists" if p.is_dir() else "absent (will be created)"))
    print("  %-10s %s" % ("data/", "exists" if (repo / "data").is_dir() else "ABSENT"))

    if missing_req:
        print("\n" + "!" * 72)
        print("MISSING REQUIRED PACKAGES:", ", ".join(missing_req))
        print("install with:")
        print("  python -m pip install --no-cache-dir " + " ".join(missing_req))
        if "torch" in missing_req:
            print("  torch MUST come from the CPU wheel (the default wheel drags ~2 GB of CUDA):")
            print("    python -m pip install --no-cache-dir torch "
                  "--index-url https://download.pytorch.org/whl/cpu")
        print("!" * 72)
        return 1

    if missing_opt:
        print("\noptional packages absent (only some steps need them): "
              + ", ".join(missing_opt))
    if missing_ds:
        print("\nDATA MISSING: %d dataset CSV(s).  run_all.bat will stop at its data check."
              % len(missing_ds))
        print("  run:  fetch_datasets.bat      (answer I AGREE; rxn_yields asks a second time: yes)")
        return 2
    print("\nRESULT: environment OK -- run_all.bat can start.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

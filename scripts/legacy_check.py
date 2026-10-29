#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke-test the frozen pipeline in archive/ -- load a few datasets, print shapes.

This does NOT run the paper's pipeline; it only proves the archive copy still
resolves its data files after being moved into a subfolder.

    python scripts/legacy_check.py                  # uses ./data
    python scripts/legacy_check.py --data-dir <dir> # point at another data/ copy
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONO = ROOT / "archive" / "single_dataset_full_benchmark.py"

PROBES = [
    "(AI_RCP_pub)bh_data_clean_all_whitelisted",
    "(Summit)Reizman_Suzuki_385",
    "(DoyleHTE)CN_coupling_3600",
    "(Roche66k)Amide_coupling_3917",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("archive : %s" % MONO)
    print("data    : %s" % args.data_dir)
    if not MONO.is_file():
        print("ARCHIVE MISSING")
        return 1

    spec = importlib.util.spec_from_file_location("legacy_frozen", MONO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("loaded  : %d names\n" % len(dir(mod)))

    data = Path(args.data_dir)
    bad = 0
    for stem in PROBES:
        path = data / (stem + ".csv")
        if not path.is_file():
            print("  %-52s MISSING FILE" % stem)
            bad += 1
            continue
        try:
            df, raw = mod.load_real_dataset(str(path), str(data))
            print("  %-52s %6d rows x %2d cols" % (stem, len(df), df.shape[1]))
        except Exception as exc:
            print("  %-52s FAILED: %s" % (stem, str(exc)[:60]))
            bad += 1
    print("\n%d/%d loaded" % (len(PROBES) - bad, len(PROBES)))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Run the hte equivalence / wiring tests and print one summary.

    python tests/run_equivalence.py
    python tests/run_equivalence.py --quick          # skip the 12-cell grid
    python tests/run_equivalence.py --only data ceiling
    python tests/run_equivalence.py --only grid

Each child is a normal subprocess, so a crash in one cannot hide another.  Exit
code 0 only when every selected test passes; 1 when any fails; 2 when no test
could run because the raw data directory is absent (the children themselves
exit 0 with a SKIP line in that case).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = {
    "data": ("data/features (a)", "test_data_features.py"),
    "ceiling": ("noise ceiling (b)", "test_ceiling.py"),
    "grid": ("12-cell grid (c)", "test_grid_one_dataset.py"),
    "attribution": ("SHAP attribution (extra)", "test_attribution.py"),
    "recommend": ("recommend wiring", "test_recommend.py"),
}
DEFAULT_ORDER = ["data", "ceiling", "grid", "attribution", "recommend"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=sorted(TESTS),
                    help="run only these tests (default: all)")
    ap.add_argument("--quick", action="store_true",
                    help="skip the 12-cell grid test (the only expensive one)")
    args = ap.parse_args(argv)

    names = list(args.only) if args.only else list(DEFAULT_ORDER)
    if args.quick and "grid" in names and not args.only:
        names.remove("grid")

    print("=" * 78)
    print("hte equivalence tests   python: %s" % sys.executable)
    print("=" * 78)
    results = []
    for name in names:
        label, script = TESTS[name]
        print("\n" + "-" * 78)
        print(">>> %s  (%s)" % (label, script))
        print("-" * 78, flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(HERE / script)],
                              cwd=str(HERE.parent))
        dt = time.perf_counter() - t0
        results.append((name, label, proc.returncode, dt))
        print("<<< %s: %s  (%.1f s)"
              % (script, "PASS" if proc.returncode == 0 else "FAIL", dt), flush=True)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    failed = 0
    for name, label, rc, dt in results:
        print("  %-14s %-28s %s  (%.1f s)"
              % (name, label, "PASS" if rc == 0 else "FAIL", dt))
        failed += rc != 0
    print("=" * 78)
    print("RESULT:", "ALL PASSED" if failed == 0 else "%d TEST(S) FAILED" % failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

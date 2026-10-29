"""Regenerate results/checksums.txt.

The repository ships result tables as evidence. This script hashes the frozen
pipeline, the parameter file, the alias mapping and every result CSV, so a
reader can verify that nothing has been edited since publication.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIXED = [
    str(Path("archive") / "single_dataset_full_benchmark.py"),   # the frozen pipeline (was Analysis_all_v16.py)
    "Analysis_all_v16.py",                  # kept for older layouts; skipped if absent
    str(Path("hte") / "params.json"),
    "data/alias_mapping.yaml",
]
HEADER = [
    "# sha256 checksums of the frozen legacy script and every result table",
    "# regenerate: python scripts/make_checksums.py",
]


def targets() -> list[Path]:
    files = [ROOT / p for p in FIXED]
    # Only canonical result tables are shipped evidence. Scratch and verification
    # copies (dot-directories, .reference/, rebuilt/, _probe*/) are excluded, and
    # so is results/legacy/ (superseded 16-dataset artefacts, not cited by the paper).
    def shipped(p: Path) -> bool:
        rel = p.relative_to(ROOT / "results").parts
        if p.name == "checksums.txt":
            return False
        return not any(part.startswith((".", "_"))
                       or part in ("rebuilt", "examples", "legacy")
                       for part in rel)

    files += sorted(p for p in (ROOT / "results").rglob("*.csv") if shipped(p))
    return [p for p in files if p.exists()]


def main() -> int:
    lines = list(HEADER) + [""]
    for p in targets():
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append("%s  %s" % (digest, p.relative_to(ROOT).as_posix()))
    out = ROOT / "results" / "checksums.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %s (%d entries)" % (out, len(lines) - len(HEADER) - 1))
    for line in lines[3:]:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

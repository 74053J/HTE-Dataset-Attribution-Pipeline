# -*- coding: utf-8 -*-
"""The corpus definition, read from ``data/alias_mapping.yaml``.

Every list of datasets the package needs -- which sixteen files exist, which
thirteen the manuscript reports, which two have no computable ceiling -- is
stated **once**, in the mapping YAML next to the column mapping.  Nothing in the
code names a dataset.

``data/alias_mapping.yaml`` entries carry:

    alias: {source: <file stem>, source_group: ..., reaction: ...,
            corpus: true/false,        # appears in the manuscript's tables
            ceiling: false}            # no condition ever repeats

``corpus`` defaults to true and ``ceiling`` defaults to true, so a new dataset
only needs ``source``.
"""
from __future__ import annotations

from pathlib import Path

YAML_NAME = "alias_mapping.yaml"


def _yaml_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(__file__).resolve().parents[1] / "data" / YAML_NAME


def _entries(path: str | Path | None = None) -> list:
    """[(alias, spec)] in file order."""
    import yaml

    doc = yaml.safe_load(_yaml_path(path).read_text(encoding="utf-8")) or {}
    ds = doc.get("datasets") or {}
    return [(str(k), dict(v)) for k, v in ds.items()]


def stems(path: str | Path | None = None) -> dict:
    """alias -> raw file stem (no ``.csv``)."""
    return {a: str(s["source"]) for a, s in _entries(path)}


def dataset_order(path: str | Path | None = None) -> list:
    """[(alias, stem)] for every file in the mapping -- the grid's row order."""
    return [(a, str(s["source"])) for a, s in _entries(path)]


def corpus(path: str | Path | None = None) -> list:
    """The aliases the manuscript reports (``corpus: true``, the default)."""
    return [a for a, s in _entries(path) if bool(s.get("corpus", True))]


def no_ceiling(path: str | Path | None = None) -> list:
    """The aliases with no repeated condition, so no ceiling is defined."""
    return [a for a, s in _entries(path) if s.get("ceiling", True) is False]


def donors(path: str | Path | None = None) -> list:
    """The transfer donors of §3.5 (``donor: true``)."""
    return [a for a, s in _entries(path) if bool(s.get("donor", False))]


def case_study(path: str | Path | None = None) -> str:
    """The dataset the SHAP / ligandxbase / acquisition figures are drawn on.

    Flagged ``case_study: true`` in the table; there must be exactly one.
    """
    hits = [a for a, s in _entries(path) if bool(s.get("case_study", False))]
    if len(hits) != 1:
        raise ValueError("alias_mapping.yaml must flag exactly one case_study "
                         "dataset, found %d: %s" % (len(hits), hits))
    return hits[0]

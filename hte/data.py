"""Dataset loading and column normalisation.

This module is a thin façade over :mod:`hte.v16_core`, which holds the loader
itself (copied verbatim from the frozen ``archive/Analysis_all_v16.py``, the
code that produced the paper).

What lives where:

* the loader, the alias pass, the loading-axis resolver and the placeholders
  are in :mod:`hte.v16_core`;
* every per-file decision -- which raw column feeds which slot, and which slot a
  file must NOT use -- comes from ``data/col_mappings.csv``
  (:mod:`hte.mapping`).  There is no per-dataset table in the code;
* the mutable globals ``LOADING_AXIS`` and ``CATALYST_SLOT_LABEL`` stay the ones
  in :mod:`hte.v16_core`; :pep:`562` forwards them, so ``hte.data.LOADING_AXIS``
  is always the live object rather than a stale copy.

Historically this module carried a second, complete copy of the loader plus a
hard-coded ``catalyst``/``ligand`` routing table keyed by dataset name.  Both
were removed on 2026-10-27; the routing is 13 override rows in the mapping CSV.
"""
from __future__ import annotations

from . import params as P
from .v16_core import (
    apply_loading_axis_alias,
    create_default_col_mappings,
    load_real_dataset,
    resolve_loading_axis,
)

#: the axes the virtual library enumerates (was previously read from here)
ENUM_AXES = list(P.ENUM_AXES)

#: names that must stay *live* module globals of :mod:`hte.v16_core`
_LIVE = ("LOADING_AXIS", "CATALYST_SLOT_LABEL", "LOADING_AXIS_CANDIDATES",
         "LOADING_AXIS_LABEL", "LOADING_AXIS_UNIT", "LOADING_AXIS_MIN_DISTINCT")


def loading_axis_available():
    """True when this dataset has a usable amount axis (see ``resolve_loading_axis``)."""
    from . import v16_core
    return v16_core.LOADING_AXIS.get("col") is not None


def __getattr__(name):
    if name in _LIVE:
        from . import v16_core
        return getattr(v16_core, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = ["load_real_dataset", "create_default_col_mappings",
           "resolve_loading_axis", "apply_loading_axis_alias",
           "loading_axis_available", "ENUM_AXES"]

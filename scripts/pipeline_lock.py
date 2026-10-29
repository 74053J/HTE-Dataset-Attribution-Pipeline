#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Machine-wide pipeline lock, usable from a .bat file.

There are two ways to hold it, because a batch file cannot hold a kernel object:

* ``single_dataset`` -- a live Python process.  The record carries its real PID, so the
  lock is known to be stale the moment that process disappears.
* ``run_all`` -- a batch run.  A batch has no PID a helper can observe, so the record
  carries a TTL (default 6 h) and is released explicitly when the run finishes.  A killed
  run therefore blocks for at most the TTL instead of for ever.

Usage
-----
    python scripts/pipeline_lock.py acquire --kind run_all [--ttl-minutes 360]
        exit 0 acquired, 3 busy, 1 error
    python scripts/pipeline_lock.py release          # only if this process/host owns it
    python scripts/pipeline_lock.py status           # 0 free, 3 busy
    python scripts/pipeline_lock.py release --force  # break a lock whose run died

The single-dataset pipeline also takes a Windows named mutex, which is crash-safe; both
sides consult this same file, so a dataset run and a batch run never overlap.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(ROOT, ".pipeline_owner.json")
DEFAULT_TTL_MIN = 360


def _pid_alive(pid: int) -> bool:
    if not pid or pid < 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    k32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if h:
        k32.CloseHandle(h)
        return True
    return False


def read_holder():
    try:
        with open(LOCK, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def is_stale(d) -> bool:
    """A record is stale when its owner is provably gone, or when its TTL has expired."""
    if not d:
        return True
    kind = str(d.get("kind", ""))
    pid = int(d.get("pid", -1) or -1)
    if kind == "single_dataset":
        return not _pid_alive(pid)
    try:
        started = float(d.get("started_ts", 0))
    except Exception:
        return True
    ttl = float(d.get("ttl_minutes", DEFAULT_TTL_MIN)) * 60.0
    return started <= 0 or (time.time() - started) > ttl


def holder_text() -> str:
    d = read_holder()
    if not d:
        return "holder unknown"
    return "pid=%s, kind=%s, started %s" % (d.get("pid"), d.get("kind"), d.get("started"))


def _write(kind: str, ttl_minutes: float) -> None:
    with open(LOCK, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "kind": kind,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "started_ts": time.time(),
                   "ttl_minutes": float(ttl_minutes)}, fh, ensure_ascii=False)


def acquire(kind: str, ttl_minutes: float = DEFAULT_TTL_MIN) -> int:
    d = read_holder()
    if d and not is_stale(d):
        print("[lock] busy: %s" % holder_text())
        return 3
    if d:
        print("[lock] reclaiming stale record (%s)" % holder_text())
    try:
        _write(kind, ttl_minutes)
    except Exception as exc:                                   # noqa: BLE001
        print("[lock] could not write %s: %s" % (LOCK, exc))
        return 1
    print("[lock] acquired for %s" % kind)
    return 0


def release(force: bool = False) -> int:
    d = read_holder()
    if not d:
        return 0
    own = int(d.get("pid", -1)) == os.getpid()
    if not own and not force:
        print("[lock] refusing to release a record owned by %s" % holder_text())
        return 1
    try:
        os.remove(LOCK)
        print("[lock] released")
    except Exception as exc:                                   # noqa: BLE001
        print("[lock] could not remove %s: %s" % (LOCK, exc))
        return 1
    return 0


def status() -> int:
    d = read_holder()
    if d and not is_stale(d):
        print("[lock] busy: %s" % holder_text())
        return 3
    print("[lock] free" + ("" if not d else " (stale record present)"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="machine-wide pipeline lock")
    ap.add_argument("action", choices=("acquire", "release", "status"))
    ap.add_argument("--kind", default="run_all")
    ap.add_argument("--ttl-minutes", type=float, default=DEFAULT_TTL_MIN)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.action == "acquire":
        return acquire(a.kind, a.ttl_minutes)
    if a.action == "release":
        return release(a.force)
    return status()


if __name__ == "__main__":
    sys.exit(main())

"""Probe PyPI directly; tell ``setup_env.bat`` whether to use the mirror.

Usage
    python scripts/_net_probe.py                 # human line: "ok 96 KB/s"
    python scripts/_net_probe.py --choose 60     # prints the pip flag to use
                                                # (empty stdout = direct PyPI)
    python scripts/_net_probe.py --choose 60 -v  # same, plus the human line on stderr

Proxies are ignored on purpose: the default install path assumes none, and a
machine-local proxy stored without a scheme in the registry makes pip fail with
``check_hostname requires server_hostname``.
"""

from __future__ import annotations

import ssl
import sys
import time
import urllib.request
from pathlib import Path

URL = "https://pypi.org/simple/"
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
TIMEOUT = 12.0
CHUNK = 1 << 16


def probe() -> tuple[str, int]:
    """Return (state, KB/s); state is "ok" or "fail"."""
    ctx = ssl.create_default_context()
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),          # direct: ignore env and registry
        urllib.request.HTTPSHandler(context=ctx),
    )
    req = urllib.request.Request(URL, headers={"User-Agent": "setup-env-probe"})
    t0 = time.perf_counter()
    try:
        with opener.open(req, timeout=TIMEOUT) as resp:
            data = resp.read(CHUNK)
        dt = max(time.perf_counter() - t0, 1e-6)
        return "ok", int(len(data) / 1024.0 / dt)
    except Exception:
        return "fail", 0


def main(argv: list[str]) -> int:
    if "--decide" in argv:
        i = argv.index("--decide")
        try:
            speed_min = int(argv[i + 1])
        except (IndexError, ValueError):
            speed_min = 60
        out = None
        if "--out" in argv:
            j = argv.index("--out")
            out = argv[j + 1] if j + 1 < len(argv) else None
        state, speed = probe()
        use_mirror = state != "ok" or speed < speed_min
        if out:
            Path(out).write_text(
                ("--index-url %s" % MIRROR) if use_mirror else "", encoding="utf-8")
        print("direct PyPI: %s, %d KB/s (need %d) -> %s"
              % (state, speed, speed_min,
                 "Tsinghua mirror" if use_mirror else "direct PyPI"))
        return 0

    if "--choose" in argv:
        i = argv.index("--choose")
        try:
            speed_min = int(argv[i + 1])
        except (IndexError, ValueError):
            speed_min = 60
        state, speed = probe()
        use_mirror = state != "ok" or speed < speed_min
        if "-v" in argv:
            print("direct PyPI: %s, %d KB/s (need %d) -> %s"
                  % (state, speed, speed_min,
                     "Tsinghua mirror" if use_mirror else "direct PyPI"),
                  file=sys.stderr)
        if use_mirror:
            print("--index-url %s" % MIRROR)
        return 0

    state, speed = probe()
    print("%s %d KB/s" % (state, speed))
    return 0 if state == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

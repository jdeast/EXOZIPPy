"""Verify the 8.6.17 port: every ported config must build, and where a
pre-refactor start logp was measured it must MATCH.

Bit-identical logp across a pure representation change is the strongest
check available -- it says the port preserved the model, not merely that
the YAML parses.  v7 already passed it (81373.201 both sides).
"""

import io
import logging
import os

import numpy as np
import yaml

logging.disable(logging.WARNING)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs"))
from exozippy.system import System  # noqa: E402

# measured BEFORE the refactor; None = no baseline recorded
KNOWN = {
    "DC2018_128_severed_v3.yaml": 80519.222,
    "DC2018_128_severed_v4.yaml": None,
    "DC2018_128_severed_v5.yaml": None,
    "DC2018_128_severed_v6.yaml": 80507.376,
    "DC2018_128_severed_v7.yaml": 81373.201,
    "DC2018_128_tightpriors.yaml": 80865.224,
}
print(
    "%-34s %8s %8s %14s %14s %s"
    % ("config", "RVs", "elems", "logp", "baseline", "verdict"),
    flush=True,
)
for cfg, base in KNOWN.items():
    try:
        d = yaml.safe_load(io.open(cfg, encoding="utf-8"))
        s = System(d, user_params=None)
        s.prepare()
        m = s.build_model()
        ip = m.initial_point()
        n = sum(int(np.asarray(ip[v.name]).size) for v in m.value_vars)
        lp = float(m.compile_logp()(ip))
        if base is None:
            v = "built (no baseline)"
        elif abs(lp - base) < 1e-3:
            v = "MATCH"
        else:
            v = "*** MISMATCH %+.3f ***" % (lp - base)
        print(
            "%-34s %8d %8d %14.3f %14s %s"
            % (
                cfg,
                len(m.free_RVs),
                n,
                lp,
                "%.3f" % base if base else "--",
                v,
            ),
            flush=True,
        )
    except Exception as e:
        print(
            "%-34s FAILED %s: %s" % (cfg, type(e).__name__, str(e)[:90]),
            flush=True,
        )

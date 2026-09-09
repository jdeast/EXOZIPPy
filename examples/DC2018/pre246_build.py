"""Build the PRE-PORT tightpriors config against PRE-#246 code.

Separates the two candidates for the +1535 nats:
  * reproduces the 80865.224 baseline -> old code + old config are
    self-consistent, so the shift came in with #242/#246 and NOT with the
    port;
  * does not reproduce it -> the baseline was never a valid comparison.  It
    was taken 2026-09-04 with the MODULE python at an unrecorded sha, before
    two merges.

NOTE this file lives in the repo, not the scratchpad: /tmp is NODE-LOCAL on
this cluster, so a job submitted from the login node cannot see anything
written to /tmp there.  The first attempt at this test died exactly that
way.
"""

import io
import logging
import os
import sys

import numpy as np
import yaml

logging.disable(logging.WARNING)
WT = sys.argv[1]
sys.path.insert(0, os.path.join(WT, "src"))
os.chdir(os.path.join(WT, "examples/DC2018/configs"))
import exozippy  # noqa: E402
from exozippy.system import System  # noqa: E402

print("exozippy from:", os.path.dirname(exozippy.__file__), flush=True)
cfg = yaml.safe_load(io.open("DC2018_128_tightpriors.yaml", encoding="utf-8"))
s = System(cfg, user_params=None)
s.prepare()
m = s.build_model()
ip = m.initial_point()
n = sum(int(np.asarray(ip[v.name]).size) for v in m.value_vars)
lp = float(m.compile_logp()(ip))
print(
    "PRE-#246 code + PRE-port config: %d RVs / %d elems  logp = %.3f"
    % (len(m.free_RVs), n, lp),
    flush=True,
)
print(
    "baseline                        : 80865.224  (delta %+.3f)"
    % (lp - 80865.224),
    flush=True,
)
print(
    "post-port on current code       : 82400.527  (delta %+.3f)"
    % (lp - 82400.527),
    flush=True,
)

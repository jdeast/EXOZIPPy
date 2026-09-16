"""Where do tightpriors' +1535 nats live?  Per-term, against v3.

Two hypotheses are already dead: the t_0 bound (worth 0.41 nats, measured)
and PR #242's event-rate recount (v3 and v6 baselines predate #242 too and
still match bit-identically).  So stop guessing and read the terms.

v3 is the control: same event, same data, MATCHES its pre-refactor
baseline.  Any term that differs between them by ~1535 is the answer;
if none does, the difference is spread and the story is different.
"""

import io
import logging
import os

import numpy as np
import pytensor
import yaml

logging.disable(logging.WARNING)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs"))
from exozippy.system import System  # noqa: E402


def terms(cfg_name):
    cfg = yaml.safe_load(io.open(cfg_name, encoding="utf-8"))
    s = System(cfg, user_params=None)
    s.prepare()
    m = s.build_model()
    ip = m.initial_point()
    ts = m.logp(sum=False)
    names = [
        getattr(t, "name", None) or "term%d" % i for i, t in enumerate(ts)
    ]
    fn = pytensor.function(m.value_vars, ts, on_unused_input="ignore")
    vals = fn(*[ip[v.name] for v in m.value_vars])
    return {n: float(np.sum(np.asarray(v))) for n, v in zip(names, vals)}


tp = terms("DC2018_128_tightpriors.yaml")
v3 = terms("DC2018_128_severed_v3.yaml")
print(
    "tightpriors total %.3f (baseline 80865.224, delta %+.3f)"
    % (sum(tp.values()), sum(tp.values()) - 80865.224),
    flush=True,
)
print(
    "v3          total %.3f (baseline 80519.222, delta %+.3f)"
    % (sum(v3.values()), sum(v3.values()) - 80519.222),
    flush=True,
)

print("\n%-46s %14s %14s" % ("term", "tightpriors", "v3"), flush=True)
for k in sorted(set(tp) | set(v3), key=lambda k: -abs(tp.get(k, 0.0))):
    a, b = tp.get(k), v3.get(k)
    if a is None and abs(b or 0) < 0.5:
        continue
    print(
        "%-46s %14s %14s"
        % (
            k[:46],
            "%.3f" % a if a is not None else "--",
            "%.3f" % b if b is not None else "--",
        ),
        flush=True,
    )

"""Is tightpriors' +1535 nat 'mismatch' a bad port, or a stale baseline?

The 80865.224 baseline was measured BEFORE the t_0 key was fixed: at that
point the params file said `lens.DC2018_128.t_0` -- the RUN name, which
matches no instance -- so the t_0 tightening was silently absent (that is
review 2.3.16).  The key was corrected afterwards and the BOUNDS were
re-verified, but the start logp was never re-measured.

So the prediction is: drop the t_0 bound from the ported config and the
logp should fall back to ~80865, because that is the model the baseline
actually measured.  If it does, the port is fine and the baseline is stale.
If it does not, the port changed something and needs investigating.
"""

import copy
import io
import logging
import os

import numpy as np
import yaml

logging.disable(logging.WARNING)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs"))
from exozippy.system import System  # noqa: E402

cfg = yaml.safe_load(io.open("DC2018_128_tightpriors.yaml", encoding="utf-8"))
base = yaml.safe_load(
    io.open("DC2018_128_tightpriors.params.yaml", encoding="utf-8")
)

for label, params in (
    ("AS PORTED (t_0 bound active)", base),
    (
        "t_0 BOUND REMOVED",
        {k: v for k, v in base.items() if k != "source.Source.t_0"},
    ),
):
    s = System(copy.deepcopy(cfg), user_params=copy.deepcopy(params))
    s.prepare()
    m = s.build_model()
    ip = m.initial_point()
    lp = float(m.compile_logp()(ip))
    t0 = [p for p in s.get_all_parameters() if p.label.endswith(".t_0")]
    b = (
        (
            "%.1f..%.1f"
            % (np.atleast_1d(t0[0].lower)[0], np.atleast_1d(t0[0].upper)[0])
        )
        if t0
        else "?"
    )
    print("%-32s logp = %12.3f   t_0 prior %s" % (label, lp, b), flush=True)

print("\nbaseline measured pre-fix: 80865.224", flush=True)
print(
    "If 't_0 BOUND REMOVED' reproduces it, the baseline was stale and the",
    flush=True,
)
print("port is sound.", flush=True)

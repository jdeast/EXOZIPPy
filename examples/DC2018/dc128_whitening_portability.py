"""Why are raw trace coordinates not portable across a rebuild? (2.4.15)

Two scripts have now produced confident nonsense by mapping a trace's stored
`*_raw` values back to physical ones through a freshly built model:
dc128_severed_v3_pull.py (rho = 1.5e-6, theta_E = 1e-4 mas) and
dc128_severed_v3_followup.py, whose guard caught it -- the rebuilt graph
missed the cold group's stored `lens.log_rho` by 3.65 in log10.

The HYPOTHESIS is the whitening state: init_scale/bound_scale are
CURVATURE-MEASURED at the start point (whitening.md), and the affine map
between raw and physical is exactly those scales.  A rebuild that re-runs
the probe from a different start therefore gets a different map, and raw
coordinates silently mean something else.

This tests it WITHOUT the trace, because the trace is not needed: build the
same config twice in one process and compare the two builds' own
raw->physical maps.  If two builds of the SAME config already disagree, the
mechanism is established and nothing about the trace is implicated.  If they
agree, the hypothesis is wrong and the cause is elsewhere (column mapping,
element order), which is worth knowing too -- so this script reports both
outcomes rather than only the one it expects.

Probes at three raw values per parameter so a pure OFFSET difference and a
pure SCALE difference can be told apart.
"""

import io
import logging
import os

import numpy as np
import yaml

logging.disable(logging.WARNING)
os.chdir("/home/jeastman/python/EXOZIPPy/examples/DC2018/configs")

import pytensor  # noqa: E402

from exozippy.system import System  # noqa: E402

CFG = "DC2018_128_severed_v3.yaml"
WATCH = ["lens.log_rho", "lens.log_theta_E", "star.radius", "star.distance"]
PROBES = [-1.0, 0.0, 1.0]


def build_maps(label):
    cfg = yaml.safe_load(io.open(CFG, encoding="utf-8"))
    s = System(cfg, user_params=None)
    s.prepare()
    m = s.build_model()
    ip = m.initial_point()
    dets = {d.name: d for d in m.deterministics}
    names = [v.name for v in m.value_vars]
    out = {}
    for w in WATCH:
        raw_name = w + "_raw"
        if w not in dets or raw_name not in names:
            continue
        fn = pytensor.function(m.value_vars, dets[w], on_unused_input="ignore")
        base = [np.asarray(ip[n]) for n in names]
        k = names.index(raw_name)
        vals = []
        for probe in PROBES:
            args = [np.array(b, copy=True) for b in base]
            args[k] = np.full_like(np.atleast_1d(args[k]), probe, dtype=float)
            vals.append(float(np.atleast_1d(np.asarray(fn(*args)))[0]))
        out[w] = vals
        print("  %-9s %-22s raw %s -> phys %s"
              % (label, w, PROBES, ["%.6g" % v for v in vals]), flush=True)
    for attr in ("whitening_state", "_whitening_state", "whitening"):
        st = getattr(s, attr, None)
        if st is not None:
            print("  %-9s system.%s present: %s"
                  % (label, attr, type(st).__name__), flush=True)
            break
    return out


print("=== BUILD 1 ===", flush=True)
a = build_maps("build-1")
print("=== BUILD 2 (same config, same process) ===", flush=True)
b = build_maps("build-2")

print("\n=== DO TWO BUILDS OF THE SAME CONFIG AGREE? ===", flush=True)
worst = 0.0
for w in WATCH:
    if w not in a or w not in b:
        continue
    d = np.abs(np.array(a[w]) - np.array(b[w]))
    worst = max(worst, float(d.max()))
    sa = (a[w][2] - a[w][0]) / 2.0
    sb = (b[w][2] - b[w][0]) / 2.0
    print("  %-22s max|diff| %.3e   slope %.6g vs %.6g   offset %.6g vs %.6g"
          % (w, d.max(), sa, sb, a[w][1], b[w][1]), flush=True)

print("\nworst disagreement across watched parameters: %.3e" % worst,
      flush=True)
if worst > 1e-9:
    print("""CONFIRMED: two builds of the SAME config produce DIFFERENT
raw->physical maps, so a stored raw coordinate has no build-independent
meaning and 2.4.15 is established.  Compare the slope/offset columns: a
slope difference is a whitening SCALE difference, an offset-only difference
is a re-centred start.""", flush=True)
else:
    print("""NOT CONFIRMED: two builds agree to machine precision, so the
map is build-stable and the 3.65 discrepancy against the TRACE has some
other cause -- most likely that the RUN's whitening was probed from a
polished/different start than a fresh build gets, or an element-order
mismatch.  Next step is to compare against the run's persisted whitening
state rather than against another fresh build.""", flush=True)

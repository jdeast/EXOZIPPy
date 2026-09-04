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
        # THE FIX (2.4.15 trap 1).  model.deterministics are expressed over
        # the RANDOM variables, so compiling them against value_vars gives a
        # function that NEVER READS ITS INPUTS and evaluates a graph still
        # containing RVs -- i.e. it returns a fresh PRIOR DRAW.  The first
        # version of this script did exactly that: every probe returned
        # -5.96195 and the two builds differed only because each drew a new
        # sample, which the script then misreported as a whitening
        # difference.  replace_rvs_by_values reconnects the value vars.
        node = m.replace_rvs_by_values([dets[w]])[0]
        fn = pytensor.function(m.value_vars, node, on_unused_input="ignore")
        base = [np.asarray(ip[n]) for n in names]
        k = names.index(raw_name)
        nel = int(np.atleast_1d(base[k]).size)

        def sweep(el):
            """Sweep RAW element `el`; return the PHYSICAL element that moves.

            The raw and physical vectors DO NOT LINE UP when any element is
            pinned: on severed-v3 star.radius_raw has ONE element (the free
            Source) while the physical star.radius has TWO (Lens pinned +
            Source), so raw[0] drives physical[1].  The previous version
            assumed index equality, read the pinned physical[0], saw a flat
            response and aborted -- the second false positive in this file.
            So the responding element is DISCOVERED, not assumed.
            """
            rows = []
            for probe in PROBES:
                args = [np.array(b, copy=True) for b in base]
                v = np.atleast_1d(np.array(args[k], dtype=float))
                v[el] = probe
                args[k] = v.reshape(np.asarray(args[k]).shape)
                rows.append(np.atleast_1d(np.asarray(fn(*args), dtype=float)))
            stack = np.vstack(rows)
            spread = stack.max(axis=0) - stack.min(axis=0)
            j = int(np.argmax(spread))
            return [float(r[j]) for r in rows], j

        # PER ELEMENT, because a vector Parameter can mix pinned and free
        # elements.  The first version swept the WHOLE vector and read
        # element 0; on severed-v3 that is star.Lens.radius, which is
        # PINNED (sigma: 0), so the response was flat and the guard aborted
        # on a false positive.  A flat element is now reported as pinned --
        # which is correct behaviour, not the trap-1 bug -- and only an
        # ALL-flat parameter is treated as evidence of the bug.
        swept = {el: sweep(el) for el in range(nel)}
        per_el = {el: v for el, (v, _j) in swept.items()}
        phys_el = {el: j for el, (_v, j) in swept.items()}
        live = {el: v for el, v in per_el.items() if max(v) - min(v) > 0.0}
        for el, v in per_el.items():
            print("  %-9s %-16s raw[%d] -> phys[%d]  %s -> %s%s"
                  % (label, w, el, phys_el[el], PROBES,
                     ["%.6g" % x for x in v],
                     "" if el in live else "   (flat: pinned/inactive)"),
                  flush=True)
        if not live:
            raise SystemExit(
                "ABORT: NO element of %s responds to %s.  If this parameter"
                " is not entirely pinned, the compiled function is ignoring"
                " its inputs -- see 2.4.15 trap 1.  Do not interpret"
                " anything below." % (w, raw_name))
        el0 = sorted(live)[0]
        vals = live[el0]
        out.setdefault("_element", {})[w] = phys_el[el0]
        # SELF-CHECK, and it is not optional: a flat response means the
        # input is being ignored, which is the bug above rather than a
        # result.  `on_unused_input="ignore"` is required here (some value
        # vars genuinely are unused by a given Deterministic) and it is also
        # what silences the only warning, so this check is the only guard.
        out[w] = vals
    for attr in ("whitening_state", "_whitening_state", "whitening"):
        st = getattr(s, attr, None)
        if st is not None:
            print("  %-9s system.%s present: %s"
                  % (label, attr, type(st).__name__), flush=True)
            break
    return out


# The run's own map, fitted from the cold group where raw and physical
# COEXIST.  This is the reference the rebuild has to reproduce, and it is
# the comparison the first version of this script failed to make: it
# compared two rebuilds against each other, which cannot distinguish "the
# map moved" from "both builds are broken the same way".
TRACE = "fitresults_severed_v3/DC2018_128_trace.nc"


def run_map(w, el=0):
    """(slope, intercept, raw_lo, raw_hi) of the RUN's raw->physical map.

    Fitted, not assumed: the stored pair is exact, but the fit is only valid
    over the raw range the cold chain actually visited, so that range is
    returned with it and reported.  A whitened coordinate's map is affine by
    construction, so a 2-parameter fit is the right model -- the residual is
    reported to show it holds.
    """
    import xarray as xr

    ds = xr.open_dataset(TRACE, group="posterior")
    if w not in ds.data_vars or (w + "_raw") not in ds.data_vars:
        return None
    def col(name):
        da = ds[name].isel(draw=slice(0, None, 200))
        extra = [d for d in da.dims if d not in ("chain", "draw")]
        if extra:
            e = min(el, da.sizes[extra[0]] - 1)
            da = da.isel({extra[0]: e})
        return np.asarray(da).ravel()

    r, p = col(w + "_raw"), col(w)
    m = np.isfinite(r) & np.isfinite(p)
    r, p = r[m], p[m]
    if r.size < 100 or (r.max() - r.min()) < 1e-9:
        return None
    sl, ic = np.polyfit(r, p, 1)
    res = float(np.abs(p - (sl * r + ic)).max())
    return float(sl), float(ic), float(r.min()), float(r.max()), res


print("=== BUILD 1 ===", flush=True)
a = build_maps("build-1")
print("=== BUILD 2 (same config, same process) ===", flush=True)
b = build_maps("build-2")

print("\n=== IS THE REBUILD SELF-CONSISTENT? (build 1 vs build 2) ===",
      flush=True)
for w in WATCH:
    if w in a and w in b and w != "_element":
        d = float(np.abs(np.array(a[w]) - np.array(b[w])).max())
        print("  %-22s max|diff| %.3e %s"
              % (w, d, "" if d < 1e-9 else "<- two rebuilds also disagree"),
              flush=True)

print("\n=== DOES THE REBUILD REPRODUCE THE RUN'S MAP? ===", flush=True)
print("%-22s %11s %11s %11s %11s %9s"
      % ("parameter", "slope(new)", "slope(run)", "icept(new)", "icept(run)",
         "ratio"), flush=True)
verdict = []
for w in WATCH:
    if w not in a or w == "_element":
        continue
    rm = run_map(w, a.get("_element", {}).get(w, 0))
    if rm is None:
        print("  %-20s (no stored raw/physical pair in the trace)" % w,
              flush=True)
        continue
    sl_r, ic_r, lo, hi, res = rm
    sl_n = (a[w][-1] - a[w][0]) / (PROBES[-1] - PROBES[0])
    ic_n = a[w][0] - sl_n * PROBES[0]
    ratio = sl_n / sl_r if sl_r else float("nan")
    print("%-22s %11.5g %11.5g %11.5g %11.5g %9.3g"
          % (w, sl_n, sl_r, ic_n, ic_r, ratio), flush=True)
    print("      run fit valid over raw %.4f..%.4f, max residual %.2e"
          % (lo, hi, res), flush=True)
    # evaluate both maps at the middle of the run's own sampled range
    mid = 0.5 * (lo + hi)
    print("      at raw=%.4f:  rebuild %.5g   run %.5g   DIFF %.5g"
          % (mid, sl_n * mid + ic_n, sl_r * mid + ic_r,
             (sl_n * mid + ic_n) - (sl_r * mid + ic_r)), flush=True)
    verdict.append(abs(ratio - 1.0))

print("", flush=True)
if verdict and max(verdict) > 0.01:
    print("""CONFIRMED (2.4.15 trap 2): the rebuild's raw->physical map does NOT
match the run's, so a stored raw coordinate has no build-independent meaning.
A SLOPE ratio far from 1 is a whitening SCALE difference, which is the
expected signature -- init_scale/bound_scale are curvature-measured at the
start point, so a rebuild probing from a different start gets a different
affine map.  Consequence: post-hoc analysis must read the stored posterior
Deterministics, and reconstructing from raw needs the RUN's whitening state.
""", flush=True)
elif verdict:
    print("""NOT CONFIRMED: the rebuild reproduces the run's map to better than
1%%, so raw coordinates ARE portable and 2.4.15 trap 2 should be withdrawn.
In that case the whole 3.65 discrepancy was trap 1 (the prior-draw bug) and
the hot-chain analysis can proceed on a corrected script.
""", flush=True)
else:
    print("INCONCLUSIVE: no watched parameter had both a rebuild map and a "
          "usable run fit.", flush=True)

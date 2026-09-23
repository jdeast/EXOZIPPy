"""Where do nested sampling's nats go?  Per-parameter H for DC2018-128.

Nested sampling costs about  nlive * H  iterations, where H is the
information gain (the prior->posterior KL divergence) in nats.  On the
observable arm H ~= 106, nlive = 500, and the run needed ~53,000 iterations
-- so H is not a diagnostic, it IS the budget.  The three arms differ by
20x in cost for exactly this reason (physical H ~ 2400, all-swaps ~415,
observable ~106).

The useful consequence: H is additive over independent coordinates, so it
can be decomposed per parameter and the cost read off a line at a time.
For a uniform prior on [lo, hi] and an approximately Gaussian posterior of
width s,

    H_i  ~=  log( (hi - lo) / (s * sqrt(2 pi e)) )

which is just "how many e-foldings did the data shrink this coordinate by".
A parameter the data pin to 1e-4 of its prior costs ~9 nats; one whose
posterior fills its prior costs ~0 and is free.

WHAT THIS CAN AND CANNOT TELL YOU.  A coordinate swap helps only where the
nats come from a prior that is a bad shape for the posterior -- a wide flat
box around a tight correlated ridge.  Where a parameter is genuinely
measured, the nats are real information and NO reparameterization removes
them; only a tighter prior would, and that is a physics decision rather
than a coordinate one.  So the ranked list below separates the two: large
H on a parameter the data measure is immovable, large H on a nuisance or on
one half of a known degeneracy is a swap candidate.

Priors are read from the run's own startup table (run.py prints
"U(lo, hi)" per sampled parameter), posterior widths from its trace, so the
two always refer to the same fit.

Run:  python3 dc128_h_decomposition.py [--log <cfg log>] [--trace <nc>]
"""

import argparse
import math
import re

import numpy as np

# "  name | value | scale | units | logprob | U(lo, hi)[*]"
ROW = re.compile(
    r"^\s*([A-Za-z_][\w.]*)\s*\|\s*(-?[\d.eE+-]+)\s*\|.*?\|\s*"
    r"([UN])\(([^)]*)\)\s*\*?\s*$"
)
HALF_LOG_2PIE = 0.5 * math.log(2.0 * math.pi * math.e)


def parse_priors(path):
    """{param: (kind, lo, hi)} from the startup table."""
    out = {}
    for line in open(path, errors="ignore"):
        m = ROW.match(line.rstrip())
        if not m:
            continue
        name, _val, kind, args = m.groups()
        try:
            lo, hi = [float(x) for x in args.split(",")[:2]]
        except ValueError:
            continue
        out[name] = (kind, lo, hi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="dc18_cfg.15363589.log")
    ap.add_argument(
        "--trace",
        default="configs/fitresults_mulens_hoggcap_observable/"
        "DC2018_128_trace.nc",
    )
    ap.add_argument("--nlive", type=float, default=500.0)
    args = ap.parse_args()

    priors = parse_priors(args.log)
    print("parsed %d priors from %s" % (len(priors), args.log), flush=True)

    # comp.INSTANCE.param -> element index, in first-appearance order per
    # (component, parameter).  That is the order the component builds its
    # vector in, so it is the order the trace's trailing dim carries.
    instance_index, seen = {}, {}
    for name in priors:
        parts = name.split(".")
        if len(parts) != 3:
            continue
        key = (parts[0], parts[2])
        order = seen.setdefault(key, [])
        if parts[1] not in order:
            order.append(parts[1])
        instance_index[name] = order.index(parts[1])

    import xarray as xr

    ds = xr.open_dataset(args.trace, group="posterior")

    rows = []
    for name, (kind, lo, hi) in sorted(priors.items()):
        # The trace stores the PHYSICAL name; per-element vars carry a
        # trailing dim.  Match on the component.parameter stem.
        stem = name
        parts = name.split(".")
        if len(parts) == 3:  # comp.instance.param
            stem = "%s.%s" % (parts[0], parts[2])
        if stem not in ds.data_vars:
            continue
        da = ds[stem]
        extra = [d for d in da.dims if d not in ("chain", "draw")]
        arr = np.asarray(da)
        if extra:
            arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
            # A per-instrument parameter is ONE vector in the trace but has
            # one startup-table row PER INSTANCE.  Emitting every element for
            # every instance counts each quantity n_instance times over --
            # it inflated the first version of this table by ~70 nats.  Map
            # the instance to its own element instead.
            idx = instance_index.get(name)
            if idx is None or idx >= arr.shape[2]:
                continue
            cols = [arr[:, :, idx]]
        else:
            cols = [arr]
        for k, c in enumerate(cols):
            s = float(np.nanstd(c))
            if not np.isfinite(s) or s <= 0:
                continue
            width = abs(hi - lo)
            if kind != "U" or not np.isfinite(width) or width <= 0:
                continue
            h = math.log(width) - math.log(s) - HALF_LOG_2PIE
            rows.append((max(h, 0.0), name, width, s))

    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    print(
        "\n%-42s %12s %12s %8s %6s"
        % ("parameter", "prior width", "post. sd", "H (nats)", "% of H")
    )
    print("-" * 88)
    for h, name, w, s in rows:
        print(
            "%-42s %12.4g %12.4g %8.2f %5.1f%%"
            % (name, w, s, h, 100 * h / total if total else 0)
        )
    print("-" * 88)
    print(
        "%-42s %12s %12s %8.2f"
        % ("TOTAL (sum of independent terms)", "", "", total)
    )
    print(
        "\nNS cost implied: nlive * H = %.0f * %.1f = %.0f iterations"
        % (args.nlive, total, args.nlive * total)
    )
    print("The run's own measured H (logLmax - logZ) was ~106 nats; a sum")
    print("far above that means the coordinates are CORRELATED (the joint")
    print("posterior is smaller than the product of its margins), and the")
    print("excess is exactly what a well-chosen swap can recover.")


if __name__ == "__main__":
    main()

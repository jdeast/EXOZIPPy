"""Is event 227's planet detectable at all, at MMEXOFAST's detection threshold?

`AnomalyFinderGridSearch.check_successive` is the gate that rejects every
window on 6 of the 44 DC2018 events, and it is a SIGNAL-DETECTION test, not
the data-sufficiency test its name and my PR #13 error message suggest:

    "at least three successive points >=2 sigma away from the zero-residual
     curve"   ->   n_success > 3 on |flux/err_flux| > 2

The instrumented gate showed 99.92% of windows have more than 5 good points,
so the point count is never the blocker; the sole discriminator between event
62 (550 windows pass) and events 227/8 (zero) is this test.

So the question is whether those events contain FOUR CONSECUTIVE 2-sigma
residuals anywhere.  That is measurable directly from the data and the
point-lens fit, with no reference to the truth table -- in particular no
reference to alpha, whose DC2018 convention this project has established is
unmappable to ours.

The longest run in the FULL residual series is an UPPER BOUND on the longest
run in any window: a window is a contiguous time interval, so a run inside one
is a run in the full series.  Therefore if the full-series maximum is below 4,
no window can pass and the null detection is CORRECT -- the planet is simply
not in the photometry at this threshold.  If it is 4 or more, the threshold or
the windowing is costing real detections.

(One caveat the measurement itself raises: check_successive sorts by time but
never checks contiguity, so a window spanning an observing gap could
concatenate two separated deviations into a spurious "run".  That would make
the bound non-strict in the other direction; it is reported, not assumed.)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc  # noqa: E402
from dc18_highmag_landscape import (  # noqa: E402
    _chi2_linear_flux,
    _fit_pspl,
    _load,
)


def _longest_run(mask):
    """Longest run of consecutive True values."""
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def one_event(event, band, sigma_cut, need_run):
    import MulensModel as mm

    t, flux, ferr, mm = _load(event, band)
    truth, _ = dc.load_truth(dc.data_dir_or_raise(None), event)
    pspl, chi2 = _fit_pspl(t, flux, ferr, mm, truth)

    model = mm.Model({k: pspl[k] for k in ("t_0", "u_0", "t_E")})
    A = model.get_magnification(t)
    # Same linear (f_source, f_blend) solve the chi2 uses, so the residuals
    # are the ones the pipeline's point-lens fit would leave behind.
    w = 1.0 / ferr**2
    a11 = np.sum(w * A * A)
    a12 = np.sum(w * A)
    a22 = np.sum(w)
    b1 = np.sum(w * A * flux)
    b2 = np.sum(w * flux)
    det = a11 * a22 - a12 * a12
    fs = (a22 * b1 - a12 * b2) / det
    fb = (a11 * b2 - a12 * b1) / det
    resid = (flux - (fs * A + fb)) / ferr

    order = np.argsort(t)
    dev = np.abs(resid[order]) > sigma_cut
    longest = _longest_run(dev)
    # Where the longest run sits, in days from the fitted peak -- an anomaly
    # should not be at the peak itself.
    idx = 0
    best = run = 0
    for i, v in enumerate(dev):
        run = run + 1 if v else 0
        if run > best:
            best, idx = run, i - run + 1
    return dict(
        event=event,
        n_points=int(t.size),
        pspl_chi2=round(float(chi2), 1),
        pspl_u_0=round(float(pspl["u_0"]), 5),
        truth_u_0=round(float(truth["u_0"]), 5),
        truth_s=round(float(truth["s"]), 4),
        truth_q=float(truth["q"]),
        n_above=int(dev.sum()),
        frac_above=round(float(dev.mean()), 5),
        longest_run=int(longest),
        run_starts_days_from_peak=round(float(t[order][idx] - pspl["t_0"]), 3),
        passes_threshold=bool(longest > need_run),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default="227,8,1,124,208,289,62,12,32")
    ap.add_argument("--band", default="W149")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument(
        "--need-run",
        type=int,
        default=3,
        help="check_successive requires n_success > this (i.e. 4 points)",
    )
    args = ap.parse_args()

    print(
        f"threshold: more than {args.need_run} consecutive "
        f"|resid| > {args.sigma} sigma  (band {args.band})\n",
        flush=True,
    )
    hdr = (
        f"{'ev':>4s} {'pts':>6s} {'truth u_0':>10s} {'truth s':>8s} "
        f"{'truth q':>9s} {'n>2sig':>7s} {'frac':>8s} {'longest':>8s} "
        f"{'d from pk':>10s} {'passes':>7s}"
    )
    print(hdr, flush=True)
    for ev in [int(e) for e in args.events.split(",") if e.strip()]:
        try:
            r = one_event(ev, args.band, args.sigma, args.need_run)
            print(
                f"{r['event']:4d} {r['n_points']:6d} {r['truth_u_0']:10.5f} "
                f"{r['truth_s']:8.4f} {r['truth_q']:9.2e} "
                f"{r['n_above']:7d} {r['frac_above']:8.5f} "
                f"{r['longest_run']:8d} "
                f"{r['run_starts_days_from_peak']:10.2f} "
                f"{str(r['passes_threshold']):>7s}",
                flush=True,
            )
        except BaseException as exc:
            print(f"{ev:4d}  FAILED {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()

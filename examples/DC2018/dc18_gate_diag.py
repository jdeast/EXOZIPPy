"""Why does AnomalyFinder's data-sufficiency gate reject EVERY window on 6 of 44?

Six of the 44 DC2018 planetary events (1, 8, 124, 208, 227, 289) produce zero
fittable anomaly-grid windows out of 94375, where healthy events get 480-550
and event 62 gets 550.  That is 14% of the sample getting no binary-lens seeds,
for a reason unrelated to the high_mag solver -- and it will scale to the
293-event set.

Event 227 makes it hard to believe it is a genuine data property: its truth is
q = 8.1e-3 at s = 0.496, a close binary with a nearly 1% mass ratio whose
planetary caustics sit at |1/s - s| = 1.52 Einstein radii while the source
passes at u_0 = 0.54.  An anomaly should be there, on a light curve with 38568
points at ~22 points/day.

The gate is in AnomalyFinderGridSearch.do_fits:

    if len(trimmed_datasets) >= 1:
        n_tot = np.sum(np.hstack([d.good for d in trimmed_datasets]))
        successive = self.check_successive(trimmed_datasets)
        if (n_tot > 5) and (successive):
            do_fit = True

so a window is rejected when the trim leaves no datasets, or fewer than 6 good
points, or no successive coverage.  This instruments WHICH of those three fires
and with what numbers, per window, rather than inferring it -- my earlier
attempt recorded `n_good_residual_points` and read 0 even for the HEALTHY
events, so that instrumentation was broken and told me nothing.

Run 227 (empty) against 62 (healthy, 550 windows, same code path) as control.
"""

import argparse
import collections
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc  # noqa: E402


def one_event(event, max_records):
    out = {"event": event}
    try:
        import mmexofast.gridsearches as gs
        from astropy.coordinates import SkyCoord

        from exozippy.components.mulensing import mmexofast_support

        tally = collections.Counter()
        n_tot_seen = []
        n_trimmed_seen = []
        real_do_fits = gs.AnomalyFinderGridSearch.do_fits
        real_trim = gs.AnomalyFinderGridSearch.get_trimmed_datasets
        real_succ = gs.AnomalyFinderGridSearch.check_successive

        def spy_do_fits(self, parameters, verbose=False):
            trimmed = real_trim(self, parameters, verbose=verbose)
            n_ds = len(trimmed)
            n_trimmed_seen.append(n_ds)
            if n_ds < 1:
                tally["no_datasets_after_trim"] += 1
            else:
                n_tot = int(np.sum(np.hstack([d.good for d in trimmed])))
                n_tot_seen.append(n_tot)
                succ = bool(real_succ(self, trimmed))
                if n_tot <= 5 and not succ:
                    tally["too_few_AND_not_successive"] += 1
                elif n_tot <= 5:
                    tally["too_few_points"] += 1
                elif not succ:
                    tally["not_successive"] += 1
                else:
                    tally["PASSED"] += 1
            return real_do_fits(self, parameters, verbose=verbose)

        gs.AnomalyFinderGridSearch.do_fits = spy_do_fits
        try:
            d = dc.data_dir_or_raise(None)
            files = dc.light_curve_files(d, event, ("W149", "Z087"))
            ra, dec = dc.event_coords(d, event)
            coords = SkyCoord(ra, dec, unit="deg").to_string(style="hmsdms")
            jpath = Path("gatediag") / f"ev{event:03d}_mmexofast.json"
            jpath.parent.mkdir(parents=True, exist_ok=True)
            if jpath.exists():
                jpath.unlink()  # a cache would hide the behaviour under study
            try:
                mmexofast_support.run_or_load(
                    jpath,
                    list(files.values()),
                    coords=coords,
                    fit_type="binary_lens",
                    renormalize_errors=True,
                    no_parallax=True,
                    options={
                        "limb_darkening_coeffs_gamma": {
                            "W149": 0.0,
                            "Z087": 0.0,
                        },
                        "pool": 1,
                        "stop_before": (
                            "search_for_anomaly:"
                            "get_anomaly_light_curve_parameters"
                        ),
                    },
                )
                out["raised"] = None
            except BaseException as exc:
                out["raised"] = f"{type(exc).__name__}: {exc}"[:160]
        finally:
            # try/FINALLY so a failure cannot leave the patch (or, in the
            # architecture post-hoc script, a log handler) in place for the
            # next event.
            gs.AnomalyFinderGridSearch.do_fits = real_do_fits

        out["gate_tally"] = dict(tally)
        out["windows_examined"] = int(sum(tally.values()))
        if n_tot_seen:
            a = np.asarray(n_tot_seen)
            out["n_good_in_window"] = {
                "min": int(a.min()),
                "median": int(np.median(a)),
                "max": int(a.max()),
                "mean": round(float(a.mean()), 2),
                "frac_gt5": round(float(np.mean(a > 5)), 4),
            }
        if n_trimmed_seen:
            b = np.asarray(n_trimmed_seen)
            out["datasets_after_trim"] = {
                "min": int(b.min()),
                "max": int(b.max()),
                "frac_zero": round(float(np.mean(b == 0)), 4),
            }
        truth, _ = dc.load_truth(dc.data_dir_or_raise(None), event)
        out["truth"] = {
            k: round(float(truth[k]), 6)
            for k in ("t_0", "u_0", "t_E", "s", "q", "rho")
        }
    except BaseException as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc().splitlines()[-4:]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--events",
        default="227,62",
        help="empty-grid event(s) first, then a HEALTHY control",
    )
    ap.add_argument("--max-records", type=int, default=0)
    args = ap.parse_args()

    for ev in [int(e) for e in args.events.split(",") if e.strip()]:
        r = one_event(ev, args.max_records)
        print(f"\n===== event {ev} =====", flush=True)
        for k, v in r.items():
            if k != "event":
                print(f"    {k}: {v}", flush=True)


if __name__ == "__main__":
    main()

"""Why does AnomalyFinder's grid come back all-NaN on 4 of the 44 DC2018 events?

Events 1, 124, 208 and 289 die identically inside
``search_for_anomaly:run_anomaly_search`` with ``ValueError: All-NaN slice
encountered``.  Reading the code, ``AnomalyFinderGridSearch.best`` takes
``np.nanargmax(self.anomalies[:, 5])``, and column 5 is
``results[:, 3] - results[:, j-1]`` -- both of which ``do_fits`` leaves at NaN
whenever its data-sufficiency gate (>= 5 good points in the trimmed window AND
``check_successive``) rejects a grid point.  So the hypothesis is: EVERY grid
point was rejected, and the all-NaN column is the honest report of that rather
than a numerical accident.

This measures it instead of assuming it.  For each event it captures the live
grid-search object, then reports how many grid points produced a finite chi2 in
each column.  Run it on failing AND working events -- a working event is the
control that says the instrumentation reads what I think it reads.

    python dc18_afgrid_diag.py --events 124,208,12,32
"""

import argparse
import multiprocessing as mp
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc  # noqa: E402


def one_event(event):
    out = {"event": event}
    try:
        import mmexofast.gridsearches as gs
        from astropy.coordinates import SkyCoord

        from exozippy.components.mulensing import mmexofast_support

        # Capture the live grid object: the exception is raised from a
        # property deep inside, so there is no other handle on it.
        captured = []
        orig_run = gs.AnomalyFinderGridSearch.run

        def spy_run(self, *a, **kw):
            captured.append(self)
            return orig_run(self, *a, **kw)

        gs.AnomalyFinderGridSearch.run = spy_run

        data_dir = dc.data_dir_or_raise(None)
        files = dc.light_curve_files(data_dir, event, ("W149", "Z087"))
        ra, dec = dc.event_coords(data_dir, event)
        coords = SkyCoord(ra, dec, unit="deg").to_string(style="hmsdms")
        jpath = Path("afdiag") / f"ev{event:03d}_mmexofast.json"
        jpath.parent.mkdir(parents=True, exist_ok=True)

        try:
            mmexofast_support.run_or_load(
                jpath,
                list(files.values()),
                coords=coords,
                fit_type="binary_lens",
                renormalize_errors=True,
                no_parallax=True,
                options={
                    "limb_darkening_coeffs_gamma": {"W149": 0.0, "Z087": 0.0},
                    "pool": 1,
                    # Stop right AFTER the step under study, so a working
                    # event costs the same as a failing one.
                    "stop_before": "search_for_anomaly:"
                    "get_anomaly_light_curve_parameters",
                },
            )
            out["raised"] = None
        except BaseException as exc:
            out["raised"] = f"{type(exc).__name__}: {exc}"
            out["trace_tail"] = (
                traceback.format_exc().strip().splitlines()[-6:]
            )
        finally:
            gs.AnomalyFinderGridSearch.run = orig_run

        if not captured:
            out["note"] = "grid search never ran"
            return out

        g = captured[-1]
        r = getattr(g, "results", None)
        if r is None:
            out["note"] = "results is None"
            return out
        r = np.asarray(r, dtype=float)
        out["results_shape"] = list(r.shape)
        # Columns are [chi2_j1, chi2_j2, chi2_flat, chi2_zero]; do_fits leaves
        # every one of them NaN together when the gate rejects a grid point.
        out["finite_per_column"] = [
            int(np.isfinite(r[:, c]).sum()) for c in range(r.shape[1])
        ]
        out["rows_all_nan"] = int((~np.isfinite(r)).all(axis=1).sum())
        out["rows_any_finite"] = int(np.isfinite(r).any(axis=1).sum())
        # The column the crash actually reads: dchi2_zero for j=1 and j=2.
        for j in (1, 2):
            col5 = r[:, 3] - r[:, j - 1]
            out[f"dchi2_zero_j{j}_finite"] = int(np.isfinite(col5).sum())
        n_data = sum(int(np.sum(d.good)) for d in getattr(g, "residuals", []))
        out["n_good_residual_points"] = n_data
    except BaseException as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--events",
        default="124,208,1,289,12,32",
        help="comma-separated; include working events as controls",
    )
    ap.add_argument("--ncpu", type=int, default=6)
    args = ap.parse_args()

    events = [int(e) for e in args.events.split(",") if e.strip()]
    with mp.Pool(min(args.ncpu, len(events))) as pool:
        rows = pool.map(one_event, events)

    for r in sorted(rows, key=lambda x: x["event"]):
        print(f"\n===== event {r['event']} =====")
        for k, v in r.items():
            if k != "event":
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()

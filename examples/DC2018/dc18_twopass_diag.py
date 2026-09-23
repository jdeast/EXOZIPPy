"""Who calls MMEXOFASTFitter.fit() a second time after the anomaly search fails?

Events 227 and 008 -- exactly the two whose `run_anomaly_search` died with
"All-NaN slice encountered" -- log the workflow plan TWICE and run the whole
pipeline twice.  Every other event logs it once.  So the failure triggers a
re-entry into fit(), and that re-entry is consequential: on event 227 the FIRST
pass found the true event (t_0 within 0.37 d of truth, u_0 = 0.606 against
0.543, chi2 39470.6) and the SECOND found a spurious candidate 337 days away
with a WORSE chi2 (40395.8) -- and it is the second one the classifier saw,
which is why an A_max ~ 2 event was labelled high_mag and got no seeds.

Static reading did not settle who re-enters.  `MMEXOFASTFitter.fit()` breaks on
a failed step and does not retry; `WorkflowStep.run()` re-raises only when
`required`; `run_or_load` calls `fitter.fit()` exactly once.  So this
instruments it instead of guessing again: spy on fit() and on the failing
grid property, printing the call stack each time.
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dc18_common as dc  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", type=int, default=227)
    args = ap.parse_args()

    import mmexofast as mmexo
    from astropy.coordinates import SkyCoord

    from exozippy.components.mulensing import mmexofast_support

    calls = {"fit": 0}
    real_fit = mmexo.MMEXOFASTFitter.fit

    def spy_fit(self, *a, **kw):
        calls["fit"] += 1
        print(
            f"\n########## MMEXOFASTFitter.fit() CALL #{calls['fit']} ##########",
            flush=True,
        )
        print("".join(traceback.format_stack()[-9:-1]), flush=True)
        print(
            f"    fit_type={self.fit_type!r} "
            f"completed_steps={len(getattr(self, 'completed_steps', []))}",
            flush=True,
        )
        try:
            return real_fit(self, *a, **kw)
        except BaseException as exc:
            print(
                f"    fit() #{calls['fit']} RAISED "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

    mmexo.MMEXOFASTFitter.fit = spy_fit

    d = dc.data_dir_or_raise(None)
    files = dc.light_curve_files(d, args.event, ("W149", "Z087"))
    ra, dec = dc.event_coords(d, args.event)
    coords = SkyCoord(ra, dec, unit="deg").to_string(style="hmsdms")
    out = Path("twopass") / f"ev{args.event:03d}_mmexofast.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()  # never let a cache hide the behaviour under study

    try:
        mmexofast_support.run_or_load(
            out,
            list(files.values()),
            coords=coords,
            fit_type="binary_lens",
            renormalize_errors=True,
            no_parallax=True,
            options={
                "limb_darkening_coeffs_gamma": {"W149": 0.0, "Z087": 0.0},
                "pool": 1,
                "stop_before": "fit_binary_lens:fit_binary_lens_models",
            },
        )
        print("\nrun_or_load returned normally", flush=True)
    except BaseException as exc:
        print(f"\nrun_or_load RAISED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()

    print(f"\n===== fit() was called {calls['fit']} time(s) =====", flush=True)


if __name__ == "__main__":
    main()

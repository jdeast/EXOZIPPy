"""Localize the transit leg's +3 sigma tc bias (7.13.1).

The bias is real and systematic: over 28 converged realizations the fitted
mid-time sits +3.2 +/- 0.45 sigma above truth, with a PIT mean of 0.081.
Three explanations have already been ELIMINATED by measurement rather than
argument:

  * a time-SCALE conversion (UTC->TDB and the like).  instrument.py defaults
    to time_scale "tdb" / time_frame "bjd", and the harness writes plain
    BJD_TDB, so nothing is converted.
  * light-travel time across the orbit.  dtc / (a/c) has median 0.68 and sd
    0.64 over the 30 rows, ranging 0.63-3.79; a straight LTT omission would
    pin that ratio at 1.0.
  * a tc/period trade-off in a 3-transit fit.  corr(pull_tc, pull_period) =
    +0.017, i.e. nothing.

What the correlations DO say is corr(dtc_seconds, sigma_tc) = +0.903: the
offset scales with the uncertainty, so it is a roughly constant PULL rather
than a fixed physical shift.

So this script stops correlating and compares the two light curves directly.
It builds the model at the injected truth, compiles a function from
Transit._model_flux_node (the per-observation prediction the likelihood uses
-- kept as a plain attribute precisely so tests can do this), evaluates it on
a fine grid through the first transit, and reports where ITS minimum falls
against the analytic generator's.  If the two curves' centres differ, the
mismatch is visible directly and in seconds; if they agree, the bias is in
the inference rather than the forward model and the next suspect is the
prior on tc.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inject_recover as ir  # noqa: E402


def main():
    seed = int(os.environ.get("DIAG_SEED", "1010"))
    snr = float(os.environ.get("DIAG_SNR", "100"))
    nep = int(os.environ.get("DIAG_NEP", "400"))

    rng = np.random.default_rng(seed)
    wd = Path(tempfile.mkdtemp(prefix="diag_tc_"))
    os.chdir(wd)
    config, params, sampler, truth, checks, _tc = ir.make_transit(
        rng, snr, nep, wd
    )
    tc_true = truth["orbit.tc"]
    t14 = truth["planet.t14"]
    print(
        "truth: tc=%.8f  P=%.6f  p=%.5f  b=%.4f  t14=%.6f d"
        % (
            tc_true,
            truth["orbit.period"],
            truth["planet.p"],
            truth["planet.b"],
            t14,
        ),
        flush=True,
    )

    # EVALUATE AT EXACT TRUTH, not at the harness's perturbed start.  The
    # first version of this diagnostic did not, which made its 198 s
    # "MODEL - GENERATOR" offset meaningless: the params carried ~5% start
    # perturbations AND (at the time) a chord/p/b inconsistency that put the
    # model's b at 0.794 against the generator's 0.547, so the two curves
    # were of different geometries and of course disagreed.  Comparing
    # forward models requires identical parameters.
    p_true = truth["planet.p"]
    b_true = truth["planet.b"]
    params["planet.b.p"] = {"initval": float(p_true)}
    params["orbit.b.chord"]["initval"] = float(
        np.sqrt(max((1.0 + p_true) ** 2 - b_true**2, 1e-12))
    )
    params["orbit.b.period"] = {"initval": float(truth["orbit.period"])}
    params["orbit.b.tc"] = {"initval": float(tc_true)}

    import pytensor

    from exozippy.system import System

    # A FINE GRID through the first transit, replacing the scattered
    # realization times: the question is where the model's centre is, and a
    # grid answers it without the random sampling adding noise.
    grid = np.linspace(tc_true - 0.9 * t14, tc_true + 0.9 * t14, 4001)
    np.savetxt(
        wd / "synth.trn",
        np.column_stack([grid, np.ones_like(grid), np.full_like(grid, 1e-4)]),
    )

    system = System(dict(config), user_params=dict(params))
    system.prepare()
    model = system.build_model()
    node = system.transit._model_flux_node
    ip = model.initial_point()
    fn = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    flux_model = np.asarray(fn(*[ip[v.name] for v in model.value_vars]))

    # The generator's own curve on the same grid, same truth.
    p_r = truth["planet.p"]
    period = truth["orbit.period"]
    G_SUN = 2942.2062
    mstar = 10 ** params["star.A.logmass"]["initval"]
    rstar = params["star.A.radius"]["initval"]
    a = (G_SUN * mstar * period**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
    ar = a / rstar
    b = truth["planet.b"]
    cosi = b / ar
    phase = 2.0 * np.pi * (grid - tc_true) / period
    z = ar * np.sqrt(np.sin(phase) ** 2 + (cosi * np.cos(phase)) ** 2)
    z = np.where(np.cos(phase) >= 0.0, z, ar)
    flux_gen = 1.0 - ir._overlap_area(z, p_r) / np.pi

    def centroid(t, f):
        """Flux-weighted centre of the decrement -- robust to a flat bottom,
        where an argmin is quantisation noise."""
        w = np.maximum(1.0 - f, 0.0)
        return float((t * w).sum() / w.sum()) if w.sum() > 0 else float("nan")

    c_model = centroid(grid, flux_model)
    c_gen = centroid(grid, flux_gen)
    print(
        "\ndepth  model=%.6f  generator=%.6f"
        % (1 - flux_model.min(), 1 - flux_gen.min()),
        flush=True,
    )
    print(
        "centre model    = %.8f  (%.2f s vs truth)"
        % (c_model, (c_model - tc_true) * 86400),
        flush=True,
    )
    print(
        "centre generator= %.8f  (%.2f s vs truth)"
        % (c_gen, (c_gen - tc_true) * 86400),
        flush=True,
    )
    print(
        "MODEL - GENERATOR = %.2f s" % ((c_model - c_gen) * 86400), flush=True
    )
    resid = flux_model - flux_gen
    print(
        "\nmax |model - generator| flux residual = %.3e (depth %.4f)"
        % (np.abs(resid).max(), 1 - flux_gen.min()),
        flush=True,
    )
    # Where does the residual peak?  An asymmetric residual shifts the centre;
    # a symmetric one only changes the depth.
    k = int(np.argmax(np.abs(resid)))
    print(
        "largest residual at t - tc = %+.4f d (%+.2f t14)"
        % (grid[k] - tc_true, (grid[k] - tc_true) / t14),
        flush=True,
    )
    lo = resid[grid < tc_true].sum()
    hi = resid[grid > tc_true].sum()
    print(
        "residual sum before/after tc: %+.4e / %+.4e  (asymmetry %+.4e)"
        % (lo, hi, hi - lo),
        flush=True,
    )


if __name__ == "__main__":
    main()

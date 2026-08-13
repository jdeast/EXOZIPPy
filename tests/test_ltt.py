"""Analytic light-travel-time (Roemer delay) helper (components/ltt.py).

Phase 1 validation: analytic spot-checks (no external reference files/IDL
run available in this environment -- see the design doc), plus a gradient/
NUTS-sampling smoke test (design doc Validation item 3) proving
solve_delay's pt.switch is actually differentiable through the branch
selection, not just correct in forward value. Not wired into any consumer
component yet.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from exozippy.components import ltt
from exozippy.constants import C_LIGHT_RSUN_PER_DAY, RSUN_TO_AU

# The IAU-defined light time for one astronomical unit (tau_A) -- an
# independent literature value, NOT re-derived from C_LIGHT_RSUN_PER_DAY/
# RSUN_TO_AU, so comparing against it is a real external check rather than
# a tautology about our own constants agreeing with themselves.
_TAU_A_SECONDS = 499.004854

_SECONDS_PER_DAY = 86400.0


def _eval(tensor_expr, **float_inputs):
    """Evaluate a PyTensor expression built from dscalar symbols fed as
    float64 -- deliberately not pt.as_tensor_variable(<python float>),
    which autocasts to float32 (the smallest dtype that fits the literal)
    and silently loses precision. Same reasoning/pattern as
    test_torres.py's _call helper.
    """
    symbols = {name: pt.dscalar(name) for name in float_inputs}
    givens = {symbols[name]: float(v) for name, v in float_inputs.items()}
    return float(tensor_expr(**symbols).eval(givens))


def test_delay_amplitude_matches_light_time_per_au():
    """
    Given a body offset by exactly 1 AU along the line of sight with zero
    velocity and acceleration (az forced to exactly 0, so solve_delay takes
    the linear-fallback branch exactly rather than approximately),
    When solve_delay is evaluated,
    Then |delay| matches tau_A = 499.004854 s, the independently known
    IAU light-time-per-AU constant.
    """
    z_rsun = (1.0 / RSUN_TO_AU) * 1.0  # 1 AU, in R_sun

    delay_days = _eval(
        lambda z: ltt.solve_delay(z, 0.0, 0.0, z0=0.0),
        z=z_rsun,
    )

    assert abs(delay_days) * _SECONDS_PER_DAY == pytest.approx(
        _TAU_A_SECONDS, abs=1e-3
    )


def test_face_on_circular_orbit_uses_linear_fallback_and_gives_zero_delay():
    """
    Given a face-on (i=0, sin_i=0) circular (e=0) orbit,
    When line_of_sight_kinematics/retarded_time are evaluated at several
    times spanning a full period,
    Then z, vz, and az are all exactly 0 (no line-of-sight motion at all,
    so this exercises the |az| < 1e-10 linear-fallback branch of
    solve_delay identically, not just approximately), and the delay is
    exactly 0 -- confirming the pt.switch selects the fallback branch
    without dividing by az=0 in the quadratic branch (which would NaN).
    """
    period = 3.0
    n_val = 2.0 * np.pi / period
    a_rel = 10.0

    t_s, tp_s, n_s, ecc_s, sinw_s, cosw_s, sini_s, a_s = (
        pt.dscalar("t"),
        pt.dscalar("tp"),
        pt.dscalar("n"),
        pt.dscalar("ecc"),
        pt.dscalar("sinw"),
        pt.dscalar("cosw"),
        pt.dscalar("sin_i"),
        pt.dscalar("a_rel"),
    )
    z_expr, vz_expr, az_expr = ltt.line_of_sight_kinematics(
        t_s, tp_s, n_s, ecc_s, sinw_s, cosw_s, sini_s, a_s
    )
    t_corr_expr, delay_expr = ltt.retarded_time(
        t_s, tp_s, n_s, ecc_s, sinw_s, cosw_s, sini_s, a_s
    )

    for t_val in [0.0, 0.3 * period, 0.5 * period, 0.9 * period]:
        givens = {
            t_s: t_val,
            tp_s: 0.0,
            n_s: n_val,
            ecc_s: 0.0,
            sinw_s: 0.0,
            cosw_s: 1.0,
            sini_s: 0.0,  # face-on
            a_s: a_rel,
        }
        z_val = float(z_expr.eval(givens))
        vz_val = float(vz_expr.eval(givens))
        az_val = float(az_expr.eval(givens))
        delay_val = float(delay_expr.eval(givens))

        assert z_val == 0.0
        assert vz_val == 0.0
        assert az_val == 0.0
        assert np.isfinite(delay_val)
        assert delay_val == 0.0


def test_circular_edgeon_secondary_eclipse_offset_is_2a_over_c():
    """
    Given a circular (e=0), edge-on (i=90 deg, sin_i=1) orbit with
    omega=0 (so primary conjunction -- sin(f+w)=+1, EXOZIPPy's "+ = toward
    observer" convention -- falls at t = tp + period/4, secondary at
    t = tp + 3*period/4),
    When retarded_time's delay is evaluated at both conjunction times,
    Then the two delays differ by 2*a/c in magnitude -- the classic result
    (Eastman, Gaudi & Agol 2013, Sec. V.3; also EXOFASTv2's target2bjd.pro
    docstring) that a circular edge-on orbit's primary and secondary events
    are offset by 2a/c relative to the target-barycenter-frame half-period
    separation, since the two conjunctions sit at the near and far
    extrema of the line-of-sight coordinate.

    a_rel = 1 AU and period = 365.25 days (Earth-like) are chosen so the
    orbital velocity is small compared to c: v/c ~ 1e-4, so the quadratic
    formula's departure from the leading-order a/c approximation is
    O((v/c)^2) ~ 1e-8 relative -- far below the rel=1e-6 tolerance used
    here, so "2a/c" is not just approximately but very precisely expected.
    """
    a_rel = 1.0 / RSUN_TO_AU  # 1 AU, in R_sun
    period = 365.25
    n_val = 2.0 * np.pi / period
    tp_val = 0.0

    t_s, tp_s, n_s, ecc_s, sinw_s, cosw_s, sini_s, a_s = (
        pt.dscalar("t"),
        pt.dscalar("tp"),
        pt.dscalar("n"),
        pt.dscalar("ecc"),
        pt.dscalar("sinw"),
        pt.dscalar("cosw"),
        pt.dscalar("sin_i"),
        pt.dscalar("a_rel"),
    )
    _, delay_expr = ltt.retarded_time(
        t_s, tp_s, n_s, ecc_s, sinw_s, cosw_s, sini_s, a_s
    )

    base_givens = {
        tp_s: tp_val,
        n_s: n_val,
        ecc_s: 0.0,
        sinw_s: 0.0,  # omega = 0
        cosw_s: 1.0,
        sini_s: 1.0,  # edge-on
        a_s: a_rel,
    }

    t_primary = tp_val + period / 4.0
    t_secondary = tp_val + 3.0 * period / 4.0

    delay_primary = float(delay_expr.eval({**base_givens, t_s: t_primary}))
    delay_secondary = float(delay_expr.eval({**base_givens, t_s: t_secondary}))

    a_over_c_days = a_rel / C_LIGHT_RSUN_PER_DAY
    observed_diff = delay_secondary - delay_primary

    assert abs(observed_diff) == pytest.approx(2.0 * a_over_c_days, rel=1e-6)


def test_retarded_time_gradient_is_finite_at_az_zero_via_numpyro():
    """
    Given an edge-on (sin_i=1), circular (e=0), omega=0 orbit evaluated at
    t = tp exactly -- where sin(f+w) = 0, so z = vz = az = 0 identically,
    the exact degenerate point the linear-fallback branch exists for --
    When retarded_time's delay is embedded in a PyMC model's likelihood and
    (a) the logp's gradient is evaluated directly at that point, and (b) a
    short NUTS chain is sampled with nuts_sampler="numpyro" in a
    neighborhood around it,
    Then both the direct gradient and the full sampled trace are finite
    throughout -- locking in this property (needed for NUTS, the whole
    reason for using exoplanet's closed-form method over EXOFASTv2's
    iterative one) as a permanent regression check, independent of why it
    holds.

    This is NOT a reproduced-bug regression test, and should not be read as
    one: solve_delay's dead quadratic branch does compute a genuine c/az
    singularity at az == 0 (confirmed directly: pt.grad of the isolated
    expression c/az is -inf there), which is exactly the shape of the
    JAX/PyTensor "where-trap" the design doc's house-rule warns about --
    but empirically, pt.grad of solve_delay's full pt.switch expression
    stayed finite (0.0) at az == 0 even WITHOUT the az_safe guard, checked
    directly via pt.grad in both the default and mode="JAX" backends, and
    in a vectorized/summed multi-element case mixing az == 0 with az != 0.
    So for this specific formula, pt.switch's gradient did not propagate
    the unselected branch's singularity, and this test does not go red on
    the unguarded solve_delay. The az_safe guard (see solve_delay) is kept
    anyway as defensive practice -- it removes a real forward-pass
    RuntimeWarning and does not rely on this gradient behavior being a
    guaranteed property of pt.switch across versions/backends -- but its
    necessity for gradient correctness is not demonstrated by this test.
    """
    a_rel = 1.0 / RSUN_TO_AU  # 1 AU, in R_sun
    period = 365.25
    n_val = 2.0 * np.pi / period
    tp_true = 10.0

    # t_obs[0] == tp_true exactly: at omega=0, e=0, this is precisely the
    # az=0 crossing (sin(f+w) = sin(n*(t-tp)) = 0 there). The rest spread
    # across the orbit so the likelihood is well-posed, not degenerate.
    t_obs = (
        np.asarray(
            [0.0, period / 8.0, period / 3.0, period * 0.6, period * 0.9],
            dtype=np.float64,
        )
        + tp_true
    )

    def _delay_expr(tp):
        return ltt.retarded_time(
            pt.as_tensor_variable(t_obs),
            tp,
            pt.constant(np.float64(n_val)),
            pt.constant(np.float64(0.0)),  # ecc
            pt.constant(np.float64(0.0)),  # sinw
            pt.constant(np.float64(1.0)),  # cosw
            pt.constant(np.float64(1.0)),  # sin_i, edge-on
            pt.constant(np.float64(a_rel)),
        )[1]

    # Self-consistent synthetic "observed" delays at the true tp (a plain
    # numeric eval, tp as a constant, not yet inside the PyMC model).
    observed_delay = _delay_expr(pt.constant(np.float64(tp_true))).eval()
    assert np.all(np.isfinite(observed_delay))
    assert observed_delay[0] == pytest.approx(0.0, abs=1e-12)  # the az=0 point

    with pm.Model() as model:
        tp_rv = pm.Normal("tp", mu=tp_true, sigma=0.1, initval=tp_true)
        delay_model = _delay_expr(tp_rv)
        pm.Normal("obs", mu=delay_model, sigma=1e-4, observed=observed_delay)

        # (a) Direct gradient check AT the exact az=0 point: initval=tp_true
        # makes model.initial_point() land exactly there, deterministically
        # -- not left to chance that NUTS wanders near it.
        point = model.initial_point()
        logp_val = model.compile_logp()(point)
        grad_val = model.compile_dlogp()(point)

        assert np.isfinite(logp_val), f"non-finite logp at az=0: {logp_val}"
        assert np.all(np.isfinite(grad_val)), (
            f"non-finite gradient at az=0: {grad_val}"
        )

        # (b) Actual short NUTS run via numpyro -- smoke test that sampling
        # in a neighborhood of the az=0 point completes with finite draws.
        idata = pm.sample(
            draws=25,
            tune=25,
            chains=1,
            cores=1,
            nuts_sampler="numpyro",
            progressbar=False,
            random_seed=0,
        )

    tp_samples = idata.posterior["tp"].values
    assert np.all(np.isfinite(tp_samples))

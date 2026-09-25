"""Periodic parameters are recentered about their mode before reporting.

A derived angle comes out of arctan2 on the (-180, 180] cut and a derived
epoch (tp, ts) lands wherever the Tc -> Tp inversion puts it within the
period.  A posterior that happens to straddle that cut used to be summarized
in two pieces: the median fell in the empty middle and the interval spanned
the whole period -- a plausible-looking number for a poorly constrained
omega, so nothing flagged it.  The physics is invariant to the cut, so it is
purely a reporting artifact.

JDE, 2026-09-25: "when deterministic periodic parameters (tp, omega, ts, etc)
are derived, we need to make sure artificial boundaries don't split them and
skew the reported 68% CI.  In exofastv2, we center any periodic parameter at
its mode before reporting."  EXOFASTv2's rule (exofast_recenter.pro,
summarizepar.pro) is what these tests pin: histogram mode, then whole-period
shifts into (mode - P/2, mode + P/2].
"""

import math

import numpy as np
import pytest
import yaml

from exozippy.components.parameter import (
    Parameter,
    recenter_periodic,
    recenter_periodic_1d,
)

N = 20_000
TC = 2450000.0


def _wrap(x, period, center=0.0):
    """Fold ``x`` into ``[center - period/2, center + period/2)`` -- the cut."""
    return center + np.mod(x - center + period / 2.0, period) - period / 2.0


@pytest.fixture
def split_angle():
    """N(179, 3) in degrees, and the same draws folded onto the (-180, 180] cut."""
    rng = np.random.default_rng(1)
    unwrapped = rng.normal(179.0, 3.0, N)
    return unwrapped, _wrap(unwrapped, 360.0)


def _angle_param(**kw):
    defaults = dict(
        label="orbit.b.omega",
        unit="deg",
        internal_unit="rad",
        periodic={"value": 360.0, "unit": "deg"},
    )
    defaults.update(kw)
    return Parameter(**defaults)


# ---------------------------------------------------------------------------
# The recentering itself
# ---------------------------------------------------------------------------


def test_a_split_angle_is_rejoined_at_its_mode(split_angle):
    """
    Given draws of N(179, 3) deg folded onto the (-180, 180] cut, so ~37% of
    them sit near -180,
    When they are recentered with a 360 deg period,
    Then every draw equals its unfolded value: one contiguous distribution.
    """
    unwrapped, wrapped = split_angle
    assert np.ptp(wrapped) > 300.0  # the fold really did split them

    out = recenter_periodic_1d(wrapped, 360.0)

    assert np.allclose(out, unwrapped, atol=1e-9)


def test_the_summary_of_split_draws_is_the_summary_of_the_unsplit_ones(
    split_angle,
):
    """
    Given the same folded draws,
    When _summarize_array is asked for them with the period,
    Then it reports the unfolded median and interval -- and without the period
    it reports the artifact: a median near zero and a +/- 180 interval.
    """
    unwrapped, wrapped = split_angle

    fixed = Parameter._summarize_array(wrapped, period=360.0)
    truth = Parameter._summarize_array(unwrapped)
    broken = Parameter._summarize_array(wrapped)

    assert math.isclose(fixed.median, truth.median, abs_tol=1e-9)
    assert math.isclose(fixed.err_minus, truth.err_minus, abs_tol=1e-9)
    assert math.isclose(fixed.err_plus, truth.err_plus, abs_tol=1e-9)
    # The artifact this exists to remove: without the period the median lands
    # in the empty middle and the interval spans the range.
    assert max(broken.err_minus, broken.err_plus) > 100.0


def test_draws_already_within_one_period_are_returned_bit_identical():
    """
    Given draws that do not come near the cut,
    When they are recentered,
    Then not one of them moves by a single bit: the shift is by whole periods
    only, and a draw needing none is untouched.
    """
    x = np.random.default_rng(2).normal(10.0, 3.0, N)

    out = recenter_periodic_1d(x, 360.0)

    assert np.array_equal(out, x)


def test_draws_scattered_over_hundreds_of_periods_land_in_one():
    """
    Given a runaway: N(0, 3) plus a random whole number of periods in
    [-500, 500] per draw, so a single 100-bin histogram has bins ~3.6
    periods wide,
    When recentered,
    Then every draw sits within one period of every other and each is the
    base distribution shifted by ONE common whole number of periods (the
    second pass makes the mode exact once the first has folded the span
    down to one period; which epoch that is, is the mode's business).
    """
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 3.0, N)
    scattered = base + 360.0 * rng.integers(-500, 501, N)

    out = recenter_periodic_1d(scattered, 360.0)

    assert np.ptp(out) < 360.0
    shifts = (out - base) / 360.0
    assert np.allclose(shifts, np.round(shifts), atol=1e-9)
    assert len(set(np.round(shifts).astype(int))) == 1


def test_non_finite_draws_are_left_in_place(split_angle):
    """
    Given folded draws with a NaN among them,
    When recentered,
    Then the NaN stays where it was and the rest are rejoined.
    """
    unwrapped, wrapped = split_angle
    wrapped = wrapped.copy()
    wrapped[7] = np.nan

    out = recenter_periodic_1d(wrapped, 360.0)

    assert np.isnan(out[7])
    keep = np.ones(N, bool)
    keep[7] = False
    assert np.allclose(out[keep], unwrapped[keep], atol=1e-9)


def test_a_vector_parameter_is_recentered_element_by_element():
    """
    Given a (2, n) array whose first row straddles the cut and whose second
    does not, with one period per row,
    When recentered,
    Then the first row is rejoined and the second is bit-identical.
    """
    rng = np.random.default_rng(4)
    row0 = rng.normal(179.0, 3.0, N)
    row1 = rng.normal(45.0, 3.0, N)
    arr = np.vstack([_wrap(row0, 360.0), row1])

    out = recenter_periodic(arr, [360.0, 360.0])

    assert np.allclose(out[0], row0, atol=1e-9)
    assert np.array_equal(out[1], row1)


def test_a_period_below_the_values_resolution_is_refused_not_applied(caplog):
    """
    Given epochs near 2.45e6 and a period of 1e-12 d (below one ulp there),
    When recentered,
    Then the draws come back unchanged with a warning, rather than mangled by a
    modulus the floating point cannot represent (EXOFASTv2's guard).
    """
    x = TC + np.random.default_rng(5).normal(0.0, 0.1, 1000)

    with caplog.at_level("WARNING"):
        out = recenter_periodic_1d(x, 1e-12)

    assert out is x
    assert "resolution" in caplog.text


# ---------------------------------------------------------------------------
# The declaration on the Parameter
# ---------------------------------------------------------------------------


def test_a_constant_period_follows_the_users_unit_relabel():
    """
    Given omega declared periodic at 360 deg but relabeled by the user in
    radians,
    When its period is resolved,
    Then it is 2 pi, not 360: the constant carries its own unit and goes
    through the parameter's from_internal like every other value.
    """
    p = _angle_param(unit="rad")

    (period,) = p.periodic_period()

    assert math.isclose(period, 2.0 * math.pi, rel_tol=1e-12)


def test_an_epoch_takes_its_period_from_the_sibling_parameters_median():
    """
    Given orbit.b.period with draws around 3.0 d and orbit.b.tp declared
    periodic on `period`, with tp draws of N(tc + 1.5, 0.05) folded into
    [tc - 1.5, tc + 1.5) -- i.e. split down the middle,
    When tp is recentered through the lookup,
    Then its stored posterior is one contiguous distribution of width ~0.05 d,
    a whole number of periods from the truth, and the period it recorded is
    the sibling's median.
    """
    rng = np.random.default_rng(6)
    period = Parameter(label="orbit.b.period", unit="d", internal_unit="d")
    period.posterior = 3.0 + rng.normal(0.0, 1e-4, N)
    tp = Parameter(
        label="orbit.b.tp",
        unit="d",
        internal_unit="d",
        periodic={"param": "period"},
    )
    truth = TC + 1.5 + rng.normal(0.0, 0.05, N)
    wrapped = _wrap(truth, 3.0, center=TC)
    tp.posterior = wrapped.copy()
    assert np.ptp(wrapped) > 2.5
    lookup = {period.label: period, tp.label: tp}

    tp.recenter_posterior(lookup)

    post = np.asarray(tp.posterior)
    assert np.ptp(post) < 0.6
    per_used = float(tp._recenter_period[0])
    assert math.isclose(per_used, float(np.median(period.posterior)))
    assert per_used != 3.0  # the median, not the round number the fold used
    shifts = (post - wrapped) / per_used
    assert np.allclose(shifts, np.round(shifts), atol=1e-9)
    # Two pieces: one stayed, the other moved by exactly one period.
    assert set(np.round(shifts).astype(int)) in ({0, 1}, {0, -1})
    assert np.allclose(post, truth, atol=1e-5)
    s = tp.compute_summary()
    assert math.isclose(s.err_plus, 0.05, rel_tol=0.1)
    assert math.isclose(s.err_minus, 0.05, rel_tol=0.1)


def test_a_sibling_without_draws_lends_its_start_value():
    """
    Given the sibling period has no posterior (fixed),
    When tp's period is resolved,
    Then it is the sibling's start value in tp's user unit.
    """
    period = Parameter(
        label="orbit.b.period", unit="d", internal_unit="d", initval=2.5
    )
    tp = Parameter(
        label="orbit.b.tp",
        unit="d",
        internal_unit="d",
        periodic={"param": "period"},
    )

    (per,) = tp.periodic_period({period.label: period, tp.label: tp})

    assert per == 2.5


def test_a_missing_sibling_warns_and_leaves_the_draws_alone(caplog):
    """
    Given tp declared periodic on a sibling the lookup does not hold,
    When it is recentered,
    Then nothing moves and a warning names the sibling: reporting the old way
    is wrong only when the draws straddle the cut, while a guessed period
    would move every reported epoch.
    """
    tp = Parameter(
        label="orbit.b.tp",
        unit="d",
        internal_unit="d",
        periodic={"param": "period"},
    )
    draws = TC + np.random.default_rng(8).normal(0.0, 0.1, 100)
    tp.posterior = draws.copy()

    with caplog.at_level("WARNING"):
        tp.recenter_posterior({tp.label: tp})

    assert np.array_equal(np.asarray(tp.posterior), draws)
    assert tp._recenter_period is None
    assert "orbit.b.period" in caplog.text


def test_each_mode_is_recentered_on_its_own(split_angle):
    """
    Given a bimodal omega whose SECOND mode straddles the cut while the first
    (larger) one does not -- so the global recentering, about the first
    mode, leaves the second split,
    When the per-mode summaries are computed,
    Then the second mode is summarized as the one distribution it is.
    """
    unwrapped, wrapped = split_angle
    rng = np.random.default_rng(9)
    big = rng.normal(0.0, 3.0, 2 * N)
    p = _angle_param()
    p.posterior = np.concatenate([big, wrapped])
    labels = np.concatenate([np.zeros(2 * N, int), np.ones(N, int)])
    lookup = {p.label: p}

    p.recenter_posterior(lookup)
    # Global recentering is about the big mode near 0, so the split second
    # mode is still split in the stored draws ...
    assert np.ptp(np.asarray(p.posterior)[2 * N :]) > 300.0
    m0, m1 = p.compute_mode_summaries(labels, 2)

    # ... and the per-mode summary still rejoins it.
    truth = Parameter._summarize_array(unwrapped)
    assert math.isclose(m1.median, truth.median, abs_tol=1e-9)
    assert math.isclose(m1.err_plus, truth.err_plus, abs_tol=1e-9)
    assert abs(m0.median) < 0.5


def test_the_setter_drops_the_recorded_period_with_the_other_caches():
    """
    Given a recentered Parameter,
    When new draws are assigned,
    Then the recorded period goes with the summaries it belonged to, so a
    second report cannot recenter new draws by a stale sibling median.
    """
    p = _angle_param()
    p.posterior = np.random.default_rng(10).normal(170.0, 5.0, 1000)
    p.recenter_posterior({p.label: p})
    assert p._recenter_period is not None

    p.posterior = np.zeros(10)

    assert p._recenter_period is None


@pytest.mark.parametrize(
    "spec",
    [
        "360",
        360.0,
        {"value": 360.0},
        {"period": 360.0},
        {"value": -1.0, "unit": "deg"},
        {"value": 360.0, "unit": "deg", "param": "period"},
        {"param": ""},
    ],
)
def test_a_malformed_declaration_fails_at_construction(spec):
    """
    Given a periodic: declaration that is not one of the two documented dicts,
    When the Parameter is built,
    Then it raises then and there, not at wrap-up.
    """
    with pytest.raises(ValueError, match="periodic"):
        _angle_param(periodic=spec)


def test_a_period_in_an_incompatible_unit_fails_at_construction():
    """
    Given an angle whose period is declared in days,
    When the Parameter is built,
    Then it raises naming the internal unit it cannot reach.
    """
    with pytest.raises(ValueError, match="not convertible"):
        _angle_param(periodic={"value": 3.0, "unit": "d"})


# ---------------------------------------------------------------------------
# The shipped declarations
# ---------------------------------------------------------------------------


def _defaults(component, block):
    from pathlib import Path

    import exozippy.components as comps

    path = Path(comps.__file__).parent / component / "defaults.yaml"
    return yaml.safe_load(open(path))[block]


def test_the_orbit_declares_its_angles_and_epochs_periodic():
    """
    Given the orbit component's defaults,
    When the periodic declarations are read,
    Then the three angles carry a 360 deg period, the four epochs take the
    orbital period from `period`, and nothing else on the orbit is periodic
    (inc is a geometric angle on [0, 180], not a cyclic one).
    """
    params = _defaults("orbit", "orbit")

    declared = {k: v["periodic"] for k, v in params.items() if "periodic" in v}

    assert declared == {
        "omega": {"value": 360.0, "unit": "deg"},
        "bigomega": {"value": 360.0, "unit": "deg"},
        "lam": {"value": 360.0, "unit": "deg"},
        "tp": {"param": "period"},
        "tp_target": {"param": "period"},
        "ts": {"param": "period"},
        "ts_target": {"param": "period"},
    }
    assert "period" in params


def test_the_lens_trajectory_angle_is_periodic():
    """
    Given the mulensing defaults,
    When alpha's declaration is read,
    Then it carries a 360 deg period (it is arctan2 of xalpha, yalpha).
    """
    params = _defaults("mulensing", "lens")

    assert params["alpha"]["periodic"] == {"value": 360.0, "unit": "deg"}


# ---------------------------------------------------------------------------
# End to end: distribute_posterior recenters what the trace hands it
# ---------------------------------------------------------------------------

_LC = dict(baseline=1.0, depth=0.01, P=3.2, tc=2459100.0, err=4.0e-4)
N_CHAIN, N_DRAW = 2, 400


def _write_lc(path, n=180):
    t = np.linspace(_LC["tc"] - 0.2, _LC["tc"] + 0.2, n)
    in_transit = np.abs(t - _LC["tc"]) < 0.04
    flux = _LC["baseline"] - _LC["depth"] * in_transit
    np.savetxt(path, np.column_stack([t, flux, np.full_like(t, _LC["err"])]))


@pytest.fixture(scope="module")
def built_system(tmp_path_factory):
    """A real, fully built one-planet System (the fixture test_second_report_staleness uses)."""
    from exozippy.system import System

    lc_path = tmp_path_factory.mktemp("periodic") / "per.TESS.dat"
    _write_lc(lc_path)
    config = {
        "run": {"name": "periodic"},
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "TESS", "file": str(lc_path), "band": "TESS"}],
    }
    user_params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.1},
        "orbit.b.period": {"initval": _LC["P"]},
        "orbit.b.tc": {"initval": _LC["tc"]},
    }
    system = System(config, user_params=user_params)
    system.prepare()
    system.build_model()
    return system


def _trace_with_split_periodics(system, omega_truth, tp_truth):
    """Every label present (so nothing is re-derived); omega and tp folded onto their cuts."""
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(11)
    size = (N_CHAIN, N_DRAW)
    data_vars = {}
    for label, par in system.get_parameter_lookup().items():
        n = par._n_elements()
        if label == "orbit.omega":
            arr = _wrap(omega_truth, 360.0).reshape(size)
        elif label == "orbit.tp":
            arr = _wrap(tp_truth, _LC["P"], center=_LC["tc"]).reshape(size)
        elif label == "orbit.period":
            arr = _LC["P"] + rng.normal(0.0, 1e-5, size)
        elif n == 1:
            arr = 1.0 + rng.normal(0.0, 0.01, size)
        else:
            arr = 1.0 + rng.normal(0.0, 0.01, size + (n,))
        dims = ["chain", "draw"] + ([f"{label}_dim"] if arr.ndim == 3 else [])
        data_vars[label] = xr.DataArray(arr, dims=dims)
    return az.from_dict({"posterior": xr.Dataset(data_vars)})


def test_distribute_posterior_rejoins_a_split_omega_and_tp(built_system):
    """
    Given a trace whose omega is N(179, 3) deg folded onto (-180, 180] and
    whose tp is N(tc + P/2, 0.05 d) folded into [tc - P/2, tc + P/2),
    When the posterior is distributed to the System's Parameters,
    Then both are reported as one distribution each: omega's summary is the
    unfolded one, and tp's interval is ~0.05 d with its median a whole number
    of periods (taken from orbit.period's median) from the truth.  The model
    labels are component-level vectors (orbit.tp, shape (n_orbits,)), not the
    config's per-instance spelling, which is why the sibling is found by
    the component prefix.
    """
    rng = np.random.default_rng(12)
    omega_truth = rng.normal(179.0, 3.0, N_CHAIN * N_DRAW)
    tp_truth = (
        _LC["tc"] + _LC["P"] / 2 + rng.normal(0.0, 0.05, N_CHAIN * N_DRAW)
    )
    system = built_system
    lookup = system.get_parameter_lookup()

    system.distribute_posterior(
        _trace_with_split_periodics(system, omega_truth, tp_truth)
    )

    omega = lookup["orbit.omega"].ensure_summary()
    truth = Parameter._summarize_array(omega_truth)
    assert math.isclose(omega.median, truth.median, abs_tol=1e-9)
    assert math.isclose(omega.err_minus, truth.err_minus, abs_tol=1e-9)
    assert math.isclose(omega.err_plus, truth.err_plus, abs_tol=1e-9)
    # The stored draws themselves are contiguous -- what the corner plot sees.
    assert np.ptp(np.asarray(lookup["orbit.omega"].posterior)) < 60.0

    tp = lookup["orbit.tp"].ensure_summary()
    assert math.isclose(tp.err_plus, 0.05, rel_tol=0.15)
    assert math.isclose(tp.err_minus, 0.05, rel_tol=0.15)
    n_per = (tp.median - np.median(tp_truth)) / _LC["P"]
    assert math.isclose(n_per, round(n_per), abs_tol=0.01)

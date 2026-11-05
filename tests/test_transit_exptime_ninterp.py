"""
Tests for exptime_s/ninterp exposure-time smearing on the transit component
(the machinery is the Instrument base's -- instrument.md, "Exposure
smearing" -- with transit its first caller):
  - exptime_s (SECONDS) / ninterp are fixed per-instrument properties read
    straight off the config at construction, not fitted parameters
  - the retired minutes key exptime: raises naming its replacement, and a
    component that does not smear refuses the keys
  - invalid smearing configs (ninterp < 1, exptime_s <= 0, or ninterp > 1
    without an exptime_s) warn and fall back to no smearing
  - the sub-exposure grid is EXOFASTv2's midpoint Riemann rule: ninterp
    cell centers across [-exptime/2, +exptime/2], never the exposure edges
  - ninterp==1 must reproduce the pre-existing instantaneous model exactly,
    regardless of exptime
  - a long exptime with ninterp>1 measurably smooths ingress/egress
  - mixed ninterp/exptime across instruments in one component is handled
    per-instrument: observations are partitioned into one group per
    distinct ninterp (each group's grid exactly its own ninterp wide),
    not padded to a single uniform ninterp

All light curves here are synthetic (not the kelt4 example).
"""

import logging

import numpy as np
import pytensor
import pytest

from exozippy.system import System


def _write_lc(path, t, flux=None, err=None):
    """Write a whitespace-separated time/flux/err light curve file."""
    n = len(t)
    if flux is None:
        flux = np.ones(n)
    if err is None:
        err = np.full(n, 1e-3)
    np.savetxt(path, np.column_stack([t, flux, err]))
    return str(path)


def _config(lc_files, transit_extra=None):
    """One star/planet/orbit, one V band, N transit instruments on lc_files."""
    transit_extra = transit_extra or [{}] * len(lc_files)
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "V", "filter": "V", "ld_law": "quadratic"}],
        "transit": [
            {"name": f"inst{i}", "file": f, "band": "V", **extra}
            for i, (f, extra) in enumerate(zip(lc_files, transit_extra))
        ],
    }


def _params():
    """A short-period, near-central, hot-Jupiter-like synthetic system.

    Short period + near-zero impact parameter gives a fast, sharp-edged
    transit (ingress/egress ~11.5 min, total duration ~100 min at these
    values) so exposure smearing over tens of minutes has a clearly
    measurable effect.
    """
    return {
        "star.0.radius": {"initval": 1.0, "sigma": 0.05},
        "star.0.mass": {"initval": 1.0, "sigma": 0.05},
        "star.0.teff": {"initval": 5800, "sigma": 100},
        "star.0.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": 0.5},
        "orbit.0.tc": {"initval": 2459634.3},
        "orbit.0.cosi": {"initval": 0.001},
        "planet.0.radius": {"initval": 1.2},
    }


TC = 2459634.3
T14 = 0.0696  # days; total transit duration at the _params() geometry


def _plot_param_values(system):
    """Reproduce transit.py's own plot_params -> compiled-function argument
    conversion (see Transit.plot_data) at the start point, so the plotted
    model can be evaluated without a sampler point."""
    return [
        float(np.squeeze(np.asarray(p.initval)))
        if getattr(p.value, "ndim", 0) == 0
        else np.atleast_1d(p.initval)
        for p in system.plot_params
    ]


def _model_flux_at_initial_point(system, model):
    """Evaluate transit's per-observation model-flux tensor at the model's
    initial point (same givens-substitution pattern already used in
    test_transit_band.py for transit.dilution). This is a plain attribute
    (Transit._model_flux_node), not a Deterministic -- it's (N_obs,), and
    registering it as a Deterministic would add N_obs floats per draw per
    chain to every trace, so it's deliberately kept out of the model graph's
    named/traced variables and only compiled here for testing."""
    ip = model.initial_point()
    givens = [
        (rv, np.asarray(ip[rv.name])) for rv in model.free_RVs if rv.name in ip
    ]
    fn = pytensor.function(
        [],
        system.transit._model_flux_node,
        givens=givens,
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )
    return fn()


def test_ninterp_one_matches_instantaneous_model(tmp_path_factory):
    """
    Given a transit instrument with ninterp=1 and a nonzero exptime, and
    an otherwise identical instrument with no smearing keys at all (the
    instantaneous model),
    When both models are built,
    Then their transit.model_flux agree to floating-point precision --
    ninterp=1 must ignore exptime and short-circuit to the original
    single-point evaluation.  (The plotted model used to serve as the
    instantaneous reference here; it now smears exactly as the likelihood
    does, so the reference is a sibling system without the keys.)
    """
    d = tmp_path_factory.mktemp("ninterp_one")
    t = np.linspace(TC - 0.55 * T14, TC + 0.55 * T14, 401)
    lc = _write_lc(d / "lc.dat", t)

    config = _config([lc], [{"exptime_s": 1800.0, "ninterp": 1}])
    system = System(config, user_params=_params())
    system.prepare()
    model = system.build_model()
    model_flux = _model_flux_at_initial_point(system, model)

    config_instant = _config([lc])
    system_instant = System(config_instant, user_params=_params())
    system_instant.prepare()
    model_instant = system_instant.build_model()
    reference = _model_flux_at_initial_point(system_instant, model_instant)

    assert system.transit.exptime_s == [1800.0]
    np.testing.assert_allclose(model_flux, reference, atol=1e-12)


def test_invalid_smearing_config_warns_and_falls_back(
    tmp_path_factory, caplog
):
    """
    Given four transit instruments -- three with invalid smearing configs
    (ninterp=0; a negative exptime_s; ninterp>1 with no exptime_s at all)
    and one valid control (exptime_s=1800, ninterp=3),
    When the system is constructed and prepared,
    Then each invalid instrument gets one warning naming it and is reset to
    the inert defaults (ninterp=1, exptime_s=60: no smearing), while the valid
    instrument is untouched and gets no warning -- a bad config degrades to
    the instantaneous model loudly instead of crashing in the grid builder
    or silently smearing over a made-up exposure time.
    """
    d = tmp_path_factory.mktemp("invalid_smearing")
    t = np.linspace(TC - 0.1, TC + 0.1, 5)
    lcs = [_write_lc(d / f"lc{i}.dat", t) for i in range(4)]

    config = _config(
        lcs,
        [
            {"ninterp": 0},
            {"exptime_s": -300.0, "ninterp": 3},
            {"ninterp": 3},
            {"exptime_s": 1800.0, "ninterp": 3},
        ],
    )
    with caplog.at_level(logging.WARNING, logger="exozippy"):
        system = System(config, user_params=_params())
        system.prepare()

    tr = system.transit
    assert tr.ninterp == [1, 1, 1, 3]
    assert tr.exptime_s == [60.0, 60.0, 60.0, 1800.0]
    # Only the accepted exptime_s is the file's own (the summary figure
    # labels a TESS row with its cadence by this): the rejected -300 is not.
    assert tr.exptime_stated == [False, False, False, True]

    smearing_warnings = [
        r.message for r in caplog.records if "exposure smearing" in r.message
    ]
    assert len(smearing_warnings) == 3
    for name in ("inst0", "inst1", "inst2"):
        assert any(name in msg for msg in smearing_warnings)
    assert not any("inst3" in msg for msg in smearing_warnings)

    # The fallback really disables smearing: only the valid instrument's
    # rows land in a width-3 group.
    groups_by_width = {
        grid.shape[1]: rows for rows, grid, weights in tr._exposure_groups
    }
    assert set(groups_by_width.keys()) == {1, 3}
    inst3_rows = np.nonzero(tr.inst_map == 3)[0]
    assert sorted(groups_by_width[3]) == list(inst3_rows)


@pytest.mark.slow
def test_long_exptime_with_ninterp_smooths_ingress_egress(tmp_path_factory):
    """
    Given two otherwise-identical systems -- one with ninterp=1, one with a
    long exptime and ninterp=21 -- evaluated on the same fine time grid
    spanning one transit,
    When comparing the discrete slope (d flux / d t) of transit.model_flux,
    Then the oversampled model's steepest slope (at ingress/egress) is
    substantially smaller than the instantaneous model's: exposure smearing
    measurably softens the sharp ingress/egress edges.
    """
    d = tmp_path_factory.mktemp("ninterp_smooth")
    t = np.linspace(TC - 0.55 * T14, TC + 0.55 * T14, 801)

    lc_instant = _write_lc(d / "lc_instant.dat", t)
    config_instant = _config(
        [lc_instant], [{"exptime_s": 1800.0, "ninterp": 1}]
    )
    system_instant = System(config_instant, user_params=_params())
    system_instant.prepare()
    model_instant = system_instant.build_model()
    flux_instant = _model_flux_at_initial_point(system_instant, model_instant)

    lc_smeared = _write_lc(d / "lc_smeared.dat", t)
    config_smeared = _config(
        [lc_smeared], [{"exptime_s": 3600.0, "ninterp": 21}]
    )
    system_smeared = System(config_smeared, user_params=_params())
    system_smeared.prepare()
    model_smeared = system_smeared.build_model()
    flux_smeared = _model_flux_at_initial_point(system_smeared, model_smeared)

    dt = np.diff(t)
    max_slope_instant = np.max(np.abs(np.diff(flux_instant) / dt))
    max_slope_smeared = np.max(np.abs(np.diff(flux_smeared) / dt))

    assert max_slope_smeared < 0.5 * max_slope_instant


def test_mixed_ninterp_across_instruments_is_grouped_per_instrument(
    tmp_path_factory,
):
    """
    Given two transit instruments in one component with different
    exptime/ninterp (inst0: ninterp=1; inst1: exptime=60min, ninterp=3),
    When load_data builds the oversampling groups,
    Then observations are partitioned into one group per distinct ninterp
    value: inst0's rows form a width-1 group (grid is just each row's own
    timestamp, weight 1) and inst1's rows form a width-3 group (3 midpoint
    sub-times per row -- EXOFASTv2's (j + 0.5)/n - 0.5 cell centers, so
    +/- exptime/3 for n=3, never the exposure edges -- each weighted 1/3)
    -- each instrument's own ninterp/exptime is honored, and neither group
    is padded out to the other's width (no wasted sub-samples).
    """
    d = tmp_path_factory.mktemp("mixed_ninterp")
    t0 = np.linspace(TC - 0.1, TC + 0.1, 5)
    t1 = np.linspace(TC - 0.1, TC + 0.1, 5)
    lc0 = _write_lc(d / "lc0.dat", t0)
    lc1 = _write_lc(d / "lc1.dat", t1)

    config = _config(
        [lc0, lc1],
        [{"ninterp": 1}, {"exptime_s": 3600.0, "ninterp": 3}],
    )
    system = System(config, user_params=_params())
    system.prepare()

    tr = system.transit
    groups_by_width = {
        grid.shape[1]: (rows, grid, weights)
        for rows, grid, weights in tr._exposure_groups
    }
    assert set(groups_by_width.keys()) == {1, 3}

    inst0_rows = np.nonzero(tr.inst_map == 0)[0]
    rows0, grid0, weights0 = groups_by_width[1]
    assert sorted(rows0) == list(inst0_rows)
    order0 = np.argsort(rows0)
    np.testing.assert_allclose(weights0, [1.0])
    np.testing.assert_allclose(grid0[order0], t0[:, None])

    inst1_rows = np.nonzero(tr.inst_map == 1)[0]
    rows1, grid1, weights1 = groups_by_width[3]
    assert sorted(rows1) == list(inst1_rows)
    order1 = np.argsort(rows1)
    exptime_days = 3600.0 / 86400.0
    # Midpoint Riemann cell centers, (j + 0.5)/3 - 0.5: NOT the exposure
    # edges +/- exptime/2 (EXOFASTv2 parity, exofast_chi2v2.pro).
    expected_offsets = np.array([-1.0, 0.0, 1.0]) / 3.0 * exptime_days
    np.testing.assert_allclose(weights1, [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(
        grid1[order1], t1[:, None] + expected_offsets[None, :]
    )


def test_shared_ninterp_different_exptime_broadcasts_per_observation(
    tmp_path_factory,
):
    """
    Given two transit instruments sharing ninterp=3 but with different
    exposure times (inst0: 30 min, inst1: 60 min),
    When load_data builds the oversampling groups,
    Then both instruments land in the single width-3 group (grouping keys
    on ninterp only), and each row's sub-exposure offsets scale with its
    OWN instrument's exptime (+/- 10 min for inst0, +/- 20 min for inst1
    at the midpoint (j + 0.5)/3 - 0.5 cell centers) -- pinning the
    per-observation exptime broadcast inside a shared group, which the
    mixed-ninterp tests above never exercise.
    """
    d = tmp_path_factory.mktemp("shared_ninterp")
    t0 = np.linspace(TC - 0.1, TC + 0.1, 5)
    t1 = np.linspace(TC - 0.05, TC + 0.05, 7)
    lc0 = _write_lc(d / "lc0.dat", t0)
    lc1 = _write_lc(d / "lc1.dat", t1)

    config = _config(
        [lc0, lc1],
        [
            {"exptime_s": 1800.0, "ninterp": 3},
            {"exptime_s": 3600.0, "ninterp": 3},
        ],
    )
    system = System(config, user_params=_params())
    system.prepare()

    tr = system.transit
    assert len(tr._exposure_groups) == 1
    rows, grid, weights = tr._exposure_groups[0]
    assert grid.shape == (12, 3)
    np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3])

    unit_offsets = np.array([-1.0, 0.0, 1.0]) / 3.0  # midpoint cell centers
    order = np.argsort(rows)
    times = np.concatenate([t0, t1])
    exptime_days = np.array([1800.0] * 5 + [3600.0] * 7) / 86400.0
    expected = times[:, None] + unit_offsets[None, :] * exptime_days[:, None]
    np.testing.assert_allclose(grid[order], expected)


def test_mixed_ninterp_model_flux_matches_instantaneous_only_for_ninterp_one(
    tmp_path_factory,
):
    """
    Given one mixed-ninterp component (inst0: ninterp=1; inst1: long
    exptime, ninterp=21), both instruments sampling the same fine time grid
    across one transit,
    When transit.model_flux is evaluated,
    Then inst0's rows match the plotted model for inst0 (ninterp=1, so
    the instantaneous model) to floating-point precision, while inst1's
    rows differ measurably from that same instantaneous curve (the two
    instruments share the band, so inst0's curve IS inst1's un-smeared
    model) -- proving the grouped oversampling actually reaches the
    model's per-observation output correctly for each instrument
    independently, not just the grid/weight arrays checked in the test
    above.
    """
    d = tmp_path_factory.mktemp("mixed_ninterp_flux")
    t = np.linspace(TC - 0.55 * T14, TC + 0.55 * T14, 401)
    lc0 = _write_lc(d / "lc0.dat", t)
    lc1 = _write_lc(d / "lc1.dat", t)

    config = _config(
        [lc0, lc1],
        [
            {"exptime_s": 1800.0, "ninterp": 1},
            {"exptime_s": 3600.0, "ninterp": 21},
        ],
    )
    system = System(config, user_params=_params())
    system.prepare()
    model = system.build_model()

    model_flux = _model_flux_at_initial_point(system, model)

    tr = system.transit
    inst0_rows = tr.inst_map == 0
    inst1_rows = tr.inst_map == 1

    param_values = _plot_param_values(system)
    baseline = np.atleast_1d(tr.baseline.initval)

    ref0, _ = tr._lc_at_times(param_values, 0, t)
    # inst1's instantaneous reference: the same band's un-smeared curve,
    # on inst1's own baseline (which multiplies the relative model).
    ref1 = baseline[1] * (ref0 / baseline[0])

    # inst0 (ninterp=1): exact match, exptime is ignored.
    np.testing.assert_allclose(model_flux[inst0_rows], ref0, atol=1e-12)

    # inst1 (ninterp=21, exptime=60min): smeared, so it must differ
    # measurably from the instantaneous reference.
    max_diff_inst1 = np.max(np.abs(model_flux[inst1_rows] - ref1))
    assert max_diff_inst1 > 1e-4


def test_plotted_model_matches_likelihood_model(tmp_path_factory):
    """
    Given a transit instrument with a long exptime and ninterp>1 (Jason's
    PR #20 review, point 2: plots must use the smeared model),
    When the plotted model -- Transit._lc_model compiled on the plot grid
    layout, the one function plot()/plot_data evaluate -- is evaluated at
    the data's own timestamps,
    Then it matches transit.model_flux (the likelihood's own smeared
    model) to floating-point precision, whereas the instantaneous model
    differs from it measurably for this same ninterp=21 instrument (the
    test above).  The plotting path is the likelihood code run on other
    times, so smearing is not re-implemented anywhere (reviews 1.5.6,
    7.14.1); tests/test_model_builder_parity.py extends this to every
    phase-curve term on a dense grid.
    """
    d = tmp_path_factory.mktemp("plot_matches_fit")
    t = np.linspace(TC - 0.55 * T14, TC + 0.55 * T14, 401)
    lc = _write_lc(d / "lc.dat", t)

    config = _config([lc], [{"exptime_s": 3600.0, "ninterp": 21}])
    system = System(config, user_params=_params())
    system.prepare()
    model = system.build_model()

    model_flux = _model_flux_at_initial_point(system, model)

    tr = system.transit
    param_values = _plot_param_values(system)
    baseline = float(np.atleast_1d(tr.baseline.initval)[0])

    plotted, plotted_matrix = tr._lc_at_times(param_values, 0, tr.time)
    np.testing.assert_allclose(plotted, model_flux, atol=1e-12)

    # The per-planet terms (the phased panels' entry point) sum to the
    # full curve minus the baseline.
    np.testing.assert_allclose(
        baseline + plotted_matrix.sum(axis=1), plotted, atol=1e-12
    )


def test_non_numeric_exptime_names_the_instrument(tmp_path_factory):
    """
    Given a transit instrument whose exptime_s is the string "abc",
    When the system is constructed (the base parses the exposure keys at
    construction, like mask: and columns:),
    Then it raises a ValueError naming the instrument and the offending
    value, chaining the original conversion error -- the bare float() used
    to surface it as "could not convert string to float: 'abc'" with no
    instrument in it (review 2.14.3).  Unlike the RANGE checks, which warn
    and fall back, a non-numeric value has no sensible fallback.
    """
    d = tmp_path_factory.mktemp("bad_exptime")
    t = np.linspace(TC - 0.1, TC + 0.1, 5)
    lcs = [_write_lc(d / "lc0.dat", t), _write_lc(d / "lc1.dat", t)]
    config = _config(lcs, [{"exptime_s": "abc", "ninterp": 3}, {}])

    with pytest.raises(ValueError, match=r"transit\[inst0\].*abc") as excinfo:
        System(config, user_params=_params())
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_the_retired_minutes_key_raises_naming_exptime_s(tmp_path_factory):
    """
    Given a transit instrument written with the retired exptime: 30 (the
    exposure time in MINUTES before the base owned it),
    When the system is constructed,
    Then it raises naming the instrument, the new key exptime_s and the
    converted value 1800 -- read as seconds, the old value would silently
    smear over 30 s instead of 30 min, so it is never accepted.
    """
    d = tmp_path_factory.mktemp("retired_exptime")
    lc = _write_lc(d / "lc.dat", np.linspace(TC - 0.1, TC + 0.1, 5))
    config = _config([lc], [{"exptime": 30, "ninterp": 10}])

    with pytest.raises(
        ValueError, match=r"transit\[inst0\].*exptime_s: 1800.*= 30 min"
    ):
        System(config, user_params=_params())


def test_a_component_that_does_not_smear_refuses_the_keys():
    """
    Given an rvinstrument file carrying exptime_s (the RV model is evaluated
    at each timestamp: its exposure averaging is not wired yet),
    When the system is constructed,
    Then it raises naming the component and the key, instead of ignoring a
    stated exposure.
    """
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "rvinstrument": [
            {"name": "HIRES", "file": "unused.rv", "exptime_s": 600.0}
        ],
    }
    with pytest.raises(
        NotImplementedError, match=r"rvinstrument\[HIRES\].*exptime_s"
    ):
        System(config, user_params=_params())


def _smear_host(exptime_s, ninterp, rule):
    """An object carrying just the base's exposure description, so
    Instrument.smear can be exercised without a model."""
    from types import SimpleNamespace

    from exozippy.components.instrument import Instrument

    host = SimpleNamespace(
        supports_exposure=True,
        EXPOSURE_RULE=rule,
        exptime_s=exptime_s,
        ninterp=ninterp,
    )
    host._blocks_inst_map = Instrument._blocks_inst_map
    host._node_mean = Instrument._node_mean
    host._exposure_partition = lambda im: Instrument._exposure_partition(
        host, im
    )
    return host, Instrument.smear


@pytest.mark.parametrize("rule", ["midpoint", "gauss"])
def test_smear_averages_every_tuple_member_with_one_set_of_weights(rule):
    """
    Given two files with different node counts (1 and 4) interleaved in the
    row blocks, and a model returning a TUPLE -- a scalar observable
    (n_g, k_g) and a 2-vector observable (n_g, k_g, 2) --
    When Instrument.smear builds the exposure average,
    Then every member comes back (N, ...) in ROW order, each equal to the
    rule-weighted mean of the model at that row's own sub-exposure nodes:
    the helper is observable-shape agnostic, which is what lets a vector
    observable be averaged as a vector before anything is projected from it.
    """
    import pytensor.tensor as pt

    from exozippy.components import exposure

    host, smear = _smear_host([60.0, 1800.0], [1, 4], rule)
    t_np = np.array([0.0, 0.3, 0.1, 0.4, 0.2])
    blocks = [(0, np.array([0, 2])), (1, np.array([1, 3, 4]))]

    def model(tt):
        return np.sin(40.0 * tt), np.stack([tt**2, np.cos(30.0 * tt)], -1)

    def fn(t_sub, rows):
        return (
            pt.sin(40.0 * t_sub),
            pt.stack([t_sub**2, pt.cos(30.0 * t_sub)], axis=-1),
        )

    t = pt.dvector("t")
    scalar, vector = smear(host, fn, t, blocks)
    got_s, got_v = pytensor.function([t], [scalar, vector])(t_np)

    frac, w = exposure.RULES[rule](4)
    expect_s = np.empty(5)
    expect_v = np.empty((5, 2))
    for r, tr in enumerate(t_np):
        if r in (0, 2):
            sub = np.array([tr])
            ww = np.ones(1)
        else:
            sub = tr + frac * 1800.0 / 86400.0
            ww = w
        ms, mv = model(sub)
        expect_s[r] = np.sum(ms * ww)
        expect_v[r] = np.sum(mv * ww[:, None], axis=0)
    np.testing.assert_allclose(got_s, expect_s, rtol=0, atol=1e-14)
    np.testing.assert_allclose(got_v, expect_v, rtol=0, atol=1e-14)


def test_the_two_rules_integrate_what_they_promise():
    """
    Given the midpoint rule (EXOFASTv2's grid) and the Gauss-Legendre rule,
    When each averages polynomials over one unit exposure,
    Then midpoint's nodes are EXOFASTv2's (j + 0.5)/n - 0.5 cell centers
    with weights 1/n, exact through degree 1 only, while n-node
    Gauss-Legendre is exact through degree 2n - 1 -- the convergence gap
    that makes a Gauss rule the better default where parity does not bind.
    """
    from exozippy.components import exposure

    x, w = exposure.midpoint_rule(10)
    # EXOFASTv2 spells it dindgen(n)/n - (n-1)/(2n); equal to rounding.
    np.testing.assert_allclose(
        x, np.arange(10) / 10 - (10 - 1) / (2 * 10), rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(w, np.full(10, 0.1))
    # mean of x^2 over [-1/2, 1/2] is 1/12; midpoint misses it.
    assert abs(np.sum(w * x**2) - 1.0 / 12.0) > 1e-4

    for n in (1, 2, 3, 5):
        x, w = exposure.gauss_legendre_rule(n)
        assert np.sum(w) == pytest.approx(1.0, abs=1e-15)
        for deg in range(2 * n):
            exact = 0.0 if deg % 2 else 0.5**deg / (deg + 1)
            assert np.sum(w * x**deg) == pytest.approx(exact, abs=1e-15)

"""
The plotted transit and RV models ARE the likelihood code run on other
times (reviews 1.5.5, 1.5.6, 7.14.1; JDE: "the plotting code should be
running the compiled likelihood code directly to generate its models to
plot/subtract").

Each data component has ONE expression builder -- ``Transit._lc_model``,
``RVInstrument._rv_model`` -- that takes a time tensor plus concrete
per-file row blocks.  ``build_likelihood`` calls it with the data's times
and blocks; ``compile_plotters`` calls it with a plot grid laid out in the
same per-file blocks.  These tests pin what that buys, with every optional
term switched ON so a drift in any of them would show:

  transit -- beaming, ellipsoidal, thermal, reflection, SED dilution, a
    detrend column, exposure smearing (ninterp=4) and light-travel time:
    the plotted model equals the likelihood's node at the DATA times, and
    on a DENSE grid it equals the likelihood a sibling system builds on
    that grid as its data.  Beaming and ellipsoidal are therefore un-
    smeared in the plot exactly as in the likelihood (1.5.6 was the NumPy
    re-smearing of the whole instantaneous curve, phase-curve terms
    included).  With two planets and ellipsoidal on, the per-planet terms
    telescope EXACTLY to the model (the old plot path folded only the
    baseline in per planet).

  RV -- two instruments, one with `rm:`, a detrend column, two orbits: the
    plotted model equals the likelihood at the data times for BOTH
    instruments and on a dense grid; the RM-free instrument's curve carries
    no RM signal (it is the bare Keplerian sum); the RM anomaly appears on
    its own instrument's curve and nowhere else, in the panels and in the
    phased-data cleaning (1.5.5: the old plot path put RM in a matrix
    column for every instrument, so phasing another orbit subtracted a
    spurious RM bump from the non-RM instrument's in-transit points).

The likelihood mu is read off the model's own observed node, not off the
retained attribute the plotters compile, so the comparison is between two
independently reached graphs.
"""

import os
import shutil

import numpy as np
import pytensor
import pytest
import yaml

from exozippy.system import System

_TC = 2459634.3
_PERIOD = 2.99
_EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
)


def _eval_at(node, system, param_values):
    """``node`` compiled against ``system.plot_params`` -- the SAME
    positional inputs the plotters take -- and evaluated at
    ``param_values``.

    Not a givens-substitution of the raw free RVs: the point's physical
    values are the raw ones pushed through the bounded transform once, and
    the two round trips disagree at ~1e-11 relative on a wide-bounded
    parameter (transit.baseline's bounds), which is exactly the noise this
    file must not mistake for a plot-vs-likelihood drift.  Feeding both
    graphs the same physical inputs is what a parity test is for.
    """
    fn = pytensor.function(
        [p.value for p in system.plot_params],
        node,
        on_unused_input="ignore",
    )
    return np.asarray(fn(*param_values))


def _likelihood_mu(system, model, name, param_values):
    """The mu tensor wired into the observed node ``name`` -- what the
    sampler scores against -- evaluated at ``param_values`` (see
    ``_eval_at``).  A sibling system with the same config has the same
    ``plot_params`` in the same order, so one system's values can be fed
    to another's mu."""
    return _eval_at(
        model.named_vars[name].owner.inputs[2], system, param_values
    )


def _fed_detrend_term(comp, system, param_values, sibling_comp):
    """The detrend term a SIBLING component's mu carries when fed THIS
    system's ``param_values``.

    A pinned parameter is still a plot input, and the detrend coefficient
    is sampled in WHITENED units (instrument.md): the same user value per
    raw column unit is a different internal number on each file, because
    each file's column has its own standard deviation.  So the sibling's
    mu, fed this system's internal coefficient, carries
    ``sibling.detrend_matrix @ c_this`` -- not the sibling's own
    ``detrend_at_data``.
    """
    labels = [p.label for p in system.plot_params]
    c_fed = np.atleast_1d(
        param_values[labels.index(comp.detrend_coeffs.label)]
    )
    return np.asarray(sibling_comp.detrend_matrix @ c_fed, dtype=float)


def _build(config, params):
    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    return system, model, point


# ---------------------------------------------------------------------------
# Transit: every term on
# ---------------------------------------------------------------------------

_THERMAL_PPM = 5000.0
_REFLECT_PPM = 2000.0
_ELLIP_PPM = 300.0
_BEAM_PPM = 4000.0


def _write_lc(path, t, rng):
    """Flux exactly 1 (so both systems seed the same baseline), plus one
    detrend column of pseudo-random values."""
    x = rng.normal(0.0, 1.0, len(t))
    np.savetxt(
        path,
        np.column_stack([t, np.ones_like(t), np.full_like(t, 1e-3), x]),
    )
    return str(path)


def _transit_config(lc_file, sed_file):
    """Two identical stars (SED dilution 0.5), one planet with every
    phase-curve term on, exposure smearing on, LTT on by default."""
    return {
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "planet": [{"name": "b", "fitbeam": True}],
        "orbit": [{"name": "b"}],
        "band": [
            {
                "name": "V",
                "filter": "V",
                "ld_law": "quadratic",
                "star_ndx": 0,
                "fitthermal": True,
                "fitreflect": True,
                "fitellip": True,
            }
        ],
        "transit": [
            {
                "name": "inst0",
                "file": lc_file,
                "band": "V",
                "exptime": 30.0,
                "ninterp": 4,
            }
        ],
        "sed": {"file": sed_file},
    }


def _transit_params():
    return {
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.08},
        "star.B.radius": {"initval": 1.0, "sigma": 0.05},
        "star.B.mass": {"initval": 1.0, "sigma": 0.05},
        "star.B.teff": {"initval": 5800, "sigma": 100},
        "star.B.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _TC},
        "orbit.0.cosi": {"initval": 0.05},
        # Eccentric, so the Kepler solve and the LTT delay are real work.
        "orbit.0.secosw": {"initval": 0.1, "sigma": 0.0},
        "orbit.0.sesinw": {"initval": 0.2, "sigma": 0.0},
        "planet.0.radius": {"initval": 1.7},
        "planet.0.beam": {"initval": _BEAM_PPM, "sigma": 0.0},
        "band.V.thermal": {"initval": _THERMAL_PPM, "sigma": 0.0},
        "band.V.reflect": {"initval": _REFLECT_PPM, "sigma": 0.0},
        "band.V.ellipsoidal": {"initval": _ELLIP_PPM, "sigma": 0.0},
        "transit.detrend_coeffs": {"initval": 0.002, "sigma": 0},
    }


@pytest.fixture(scope="module")
def transit_everything_on(tmp_path_factory):
    """The all-terms-on system on a landmark-sampled light curve, plus a
    sibling whose DATA are a dense grid over one full orbit."""
    d = tmp_path_factory.mktemp("builder_parity_lc")
    rng = np.random.default_rng(1)
    sed_file = d / "two_star.sed"
    sed_file.write_text("model: NextGen\nfilters: []\n")

    centers = [
        _TC,
        _TC + _PERIOD / 4.0,
        _TC + _PERIOD / 2.0,
        _TC + 0.8 * _PERIOD,
    ]
    t_sparse = np.concatenate(
        [np.linspace(c - 0.05, c + 0.05, 25) for c in centers]
    )
    lc_sparse = _write_lc(d / "sparse.dat", t_sparse, rng)
    system, model, point = _build(
        _transit_config(lc_sparse, str(sed_file)), _transit_params()
    )

    t_dense = np.linspace(_TC - 0.5 * _PERIOD, _TC + 0.5 * _PERIOD, 1500)
    lc_dense = _write_lc(d / "dense.dat", t_dense, rng)
    dense = _build(_transit_config(lc_dense, str(sed_file)), _transit_params())

    assert "transit.dilution" in model.named_vars  # dilution really is on
    assert system.transit.ninterp == [4]
    return system, model, point, dense


def test_transit_plotted_model_equals_likelihood_at_the_data(
    transit_everything_on,
):
    """
    Given beaming, ellipsoidal, thermal, reflection, SED dilution, a detrend
    column, ninterp=4 smearing and LTT all on,
    When the plotted model (the grid layout, fed the data's own times) is
    compared with the likelihood's mu at the observations,
    Then plotted + the fitted detrend term equals mu to floating-point
    precision -- and so does the likelihood's own node compiled for the
    plots, which is what the phased panels subtract.
    """
    system, model, point, _ = transit_everything_on
    tr = system.transit
    param_values = tr._point_to_plot_params(point, system)
    mu = _likelihood_mu(system, model, "transit_likelihood", param_values)
    detrend = tr.detrend_at_data(point)
    assert np.max(np.abs(detrend)) > 1e-4  # the detrend term is live

    plotted, plotted_terms = tr._lc_at_times(param_values, 0, tr.time)
    np.testing.assert_allclose(plotted + detrend, mu, rtol=0, atol=1e-14)

    phys, terms = tr._eval_lc_data(param_values)
    np.testing.assert_allclose(phys + detrend, mu, rtol=0, atol=1e-14)
    np.testing.assert_allclose(terms, plotted_terms, rtol=0, atol=1e-14)


def test_transit_plotted_model_equals_likelihood_on_a_dense_grid(
    transit_everything_on,
):
    """
    Given the same all-terms-on system,
    When its plotted model is evaluated on a dense grid over one full
    orbit and compared with the likelihood a SIBLING system builds with
    that grid as its data (same parameters, its own detrend term removed),
    Then they agree to floating-point precision everywhere -- through
    transit, both quadratures and secondary eclipse.  Beaming and
    ellipsoidal are un-smeared in both by construction (review 1.5.6).
    """
    system, model, point, (dense, dense_model, dense_point) = (
        transit_everything_on
    )
    tr = system.transit
    param_values = tr._point_to_plot_params(point, system)
    t_dense = dense.transit.time

    plotted, _ = tr._lc_at_times(param_values, 0, t_dense)
    # The sibling's likelihood at THIS system's parameter values (same
    # config, same plot_params order); its own pinned detrend term removed.
    mu_dense = _likelihood_mu(
        dense, dense_model, "transit_likelihood", param_values
    )
    reference = mu_dense - _fed_detrend_term(
        tr, system, param_values, dense.transit
    )

    np.testing.assert_allclose(plotted, reference, rtol=0, atol=1e-14)
    # The phase curve is really there: ppm-level structure away from the
    # transit and the eclipse.
    out = np.abs(((t_dense - _TC) / _PERIOD + 0.5) % 1.0 - 0.5) > 0.1
    assert np.ptp(plotted[out]) > 1e-3


def test_transit_plot_data_traces_come_from_the_same_expression(
    transit_everything_on,
):
    """
    Given the same system,
    When plot_data builds its charts,
    Then the unphased model trace equals the plotted model on its own grid
    and the phased trace equals the planet's term on the period window --
    the panels have no third path.
    """
    system, model, point, _ = transit_everything_on
    tr = system.transit
    param_values = tr._point_to_plot_params(point, system)
    specs = tr.plot_data(system, point)

    unphased = [s for s in specs if not s.meta["phase_folded"]][0]
    trace = [t for t in unphased.traces if t.role == "model"][0]
    expected, _ = tr._lc_at_times(param_values, 0, trace.x)
    np.testing.assert_allclose(trace.y, expected, rtol=0, atol=1e-14)

    phased = [s for s in specs if s.meta["phase_folded"]][0]
    trace = [t for t in phased.traces if t.role == "model"][0]
    tc = phased.meta["tc"]
    _, terms = tr._lc_at_times(param_values, 0, trace.x + tc)
    # The fold maps both ends of the period window, tc -/+ P/2, onto phase
    # -0.5, so the two endpoint samples share an x and were computed one
    # period apart -- equal to ~1e-11, not to 1e-14.  Compare the interior.
    np.testing.assert_allclose(
        trace.y[1:-1], terms[1:-1, 0], rtol=0, atol=1e-14
    )
    assert len(unphased.param_deps) > 0 and len(phased.param_deps) > 0


def test_two_planet_ellipsoidal_terms_telescope_exactly(tmp_path_factory):
    """
    Given two planets on one light curve with ellipsoidal variation on,
    When the per-planet terms the phased panels draw are summed,
    Then baseline + sum equals the model to floating-point precision --
    each term is the change its planet's step makes to the RUNNING model,
    so the ellipsoidal factor multiplies the earlier planets' terms exactly
    as it does in the likelihood.  The old plot path folded only the
    baseline in per planet, exact for one planet and not for two.
    """
    d = tmp_path_factory.mktemp("builder_parity_two_planets")
    t = np.concatenate(
        [
            np.linspace(_TC - 0.06, _TC + 0.06, 40),
            np.linspace(_TC + 0.7, _TC + 0.9, 40),
        ]
    )
    np.savetxt(
        d / "lc.dat",
        np.column_stack([t, np.ones_like(t), np.full_like(t, 1e-3)]),
    )
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}, {"name": "c", "orbit_ndx": 1}],
        "orbit": [{"name": "b"}, {"name": "c"}],
        "band": [
            {
                "name": "V",
                "filter": "V",
                "ld_law": "quadratic",
                "fitellip": True,
            }
        ],
        "transit": [{"name": "inst0", "file": str(d / "lc.dat"), "band": "V"}],
    }
    params = {
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.b.period": {"initval": _PERIOD},
        "orbit.b.tc": {"initval": _TC},
        "orbit.b.cosi": {"initval": 0.05},
        "orbit.c.period": {"initval": 1.6 * _PERIOD},
        "orbit.c.tc": {"initval": _TC + 0.8},
        "orbit.c.cosi": {"initval": 0.02},
        "planet.b.radius": {"initval": 1.7},
        "planet.c.radius": {"initval": 1.2},
        "band.V.ellipsoidal": {"initval": _ELLIP_PPM, "sigma": 0.0},
    }
    system, model, point = _build(config, params)
    tr = system.transit
    assert system.planet.n_elements == 2
    param_values = tr._point_to_plot_params(point, system)

    phys, terms = tr._eval_lc_data(param_values)
    baseline = tr._point_value(point, tr.baseline, 0)
    np.testing.assert_allclose(
        baseline + terms.sum(axis=1), phys, rtol=0, atol=1e-13
    )
    mu = _likelihood_mu(system, model, "transit_likelihood", param_values)
    np.testing.assert_allclose(phys, mu, rtol=0, atol=1e-14)
    # Both planets contribute, and the ellipsoidal term is live (a
    # nonzero term far from either planet's transit).
    assert np.max(np.abs(terms[:, 0])) > 1e-3
    assert np.max(np.abs(terms[:, 1])) > 1e-4


# ---------------------------------------------------------------------------
# RV: two instruments, one with rm:, a detrend column, two orbits
# ---------------------------------------------------------------------------


def _kelt17_split(tmp_path, n_rm=40, dense=False):
    """The KELT-17 example with its RV file split across two instruments
    (only TRES_RM tagged `rm: b`), a detrend column appended to both, and
    a second (non-transiting, RM-free) orbit c.  ``dense`` replaces the
    times with a dense in-transit grid (TRES_RM) and a dense full-orbit
    grid (TRES_ORB) so a sibling likelihood exists on a plot grid."""
    exdir = os.path.join(_EXAMPLES, "kelt17")
    if not os.path.exists(os.path.join(exdir, "kelt17.yaml")):
        pytest.skip("kelt17 example not present")
    work = str(tmp_path)
    for name in os.listdir(exdir):
        shutil.copy(os.path.join(exdir, name), work)

    rng = np.random.default_rng(7)
    rv = np.loadtxt(os.path.join(exdir, "KELT-17.TRES.rv"))
    with open(os.path.join(exdir, "kelt17.params.yaml")) as fh:
        params = yaml.safe_load(fh)
    tc = float(params["orbit.b.tc"]["initval"])
    period = float(params["orbit.b.period"]["initval"])
    if dense:
        t_rm = np.linspace(tc - 0.12, tc + 0.12, 300)
        t_orb = np.linspace(tc - 0.5 * period, tc + 0.5 * period, 300)
        rm_rows = np.column_stack(
            [t_rm, np.zeros_like(t_rm), np.full_like(t_rm, 100.0)]
        )
        orb_rows = np.column_stack(
            [t_orb, np.zeros_like(t_orb), np.full_like(t_orb, 100.0)]
        )
    else:
        rm_rows, orb_rows = rv[:n_rm], rv[n_rm:]
    for fname, rows in (("rm_inst.rv", rm_rows), ("orb_inst.rv", orb_rows)):
        x = rng.normal(0.0, 1.0, len(rows))
        np.savetxt(os.path.join(work, fname), np.column_stack([rows, x]))

    with open(os.path.join(exdir, "kelt17.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    cfg["rvinstrument"] = [
        {"name": "TRES_RM", "file": "rm_inst.rv", "rm": "b", "rm_band": "V"},
        {"name": "TRES_ORB", "file": "orb_inst.rv"},
    ]
    cfg["planet"].append({"name": "c", "orbit_ndx": 1})
    cfg["orbit"].append({"name": "c", "primary": ["A"], "companion": ["c"]})
    params["orbit.c.period"] = {"initval": 11.3}
    params["orbit.c.tc"] = {"initval": tc + 4.1}
    params["orbit.c.secosw"] = {"initval": 0.0, "sigma": 0}
    params["orbit.c.sesinw"] = {"initval": 0.0, "sigma": 0}
    params["planet.c.mass"] = {"initval": 3.0}
    # One coefficient per (instrument, column); a nonzero pinned value so
    # the detrend term is live on both files.
    params["rvinstrument.detrend_coeffs"] = {"initval": 0.001, "sigma": 0}

    cwd = os.getcwd()
    try:
        os.chdir(work)
        system, model, point = _build(cfg, params)
    finally:
        os.chdir(cwd)
    return system, model, point


@pytest.fixture(scope="module")
def rm_split(tmp_path_factory):
    system, model, point = _kelt17_split(tmp_path_factory.mktemp("rm_split"))
    dense = _kelt17_split(tmp_path_factory.mktemp("rm_dense"), dense=True)
    rv = system.rvinstrument
    assert rv.rm_orbit == ["b", None]
    assert len(rv._plot_orbit_map) == 2
    return system, model, point, dense


def _keplerian_matrix(system, param_values, t):
    """The bare Keplerian per-orbit RV matrix (no RM) at ``t``, in
    internal units, straight from the orbit component."""
    import pytensor.tensor as pt

    rv = system.rvinstrument
    K_vec, omap = rv._orbit_rv_terms(system, rv.star_ndx[0])
    node = system.orbit.get_radial_velocity(
        pt.as_tensor_variable(np.asarray(t, dtype=float)), K_vec, omap
    )
    return _eval_at(node, system, param_values)


def _keplerian_sum(system, param_values, t):
    return _keplerian_matrix(system, param_values, t).sum(axis=1)


def test_rv_plotted_model_equals_likelihood_at_the_data(rm_split):
    """
    Given two RV instruments (one with rm:), a detrend column and two
    orbits,
    When each instrument's plotted model at its own observation times is
    compared with the likelihood's mu on its rows,
    Then plotted + gamma + the fitted detrend term equals mu for BOTH
    instruments to floating-point precision, and so does the likelihood's
    own per-orbit matrix compiled for the plots.
    """
    system, model, point, _ = rm_split
    rv = system.rvinstrument
    param_values = rv._point_to_plot_params(point, system)
    mu = _likelihood_mu(system, model, "rvinstrument.model", param_values)
    detrend = rv.detrend_at_data(point)
    assert np.max(np.abs(detrend)) > 0.0

    for i in range(rv.n_elements):
        rows = rv.rows(i)
        gamma = rv._point_value(point, rv.gamma, i)
        plotted, _ = rv._rv_at_times(param_values, i, rv.time[rows])
        np.testing.assert_allclose(
            plotted + gamma + detrend[rows], mu[rows], rtol=0, atol=1e-14
        )

    matrix = np.asarray(rv._rv_data_fn(*param_values))
    gamma_all = np.array(
        [rv._point_value(point, rv.gamma, i) for i in rv.inst_map]
    )
    np.testing.assert_allclose(
        matrix.sum(axis=1) + gamma_all + detrend, mu, rtol=0, atol=1e-14
    )


def test_rv_plotted_model_equals_likelihood_on_a_dense_grid(rm_split):
    """
    Given the same system,
    When each instrument's plotted model on a dense grid (through the
    transit for the RM file, a full orbit for the other) is compared with
    the likelihood a SIBLING system builds with those grids as its data,
    Then they agree to floating-point precision, RM anomaly included.
    """
    system, model, point, (dense, dense_model, dense_point) = rm_split
    rv = system.rvinstrument
    param_values = rv._point_to_plot_params(point, system)
    # The sibling's likelihood at THIS system's parameter values (same
    # config, same plot_params order -- gamma included, so it is this
    # system's gamma that comes back off); the sibling's own pinned
    # detrend term removed.
    mu_dense = _likelihood_mu(
        dense, dense_model, "rvinstrument.model", param_values
    )
    drv = dense.rvinstrument
    detrend_dense = _fed_detrend_term(rv, system, param_values, drv)

    for i in range(drv.n_elements):
        rows = drv.rows(i)
        gamma = rv._point_value(point, rv.gamma, i)
        reference = mu_dense[rows] - gamma - detrend_dense[rows]
        plotted, _ = rv._rv_at_times(param_values, i, drv.time[rows])
        np.testing.assert_allclose(plotted, reference, rtol=0, atol=1e-14)


def test_rm_free_instrument_plots_no_rm_signal(rm_split):
    """
    Given the same system and a dense grid through the transit,
    When both instruments' plotted models are evaluated on it,
    Then the RM-free instrument's curve IS the bare Keplerian sum (no RM
    signal at all), while the RM instrument's curve differs from it by
    hundreds of m/s in transit -- the anomaly is drawn on the curve of the
    file whose likelihood scored it and on no other (review 1.5.5).
    """
    system, model, point, _ = rm_split
    rv = system.rvinstrument
    param_values = rv._point_to_plot_params(point, system)
    tc = rv._point_value(point, system.orbit.tc, 0)
    t = np.linspace(tc - 0.08, tc + 0.08, 200)

    y_orb, _ = rv._rv_at_times(param_values, 1, t)
    y_rm, _ = rv._rv_at_times(param_values, 0, t)
    keplerian = _keplerian_sum(system, param_values, t)

    np.testing.assert_allclose(y_orb, keplerian, rtol=0, atol=1e-14)
    anomaly_ms = (y_rm - y_orb) * rv._rv_factor()
    assert np.max(np.abs(anomaly_ms)) > 100.0


def test_rm_appears_only_on_its_own_instrument_in_the_charts(rm_split):
    """
    Given the same system,
    When plot_data builds its charts,
    Then the unphased chart's "model" trace is the RM-free reference and a
    "TRES_RM model" trace carries the anomaly over that file's own span;
    the phased chart for orbit b has the same pair while orbit c's (where
    neither file has RM) has the reference alone; and the RM parameters
    reach param_deps so a GUI slider on them refreshes the charts.
    """
    system, model, point, _ = rm_split
    rv = system.rvinstrument
    specs = rv.plot_data(system, point)

    unphased = [s for s in specs if not s.meta["phase_folded"]][0]
    names = [t.name for t in unphased.traces if t.role == "model"]
    assert names == ["model", "TRES_RM model"]
    rm_trace = unphased.traces[names.index("TRES_RM model")]
    t_rm = rv.time[rv.rows(0)]
    assert rm_trace.x.min() == pytest.approx(t_rm.min())
    assert rm_trace.x.max() == pytest.approx(t_rm.max())
    assert any(lbl.endswith("svcoslam") for lbl in unphased.param_deps)

    phased = {s.meta["orbit"]: s for s in specs if s.meta["phase_folded"]}
    assert [t.name for t in phased["b"].traces if t.role == "model"] == [
        "model",
        "TRES_RM model",
    ]
    assert [t.name for t in phased["c"].traces if t.role == "model"] == [
        "model"
    ]
    # In the RM orbit's panel the two curves differ exactly by the anomaly
    # (hundreds of m/s at mid-transit, phase 0.25), nowhere else.
    ref, rm = (t for t in phased["b"].traces if t.role == "model")
    diff = np.abs(rm.y - ref.y)
    in_transit = np.abs(rm.x - 0.25) < 0.02
    assert np.max(diff[in_transit]) > 100.0
    assert np.max(diff[~in_transit]) < 1e-6


def test_phased_cleaning_subtracts_no_rm_from_the_rm_free_instrument(
    rm_split,
):
    """
    Given the same system,
    When the phased panel for orbit c removes orbit b's signal from the
    data ("other orbits"),
    Then the RM-free instrument's points have exactly the Keplerian orbit-b
    signal removed and the RM instrument's points have Keplerian + RM
    removed -- each file's rows carry the model its likelihood scored.
    The old plot path evaluated its RM-for-everyone matrix at the data
    times, so the non-RM instrument's in-transit points lost a spurious RM
    bump here (review 1.5.5).
    """
    system, model, point, _ = rm_split
    rv = system.rvinstrument
    param_values = rv._point_to_plot_params(point, system)
    shared = rv._phased_shared(system, point)
    matrix = shared["data_rv_matrix"]
    col_b = list(rv._plot_orbit_map).index(0)

    # TRES_ORB's rows: every column is Keplerian, orbit b's included.
    rows_orb = rv.rows(1)
    np.testing.assert_allclose(
        matrix[rows_orb],
        _keplerian_matrix(system, param_values, rv.time[rows_orb]),
        rtol=0,
        atol=1e-14,
    )
    # TRES_RM's rows: orbit c's column is Keplerian, orbit b's column is
    # Keplerian + RM (hundreds of m/s in transit).
    rows_rm = rv.rows(0)
    kep_rm = _keplerian_matrix(system, param_values, rv.time[rows_rm])
    other_c = np.ones(matrix.shape[1], dtype=bool)
    other_c[col_b] = False
    np.testing.assert_allclose(
        matrix[rows_rm][:, other_c], kep_rm[:, other_c], rtol=0, atol=1e-14
    )
    excess_ms = (
        matrix[rows_rm][:, col_b] - kep_rm[:, col_b]
    ) * rv._rv_factor()
    assert np.max(np.abs(excess_ms)) > 100.0

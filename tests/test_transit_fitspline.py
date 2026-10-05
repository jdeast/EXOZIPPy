"""fitspline: EXOFASTv2's Kepler-spline detrending, co-fit as detrend columns.

  - spline.py: the basis is a port of keplerspline.pro / bspline_bkpts.pro
    (gap split at diff(t) > splinespace, long(range/bkspace)+1 breakpoints,
    3 padding knots per side, a cubic), pinned by VALUE as well as shape.
  - Transit: the per-file fitspline/splinespace keys; fitspline off leaves
    the detrend path byte-for-byte unchanged; the spline columns land in
    that file's detrend block with ONE column dropped, which is what makes
    the block non-singular after whitening (pinning the baseline would not);
    a synthetic low-frequency wobble is flattened and the transit depth
    recovered; the plot path (detrend_at_data), caption and prose carry it.

Rationale for every ruling: components/instrument.md, "fitspline".
"""

import logging

import numpy as np
import pymc as pm
import pytest

from conftest import _DummyConfigManager
from exozippy.components.transit import spline
from exozippy.components.transit.transit import Transit
from exozippy.system import System

# ---------------------------------------------------------------------------
# spline.py: the basis
# ---------------------------------------------------------------------------

# gj1214's MIRILRS light curve spans 1.6486 d: at the default 0.75 d spacing
# keplerspline gives long(1.6486/0.75)+1 = 3 breakpoints -> 5 coefficients.
_SPAN = 1.6486


def test_knot_vector_matches_bspline_bkpts():
    """
    Given 3 evenly spaced breakpoints on the rescaled [0, 1] segment,
    When the full knot vector is built,
    Then it is the breakpoints plus 3 padding knots per side at the same
    spacing -- bspline_bkpts.pro's `for i=1, nord-1` loop with nord=4.
    """
    knots = spline.segment_knots(3)
    np.testing.assert_allclose(
        knots, [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    )


@pytest.mark.parametrize(
    "span, splinespace, n_bkpts",
    [
        (_SPAN, 0.75, 3),  # MIRILRS
        (_SPAN, 0.6, 3),  # the BIC-preferred spacing: still 3
        (1.5, 0.75, 3),  # exact multiple: long(2.0) + 1
        (0.1, 0.75, 2),  # a ground-based visit: clamped to 2 -> one cubic
        (13.0, 0.75, 18),  # a TESS orbit
    ],
)
def test_breakpoint_count_is_keplerspline_rule(span, splinespace, n_bkpts):
    """
    Given a segment span and a spacing,
    When the breakpoint count is computed,
    Then it is long(range/bkspace) + 1, never fewer than 2.
    """
    assert spline.segment_n_bkpts(span, splinespace) == n_bkpts


def test_basis_shape_and_partition_of_unity():
    """
    Given MIRILRS-like times (one gap-free segment of 1.6486 d),
    When the basis is built at the default spacing,
    Then it has 5 columns (3 breakpoints + 2) and every row sums to 1.
    """
    t = np.linspace(0.0, _SPAN, 400) + 2459781.0
    basis = spline.spline_basis(t, spline.DEFAULT_SPLINESPACE)
    assert basis.shape == (400, 5)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-14)


def test_basis_values_at_segment_start_are_the_cubic_knot_values():
    """
    Given uniform knots,
    When the basis is evaluated at the first breakpoint,
    Then the nonzero values are the textbook uniform cubic B-spline values
    at a knot, (1/6, 2/3, 1/6) -- a value that owes nothing to scipy or to
    this module, so a wrong knot vector or degree cannot reproduce it.
    """
    t = np.linspace(0.0, _SPAN, 400)
    basis = spline.spline_basis(t, 0.75)
    np.testing.assert_allclose(basis[0], [1 / 6, 2 / 3, 1 / 6, 0.0, 0.0])
    np.testing.assert_allclose(basis[-1], [0.0, 0.0, 1 / 6, 2 / 3, 1 / 6])


def test_basis_reproduces_any_cubic_in_time():
    """
    Given a cubic polynomial in time,
    When it is least-squares fit with the basis,
    Then the fit is exact: a cubic spline space contains every cubic, which
    is also why a user's time-polynomial detrend column is degenerate with
    it (see test_time_detrend_column_with_fitspline_raises).
    """
    t = np.linspace(0.0, _SPAN, 300)
    y = 1.0 + 0.3 * t - 0.2 * t**2 + 0.05 * t**3
    basis = spline.spline_basis(t, 0.75)
    coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    np.testing.assert_allclose(basis @ coeffs, y, atol=1e-12)


def test_gap_wider_than_splinespace_starts_a_new_segment():
    """
    Given two 1.6486 d chunks separated by a 1 d gap (> 0.75 d),
    When the basis is built,
    Then each chunk gets its own 5-column block, zero on the other chunk.
    """
    a = np.linspace(0.0, _SPAN, 200)
    b = a + _SPAN + 1.0
    t = np.concatenate([a, b])
    basis = spline.spline_basis(t, 0.75)
    assert basis.shape == (400, 10)
    assert np.all(basis[:200, 5:] == 0.0)
    assert np.all(basis[200:, :5] == 0.0)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-14)


def test_gap_equal_to_splinespace_does_not_split():
    """
    Given a gap of exactly splinespace,
    When the segments are found,
    Then there is one segment: keplerspline.pro splits on diff(t) GT ndays.
    """
    t = np.array([0.0, 0.1, 0.2, 0.95, 1.0])  # the 0.2 -> 0.95 gap is 0.75
    assert len(spline.split_segments(t, 0.75)) == 1
    assert len(spline.split_segments(t, 0.7499)) == 2


def _noisy_wobble(t, seed):
    rng = np.random.default_rng(seed)
    return (
        1.0
        + 1e-3 * np.sin(2.0 * np.pi * t / 1.3)
        + 1e-4 * rng.standard_normal(t.size)
    )


@pytest.mark.parametrize(
    "span, splinespace, n_coeffs",
    [
        (_SPAN, 0.75, 5),  # MIRILRS
        (3.0, 0.75, 7),  # exact multiple: long(4.0) + 1 -> 5 breakpoints
        (0.1, 0.75, 4),  # shorter than the spacing: one cubic
        (13.0, 0.75, 20),  # a TESS orbit
        (5.3, 0.5, 13),
    ],
)
def test_fit_matches_vanderburg_keplersplinev2(span, splinespace, n_coeffs):
    """
    Given a noisy low-frequency wobble on one gap-free segment,
    When it is least-squares fit with this basis and with Vanderburg's own
    Python keplerspline (tests/third_party/keplersplinev2.py, pydl's
    bspline.iterfit) at maxiter=1 -- one pass, no outlier clipping,
    Then the two fitted curves agree to rounding: the same knots, so the
    same spline space, built independently of this port.
    """
    from third_party import keplersplinev2

    t = np.linspace(0.0, span, 500)
    f = _noisy_wobble(t, seed=1)
    basis = spline.spline_basis(t, splinespace)
    assert basis.shape[1] == n_coeffs
    coeffs, *_ = np.linalg.lstsq(basis, f, rcond=None)
    theirs = keplersplinev2.kepler_spline(
        t, f, bkspace=splinespace, maxiter=1
    )[0]
    np.testing.assert_allclose(basis @ coeffs, theirs, rtol=0, atol=1e-12)


def test_gapped_fit_matches_vanderburg_keplersplinev2():
    """
    Given two chunks separated by a gap wider than splinespace,
    When they are fit with this basis and with keplersplinev2's split() +
    kepler_spline(maxiter=1) per chunk (gap_width = splinespace, as
    EXOFASTv2 uses ndays for both),
    Then both find 2 segments and the fitted curves agree to rounding.
    """
    from third_party import keplersplinev2

    t = np.concatenate(
        [np.linspace(0.0, 1.6, 300), np.linspace(3.0, 4.9, 350)]
    )
    f = _noisy_wobble(t, seed=2)
    basis = spline.spline_basis(t, 0.75)
    coeffs, *_ = np.linalg.lstsq(basis, f, rcond=None)
    chunks_t, chunks_f = keplersplinev2.split(t, f, gap_width=0.75)
    assert len(chunks_t) == len(spline.split_segments(t, 0.75)) == 2
    theirs = np.concatenate(
        [
            keplersplinev2.kepler_spline(a, b, bkspace=0.75, maxiter=1)[0]
            for a, b in zip(chunks_t, chunks_f)
        ]
    )
    np.testing.assert_allclose(basis @ coeffs, theirs, rtol=0, atol=1e-12)


def test_zero_span_segment_raises_naming_the_file():
    """
    Given a point isolated from the rest by a gap wider than splinespace,
    When the basis is built,
    Then it raises naming the label (keplerspline.pro divides by the zero
    span there; there is no spline to fit through one instant).
    """
    t = np.array([0.0, 0.1, 0.2, 0.3, 5.0])
    with pytest.raises(ValueError, match=r"\[transit\[MIRI\]\].*no time span"):
        spline.spline_basis(t, 0.75, label="transit[MIRI]")


# ---------------------------------------------------------------------------
# Transit: the per-file keys
# ---------------------------------------------------------------------------


def _transit(tmp_path, entries):
    path = tmp_path / "lc.dat"
    path.write_text("2459781.0 1.0 0.001\n")
    config = [
        {"name": f"f{i}", "file": str(path), "band": "TESS", **e}
        for i, e in enumerate(entries)
    ]
    return Transit(config, _DummyConfigManager())


def test_keys_default_off_with_exofast_spacing(tmp_path):
    """
    Given a transit entry with neither key,
    When the component is constructed,
    Then fitspline is False and splinespace is EXOFASTv2's 0.75 d.
    """
    tr = _transit(tmp_path, [{}])
    assert tr.fitspline == [False]
    assert tr.splinespace == [0.75]


def test_keys_are_per_file(tmp_path):
    """
    Given three files, only the second fitting a spline at 0.6 d,
    When the component is constructed,
    Then each file carries its own values.
    """
    tr = _transit(
        tmp_path,
        [{}, {"fitspline": True, "splinespace": 0.6}, {"fitspline": False}],
    )
    assert tr.fitspline == [False, True, False]
    assert tr.splinespace == [0.75, 0.6, 0.75]


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_non_boolean_fitspline_raises(tmp_path, bad):
    """
    Given fitspline spelled as EXOFASTv2's 0/1, a string, or null,
    When the component is constructed,
    Then it raises naming the file -- a guessed boolean silently changes
    the model.
    """
    with pytest.raises(ValueError, match=r"\[transit\[f0\]\] fitspline"):
        _transit(tmp_path, [{"fitspline": bad}])


@pytest.mark.parametrize("bad", [0, -0.5, "abc", float("nan"), True])
def test_bad_splinespace_raises(tmp_path, bad):
    """
    Given a non-positive, non-numeric, NaN or boolean splinespace,
    When the component is constructed,
    Then it raises naming the file.
    """
    with pytest.raises(ValueError, match=r"\[transit\[f0\]\] splinespace"):
        _transit(tmp_path, [{"fitspline": True, "splinespace": bad}])


def test_splinespace_without_fitspline_warns(tmp_path, caplog):
    """
    Given splinespace on a file that does not set fitspline,
    When the component is constructed,
    Then it warns that splinespace is ignored.
    """
    with caplog.at_level(logging.WARNING):
        _transit(tmp_path, [{"splinespace": 0.5}])
    assert "splinespace is set but fitspline is not true" in caplog.text


# ---------------------------------------------------------------------------
# Transit: built systems
# ---------------------------------------------------------------------------

_TC = 2459781.8
_PERIOD = 1.58040453
_T0 = _TC - 0.8
_ERR = 2e-4
# GJ 1214b-like planet: the truth is set through planet.radius (p is
# derived), and p_true is read back from the built truth model rather than
# hand-converted.
_RADIUS_TRUE = 0.2427  # jupiterRad; with star.radius 0.215 -> p ~0.116
# A low-frequency baseline wobble over the 1.65 d span, 10x the noise and
# HIGH at the transit, so a constant baseline reads the transit as shallower.
_WOBBLE = 2e-3


def _wobble(t):
    return _WOBBLE * np.cos(2.0 * np.pi * (t - _TC) / 1.65)


def _params(radius_sigma=None, radius=_RADIUS_TRUE):
    # Everything but the radius, the baseline and the detrend coefficients is
    # pinned ON THE SAMPLED COORDINATE (logmass, logP, q1/q2 -- a sigma on
    # the derived mass/period/u1/u2 would be ignored), so the depth has no
    # degenerate partner (stellar density, limb darkening) and the test
    # isolates what the spline does to it.
    p = {
        "star.0.radius": {"initval": 0.215, "sigma": 0.0},
        "star.0.logmass": {"initval": float(np.log10(0.178)), "sigma": 0.0},
        "orbit.0.logP": {"initval": float(np.log10(_PERIOD)), "sigma": 0.0},
        "orbit.0.tc": {"initval": _TC, "sigma": 0.0},
        "orbit.0.cosi": {"initval": 0.02, "sigma": 0.0},
        "orbit.0.secosw": {"initval": 0.0, "sigma": 0.0},
        "orbit.0.sesinw": {"initval": 0.0, "sigma": 0.0},
        "band.TESS.q1": {"initval": 0.25, "sigma": 0.0},
        "band.TESS.q2": {"initval": 0.3, "sigma": 0.0},
        "transit.0.jitter_variance": {"initval": 0.0, "sigma": 0.0},
        "planet.0.radius": {"initval": radius},
    }
    if radius_sigma is not None:
        p["planet.0.radius"]["sigma"] = radius_sigma
    return p


def _config(lc_file, **entry):
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "TESS", "filter": "TESS", "ld_law": "quadratic"}],
        "transit": [
            {"name": "MIRI", "file": lc_file, "band": "TESS", **entry}
        ],
    }


def _build(lc_file, entry, params):
    """(system, model, raw start, internal start)."""
    system = System(_config(lc_file, **entry), user_params=params)
    system.prepare()
    model = system.build_model()
    raw = system.get_raw_start(model)
    with model:
        internal = system.get_internal_point(model, raw)
    return system, model, raw, internal


def _model_flux(system, model, raw_point):
    """build_likelihood's own model node (baseline + detrend + transit) --
    what the sampler fits, not a separately compiled plot curve."""
    fn = _compile_at_values(model, system.transit._model_flux_node)
    return np.asarray(fn(_raw(model, raw_point)))


def _node_value(model, param, raw_point):
    """``param``'s element 0 evaluated from the model's OWN tensor at a raw
    point.  Not ``_point_value``: that falls back to the initval for a
    parameter absent from the point (a derived one like planet.p is never a
    named Deterministic), which would let a recovery test pass vacuously."""
    fn = _compile_at_values(model, param.value)
    return float(np.atleast_1d(fn(_raw(model, raw_point)))[0])


def _compile_at_values(model, node):
    """Compile ``node`` as a function of the raw point.  The RVs must be
    replaced by their value variables first: compiled as-is, the graph
    still contains the random variables and DRAWS from their priors."""
    (node,) = model.replace_rvs_by_values([node])
    return model.compile_fn(
        node, inputs=model.value_vars, on_unused_input="ignore"
    )


def _raw(model, point):
    """Just the model's value variables (find_MAP's point carries more)."""
    return {v.name: point[v.name] for v in model.value_vars}


@pytest.fixture(scope="module")
def truth_lc(tmp_path_factory):
    """EXOZIPPy's own transit at the truth (so the test asserts RECOVERY,
    not agreement between two light-curve codes), plus the wobble, white
    noise and an airmass-like extra column so the user-column path is
    exercised alongside the spline.  Returns (path, t, transit_only,
    p_true)."""
    d = tmp_path_factory.mktemp("fitspline")
    t = np.linspace(_T0, _T0 + 1.65, 1200)
    flat = d / "flat.dat"
    np.savetxt(
        flat, np.column_stack([t, np.ones_like(t), np.full_like(t, _ERR)])
    )
    system, model, raw, _ = _build(str(flat), {}, _params(radius_sigma=0.0))
    baseline = _node_value(model, system.transit.baseline, raw)
    transit_only = _model_flux(system, model, raw) - baseline
    p_true = _node_value(model, system.planet.p, raw)
    assert 0.10 < p_true < 0.13  # GJ 1214b-like, not a unit slip

    rng = np.random.default_rng(1214)
    airmass = 1.2 + 0.1 * np.sin(2.0 * np.pi * (t - _T0) / 0.37)
    flux = 1.0 + transit_only + _wobble(t) + rng.normal(0.0, _ERR, t.size)
    path = d / "wobble.dat"
    np.savetxt(
        path, np.column_stack([t, flux, np.full_like(t, _ERR), airmass])
    )
    return str(path), t, transit_only, p_true


@pytest.fixture(scope="module")
def spline_on(truth_lc):
    return _build(
        truth_lc[0], {"fitspline": True}, _params(radius=0.9 * _RADIUS_TRUE)
    )


@pytest.fixture(scope="module")
def spline_off(truth_lc):
    return _build(truth_lc[0], {}, _params(radius=0.9 * _RADIUS_TRUE))


def test_fitspline_off_is_the_unchanged_detrend_path(
    truth_lc, spline_off, monkeypatch
):
    """
    Given the same file with fitspline absent and with fitspline: false,
    When both systems are built,
    Then the detrend matrices, column counts and start logp are identical,
    and the spline builder is never even called on the off path.
    """
    system_absent, model_absent, raw_absent, _ = spline_off

    def _boom(*args, **kwargs):
        raise AssertionError("_spline_columns called with fitspline off")

    monkeypatch.setattr(Transit, "_spline_columns", _boom)
    system_false, model_false, raw_false, _ = _build(
        truth_lc[0], {"fitspline": False}, _params(radius=0.9 * _RADIUS_TRUE)
    )

    tr_a, tr_f = system_absent.transit, system_false.transit
    assert tr_a.n_detrend_per_inst == tr_f.n_detrend_per_inst == [1]
    np.testing.assert_array_equal(tr_a.detrend_matrix, tr_f.detrend_matrix)
    assert model_absent.compile_logp()(
        raw_absent
    ) == model_false.compile_logp()(raw_false)


def test_spline_columns_join_the_file_block_minus_one(spline_on, spline_off):
    """
    Given one airmass column and fitspline at 0.75 d over 1.65 d,
    When the system is built,
    Then the file's block is 1 user column + (5 - 1) spline columns, the
    user column is placed exactly as without the spline, and the spline
    columns are the whitened basis minus its last column.
    """
    on, off = spline_on[0].transit, spline_off[0].transit
    assert on.n_detrend_per_inst == [1 + 4]
    np.testing.assert_array_equal(
        on.detrend_matrix[:, 0], off.detrend_matrix[:, 0]
    )

    basis = spline.spline_basis(on.time, 0.75)[:, :-1]
    whitened = (basis - basis.mean(axis=0)) / basis.std(axis=0)
    np.testing.assert_allclose(on.detrend_matrix[:, 1:], whitened, atol=1e-12)


def test_block_is_full_rank_and_logp_finite(spline_on):
    """
    Given fitspline on,
    When the design [1 | whitened block] is inspected and logp evaluated,
    Then it has full column rank (the degeneracy with the baseline is
    resolved) and logp and its gradient are finite at the start.
    """
    system, model, raw, _ = spline_on
    X = system.transit.detrend_matrix
    design = np.column_stack([np.ones(X.shape[0]), X])
    assert np.linalg.matrix_rank(design) == design.shape[1]
    assert np.isfinite(model.compile_logp()(raw))
    assert np.all(np.isfinite(model.compile_dlogp()(raw)))


def test_pinning_the_baseline_instead_would_leave_the_block_singular():
    """
    Given the FULL 5-column basis, whitened the way _build_block_detrend
    whitens it,
    When its rank is measured,
    Then it is 4 of 5: mean subtraction moves the partition-of-unity
    degeneracy inside the spline block, so pinning the baseline cannot fix
    it and dropping one column does (instrument.md, "fitspline").
    """
    t = np.linspace(0.0, _SPAN, 500)
    basis = spline.spline_basis(t, 0.75)
    whitened = (basis - basis.mean(axis=0)) / basis.std(axis=0)
    assert np.linalg.matrix_rank(whitened) == 4
    dropped = whitened[:, :-1]
    assert np.linalg.matrix_rank(dropped) == dropped.shape[1]


def test_time_detrend_column_with_fitspline_raises(tmp_path):
    """
    Given a detrend column that is the time itself, plus fitspline,
    When the system is prepared,
    Then it raises naming the file: a linear-in-time column lies inside the
    cubic spline's span, so the pair is exactly degenerate.
    """
    t = np.linspace(_T0, _T0 + 1.65, 300)
    path = tmp_path / "timecol.dat"
    np.savetxt(
        path, np.column_stack([t, np.ones_like(t), np.full_like(t, _ERR), t])
    )
    system = System(_config(str(path), fitspline=True), user_params=_params())
    with pytest.raises(
        ValueError, match=r"\[transit\[MIRI\]\] fitspline: .*degenerate"
    ):
        system.prepare()


def _fit(system, model, raw):
    """MAP of the free parameters (planet radius, baseline, detrend
    coefficients).  Returns (p at the MAP, the raw MAP point)."""
    with model:
        mp = pm.find_MAP(start=raw, progressbar=False, maxeval=10000)
    return _node_value(model, system.planet.p, mp), mp


@pytest.fixture(scope="module")
def fits(spline_on, spline_off):
    return {
        "on": _fit(*spline_on[:3]),
        "off": _fit(*spline_off[:3]),
    }


def test_spline_flattens_the_wobble(spline_on, spline_off, truth_lc, fits):
    """
    Given a transit plus a 2e-3 low-frequency wobble (10x the noise),
    When the model is fit to MAP with and without fitspline,
    Then with it the out-of-transit residual is white at the noise level,
    and without it the residual still carries the wobble.
    """
    oot = truth_lc[2] == 0.0
    rms = {}
    for tag, built in (("on", spline_on), ("off", spline_off)):
        system, model = built[0], built[1]
        resid = system.transit.flux - _model_flux(system, model, fits[tag][1])
        rms[tag] = np.std(resid[oot])
    assert rms["on"] < 1.1 * _ERR
    assert rms["off"] > 3.0 * _ERR


def test_spline_recovers_the_transit_depth(truth_lc, fits):
    """
    Given the same fits,
    When the radius ratio is read off the MAP,
    Then with fitspline it is recovered to <1%, while without it the wobble
    (high at the transit) biases it low by >3%.
    """
    p_true = truth_lc[3]
    p_on, p_off = fits["on"][0], fits["off"][0]
    assert abs(p_on - p_true) / p_true < 0.01
    assert (p_true - p_off) / p_true > 0.03


def test_detrend_at_data_includes_the_spline(spline_on):
    """
    Given a point with nonzero coefficients on every detrend column,
    When detrend_at_data (the plot correction) is evaluated,
    Then it equals the likelihood's own X.c term, spline columns included:
    zeroing just the spline coefficients changes it.
    """
    tr = spline_on[0].transit
    coeffs = np.random.default_rng(0).normal(0.0, 1e-3, tr.total_detrend_cols)
    full = tr.detrend_at_data({tr.detrend_coeffs.label: coeffs})
    user_only = coeffs.copy()
    user_only[1:] = 0.0
    without = tr.detrend_at_data({tr.detrend_coeffs.label: user_only})

    np.testing.assert_allclose(full, tr.detrend_matrix @ coeffs)
    np.testing.assert_allclose(
        full - without, tr.detrend_matrix[:, 1:] @ coeffs[1:]
    )
    assert np.max(np.abs(full - without)) > 1e-4


def test_caption_names_the_spline_only_when_fit(spline_on, spline_off):
    """
    Given fitspline on and off,
    When the detrend caption is read,
    Then the spline sentence is appended only when a file fits a spline,
    and the off caption is exactly the base Instrument caption.
    """
    from exozippy.components.instrument import Instrument

    on, off = spline_on[0].transit, spline_off[0].transit
    assert on.detrend_caption().startswith(Instrument.detrend_caption(on))
    assert "cubic B-spline" in on.detrend_caption()
    assert off.detrend_caption() == Instrument.detrend_caption(off)


def test_prose_cites_vanderburg_only_when_fit(spline_on, spline_off):
    """
    Given fitspline on and off,
    When the modeling-draft collector is read,
    Then the fitspline sentence (citing Vanderburg:2014, naming the file and
    the spacing) is present only when a file fits a spline.
    """
    from exozippy.outputs.prose import get_collector

    on = [
        s
        for s in get_collector(spline_on[0]).sentences()
        if s.key == "transit.fitspline"
    ]
    off = [
        s
        for s in get_collector(spline_off[0]).sentences()
        if s.key == "transit.fitspline"
    ]
    assert len(on) == 1 and off == []
    assert on[0].cite_keys() == ["Vanderburg:2014"]
    assert "MIRI light curve" in on[0].text
    assert "every 0.75~d" in on[0].text

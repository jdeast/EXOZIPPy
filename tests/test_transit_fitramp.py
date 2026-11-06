"""fitramp: EXOFASTv2's exponential detector ramp (FITRAMP) on a transit file.

  - The model: R(t) = 1 - A * (1 - exp(-(t - t0) / tau)), t0 the file's
    MINIMUM time, multiplied into the model with the baseline; R(t0) == 1
    exactly and R -> 1 - A late (exofast_chi2v2.pro's -A form).
  - The seed: a least-squares fit of the DECAYING form lands near an
    injected ramp on its own (EXOFASTv2's ramp_func fits the opposite sign,
    so its seed is deliberately not ported), and MAP recovers it.
  - Opt-in: a file without fitramp has no ramp parameters and an unchanged
    model, logp and phased panel, bit for bit.
  - tau sits on the logit transform over [1e-5 d, 10 x span]: the gradient
    is finite next to both walls, under the C backend and under numpyro.

Rationale for every ruling: components/instrument.md, "fitramp".
"""

import numpy as np
import pymc as pm
import pytensor
import pytest
import yaml

from conftest import _DummyConfigManager
from exozippy.components.transit import ramp
from exozippy.components.transit.transit import Transit
from exozippy.system import System

_TC = 2459781.8
_PERIOD = 1.58040453
_ERR = 2e-4
_N = 1000
# A JWST-like visit: 0.5 d centered on the transit.
_T_START = _TC - 0.25
_SPAN = 0.5
_RADIUS_TRUE = 0.2427  # jupiterRad; with star.radius 0.215 -> p ~0.116
# The injected ramp: 20x the per-point noise, e-folding in ~1 h.
_A_TRUE = 0.004
_TAU_TRUE = 0.04


def _params(radius=_RADIUS_TRUE, radius_sigma=None, extra=None):
    # Everything but the radius, baseline and ramp is pinned on the sampled
    # coordinate, as in test_transit_fitspline, so the depth has no
    # degenerate partner and the test isolates the ramp.
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
    p.update(extra or {})
    return p


def _config(files):
    """``files``: [(name, path, extra entry keys), ...]."""
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "TESS", "filter": "TESS", "ld_law": "quadratic"}],
        "transit": [
            {"name": name, "file": path, "band": "TESS", **entry}
            for name, path, entry in files
        ],
    }


def _build(files, params):
    """(system, model, raw start, internal start), plotters compiled."""
    system = System(_config(files), user_params=params)
    system.prepare()
    model = system.build_model()
    raw = system.get_raw_start(model)
    with model:
        internal = system.get_internal_point(model, raw)
    return system, model, raw, internal


def _compile_at_values(model, node):
    """``node`` as a function of the raw point (RVs replaced by values)."""
    (node,) = model.replace_rvs_by_values([node])
    return model.compile_fn(
        node, inputs=model.value_vars, on_unused_input="ignore"
    )


def _raw(model, point):
    return {v.name: point[v.name] for v in model.value_vars}


def _node_value(model, param, raw_point, i=0):
    """Element ``i`` of ``param`` from the model's OWN tensor."""
    fn = _compile_at_values(model, param.value)
    return float(np.atleast_1d(fn(_raw(model, raw_point)))[i])


def _model_flux(system, model, raw_point):
    fn = _compile_at_values(model, system.transit._model_flux_node)
    return np.asarray(fn(_raw(model, raw_point)))


def _values(system, point, **replace):
    """Plot-parameter values at ``point`` with transit params replaced
    (``ramp_amp=[...]`` etc., in internal units)."""
    tr = system.transit
    values = list(tr._point_to_plot_params(point, system))
    labels = [p.label for p in system.plot_params]
    for name, val in replace.items():
        values[labels.index(getattr(tr, name).label)] = np.atleast_1d(
            np.asarray(val, dtype=float)
        )
    return values


def _mu(system, model, values):
    """The mu the likelihood scores, read off the observed node."""
    node = model.named_vars["transit_likelihood"].owner.inputs[2]
    fn = pytensor.function(
        [p.value for p in system.plot_params], node, on_unused_input="ignore"
    )
    return np.asarray(fn(*values))


def _ramp(t, t0, amp, tau):
    """The ramp written out independently (the test's reference)."""
    return 1.0 - amp * (1.0 - np.exp(-(t - t0) / tau))


# ---------------------------------------------------------------------------
# Data: EXOZIPPy's own transit at the truth, times an injected ramp
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def truth(tmp_path_factory):
    """Returns (dir, ramp file, no-ramp file, t, transit_only, truth radius
    in internal units).  The file's
    rows are written SHUFFLED, so t0 = min(time) is tested, not row 0."""
    d = tmp_path_factory.mktemp("fitramp")
    t = np.linspace(_T_START, _T_START + _SPAN, _N)
    flat = d / "flat.dat"
    np.savetxt(
        flat, np.column_stack([t, np.ones_like(t), np.full_like(t, _ERR)])
    )
    system, model, raw, _ = _build(
        [("MIRI", str(flat), {})], _params(radius_sigma=0.0)
    )
    baseline = _node_value(model, system.transit.baseline, raw)
    transit_only = _model_flux(system, model, raw) - baseline
    # The truth radius as the model's own tensor holds it (internal unit).
    radius_internal = _node_value(model, system.planet.radius, raw)
    assert transit_only.min() < -0.01  # a GJ 1214b-like dip, not a unit slip

    rng = np.random.default_rng(1214)
    flux = (1.0 + transit_only) * _ramp(t, t[0], _A_TRUE, _TAU_TRUE)
    flux = flux + rng.normal(0.0, _ERR, t.size)
    order = rng.permutation(t.size)
    path = d / "ramp.dat"
    np.savetxt(
        path,
        np.column_stack([t, flux, np.full_like(t, _ERR)])[order],
    )
    # The same times with NO ramp, for the opt-in identity tests.
    noramp = d / "noramp.dat"
    np.savetxt(
        noramp,
        np.column_stack(
            [
                t,
                1.0 + transit_only + rng.normal(0.0, _ERR, t.size),
                np.full_like(t, _ERR),
            ]
        ),
    )
    return d, str(path), str(noramp), t, transit_only, radius_internal


@pytest.fixture(scope="module")
def ramp_on(truth):
    return _build(
        [("MIRI", truth[1], {"fitramp": True})],
        _params(radius=0.95 * _RADIUS_TRUE),
    )


# ---------------------------------------------------------------------------
# The per-file key
# ---------------------------------------------------------------------------


def _transit(tmp_path, entries):
    path = tmp_path / "lc.dat"
    path.write_text("2459781.0 1.0 0.001\n")
    config = [
        {"name": f"f{i}", "file": str(path), "band": "TESS", **e}
        for i, e in enumerate(entries)
    ]
    return Transit(config, _DummyConfigManager())


def test_fitramp_defaults_off_and_is_per_file(tmp_path):
    """
    Given three files, only the second setting fitramp,
    When the component is constructed,
    Then each file carries its own flag and the default is off.
    """
    tr = _transit(tmp_path, [{}, {"fitramp": True}, {"fitramp": False}])
    assert tr.fitramp == [False, True, False]


@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_non_boolean_fitramp_raises(tmp_path, bad):
    """
    Given fitramp spelled as EXOFASTv2's 0/1, a string or null,
    When the component is constructed,
    Then it raises naming the file.
    """
    with pytest.raises(ValueError, match=r"\[transit\[f0\]\] fitramp"):
        _transit(tmp_path, [{"fitramp": bad}])


def test_tau_floor_is_the_defaults_lower_bound():
    """
    Given ramp.TAU_FLOOR (the seed's floor) and defaults.yaml's
    ramp_tau.lower (the sampled bound),
    When both are read,
    Then they are the same number, so the seed cannot start below the
    bound it is sampled inside.
    """
    from pathlib import Path

    import exozippy.components.transit as pkg

    with open(Path(pkg.__file__).parent / "defaults.yaml") as f:
        defaults = yaml.safe_load(f)
    assert defaults["transit"]["ramp_tau"]["lower"] == ramp.TAU_FLOOR
    assert defaults["transit"]["ramp_tau"]["lower"] > 0


def test_zero_span_ramp_file_raises(tmp_path):
    """
    Given fitramp on a file whose times do not span an interval,
    When the system is prepared,
    Then it raises naming the file -- there is no ramp to fit.
    """
    path = tmp_path / "flat.dat"
    t = np.full(5, _TC)
    np.savetxt(path, np.column_stack([t, np.ones(5), np.full(5, _ERR)]))
    system = System(
        _config([("MIRI", str(path), {"fitramp": True})]),
        user_params=_params(),
    )
    with pytest.raises(ValueError, match=r"\[transit\[MIRI\]\] fitramp"):
        system.prepare()


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def test_t0_is_the_minimum_time_and_the_bound_is_ten_spans(ramp_on, truth):
    """
    Given a ramp file written with its rows shuffled,
    When the system is built,
    Then t0 is the minimum time and tau's upper bound is 10 x the span.
    """
    system = ramp_on[0]
    tr = system.transit
    t = truth[3]
    assert tr.ramp_t0[0] == t.min()
    assert tr.ramp_span[0] == pytest.approx(_SPAN, rel=1e-12)
    assert np.atleast_1d(tr.ramp_tau.upper)[0] == pytest.approx(
        10.0 * _SPAN, rel=1e-12
    )
    assert np.atleast_1d(tr.ramp_tau.lower)[0] == ramp.TAU_FLOOR


def test_ramp_is_one_at_t0_and_levels_to_one_minus_a(ramp_on):
    """
    Given A = 0.003 and tau = 0.002 d,
    When the compiled model is evaluated out of transit at t0 and 60 tau
    later,
    Then model / baseline is exactly 1 at t0 and 1 - A late, and at every
    observation the likelihood's ramp node equals the ramp written out.
    """
    system, model, _, point = ramp_on
    tr = system.transit
    amp, tau = 0.003, 0.002
    values = _values(system, point, ramp_amp=[amp], ramp_tau=[tau])
    b = tr._point_value(point, tr.baseline, 0)
    t0 = tr.ramp_t0[0]

    times = np.array([t0, t0 + 60.0 * tau])
    full, matrix = tr._lc_at_times(values, 0, times)
    assert np.all(matrix == 0.0)  # both out of transit: rel == 1 exactly
    assert full[0] == b  # R(t0) == 1 exactly (expm1(0) == 0)
    np.testing.assert_allclose(full[1] / b, 1.0 - amp, rtol=1e-14)

    _, _, data_ramp = tr._eval_lc_data_and_ramp(values)
    assert data_ramp[np.argmin(tr.time)] == 1.0
    np.testing.assert_allclose(
        data_ramp, _ramp(tr.time, t0, amp, tau), rtol=1e-14, atol=0
    )


def test_ramp_multiplies_with_the_baseline_in_the_likelihood(ramp_on):
    """
    Given the ramp file at its start,
    When the likelihood's mu is read off the observed node,
    Then mu == ramp * (baseline + sum(planet terms)): the ramp multiplies
    the whole model, and the planet terms carry none of it.
    """
    system, model, _, point = ramp_on
    tr = system.transit
    values = _values(system, point)
    phys, terms, data_ramp = tr._eval_lc_data_and_ramp(values)
    b = tr._point_value(point, tr.baseline, 0)
    np.testing.assert_allclose(
        phys, data_ramp * (b + terms.sum(axis=1)), rtol=1e-14
    )
    np.testing.assert_allclose(phys, _mu(system, model, values), rtol=1e-14)
    assert data_ramp.min() < 1.0 - 1e-4  # the seeded ramp is not flat


def test_phased_panel_divides_the_ramp_out_of_the_data(ramp_on):
    """
    Given a ramp file,
    When the phased arrays are built,
    Then the data and error bars are divided by baseline * ramp, the model
    curve (a grid at the seeded epoch) by the baseline alone, and every
    (data - model) / err is the likelihood's own normalized residual -- so
    no ramp leaks into the fold.
    """
    system, model, _, point = ramp_on
    tr = system.transit
    values = _values(system, point)
    _, terms, data_ramp = tr._eval_lc_data_and_ramp(values)
    b = tr._point_value(point, tr.baseline, 0)
    prep = tr._phased_lc_arrays(system, point, 0, 0)

    norm = b * data_ramp
    np.testing.assert_allclose(prep["yerr_data"], tr.err / norm, rtol=1e-14)
    mu = _mu(system, model, values)
    np.testing.assert_allclose(
        (prep["y_data"] - terms[:, 0] / b) / prep["yerr_data"],
        (tr.flux - mu) / tr.err,
        rtol=1e-9,
        atol=1e-9,
    )
    assert np.all(np.isfinite(prep["y_model"]))
    assert -0.05 < prep["y_model"].min() < -1e-3  # a relative dip


# ---------------------------------------------------------------------------
# Seed and recovery
# ---------------------------------------------------------------------------


def test_seed_alone_lands_near_the_injected_ramp(ramp_on):
    """
    Given a transit times an injected A = 0.004, tau = 0.04 d ramp,
    When the data-derived seed is read (before any fit),
    Then it lands near the truth: the decaying form fits the decaying data
    (EXOFASTv2's sign-flipped ramp_func would not).
    """
    system, model, raw, _ = ramp_on
    tr = system.transit
    amp, tau = tr.ramp_seed[0]
    assert amp == pytest.approx(_A_TRUE, rel=0.10)
    assert tau == pytest.approx(_TAU_TRUE, rel=0.20)
    assert tr.baseline_init[0] == pytest.approx(1.0, abs=5e-4)
    # ...and it IS the model's start.
    assert _node_value(model, tr.ramp_amp, raw) == pytest.approx(amp)
    assert _node_value(model, tr.ramp_tau, raw) == pytest.approx(tau)


def test_seed_on_a_flat_light_curve_finds_no_ramp():
    """
    Given white noise with no ramp,
    When the seed is fit,
    Then A is consistent with zero (no invented ramp).
    """
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, _SPAN, _N)
    flux = 1.0 + rng.normal(0.0, _ERR, t.size)
    _, amp, _ = ramp.seed_ramp(t, flux, np.full_like(t, _ERR), ramp.TAU_FLOOR)
    assert abs(amp) < 5 * _ERR


@pytest.fixture(scope="module")
def ramp_map(ramp_on):
    system, model, raw, _ = ramp_on
    with model:
        mp = pm.find_MAP(start=raw, progressbar=False, maxeval=10000)
    return mp


def test_map_recovers_the_injected_ramp_and_depth(truth, ramp_on, ramp_map):
    """
    Given the injected ramp and a radius started 5% low,
    When the model is fit to MAP,
    Then A, tau and the radius are recovered and the residual is white at
    the noise level.
    """
    system, model, _, _ = ramp_on
    tr = system.transit
    amp = _node_value(model, tr.ramp_amp, ramp_map)
    tau = _node_value(model, tr.ramp_tau, ramp_map)
    radius = _node_value(model, system.planet.radius, ramp_map)
    assert amp == pytest.approx(_A_TRUE, rel=0.03)
    assert tau == pytest.approx(_TAU_TRUE, rel=0.05)
    assert radius == pytest.approx(truth[5], rel=0.01)
    resid = tr.flux - _model_flux(system, model, ramp_map)
    assert np.std(resid) < 1.1 * _ERR


# ---------------------------------------------------------------------------
# Opt-in: a file without fitramp is unchanged
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ramp_off(truth):
    return _build(
        [("MIRI", truth[2], {})], _params(radius=0.95 * _RADIUS_TRUE)
    )


def test_without_fitramp_nothing_is_built(truth, ramp_off, monkeypatch):
    """
    Given the same file with fitramp absent and with fitramp: false,
    When both systems are built,
    Then neither has ramp parameters, the ramp builder is never called,
    the start logp is identical, and the data ramp is exactly 1.0.
    """
    system_a, model_a, raw_a, point_a = ramp_off

    def _boom(*args, **kwargs):
        raise AssertionError("_ramp_profile called with fitramp off")

    monkeypatch.setattr(Transit, "_ramp_profile", _boom)
    system_f, model_f, raw_f, _ = _build(
        [("MIRI", truth[2], {"fitramp": False})],
        _params(radius=0.95 * _RADIUS_TRUE),
    )
    for system, model in ((system_a, model_a), (system_f, model_f)):
        tr = system.transit
        assert "ramp_amp" not in tr.manifest
        assert "ramp_tau" not in tr.manifest
        assert not hasattr(tr, "ramp_amp")
        assert not any("ramp" in v.name for v in model.free_RVs)
    assert model_a.compile_logp()(raw_a) == model_f.compile_logp()(raw_f)
    _, _, data_ramp = system_a.transit._eval_lc_data_and_ramp(
        _values(system_a, point_a)
    )
    assert np.all(data_ramp == 1.0)


def test_without_fitramp_the_phased_panel_is_the_old_expression(ramp_off):
    """
    Given no fitramp,
    When the phased arrays are built,
    Then they are BIT-FOR-BIT the pre-ramp expression
    ((flux - baseline - other - gp) / baseline, err / baseline): the ramp
    is exactly 1.0, so multiplying and dividing by it is exact.
    """
    system, _, _, point = ramp_off
    tr = system.transit
    shared = tr._phased_lc_shared(system, point)
    prep = tr._phased_lc_arrays(system, point, 0, 0, shared=shared)
    b = tr._point_value(point, tr.baseline, 0)
    rows = tr.rows(0)
    other = np.sum(shared["data_terms"][rows][:, []], axis=1)
    old_y = (
        shared["flux_corrected"][rows]
        - b
        - other
        - shared["extra_signals"][rows]
    ) / b
    old_err = shared["err_corrected"][rows] / b
    np.testing.assert_array_equal(prep["y_data"], old_y)
    np.testing.assert_array_equal(prep["yerr_data"], old_err)


def test_a_mixed_config_leaves_the_other_file_untouched(truth):
    """
    Given two files, only the first with fitramp,
    When the system is built,
    Then the second file's ramp is exactly 1.0 at every row and its ramp
    elements are pinned (not sampled), while the first file's ramp is not
    flat.
    """
    params = _params(radius=0.95 * _RADIUS_TRUE)
    params["transit.1.jitter_variance"] = {"initval": 0.0, "sigma": 0.0}
    system, model, raw, point = _build(
        [
            ("MIRI", truth[1], {"fitramp": True}),
            ("OTHER", truth[2], {}),
        ],
        params,
    )
    tr = system.transit
    _, _, data_ramp = tr._eval_lc_data_and_ramp(_values(system, point))
    assert np.all(data_ramp[tr.rows(1)] == 1.0)
    assert data_ramp[tr.rows(0)].min() < 1.0 - 1e-4
    assert tr.ramp_amp.element_is_sampled(0)
    assert tr.ramp_tau.element_is_sampled(0)
    assert not tr.ramp_amp.element_is_sampled(1)
    assert not tr.ramp_tau.element_is_sampled(1)
    assert np.isfinite(model.compile_logp()(raw))


# ---------------------------------------------------------------------------
# Exposure smearing: the ramp is averaged on the planet terms' own nodes
# ---------------------------------------------------------------------------


def test_ramp_is_averaged_over_each_exposure(truth):
    """
    Given the ramp file with exptime_s = 300 s and ninterp = 5, and a
    short tau = 0.002 d (so the ramp curves visibly within one exposure),
    When the likelihood's ramp node is read at the observations,
    Then it equals the node-weighted mean of the ramp written out on each
    row's own sub-exposure nodes (nodes before t0 held at R = 1), differs
    from the timestamp ramp where it curves, and the model is still
    ramp * (baseline + sum(planet terms)).
    """
    system, model, raw, point = _build(
        [
            (
                "MIRI",
                truth[1],
                {"fitramp": True, "exptime_s": 300.0, "ninterp": 5},
            )
        ],
        _params(radius=0.95 * _RADIUS_TRUE),
    )
    tr = system.transit
    amp, tau = 0.003, 0.002
    values = _values(system, point, ramp_amp=[amp], ramp_tau=[tau])
    phys, terms, data_ramp = tr._eval_lc_data_and_ramp(values)

    t0 = tr.ramp_t0[0]
    offsets, weights = tr.exposure_nodes(0)
    assert offsets.size == 5
    t_sub = tr.time[:, None] + offsets[None, :]
    expected = _ramp(np.maximum(t_sub, t0), t0, amp, tau) @ weights
    np.testing.assert_allclose(data_ramp, expected, rtol=1e-13, atol=0)
    # The average is not the timestamp value where the ramp curves.
    assert np.max(np.abs(data_ramp - _ramp(tr.time, t0, amp, tau))) > 1e-6

    b = tr._point_value(point, tr.baseline, 0)
    np.testing.assert_allclose(
        phys, data_ramp * (b + terms.sum(axis=1)), rtol=1e-14
    )
    np.testing.assert_allclose(phys, _mu(system, model, values), rtol=1e-14)


# ---------------------------------------------------------------------------
# Gradients next to tau's walls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tau_start",
    [0.98 * 10.0 * _SPAN, 1.02 * ramp.TAU_FLOOR],
    ids=["near_upper", "near_lower"],
)
def test_gradient_is_finite_next_to_the_tau_bounds(truth, tau_start):
    """
    Given tau started 2% inside either wall of [1e-5 d, 10 x span],
    When logp and its gradient are evaluated at the start,
    Then both are finite on every element (the logit transform has no
    chi2 = infinity cliff, unlike EXOFASTv2's rejection).
    """
    system, model, raw, _ = _build(
        [("MIRI", truth[1], {"fitramp": True})],
        _params(extra={"transit.0.ramp_tau": {"initval": tau_start}}),
    )
    assert _node_value(model, system.transit.ramp_tau, raw) == pytest.approx(
        tau_start, rel=1e-6
    )
    assert np.isfinite(model.compile_logp()(raw))
    grad = model.compile_dlogp()(raw)
    assert np.all(np.isfinite(grad))


@pytest.mark.slow
def test_numpyro_samples_finitely_next_to_the_upper_bound(truth):
    """
    Given tau started 2% below its 10 x span upper bound,
    When a short chain is sampled with nuts_sampler="numpyro",
    Then every draw is finite and tau stays inside its bounds.

    The house rule (test_element_parameterization): a finite C-backend
    gradient is not proof the JAX path is sound, so the JAX path is run.
    """
    system, model, raw, _ = _build(
        [("MIRI", truth[1], {"fitramp": True})],
        _params(
            extra={"transit.0.ramp_tau": {"initval": 0.98 * 10.0 * _SPAN}}
        ),
    )
    with model:
        idata = pm.sample(
            draws=30,
            tune=30,
            chains=1,
            cores=1,
            nuts_sampler="numpyro",
            initvals=raw,
            progressbar=False,
            random_seed=0,
        )
    for var in idata.posterior.data_vars:
        assert np.all(np.isfinite(idata.posterior[var].values)), (
            f"non-finite draws for {var}"
        )
    tau = idata.posterior[system.transit.ramp_tau.label].values
    assert np.all((tau > ramp.TAU_FLOOR) & (tau <= 10.0 * _SPAN))

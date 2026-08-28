"""The microlensing likelihood is Gaussian in FLUX, and only in flux.

Photon-counting noise is (approximately) Gaussian in flux; a magnitude is a
logarithm of it.  Modeling the light curve in magnitudes therefore
  (a) is only a first-order approximation, degrading exactly where the data are
      faint, and
  (b) is undefined for the non-positive fluxes difference imaging routinely
      produces -- which the old code papered over by clamping them to 1e-30,
      i.e. ~75 mag, and then feeding those fabricated points to the fit.

These tests pin the three claims that matter:

  1. non-positive fluxes are first class -- no clamp, finite logp, finite
     gradient (test_negative_fluxes_*);
  2. ``data_format`` is now purely a statement about the FILE -- a magnitude
     file and its exactly converted flux twin build the identical model
     (test_magnitude_file_and_flux_twin_*);
  3. a magnitude file gives back the posterior it used to, to the conversion's
     own O(sigma_mag) precision, and the discrepancy shrinks proportionally
     with the photometric error (test_magnitude_posterior_matches_*).
"""

import os

import numpy as np
import pytensor
import pytest

from exozippy.system import System

# d mag / d ln flux; sigma_mag = K * sigma_flux / flux.
K = 2.5 / np.log(10.0)

T0 = 2460025.0
U0 = 0.30
TE = 25.0


def _pspl_flux(t, f_source, f_blend, t0=T0, u0=U0, tE=TE):
    """Paczynski light curve in flux (no parallax; the fit adds none)."""
    tau = (t - t0) / tE
    u = np.sqrt(tau**2 + u0**2)
    A = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    return f_source * A + f_blend


def _config(fmt, inst_extra=None):
    entry = {"name": "LC", "file": "lc.dat", "data_format": fmt}
    entry.update(inst_extra or {})
    return {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [
            {
                "name": "Lens",
                "lenses": ["star.0"],
                "sources": ["star.1"],
                "finite_source": False,
                "t0_par": T0,
                # Never shell out to MMEXOFAST from a unit test; the start
                # values below are all the bootstrap needs.
                "mmexofast": False,
            }
        ],
        "mulensinstrument": [entry],
    }


def _params(extra=None):
    p = {
        "lens.Lens.t_0": {"initval": T0},
        "lens.Lens.u_0": {"initval": U0},
        "lens.Lens.t_E": {"initval": TE},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
    }
    for nm in ("Lens", "Source"):
        p[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        p[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}
    p.update(extra or {})
    return p


def _build(tmpdir, rows, fmt, extra_params=None, inst_extra=None):
    """Write ``rows`` as lc.dat in ``tmpdir`` and build the system there."""
    path = os.path.join(str(tmpdir), "lc.dat")
    np.savetxt(path, rows, fmt="%.17g")
    cwd = os.getcwd()
    os.chdir(str(tmpdir))
    try:
        system = System(
            _config(fmt, inst_extra), user_params=_params(extra_params)
        )
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model


def _observation(model):
    """(mu, sigma, observed) of the single mulensing Normal, as callables."""
    obs = [v for v in model.observed_RVs if "mulens" in v.name][0]
    ins = obs.owner.inputs
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values([ins[-2], ins[-1]]),
        on_unused_input="ignore",
    )
    data = np.asarray(obs.tag.observations.eval(), dtype=float).ravel()
    return fn, data


def _eval_mu_sigma(model, fn, point):
    return [
        np.asarray(a, dtype=float).ravel()
        for a in fn(*[point[v.name] for v in model.value_vars])
    ]


def _eval_node(model, node, point):
    """Evaluate a model node at ``point``.

    ``replace_rvs_by_values`` is not optional: a Parameter's ``.value`` graph
    is written over the RandomVariables, so compiling it against value_vars
    silently DRAWS FROM THE PRIOR instead of reading the point.
    """
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values([node]),
        on_unused_input="ignore",
    )
    out = fn(*[point[v.name] for v in model.value_vars])[0]
    return np.atleast_1d(np.asarray(out, dtype=float))


# ---------------------------------------------------------------------------
# 1. Negative fluxes are first class
# ---------------------------------------------------------------------------


DIA_FSOURCE = 1.0
DIA_FBLEND = -0.45  # heavy negative blending, as OGLE difference imaging shows
DIA_ERR = 0.6  # a faint source: baseline S/N below 1


@pytest.fixture(scope="module")
def dia_system(tmp_path_factory):
    """A difference-imaging-style light curve: heavy negative blending and a
    baseline the errors straddle, so a large fraction of the off-peak epochs
    come out below zero.  This is the data the flux format exists for.

    Under the old magnitude likelihood every one of those points was clamped
    to 1e-30 and entered the fit as a fabricated ~75 mag measurement.

    The true (f_source, f_blend) are given as start values so the test is
    about the likelihood rather than about the NNLS bootstrap, which cannot
    return a negative f_blend.
    """
    rng = np.random.default_rng(20260811)
    t = np.linspace(T0 - 90.0, T0 + 90.0, 600)
    f_true = _pspl_flux(t, f_source=DIA_FSOURCE, f_blend=DIA_FBLEND)
    err = np.full(t.size, DIA_ERR)
    flux = f_true + rng.normal(0.0, err)
    rows = np.column_stack([t, flux, err])

    f_total = DIA_FSOURCE + DIA_FBLEND
    extra = {
        "mulensinstrument.LC.log_f_total": {"initval": np.log10(f_total)},
        "mulensinstrument.LC.q_source": {"initval": DIA_FSOURCE / f_total},
    }
    tmpdir = tmp_path_factory.mktemp("dia")
    system, model = _build(tmpdir, rows, "flux", extra_params=extra)
    # Compare against what is actually ON DISK, so the assertion is about the
    # loader and not about np.savetxt's decimal formatting.
    on_disk = np.loadtxt(os.path.join(str(tmpdir), "lc.dat"))
    return system, model, on_disk


def test_negative_fluxes_reach_the_model_unclamped(dia_system):
    """
    Given a difference-imaging light curve with genuinely negative fluxes,
    When it is loaded,
    Then the negative values arrive in the model bit-for-bit, with their own
    errors -- nothing is floored at 1e-30 (~75 mag) the way the magnitude
    branch had to.
    """
    system, _model, rows = dia_system
    inst = system.mulensinstrument

    n_negative = int(np.sum(rows[:, 1] < 0.0))
    assert n_negative > 50, "fixture is meant to contain negative fluxes"

    # rtol is the decimal round-trip of the file, not a modeling tolerance:
    # pandas' and numpy's text parsers can differ in the last bit.  The claim
    # under test is that no value is altered, in particular not floored.
    np.testing.assert_allclose(inst.flux, rows[:, 1], rtol=1e-13)
    np.testing.assert_allclose(inst.err, rows[:, 2], rtol=1e-13)
    assert inst.flux.min() < 0.0
    assert int(np.sum(inst.flux < 0.0)) == n_negative
    # The magnitude array is gone; there is no second likelihood branch.
    assert not hasattr(inst, "mag")


def test_negative_fluxes_give_finite_logp_and_gradient(dia_system):
    """
    Given the same light curve,
    When logp and dlogp are evaluated at the start point,
    Then both are finite -- the negative epochs contribute an ordinary
    Gaussian term instead of a fabricated 75 mag outlier.
    """
    _system, model, _rows = dia_system
    point = model.initial_point()

    logp = float(np.asarray(model.compile_logp()(point)))
    assert np.isfinite(logp)

    grads = model.compile_dlogp()(point)
    grads = np.concatenate(
        [np.atleast_1d(np.asarray(g)).ravel() for g in [grads]]
    )
    assert np.all(np.isfinite(grads))
    assert np.any(grads != 0.0)


def test_negative_flux_epochs_are_ordinary_chi2_terms(dia_system):
    """
    Given the same light curve,
    When the per-point chi2 of the negative-flux epochs is measured at the
    start point,
    Then it is of order one per point.

    This is the statistical content of the change: under the magnitude
    likelihood these same epochs sat at 75 mag against a model near the true
    (finite) magnitude, i.e. they were enormous, fabricated outliers.
    """
    _system, model, _rows = dia_system
    point = model.initial_point()
    fn, data = _observation(model)
    mu, sigma = _eval_mu_sigma(model, fn, point)

    neg = data < 0.0
    chi2_per_point = (
        np.sum(((data[neg] - mu[neg]) / sigma[neg]) ** 2) / neg.sum()
    )
    assert 0.2 < chi2_per_point < 3.0, chi2_per_point


# ---------------------------------------------------------------------------
# 2. data_format is only a statement about the file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def twin_systems(tmp_path_factory):
    """The same light curve written twice: once in magnitudes, once as the
    exact flux equivalent (F = 10**(-0.4 m), sigma_F = F * sigma_m / K)."""
    rng = np.random.default_rng(11223344)
    t = np.linspace(T0 - 90.0, T0 + 90.0, 500)
    # An ordinary, positive-flux magnitude survey light curve.
    f_true = _pspl_flux(t, f_source=8.0e-8, f_blend=2.0e-8)
    sigma_m = np.full(t.size, 0.01)
    flux = f_true * (1.0 + rng.normal(0.0, sigma_m / K))
    mag = -2.5 * np.log10(flux)
    err_f = flux * sigma_m / K

    mag_rows = np.column_stack([t, mag, sigma_m])
    flux_rows = np.column_stack([t, flux, err_f])

    mag_sys = _build(tmp_path_factory.mktemp("mag"), mag_rows, "magnitude")
    flux_sys = _build(tmp_path_factory.mktemp("flux"), flux_rows, "flux")
    return mag_sys, flux_sys


def test_magnitude_file_and_flux_twin_load_to_the_same_arrays(twin_systems):
    """
    Given a magnitude file and the exactly converted flux file,
    When both are loaded,
    Then the modeled observable and its errors agree to float round-off --
    data_format no longer selects a likelihood, only a file layout.
    """
    (mag_sys, _), (flux_sys, _) = twin_systems

    np.testing.assert_allclose(
        mag_sys.mulensinstrument.flux,
        flux_sys.mulensinstrument.flux,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        mag_sys.mulensinstrument.err,
        flux_sys.mulensinstrument.err,
        rtol=1e-12,
    )


def test_magnitude_file_and_flux_twin_give_the_same_logp(twin_systems):
    """
    Given the same two files,
    When each built model's logp is evaluated at its own start point,
    Then the values agree to round-off (the start values come from the same
    flux bootstrap, which never round-trips through magnitudes).
    """
    (_, mag_model), (_, flux_model) = twin_systems
    lp_mag = float(
        np.asarray(mag_model.compile_logp()(mag_model.initial_point()))
    )
    lp_flux = float(
        np.asarray(flux_model.compile_logp()(flux_model.initial_point()))
    )
    assert np.isfinite(lp_mag)
    assert lp_mag == pytest.approx(lp_flux, rel=1e-9, abs=1e-6)


def test_observation_node_is_a_normal_in_flux(twin_systems):
    """
    Given a magnitude-format light curve,
    When the built observation node is inspected,
    Then the observed values are the converted FLUXES and sigma is the flux
    error times err_scale -- there is no magnitude anywhere in the graph.
    """
    (mag_sys, mag_model), _ = twin_systems
    point = mag_model.initial_point()
    fn, data = _observation(mag_model)
    _mu, sigma = _eval_mu_sigma(mag_model, fn, point)

    inst = mag_sys.mulensinstrument
    np.testing.assert_allclose(data, inst.flux, rtol=1e-12)

    err_scale = _eval_node(mag_model, inst.err_scale.value, point)
    np.testing.assert_allclose(sigma, inst.err * err_scale[0], rtol=1e-10)


# ---------------------------------------------------------------------------
# 3. A magnitude file keeps the posterior it used to have, to O(sigma_mag)
# ---------------------------------------------------------------------------


def _profiles(f_true, sigma_rel, seed):
    """Flux-space and (old) magnitude-space log-likelihood profiles.

    ``f_true`` is a real model light curve from the built system.  ONE data
    set is drawn around it at fractional error ``sigma_rel`` (i.e. sigma_mag =
    K*sigma_rel), and both likelihoods are profiled over the same data along
    an overall flux normalization ``c``.  Sharing the data matters: the
    noise-driven scatter of the estimate is then common to the two profiles
    and cancels in their difference, leaving only the systematic term.

    The scan spans +/-4 posterior sigmas (sigma_c = sigma_rel/sqrt(N)), so the
    profiles are compared over the range the posterior actually occupies
    rather than over an arbitrary fixed interval.

    Returns (c_grid, sigma_c, ll_flux, ll_mag), each profile relative to its
    own maximum.
    """
    rng = np.random.default_rng(seed)
    n = f_true.size
    sig_f = sigma_rel * f_true
    f_obs = f_true + rng.normal(0.0, sig_f)
    m_obs = -2.5 * np.log10(f_obs)
    # Exactly the conversion the old load_data did on a magnitude file, run
    # backwards: the magnitude branch's sigma for these same data.
    sig_m = K * sig_f / f_obs

    sigma_c = sigma_rel / np.sqrt(n)
    c = 1.0 + np.linspace(-4.0, 4.0, 401) * sigma_c
    ll_f = np.empty_like(c)
    ll_m = np.empty_like(c)
    for j, cj in enumerate(c):
        mu = cj * f_true
        ll_f[j] = np.sum(-0.5 * ((f_obs - mu) / sig_f) ** 2)
        ll_m[j] = np.sum(-0.5 * ((m_obs + 2.5 * np.log10(mu)) / sig_m) ** 2)
    return c, sigma_c, ll_f - ll_f.max(), ll_m - ll_m.max()


def _vertex(c, ll):
    """Location of a (near-parabolic) profile's maximum, by quadratic fit."""
    a, b, _ = np.polyfit(c, ll, 2)
    return -b / (2.0 * a)


def _map_offset_in_sigmas(f_true, sigma_rel, seed=987654):
    """|MAP_flux - MAP_mag| in units of the posterior sigma, and the two
    profile curvatures."""
    c, sigma_c, ll_f, ll_m = _profiles(f_true, sigma_rel, seed)
    curv_f = np.polyfit(c, ll_f, 2)[0]
    curv_m = np.polyfit(c, ll_m, 2)[0]
    shift = abs(_vertex(c, ll_f) - _vertex(c, ll_m)) / sigma_c
    return shift, curv_f, curv_m


@pytest.fixture(scope="module")
def model_lightcurve(twin_systems):
    """The fitted model flux of the magnitude-format system at its start
    point -- a real curve out of the real graph, used as the truth for the
    statistical comparison below."""
    (_, mag_model), _ = twin_systems
    fn, _data = _observation(mag_model)
    mu, _sigma = _eval_mu_sigma(mag_model, fn, mag_model.initial_point())
    assert np.all(mu > 0.0)
    return mu


@pytest.mark.parametrize(
    "sigma_rel, tol_sigmas, tol_width",
    [
        # sigma_mag = 0.011 -> the two MAPs sit 1.5*sigma_rel*sqrt(N) = 0.34
        # posterior sigma apart, and the posterior WIDTHS agree to 2%.
        (0.010, 0.5, 0.02),
        # 10x better photometry -> 10x closer MAPs, and widths to 0.2%.
        (0.001, 0.05, 0.002),
    ],
)
def test_magnitude_posterior_matches_flux_to_first_order(
    model_lightcurve, sigma_rel, tol_sigmas, tol_width
):
    """
    Given a real model light curve and one data set at fractional error
    sigma_rel,
    When the flux-space and (old) magnitude-space log-likelihoods are profiled
    over the model's overall flux normalization,
    Then their maxima agree to well within a posterior sigma and their widths
    agree to a fraction of a percent -- and both tolerances scale with
    sigma_rel, which is the precision claim the conversion makes.

    Where the tolerances come from, rather than from tuning.  A magnitude
    residual x carries a flux residual (1 + x/(2K)) times as large, so
    chi2_flux = chi2_mag + sum_i x_i^3/(K*sigma_m^2).  Differentiating the
    extra term shifts the estimate by 1.5*sigma_m^2/K in magnitude, i.e. by
    1.5*sigma_rel^2 in the flux normalization c, against a posterior width
    sigma_c = sigma_rel/sqrt(N).  The offset in posterior sigmas is therefore
    1.5*sigma_rel*sqrt(N) -- 0.34 here at sigma_rel = 0.01 with N = 500, and
    0.034 at 0.001.  Note the direction of the correctness argument: it is the
    magnitude fit that carried this bias (its noise is not Gaussian in
    magnitudes), and the flux fit that does not.

    The two profiles share one data set, so the noise-driven scatter of the
    estimate is common to both and cancels; what is left is exactly the
    systematic term above.
    """
    shift, curv_f, curv_m = _map_offset_in_sigmas(model_lightcurve, sigma_rel)

    assert shift < tol_sigmas, f"MAPs differ by {shift:.3f} posterior sigma"
    assert abs(curv_f / curv_m - 1.0) < tol_width, (curv_f, curv_m)


def test_first_order_discrepancy_shrinks_with_the_photometric_error(
    model_lightcurve,
):
    """
    Given the same comparison at two photometric precisions a factor 10 apart,
    When the MAP offsets are compared in posterior sigmas,
    Then the better photometry agrees at least 5x better.

    A genuinely different statistical model would not converge like this; a
    first-order approximation does.  This is what makes "the magnitude branch
    was an approximation to this one" a measurement rather than an assertion.
    """
    shifts = [
        _map_offset_in_sigmas(model_lightcurve, s)[0] for s in (0.010, 0.001)
    ]
    assert shifts[0] / shifts[1] > 5.0, shifts


# ---------------------------------------------------------------------------
# 4. Optional-feature amplitudes follow the light curve's flux scale
# ---------------------------------------------------------------------------


def test_gp_and_outlier_amplitudes_are_capped_on_the_flux_scale():
    """
    Given two light curves whose flux zeropoints differ by ten orders of
    magnitude (a magnitude file vs difference-imaging counts),
    When _scale_flux_amplitudes rewrites the manifest,
    Then each element's GP amplitude and outlier-scale bounds and starts are
    proportional to that light curve's own baseline flux.

    Fixed numbers cannot do this job once the observable is flux in the file's
    arbitrary system -- which is exactly why the old magnitude caps (5 mag,
    10 mag) could be constants and these cannot.
    """
    from exozippy.components.mulensing.mulensinstrument import MulensInstrument

    inst = MulensInstrument.__new__(MulensInstrument)
    f_total = np.array([6.3e-8, 1.2e4])
    manifest = {
        "gp_rot_sigma": {"overrides": {"sigma": [0.0, np.nan]}},
        "out_scale": {},
    }
    inst._scale_flux_amplitudes(manifest, f_total)

    for name, (cap, start) in MulensInstrument._FLUX_AMPLITUDE_CAPS.items():
        if name not in manifest:
            continue
        ov = manifest[name]["overrides"]
        np.testing.assert_allclose(ov["upper"], cap * f_total, rtol=1e-12)
        np.testing.assert_allclose(ov["initval"], start * f_total, rtol=1e-12)

    # The pin installed by _register_gp for the opted-out element survives.
    assert manifest["gp_rot_sigma"]["overrides"]["sigma"][0] == 0.0


@pytest.fixture(scope="module")
def gp_system(tmp_path_factory):
    """A magnitude-scale light curve (fluxes ~1e-7) that requests an SHO GP."""
    rng = np.random.default_rng(4242)
    t = np.linspace(T0 - 90.0, T0 + 90.0, 600)
    f_true = _pspl_flux(t, f_source=8.0e-8, f_blend=2.0e-8)
    err = np.full(t.size, 1.0e-9)
    rows = np.column_stack([t, f_true + rng.normal(0.0, err), err])
    return _build(
        tmp_path_factory.mktemp("gp"), rows, "flux", inst_extra={"gp": "sho"}
    )


def test_gp_amplitude_bound_follows_the_light_curve_flux_scale(gp_system):
    """
    Given a light curve whose fluxes are ~1e-7 and which requested an SHO GP,
    When the model is built,
    Then the GP amplitude's upper bound is 100x that light curve's own
    baseline flux and its start is the median error bar (the data-driven hint,
    which outranks the per-element default) -- not the fixed magnitude-era
    numbers, which would have put the entire prior 8 orders of magnitude above
    any physical amplitude.
    """
    system, _model = gp_system
    inst = system.mulensinstrument
    f_total = np.asarray(inst.fs_init, dtype=float)

    np.testing.assert_allclose(
        inst.gp_sho_sigma.upper, 1.0e2 * f_total, rtol=1e-10
    )
    np.testing.assert_allclose(inst.gp_sho_sigma.lower, [0.0])
    np.testing.assert_allclose(
        inst.gp_sho_sigma.initval, [np.median(inst.err)], rtol=1e-10
    )


def test_gp_plot_curve_is_built_in_flux(gp_system):
    """
    Given the same GP light curve and a fitted point,
    When plot_data builds the "model + GP" curve,
    Then it exists and is finite: the GP conditional mean is added to the
    model FLUX in the instrument's own system (where celerite2 conditioned it)
    and only then mapped onto the reference system.
    """
    system, model = gp_system
    point = model.initial_point()
    system.compile_plotter_functions(model)

    spec = system.mulensinstrument.plot_data(system, point)[0]
    gp_traces = [tr for tr in spec.traces if "GP" in tr.name]
    assert gp_traces, [tr.name for tr in spec.traces]
    for tr in gp_traces:
        assert np.all(np.isfinite(tr.y))


# ---------------------------------------------------------------------------
# 5. The plot path drops, rather than fabricates, non-positive fluxes
# ---------------------------------------------------------------------------


def test_aligned_plot_specs_are_finite_and_drop_non_positive_fluxes(
    dia_system,
):
    """
    Given the difference-imaging light curve and a fitted point,
    When plot_data builds the aligned lightcurve specs,
    Then the model traces are finite everywhere, the positive-flux epochs get
    their delta-magnitudes, and the non-positive ones come back NaN (with NaN
    error bars) instead of being dragged to a ~75 mag spike.

    With one instrument the aligner maps the reference system onto itself, so
    the data's delta-magnitude must be exactly -2.5*log10(F) - baseline: that
    pins the affine flux alignment, not just its finiteness.
    """
    system, model, rows = dia_system
    point = model.initial_point()
    system.compile_plotter_functions(model)

    specs = system.mulensinstrument.plot_data(system, point)
    assert specs, "expected at least the lightcurve spec"
    spec = specs[0]

    models = [tr for tr in spec.traces if tr.role == "model"]
    assert models
    for tr in models:
        assert np.all(np.isfinite(tr.y)), tr.name

    data = [tr for tr in spec.traces if tr.role == "data"]
    assert len(data) == 1
    y = np.asarray(data[0].y, dtype=float)

    pos = rows[:, 1] > 0.0
    assert np.all(np.isnan(y[~pos]))
    assert np.all(np.isfinite(y[pos]))

    fs, fb = system.mulensinstrument._compiled_flux(
        *system.mulensinstrument._point_to_plot_params(point, system)
    )
    baseline = -2.5 * np.log10(
        float(np.atleast_1d(fs)[0] + np.atleast_1d(fb)[0])
    )
    np.testing.assert_allclose(
        y[pos], -2.5 * np.log10(rows[pos, 1]) - baseline, rtol=1e-10
    )

    yerr = np.asarray(data[0].yerr, dtype=float)
    assert yerr.shape == (2, rows.shape[0])
    assert np.all(np.isnan(yerr[:, ~pos]))


def test_plot_specs_drop_non_positive_fluxes_instead_of_clamping(dia_system):
    """
    Given the difference-imaging light curve,
    When the data-only charts are built,
    Then the negative-flux epochs come back as NaN (not drawn) rather than as
    the ~75 mag spikes the old 1e-30 clamp produced, and the positive ones
    keep their magnitudes.
    """
    system, _model, rows = dia_system
    specs = system.mulensinstrument.plot_data(system, point=None)
    y = specs[0].traces[0].y

    neg = rows[:, 1] <= 0.0
    assert np.all(np.isnan(y[neg]))
    np.testing.assert_allclose(
        y[~neg], -2.5 * np.log10(rows[~neg, 1]), rtol=1e-12
    )
    assert np.nanmax(y) < 40.0

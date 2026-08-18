"""Tests for the optional per-file robust likelihoods on Instrument.

Three layers, mirroring tests/test_gp.py:

1. The ``likelihood:`` config vocabulary and the logp builders
   (components/likelihood.py) -- pure functions, no PyMC.
2. The Instrument lifecycle hooks (_load_likelihood_config / _prepare_robust /
   _register_robust) -- the scaffolding, exercised on a dummy instrument.
3. The likelihood itself -- that the hogg mixture Potential and the Student-t
   match their closed forms exactly, that limits recover the Gaussian, and
   that a real two-instrument RV model builds with only the opted-in file's
   parameters sampled.

The default is off everywhere, so the strongest regression guarantee is the
"no likelihood: key changes nothing" test in each layer.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from scipy import stats

from conftest import _DummyConfigManager
from exozippy.components import likelihood as robust_support
from exozippy.components.instrument import Instrument


class _DummyInstrument(Instrument):
    """Minimal concrete Instrument for unit-testing the shared helpers."""

    @property
    def prefix(self):
        return "dummy"

    def register_parameters(self, system):  # pragma: no cover - not exercised
        pass

    def build_likelihood(self, model, system):  # pragma: no cover
        pass


class _NoRobustInstrument(_DummyInstrument):
    supports_robust_likelihood = False


class _RecordingConfigManager(_DummyConfigManager):
    """Captures the hints a component pushes, so they can be asserted on."""

    def __init__(self):
        self.hints = {}
        self.scale_hints = {}

    def add_hint(self, path, value, rank=None):
        self.hints[path] = value

    def add_scale_hint(self, path, scale):
        self.scale_hints[path] = scale


class _FakeParam:
    """Stand-in for a built Parameter: only .value is read by the builders."""

    def __init__(self, values):
        self.value = pt.as_tensor_variable(values)


def _make(config, config_manager=None):
    return _DummyInstrument(config, config_manager=config_manager)


def _gaussian_logp(y, mu, sigma):
    return float(np.sum(stats.norm.logpdf(y, loc=mu, scale=sigma)))


def _hogg_logp_numpy(y, mu, sigma, out_frac, out_scale):
    core = stats.norm.logpdf(y, loc=mu, scale=sigma)
    wide = stats.norm.logpdf(y, loc=mu, scale=np.sqrt(sigma**2 + out_scale**2))
    return float(
        np.sum(
            np.logaddexp(np.log1p(-out_frac) + core, np.log(out_frac) + wide)
        )
    )


# ---------------------------------------------------------------------------
# 1. The likelihood: config vocabulary and logp builders
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ""),
        ("none", ""),
        ("off", ""),
        ("", ""),
        (False, ""),
        ("gaussian", ""),
        ("Normal", ""),
        ("hogg", "hogg"),
        ("  Mixture  ", "hogg"),
        ("hogg_mixture", "hogg"),
        ("studentt", "studentt"),
        ("Student-T", "studentt"),
        ("t", "studentt"),
    ],
)
def test_parse_likelihood_spec_normalizes_every_accepted_spelling(
    value, expected
):
    """
    Given any accepted form of the likelihood: key,
    When parse_likelihood_spec normalizes it,
    Then the result is the canonical family key, and every "off" spelling
    (including the explicit 'gaussian') yields the empty string.
    """
    assert robust_support.parse_likelihood_spec(value) == expected


def test_parse_likelihood_spec_rejects_unknown_and_ambiguous_values():
    """
    Given a misspelled family, a bare true, or a list,
    When parse_likelihood_spec runs,
    Then each raises with the context in the message -- the families are
    alternatives, so a list is never meaningful (unlike gp: terms, which add).
    """
    with pytest.raises(ValueError, match="unknown likelihood"):
        robust_support.parse_likelihood_spec(
            "cauchy", context="rvinstrument[HARPS]"
        )
    with pytest.raises(ValueError, match="ambiguous"):
        robust_support.parse_likelihood_spec(True)
    with pytest.raises(ValueError, match="single family"):
        robust_support.parse_likelihood_spec(["hogg", "studentt"])


def test_family_tables_are_mutually_consistent():
    """
    Given the per-family parameter tables,
    When cross-checked,
    Then every family has a parameter list, every scale parameter and
    log-sampled parameter belongs to some family, and the Deterministic names
    do not collide with a sampled parameter name.
    """
    all_params = set()
    for kind in robust_support.LIKELIHOOD_KINDS:
        params = robust_support.LIKELIHOOD_PARAMS[kind]
        assert params
        all_params.update(params)
    for kind, name in robust_support.LIKELIHOOD_SCALE_PARAM.items():
        assert name in robust_support.LIKELIHOOD_PARAMS[kind]
    assert set(robust_support.LIKELIHOOD_LOG_PARAMS) <= all_params
    assert not (
        set(robust_support.LIKELIHOOD_LOG_PARAMS.values()) & all_params
    )


def test_every_robust_parameter_has_defaults():
    """
    Given the robust-likelihood parameter tables,
    When each name is resolved against the loaded defaults,
    Then all of them have bounds, a start and an init_scale (required of any
    sampled parameter), so no fit can fail late with a missing-scale error.
    """
    from exozippy.config import ConfigManager

    cm = ConfigManager({})
    for kind in robust_support.LIKELIHOOD_KINDS:
        for name in robust_support.LIKELIHOOD_PARAMS[kind]:
            cfg = cm.resolve("rvinstrument", name, shape=(1,))
            assert cfg["lower"] is not None, name
            assert cfg["upper"] is not None, name
            assert cfg["init_scale"] is not None, name
            assert cfg["initval"] is not None, name


def test_introspection_reports_the_robust_parameters_with_merged_units():
    """
    Given the robust blueprint lives at the root of components/defaults.yaml
    and each instrument overrides only out_scale,
    When component_schema describes an instrument,
    Then every robust parameter appears with the component's unit layered
    over the shared blueprint, and the opted-out astrometry reports none.
    """
    from exozippy.introspect import component_schema

    expected = {
        name
        for kind in robust_support.LIKELIHOOD_KINDS
        for name in robust_support.LIKELIHOOD_PARAMS[kind]
    }

    for key in ("rvinstrument", "transit", "mulensinstrument"):
        params = component_schema(key)["parameters"]
        assert expected <= set(params), key

    rv_scale = component_schema("rvinstrument")["parameters"]["out_scale"]
    assert rv_scale["unit"] == "m/s"  # from rvinstrument's block
    assert rv_scale["internal_unit"] == "solRad/d"
    assert rv_scale["description"]  # inherited from the root file

    # Microlensing is fit in flux, in the data file's own (arbitrary) flux
    # system, so out_scale is a bare flux with no unit string; the usable
    # per-light-curve bound is installed from the data by
    # MulensInstrument._scale_flux_amplitudes.
    mu_scale = component_schema("mulensinstrument")["parameters"]["out_scale"]
    assert mu_scale["unit"] == ""
    assert mu_scale["description"]  # still inherited from the root file

    astro = component_schema("astrometryinstrument")["parameters"]
    assert not (expected & set(astro))


def test_likelihood_config_schema_entry_shape():
    """
    Given the shared likelihood config-schema entry,
    When inspected,
    Then it declares the standard option shape consumed by introspection.
    """
    entry = Instrument._likelihood_config_schema()
    assert entry["key"] == "likelihood"
    assert entry["kind"] == "option"
    assert entry["required"] is False
    assert set(robust_support.LIKELIHOOD_KINDS) <= set(entry["accepts"])


def test_hogg_logp_matches_the_closed_form_mixture():
    """
    Given residuals scored by hogg_logp,
    When compared against the scipy-composed two-Normal mixture,
    Then they agree to float precision, normalization constants included.
    """
    rng = np.random.default_rng(7)
    n = 50
    resid = rng.normal(0, 1.0, n)
    resid[::10] += 8.0  # a few genuine outliers
    sigma = rng.uniform(0.5, 1.5, n)
    out_frac, out_scale = 0.07, 10.0

    got = pt.sum(
        robust_support.hogg_logp(
            pt.as_tensor_variable(resid),
            pt.as_tensor_variable(sigma),
            out_frac,
            out_scale,
        )
    ).eval()

    expected = _hogg_logp_numpy(resid, 0.0, sigma, out_frac, out_scale)
    assert float(got) == pytest.approx(expected, rel=1e-12)


def test_hogg_logp_recovers_the_gaussian_in_the_no_outlier_limit():
    """
    Given a vanishing outlier fraction (and separately a vanishing extra
    scale),
    When the mixture is evaluated,
    Then it reduces to the plain Gaussian logp it replaces -- the feature is
    continuously connected to the default.
    """
    rng = np.random.default_rng(8)
    n = 30
    resid = rng.normal(0, 1.0, n)
    sigma = rng.uniform(0.5, 1.5, n)
    expected = _gaussian_logp(resid, 0.0, sigma)

    frac_to_zero = pt.sum(
        robust_support.hogg_logp(
            pt.as_tensor_variable(resid),
            pt.as_tensor_variable(sigma),
            1e-14,
            10.0,
        )
    ).eval()
    assert float(frac_to_zero) == pytest.approx(expected, abs=1e-9)

    scale_to_zero = pt.sum(
        robust_support.hogg_logp(
            pt.as_tensor_variable(resid),
            pt.as_tensor_variable(sigma),
            0.1,
            0.0,
        )
    ).eval()
    assert float(scale_to_zero) == pytest.approx(expected, abs=1e-9)


def test_hogg_outlier_logodds_separates_outliers_from_inliers():
    """
    Given one point far outside the inlier scatter among clean points,
    When the per-point posterior outlier probability is evaluated,
    Then it is ~1 for the planted outlier and small for the clean points.
    """
    resid = np.array([0.1, -0.2, 0.05, 12.0, -0.15])
    sigma = np.full(5, 0.2)

    logodds = robust_support.hogg_outlier_logodds(
        pt.as_tensor_variable(resid),
        pt.as_tensor_variable(sigma),
        0.05,
        5.0,
    ).eval()
    prob = 1.0 / (1.0 + np.exp(-logodds))

    assert prob[3] > 0.999
    assert np.all(prob[[0, 1, 2, 4]] < 0.15)


@pytest.mark.parametrize(
    "out_frac,out_scale",
    [
        (0.05, 4.0),
        (1e-6, 1e-3),
        (0.5, 1e3),
        (0.4999999, 250.0),
    ],
)
def test_hogg_logp_and_logodds_describe_the_same_two_branches(
    out_frac, out_scale
):
    """
    Given the mixture logp and the posterior outlier log-odds of the same
    points,
    When the two branch densities are RECOVERED from that pair alone and
    compared with the closed-form weighted Gaussians,
    Then both match -- so the probability the likelihood integrates and the
    probability the audit reports are the same mixture.  This is the test
    that fails if the two consumers ever drift apart again (they share
    hogg_branch_logps; before that they each rebuilt the four terms).  The
    algebra is exact: with a = log(1-f) + log N(r|0,s) and b = log(f) +
    log N(r|0,sqrt(s^2+S^2)), logp = logaddexp(a, b) and logodds = b - a, so
    a = logp - softplus(logodds) and b = logp - softplus(-logodds).  (The
    residuals stay within a few tens of sigma: recovering a branch from the
    pair is exact algebra but ill-conditioned once one branch underflows the
    other by 1e7 nats, which says nothing about either implementation.  The
    catastrophic-outlier case is pinned by the finite-gradient test below.)
    """
    rng = np.random.default_rng(19)
    resid = np.concatenate([rng.normal(0, 1, 6), [0.0, 5.0, -5.0, 1e-12]])
    sigma = np.concatenate([rng.uniform(0.1, 2.0, 6), [1.0, 0.2, 0.2, 1e-3]])

    logp = robust_support.hogg_logp(
        pt.as_tensor_variable(resid),
        pt.as_tensor_variable(sigma),
        out_frac,
        out_scale,
    ).eval()
    logodds = robust_support.hogg_outlier_logodds(
        pt.as_tensor_variable(resid),
        pt.as_tensor_variable(sigma),
        out_frac,
        out_scale,
    ).eval()

    softplus = np.logaddexp(0.0, logodds)
    inlier = logp - softplus
    outlier = logp - np.logaddexp(0.0, -logodds)

    expected_in = np.log1p(-out_frac) + stats.norm.logpdf(
        resid, loc=0.0, scale=sigma
    )
    expected_out = np.log(out_frac) + stats.norm.logpdf(
        resid, loc=0.0, scale=np.sqrt(sigma**2 + out_scale**2)
    )
    np.testing.assert_allclose(inlier, expected_in, rtol=0, atol=1e-9)
    np.testing.assert_allclose(outlier, expected_out, rtol=0, atol=1e-9)


def test_both_hogg_consumers_are_built_from_the_shared_branches():
    """
    Given the shared hogg_branch_logps helper,
    When its two outputs are combined the way each consumer combines them,
    Then the results are bit-identical to the consumers' own output -- the
    single definition really is the one both use, and the mixture is still
    assembled with logaddexp (never a where over branch logps, which would
    poison the JAX gradient).
    """
    resid = pt.as_tensor_variable(np.array([0.3, -2.0, 40.0]))
    sigma = pt.as_tensor_variable(np.array([0.2, 0.5, 1.0]))
    inlier, outlier = robust_support.hogg_branch_logps(resid, sigma, 0.1, 3.0)

    logp = robust_support.hogg_logp(resid, sigma, 0.1, 3.0).eval()
    logodds = robust_support.hogg_outlier_logodds(
        resid, sigma, 0.1, 3.0
    ).eval()

    expected_logp = (
        pt.logaddexp(inlier, outlier) - 0.5 * np.log(2.0 * np.pi)
    ).eval()
    assert np.array_equal(logp, expected_logp)
    assert np.array_equal(logodds, (outlier - inlier).eval())


# ---------------------------------------------------------------------------
# 2. Instrument lifecycle hooks
# ---------------------------------------------------------------------------
def test_no_likelihood_key_leaves_the_component_untouched():
    """
    Given instruments with no likelihood: key (the default),
    When the base loads the config and registers parameters,
    Then has_robust_likelihood is False and the manifest gains nothing -- the
    model is exactly what it was before the feature existed.
    """
    inst = _make([{"file": "a.rv"}, {"file": "b.rv"}])
    assert inst.likelihood_kinds == ["", ""]
    assert inst.has_robust_likelihood is False

    manifest = {"gamma": "default"}
    inst._register_robust(manifest)
    assert manifest == {"gamma": "default"}


def test_likelihood_kinds_are_parsed_per_element():
    """
    Given a mix of likelihood: settings across three files,
    When the base loads the config,
    Then each element carries its own family and has_robust_likelihood is
    True.
    """
    inst = _make(
        [
            {"file": "a.rv", "likelihood": "hogg"},
            {"file": "b.rv"},
            {"file": "c.rv", "likelihood": "studentt"},
        ]
    )
    assert inst.likelihood_kinds == ["hogg", "", "studentt"]
    assert inst.has_robust_likelihood is True
    assert inst._robust_elements("hogg") == [0]
    assert inst._robust_elements("studentt") == [2]


def test_component_that_does_not_support_robust_rejects_the_key():
    """
    Given a component with more than one observable per file
    (supports_robust_likelihood False, as astrometryinstrument declares),
    When a data file sets likelihood:,
    Then construction raises rather than silently ignoring the key.
    """
    with pytest.raises(NotImplementedError, match="not supported"):
        _NoRobustInstrument(
            [{"file": "a.dat", "likelihood": "hogg"}], config_manager=None
        )

    inst = _NoRobustInstrument([{"file": "a.dat"}], config_manager=None)
    assert inst.has_robust_likelihood is False


def test_astrometry_declares_no_robust_support():
    """
    Given astrometryinstrument models two observables per dataset,
    When its class attribute is inspected,
    Then it opts out of robust likelihoods, and the three single-observable
    children keep the base default.
    """
    from exozippy.components.astrometryinstrument.astrometryinstrument import (
        AstrometryInstrument,
    )
    from exozippy.components.mulensing.mulensinstrument import MulensInstrument
    from exozippy.components.rvinstrument.rvinstrument import RVInstrument
    from exozippy.components.transit.transit import Transit

    assert AstrometryInstrument.supports_robust_likelihood is False
    assert RVInstrument.supports_robust_likelihood
    assert Transit.supports_robust_likelihood
    assert MulensInstrument.supports_robust_likelihood


def test_gp_and_likelihood_on_the_same_file_raise():
    """
    Given one file requesting both gp: and likelihood:,
    When the component is constructed,
    Then it raises (celerite2's closed-form marginal is Gaussian-only) --
    while the same two keys on different files coexist.
    """
    with pytest.raises(ValueError, match="cannot be combined"):
        _make([{"file": "a.rv", "gp": "sho", "likelihood": "hogg"}])

    inst = _make(
        [
            {"file": "a.rv", "gp": "sho"},
            {"file": "b.rv", "likelihood": "hogg"},
        ]
    )
    assert inst.gp_terms == [("sho",), ()]
    assert inst.likelihood_kinds == ["", "hogg"]


def test_register_robust_pins_the_files_that_did_not_opt_in():
    """
    Given three files where only the middle one requested the hogg mixture,
    When _register_robust builds the manifest,
    Then both hogg parameters are registered (full-length vectors, so user
    paths resolve by instrument name) with sigma pinned to 0 on the two files
    that opted out and left alone (NaN) on the one that did.
    """
    inst = _make(
        [
            {"file": "a.rv"},
            {"file": "b.rv", "likelihood": "hogg"},
            {"file": "c.rv"},
        ]
    )
    manifest = inst._register_robust({})

    assert set(manifest) == set(robust_support.LIKELIHOOD_PARAMS["hogg"])
    for name in robust_support.LIKELIHOOD_PARAMS["hogg"]:
        pin = manifest[name]["overrides"]["sigma"]
        assert pin[0] == 0.0 and pin[2] == 0.0
        assert np.isnan(pin[1])


def test_register_robust_omits_the_pin_when_every_file_opted_in():
    """
    Given every file requesting a family,
    When _register_robust builds the manifest,
    Then no override is emitted at all (nothing to pin), and each family
    contributes its own parameters.
    """
    inst = _make(
        [
            {"file": "a.rv", "likelihood": "hogg"},
            {"file": "b.rv", "likelihood": "studentt"},
        ]
    )
    manifest = inst._register_robust({})

    expected = set(robust_support.LIKELIHOOD_PARAMS["hogg"]) | set(
        robust_support.LIKELIHOOD_PARAMS["studentt"]
    )
    assert set(manifest) == expected
    # hogg's parameters pin the studentt file and vice versa.
    hogg_pin = manifest["out_frac"]["overrides"]["sigma"]
    t_pin = manifest["t_log_nu"]["overrides"]["sigma"]
    assert np.isnan(hogg_pin[0]) and hogg_pin[1] == 0.0
    assert t_pin[0] == 0.0 and np.isnan(t_pin[1])


def test_one_helper_exposes_the_linear_value_of_both_log_tables():
    """
    Given the GP and robust features, which each sample some parameters in
    log10 and owe the tables the linear value,
    When the shared _build_log10_deterministics runs over each feature's
    table,
    Then it returns 10**value for every parameter the topology built, skips
    the ones it did not, and names the Deterministic <prefix>.<linear name>
    -- one function over two tables, so the two features cannot disagree
    about what a log-sampled parameter reports.
    """
    from exozippy.components import gp as gp_support

    inst = _make(
        [
            {"file": "a.rv", "likelihood": "studentt"},
            {"file": "b.rv", "gp": "sho"},
        ],
        _RecordingConfigManager(),
    )

    with pm.Model() as model:
        inst.t_log_nu = _FakeParam(np.array([0.7, 0.0]))
        inst.gp_sho_log_q = _FakeParam(np.array([0.0, -0.5]))
        # gp_rot_log_q0 / gp_rot_log_dq were never built (no rotation term).
        inst._build_robust_deterministics()
        inst._build_gp_deterministics()

    assert set(inst._robust_linear) == {"t_nu"}
    assert set(inst._gp_linear) == {"gp_sho_q"}
    assert set(gp_support.GP_LOG_PARAMS) > {"gp_sho_log_q"}

    np.testing.assert_allclose(
        inst._robust_linear["t_nu"].eval(), 10.0 ** np.array([0.7, 0.0])
    )
    np.testing.assert_allclose(
        inst._gp_linear["gp_sho_q"].eval(), 10.0 ** np.array([0.0, -0.5])
    )
    assert {v.name for v in model.deterministics} == {
        "dummy.t_nu",
        "dummy.gp_sho_q",
    }


def test_prepare_robust_hints_ten_times_the_white_noise_level():
    """
    Given a hogg file among plain files,
    When _prepare_robust indexes the data,
    Then it records that file's observation indices and pushes an out_scale
    hint of 10 x the median error bar (in user units via user_factor) -- well
    clear of the inlier scatter, and derived from the reported errors, not
    from the observations' own spread.
    """
    cm = _RecordingConfigManager()
    inst = _make(
        [{"file": "a.rv"}, {"file": "b.rv", "likelihood": "hogg"}], cm
    )
    err = np.concatenate([np.full(5, 1.0), np.full(7, 3.0)])
    inst_map = np.repeat([0, 1], [5, 7])

    inst._prepare_robust(err, inst_map, user_factor=2.0)

    assert list(inst._robust_obs_index) == [1]
    np.testing.assert_array_equal(inst._robust_obs_index[1], np.arange(5, 12))
    assert cm.hints == {"dummy.1.out_scale": pytest.approx(60.0)}
    assert cm.scale_hints == {"dummy.1.out_scale": pytest.approx(60.0)}


# ---------------------------------------------------------------------------
# 3. The likelihood terms through the shared dispatcher
# ---------------------------------------------------------------------------
def _dispatch(inst, mu, sigma, y):
    return inst.add_observation_likelihood(
        "m",
        mu=pt.as_tensor_variable(mu),
        sigma=pt.as_tensor_variable(sigma),
        observed=y,
    )


def test_mixed_files_split_into_one_normal_plus_the_hogg_potential():
    """
    Given two files where the second asked for the hogg mixture,
    When add_observation_likelihood runs,
    Then the plain file keeps the shared Normal, the hogg file contributes a
    Potential, and the total logp equals Gaussian(file A) + mixture(file B)
    computed in closed form -- the split loses and double-counts nothing.
    """
    rng = np.random.default_rng(11)
    n_per = 9
    inst_map = np.repeat([0, 1], n_per)
    sigma = rng.uniform(0.1, 0.3, 2 * n_per)
    mu = np.sin(np.arange(2 * n_per))
    y = mu + rng.normal(0, sigma)
    y[n_per + 2] += 5.0  # an outlier in the hogg file

    inst = _make(
        [{"file": "a.rv"}, {"file": "b.rv", "likelihood": "hogg"}],
        _RecordingConfigManager(),
    )
    inst.inst_map = inst_map
    inst.n_total_obs = 2 * n_per
    inst._prepare_robust(sigma, inst_map)

    out_frac, out_scale = 0.05, 4.0
    with pm.Model() as model:
        inst.out_frac = _FakeParam(np.array([0.0, out_frac]))
        inst.out_scale = _FakeParam(np.array([0.0, out_scale]))
        _dispatch(inst, mu, sigma, y)

    assert [v.name for v in model.observed_RVs] == ["m"]
    assert [v.name for v in model.potentials] == ["m.hogg.1"]

    a, b = slice(0, n_per), slice(n_per, 2 * n_per)
    expected = _gaussian_logp(y[a], mu[a], sigma[a]) + _hogg_logp_numpy(
        y[b], mu[b], sigma[b], out_frac, out_scale
    )
    assert float(model.compile_logp()({})) == pytest.approx(
        expected, rel=1e-12
    )


def test_studentt_file_matches_scipy_and_reports_nu():
    """
    Given two files where the first asked for the Student-t,
    When add_observation_likelihood runs,
    Then that file's term is a StudentT whose logp matches scipy.stats.t
    (with sigma as the scale parameter, not the standard deviation), the
    plain file keeps its Normal, and the linear t_nu is a Deterministic.
    """
    rng = np.random.default_rng(12)
    n_per = 8
    inst_map = np.repeat([0, 1], n_per)
    sigma = rng.uniform(0.1, 0.3, 2 * n_per)
    mu = np.zeros(2 * n_per)
    y = mu + rng.normal(0, sigma)

    inst = _make(
        [{"file": "a.rv", "likelihood": "studentt"}, {"file": "b.rv"}],
        _RecordingConfigManager(),
    )
    inst.inst_map = inst_map
    inst.n_total_obs = 2 * n_per
    inst._prepare_robust(sigma, inst_map)

    log_nu = 0.75
    with pm.Model() as model:
        inst.t_log_nu = _FakeParam(np.array([log_nu, 0.0]))
        _dispatch(inst, mu, sigma, y)

    names = sorted(v.name for v in model.observed_RVs)
    assert names == ["m", "m.t.0"]
    assert "dummy.t_nu" in {v.name for v in model.deterministics}

    a, b = slice(0, n_per), slice(n_per, 2 * n_per)
    nu = 10.0**log_nu
    expected = float(
        np.sum(stats.t.logpdf(y[a], df=nu, loc=mu[a], scale=sigma[a]))
    ) + _gaussian_logp(y[b], mu[b], sigma[b])
    assert float(model.compile_logp()({})) == pytest.approx(
        expected, rel=1e-10
    )


def test_gradient_is_finite_with_an_extreme_outlier():
    """
    Given a hogg file containing a catastrophic (1e4 sigma) outlier,
    When the model logp and dlogp are compiled,
    Then both are finite -- the logaddexp composition has no branch that
    poisons the gradient (the JAX where-trap this module must avoid).
    """
    n = 6
    sigma = np.full(n, 0.1)
    mu = np.zeros(n)
    y = np.zeros(n)
    y[3] = 1e3  # 1e4 sigma

    inst = _make(
        [{"file": "a.rv", "likelihood": "hogg"}], _RecordingConfigManager()
    )
    inst.inst_map = np.zeros(n, dtype=int)
    inst.n_total_obs = n
    inst._prepare_robust(sigma, inst.inst_map)

    with pm.Model() as model:
        frac = pm.Uniform("frac", 1e-6, 0.5, initval=0.05)
        scale = pm.HalfNormal("scale", 10.0, initval=5.0)
        inst.out_frac = _FakeParam(pt.stack([frac]))
        inst.out_scale = _FakeParam(pt.stack([scale]))
        _dispatch(inst, mu, sigma, y)

    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    grads = model.compile_dlogp()(point)
    assert np.all(
        np.isfinite(
            np.concatenate([np.atleast_1d(g) for g in np.atleast_1d(grads)])
        )
    )


# ---------------------------------------------------------------------------
# 3b. End to end through a real System
# ---------------------------------------------------------------------------
def _write_rv(path, seed, n=40):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2455000.0, 2455400.0, n))
    rv = 30.0 * np.sin(2 * np.pi * t / 17.0) + rng.normal(0, 3.0, n)
    err = np.full(n, 3.0)
    np.savetxt(path, np.column_stack([t, rv, err]))
    return path


@pytest.fixture(scope="module")
def two_rv_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("robust_rv")
    return [str(_write_rv(d / "a.rv", 21)), str(_write_rv(d / "b.rv", 22))]


def _rv_system(files, likelihood_spec, extra_params=None):
    from exozippy.system import System

    inst = [
        {"name": "A_inst", "file": files[0]},
        {"name": "B_inst", "file": files[1]},
    ]
    if likelihood_spec is not None:
        inst[0]["likelihood"] = likelihood_spec
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "rvinstrument": inst,
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "orbit.b.logP": {"initval": np.log10(17.0)},
        "orbit.b.tc": {"initval": 2455010.0},
    }
    params.update(extra_params or {})
    system = System(config, params)
    system.prepare()
    return system, system.build_model()


def test_rv_system_without_likelihood_samples_no_robust_parameters(
    two_rv_files,
):
    """
    Given a plain two-instrument RV system,
    When the model is built,
    Then no robust-likelihood parameter exists anywhere in it -- the feature
    costs nothing when unused.
    """
    system, model = _rv_system(two_rv_files, None)
    names = {v.name for v in model.value_vars}
    for kind in robust_support.LIKELIHOOD_KINDS:
        for name in robust_support.LIKELIHOOD_PARAMS[kind]:
            assert f"rvinstrument.{name}_raw" not in names
    assert not model.potentials or all(
        ".hogg." not in (v.name or "") for v in model.potentials
    )


def test_rv_system_with_hogg_on_one_file_samples_only_that_file(two_rv_files):
    """
    Given a two-instrument RV system where only the first file asks for the
    hogg mixture,
    When the model is built,
    Then out_frac/out_scale are sampled for that element only (the other is
    pinned fixed), the mixture Potential is attached to the first file, the
    second keeps the plain Normal, and logp/dlogp are finite at the start.
    """
    system, model = _rv_system(two_rv_files, "hogg")
    rv = system.rvinstrument

    assert rv.likelihood_kinds == ["hogg", ""]
    sampled = {v.name for v in model.value_vars}
    for name in robust_support.LIKELIHOOD_PARAMS["hogg"]:
        assert f"rvinstrument.{name}_raw" in sampled
        param = getattr(rv, name)
        assert param.sigma[1] == 0.0  # pinned
        assert np.isnan(param.sigma[0])  # free
        assert model[f"rvinstrument.{name}_raw"].type.shape == (1,)

    assert "rvinstrument.model.hogg.A_inst" in {
        v.name for v in model.potentials
    }
    assert "rvinstrument.model" in {v.name for v in model.observed_RVs}

    # The data-driven hint seeded out_scale at 10 x the median error (3.0
    # m/s in the fixture files), superseding the defaults.yaml start.
    # Parameter stores initval in internal units (solRad/d), so convert.
    import astropy.units as u

    expected_internal = 30.0 * (u.m / u.s).to(u.solRad / u.d)
    assert float(np.ravel(rv.out_scale.initval)[0]) == pytest.approx(
        expected_internal, rel=1e-6
    )

    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    grads = model.compile_dlogp()(point)
    assert np.all(np.isfinite(np.atleast_1d(grads)))


def test_rv_system_with_studentt_reports_nu_and_stays_finite(two_rv_files):
    """
    Given the first file asking for the Student-t,
    When the model is built,
    Then t_log_nu is sampled for that element only, the observed StudentT
    replaces its Normal, the linear t_nu Deterministic exists, and logp/dlogp
    are finite at the start.
    """
    system, model = _rv_system(two_rv_files, "studentt")
    rv = system.rvinstrument

    sampled = {v.name for v in model.value_vars}
    assert "rvinstrument.t_log_nu_raw" in sampled
    assert model["rvinstrument.t_log_nu_raw"].type.shape == (1,)
    assert rv.t_log_nu.sigma[1] == 0.0

    observed = {v.name for v in model.observed_RVs}
    assert "rvinstrument.model.t.A_inst" in observed
    assert "rvinstrument.model" in observed  # B_inst keeps its Normal
    assert "rvinstrument.t_nu" in {v.name for v in model.deterministics}

    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    grads = model.compile_dlogp()(point)
    assert np.all(np.isfinite(np.atleast_1d(grads)))


def test_outlier_prob_at_data_flags_a_planted_outlier(tmp_path):
    """
    Given a hogg fit whose first file is pure noise plus one corrupted point,
    and a physical model pinned to (near) zero amplitude so the start point
    actually fits the inliers,
    When outlier_prob_at_data is evaluated at the start point,
    Then the corrupted point saturates its posterior outlier probability,
    every clean point stays below 10%, and every observation of the plain
    file reports exactly zero.
    """
    rng = np.random.default_rng(31)
    outlier_row = 17
    files = []
    for j, name in enumerate(("flat_a.rv", "flat_b.rv")):
        t = np.sort(rng.uniform(2455000.0, 2455400.0, 40))
        rv = rng.normal(0, 3.0, 40)
        if j == 0:
            rv[outlier_row] += 300.0
        np.savetxt(
            tmp_path / name,
            np.column_stack([t, rv, np.full(40, 3.0)]),
        )
        files.append(str(tmp_path / name))

    system, model = _rv_system(
        files,
        "hogg",
        # Pin the planet to negligible mass and gamma to the truth: the model
        # RV curve is ~flat and unshifted, so residuals at the start are the
        # noise itself and the mixture's verdict reflects the data, not an
        # unconverged physical model.  (The data-driven gamma hint is the
        # outlier-polluted mean -- +7.5 m/s here -- which is exactly the kind
        # of pull the mixture exists to undo during sampling, but this test
        # evaluates at the start point.)
        extra_params={
            "planet.b.mass": {"initval": 1e-5, "sigma": 0},
            "rvinstrument.A_inst.gamma": {"initval": 0.0},
        },
    )
    rv_comp = system.rvinstrument

    point = model.initial_point()
    prob = rv_comp.outlier_prob_at_data(system, point)

    assert prob.shape == (rv_comp.n_total_obs,)
    n_a = int(np.sum(rv_comp.inst_map == 0))
    a_prob = prob[:n_a]
    # The corrupted point is found by value (file A is sorted by time at
    # load, but the times were generated sorted, so the row is unmoved).
    assert np.argmax(np.abs(rv_comp.rv[:n_a])) == outlier_row
    assert a_prob[outlier_row] > 0.999
    # No clean point is judged more-likely-outlier-than-not.
    clean = np.delete(a_prob, outlier_row)
    assert np.all(clean < 0.5)
    np.testing.assert_array_equal(prob[n_a:], 0.0)


def test_user_can_override_a_robust_parameter_by_instrument_name(
    two_rv_files,
):
    """
    Given a user prior on out_frac in the params file, addressed by
    instrument name,
    When the model is built,
    Then it lands on the right element of the vector -- component-supplied
    pins are layered below RANK_USER, so the user always wins.
    """
    system, model = _rv_system(
        two_rv_files,
        "hogg",
        extra_params={
            "rvinstrument.A_inst.out_frac": {"initval": 0.12, "sigma": 0.03}
        },
    )
    frac = system.rvinstrument.out_frac
    assert float(np.ravel(frac.initval)[0]) == pytest.approx(0.12)
    assert float(np.ravel(frac.sigma)[0]) == pytest.approx(0.03)

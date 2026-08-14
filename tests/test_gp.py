"""Tests for the optional per-file Gaussian-process noise on Instrument.

Three layers:

1. The ``gp:`` config vocabulary and the kernel catalogue (components/gp.py) --
   pure functions, no PyMC.
2. The Instrument lifecycle hooks (_load_gp_config / _prepare_gp /
   _register_gp) -- the scaffolding, exercised on a dummy instrument.
3. The likelihood itself -- that a celerite2 marginal with a vanishing kernel
   amplitude reproduces the independent-Gaussian logp it replaces, that the
   per-file time sort is applied consistently to every array, and that a real
   two-instrument RV model builds with only the GP file's hyperparameters
   sampled.

The default is off everywhere, so the strongest regression guarantee is the
"no gp: key changes nothing" test in each layer.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from conftest import _DummyConfigManager
from exozippy.components import gp as gp_support
from exozippy.components.instrument import Instrument

# Most of this file tests EXOZIPPy's own GP scaffolding -- spec parsing,
# parameter registration, pinning, hints -- and needs no kernel at all. The
# tests that build a real celerite2 kernel are marked below.
#
# celerite2/pymc/ops.py does `from pytensor.link.jax.dispatch import
# jax_funcify` at module scope, so celerite2's PyMC backend requires jax at
# IMPORT time -- note that is separate from its build-time BUILD_JAX option,
# which is genuinely optional. On a platform that ships no jax (Intel macOS,
# where jaxlib's last x86_64 wheel predates the jax.ffi exoplanet-core
# needs) the GP feature is therefore unavailable, and these tests must SKIP
# rather than fail: an unavailable feature is not a defect.
#
# The condition is celerite2.pymc importing, not jax importing, because that
# is exactly what gp.py needs -- and it implies jax anyway.
try:
    import celerite2.pymc  # noqa: F401

    _CELERITE2_PYMC_IMPORT_ERROR = None
except ImportError as _exc:  # pragma: no cover - platform dependent
    _CELERITE2_PYMC_IMPORT_ERROR = _exc

needs_celerite2_pymc = pytest.mark.skipif(
    _CELERITE2_PYMC_IMPORT_ERROR is not None,
    reason=(
        "celerite2's PyMC backend is unimportable here "
        f"({_CELERITE2_PYMC_IMPORT_ERROR}); it imports jax unconditionally, "
        "so GP noise is unavailable on platforms that ship no jax"
    ),
)


class _DummyInstrument(Instrument):
    """Minimal concrete Instrument for unit-testing the shared GP helpers."""

    @property
    def prefix(self):
        return "dummy"

    def register_parameters(self, system):  # pragma: no cover - not exercised
        pass

    def build_likelihood(self, model, system):  # pragma: no cover
        pass


class _NoGPInstrument(_DummyInstrument):
    supports_gp = False


class _RecordingConfigManager(_DummyConfigManager):
    """Captures the hints a component pushes, so they can be asserted on."""

    def __init__(self):
        self.hints = {}
        self.scale_hints = {}

    def add_hint(self, path, value, rank=None):
        self.hints[path] = value

    def add_scale_hint(self, path, scale):
        self.scale_hints[path] = scale


def _make(config, config_manager=None):
    return _DummyInstrument(config, config_manager=config_manager)


# ---------------------------------------------------------------------------
# 1. The gp: config vocabulary
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ()),
        ("none", ()),
        ("off", ()),
        ("", ()),
        (False, ()),
        ([], ()),
        ("rotation", ("rotation",)),
        ("RotationTerm", ("rotation",)),
        ("  ROT  ", ("rotation",)),
        ("sho", ("sho",)),
        ("SHOTerm", ("sho",)),
        (["rotation", "sho"], ("rotation", "sho")),
        (["sho", "rotation"], ("rotation", "sho")),
        (["sho", "sho"], ("sho",)),
        (["rotation", "none"], ("rotation",)),
    ],
)
def test_parse_gp_spec_normalizes_every_accepted_spelling(value, expected):
    """
    Given any accepted form of the gp: key,
    When parse_gp_spec normalizes it,
    Then the result is a deduped tuple in canonical GP_TERMS order, and every
    "off" spelling yields the empty tuple.
    """
    assert gp_support.parse_gp_spec(value) == expected


def test_parse_gp_spec_rejects_unknown_terms():
    """
    Given a misspelled or unsupported kernel name,
    When parse_gp_spec runs,
    Then it raises with the context and the supported terms listed.
    """
    with pytest.raises(ValueError, match="unknown GP term"):
        gp_support.parse_gp_spec("matern", context="rvinstrument[HARPS]")


def test_parse_gp_spec_rejects_bare_true():
    """
    Given `gp: true`, which names no kernel,
    When parse_gp_spec runs,
    Then it raises rather than guessing a term.
    """
    with pytest.raises(ValueError, match="ambiguous"):
        gp_support.parse_gp_spec(True)


def test_term_tables_are_mutually_consistent():
    """
    Given the per-term parameter tables,
    When cross-checked,
    Then every term has a parameter list and an amplitude, the amplitude is
    one of that term's parameters, and every log-sampled parameter belongs to
    some term.
    """
    all_params = set()
    for kind in gp_support.GP_TERMS:
        params = gp_support.GP_TERM_PARAMS[kind]
        amp = gp_support.GP_AMPLITUDE_PARAM[kind]
        assert amp in params
        all_params.update(params)
    assert set(gp_support.GP_LOG_PARAMS) <= all_params
    # The linear names the Deterministics use must not collide with a sampled
    # parameter name (they share the component's namespace).
    assert not (set(gp_support.GP_LOG_PARAMS.values()) & all_params)


def test_every_gp_parameter_has_defaults():
    """
    Given the GP parameter tables,
    When each name is resolved against the loaded defaults,
    Then all of them have bounds and an init_scale (required of any sampled
    parameter), so no fit can fail late with a missing-scale error.
    """
    from exozippy.config import ConfigManager

    cm = ConfigManager({})
    for kind in gp_support.GP_TERMS:
        for name in gp_support.GP_TERM_PARAMS[kind]:
            cfg = cm.resolve("rvinstrument", name, shape=(1,))
            assert cfg["lower"] is not None, name
            assert cfg["upper"] is not None, name
            assert cfg["init_scale"] is not None, name
            assert cfg["initval"] is not None, name


def test_introspection_reports_every_gp_parameter_with_merged_units():
    """
    Given the GP blueprint lives at the root of components/defaults.yaml and
    each instrument overrides only the amplitude,
    When component_schema describes an instrument,
    Then every GP parameter appears, with the component's unit/bounds layered
    over the shared blueprint -- the same view ConfigManager.resolve builds.
    A component that opts out of GP support reports none.
    """
    from exozippy.introspect import component_schema

    expected = {
        name
        for kind in gp_support.GP_TERMS
        for name in gp_support.GP_TERM_PARAMS[kind]
    }

    for key in ("rvinstrument", "transit", "mulensinstrument"):
        params = component_schema(key)["parameters"]
        assert expected <= set(params), key

    rv_amp = component_schema("rvinstrument")["parameters"]["gp_rot_sigma"]
    assert rv_amp["unit"] == "m/s"  # from rvinstrument's block
    assert rv_amp["internal_unit"] == "solRad/d"
    assert rv_amp["description"]  # inherited from the root file

    rv_period = component_schema("rvinstrument")["parameters"]["gp_rot_period"]
    assert rv_period["unit"] == "d"  # entirely from the root file

    astro = component_schema("astrometryinstrument")["parameters"]
    assert not [p for p in astro if p.startswith("gp_")]


def test_gp_config_schema_entry_shape():
    """
    Given the shared gp config-schema entry,
    When inspected,
    Then it declares the standard option shape consumed by introspection.
    """
    entry = Instrument._gp_config_schema()
    assert entry["key"] == "gp"
    assert entry["kind"] == "option"
    assert entry["required"] is False
    assert set(entry["accepts"]) == set(gp_support.GP_TERMS) | {"none"}


# ---------------------------------------------------------------------------
# 2. Instrument lifecycle hooks
# ---------------------------------------------------------------------------
def test_no_gp_key_leaves_the_component_untouched():
    """
    Given instruments with no gp: key (the default),
    When the base loads the GP config and registers parameters,
    Then has_gp is False and the manifest gains nothing -- the model is
    exactly what it was before the feature existed.
    """
    inst = _make([{"file": "a.rv"}, {"file": "b.rv"}])
    assert inst.gp_terms == [(), ()]
    assert inst.has_gp is False

    manifest = {"gamma": "default"}
    inst._register_gp(manifest)
    assert manifest == {"gamma": "default"}
    assert inst.sampler_requirements() == {}


def test_gp_terms_are_parsed_per_element():
    """
    Given a mix of gp: settings across three files,
    When the base loads the GP config,
    Then each element carries its own terms and has_gp is True.
    """
    inst = _make(
        [
            {"file": "a.rv", "gp": "rotation"},
            {"file": "b.rv"},
            {"file": "c.rv", "gp": ["sho", "rotation"]},
        ]
    )
    assert inst.gp_terms == [("rotation",), (), ("rotation", "sho")]
    assert inst.has_gp is True
    assert inst._gp_elements("rotation") == [0, 2]
    assert inst._gp_elements("sho") == [2]


def test_component_that_does_not_support_gp_rejects_the_key():
    """
    Given a component with more than one observable per file (supports_gp
    False, as astrometryinstrument declares),
    When a data file sets gp:,
    Then construction raises rather than silently ignoring the key.
    """
    with pytest.raises(NotImplementedError, match="not supported"):
        _NoGPInstrument([{"file": "a.dat", "gp": "sho"}], config_manager=None)

    # ... but the same component is unaffected without the key.
    inst = _NoGPInstrument([{"file": "a.dat"}], config_manager=None)
    assert inst.has_gp is False


def test_astrometry_declares_no_gp_support():
    """
    Given astrometryinstrument models two observables per dataset,
    When its class attribute is inspected,
    Then it opts out of GP support (so the base raises on a gp: key).
    """
    from exozippy.components.astrometryinstrument.astrometryinstrument import (
        AstrometryInstrument,
    )
    from exozippy.components.mulensing.mulensinstrument import MulensInstrument
    from exozippy.components.rvinstrument.rvinstrument import RVInstrument
    from exozippy.components.transit.transit import Transit

    assert AstrometryInstrument.supports_gp is False
    # the three single-observable children keep the base default
    assert RVInstrument.supports_gp
    assert Transit.supports_gp
    assert MulensInstrument.supports_gp


def test_register_gp_pins_the_files_that_did_not_opt_in():
    """
    Given three files where only the middle one requested a rotation GP,
    When _register_gp builds the manifest,
    Then every rotation parameter is registered once (full-length vectors, so
    user paths resolve by instrument name) with sigma pinned to 0 on the two
    files that opted out and left alone (NaN) on the one that did.
    """
    inst = _make(
        [
            {"file": "a.rv"},
            {"file": "b.rv", "gp": "rotation"},
            {"file": "c.rv"},
        ]
    )
    manifest = {}
    inst._register_gp(manifest)

    assert set(manifest) == set(gp_support.GP_TERM_PARAMS["rotation"])
    for name in gp_support.GP_TERM_PARAMS["rotation"]:
        pin = manifest[name]["overrides"]["sigma"]
        assert pin[0] == 0.0 and pin[2] == 0.0
        assert np.isnan(pin[1])


def test_register_gp_omits_the_pin_when_every_file_opted_in():
    """
    Given every file requesting the same term,
    When _register_gp builds the manifest,
    Then no override is emitted at all (nothing to pin).
    """
    inst = _make(
        [{"file": "a.rv", "gp": "sho"}, {"file": "b.rv", "gp": "sho"}]
    )
    manifest = {}
    inst._register_gp(manifest)
    assert set(manifest) == set(gp_support.GP_TERM_PARAMS["sho"])
    assert all(entry == {} for entry in manifest.values())


def test_register_gp_registers_both_terms_independently():
    """
    Given one file asking for rotation and another for sho,
    When _register_gp builds the manifest,
    Then both parameter sets appear, each pinned off on the file that did not
    ask for that term -- the two kernels never share a hyperparameter.
    """
    inst = _make(
        [{"file": "a.rv", "gp": "rotation"}, {"file": "b.rv", "gp": "sho"}]
    )
    manifest = {}
    inst._register_gp(manifest)

    expected = set(gp_support.GP_TERM_PARAMS["rotation"]) | set(
        gp_support.GP_TERM_PARAMS["sho"]
    )
    assert set(manifest) == expected
    rot_pin = manifest["gp_rot_sigma"]["overrides"]["sigma"]
    sho_pin = manifest["gp_sho_sigma"]["overrides"]["sigma"]
    assert np.isnan(rot_pin[0]) and rot_pin[1] == 0.0
    assert sho_pin[0] == 0.0 and np.isnan(sho_pin[1])


def test_prepare_gp_sorts_each_file_by_time_and_hints_the_white_noise_level():
    """
    Given interleaved, unsorted observations from two files (only the second
    of which asked for a GP),
    When _prepare_gp runs,
    Then the recorded indices select that file's rows in ascending time order,
    and the amplitude hint is the median error bar scaled to user units.
    """
    cm = _RecordingConfigManager()
    inst = _make([{"file": "a.rv"}, {"file": "b.rv", "gp": "sho"}], cm)

    time = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0])
    err = np.array([0.1, 0.5, 0.2, 0.4, 0.6, 0.3])
    inst_map = np.array([0, 1, 0, 1, 1, 0])

    inst._prepare_gp(time, err, inst_map, user_factor=10.0)

    idx = inst._gp_obs_index[1]
    assert list(idx) == [1, 3, 4]  # rows of file b
    assert list(time[idx]) == [1.0, 3.0, 7.0]  # ascending
    assert 0 not in inst._gp_obs_index  # file a has no GP

    # median([0.5, 0.4, 0.6]) = 0.5, times the user_factor
    assert cm.hints["dummy.1.gp_sho_sigma"] == pytest.approx(5.0)
    assert cm.scale_hints["dummy.1.gp_sho_sigma"] == pytest.approx(5.0)
    assert "dummy.0.gp_sho_sigma" not in cm.hints


def test_prepare_gp_hints_both_amplitudes_when_both_terms_are_used():
    """
    Given a file asking for both kernels,
    When _prepare_gp runs,
    Then each term's own amplitude gets the hint (they are separate
    parameters and must both start at a sensible scale).
    """
    cm = _RecordingConfigManager()
    inst = _make([{"file": "a.rv", "gp": ["rotation", "sho"]}], cm)
    inst._prepare_gp(np.arange(5.0), np.full(5, 2.0), np.zeros(5, dtype=int))

    assert cm.hints["dummy.0.gp_rot_sigma"] == pytest.approx(2.0)
    assert cm.hints["dummy.0.gp_sho_sigma"] == pytest.approx(2.0)


def test_prepare_gp_skips_the_hint_for_degenerate_errors():
    """
    Given a file whose error column is all zeros,
    When _prepare_gp runs,
    Then no amplitude hint is pushed -- a zero start would pin the logit
    transform against its lower bound -- but the sort index is still recorded.
    """
    cm = _RecordingConfigManager()
    inst = _make([{"file": "a.rv", "gp": "sho"}], cm)
    inst._prepare_gp(np.arange(5.0), np.zeros(5), np.zeros(5, dtype=int))

    assert cm.hints == {}
    assert list(inst._gp_obs_index[0]) == [0, 1, 2, 3, 4]


def test_prepare_gp_rejects_a_file_with_too_few_observations():
    """
    Given a GP requested on a file with fewer than 3 epochs,
    When _prepare_gp runs,
    Then it raises: there is nothing for a covariance kernel to constrain.
    """
    inst = _make(
        [{"file": "a.rv", "gp": "rotation"}], _RecordingConfigManager()
    )
    with pytest.raises(ValueError, match="at least 3 observations"):
        inst._prepare_gp(np.arange(2.0), np.ones(2), np.zeros(2, dtype=int))


def test_prepare_gp_is_a_noop_without_a_gp_key():
    """
    Given no file requesting a GP,
    When _prepare_gp runs,
    Then nothing is recorded and no hint is pushed.
    """
    cm = _RecordingConfigManager()
    inst = _make([{"file": "a.rv"}], cm)
    inst._prepare_gp(np.arange(5.0), np.ones(5), np.zeros(5, dtype=int))
    assert inst._gp_obs_index == {}
    assert inst._gp_time is None
    assert cm.hints == {}


def test_gp_imposes_no_sampler_constraint():
    """
    Given a file with a GP,
    When sampler_requirements is queried,
    Then it is empty: celerite2 registers a JAX implementation of every op it
    uses with PyTensor's JAX linker, so the JAX-funcifying samplers work. (An
    earlier version of this feature wrongly excluded numpyro/blackjax by
    analogy with transit's exoplanet_core op; see the note in instrument.py.)
    """
    inst = _make([{"file": "a.rv", "gp": "rotation"}])
    assert inst.sampler_requirements() == {}


@needs_celerite2_pymc
def test_celerite_ops_have_a_registered_jax_implementation():
    """
    Given celerite2's PyTensor ops,
    When the JAX linker's dispatch table is consulted,
    Then a conversion is registered for them -- the fact the no-constraint
    decision above rests on. If a celerite2 upgrade drops it, this fails here
    rather than at someone's first numpyro run.
    """
    from celerite2.pymc.ops import _CeleriteOp
    from pytensor.link.jax.dispatch import jax_funcify

    assert _CeleriteOp in jax_funcify.registry


# ---------------------------------------------------------------------------
# 3. The likelihood
# ---------------------------------------------------------------------------
def _celerite_logp(kernel_kind, params, t, y, mu, sigma):
    """logp of one celerite2 marginal likelihood, evaluated numerically."""
    with pm.Model() as model:
        kernel = gp_support.build_term(kernel_kind, params)
        gp_support.marginal_likelihood(
            "obs",
            kernel,
            t=t,
            yerr=pt.as_tensor_variable(sigma),
            mean=pt.as_tensor_variable(mu),
            observed=y,
        )
    return float(model.compile_logp()({}))


@needs_celerite2_pymc
def test_vanishing_amplitude_reproduces_the_independent_gaussian_logp():
    """
    Given a GP whose kernel amplitude is driven to ~0,
    When its marginal likelihood is evaluated,
    Then it matches sum(Normal.logpdf(y; mu, sigma)) -- the likelihood the GP
    replaces. This pins the mean/yerr/observed wiring: a swapped or misscaled
    argument would not reduce to the white-noise limit.
    """
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 30, 40))
    mu = 0.5 * np.sin(t)
    sigma = rng.uniform(0.05, 0.15, 40)
    y = mu + rng.normal(0, sigma)

    expected = float(
        np.sum(
            -0.5 * ((y - mu) / sigma) ** 2
            - np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
        )
    )

    rot = _celerite_logp(
        "rotation",
        {"sigma": 1e-8, "period": 12.0, "Q0": 2.0, "dQ": 1.0, "f": 0.5},
        t,
        y,
        mu,
        sigma,
    )
    sho = _celerite_logp(
        "sho", {"sigma": 1e-8, "rho": 5.0, "Q": 1.0 / 3.0}, t, y, mu, sigma
    )

    assert rot == pytest.approx(expected, rel=1e-8)
    assert sho == pytest.approx(expected, rel=1e-8)


@needs_celerite2_pymc
def test_a_real_kernel_prefers_correlated_data_over_white_noise():
    """
    Given data with a genuine smooth correlated component,
    When the same data are scored with a real GP kernel and with independent
    Gaussian errors,
    Then the GP logp is higher -- the kernel is actually being used, not
    silently bypassed.
    """
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 40, 60))
    sigma = np.full(60, 0.05)
    y = 0.4 * np.sin(2 * np.pi * t / 11.0) + rng.normal(0, sigma)
    mu = np.zeros_like(t)

    white = float(
        np.sum(
            -0.5 * ((y - mu) / sigma) ** 2
            - np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
        )
    )
    gp = _celerite_logp(
        "rotation",
        {"sigma": 0.4, "period": 11.0, "Q0": 5.0, "dQ": 1.0, "f": 0.1},
        t,
        y,
        mu,
        sigma,
    )

    assert gp > white


@needs_celerite2_pymc
def test_build_kernel_sums_the_requested_terms():
    """
    Given a file asking for both terms,
    When build_kernel runs,
    Then the resulting kernel's covariance equals the sum of the two
    individual kernels' (celerite2 terms add).
    """
    rot_p = {"sigma": 0.3, "period": 8.0, "Q0": 2.0, "dQ": 1.0, "f": 0.4}
    sho_p = {"sigma": 0.2, "rho": 3.0, "Q": 1.0}
    tau = pt.as_tensor_variable(np.linspace(0.0, 5.0, 11))

    both = gp_support.build_kernel(
        ("rotation", "sho"), {"rotation": rot_p, "sho": sho_p}
    )
    rot = gp_support.build_term("rotation", rot_p)
    sho = gp_support.build_term("sho", sho_p)

    assert np.allclose(
        both.get_value(tau).eval(),
        (rot.get_value(tau) + sho.get_value(tau)).eval(),
    )


def test_add_observation_likelihood_without_gp_is_the_plain_normal():
    """
    Given no file requesting a GP,
    When add_observation_likelihood runs,
    Then it emits a single pm.Normal over all observations, with the same
    logp the component produced before this feature existed.
    """
    rng = np.random.default_rng(2)
    y = rng.normal(size=12)
    mu = np.zeros(12)
    sigma = np.full(12, 0.2)

    inst = _make([{"file": "a.rv"}, {"file": "b.rv"}])
    inst.inst_map = np.repeat([0, 1], 6)
    inst.n_total_obs = 12

    with pm.Model() as model:
        rv = inst.add_observation_likelihood(
            "m",
            mu=pt.as_tensor_variable(mu),
            sigma=pt.as_tensor_variable(sigma),
            observed=y,
        )
        assert rv is not None
        assert [v.name for v in model.observed_RVs] == ["m"]

    expected = float(
        np.sum(
            -0.5 * ((y - mu) / sigma) ** 2
            - np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
        )
    )
    assert float(model.compile_logp()({})) == pytest.approx(expected)


@needs_celerite2_pymc
def test_mixed_files_split_into_one_normal_plus_one_gp_per_file():
    """
    Given three files where the middle one asked for a GP,
    When add_observation_likelihood runs,
    Then the two plain files share a single Normal, the GP file gets its own
    celerite2 term, and with a vanishing GP amplitude the total logp equals
    the all-Normal logp -- i.e. the split loses and double-counts nothing.
    """
    rng = np.random.default_rng(3)
    n_per = 8
    time = np.concatenate(
        [np.sort(rng.uniform(0, 20, n_per)) for _ in range(3)]
    )
    inst_map = np.repeat([0, 1, 2], n_per)
    sigma = rng.uniform(0.05, 0.2, 3 * n_per)
    mu = np.sin(time)
    y = mu + rng.normal(0, sigma)

    inst = _make(
        [{"file": "a.rv"}, {"file": "b.rv", "gp": "sho"}, {"file": "c.rv"}],
        _RecordingConfigManager(),
    )
    inst.inst_map = inst_map
    inst.n_total_obs = len(time)
    inst._prepare_gp(time, sigma, inst_map)

    with pm.Model() as model:
        # Stand in for the Parameters stage 5 would have built, with an
        # amplitude small enough that the GP reduces to white noise.
        inst.gp_sho_sigma = _FakeParam(np.array([0.0, 1e-8, 0.0]))
        inst.gp_sho_rho = _FakeParam(np.array([1.0, 5.0, 1.0]))
        inst._gp_linear = {
            "gp_sho_q": pt.as_tensor_variable([1.0, 1.0 / 3.0, 1.0])
        }
        inst._build_gp_deterministics = lambda: None

        inst.add_observation_likelihood(
            "m",
            mu=pt.as_tensor_variable(mu),
            sigma=pt.as_tensor_variable(sigma),
            observed=y,
        )

    names = sorted(v.name for v in model.observed_RVs)
    assert names == ["m", "m.gp.1"]

    expected = float(
        np.sum(
            -0.5 * ((y - mu) / sigma) ** 2
            - np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
        )
    )
    assert float(model.compile_logp()({})) == pytest.approx(expected, rel=1e-8)


@needs_celerite2_pymc
def test_gp_likelihood_is_invariant_to_the_input_ordering():
    """
    Given the same observations presented in time order and shuffled,
    When each is scored through add_observation_likelihood,
    Then the logp is identical -- _prepare_gp's per-file sort permutation is
    applied to the times, the model, the sigmas and the data alike. (Feeding
    celerite2 unsorted times silently returns a wrong answer, so this is the
    test that a reordering bug would break.)
    """
    rng = np.random.default_rng(4)
    n = 20
    time = np.sort(rng.uniform(0, 25, n))
    sigma = rng.uniform(0.05, 0.2, n)
    mu = 0.3 * np.cos(time)
    y = mu + rng.normal(0, sigma)
    perm = rng.permutation(n)

    def logp(order):
        inst = _make(
            [{"file": "a.rv", "gp": "rotation"}], _RecordingConfigManager()
        )
        inst.inst_map = np.zeros(n, dtype=int)
        inst.n_total_obs = n
        inst._prepare_gp(time[order], sigma[order], inst.inst_map)
        with pm.Model() as model:
            inst.gp_rot_sigma = _FakeParam(np.array([0.3]))
            inst.gp_rot_period = _FakeParam(np.array([9.0]))
            inst.gp_rot_f = _FakeParam(np.array([0.4]))
            inst._gp_linear = {
                "gp_rot_q0": pt.as_tensor_variable([3.0]),
                "gp_rot_dq": pt.as_tensor_variable([2.0]),
            }
            inst._build_gp_deterministics = lambda: None
            inst.add_observation_likelihood(
                "m",
                mu=pt.as_tensor_variable(mu[order]),
                sigma=pt.as_tensor_variable(sigma[order]),
                observed=y[order],
            )
        return float(model.compile_logp()({}))

    assert logp(perm) == pytest.approx(logp(np.arange(n)), rel=1e-10)


class _FakeParam:
    """Stand-in for a built Parameter: only .value is read by _gp_kernel."""

    def __init__(self, values):
        self.value = pt.as_tensor_variable(values)


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
    d = tmp_path_factory.mktemp("gp_rv")
    return [str(_write_rv(d / "a.rv", 11)), str(_write_rv(d / "b.rv", 12))]


def _rv_system(files, gp_spec):
    from exozippy.system import System

    inst = [
        {"name": "A_inst", "file": files[0]},
        {"name": "B_inst", "file": files[1]},
    ]
    if gp_spec is not None:
        inst[0]["gp"] = gp_spec
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
    system = System(config, params)
    system.prepare()
    return system, system.build_model()


def test_rv_system_without_gp_samples_no_gp_parameters(two_rv_files):
    """
    Given a two-instrument RV system with no gp: key,
    When the model is built,
    Then no GP parameter exists at all -- the default costs nothing.
    """
    system, model = _rv_system(two_rv_files, None)
    assert system.rvinstrument.has_gp is False
    assert not [v.name for v in model.value_vars if ".gp_" in v.name]


@needs_celerite2_pymc
def test_rv_system_with_gp_on_one_file_samples_only_that_file(two_rv_files):
    """
    Given a two-instrument RV system where only the first file asks for a
    rotation GP,
    When the model is built,
    Then the rotation hyperparameters are sampled, the second file's elements
    of those same vectors are pinned fixed (sigma 0), the GP marginal
    likelihood is attached to the first file only, and the logp and its
    gradient are finite at the start point.
    """
    system, model = _rv_system(two_rv_files, "rotation")
    rv = system.rvinstrument

    assert rv.gp_terms == [("rotation",), ()]
    sampled = {v.name for v in model.value_vars}
    for name in gp_support.GP_TERM_PARAMS["rotation"]:
        assert f"rvinstrument.{name}_raw" in sampled
        param = getattr(rv, name)
        # element 0 opted in (free), element 1 was pinned
        assert param.sigma[1] == 0.0
        assert np.isnan(param.sigma[0])
        # only the opted-in element is sampled, so the raw vector is length 1
        assert model[f"rvinstrument.{name}_raw"].type.shape == (1,)

    observed = {v.name for v in model.observed_RVs}
    assert "rvinstrument.model.gp.A_inst" in observed
    assert "rvinstrument.model" in observed  # B_inst keeps a plain Normal
    assert "rvinstrument.model.gp.B_inst" not in observed

    # The linear quality factors are reported alongside the sampled logs.
    det = {v.name for v in model.deterministics}
    assert "rvinstrument.gp_rot_q0" in det
    assert "rvinstrument.gp_rot_dq" in det

    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    assert np.all(
        np.isfinite(
            np.concatenate(
                [np.atleast_1d(g) for g in [model.compile_dlogp()(point)]]
            )
        )
    )


@needs_celerite2_pymc
def test_gp_prediction_is_consistent_between_the_data_and_grid_evaluators(
    two_rv_files,
):
    """
    Given a fitted GP,
    When the conditional mean is asked for at the observation times and again
    on a grid that happens to equal those same times,
    Then the two agree. They go through different celerite2 code paths (the
    factorization shortcut vs the general_matmul ops), so this catches a
    mis-wired mean, a stale factorization, or a permutation applied to one
    path and not the other.
    """
    system, model = _rv_system(two_rv_files, "rotation")
    rv = system.rvinstrument
    for comp in system.active_components.values():
        comp.compile_plotters(model, system)
    assert rv.has_gp_plotters()

    point = system.get_internal_point(model, system.get_raw_start(model))

    at_data = rv.gp_mean_at_data(system, point)
    idx = rv._gp_obs_index[0]
    on_grid = rv.gp_mean_on_grid(system, point, 0, rv.time[idx])

    assert np.allclose(on_grid, at_data[idx], rtol=0, atol=1e-8)
    # Observations belonging to the non-GP instrument stay exactly zero, so a
    # caller can subtract the vector unconditionally.
    other = np.setdiff1d(np.arange(rv.n_total_obs), idx)
    assert np.all(at_data[other] == 0.0)
    # ... and a file without a GP predicts zeros on any grid.
    assert np.all(rv.gp_mean_on_grid(system, point, 1, rv.time[:5]) == 0.0)


@needs_celerite2_pymc
def test_plots_put_the_gp_in_the_unphased_model_and_out_of_the_phased_data(
    two_rv_files,
):
    """
    Given a GP on the first of two RV instruments,
    When the plot payloads are built,
    Then the unphased chart gains a "model+GP" curve for that instrument only
    (the model the likelihood actually fits), while the phased chart's data
    has the GP removed -- differing from the no-GP background subtraction by
    exactly the GP conditional mean.
    """
    system, model = _rv_system(two_rv_files, "rotation")
    rv = system.rvinstrument
    for comp in system.active_components.values():
        comp.compile_plotters(model, system)
    point = system.get_internal_point(model, system.get_raw_start(model))

    # Unphased: one extra full-model curve, for the GP instrument only.
    curves = rv._eval_unphased_gp_models(system, point)
    assert [i for i, _, _ in curves] == [0]
    _, t_gp, y_gp = curves[0]
    # It spans only that instrument's own data range: the conditional mean
    # decays to zero away from the data and would draw a misleading flat line.
    mask = rv.inst_map == 0
    assert t_gp.min() == pytest.approx(rv.time[mask].min())
    assert t_gp.max() == pytest.approx(rv.time[mask].max())
    assert np.all(np.isfinite(y_gp))

    specs = rv.plot_data(system, point)
    unphased = next(s for s in specs if s.id.endswith("unphased"))
    model_names = [tr.name for tr in unphased.traces if tr.role == "model"]
    assert "model" in model_names
    assert any("model+GP" in n for n in model_names)

    # Phased: the GP is folded into the per-observation background that gets
    # subtracted from the data. This system has exactly one orbit, so the
    # "other orbits" term is identically zero and the background must equal
    # the GP conditional mean on the nose.
    prep = rv._phased_arrays(system, point, 0, rv._plot_orbit_map[0])
    gp_at_data = rv.gp_mean_at_data(system, point)
    assert np.any(gp_at_data != 0.0)  # the GP is doing something
    assert np.allclose(prep["other_signals"], gp_at_data, rtol=0, atol=1e-12)


@needs_celerite2_pymc
def test_user_can_override_a_gp_hyperparameter_prior(two_rv_files):
    """
    Given a user prior on a GP hyperparameter in the params file, addressed by
    instrument name,
    When the model is built,
    Then it lands on the right element of the vector -- component-supplied
    pins are layered below RANK_USER, so the user always wins.
    """
    from exozippy.system import System

    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "rvinstrument": [
            {"name": "A_inst", "file": two_rv_files[0], "gp": "rotation"},
            {"name": "B_inst", "file": two_rv_files[1], "gp": "rotation"},
        ],
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "orbit.b.logP": {"initval": np.log10(17.0)},
        "orbit.b.tc": {"initval": 2455010.0},
        "rvinstrument.B_inst.gp_rot_period": {"initval": 22.56, "sigma": 0.29},
    }
    system = System(config, params)
    system.prepare()
    system.build_model()

    period = system.rvinstrument.gp_rot_period
    assert period.initval[1] == pytest.approx(22.56)
    assert period.sigma[1] == pytest.approx(0.29)
    # the untouched instrument keeps the defaults.yaml start
    assert period.initval[0] != pytest.approx(22.56)

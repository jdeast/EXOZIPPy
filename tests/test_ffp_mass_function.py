"""Tests for the free-floating-planet mass function.

The FFP mass function is selected per star (`mass_function: ffp` on a star
block) and replaces, for that star only, the stellar IMF the galacticmodel
block otherwise applies to every star.  See
`galacticmodel.ffp_logmass_logp` -- the single function the functional form
lives in -- and `Star._parse_mass_functions`.
"""

import logging

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from conftest import _DummyConfigManager, _DummySystem
from exozippy.components.galacticmodel.galacticmodel import (
    GalacticModel,
    _lognormal_log_norm,
    ffp_logmass_logp,
)
from exozippy.components.star.star import (
    FFP_LOGMASS_CALIBRATION_MIN,
    Star,
)
from exozippy.config import ConfigManager
from exozippy.constants import FFP_MASS_FUNCTION_SLOPE

_RA_RAD = np.deg2rad(270.0)
_DEC_RAD = np.deg2rad(-29.0)

# star/defaults.yaml hard bounds on the sampled log10 mass (dex solMass).
_LOGMASS_LOWER = -9.0
_LOGMASS_UPPER = 2.5

_LN10 = np.log(10.0)

# Chabrier 2003 system IMF, as build_likelihood applies it.
_CHABRIER_LOG_MC = np.log10(0.22)
_CHABRIER_SIGMA = 0.57

_HBL_DEX = np.log10(0.075)  # hydrogen-burning limit, dex(solMass)


class _MockParam:
    """Minimal Parameter stand-in: initval, a PyTensor value, hard bounds."""

    def __init__(self, initval, lower=None, upper=None):
        self.initval = np.atleast_1d(np.asarray(initval, dtype=np.float64))
        self.value = pt.as_tensor_variable(self.initval)
        self.lower = lower
        self.upper = upper
        self.prior_contributions = []

    def add_prior_contribution(self, *args, **kwargs):
        """Reporting-only hook (see parameter.PriorContribution).

        build_likelihood declares what its potentials ARE so the reported
        tables can describe them; the declaration changes no math.  Recorded
        rather than dropped so a test could assert on it.
        """
        self.prior_contributions.append((args, kwargs))


class _MockStar:
    """Stand-in for the Star component, with n stars and a mass-function mask."""

    def __init__(self, logmass, ffp_mask=None, alpha=FFP_MASS_FUNCTION_SLOPE):
        n = len(np.atleast_1d(logmass))
        self.ra = _MockParam([_RA_RAD] * n)
        self.dec = _MockParam([_DEC_RAD] * n)
        self.logmass = _MockParam(
            logmass,
            lower=np.full(n, _LOGMASS_LOWER),
            upper=np.full(n, _LOGMASS_UPPER),
        )
        self.distance = _MockParam([8000.0] * n)
        self.pm_ra = _MockParam([0.0] * n)
        self.pm_dec = _MockParam([0.0] * n)
        self.rv = _MockParam([0.0] * n)
        if ffp_mask is not None:
            self.ffp_mask = np.asarray(ffp_mask, dtype=bool)
            self.ffp_alpha = np.full(n, alpha, dtype=float)


class _MockSystem:
    def __init__(self, star):
        self.star = star


def _imf_potential(star, imf=None):
    """Numeric value of the galacticmodel.imf_prior potential for a star."""
    cfg = [{}] if imf is None else [{"IMF": imf}]
    gm = GalacticModel(cfg, _DummyConfigManager())
    with pm.Model() as model:
        gm.build_likelihood(model, _MockSystem(star))
    return float(model.named_vars["galacticmodel.imf_prior"].eval())


def _ffp_closed_form(x, alpha, lower=_LOGMASS_LOWER, upper=_LOGMASS_UPPER):
    """log p(x) for dN/dlogM ~ M^-alpha, normalized on [lower, upper].

    Written out independently of the implementation: p(x) ~ 10^(-alpha x), so
    Z = (10^(-alpha*upper) - 10^(-alpha*lower)) / (-alpha ln10).
    """
    k = -alpha
    z = (10.0 ** (k * upper) - 10.0 ** (k * lower)) / (k * _LN10)
    return k * _LN10 * x - np.log(z)


def _chabrier_closed_form(x, lower=_LOGMASS_LOWER, upper=_LOGMASS_UPPER):
    param = _MockParam(
        x, lower=np.atleast_1d(lower), upper=np.atleast_1d(upper)
    )
    return -0.5 * ((x - _CHABRIER_LOG_MC) / _CHABRIER_SIGMA) ** 2 - (
        _lognormal_log_norm(_CHABRIER_LOG_MC, _CHABRIER_SIGMA, param)
    )


# ---------------------------------------------------------------------------
# The mass function itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("logmass", [-2.5406, -4.0])
def test_ffp_logp_equals_the_closed_form(logmass):
    """
    Given the Sumi+2023 FFP mass function on star.logmass's default support,
    When ffp_logmass_logp is evaluated at a mass,
    Then it equals the independently written closed form
      -alpha*ln10*x - log Z, with Z the integral of 10^(-alpha x) over the
      support.

    -2.5406 dex is 3 Mjup, the case this feature exists for.
    """
    # Arrange
    param = _MockParam(logmass, lower=_LOGMASS_LOWER, upper=_LOGMASS_UPPER)

    # Act
    got = float(np.atleast_1d(ffp_logmass_logp(logmass, 0.96, param))[0])

    # Assert
    assert got == pytest.approx(_ffp_closed_form(logmass, 0.96), rel=1e-12)


def test_ffp_logp_falls_by_alpha_ln10_per_dex():
    """
    Given the FFP mass function,
    When its logp is compared one dex apart,
    Then the difference is exactly -alpha*ln10, NOT (1-alpha)*ln10.

    The normalization cancels in a difference, so this isolates the change of
    variables: Sumi+2023 quote dN/dlogM, already a density in the sampled
    coordinate, so there is no |dM/dx| = M ln10 Jacobian.  Applying Salpeter's
    (1 - alpha) exponent here would be off by a whole factor of M.
    """
    # Arrange
    alpha = 0.96
    param = _MockParam(0.0, lower=_LOGMASS_LOWER, upper=_LOGMASS_UPPER)

    # Act
    hi = float(np.atleast_1d(ffp_logmass_logp(-3.0, alpha, param))[0])
    lo = float(np.atleast_1d(ffp_logmass_logp(-4.0, alpha, param))[0])

    # Assert
    assert hi - lo == pytest.approx(-alpha * _LN10, rel=1e-12)
    assert hi - lo != pytest.approx((1.0 - alpha) * _LN10, rel=1e-3)


def test_ffp_density_integrates_to_one_over_the_support():
    """
    Given the FFP mass function normalized over star.logmass's hard bounds,
    When exp(logp) is integrated numerically over that support,
    Then the result is 1 -- it is a proper density, so its logp is comparable
      with the IMF branches over the same support.
    """
    # Arrange
    param = _MockParam(0.0, lower=_LOGMASS_LOWER, upper=_LOGMASS_UPPER)
    x = np.linspace(_LOGMASS_LOWER, _LOGMASS_UPPER, 400001)

    # Act
    integral = np.trapezoid(np.exp(ffp_logmass_logp(x, 0.96, param)), x)

    # Assert
    assert integral == pytest.approx(1.0, rel=1e-6)


def test_ffp_density_integrates_to_one_for_a_user_slope():
    """
    Given a user-supplied slope far from the published value,
    When the density is integrated over the support,
    Then it is still 1 -- the normalizer tracks alpha rather than being a
      constant fitted to the default.
    """
    # Arrange
    param = _MockParam(0.0, lower=_LOGMASS_LOWER, upper=_LOGMASS_UPPER)
    x = np.linspace(_LOGMASS_LOWER, _LOGMASS_UPPER, 400001)

    # Act
    integral = np.trapezoid(np.exp(ffp_logmass_logp(x, 1.8, param)), x)

    # Assert
    assert integral == pytest.approx(1.0, rel=1e-6)


def test_ffp_stops_penalizing_a_jupiter_mass_lens_relative_to_a_star():
    """
    Given a 3 Mjup lens and a 0.5 Msun star,
    When each mass prior's preference between the two is measured,
    Then Chabrier charges ~5 nats for being 3 Mjup while the FFP function
      *prefers* it by ~5 nats -- a swing of ~10 nats, which is the penalty
      this feature removes.

    Stated as a DIFFERENCE within one prior on purpose.  Each prior's
    normalizer cancels, so this is the comparison that means something; the
    absolute logp of the two priors at one mass is not comparable in any
    useful way, because the FFP density is spread over the whole (arbitrary,
    default) support and its normalizer is dominated by the lower bound.
    That is exactly why selecting it warns about that bound.
    """
    # Arrange
    ffp_x = np.log10(3.0 / 1047.57)  # 3 Mjup in solMass
    star_x = np.log10(0.5)
    param = _MockParam(
        [ffp_x, star_x],
        lower=np.full(2, _LOGMASS_LOWER),
        upper=np.full(2, _LOGMASS_UPPER),
    )

    # Act
    ffp = ffp_logmass_logp(np.array([ffp_x, star_x]), 0.96, param)
    chabrier = _chabrier_closed_form(np.array([ffp_x, star_x]))

    # Assert
    assert ffp[0] - ffp[1] == pytest.approx(4.95, abs=0.05)
    assert chabrier[0] - chabrier[1] == pytest.approx(-5.25, abs=0.05)
    assert (ffp[0] - ffp[1]) - (chabrier[0] - chabrier[1]) > 9.0


# ---------------------------------------------------------------------------
# Per-star selection inside the galacticmodel potential
# ---------------------------------------------------------------------------


def test_default_path_is_untouched_by_the_ffp_branch():
    """
    Given stars that did not select a mass function (and a star object from
      before this feature existed, with no ffp_mask attribute at all),
    When the IMF potential is built,
    Then both give the identical Chabrier value -- the default model is
      byte-for-byte what it was.
    """
    # Arrange
    logmass = [np.log10(0.5), np.log10(0.3)]
    legacy = _MockStar(logmass)  # no ffp_mask attribute
    modern = _MockStar(logmass, ffp_mask=[False, False])

    # Act
    legacy_val = _imf_potential(legacy)
    modern_val = _imf_potential(modern)

    # Assert
    expected = float(np.sum(_chabrier_closed_form(np.array(logmass))))
    assert legacy_val == modern_val
    assert legacy_val == pytest.approx(expected, rel=1e-12)


def test_mixed_system_puts_the_right_prior_on_each_star():
    """
    Given a realistic microlensing system -- a STELLAR source and an
      FFP LENS -- with mass_function: ffp on the lens only,
    When the IMF potential is built,
    Then it is exactly ffp(lens) + chabrier(source): each star draws its own
      mass prior, and neither leaks into the other.

    This is why the selection has to be per star rather than a
    galacticmodel-level switch: the potential is one sum over the whole
    star.logmass vector.
    """
    # Arrange
    lens_x = np.log10(3.0 / 1047.57)  # 3 Mjup FFP lens
    source_x = np.log10(1.1)  # ordinary bulge source star
    star = _MockStar([lens_x, source_x], ffp_mask=[True, False])

    # Act
    got = _imf_potential(star)

    # Assert
    expected = float(
        np.atleast_1d(
            ffp_logmass_logp(lens_x, FFP_MASS_FUNCTION_SLOPE, star.logmass)
        )[0]
    ) + float(np.atleast_1d(_chabrier_closed_form(source_x))[0])
    assert got == pytest.approx(expected, rel=1e-12)

    # ... and it is genuinely different from charging both the stellar IMF
    both_stellar = _imf_potential(_MockStar([lens_x, source_x]))
    assert abs(got - both_stellar) > 4.0


def test_mixed_system_works_under_salpeter_too():
    """
    Given IMF: Salpeter and one star opting into the FFP mass function,
    When the IMF potential is built,
    Then the FFP star draws the FFP prior and the other draws Salpeter --
      the per-star choice overrides whichever stellar IMF is configured.
    """
    # Arrange
    lens_x = -3.0
    source_x = np.log10(1.1)
    star = _MockStar([lens_x, source_x], ffp_mask=[True, False])

    # Act
    got = _imf_potential(star, imf="salpeter")

    # Assert: Salpeter is dN/dM ~ M^-2.35, so in log mass its exponent is
    # (1 - 2.35) = -1.35 -- i.e. the same closed form with "alpha" = 1.35.
    expected = _ffp_closed_form(lens_x, FFP_MASS_FUNCTION_SLOPE) + (
        _ffp_closed_form(source_x, 1.35)
    )
    assert got == pytest.approx(expected, rel=1e-12)


def test_user_alpha_takes_effect_in_the_potential():
    """
    Given a star whose mass_function names a non-default alpha,
    When the IMF potential is built,
    Then the potential uses that slope, not the published default.
    """
    # Arrange
    x = -4.0
    star = _MockStar([x], ffp_mask=[True], alpha=1.43)

    # Act
    got = _imf_potential(star)

    # Assert
    assert got == pytest.approx(_ffp_closed_form(x, 1.43), rel=1e-12)
    assert got != pytest.approx(
        _ffp_closed_form(x, FFP_MASS_FUNCTION_SLOPE), rel=1e-3
    )


def test_ffp_gradient_is_finite():
    """
    Given the FFP prior on a star's logmass,
    When the potential's gradient wrt the mass is taken,
    Then it is finite and equals the analytic -alpha*ln10 (a linear tilt, the
      friendliest possible geometry for NUTS).
    """
    # Arrange
    x = pt.dscalar("logmass")
    star = _MockStar([-3.0], ffp_mask=[True])
    star.logmass.value = pt.stack([x])

    # Act
    with pm.Model() as model:
        GalacticModel([{}], _DummyConfigManager()).build_likelihood(
            model, _MockSystem(star)
        )
    node = model.named_vars["galacticmodel.imf_prior"]
    grad = pt.grad(node, x).eval({x: -3.0})

    # Assert
    assert np.isfinite(grad)
    assert float(grad) == pytest.approx(
        -FFP_MASS_FUNCTION_SLOPE * _LN10, rel=1e-10
    )


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _star(blocks, user_params=None):
    return Star(blocks, ConfigManager(dict(user_params or {})))


def test_absent_mass_function_defaults_to_the_stellar_imf():
    """
    Given star blocks with no mass_function key (every config written before
      this feature),
    When the component is constructed,
    Then every star draws the stellar IMF and the FFP mask is all False.
    """
    # Act
    star = _star([{"name": "A"}, {"name": "B"}])

    # Assert
    assert star.mass_functions == ["imf", "imf"]
    assert not star.ffp_mask.any()
    assert np.allclose(star.ffp_alpha, FFP_MASS_FUNCTION_SLOPE)


@pytest.mark.parametrize("spec", ["ffp", "FFP", {"kind": "ffp"}])
def test_ffp_is_selected_by_string_or_dict_and_case_insensitively(spec):
    """
    Given mass_function written as a bare string or as a dict,
    When the component is constructed,
    Then both select the FFP mass function with the published slope.
    """
    # Act
    star = _star([{"name": "Lens", "mass_function": spec}])

    # Assert
    assert star.ffp_mask.tolist() == [True]
    assert star.ffp_alpha[0] == pytest.approx(FFP_MASS_FUNCTION_SLOPE)


def test_user_alpha_is_read_from_the_dict_form():
    """
    Given a user-supplied alpha (this measurement is uncertain and Roman will
      revise it, so tracking a new fit must not require a source edit),
    When the component is constructed,
    Then that slope is stored for that star only.
    """
    # Act
    star = _star(
        [
            {"name": "Lens", "mass_function": {"kind": "ffp", "alpha": 1.25}},
            {"name": "Source"},
        ]
    )

    # Assert
    assert star.ffp_alpha[0] == pytest.approx(1.25)
    assert star.ffp_alpha[1] == pytest.approx(FFP_MASS_FUNCTION_SLOPE)


@pytest.mark.parametrize(
    "spec, match",
    [
        ("salpeter", "not implemented"),
        ({"kind": "kroupa"}, "not implemented"),
        ({"kind": "ffp", "slope": 1.0}, "unknown mass_function key"),
        ({"kind": "ffp", "logmass_floor": -7.0}, "unknown mass_function key"),
        ({"alpha": 1.0}, "must name a 'kind'"),
        ({"kind": "imf", "alpha": 1.0}, "takes no options"),
    ],
)
def test_bad_mass_function_spellings_raise(spec, match):
    """
    Given a misspelled or unsupported mass_function entry,
    When the component is constructed,
    Then it raises rather than silently ignoring the key -- a silently
      ignored mass-function key is exactly the bug PR #82 fixed for
      IMF: Salpeter.
    """
    # Act / Assert
    with pytest.raises(ValueError, match=match):
        _star([{"name": "Lens", "mass_function": spec}])


def test_ffp_without_a_galacticmodel_block_warns(caplog):
    """
    Given mass_function: ffp but no galacticmodel block,
    When parameters are registered,
    Then a warning says the key does nothing, because with no galacticmodel
      no mass prior is applied to any star at all.
    """
    # Arrange
    star = _star([{"name": "Lens", "mass_function": "ffp"}])
    system = _DummySystem()
    system.config_manager = star.config_manager

    # Act
    with caplog.at_level(logging.WARNING):
        star.register_parameters(system)

    # Assert
    assert "no 'galacticmodel' block" in caplog.text
    assert "star.Lens" in caplog.text


# ---------------------------------------------------------------------------
# Bounds: the FFP support is the user's call, so nothing clamps it
# ---------------------------------------------------------------------------


def _logmass_lower(blocks, imf=None, user_params=None):
    """Resolved star.logmass lower bound(s) for the given star blocks."""
    star = _star(blocks, user_params)
    system = _DummySystem()
    system.config_manager = star.config_manager
    if imf is not None:
        system.galacticmodel = GalacticModel(
            [{"IMF": imf}], star.config_manager
        )
    star.register_parameters(system)
    with pm.Model() as model:
        star.add_parameter(model=model, param_name="logmass", system=system)
    return np.atleast_1d(star.logmass.lower)


def test_ffp_leaves_the_default_logmass_support_alone():
    """
    Given a star that selects the FFP mass function,
    When its logmass parameter is built,
    Then the lower bound is still defaults.yaml's -9 dex.

    NOT an oversight.  Sub-stellar masses are this relation's domain, so a
    floor here would gate the very models it exists to express.  Contrast the
    Salpeter floor, which fixes a stellar IMF being used outside its domain
    of validity.
    """
    # Act
    lower = _logmass_lower(
        [{"name": "Lens", "mass_function": "ffp"}], imf="chabrier"
    )

    # Assert
    assert lower[0] == pytest.approx(_LOGMASS_LOWER)


def test_ffp_warns_that_the_lower_bound_is_a_modeling_choice(caplog):
    """
    Given a star selecting the FFP mass function with the default bound,
    When parameters are registered,
    Then a warning explains what the user needs to decide: that the density
      rises toward low mass, how concentrated the prior is at the bound, what
      -9 dex physically is, that they should set 'lower' themselves, and that
      bounds may only be tightened.
    """
    # Arrange
    star = _star([{"name": "Lens", "mass_function": "ffp"}])
    system = _DummySystem()
    system.config_manager = star.config_manager
    system.galacticmodel = GalacticModel([{}], star.config_manager)

    # Act
    with caplog.at_level(logging.WARNING):
        star.register_parameters(system)

    # Assert
    text = caplog.text
    assert "star.Lens.logmass" in text
    assert "RISES toward low mass" in text
    assert "2.21 nats per dex" in text  # alpha * ln10 at the default slope
    assert "1.04 dex" in text  # 90% of the mass, = 1/alpha
    assert "Ceres" in text
    assert "lower" in text
    assert "TIGHTENED" in text
    assert f"{FFP_LOGMASS_CALIBRATION_MIN:.4f}" in text


def test_no_bound_warning_when_the_user_already_set_lower(caplog):
    """
    Given a user who has set star.<name>.logmass's lower bound,
    When the FFP mass function is selected,
    Then no bound advisory is issued -- the decision has already been made,
      and warning anyway is how a codebase teaches people to ignore it.
    """
    # Arrange
    star = _star(
        [{"name": "Lens", "mass_function": "ffp"}],
        user_params={"star.Lens.logmass": {"lower": -7.0}},
    )
    system = _DummySystem()
    system.config_manager = star.config_manager
    system.galacticmodel = GalacticModel([{}], star.config_manager)

    # Act
    with caplog.at_level(logging.WARNING):
        star.register_parameters(system)

    # Assert
    assert "RISES toward low mass" not in caplog.text


def test_user_lower_bound_is_honored_under_ffp():
    """
    Given a user bound on an FFP star's logmass,
    When the parameter is built,
    Then it is exactly the user's -- nothing overrides it.
    """
    # Act
    lower = _logmass_lower(
        [{"name": "Lens", "mass_function": "ffp"}],
        imf="chabrier",
        user_params={"star.Lens.logmass": {"lower": -7.0}},
    )

    # Assert
    assert lower[0] == pytest.approx(-7.0)


def test_ffp_star_is_exempt_from_the_salpeter_floor(caplog):
    """
    Given IMF: Salpeter and one star opting into the FFP mass function,
    When the logmass bounds are resolved,
    Then only the stars drawing the stellar IMF are floored at the
      hydrogen-burning limit; the FFP star keeps the full support.

    The Salpeter floor exists because a STELLAR IMF has no claim below the
    hydrogen-burning limit.  A star that opted out of the stellar IMF must
    not inherit its floor -- that is the whole point of opting out.
    """
    # Act
    with caplog.at_level(logging.WARNING):
        lower = _logmass_lower(
            [
                {"name": "Lens", "mass_function": "ffp"},
                {"name": "Source"},
            ],
            imf="salpeter",
        )

    # Assert
    assert lower[0] == pytest.approx(_LOGMASS_LOWER)
    assert lower[1] == pytest.approx(_HBL_DEX)
    assert "star.Source.logmass" in caplog.text
    assert "raising the lower bound on star.Lens.logmass" not in caplog.text


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_end_to_end_mixed_system_builds_with_the_right_prior_on_each_star():
    """
    Given a full System with a stellar source and an FFP lens,
    When the model is built and the IMF potential evaluated at the start,
    Then the potential equals ffp(lens) + chabrier(source), and both the logp
      and its gradient are finite.

    The headline: this is the whole feature end to end, through the real
    config plumbing rather than a mock.
    """
    # Arrange
    from exozippy.system import System

    config = {
        "star": [
            {"name": "Lens", "mass_function": "ffp"},
            {"name": "Source"},
        ],
        "galacticmodel": [{"name": "gm"}],
    }
    user_params = {
        "star.Lens.ra": {"initval": 270.0, "sigma": 0},
        "star.Lens.dec": {"initval": -29.0, "sigma": 0},
        "star.Lens.logmass": {"initval": np.log10(3.0 / 1047.57)},
        "star.Source.ra": {"initval": 270.0, "sigma": 0},
        "star.Source.dec": {"initval": -29.0, "sigma": 0},
        "star.Source.logmass": {"initval": np.log10(1.1)},
    }

    # Act
    system = System(config, user_params)
    system.prepare()
    model = system.build_model()
    ip = model.initial_point()

    fn = pytensor.function(
        model.value_vars,
        [
            system.star.logmass.value,
            model.named_vars["galacticmodel.imf_prior"],
        ],
        on_unused_input="ignore",
    )
    values, potential = fn(*[ip[v.name] for v in model.value_vars])
    potential = float(potential)

    # Assert
    assert system.star.ffp_mask.tolist() == [True, False]
    lens_x, source_x = float(values[0]), float(values[1])
    expected = _ffp_closed_form(lens_x, FFP_MASS_FUNCTION_SLOPE) + float(
        np.atleast_1d(_chabrier_closed_form(source_x))[0]
    )
    assert potential == pytest.approx(expected, rel=1e-10)

    assert np.isfinite(float(np.asarray(model.compile_logp()(ip))))
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))

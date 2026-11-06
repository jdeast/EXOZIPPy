"""Tests for the free-floating-planet mass function.

The FFP mass function is selected per star (`mass_function: ffp` on a star
block) and replaces, for that star only, the stellar mass function the
galacticmodel block otherwise applies to every star (genulens's present-day
mass function, inside the kinematic mixture).  The FFP term alone is the
``galacticmodel.imf_prior`` potential; it is 0 for every other star.  See
`galacticmodel.ffp_logmass_logp` -- the single function the functional form
lives in -- and `Star._parse_mass_functions`.
"""

import logging

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from conftest import _DummyConfigManager, _DummySystem, _MockParam
from exozippy.components.galacticmodel import koshimoto21 as k21
from exozippy.components.galacticmodel.galacticmodel import (
    GalacticModel,
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

    def position_root(self, i):
        # No derived lens positions here (Star.position_root).
        return i


class _MockSystem:
    def __init__(self, star):
        self.star = star
        # No component models a star's mass loss (Component.initial_logmass).
        self.active_components = {}


def _potentials(star):
    """(imf_prior, kinematic_prior) numeric values for a mock star set."""
    gm = GalacticModel([{}], _DummyConfigManager())
    with pm.Model() as model:
        gm.build_likelihood(model, _MockSystem(star))
    return (
        float(model.named_vars["galacticmodel.imf_prior"].eval()),
        float(model.named_vars["galacticmodel.kinematic_prior"].eval()),
    )


def _imf_potential(star):
    """Numeric value of the galacticmodel.imf_prior potential for a star."""
    return _potentials(star)[0]


def _ffp_closed_form(x, alpha, lower=_LOGMASS_LOWER, upper=_LOGMASS_UPPER):
    """log p(x) for dN/dlogM ~ M^-alpha, normalized on [lower, upper].

    Written out independently of the implementation: p(x) ~ 10^(-alpha x), so
    Z = (10^(-alpha*upper) - 10^(-alpha*lower)) / (-alpha ln10).
    """
    k = -alpha
    z = (10.0 ** (k * upper) - 10.0 ** (k * lower)) / (k * _LN10)
    return k * _LN10 * x - np.log(z)


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
      with the stellar mass function normalized over the same support.
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
    Then K21's IMF (the stellar mass function's birth density) charges ~2.5
      nats for being 3 Mjup while the FFP function *prefers* it by ~5 nats
      -- a swing of ~7.5 nats, which is the penalty this feature removes.

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
    k21_imf = k21.log_imf(np.array([ffp_x, star_x]))

    # Assert
    assert ffp[0] - ffp[1] == pytest.approx(4.95, abs=0.05)
    assert k21_imf[0] - k21_imf[1] == pytest.approx(-2.50, abs=0.05)
    assert (ffp[0] - ffp[1]) - (k21_imf[0] - k21_imf[1]) > 7.0


# ---------------------------------------------------------------------------
# Per-star selection inside the galacticmodel potential
# ---------------------------------------------------------------------------


def test_default_path_is_untouched_by_the_ffp_branch():
    """
    Given stars that did not select a mass function (and a star object from
      before this feature existed, with no ffp_mask attribute at all),
    When the galactic potentials are built,
    Then both give identical values, and the imf_prior potential is exactly
      0: every star's mass prior is genulens's present-day mass function,
      inside the kinematic mixture.
    """
    # Arrange
    logmass = [np.log10(0.5), np.log10(0.3)]
    legacy = _MockStar(logmass)  # no ffp_mask attribute
    modern = _MockStar(logmass, ffp_mask=[False, False])

    # Act
    legacy_val = _potentials(legacy)
    modern_val = _potentials(modern)

    # Assert
    assert legacy_val == modern_val
    assert legacy_val[0] == 0.0
    assert np.isfinite(legacy_val[1])


def test_mixed_system_puts_the_right_prior_on_each_star():
    """
    Given a realistic microlensing system -- a STELLAR source and an
      FFP LENS -- with mass_function: ffp on the lens only,
    When the galactic potentials are built,
    Then imf_prior is exactly ffp(lens), the lens's mass does not reach the
      kinematic mixture, and the source's mass does: each star draws its own
      mass prior, and neither leaks into the other.

    This is why the selection has to be per star rather than a
    galacticmodel-level switch: one prior covers the whole star.logmass
    vector.
    """
    # Arrange
    lens_x = np.log10(3.0 / 1047.57)  # 3 Mjup FFP lens
    source_x = np.log10(1.1)  # ordinary bulge source star
    star = _MockStar([lens_x, source_x], ffp_mask=[True, False])

    # Act
    imf, kin = _potentials(star)
    _, kin_lens_moved = _potentials(
        _MockStar([lens_x - 1.0, source_x], ffp_mask=[True, False])
    )
    _, kin_source_moved = _potentials(
        _MockStar([lens_x, source_x - 0.3], ffp_mask=[True, False])
    )

    # Assert
    expected = float(
        np.atleast_1d(
            ffp_logmass_logp(lens_x, FFP_MASS_FUNCTION_SLOPE, star.logmass)
        )[0]
    )
    assert imf == pytest.approx(expected, rel=1e-12)
    assert kin_lens_moved == kin
    assert abs(kin_source_moved - kin) > 1e-3

    # ... and it is genuinely different from charging both the stellar
    # mass function
    both_imf, both_kin = _potentials(_MockStar([lens_x, source_x]))
    assert both_imf == 0.0
    assert abs((imf + kin) - (both_imf + both_kin)) > 1.0


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
    Then every star draws the stellar mass function and the FFP mask is all
      False.
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


def _logmass_lower(blocks, with_gm=False, user_params=None):
    """Resolved star.logmass lower bound(s) for the given star blocks."""
    star = _star(blocks, user_params)
    system = _DummySystem()
    system.config_manager = star.config_manager
    if with_gm:
        system.galacticmodel = GalacticModel([{}], star.config_manager)
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
    floor here would gate the very models it exists to express.
    """
    # Act
    lower = _logmass_lower(
        [{"name": "Lens", "mass_function": "ffp"}], with_gm=True
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
        with_gm=True,
        user_params={"star.Lens.logmass": {"lower": -7.0}},
    )

    # Assert
    assert lower[0] == pytest.approx(-7.0)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_mixed_system_builds_with_the_right_prior_on_each_star():
    """
    Given a full System with a stellar source and an FFP lens,
    When the model is built and the imf_prior potential evaluated at the
      start,
    Then the potential equals ffp(lens) alone (the source's mass prior is
      inside the kinematic mixture), and both the logp and its gradient are
      finite.

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
    lens_x = float(values[0])
    expected = _ffp_closed_form(lens_x, FFP_MASS_FUNCTION_SLOPE)
    assert potential == pytest.approx(expected, rel=1e-10)

    assert np.isfinite(float(np.asarray(model.compile_logp()(ip))))
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))

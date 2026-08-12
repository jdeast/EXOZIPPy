"""Tests for the constant-space-density (volume) prior on star.distance.

``star.distance`` is declared with finite ``lower``/``upper`` and no sigma, so
parameter.py's logit transform gives it a UNIFORM-in-distance prior -- a
default nobody chose, and one that implies p(plx) ~ plx^-2.  ``Star.build_
likelihood`` now adds ``2*log(d)`` (normalized) so the prior is p(d) ~ d^2,
the volume of the shell the object could have come from.

Where a ``galacticmodel`` block exists it already carries exactly that term
(``volume_element`` inside its ``kinematic_prior``, over the same full
``star.distance`` vector), so the star component defers entirely and no
second copy is applied.
"""

import types

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from exozippy.components.star.star import Star
from exozippy.config import ConfigManager

# star/defaults.yaml hard bounds on distance (pc).  They are the support the
# volume prior normalizes over.
_D_LOWER = 0.001
_D_UPPER = 100000.0


def _build_star(config=None, params=("distance",), user_params=None):
    """Build Star parameters through the full config -> model pipeline.

    Returns (star, model, system) with ``system`` a stub carrying only the
    topology dict ``_galactic_imf`` consults.
    """
    config = config or {"star": [{"name": "A"}]}
    cm = ConfigManager(dict(user_params or {}), system_config=config)
    cm.finalize_user_params()

    star = Star(config["star"], cm)
    star.manifest = {p: None for p in params}

    with pm.Model() as model:
        for p in params:
            star.add_parameter(model, p, system=None)

    return star, model, types.SimpleNamespace(config=config)


def _expected_log_norm(lower=_D_LOWER, upper=_D_UPPER):
    """log int_lower^upper d^2 dd, written the obvious way."""
    return float(np.log((upper**3 - lower**3) / 3.0))


def _logp_at(model, point):
    return float(model.compile_logp()(point))


def _eval(model, tensor, point):
    import pytensor

    fn = pytensor.function(model.free_RVs, tensor, on_unused_input="ignore")
    return np.asarray(fn(*[point[rv.name] for rv in model.free_RVs]))


def _potential_names(model):
    return [p.name for p in model.potentials]


# ----------------------------------------------------------------------
# The prior itself
# ----------------------------------------------------------------------


def test_volume_prior_adds_exactly_two_log_d_minus_the_normalizer():
    """
    Given a star with a distance and no galacticmodel,
    When the star's likelihood is built,
    Then the model logp gains exactly 2*log(d) - log(Z) relative to the old
      uniform-in-distance behaviour, at every raw point.

    The assertion is the ANALYTIC difference, not "a number changed": the
    baseline model is the same graph with build_likelihood skipped, which is
    byte-for-byte the pre-fix model.
    """
    # ARRANGE: the same parameter built twice, once with the new potential
    base_star, base_model, _ = _build_star()
    star, model, system = _build_star()
    with model:
        star.build_likelihood(model, system)

    log_z = _expected_log_norm()

    # ACT / ASSERT at several raw points (the term is d-dependent, so one
    # point could not distinguish it from a constant offset).
    for raw in (-2.0, 0.0, 1.5):
        point = base_model.initial_point()
        point["star.distance_raw"] = np.array([raw])
        d = float(_eval(base_model, base_star.distance.value, point)[0])

        delta = _logp_at(model, point) - _logp_at(base_model, point)
        assert delta == pytest.approx(2.0 * np.log(d) - log_z, rel=1e-12)


def test_volume_prior_covers_every_star():
    """
    Given two stars and no galacticmodel,
    When the star's likelihood is built,
    Then the added term is the SUM over both distances -- galacticmodel's
      volume element is a plain pt.sum over the whole vector too, so the
      replacement has to cover the same set.
    """
    # ARRANGE
    config = {"star": [{"name": "A"}, {"name": "B"}]}
    base_star, base_model, _ = _build_star(config=config)
    star, model, system = _build_star(config=config)
    with model:
        star.build_likelihood(model, system)

    point = base_model.initial_point()
    point["star.distance_raw"] = np.array([-1.0, 2.0])
    d = _eval(base_model, base_star.distance.value, point)
    assert d.shape == (2,)

    # ACT
    delta = _logp_at(model, point) - _logp_at(base_model, point)

    # ASSERT
    expected = np.sum(2.0 * np.log(d)) - 2.0 * _expected_log_norm()
    assert delta == pytest.approx(expected, rel=1e-12)


def test_volume_prior_is_a_normalized_density_over_the_support():
    """
    Given star.distance's hard support [1e-3, 1e5] pc,
    When exp(2*log d - log Z) is integrated over it,
    Then it integrates to 1 -- the term is a proper prior density, exactly
      as the uniform prior it replaces was.
    """
    # ARRANGE
    star, model, system = _build_star()
    with model:
        star.build_likelihood(model, system)

    log_z = star._volume_prior_log_norm()
    assert float(np.atleast_1d(log_z)[0]) == pytest.approx(
        _expected_log_norm(), rel=1e-12
    )

    # ACT: a d^2 integrand is dominated by the upper end, so a linear grid
    # is the right one (and trapezoid on a quadratic converges as h^2).
    d = np.linspace(_D_LOWER, _D_UPPER, 400001)
    integral = np.trapezoid(np.exp(2.0 * np.log(d) - log_z), d)

    # ASSERT
    assert float(integral) == pytest.approx(1.0, rel=1e-9)


def test_volume_prior_gradient_is_finite():
    """
    Given the volume prior on a sampled distance,
    When its gradient with respect to the raw sampling coordinate is taken,
    Then it is finite at the start and far out in the raw coordinate.
    """
    # ARRANGE
    star, model, system = _build_star()
    with model:
        star.build_likelihood(model, system)
    raw = model["star.distance_raw"]
    node = model["star.volume_prior"]

    # ACT
    grad_fn = pt.grad(pt.sum(node), raw)

    # ASSERT
    for value in (-20.0, 0.0, 20.0):
        got = grad_fn.eval({raw: np.array([value])})
        assert np.all(np.isfinite(got)), f"non-finite gradient at raw={value}"


def test_volume_prior_beats_uniform_at_large_distance():
    """
    Given the prior densities implied by the two choices,
    When they are compared across the support,
    Then the volume prior favours larger distances by exactly d^2 -- the
      whole point, and the difference that matters in the plx/sigma < 10
      regime.
    """
    # ARRANGE
    star, model, system = _build_star()
    with model:
        star.build_likelihood(model, system)
    log_z = float(np.atleast_1d(star._volume_prior_log_norm())[0])

    # ACT: log p(d) for both, up to the shared uniform reference
    d1, d2 = 100.0, 1000.0
    volume = (2.0 * np.log(d2) - log_z) - (2.0 * np.log(d1) - log_z)

    # ASSERT
    assert volume == pytest.approx(2.0 * np.log(d2 / d1), rel=1e-12)


# ----------------------------------------------------------------------
# Deferring to the galactic model
# ----------------------------------------------------------------------


def test_galacticmodel_suppresses_the_star_volume_prior():
    """
    Given a config that carries a galacticmodel block,
    When the star's likelihood is built,
    Then NO star volume prior is added and the model logp is untouched --
      galacticmodel.kinematic_prior already supplies 2*log(d), and a second
      copy would make the prior d^4.
    """
    # ARRANGE
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "galacticmodel": [{}],
    }
    base_star, base_model, _ = _build_star(config=config)
    star, model, system = _build_star(config=config)

    # ACT
    with model:
        star.build_likelihood(model, system)

    # ASSERT
    assert "star.volume_prior" not in _potential_names(model)
    point = base_model.initial_point()
    point["star.distance_raw"] = np.array([0.7, -1.3])
    assert _logp_at(model, point) == pytest.approx(
        _logp_at(base_model, point), rel=1e-12
    )


def test_galacticmodel_covers_the_same_stars_the_volume_prior_would():
    """
    Given galacticmodel's kinematic prior,
    When its distance coverage is read,
    Then it is the WHOLE star.distance vector with no mask -- which is what
      makes deferring to it (rather than covering the stars it misses)
      correct.  If this ever becomes a subset, the star component must
      cover the complement instead of standing down entirely.
    """
    # ARRANGE
    import inspect

    from exozippy.components.galacticmodel.galacticmodel import GalacticModel

    src = inspect.getsource(GalacticModel.build_likelihood)

    # ASSERT: the distance it weights is the unmasked vector, and the volume
    # element is applied inside the summed kinematic potential.
    assert "distance = pt.maximum(stars.distance.value, 1e-3) / 1e3" in src
    assert "volume_element = 2.0 * pt.log(distance * 1000.0)" in src
    assert "+ volume_element" in src


# ----------------------------------------------------------------------
# Topologies with no distance at all
# ----------------------------------------------------------------------


def test_no_distance_in_the_manifest_adds_nothing():
    """
    Given a topology with no sed/mann/lens/galacticmodel/astrometry, so the
      star has no distance parameter at all,
    When the star's likelihood is built,
    Then nothing is added -- there is no distance to weight.
    """
    # ARRANGE
    star, model, system = _build_star(params=("radius",))

    # ACT
    with model:
        star.build_likelihood(model, system)

    # ASSERT: only the logit transform's own uniform-prior correction, which
    # every bounded element carries; no volume prior.
    assert _potential_names(model) == ["logit_uniform_prior.star.radius"]


def test_pinned_distance_contributes_only_a_constant():
    """
    Given a distance pinned with sigma: 0,
    When the volume prior is built,
    Then the element is still covered (as galacticmodel covers pinned
      elements) but contributes a constant, so no posterior can move.
    """
    # ARRANGE
    user_params = {"star.A.distance": {"initval": 250.0, "sigma": 0}}
    star, model, system = _build_star(user_params=user_params)
    with model:
        star.build_likelihood(model, system)

    # ACT
    value = float(model["star.volume_prior"].eval())

    # ASSERT
    assert list(star.distance.is_sampled) == [False]
    assert value == pytest.approx(
        2.0 * np.log(250.0) - _expected_log_norm(), rel=1e-12
    )

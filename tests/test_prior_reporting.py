"""Component-added priors must reach the reported tables.

``Parameter.get_prior_str`` describes a prior from the parameter's OWN
fields -- ``sigma``, ``mu``, ``lower``/``upper`` -- so a ``pm.Potential`` a
component adds in stage 6 was invisible to it.  Three shipped priors were
misreported that way, every one of them as "Uniform", which is precisely the
prior they replace:

* ``star.distance``'s d^2 volume prior (or the galactic density/kinematic
  mixture where a galacticmodel block exists);
* ``star.logmass``'s stellar IMF (Chabrier or Salpeter);
* the free-floating-planet mass function, which swaps ONE star off that IMF.

Components now declare each contribution against the parameter it acts on
(``Parameter.add_prior_contribution``), and both report paths compose it
with the parameter's own fields: run.py's startup audit table asks
``get_prior_str(latex=False)`` and ``outputs/latex.py`` asks
``to_latex_prior_def()``, which is ``get_prior_str(latex=True)`` per element.
The seam is the registration call; ``parameter.py`` never learns a
component's name.
"""

import types

import numpy as np
import pymc as pm
import pytest

from exozippy.components.galacticmodel.galacticmodel import GalacticModel
from exozippy.components.parameter import Parameter
from exozippy.components.star.star import Star
from exozippy.config import ConfigManager

_RA_RAD = np.deg2rad(270.0)
_DEC_RAD = np.deg2rad(-29.0)


# ----------------------------------------------------------------------
# Harnesses (mirroring tests/test_distance_volume_prior.py and
# tests/test_galactic_model.py, which build the same two potentials)
# ----------------------------------------------------------------------


def _build_star(config=None, params=("distance",), user_params=None):
    """Star parameters through the full config -> model pipeline."""
    config = config or {"star": [{"name": "A"}]}
    cm = ConfigManager(dict(user_params or {}), system_config=config)
    cm.finalize_user_params()

    star = Star(config["star"], cm)
    star.manifest = {p: None for p in params}

    with pm.Model() as model:
        for p in params:
            star.add_parameter(model, p, system=None)

    return star, model, types.SimpleNamespace(config=config)


def _run_galacticmodel(star_config, gm_config=None):
    """Build the galacticmodel potentials over REAL star Parameters.

    Returns the Star component, whose Parameters now carry whatever the
    galacticmodel declared.
    """
    config = {
        "star": star_config,
        "galacticmodel": gm_config or [{"name": "gm"}],
    }
    star, _, _ = _build_star(
        config=config,
        params=("distance", "logmass", "pm_ra", "pm_dec", "rv", "ra", "dec"),
    )
    # build_likelihood reads ra/dec initvals for the line of sight.
    n = star.n_elements
    star.ra.initval = np.full(n, _RA_RAD)
    star.dec.initval = np.full(n, _DEC_RAD)

    gm = GalacticModel(config["galacticmodel"], star.config_manager)
    with pm.Model() as model:
        gm.build_likelihood(model, types.SimpleNamespace(star=star))
    return star


def _both(param, index=0):
    """(text, latex) renderings of one element's prior."""
    return (
        param.get_prior_str(index, latex=False),
        param.get_prior_str(index, latex=True),
    )


# ----------------------------------------------------------------------
# star.distance: the volume prior
# ----------------------------------------------------------------------


def test_distance_reports_the_volume_prior_not_uniform():
    """
    Given a star with a distance and no galacticmodel,
    When the star's likelihood adds the d^2 volume prior,
    Then both renderings of star.distance's prior name that prior and
      neither says Uniform.
    """
    # Arrange
    star, model, system = _build_star()
    before_text, before_latex = _both(star.distance)

    # Act
    with model:
        star.build_likelihood(model, system)
    text, latex = _both(star.distance)

    # Assert -- it really did say Uniform before, and does not now
    assert "U(" in before_text and r"\mathcal{U}" in before_latex
    assert "d^2" in text
    assert "d^{2}" in latex
    assert "U(" not in text
    assert r"\mathcal{U}" not in latex


def test_volume_prior_keeps_the_support_it_normalizes_over():
    """
    Given the volume prior, which is normalized over star.distance's own
      hard bounds,
    When the prior is rendered,
    Then the bounds are still quoted -- the replacement text must not be
      less informative than the "Uniform(lower, upper)" it replaces.
    """
    # Arrange / Act
    star, model, system = _build_star()
    with model:
        star.build_likelihood(model, system)

    # Assert -- the term AND the interval it is a density over (out-of-range
    # magnitudes render as powers of ten: '1e5' in text, '10^{5}' in LaTeX)
    text, latex = _both(star.distance)
    assert "d^2" in text and "0.001" in text and "1e5" in text
    assert "d^{2}" in latex and "0.001" in latex and "10^{5}" in latex


def test_distance_reports_both_a_user_gaussian_and_the_volume_prior():
    """
    Given a user Gaussian on star.distance (a parallax measurement),
    When the volume prior is added on top of it,
    Then the rendered prior states BOTH -- the Gaussian and the d^2 term --
      because the posterior really is their product.
    """
    # Arrange
    user = {"star.0.distance": {"initval": 100.0, "sigma": 5.0}}
    star, model, system = _build_star(user_params=user)

    # Act
    with model:
        star.build_likelihood(model, system)
    text, latex = _both(star.distance)

    # Assert
    assert "N(100, 5)" in text
    assert r"\mathcal{N}(100, 5)" in latex
    assert "d^2" in text
    assert "d^{2}" in latex


def test_distance_under_a_galacticmodel_reports_the_galactic_model():
    """
    Given a galacticmodel block, to which the star component defers its
      distance prior entirely,
    When the galactic density/kinematic potential is built,
    Then star.distance is reported as the galactic model, not as Uniform
      and not as the d^2 volume prior the star component would have added.
    """
    # Arrange / Act
    star = _run_galacticmodel([{"name": "Lens"}])

    # Assert
    text, latex = _both(star.distance)
    assert "Galactic" in text and "Galactic" in latex
    assert "U(" not in text
    assert r"\mathcal{U}" not in latex


def test_star_component_stands_down_and_declares_nothing_under_a_galacticmodel():
    """
    Given a galacticmodel block,
    When Star.build_likelihood runs,
    Then it adds neither the potential nor a declaration -- the two would
      otherwise both be reported and the prior would read as d^4.
    """
    # Arrange
    config = {"star": [{"name": "A"}], "galacticmodel": [{"name": "gm"}]}
    star, model, system = _build_star(config=config)

    # Act
    with model:
        star.build_likelihood(model, system)

    # Assert
    assert star.distance.prior_contributions == []
    assert not [
        pot.name
        for pot in model.potentials
        if "volume_prior" in (pot.name or "")
    ]


# ----------------------------------------------------------------------
# star.logmass: the IMF and the FFP mass function
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "imf,expected",
    [(None, "Chabrier"), ("chabrier", "Chabrier"), ("Salpeter", "Salpeter")],
)
def test_logmass_reports_the_imf_actually_applied(imf, expected):
    """
    Given a galacticmodel whose IMF: key selects a stellar mass function,
    When its imf_prior potential is built,
    Then star.logmass is reported as that IMF, by name, and not as Uniform.
    """
    # Arrange
    gm_config = [{"name": "gm"} if imf is None else {"name": "gm", "IMF": imf}]

    # Act
    star = _run_galacticmodel([{"name": "A"}], gm_config=gm_config)

    # Assert
    text, latex = _both(star.logmass)
    assert expected in text and expected in latex
    assert "U(" not in text
    assert r"\mathcal{U}" not in latex


def test_ffp_star_reports_the_ffp_mass_function_and_its_neighbour_the_imf():
    """
    Given two stars, one with `mass_function: ffp`,
    When the imf_prior potential is built,
    Then the FFP star's logmass element reports the FFP mass function and
      the other reports the stellar IMF.

    The contribution is per ELEMENT because the choice is: one pt.sum over
    the whole logmass vector carries two different densities.
    """
    # Arrange / Act
    star = _run_galacticmodel(
        [{"name": "Source"}, {"name": "Lens", "mass_function": "ffp"}]
    )

    # Assert
    imf_text, imf_latex = _both(star.logmass, index=0)
    ffp_text, ffp_latex = _both(star.logmass, index=1)
    assert "Chabrier" in imf_text and "Chabrier" in imf_latex
    assert "FFP" in ffp_text and "FFP" in ffp_latex
    assert "FFP" not in imf_text
    assert "Chabrier" not in ffp_text


def test_ffp_contribution_quotes_the_slope_actually_used():
    """
    Given `mass_function: {kind: ffp, alpha: 1.25}`,
    When the prior is rendered,
    Then the reported slope is the one the potential used, not the
      published default.
    """
    # Arrange / Act
    star = _run_galacticmodel(
        [{"name": "Lens", "mass_function": {"kind": "ffp", "alpha": 1.25}}]
    )

    # Assert
    text, _ = _both(star.logmass)
    assert "1.25" in text


# ----------------------------------------------------------------------
# The mechanism itself
# ----------------------------------------------------------------------


def test_parameter_with_no_contribution_is_byte_for_byte_unchanged():
    """
    Given parameters covering every branch of the prior description,
    When no component has declared anything against them,
    Then get_prior_str returns exactly what the pre-existing branch returns.

    The composition step must be inert unless something opted in.
    """
    # Arrange
    cases = [
        Parameter(label="a.b", initval=1.0, sigma=0.0),
        Parameter(label="a.c", initval=1.0, mu=1.0, sigma=0.5),
        Parameter(label="a.d", initval=1.0, lower=0.0, upper=2.0),
        Parameter(label="a.e", initval=1.0, lower=0.0),
        Parameter(label="a.f", initval=1.0, expression=lambda: None),
    ]

    # Act / Assert
    for p in cases:
        for latex in (False, True):
            assert (
                p.get_prior_str(latex=latex)
                == p._own_prior_str(latex=latex)[0]
            )


def test_declaring_the_same_contribution_twice_is_a_no_op():
    """
    Given a parameter that has already been told about a contribution,
    When the identical contribution is declared again (a second
      build_model() on one System, as the GUI does),
    Then it is recorded once and the rendered text says it once.
    """
    # Arrange
    p = Parameter(label="a.b", initval=1.0, lower=0.0, upper=2.0)

    # Act
    p.add_prior_contribution("Some prior", supersedes_bounds=True)
    p.add_prior_contribution("Some prior", supersedes_bounds=True)

    # Assert
    assert len(p.prior_contributions) == 1
    assert p.get_prior_str(latex=False).count("Some prior") == 1


def test_a_pinned_element_still_reports_fixed():
    """
    Given a pinned element (sigma = 0) covered by a component potential,
    When its prior is rendered,
    Then it still reports Fixed: the potential is a constant there and
      cannot move the posterior, so Fixed remains the whole story.
    """
    # Arrange
    p = Parameter(label="a.b", initval=1.0, sigma=0.0, lower=0.0, upper=2.0)

    # Act
    p.add_prior_contribution("Some prior", supersedes_bounds=True)

    # Assert
    assert p.get_prior_str(latex=False) == "Fixed"


def test_a_multiplying_contribution_keeps_the_parameters_own_bounds():
    """
    Given a contribution that does NOT supersede the bounds (it multiplies
      whatever prior the parameter already has),
    When the prior is rendered,
    Then both the uniform and the contribution appear.
    """
    # Arrange
    p = Parameter(label="a.b", initval=1.0, lower=0.0, upper=2.0)

    # Act
    p.add_prior_contribution("Extra term")

    # Assert
    text = p.get_prior_str(latex=False)
    assert text.startswith("U(")
    assert "2)" in text
    assert "Extra term" in text


def test_latex_macro_carries_the_contribution_per_element():
    """
    Given a two-element parameter whose elements carry different
      contributions,
    When the LaTeX prior macros are emitted,
    Then each element's macro carries its own contribution.

    This is the path outputs/latex.py takes (\\...prior per element); the
    text path run.py takes is the same function with latex=False, so a
    contribution cannot reach one report and miss the other.
    """
    # Arrange
    p = Parameter(
        label="a.b",
        initval=np.array([1.0, 1.0]),
        lower=0.0,
        upper=2.0,
        shape=(2,),
    )
    p.add_prior_contribution("First prior", elements=[0])
    p.add_prior_contribution("Second prior", elements=[1])

    # Act
    macros = p.to_latex_prior_def()

    # Assert
    lines = [ln for ln in macros.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "First prior" in lines[0] and "Second prior" not in lines[0]
    assert "Second prior" in lines[1] and "First prior" not in lines[1]


def test_boolean_masks_select_elements():
    """
    Given a contribution declared with a boolean mask,
    When each element's prior is rendered,
    Then only the masked elements carry it.
    """
    # Arrange
    p = Parameter(
        label="a.b",
        initval=np.array([1.0, 1.0, 1.0]),
        lower=0.0,
        upper=2.0,
        shape=(3,),
    )

    # Act
    p.add_prior_contribution("Masked", elements=np.array([False, True, True]))

    # Assert
    assert "Masked" not in p.get_prior_str(0, latex=False)
    assert "Masked" in p.get_prior_str(1, latex=False)
    assert "Masked" in p.get_prior_str(2, latex=False)

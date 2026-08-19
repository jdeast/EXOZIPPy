"""star.radius/teff/feh exist only where something reads them (review 3.8.1).

The rule these tests defend, in three tiers:

1. INACTIVE -- the quantity is not a parameter of this configuration.  It
   supplies no value anything reads, so nothing about it is arbitrary.  Not
   user-overridable, and should not be.
2. ``pin_unselected`` through the ``"overrides"`` channel -- pinned by default
   but layered UNDER the params file, for a parameter that exists for every
   instance and is wanted only on some.
3. FREE -- anything the model reads.  Never given a value by the code.

The defect was that a microlensing source star's radius sat in tier 2 while,
under ``finite_source``, ``rho``'s deps genuinely READ it -- so a light curve's
finite-source signal was measured against a radius nobody had chosen (the
untouched 1.0 solRad default).  It is tier 3 now.  Its siblings -- the LENS
radius, and every star's teff and feh in a point-source fit -- are read by
nothing at all and moved the other way, into tier 1.

The predicate is deliberately over the whole TOPOLOGY rather than over
microlensing: readership is not a microlensing fact, which is why the old
microlensing-only predicate pinned the source star's radius and left the lens
star's, equally unread, free.
"""

import logging

import numpy as np
import pytest

from exozippy.manifest import interpret_manifest_entry
from exozippy.system import System


def _sed_block(tmp_path):
    """An `sed:` block carrying one catalog row.

    The row's numbers are irrelevant to what is being tested.  What makes an
    SED read every star's radius/teff/feh is not its photometry but its
    teffsed and fbolsed floor potentials, which are pt.sum over the WHOLE star
    vector with no mask -- and fbol is calc_fbol(luminosity(radius, teff)).
    """
    sed_file = tmp_path / "one_row.sed"
    sed_file.write_text(
        "model: NextGen\n"
        "filters:\n"
        '    - name: "2MASS/2MASS.J"\n'
        "      mag: 15.0\n"
        "      err: 0.03\n"
    )
    return {"file": str(sed_file)}


def _mulens_config(finite_source=False, extra=None):
    """A point-source (or finite-source) microlensing topology, no data."""
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [
            {
                "name": "L",
                "lens_ndx": 0,
                "source_ndx": 1,
                "finite_source": finite_source,
            }
        ],
        "galacticmodel": [{"name": "gm", "anchor_idx": 1}],
    }
    config.update(extra or {})
    return config


def _star(config, user_params=None):
    """The prepared star component of `config` (stage 3 has run)."""
    system = System(config, user_params or {})
    system.prepare()
    return system.star


def _mask(star, param):
    """The activity mask star.manifest declares for `param`, or None."""
    entry = star.manifest.get(param)
    if entry is None and param not in star.manifest:
        return None
    options = interpret_manifest_entry(entry).options
    mask = options.get("mask")
    return None if mask is None else [bool(m) for m in mask]


# ----------------------------------------------------------------------
# Tier 1: not a parameter of this configuration
# ----------------------------------------------------------------------
@pytest.mark.parametrize("param", ["radius", "teff", "feh"])
def test_point_source_microlensing_declares_no_structure_parameters(param):
    """
    Given a point-source microlensing fit with no SED, no evolutionary model
      and no empirical relation,
    When the star component registers its parameters,
    Then every star's radius, teff and feh is inactive -- nothing reads them,
      so free they would refill their prior and pinned they would change
      nothing.
    """
    # Arrange / Act
    star = _star(_mulens_config())

    # Assert -- both stars, the LENS included.  The lens star is the half the
    # old microlensing-only predicate could never reach.
    assert _mask(star, param) == [False, False]


def test_inactive_is_structural_and_not_reachable_by_a_pin():
    """
    Given the same fit,
    When the manifest entry is read,
    Then the deactivation is in the `mask` channel and NOT in `overrides`.

    The channel is the tier.  `overrides` layers under the params file and so
    would still be offering the user a value the code chose; `mask` says the
    element is not part of this model, which is a different claim and the one
    being made here.
    """
    # Arrange / Act
    entry = interpret_manifest_entry(_star(_mulens_config()).manifest["teff"])

    # Assert
    assert entry.options.get("mask") is not None
    assert "sigma" not in (entry.options.get("overrides") or {})


# ----------------------------------------------------------------------
# Tier 3: anything the model reads
# ----------------------------------------------------------------------
def test_finite_source_frees_the_source_radius_but_not_the_lens_radius():
    """
    Given the same fit with finite_source turned on,
    When the star component registers its parameters,
    Then the SOURCE star's radius is active (rho's deps read it) while the
      LENS star's is not.

    This is the per-star half of the ruling: the two stars want opposite
    answers for the same parameter, which a whole-vector pin cannot express.
    """
    # Arrange / Act
    star = _star(_mulens_config(finite_source=True))

    # Assert
    assert _mask(star, "radius") == [False, True]
    # teff/feh are unmoved -- finite_source reads the radius, nothing else.
    assert _mask(star, "teff") == [False, False]
    assert _mask(star, "feh") == [False, False]


def test_finite_source_warns_that_the_freed_radius_is_degenerate(caplog):
    """
    Given a finite-source fit with nothing else constraining the source size,
    When the star component registers its parameters,
    Then it warns that the radius is not separately identifiable and names the
      ways to break the degeneracy.

    Rope, not gates: the degeneracy is real (theta_E and d_S absorb any
    rescaling), but a finite-source measurement genuinely bounds rho, hence the
    radius, from above -- so it is fit and the caveat is stated, rather than
    pinned at a number the code picked.
    """
    # Arrange / Act
    with caplog.at_level(logging.WARNING):
        _star(_mulens_config(finite_source=True))

    # Assert
    text = caplog.text
    assert "not separately identifiable" in text
    assert "Source" in text
    for remedy in ("sed:", "mann", "torres"):
        assert remedy in text


def test_a_user_prior_makes_the_parameter_read():
    """
    Given a point-source fit whose params file puts a spectroscopic prior on
      the source teff,
    When the star component registers its parameters,
    Then that element stays active.

    A `mu`/`sigma` is a term in the logp -- the user asserting a measurement --
    so it makes the parameter read by definition.  Deactivating it would DROP
    that prior and stop reporting a quantity the user measured, which is why
    readership cannot be answered from the component topology alone.  Five
    shipped examples (kelt17, hd80606, GaiaBH1, HIP1349, kelt4) depend on this.
    """
    # Arrange / Act
    star = _star(
        _mulens_config(),
        {"star.Source.teff": {"initval": 5800.0, "sigma": 120.0}},
    )

    # Assert -- only the element the user claimed.
    assert _mask(star, "teff") == [False, True]


def test_a_user_sigma_zero_is_a_pin_and_does_not_reactivate():
    """
    Given a params file that pins the teff outright (`sigma: 0`),
    When the star component registers its parameters,
    Then the element is still inactive.

    `sigma: 0` is a pin, and the inactive role subsumes it exactly -- the value
    is held, nothing is sampled, no prior applies.  Honoring it as a "user
    prior" would make this whole fix a no-op on the very files it exists for:
    both shipped microlensing params files pinned all six that way by hand.
    """
    # Arrange / Act
    star = _star(
        _mulens_config(), {"star.Source.teff": {"initval": 5800.0, "sigma": 0}}
    )

    # Assert
    assert _mask(star, "teff") == [False, False]


# ----------------------------------------------------------------------
# The mask is computed from the CONFIG
# ----------------------------------------------------------------------
@pytest.mark.parametrize("param", ["radius", "teff", "feh"])
def test_adding_an_sed_block_reactivates_everything(param, tmp_path):
    """
    Given the same point-source fit plus an `sed:` block,
    When the star component registers its parameters,
    Then all three are active again for every star, with no user action.

    The SED's teffsed and fbolsed floor potentials are pt.sum over the WHOLE
    star vector with no mask, and fbol is calc_fbol(luminosity(radius, teff)),
    so an SED reads every star's radius and teff; _predicted_appmag_node reads
    the whole feh vector.  They are only weakly constrained -- and finding out
    how weakly is the point of letting them be fit.
    """
    # Arrange
    config = _mulens_config(extra={"sed": _sed_block(tmp_path)})

    # Act
    star = _star(config)

    # Assert -- no mask at all, i.e. the historical whole-vector manifest.
    assert _mask(star, param) is None


def test_the_consumer_predicate_names_who_reads_what(tmp_path):
    """
    Given a finite-source fit with an SED,
    When structure_consumers is asked,
    Then it returns one record per (reader, parameter, star), naming both.

    The predicate is the seam: a new consumer of a stellar structure parameter
    is registered here and the mask, the log line and the degeneracy warning
    all follow.  Answering "which stars" and "which readers" separately is how
    one of them gets forgotten.
    """
    # Arrange
    config = _mulens_config(
        finite_source=True, extra={"sed": _sed_block(tmp_path)}
    )
    system = System(config, {})
    system.prepare()

    # Act
    consumers = system.star.structure_consumers(system)

    # Assert
    labels = {c.label for c in consumers}
    assert "sed" in labels
    assert "lens(finite_source)" in labels
    # The finite source reads the SOURCE's radius, star 1, and only that.
    fs = {(c.param, c.star) for c in consumers if c.label.startswith("lens(")}
    assert fs == {("radius", 1)}


def test_planet_reads_its_host_radius_even_for_a_single_planet():
    """
    Given one planet in one orbit around star 0,
    When the star component registers its parameters,
    Then that star's radius is active.

    A regression pin with a specific cause: planet.star_map is a numpy array,
    and `stars or []` on a 1-element array reads the ELEMENT -- so array([0])
    tests False and the whole consumer vanishes.  It did, for one commit, and
    it deactivated the star radius that planet.p = radius/star.radius reads in
    every single-planet transit and RV fit.
    """
    # Arrange
    config = {
        "star": [{"name": "A"}],
        "planet": [{"name": "b", "star_ndx": 0, "orbit_ndx": 0}],
        "orbit": [{"name": "b"}],
    }

    # Act
    star = _star(config)

    # Assert
    assert _mask(star, "radius") in (None, [True])


def test_a_parameter_no_star_uses_is_still_declared():
    """
    Given a point-source fit where NO star's radius is read,
    When the manifest is read,
    Then `radius` is still declared, wholly inactive, rather than dropped.

    This deliberately departs from mode_manifest's "omit a parameter no
    instance uses" rule: star.logg, star.density and star.luminosity name
    radius (and teff) in their own deps, so dropping either would be a
    build-graph dependency error rather than a saving.
    """
    # Arrange / Act
    star = _star(_mulens_config())

    # Assert
    for param in ("radius", "teff"):
        assert param in star.manifest
        assert not np.any(_mask(star, param))

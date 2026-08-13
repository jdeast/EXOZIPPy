"""Tests for validate_sigma_has_center (config.py).

A user 'sigma' > 0 asks for a Gaussian prior.  With neither 'mu' nor 'initval'
given, Parameter.build_pymc centers that prior on whatever start value the
system resolved -- frequently one derived FROM THE DATA (a component data hint,
a relaxation-engine solution, a start value mkparam seeded from a previous
fit's MAP).  Centering a prior on the data's own best fit double-counts the
data, so the config is rejected outright rather than silently sampled.
"""

import pytest

from exozippy.config import ConfigManager, validate_sigma_has_center

CONFIG = {"star": [{"name": "A"}, {"name": "B"}]}


def test_sigma_without_center_is_fatal():
    """
    Given a params entry with sigma > 0 and neither mu nor initval,
    When the ConfigManager is constructed,
    Then a ValueError names the parameter and explains both fixes.
    """
    # ARRANGE
    user_params = {"star.A.teff": {"sigma": 100.0}}

    # ACT / ASSERT
    with pytest.raises(ValueError) as exc:
        ConfigManager(user_params, system_config=CONFIG)

    msg = str(exc.value)
    assert "star.0.teff" in msg, "the offending parameter must be named"
    assert "double-count" in msg, "the reason must be stated"
    assert "'mu'" in msg and "sigma: 0" in msg, "both fixes must be offered"


def test_sigma_zero_alone_is_legal():
    """
    Given a params entry with sigma == 0 and no mu/initval,
    When the ConfigManager is constructed,
    Then it succeeds -- sigma 0 is a fixed pin ("hold this at whatever it
      resolves to"), not a Gaussian prior, so nothing is double-counted.
    """
    # ARRANGE / ACT
    cm = ConfigManager(
        {"star.A.teff": {"sigma": 0}, "star.B.teff": {"sigma": 0.0}},
        system_config=CONFIG,
    )

    # ASSERT
    assert cm.user_params["star.0.teff"]["sigma"] == 0


def test_explicit_mu_plus_sigma_is_legal():
    """
    Given a params entry with an explicit mu and sigma > 0,
    When the ConfigManager is constructed,
    Then it succeeds -- the prior center is stated independently.
    """
    # ARRANGE / ACT
    cm = ConfigManager(
        {"star.A.teff": {"mu": 5800.0, "sigma": 100.0}}, system_config=CONFIG
    )

    # ASSERT
    assert cm.user_params["star.0.teff"]["mu"] == 5800.0


def test_initval_plus_sigma_is_legal():
    """
    Given a params entry with initval and sigma > 0 (the common shorthand for
      "prior centered here"),
    When the ConfigManager is constructed,
    Then it succeeds.
    """
    # ARRANGE / ACT
    cm = ConfigManager(
        {"star.A.teff": {"initval": 5800.0, "sigma": 100.0}},
        system_config=CONFIG,
    )

    # ASSERT
    assert cm.user_params["star.0.teff"]["initval"] == 5800.0


def test_linked_mu_plus_sigma_is_legal():
    """
    Given a params entry whose mu is a LINK expression plus sigma > 0,
    When the ConfigManager is constructed,
    Then it succeeds.

    Regression guard: extract_links deletes the link string from the entry, so
    a check reading only user_params would see no mu and hard-error on a legal
    file.  The validator must consult ConfigManager.links.
    """
    # ARRANGE / ACT
    cm = ConfigManager(
        {
            "star.B.teff": {"initval": 5000.0},
            "star.A.teff": {"mu": "star.B.teff", "sigma": 100.0},
        },
        system_config=CONFIG,
    )

    # ASSERT -- the link really was extracted (so the entry no longer has mu)
    assert "mu" in cm.links["star.0.teff"]
    assert "mu" not in cm.user_params["star.0.teff"]


def test_linked_initval_plus_sigma_is_legal():
    """
    Given a params entry whose initval is a link expression plus sigma > 0,
    When the ConfigManager is constructed,
    Then it succeeds (a linked initval states a center just as mu does).
    """
    # ARRANGE / ACT
    cm = ConfigManager(
        {
            "star.B.teff": {"initval": 5000.0},
            "star.A.teff": {"initval": "star.B.teff + 100", "sigma": 50.0},
        },
        system_config=CONFIG,
    )

    # ASSERT
    assert "initval" in cm.links["star.0.teff"]


def test_linked_sigma_without_center_is_fatal():
    """
    Given a params entry whose sigma is a link expression and which has no
      mu or initval,
    When the ConfigManager is constructed,
    Then it still raises -- a dynamic width does not excuse an unstated center.
    """
    # ARRANGE
    user_params = {
        "star.B.age": {"initval": 4.0},
        "star.A.age": {"sigma": "0.5 * star.B.age"},
    }

    # ACT / ASSERT
    with pytest.raises(ValueError, match="star.0.age"):
        ConfigManager(user_params, system_config=CONFIG)


def test_all_zero_sigma_list_is_legal():
    """
    Given a per-element sigma list that is all zeros,
    When validated,
    Then it passes (every element is a pin), while a list containing a
      non-zero width with no center is rejected.
    """
    # ARRANGE / ACT / ASSERT
    validate_sigma_has_center({"star.0.teff": {"sigma": [0.0, 0]}})

    with pytest.raises(ValueError, match="star.0.teff"):
        validate_sigma_has_center({"star.0.teff": {"sigma": [0.0, 0.5]}})


def test_absent_and_null_sigma_and_non_dict_entries_are_skipped():
    """
    Given entries with no sigma, an explicit null sigma, and a bare scalar,
    When validated,
    Then none of them is flagged (no sigma means no prior to center).
    """
    # ARRANGE / ACT / ASSERT -- must not raise
    validate_sigma_has_center(
        {
            "star.0.teff": {"lower": 3000.0, "upper": 9000.0},
            "star.0.mass": {"sigma": None},
            "star.0.radius": 1.0,
        }
    )


def test_bounds_alone_do_not_count_as_a_center():
    """
    Given a params entry with sigma > 0 and only lower/upper bounds,
    When validated,
    Then it is still rejected -- a bound is not a prior center.
    """
    # ARRANGE / ACT / ASSERT
    with pytest.raises(ValueError, match="star.0.teff"):
        validate_sigma_has_center(
            {"star.0.teff": {"sigma": 100.0, "lower": 3000.0, "upper": 9000.0}}
        )


def test_error_names_every_offender():
    """
    Given several offending entries,
    When validated,
    Then the single error names all of them, so the user fixes the file once.
    """
    # ARRANGE / ACT
    with pytest.raises(ValueError) as exc:
        validate_sigma_has_center(
            {
                "star.0.teff": {"sigma": 100.0},
                "star.1.feh": {"sigma": 0.08},
                "star.0.mass": {"initval": 1.0, "sigma": 0.05},
            }
        )

    # ASSERT
    msg = str(exc.value)
    assert "star.0.teff" in msg and "star.1.feh" in msg
    assert "star.0.mass" not in msg, "a centered entry must not be flagged"


def test_source_file_is_named_when_given():
    """
    Given a source label,
    When validation fails,
    Then the message quotes the file so the user knows what to edit.
    """
    # ARRANGE / ACT / ASSERT
    with pytest.raises(ValueError, match="my.params.yaml"):
        validate_sigma_has_center(
            {"star.0.teff": {"sigma": 100.0}}, source="my.params.yaml"
        )

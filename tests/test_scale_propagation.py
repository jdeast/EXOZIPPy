"""Forward scale propagation and the retirement of user-side init_scale.

The backward (inverse-Jacobian) scale pass is gone: sampled-parameter
scales are only preliminary now -- the whitening probe measures the real
ones from the data (tests/test_whitening.py).  What remains load-bearing is
the FORWARD Jacobian pass, which fills in derived-parameter scales because
those set the soft-bound barrier steepness on derived parameters, and the
strip-with-warning of any user init_scale.

Uses ConfigManager directly to avoid loading data files.  The star
component's mass = 10^logmass relation gives a clean 1-parent case.
"""

import logging

import numpy as np
import pytest

from exozippy.config import ConfigManager

_STAR_CONFIG = {"star": [{"name": "Lens"}]}


def _make_cm(extra_user_params):
    cm = ConfigManager(extra_user_params, system_config=_STAR_CONFIG)
    cm.finalize_user_params()
    return cm


def test_forward_pass_fills_derived_scale_from_parent():
    """
    Given a user Gaussian prior (sigma, RANK_USER) on sampled logmass,
    When the relaxation engine's forward Jacobian pass runs,
    Then mass (derived, mass = 10^logmass) gets a propagated scale of
    dmass/dlogmass * sigma = mass * ln10 * sigma, beating mass's own
    defaults.yaml scale (higher rank wins) -- this is what sets the
    soft-bound barrier steepness on the derived parameter.
    """
    # Arrange / Act
    logmass_initval = -0.3  # mass ~ 0.5 M_sun
    sigma = 0.02
    cm = _make_cm(
        {
            "star.Lens.logmass": {
                "initval": logmass_initval,
                "mu": logmass_initval,
                "sigma": sigma,
            }
        }
    )

    # Assert
    mass_scale = cm.propagated_scales.get("star.0.mass")
    assert mass_scale is not None
    expected = (10**logmass_initval) * np.log(10) * sigma
    assert mass_scale == pytest.approx(expected, rel=0.05)


def test_user_init_scale_on_derived_is_stripped_not_backpropagated(caplog):
    """
    Given the user puts init_scale on the derived 'mass',
    When the ConfigManager is constructed and solved,
    Then the key is stripped with a warning and the parent logmass scale is
    IDENTICAL to a run without it (no backward propagation happens).
    """
    # Arrange: reference run without any user init_scale
    base = _make_cm(
        {
            "star.Lens.logmass": {"initval": -0.3},
            "star.Lens.mass": {"initval": 10**-0.3},
        }
    )
    base_scale = base.propagated_scales.get("star.0.logmass")

    # Act
    with caplog.at_level(logging.WARNING):
        cm = _make_cm(
            {
                "star.Lens.logmass": {"initval": -0.3},
                "star.Lens.mass": {"initval": 10**-0.3, "init_scale": 0.05},
            }
        )

    # Assert
    assert "init_scale" not in cm.user_params["star.0.mass"]
    assert any("obsolete" in r.message for r in caplog.records)
    assert cm.propagated_scales.get("star.0.logmass") == pytest.approx(
        base_scale
    )


def test_user_sigma_still_seeds_the_scale():
    """
    Given the user puts a Gaussian prior (sigma) on sampled logmass,
    When the engine runs,
    Then that sigma is used as logmass's preliminary scale (RANK_USER) --
    sigma stays user-facing even though init_scale does not.
    """
    # Arrange / Act
    cm = _make_cm(
        {"star.Lens.logmass": {"initval": -0.3, "mu": -0.3, "sigma": 0.02}}
    )

    # Assert
    assert cm.propagated_scales.get("star.0.logmass") == pytest.approx(
        0.02, rel=0.01
    )

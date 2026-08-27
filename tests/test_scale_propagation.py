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
    Given a user Gaussian prior (sigma, PRECEDENCE_USER) on sampled logmass,
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
    Then that sigma is used as logmass's preliminary scale (PRECEDENCE_USER) --
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


# ---------------------------------------------------------------------------
# The scale sync back into user_params when there is no scale to sync
# ---------------------------------------------------------------------------

# star + planet + orbit is the smallest topology carrying Kepler's third law,
# a**3 = C * m_total * period**2.  None of a, m_total or period has an
# init_scale in defaults.yaml -- init_scale is optional -- so the forward
# Jacobian pass scores the solved a scale at rank 0, which loses to a's
# own (absent) rank-0 scale and is therefore never stored.  Naming a in a
# params file then used to send the sync step reading a key that was never
# written: KeyError straight out of prepare().
_ORBIT_CONFIG = {
    "star": [{"name": "A"}],
    "planet": [{"name": "b"}],
    "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
}


def test_scaleless_solved_target_does_not_crash_the_engine():
    """
    Given a params entry on a derived parameter (orbit a) whose relation
    parents carry no init_scale from any source,
    When the relaxation engine solves it (stage 4 of prepare),
    Then the solve completes and the entry simply gets no init_scale.

    The user period entry is load-bearing: an initval with no sigma pins the
    VALUE at PRECEDENCE_USER while leaving period scale-less, which is what drives
    the propagated scale rank to zero.
    """
    # Arrange
    cm = ConfigManager(
        {
            "orbit.b.period": {"initval": 2.9895933},
            "orbit.b.a": {"lower": 0.0, "upper": 1000.0},
        },
        system_config=_ORBIT_CONFIG,
    )

    # Act
    cm.finalize_user_params()

    # Assert: the solve ran and a got a start value ...
    entry = cm.user_params["orbit.0.a"]
    assert entry["initval"] is not None
    # ... but no preliminary scale, which is the handled state: build_pymc
    # falls back to a fraction of the bound span and the whitening probe
    # measures the real scale from the data.
    assert "init_scale" not in entry
    assert cm.propagated_scales.get("orbit.0.a") is None

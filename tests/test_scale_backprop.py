"""
Regression tests for backward scale propagation through derived parameters.

Bug fixed: init_scale on a derived parameter was silently ignored.
Fix: the relaxation engine back-propagates via the inverse Jacobian so that
the user's known scale on a derived quantity (e.g. mass from a prior fit)
informs the proposal scales on the sampled parents (logmass, …).

Uses ConfigManager directly to avoid loading data files.
"""
import logging
import numpy as np
import pytest

from exozippy.config import ConfigManager, RANK_USER, RANK_DERIVED_USER


# Minimal star config — no data files needed.
# star symbolic_physics.py has mass = 10^logmass, which gives us a clean
# 1-parent relationship to test scale back-propagation.
_STAR_CONFIG = {"star": [{"name": "Lens"}]}


def _make_cm(extra_user_params):
    cm = ConfigManager(extra_user_params, system_config=_STAR_CONFIG)
    cm.finalize_user_params()
    return cm


# ---------------------------------------------------------------------------
# Back-propagation through logmass → mass
# ---------------------------------------------------------------------------

def test_mass_init_scale_propagates_to_logmass():
    """
    Given the user specifies init_scale on 'star.Lens.mass' (a derived parameter),
    When the relaxation engine runs,
    Then logmass (the sampled parent) receives a back-propagated scale.

    Relation: mass = 10^logmass
    Jacobian: dmass/dlogmass = mass * ln10
    Inverse:  sigma_logmass = sigma_mass / (mass * ln10)
    """
    logmass_initval = -0.3   # mass ≈ 0.5 M_sun
    mass_initval = 10 ** logmass_initval
    sigma_mass = 0.05

    cm = _make_cm({
        "star.Lens.logmass": {"initval": logmass_initval},
        "star.Lens.mass":    {"initval": mass_initval, "init_scale": sigma_mass},
    })

    logmass_scale = cm.propagated_scales.get("star.0.logmass")
    assert logmass_scale is not None, "logmass should have a back-propagated scale"

    expected = sigma_mass / (mass_initval * np.log(10))
    assert logmass_scale == pytest.approx(expected, rel=0.05), (
        f"Back-propagated logmass scale {logmass_scale:.4g} "
        f"doesn't match expected {expected:.4g}"
    )


def test_user_specified_parent_scale_beats_backprop():
    """
    Given the user specifies init_scale on both 'star.Lens.mass' (derived)
      and 'star.Lens.logmass' (the sampled parent),
    When the relaxation engine runs,
    Then logmass keeps the user's explicitly-specified scale (RANK_USER beats
      the back-propagated RANK_DERIVED_USER).
    """
    cm = _make_cm({
        "star.Lens.mass":    {"initval": 0.5, "init_scale": 0.05},
        "star.Lens.logmass": {"initval": -0.3, "init_scale": 999.0},
    })

    logmass_scale = cm.propagated_scales.get("star.0.logmass")
    assert logmass_scale == pytest.approx(999.0, rel=0.01), (
        "User-specified logmass init_scale should override back-propagated scale"
    )


def test_backprop_overrides_default_scale():
    """
    Given no user-specified logmass scale exists (only a default),
    When the user provides init_scale on the derived 'mass',
    Then the back-propagated scale replaces the default-level logmass scale.
    """
    sigma_mass = 0.02
    mass_initval = 0.5
    expected = sigma_mass / (mass_initval * np.log(10))

    cm = _make_cm({
        "star.Lens.mass":    {"initval": mass_initval, "init_scale": sigma_mass},
        "star.Lens.logmass": {"initval": np.log10(mass_initval)},
    })

    logmass_scale = cm.propagated_scales.get("star.0.logmass")
    assert logmass_scale is not None
    assert logmass_scale == pytest.approx(expected, rel=0.05), (
        "Back-propagated scale should replace the default logmass scale"
    )


def test_derived_param_scale_not_warned(caplog):
    """
    Given the user provides init_scale on a derived parameter,
    When the ConfigManager runs,
    Then no WARNING is emitted about the scale being ignored.
    """
    cm_under_test = ConfigManager(
        {"star.Lens.mass": {"initval": 0.5, "init_scale": 0.05}},
        system_config=_STAR_CONFIG,
    )
    with caplog.at_level(logging.WARNING):
        cm_under_test.finalize_user_params()

    ignored = [r for r in caplog.records
               if r.levelno == logging.WARNING and "ignored" in r.message]
    assert not ignored, f"Unexpected 'ignored' warnings: {[r.message for r in ignored]}"

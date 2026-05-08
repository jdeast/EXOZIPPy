# tests/test_config_healing.py
import pytest
import numpy as np
from exozippy.config import ConfigManager


def test_config_derives_te_from_physical_input():
    """
    Given: User only provides Mass, Distance, and Proper Motion.
    When: ConfigManager initializes.
    Then: It should derive t_E and inject it into user_params.
    """
    user_params = {
        "star.Lens.mass": {"initval": 0.5},
        "star.Lens.distance": {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "lens.Lens.u_0": {"initval": 0.5},
        "lens.Lens.t_0": {"initval": 2460000.0},
        "star.Lens.pm_ra": {"initval": 5.0},  # mas/yr
        "star.Lens.pm_dec": {"initval": 0.0},
        "star.Source.pm_ra": {"initval": 0.0},
        "star.Source.pm_dec": {"initval": 0.0}
    }

    cm = ConfigManager(user_params)

    # Check if t_E was derived and injected
    assert "lens.Lens.t_E" in cm.user_params
    derived_te = cm.user_params["lens.Lens.t_E"]["initval"]

    # Manual check:
    # pi_rel = 1000/4000 - 1000/8000 = 0.125
    # theta_E = sqrt(8.144 * 0.5 * 0.125) = 0.7134
    # t_E = (0.7134 / 5.0) * 365.25 = 52.12
    assert np.isclose(derived_te, 52.12, atol=0.1)
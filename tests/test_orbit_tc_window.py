"""The hard tc window must be one full period of the period the USER set.

`tc` is periodic (tc and tc + P are the same solution), so `tc_init +/- P/2`
is the right hard bound.  The trap is *which* P: the window is computed in
stage 2 (`Orbit.register_parameters`), which runs BEFORE the relaxation
engine reconciles `period:` with `logP:`.  Reading `logP`'s resolved initval
alone therefore returned its defaults.yaml value (1.0 -> 10 d) for every fit
that seeded `period:` instead, hard-bounding tc to +/- 5 d regardless of the
real period.  Several shipped kelt4 params files seed `period:`.
"""

import numpy as np
import pytest

from exozippy.components.orbit import Orbit
from exozippy.config import ConfigManager

TC = 2460000.0


def _window(user_params, names=("b",)):
    """Return the per-orbit half-width of the manifest's tc window, in days."""
    cm = ConfigManager(dict(user_params))
    orbit = Orbit([{"name": n} for n in names], cm)
    orbit.register_parameters(system=None)
    entry = orbit.manifest["tc"]
    return (
        np.atleast_1d(entry["upper"]) - np.atleast_1d(entry["lower"])
    ) / 2.0


def test_user_seeded_period_sets_the_tc_window():
    """
    Given a params file that seeds `period: 1000` (and not logP),
    When the orbit registers its parameters,
    Then the tc window is +/- 500 d, not the +/- 5 d that logP's
      defaults.yaml initval of 1.0 would give -- a hard bound that
      excluded the true tc with no way for the sampler to reach it.
    """
    # Arrange / Act
    half = _window(
        {"orbit.0.period": {"initval": 1000.0}, "orbit.0.tc": {"initval": TC}}
    )

    # Assert
    np.testing.assert_allclose(half, [500.0])


def test_user_seeded_logP_still_sets_the_tc_window():
    """
    Given a params file that seeds `logP: 3.0` (the other legal spelling),
    When the orbit registers its parameters,
    Then the tc window is the same +/- 500 d -- the fix must not regress
      the spelling that already worked.
    """
    half = _window(
        {"orbit.0.logP": {"initval": 3.0}, "orbit.0.tc": {"initval": TC}}
    )
    np.testing.assert_allclose(half, [500.0])


def test_unseeded_orbit_falls_back_to_the_logP_default():
    """
    Given a params file that seeds neither period nor logP,
    When the orbit registers its parameters,
    Then the window is the defaults.yaml logP (1.0 -> 10 d) half-period,
      +/- 5 d, exactly as before.
    """
    half = _window({"orbit.0.tc": {"initval": TC}})
    np.testing.assert_allclose(half, [5.0])


def test_short_period_seed_narrows_the_window():
    """
    Given the KELT-4b period seeded as `period: 2.9895933`,
    When the orbit registers its parameters,
    Then the window is +/- P/2 = 1.4948 d, not the +/- 5 d the shipped
      example used to get -- one period, the real one, in both directions.
    """
    half = _window(
        {
            "orbit.0.period": {"initval": 2.9895933},
            "orbit.0.tc": {"initval": TC},
        }
    )
    np.testing.assert_allclose(half, [2.9895933 / 2.0])


@pytest.mark.parametrize("spelling", ["period", "logP"])
def test_window_is_per_orbit(spelling):
    """
    Given a two-orbit system where only the second orbit is seeded,
    When the orbits register their parameters,
    Then each orbit gets its own window -- the seeded one from its seed,
      the unseeded one from the default.
    """
    # Arrange: seed orbit "wide" at 1000 d in either spelling.
    value = 1000.0 if spelling == "period" else 3.0
    up = {
        f"orbit.wide.{spelling}": {"initval": value},
        "orbit.tight.tc": {"initval": TC},
        "orbit.wide.tc": {"initval": TC},
    }

    # Act
    half = _window(up, names=("tight", "wide"))

    # Assert
    np.testing.assert_allclose(half, [5.0, 500.0])

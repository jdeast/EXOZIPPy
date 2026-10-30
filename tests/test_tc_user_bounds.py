"""A user's lower/upper on `orbit.tc` is the support of the built transform.

Review 1.8.17: the orbit's one-period tc window (`tc_init +/- P/2`) was a
bare manifest OPTION, and options are merged OVER the resolved config, so a
params file's `tc: {lower: T0 - 0.3, upper: T0 + 0.3}` built a logit
transform on T0 +/- P/2 and the user's box applied nowhere.  The window is a
modelling default (the likelihood is finite a period away), so it is now an
option WITH A GAP (config.md): the elements the user bounded keep the user's
bound, tighter or looser, field by field; the rest keep the window exactly.
"""

import numpy as np
import pytest

from exozippy.components.orbit import Orbit
from exozippy.config import ConfigManager
from exozippy.system import System

P = 3.0
TC = 2455200.0


def _write_rv(path, seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(TC - 60.3, TC + 60.3, 40)
    rv = -60.0 * np.sin(2.0 * np.pi * (t - TC) / P) + rng.normal(
        0, 5.0, t.size
    )
    np.savetxt(path, np.column_stack([t, rv, np.full(t.size, 5.0)]))
    return str(path)


def _params(**tc_fields):
    return {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "planet.b.mass": {"initval": 0.5},
        "orbit.b.period": {"initval": P},
        "orbit.b.tc": {"initval": TC, **tc_fields},
        "orbit.b.secosw": {"initval": 0.1},
        "orbit.b.sesinw": {"initval": 0.1},
        "orbit.b.cosi": {"initval": 0.1},
    }


@pytest.fixture(scope="module")
def rv_file(tmp_path_factory):
    return _write_rv(tmp_path_factory.mktemp("tc_user_bounds") / "rv.dat")


def _tc_support(rv_file, params):
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "rvinstrument": [{"name": "inst", "file": rv_file}],
    }
    system = System(config, params)
    system.prepare()
    system.build_model()
    assert system.orbit.tc_epoch.tolist() == [0]
    tf = system.orbit.tc._raw_transform
    assert tf["use_logit"][0]
    return float(tf["lowers"][0]), float(tf["uppers"][0])


def test_a_tighter_user_box_is_the_support(rv_file):
    """
    Given a params file bounding tc to T0 +/- 0.3 d inside the +/- 1.5 d
      one-period window,
    When the model is built,
    Then the logit transform's support is the user's box, not the window.
    """
    lo, hi = _tc_support(rv_file, _params(lower=TC - 0.3, upper=TC + 0.3))
    assert (lo, hi) == (TC - 0.3, TC + 0.3)


def test_a_one_sided_user_bound_keeps_the_window_on_the_other_side(rv_file):
    """
    Given only a user `lower`, looser than the window,
    When the model is built,
    Then the support is the user's lower and the window's upper: the gap is
      per field, and the user may loosen the default as well as tighten it.
    """
    lo, hi = _tc_support(rv_file, _params(lower=TC - 2.0))
    assert (lo, hi) == (TC - 2.0, TC + P / 2.0)


def test_with_no_user_bound_the_window_is_unchanged(rv_file):
    """
    Given no user bound on tc,
    When the model is built,
    Then the support is exactly T0 +/- P/2, the window it always had.
    """
    lo, hi = _tc_support(rv_file, _params())
    assert (lo, hi) == (TC - P / 2.0, TC + P / 2.0)


def test_the_manifest_gaps_only_the_elements_the_user_bounded():
    """
    Given two orbits, one bounded by the user (instance-name spelling),
    When the orbit registers its parameters,
    Then the manifest window is NaN ("keep the resolved value") on that
      element only, and the other element's window is the bit-identical
      `tc_init +/- P/2` it always was.
    """
    cm = ConfigManager(
        {
            "orbit.b.period": {"initval": P},
            "orbit.c.period": {"initval": 2 * P},
            "orbit.b.tc": {"initval": TC, "upper": TC + 0.1},
            "orbit.c.tc": {"initval": TC + 1.0},
        }
    )
    orbit = Orbit([{"name": "b"}, {"name": "c"}], cm)
    orbit.register_parameters(system=None)
    entry = orbit.manifest["tc"]
    np.testing.assert_array_equal(entry["lower"], [TC - P / 2, TC + 1.0 - P])
    assert np.isnan(entry["upper"][0])
    assert entry["upper"][1] == TC + 1.0 + P


# ---------------------------------------------------------------------------
# The same defect on cos i's half-plane lower bound (the audit that followed
# review 1.8.17): a bare option replaced a user's `cosi: {lower: ...}`.
# ---------------------------------------------------------------------------


def test_a_user_cosi_lower_is_the_support(rv_file):
    """
    Given a params file bounding cos i below at 0.05 (i < 87.1 deg),
    When the model is built,
    Then cos i's support starts at the user's 0.05, not the component's 0.
    """
    params = _params()
    params["orbit.b.cosi"] = {"initval": 0.1, "lower": 0.05}
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "rvinstrument": [{"name": "inst", "file": rv_file}],
    }
    system = System(config, params)
    system.prepare()
    system.build_model()
    tf = system.orbit.cosi._raw_transform
    assert float(tf["lowers"][0]) == 0.05


def test_the_cosi_half_plane_is_gapped_only_where_the_user_wrote_lower():
    """
    Given two orbits, one with a user cos i lower,
    When the orbit registers its parameters,
    Then that element's option is NaN (keep the user's) and the other's is
      the bit-identical 0.0 half-plane.
    """
    cm = ConfigManager({"orbit.b.cosi": {"lower": -1.0}})
    orbit = Orbit([{"name": "b"}, {"name": "c"}], cm)
    orbit.register_parameters(system=None)
    lower = np.asarray(orbit.manifest["cosi"]["lower"], dtype=float)
    assert np.isnan(lower[0]) and lower[1] == 0.0

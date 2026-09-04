"""Per-source lens seeds addressed by the SOURCE star's name on a
SINGLE-source event (review 2.6.13).

conventions.md/config.md document `lens.<source star>.t_0` as the spelling
of the lens's per-source vector elements, and three shipped params files
(ob07224, ob09020, ob170114) use it.  The VALUES always applied --
Lens._rewrite_source_param_keys translates the keys to index form at
construction -- but the manifest attached the per-source display names only
for n_sources > 1, so on a one-source event diagnostics.check_unused_yaml
(which audits the user's own spellings against display labels) falsely
warned that every such seed "did not match any model parameter and were
not applied".  That false warning opened every run of the three examples
and misdirected review 2.6.13's diagnosis twice.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from exozippy.diagnostics import ModelAuditor
from exozippy.system import System

_T0 = 2458560.0


def _write_lc(path, n=60, span=40.0):
    rng = np.random.default_rng(7)
    t = np.linspace(_T0 - span, _T0 + span, n)
    mag = 15.0 - rng.uniform(0, 0.001, n)
    err = np.full(n, 0.01)
    np.savetxt(path, np.column_stack([t, mag, err]))
    return str(path)


def _system(tmp_path, params):
    lc = _write_lc(tmp_path / "lc.dat")
    config = {
        "star": [{"name": "L1"}, {"name": "Source"}],
        "lens": [
            {
                "name": "EV",
                "lenses": ["star.0"],
                "sources": ["star.1"],
                "mmexofast": False,
            }
        ],
        "mulensinstrument": [{"name": "OGLE", "file": lc, "filter": "I"}],
    }
    base = {
        "star.L1.mass": {"initval": 0.6},
        "star.L1.distance": {"initval": 4000.0},
        "star.Source.mass": {"initval": 1.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.radius": {"sigma": 0.0},
        "star.teff": {"sigma": 0.0},
        "star.feh": {"sigma": 0.0},
        "star.ra": {"initval": 268.0, "sigma": 0},
        "star.dec": {"initval": -29.0, "sigma": 0},
    }
    base.update(params)
    system = System(config, user_params=base)
    system.prepare()
    model = system.build_model()
    return system, model


def test_source_named_seed_reaches_the_parameter_on_a_single_source_event(
    tmp_path,
):
    """
    Given: a 1-source microlensing system whose params file seeds
      lens.Source.t_0 / lens.Source.u_0 by the SOURCE star's name -- the
      documented spelling, and the one three shipped examples use,
    When: the model is built,
    Then: the seeds land on the parameters (they always did, through
      Lens._rewrite_source_param_keys), AND the unused-yaml audit does not
      report them as unmatched -- the audit half FAILED before the fix:
      with no per-source display names on a 1-source event the audit
      falsely warned the seeds "were not applied".
    """
    # Arrange / Act
    system, model = _system(
        tmp_path,
        {
            "lens.Source.t_0": {"initval": _T0 + 1.25},
            "lens.Source.u_0": {"initval": 0.31},
        },
    )

    # Assert: the values are ON the parameters, visible to resolve().
    assert np.isclose(
        float(np.atleast_1d(system.lens.t_0.initval)[0]), _T0 + 1.25
    )
    assert np.isclose(float(np.atleast_1d(system.lens.u_0.initval)[0]), 0.31)

    # Assert: the audit behind run.py's "did not match any model parameter"
    # warning no longer lists the source-named keys.
    unused = ModelAuditor(model, system, {}).check_unused_yaml()
    assert not [k for k in unused if str(k).startswith("lens.Source.")], unused


def test_source_named_sigma_is_applied_on_a_single_source_event(tmp_path):
    """
    Given: a Gaussian prior spelled lens.Source.u_0: {initval, mu, sigma}
      on a 1-source event,
    When: the model is built,
    Then: the sigma reaches the Parameter.  A contract pin (this held
      before the fix too, through _rewrite_source_param_keys): nothing
      else asserts that a source-named PRIOR reaches a 1-source event, and
      the rewriter is the only thing standing between this spelling and a
      silently dropped prior.
    """
    # Arrange / Act
    system, _ = _system(
        tmp_path,
        {
            "lens.Source.u_0": {"initval": 0.31, "mu": 0.31, "sigma": 0.02},
        },
    )

    # Assert
    sigma = np.atleast_1d(system.lens.u_0.sigma)[0]
    assert np.isclose(float(sigma), 0.02)

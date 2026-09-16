"""Per-source trajectory seeds addressed by the SOURCE star's name on a
SINGLE-source event (review 2.6.13).

conventions.md/config.md document `source.<source star>.t_0` as the
spelling of the per-source trajectory parameters, and three shipped params
files (ob07224, ob09020, ob170114) use it.  The VALUES always applied --
pre-split Lens._rewrite_source_param_keys translated the keys to index
form at construction; post-split each `source:` instance is simply NAMED
after its body star (Source.normalize_config_block ->
bodies.derive_body_names), so the name form folds to index form at
ConfigManager construction instead.  But the manifest attached the
per-source display names only for n_sources > 1, so on a one-source event
diagnostics.check_unused_yaml (which audits the user's own spellings
against display labels) falsely warned that every such seed "did not match
any model parameter and were not applied".  That false warning opened
every run of the three examples and misdirected review 2.6.13's diagnosis
twice.
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
        "mulensevent": [{"name": "EV", "mmexofast": False}],
        "lens": [{"body": "star.L1"}],
        "source": [{"body": "star.Source"}],
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
      source.Source.t_0 / source.Source.u_0 by the SOURCE star's name --
      the documented spelling, and the one three shipped examples use,
    When: the model is built,
    Then: the seeds land on the parameters (they always did -- pre-split
      through Lens._rewrite_source_param_keys, now through the source
      instance being named after its body star), AND the unused-yaml audit
      does not report them as unmatched -- the audit half FAILED before the
      fix: with no per-source display names on a 1-source event the audit
      falsely warned the seeds "were not applied".
    """
    # Arrange / Act
    system, model = _system(
        tmp_path,
        {
            "source.Source.t_0": {"initval": _T0 + 1.25},
            "source.Source.u_0": {"initval": 0.31},
        },
    )

    # Assert: the values are ON the parameters, visible to resolve().
    assert np.isclose(
        float(np.atleast_1d(system.source.t_0.initval)[0]), _T0 + 1.25
    )
    assert np.isclose(float(np.atleast_1d(system.source.u_0.initval)[0]), 0.31)

    # Assert: the audit behind run.py's "did not match any model parameter"
    # warning no longer lists the source-named keys.
    unused = ModelAuditor(model, system, {}).check_unused_yaml()
    assert not [k for k in unused if str(k).startswith("source.Source.")], (
        unused
    )


def test_source_named_sigma_is_applied_on_a_single_source_event(tmp_path):
    """
    Given: a Gaussian prior spelled source.Source.u_0:
      {initval, mu, sigma} on a 1-source event,
    When: the model is built,
    Then: the sigma reaches the Parameter.  A contract pin (this held
      before the fix too, through _rewrite_source_param_keys): nothing
      else asserts that a source-named PRIOR reaches a 1-source event, and
      the name-to-index folding is the only thing standing between this
      spelling and a silently dropped prior.
    """
    # Arrange / Act
    system, _ = _system(
        tmp_path,
        {
            "source.Source.u_0": {
                "initval": 0.31,
                "mu": 0.31,
                "sigma": 0.02,
            },
        },
    )

    # Assert
    sigma = np.atleast_1d(system.source.u_0.sigma)[0]
    assert np.isclose(float(sigma), 0.02)

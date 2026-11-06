"""A sampled element must state its prior (review 1.2.10).

Two finite bounds (uniform, logit path) or a sigma > 0 (explicit Gaussian).
Anything else used to fall through to ``initval + init_scale * raw`` with
raw ~ N(0, 1): a prior N(initval, init_scale) centered on the START value,
which nobody stated.  JDE ruling 2026-11-01: raise, naming the element, the
values in user units, the provenance and the remedy.
"""

import numpy as np
import pymc as pm
import pytest

from exozippy.components.parameter import Parameter


def _build(**kw):
    # Infinite bounds are spelled out: a missing one is a separate
    # developer error (every sampled parameter's defaults carry both keys).
    kw = {"lower": -np.inf, "upper": np.inf, **kw}
    with pm.Model() as model:
        p = Parameter(
            label="star.test_param", initval=5.0, init_scale=1.0, **kw
        )
        p.build_pymc()
    return p, model


def test_one_bound_and_no_sigma_raises_naming_the_missing_side():
    """
    Given a sampled element with only a lower bound and no sigma,
    When it is built,
    Then it raises naming the element, its values, the missing side and the
      remedy.
    """
    with pytest.raises(ValueError) as exc:
        _build(lower=0.0)
    msg = str(exc.value)
    assert "no stated prior" in msg
    assert "star.test_param" in msg
    assert "start 5" in msg and "lower 0" in msg and "upper none" in msg
    assert "missing: upper" in msg
    assert "start value from:" in msg
    assert "'lower'/'upper'" in msg and "'sigma'" in msg


def test_no_bounds_and_no_sigma_raises():
    """
    Given a sampled element with neither bound and no sigma,
    When it is built,
    Then it raises, naming both sides as missing.
    """
    with pytest.raises(ValueError, match="missing: lower and upper"):
        _build()


def test_the_params_file_is_named_when_known():
    """
    Given the offending element came from a known params file,
    When it is built,
    Then the error names that file.
    """
    with pytest.raises(ValueError, match=r"params file: my\.params\.yaml"):
        with pm.Model():
            Parameter(
                label="star.test_param",
                initval=5.0,
                init_scale=1.0,
                lower=0.0,
                upper=np.inf,
                source_file="my.params.yaml",
            ).build_pymc()


@pytest.mark.parametrize("bounds", [{}, {"lower": 0.0}])
def test_a_sigma_states_the_prior_and_builds(bounds):
    """
    Given a sampled element with fewer than two finite bounds but a sigma,
    When it is built,
    Then it builds on the Gaussian path: the raw N(0, 1) scaled by sigma IS
      the stated prior, and a single bound becomes a soft barrier.
    """
    p, model = _build(mu=5.0, sigma=2.0, **bounds)
    assert not p._raw_transform["use_logit"][0]
    logp = model.compile_logp()(model.initial_point())
    assert np.isfinite(logp)


def test_two_finite_bounds_build():
    """
    Given a sampled element with two finite bounds and no sigma,
    When it is built,
    Then it builds on the logit path (uniform prior).
    """
    p, _ = _build(lower=0.0, upper=10.0)
    assert p._raw_transform["use_logit"][0]

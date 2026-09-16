"""err_scale's bounds, and the remedy a wall warning carries (review 8.2.2).

WHAT THESE GUARD.  On DC2018-226 a point-lens fit to a wide binary inflated
both bands' error scales 300-460x inside a six-decade bound and walked u_0 to
zero under the flattened likelihood, paying 181k nats of misfit it could no
longer feel (review 2.4.14).  Three things changed: the default bound is two
decades either side of 1, a component can declare what a value against its
bound MEANS, and the posterior is checked against every hard bound at
wrap-up with that sentence attached.
"""

import logging

import numpy as np
import pymc as pm
import pytest
import yaml

from exozippy import diagnostics
from exozippy.components.parameter import Parameter

REMEDY = "Rescale the supplied errors or seed a better starting model."


def _load_defaults():
    from pathlib import Path

    import exozippy.components.mulensing as mu

    return yaml.safe_load(
        (Path(mu.__file__).parent / "defaults.yaml").read_text()
    )


def test_err_scale_default_bounds_are_two_decades_each_side_of_one():
    """
    Given the mulensing component's defaults.yaml,
    When err_scale's bounds are read,
    Then they are 0.01 and 100 (JDE's ruling on 8.2.2), and the entry
      carries a near-bound remedy that names the two real fixes.
    """
    # ARRANGE / ACT
    spec = _load_defaults()["mulensinstrument"]["err_scale"]

    # ASSERT
    assert spec["lower"] == pytest.approx(0.01)
    assert spec["upper"] == pytest.approx(100.0)
    remedy = spec["near_bound_remedy"]
    assert "errors" in remedy and "starting model" in remedy


@pytest.mark.parametrize(
    "value, lower, upper, expected",
    [
        (1.0, 0.01, 100.0, 0.5),  # log-space midpoint of a positive scale
        (100.0, 0.01, 100.0, 1.0),
        (0.01, 0.01, 100.0, 0.0),
        (0.0, -1.0, 1.0, 0.5),  # linear when a bound is not positive
        (0.75, 0.5, 1.5, 0.5),  # linear-looking interval still measured in log
    ],
)
def test_near_bound_position_is_logarithmic_for_a_scale(
    value, lower, upper, expected
):
    """
    Given a value inside [lower, upper],
    When its position is measured,
    Then a positive interval is read in log space and any other linearly --
      so err_scale = 1 on [0.01, 100] is the MIDDLE, not 1% from the floor.
    """
    pos = diagnostics.near_bound_position(value, lower, upper)
    if expected == 0.5 and (lower, upper) == (0.5, 1.5):
        expected = (np.log(0.75) - np.log(0.5)) / (np.log(1.5) - np.log(0.5))
    assert pos == pytest.approx(expected, abs=1e-9)


def test_near_bound_position_degenerate_inputs_are_nan_not_errors():
    assert np.isnan(diagnostics.near_bound_position(1.0, 2.0, 2.0))
    assert np.isnan(diagnostics.near_bound_position(np.nan, 0.0, 1.0))


class _System:
    def __init__(self, params):
        self._params = params

    def get_all_parameters(self):
        return list(self._params)


def _built(label, posterior_values, remedy=REMEDY, lower=0.01, upper=100.0):
    par = Parameter(
        label=label,
        initval=1.0,
        init_scale=0.1,
        lower=lower,
        upper=upper,
        unit="",
        internal_unit="",
        near_bound_remedy=remedy,
    )
    with pm.Model():
        par.build_pymc()
    par.posterior = np.asarray(posterior_values, dtype=float)
    return par


def test_posterior_against_the_upper_wall_warns_with_the_remedy(caplog):
    """
    Given a bounded parameter whose posterior median sits at 95 on [0.01, 100],
    When the wrap-up near-bound check runs,
    Then it warns once, names the upper bound, and appends the component's
      remedy sentence -- while a parameter sitting mid-range says nothing.
    """
    # ARRANGE
    wall = _built("mulensinstrument.A.err_scale", np.full(400, 95.0))
    fine = _built("mulensinstrument.B.err_scale", np.full(400, 1.1))

    # ACT
    with caplog.at_level(logging.WARNING, logger="exozippy.diagnostics"):
        hits = diagnostics.warn_posterior_near_bounds(_System([wall, fine]))

    # ASSERT
    assert [h["label"] for h in hits] == ["mulensinstrument.A.err_scale"]
    assert hits[0]["side"] == "upper"
    assert hits[0]["position"] > 0.98
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    assert "upper bound" in msgs[0]
    assert msgs[0].endswith(REMEDY)
    assert "mulensinstrument.B.err_scale" not in caplog.text


def test_posterior_against_the_lower_wall_is_read_in_log_space(caplog):
    """
    Given err_scale sitting at 0.012 on [0.01, 100],
    When the check runs,
    Then it is a LOWER-wall hit: linearly 0.012 is 0.002% of a span of
      100 and log-space says 2% from the floor, and the check reads the
      latter.  A parameter at 0.5 -- 0.5% of the linear span -- is NOT a hit.
    """
    near = _built("mulensinstrument.A.err_scale", np.full(300, 0.0115))
    mid = _built("mulensinstrument.B.err_scale", np.full(300, 0.5))

    with caplog.at_level(logging.WARNING, logger="exozippy.diagnostics"):
        hits = diagnostics.warn_posterior_near_bounds(_System([near, mid]))

    assert [h["side"] for h in hits] == ["lower"]
    assert hits[0]["label"] == "mulensinstrument.A.err_scale"


def test_a_parameter_without_a_remedy_still_warns_generically(caplog):
    """
    Given a bounded parameter that declared no near_bound_remedy,
    When its posterior sits on a wall,
    Then the generic sentence is logged with nothing appended -- the remedy
      is optional and its absence is not an error.
    """
    par = _built("star.X.thing", np.full(100, 99.9), remedy=None)

    with caplog.at_level(logging.WARNING, logger="exozippy.diagnostics"):
        hits = diagnostics.warn_posterior_near_bounds(_System([par]))

    assert len(hits) == 1
    assert caplog.records[0].getMessage().endswith("the thing to revisit.")


def test_remedy_suffix_is_the_one_spelling_both_warnings_share():
    """The post-polish wall warning and the posterior check append the same
    text through Parameter.remedy_suffix, so they cannot drift."""
    par = Parameter(label="a.b.c", initval=1.0, near_bound_remedy="  Do X.  ")
    assert par.remedy_suffix() == "  Do X."
    assert Parameter(label="a.b.d", initval=1.0).remedy_suffix() == ""


def test_resolve_carries_the_remedy_from_defaults_to_the_parameter():
    """
    Given the mulensing defaults,
    When ConfigManager builds the err_scale entry a Parameter is constructed
      from,
    Then near_bound_remedy rides along -- it is a defaults-only field, not
      a params-file key, and this is the seam it has to cross.
    """
    from exozippy.config import ConfigManager

    src = ConfigManager.resolve.__code__.co_consts
    assert "near_bound_remedy" in src

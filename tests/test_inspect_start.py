"""inspect_start's read-only contract and its user_params key lookup.

inspect_start renders the startup table.  It is a diagnostic: it must report
the start the sampler is actually going to use, and it must not change it.
Two defects used to break both halves of that:

  * ``np.atleast_1d(p.initval)`` ALIASES a 1-D initval array, so writing the
    solver-reconciled value into it mutated ``Parameter.initval`` in place --
    reverting the polished starts ``System.apply_polished_starts`` had just
    stored there.  ``get_raw_starts``/``_seed_initvals_for`` then rebuilt the
    later seeds from the reverted values.
  * the lookup keys were built from name-form display labels
    (``star.B.mass``) while ``ConfigManager.user_params`` is stored in index
    form (``star.1.mass``), so the lookup succeeded or failed depending on
    whether the user had named their components.
"""

import logging
from unittest.mock import patch

import numpy as np
import pymc as pm
import pytest

from exozippy.components.parameter import Parameter
from exozippy.components.star.star import Star
from exozippy.config import ConfigManager, canonical_param_key
from exozippy.run import _user_initval, inspect_start

CONFIG = {"star": [{"name": "A"}, {"name": "B"}]}


class _Sys:
    """Duck-typed System: only what ModelAuditor/inspect_start touch."""

    def __init__(self, config_manager, params):
        self.config_manager = config_manager
        self.user_params = config_manager.user_params
        self._params = params

    def get_all_parameters(self):
        return self._params

    def get_parameter_lookup(self):
        return {p.label: p for p in self._params}


def _build_star_mass(user_params, model_name, system_config=CONFIG):
    """Build the two-element star.mass Parameter for CONFIG's named stars.

    ``system_config=None`` keeps ConfigManager from standardizing the user
    keys, so they stay in whatever form the caller wrote them -- the two
    spellings the read-only contract has to hold for.
    """
    cm = ConfigManager(user_params, system_config=system_config)
    star = Star(CONFIG["star"], cm)
    with pm.Model(name=model_name) as model:
        star.manifest = {"mass": {}}
        star.add_parameter(model=model, param_name="mass", system=None)
    p = star.mass
    return model, _Sys(cm, [p]), p


def _table_row(caplog, label):
    """The startup-table line for one element, or None."""
    for rec in caplog.records:
        if rec.getMessage().strip().startswith(label + " "):
            return rec.getMessage()
    return None


# ---------------------------------------------------------------------------
# (a) read-only with respect to Parameter.initval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system_config", [CONFIG, None])
@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_inspect_start_never_mutates_initval(mock_logp, system_config):
    """
    Given a vector Parameter whose initval was moved after construction
      (exactly what the pre-whitening seed polish does),
    When inspect_start renders the startup table,
    Then Parameter.initval is byte-identical afterwards -- the table is a
      diagnostic and the start it reports on belongs to the sampler.
    """
    # ARRANGE
    mock_logp.return_value = ({}, {})
    model, system, p = _build_star_mass(
        {"star.A.mass": {"initval": 1.0}, "star.B.mass": {"initval": 0.85}},
        f"model_readonly_{system_config is not None}",
        system_config=system_config,
    )
    polished = np.array([1.4, 0.5])
    p.initval = polished.copy()
    before = p.initval.copy()

    # ACT
    inspect_start(model, system, {}, {}, {})

    # ASSERT
    assert p.initval.tobytes() == before.tobytes()
    np.testing.assert_array_equal(p.initval, polished)


@pytest.mark.parametrize("system_config", [CONFIG, None])
@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_inspect_start_reports_the_start_the_sampler_will_use(
    mock_logp, caplog, system_config
):
    """
    Given a polished initval that differs from the user_params entry it was
      originally resolved from,
    When inspect_start renders the startup table,
    Then the table shows the LIVE initval (what get_raw_start encodes and the
      sampler begins from), not the stale user_params value.
    """
    # ARRANGE
    mock_logp.return_value = ({}, {})
    model, system, p = _build_star_mass(
        {"star.A.mass": {"initval": 1.0}, "star.B.mass": {"initval": 0.85}},
        f"model_reports_live_{system_config is not None}",
        system_config=system_config,
    )
    p.initval = np.array([1.4, 0.5])

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, {}, {}, {})

    # ASSERT
    row_a = _table_row(caplog, "star.A.mass")
    row_b = _table_row(caplog, "star.B.mass")
    assert row_a is not None and row_b is not None
    assert "1.40000000" in row_a and "1.00000000" not in row_a
    assert "0.50000000" in row_b and "0.85000000" not in row_b


# ---------------------------------------------------------------------------
# (b) the lookup no longer depends on how the user spelled the instance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["star.B.mass", "star.1.mass"])
def test_user_initval_found_by_name_or_index(key):
    """
    Given a user entry written EITHER by instance name or by index,
    When _user_initval looks the element's start value up,
    Then it finds the same value both ways -- user_params is stored in index
      form, so a name-form key only matches after canonicalization.
    """
    # ARRANGE
    model, system, p = _build_star_mass(
        {key: {"initval": 0.85}}, f"model_lookup_{key.replace('.', '_')}"
    )
    assert "star.1.mass" in system.config_manager.user_params  # index form

    # ACT
    found = _user_initval(system.config_manager, p, 1)

    # ASSERT
    assert found == pytest.approx(0.85)
    assert _user_initval(system.config_manager, p, 0) is None


@pytest.mark.parametrize("key", ["star.B.mass", "star.1.mass"])
@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_unset_element_falls_back_to_user_params_either_spelling(
    mock_logp, caplog, key
):
    """
    Given an element the Parameter carries no number for,
    When inspect_start renders the startup table,
    Then it falls back to the user/solved parameter table, identically
      whether the user named the instance or indexed it.
    """
    # ARRANGE
    mock_logp.return_value = ({}, {})
    model, system, p = _build_star_mass(
        {key: {"initval": 0.85}},
        f"model_fallback_{key.replace('.', '_')}",
    )
    p.initval = np.array([np.nan, np.nan])

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, {}, {}, {})

    # ASSERT
    row_b = _table_row(caplog, "star.B.mass")
    assert row_b is not None
    assert "0.85000000" in row_b


def test_canonical_param_key_leaves_unknown_and_short_keys_alone():
    """
    Given keys the config cannot resolve to an instance index,
    When canonical_param_key canonicalizes them,
    Then they come back unchanged -- which is how standardize_param_names
      stores them, so the two stay in agreement.
    """
    assert canonical_param_key("star.B.mass", CONFIG) == "star.1.mass"
    assert canonical_param_key("star.1.mass", CONFIG) == "star.1.mass"
    assert canonical_param_key("star.mass", CONFIG) == "star.mass"
    assert canonical_param_key("star.Z.mass", CONFIG) == "star.Z.mass"
    assert canonical_param_key("sed.errscale", CONFIG) == "sed.errscale"
    assert canonical_param_key("run", CONFIG) == "run"


def test_scalar_initval_is_also_left_alone():
    """
    Given a scalar-shaped Parameter (np.atleast_1d copies there, so it was
      never corrupted),
    When inspect_start runs,
    Then its initval is unchanged too -- pinning that the fix did not make
      the scalar path worse.
    """
    # ARRANGE
    cm = ConfigManager({}, system_config={})
    p = Parameter(label="toy.x", initval=2.0, lower=0.0, upper=10.0)
    with pm.Model(name="model_scalar_initval") as model:
        p.build_pymc()
    system = _Sys(cm, [p])

    # ACT
    with patch(
        "exozippy.diagnostics.ModelAuditor.get_aggregated_logps",
        return_value=({}, {}),
    ):
        inspect_start(model, system, {}, {}, {})

    # ASSERT
    assert np.asarray(p.initval).tolist() == [2.0]

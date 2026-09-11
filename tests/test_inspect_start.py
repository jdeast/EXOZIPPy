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

import exozippy.components
from exozippy.components.parameter import ElementExpression, Parameter
from exozippy.components.star.star import Star
from exozippy.config import ConfigManager, canonical_param_key
from exozippy.run import (
    _prints_in_startup_table,
    _user_initval,
    inspect_start,
)

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
        # star.mass is DERIVED in production (mass = 10**logmass), so
        # defaults.yaml gives it no bounds -- logmass carries the hard
        # support.  This manifest entry deliberately makes it a FREE
        # parameter as a test vehicle, and a free parameter must declare
        # its own lower/upper AND a start for EVERY element.  The
        # "overrides" initval supplies the latter and layers UNDER the
        # params file, so an element the caller seeded keeps its own value:
        # star.mass carries no defaults.yaml initval, so the unseeded
        # element used to build with a NaN start, which the transform turned
        # into log(NaN/(1-NaN)) behind a "nudged" warning.  What these tests
        # are about is the user_params key lookup, not the start value.
        star.manifest = {
            "mass": {
                "lower": 0.1,
                "upper": 250.0,
                "overrides": {"initval": 1.0},
            }
        }
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
    inspect_start(model, system, {})

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
        inspect_start(model, system, {})

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
        inspect_start(model, system, {})

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
        inspect_start(model, system, {})

    # ASSERT
    assert np.asarray(p.initval).tolist() == [2.0]


# ---------------------------------------------------------------------------
# (c) per-element units
# ---------------------------------------------------------------------------


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_per_element_units_are_reported_per_element(mock_logp, caplog):
    """
    Given a vector parameter whose elements carry DIFFERENT user units
      (a `unit:` override on one named instance only),
    When inspect_start renders the startup table,
    Then each element's value is converted with its OWN factor.

    Regression: the table converted with the whole-vector factor, which for
    a scalar input returns an n-element array; the .item() that followed
    then raised "can only convert an array of size 1" and killed the fit in
    its own startup banner.  ``Parameter.from_internal(..., index=i)`` is the
    per-element form, and ``element_factor`` is the one owner of the rule.
    """
    # ARRANGE: A in jupiterMass, B left at the defaults.yaml solMass.
    mock_logp.return_value = ({}, {})
    model, system, p = _build_star_mass(
        {"star.A.mass": {"unit": "jupiterMass", "initval": 200.0}},
        "model_per_element_units",
    )
    assert len(p.unit) == 2, "the override must survive as per-element units"

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, {})

    # ASSERT: A reports the jupiterMass number the user typed, B the solMass
    # one from the "overrides" default -- not either one in the other's unit.
    row_a = _table_row(caplog, "star.A.mass")
    row_b = _table_row(caplog, "star.B.mass")
    assert row_a is not None and row_b is not None
    assert "200.000000" in row_a
    assert "jupiterMass" in row_a
    assert "1.00000000" in row_b
    assert "solMass" in row_b


# ---------------------------------------------------------------------------
# (d) print_to_table -- review 3.14.9
# ---------------------------------------------------------------------------


def _build_star_mass_with(options, model_name):
    """The two-element star.mass vehicle, plus extra manifest options."""
    cm = ConfigManager({}, system_config=CONFIG)
    star = Star(CONFIG["star"], cm)
    with pm.Model(name=model_name) as model:
        star.manifest = {
            "mass": {
                "lower": 0.1,
                "upper": 250.0,
                "overrides": {"initval": 1.0},
                **options,
            }
        }
        star.add_parameter(model=model, param_name="mass", system=None)
    return model, _Sys(cm, [star.mass]), star.mass


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_a_sampled_parameter_marked_not_for_tables_is_not_printed(
    mock_logp, caplog
):
    """
    Given a SAMPLED parameter carrying print_to_table: false,
    When inspect_start renders the startup table,
    Then it gets no row.

    Review 3.14.9.  The startup table consulted only ``debug_print`` and the
    sampled-ness of the parameter, so the one flag that says "not for
    tables" -- a params-file key, so a user's only lever -- was the one
    thing this table ignored.
    """
    # ARRANGE
    mock_logp.return_value = ({}, {})
    model, system, p = _build_star_mass_with(
        {"print_to_table": False}, "model_ptt_false"
    )
    assert np.any(p.is_sampled), "the vehicle must be sampled to be a case"

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, {})

    # ASSERT
    assert _table_row(caplog, "star.A.mass") is None
    assert _table_row(caplog, "star.B.mass") is None


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_the_same_parameter_is_printed_with_the_flag_left_alone(
    mock_logp, caplog
):
    """
    Given the identical parameter with print_to_table at its default,
    When inspect_start renders the startup table,
    Then it gets its rows.

    The control for the test above: the suppression must come from the flag
    and not from something else about the vehicle.
    """
    mock_logp.return_value = ({}, {})
    model, system, _ = _build_star_mass_with({}, "model_ptt_default")

    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, {})

    assert _table_row(caplog, "star.A.mass") is not None
    assert _table_row(caplog, "star.B.mass") is not None


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_not_for_tables_vetoes_even_an_explicit_debug_print(mock_logp, caplog):
    """
    Given a parameter that sets BOTH debug_print: true and
      print_to_table: false,
    When inspect_start renders the startup table,
    Then print_to_table wins and there is no row.

    The precedence is deliberate and is the reason the flag is a veto rather
    than a default.  Every parameter that opts INTO this table does so with
    ``debug_print`` in its component's defaults.yaml, which a user cannot
    edit; ``print_to_table`` is a params-file key.  If ``debug_print`` won,
    a user would have no way to suppress a row at all.
    """
    mock_logp.return_value = ({}, {})
    model, system, _ = _build_star_mass_with(
        {"print_to_table": False, "debug_print": True}, "model_ptt_veto"
    )

    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, {})

    assert _table_row(caplog, "star.A.mass") is None


def test_orbit_n_is_the_only_shipped_not_for_tables_parameter():
    """
    Given the shipped defaults.yaml tree,
    When print_to_table: false is counted,
    Then orbit.n is the only user -- so wiring the flag into the startup
      table changes no shipped table today.

    The item named orbit.n as the LIVE case for 3.14.9; it is not, because
    orbit.n is derived and the sampled-only default already excluded it.
    This pins the fact behind that correction, so a future reader who finds
    a second not-for-tables parameter knows the claim has to be re-checked.
    """
    import pathlib

    root = pathlib.Path(exozippy.components.__file__).parent
    users = sorted(
        f"{path.parent.name}"
        for path in root.rglob("defaults.yaml")
        if "print_to_table: false" in path.read_text()
    )

    assert users == ["orbit"]


# ---------------------------------------------------------------------------
# (e) the flat-direction warning is sampler health, not table content
#     -- review 2.3.7
# ---------------------------------------------------------------------------


def _flat_report(p):
    """A whitening report whose probe found every element of ``p`` flat."""
    n = int(np.prod(p.shape)) if p.shape not in ((), None) else 1
    return {"multipliers": {f"{p.label}_raw": np.full(n, np.nan)}}


def _flat_warning(caplog):
    """The 'logp is flat along' warning, or None."""
    for rec in caplog.records:
        if "logp is flat along" in rec.getMessage():
            return rec.getMessage()
    return None


@pytest.mark.parametrize(
    "suppress", [{"print_to_table": False}, {"debug_print": False}]
)
@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_a_flat_sampled_parameter_is_warned_about_with_no_table_row(
    mock_logp, caplog, suppress
):
    """
    Given a SAMPLED parameter the table is told not to print
      (``print_to_table: false``, or ``debug_print: false``) whose probe
      multipliers came back NaN,
    When inspect_start runs,
    Then it gets no row AND the flat-direction warning still names both of
      its elements.

    Review 2.3.7: the collection used to sit inside the row loop, which
    `continue`s on _prints_in_startup_table, so a cosmetic table flag -- and
    ``print_to_table`` is a params-file key, i.e. the user's own -- silently
    disabled a sampler-health diagnostic whose own text says one
    unconstrained parameter destroys HMC efficiency.
    """
    # ARRANGE
    mock_logp.return_value = ({}, {})
    key = "_".join(suppress)
    model, system, p = _build_star_mass_with(suppress, f"model_flat_{key}")
    assert np.all(p.is_sampled), "the vehicle must be sampled to be a case"
    assert not _prints_in_startup_table(p), "the flag must suppress the rows"

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, {}, whiten_report=_flat_report(p))

    # ASSERT
    assert _table_row(caplog, "star.A.mass") is None
    assert _table_row(caplog, "star.B.mass") is None
    warning = _flat_warning(caplog)
    assert warning is not None, [r.getMessage() for r in caplog.records]
    assert "star.A.mass" in warning and "star.B.mass" in warning


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_a_measured_direction_is_not_called_flat(mock_logp, caplog):
    """
    Given the same suppressed parameter with FINITE probe multipliers,
    When inspect_start runs,
    Then no flat warning is emitted.

    The control for the test above: the warning has to come from the NaN
    multipliers rather than from having stopped consulting the table flags,
    or widening the collection would just warn about everything.
    """
    # ARRANGE
    mock_logp.return_value = ({}, {})
    model, system, p = _build_star_mass_with(
        {"print_to_table": False}, "model_flat_control"
    )
    report = {"multipliers": {f"{p.label}_raw": np.array([2.5, 3.0])}}

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, {}, whiten_report=report)

    # ASSERT
    assert _flat_warning(caplog) is None


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_no_whitening_report_means_no_flat_verdict(mock_logp, caplog):
    """
    Given a run with no whitening report at all (``measure_scales: false``),
    When inspect_start runs,
    Then nothing is reported flat -- unmeasured is not the same as flat, and
      a warning on every un-probed run would be noise.
    """
    mock_logp.return_value = ({}, {})
    model, system, _ = _build_star_mass_with({}, "model_flat_unprobed")

    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, {}, whiten_report=None)

    assert _flat_warning(caplog) is None


# ---------------------------------------------------------------------------
# (f) a parameter whose START is its expression's value -- review 3.14.16
# ---------------------------------------------------------------------------


def _row_value(caplog, label):
    """The Value column of one startup-table row, as a float."""
    row = _table_row(caplog, label)
    assert row is not None, f"no row for {label}"
    return float(row.split("|")[1].strip())


def _derived_pair(model_name, by_element):
    """A sampled driver plus a DERIVED parameter with no initval at all.

    ``by_element=True`` derives it one element at a time
    (``element_expressions``, so there is no whole-vector ``expression``),
    which is the shape 3.14.16 is about; ``False`` uses the whole-vector
    ``expression`` that the old gate already accepted, as the control.

    The driver starts at 0.5 on bounds [0, 1] and the derivation halves it,
    so the derived start is exactly 0.25 -- a number a draw from the
    driver's U(0, 1) prior would essentially never produce, which is what
    makes this also a test that the value is read AT THE START POINT.
    """
    driver = Parameter(
        label="comp.drv",
        initval=np.full(2, 0.5),
        init_scale=np.full(2, 0.1),
        lower=np.zeros(2),
        upper=np.ones(2),
        unit="",
        internal_unit="",
        shape=(2,),
        names=["i0", "i1"],
    )
    with pm.Model(name=model_name) as model:
        drv = driver.build_pymc()
        kwargs = (
            {
                "element_expressions": [
                    ElementExpression(
                        mask=[True, True], expr=lambda: drv * 0.5
                    )
                ]
            }
            if by_element
            else {"expression": lambda: drv * 0.5}
        )
        derived = Parameter(
            label="comp.der",
            initval=None,
            unit="",
            internal_unit="",
            shape=(2,),
            names=["i0", "i1"],
            debug_print=True,
            **kwargs,
        )
        derived.build_pymc()

    system = _Sys(ConfigManager({}, system_config={}), [driver, derived])
    point = model.initial_point()
    return model, system, derived, point


@pytest.mark.parametrize("by_element", [True, False])
@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_a_derived_parameter_with_no_initval_reports_its_start(
    mock_logp, caplog, by_element
):
    """
    Given a DERIVED parameter carrying no initval and no user entry -- so
      its start IS whatever its expression computes,
    When inspect_start renders the startup table,
    Then its rows carry that computed value, whether the derivation is
      declared per ELEMENT or for the whole vector.

    Review 3.14.16: the last-resort lookup was gated on ``p.expression is
    not None``, a whole-vector test inherited from 1.10.9, so the
    per-element form fell through it and the parameter got NO ROW at all.
    Measured on the one shipped case, ``examples/ob09020``'s
    ``lens.alpha``: the row was missing, and it now reads 189.082 deg --
    which is that example's published alpha_0 = 189.08, recorded in its own
    params file as the acceptance check.

    The posterior-side twin of this gate was widened the same way in
    System._set_comp_posterior; both tables now agree on what "derived and
    therefore evaluable" means.
    """
    # ARRANGE
    mock_logp.return_value = ({}, {})
    model, system, derived, point = _derived_pair(
        f"model_derived_{by_element}", by_element
    )
    assert derived.initval is None
    assert np.all(derived.is_derived)

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, point)

    # ASSERT
    for element in ("i0", "i1"):
        row = _table_row(caplog, f"comp.{element}.der")
        assert row is not None, [r.getMessage() for r in caplog.records]
        assert "0.25000000" in row, row
        # and it really is half the driver's own reported start, so the row
        # is the derivation's answer rather than a coincidence
        assert _row_value(caplog, f"comp.{element}.der") == pytest.approx(
            0.5 * _row_value(caplog, f"comp.{element}.drv")
        )


@patch("exozippy.diagnostics.ModelAuditor.get_aggregated_logps")
def test_the_value_comes_from_the_start_point_not_from_a_prior_draw(
    mock_logp, caplog
):
    """
    Given the same per-element derived parameter,
    When the driver's start is MOVED before the table is rendered,
    Then the reported value follows the new start.

    The property that matters and that ``p.value.eval()`` -- the spelling
    the item proposed -- does not have: evaluating the graph with its RVs
    still in place samples the driver's prior, so the row would be a random
    number that merely looks plausible.  ModelAuditor.values_at_start
    replaces the RVs by their value variables and feeds the start point,
    which is why moving the start moves the row.
    """
    # ARRANGE
    mock_logp.return_value = ({}, {})
    model, system, _derived, point = _derived_pair("model_derived_start", True)
    moved = {k: np.asarray(v) for k, v in point.items()}
    # raw 0 maps to the driver's initval (0.5); +1.0 in logit space moves it
    # to sigmoid(logit(0.5) + 1*scale), i.e. away from 0.5 but still inside
    # the bounds, so the derived value must move with it.
    key = next(k for k in moved if "comp.drv" in k)
    moved[key] = moved[key] + 1.0

    # ACT
    with caplog.at_level(logging.INFO, logger="exozippy.run"):
        inspect_start(model, system, moved)

    # ASSERT: the derived row moved with the point, and it is still exactly
    # half of what the DRIVER'S OWN node evaluates to at that same point --
    # so the row is the derivation's answer at the given start, not a
    # number from somewhere else.  (The driver's own row deliberately keeps
    # reporting its `initval`, which is the authoritative start of a
    # SAMPLED element; in a real run get_raw_start builds the point from
    # exactly that, so the two agree there.)
    from exozippy.diagnostics import ModelAuditor

    at_point = ModelAuditor(model, system, moved).values_at_start(
        [system._params[0], system._params[1]]
    )
    driver_at_point = float(at_point[id(system._params[0])][0])
    derived_value = _row_value(caplog, "comp.i0.der")

    assert derived_value == pytest.approx(0.5 * driver_at_point)
    assert derived_value != pytest.approx(0.25), derived_value

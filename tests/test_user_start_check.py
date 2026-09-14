"""``ModelAuditor.check_user_starts`` -- the user-start contract.

THE CONTRACT: when a user sets a value, the model produces that value or
says why it cannot.  Before this check the second half did not happen, and
the failure was SILENT: ob09020 pins ``mulensevent.t_E = 76.9`` and starts
at 74.48, and the only way to discover that was to compile the graph by
hand.

These tests drive the classifier with fabricated parameters rather than a
built model, following ``test_known_keys.py``'s ``ModelAuditor.__new__``
pattern -- the numeric and unit logic is what can silently rot, and it does
not need a PyMC model to exercise.

What a fake CANNOT check is that the whole path runs against a real graph,
and that is deliberately not tested here: it needs a built model per
example, which belongs in the acceptance tier rather than in a unit file.
It was verified by sweeping all 19 shipped examples when the check landed
(14 misses, all classified 'approximate'), and any regression in the
compile path shows up there rather than as a silent empty list -- the
bare ``except`` around the compile returns [] by design, so a unit test
asserting "no findings" could never distinguish the two.
"""

import numpy as np
import pytest

from exozippy import diagnostics as diagnostics_mod


class _FakeParam:
    """A Parameter with just the surface check_user_starts touches."""

    def __init__(
        self,
        label,
        names=None,
        unit="",
        factor=1.0,
        n=1,
        active=True,
        derived=False,
        sampled=True,
        value=None,
    ):
        self.label = label
        self.names = names or []
        self.unit = unit
        self._factor = factor  # internal = user * factor
        self._n = n
        self._active = active
        self._derived = derived
        self._sampled = sampled
        self.shape = () if n == 1 else (n,)
        # Opaque by default: the compiled fn is monkeypatched.  A real node
        # is passed only where the graph WALK is under test.
        self.value = object() if value is None else value

    def get_display_label(self, index=0):
        parts = self.label.split(".")
        if self.names and index < len(self.names):
            return f"{parts[0]}.{self.names[index]}.{parts[-1]}"
        if self._n > 1:
            return f"{parts[0]}.{index}.{parts[-1]}"
        return self.label

    def element_is_active(self, index=0):
        return self._active

    def element_is_derived(self, index=0):
        return self._derived

    def element_is_sampled(self, index=0):
        return self._sampled and not self._derived

    def from_internal(self, val, index=None):
        return float(val) / self._factor


class _FakeModel:
    value_vars = []

    def replace_rvs_by_values(self, nodes):
        return list(nodes)


def _auditor(params, user_params, ledger=None, solved_by=None):
    class _CM:
        _last_resolved = dict(ledger or {})
        _last_solved_by = dict(solved_by or {})

    class _Sys:
        config_manager = _CM()

    a = diagnostics_mod.ModelAuditor.__new__(diagnostics_mod.ModelAuditor)
    a.system = _Sys()
    a.model = _FakeModel()
    a.transformed_inits = {}
    a.user_params = user_params
    a.all_params = params
    # __init__ is bypassed, so mirror the one attribute the graph walk
    # reads (a sampled parameter reaches an expression as `<label>_raw`).
    a.hidden_suffixes = ["_raw", "_raw_n", "_raw_u", "_interval__", "_log__"]
    return a


def _with_produced(monkeypatch, values):
    """Make the single compiled function return `values` (internal units)."""

    def _fake_function(inputs, outputs, **kw):
        def _call(*args):
            return [np.asarray(v, dtype=float) for v in values]

        return _call

    monkeypatch.setattr(diagnostics_mod.pytensor, "function", _fake_function)


def test_wrap_if_angle_treats_degrees_as_periodic():
    """
    Given two angles 360 apart,
    When the difference is wrapped,
    Then it is zero -- and a non-angle unit is left alone.

    THIS IS A REGRESSION.  The prototype of this check reported four
    shipped examples as broken because bigomega/alpha came back on the
    other branch (352.57 vs -7.43, 210 vs -150, 337.484 vs -22.516,
    330.4 vs -29.6).  Those are the SAME start.  A warning block with
    false positives in it is a warning block nobody reads.
    """
    wrap = diagnostics_mod.ModelAuditor._wrap_if_angle
    assert wrap(360.0, "deg") == pytest.approx(0.0)
    assert wrap(-360.0, "deg") == pytest.approx(0.0)
    assert wrap(1.5, "deg") == pytest.approx(1.5)
    # Not an angle: 360 really is a difference of 360.
    assert wrap(360.0, "d") == pytest.approx(360.0)
    assert wrap(360.0, "") == pytest.approx(360.0)


def test_a_kept_pin_the_graph_misses_is_reported_as_approximate(monkeypatch):
    """
    Given a pin the LEDGER holds exactly but the graph does not reproduce,
    When the check runs,
    Then it is reported, and the reason is 'approximate', not 'overspecified'.

    This is ob09020's t_E: recorded at rank 100, never overwritten, and the
    model still starts 3.15% away because the seed path is heliocentric
    while the graph is geocentric.  Calling that 'overspecified' would send
    the user hunting for a conflicting pin that does not exist.
    """
    p = _FakeParam("mulensevent.t_E", unit="d")
    a = _auditor(
        [p],
        {"mulensevent.t_E": {"initval": 76.9}},
        ledger={"mulensevent.t_E": 76.9},
    )
    _with_produced(monkeypatch, [74.4767])

    (found,) = a.check_user_starts()
    assert found["key"] == "mulensevent.t_E"
    assert found["reason"] == "approximate"
    assert found["requested"] == pytest.approx(76.9)
    assert found["produced"] == pytest.approx(74.4767)
    assert found["rel"] == pytest.approx(-0.0315, abs=1e-4)


def test_an_overwritten_pin_is_reported_as_overspecified(monkeypatch):
    """
    Given a pin the relaxation engine RESOLVED to something else,
    When the check runs,
    Then the reason is 'overspecified' and the detail names the equation.

    The two reasons need different fixes -- drop a conflicting pin, versus
    accept an approximation -- so a report that cannot tell them apart is
    not actionable.
    """
    p = _FakeParam("orbit.b.m_total", unit="solMass")
    a = _auditor(
        [p],
        {"orbit.b.m_total": {"initval": 2.0}},
        ledger={"orbit.b.m_total": 1.25},
        solved_by={"orbit.b.m_total": "orbit.m_total (standalone solver)"},
    )
    _with_produced(monkeypatch, [1.25])

    (found,) = a.check_user_starts()
    assert found["reason"] == "overspecified"
    assert "1.25" in found["detail"]
    assert "standalone solver" in found["detail"]


def test_the_ledger_is_compared_in_user_units(monkeypatch):
    """
    Given a parameter whose internal unit differs from the file's,
    When the ledger is compared against what the user wrote,
    Then a value the engine KEPT is not reported as overwritten.

    THE RECIPROCAL-FACTOR TRAP.  ``_last_resolved`` is documented at
    config.py:890 as "internal_path -> internal value" while the params
    file is in user units, so comparing them raw calls every converted
    parameter overspecified: star.ra holds 4.6095 rad against a written
    264.105 deg.  The first draft of this check did exactly that.
    """
    # internal = user * pi/180
    p = _FakeParam("star.ra", unit="deg", factor=np.pi / 180.0)
    a = _auditor(
        [p],
        {"star.ra": {"initval": 264.105}},
        ledger={"star.ra": 264.105 * np.pi / 180.0},
    )
    # The graph reproduces it, so there is nothing to report at all.
    _with_produced(monkeypatch, [264.105 * np.pi / 180.0])
    assert a.check_user_starts() == []

    # And when the graph DOES miss, the kept pin is still 'approximate'
    # rather than 'overspecified'.
    _with_produced(monkeypatch, [260.0 * np.pi / 180.0])
    (found,) = a.check_user_starts()
    assert found["reason"] == "approximate"
    assert found["produced"] == pytest.approx(260.0)


def test_an_inactive_element_is_not_reported(monkeypatch):
    """
    Given a user value on an element the build left INACTIVE,
    When the check runs,
    Then nothing is reported.

    An inactive element carries no model node, so a mismatch there is a
    statement about a parameterization the user did not choose.
    """
    p = _FakeParam("orbit.chord", unit="", active=False)
    a = _auditor([p], {"orbit.chord": {"initval": 0.4}})
    _with_produced(monkeypatch, [0.9])
    assert a.check_user_starts() == []


def test_a_value_the_model_delivers_is_silent(monkeypatch):
    """
    Given a user value the graph reproduces,
    When the check runs,
    Then nothing is reported.

    The check must stay quiet on a healthy fit or it becomes noise, and the
    noise floor has to tolerate the last-bit difference between a seed and
    the same number reassembled through the graph.
    """
    p = _FakeParam("star.A.teff", names=["A"], unit="K")
    a = _auditor([p], {"star.A.teff": {"initval": 5800.0}})
    _with_produced(monkeypatch, [5800.0 * (1 + 1e-9)])
    assert a.check_user_starts() == []


def test_a_non_numeric_or_unmatched_entry_is_skipped(monkeypatch):
    """
    Given a params entry that is a string, a bool, or names no parameter,
    When the check runs,
    Then it is skipped rather than crashing.

    check_unused_yaml owns unmatched keys; this check must not
    double-report them, and must not choke on a non-numeric initval.
    """
    p = _FakeParam("star.A.teff", names=["A"], unit="K")
    a = _auditor(
        [p],
        {
            "star.A.teff": {"initval": 5800.0},
            "star.A.spectype": {"initval": "G2V"},
            "star.A.flag": {"initval": True},
            "nosuch.thing.here": {"initval": 1.0},
            "star.A.noinitval": {"sigma": 3.0},
        },
    )
    _with_produced(monkeypatch, [5800.0])
    assert a.check_user_starts() == []


def test_a_compile_failure_does_not_break_startup(monkeypatch):
    """
    Given the diagnostic's own compile raising,
    When the check runs,
    Then it returns empty instead of propagating.

    A diagnostic must never be the reason a fit does not start.
    """
    p = _FakeParam("mulensevent.t_E", unit="d")
    a = _auditor([p], {"mulensevent.t_E": {"initval": 76.9}})

    def _boom(*args, **kw):
        raise RuntimeError("no gradient for this Op")

    monkeypatch.setattr(diagnostics_mod.pytensor, "function", _boom)
    assert a.check_user_starts() == []


def test_an_engine_recorded_contradiction_names_the_other_pin(monkeypatch):
    """
    Given two user pins the same relation links, which cannot both hold,
    When the check classifies the resulting miss,
    Then it says 'overspecified' and NAMES the other pin.

    The mechanism alone is not advice.  examples/galactic_model pinned
    star.BulgeTarget.mass = 0.5 AND star.BulgeTarget.logmass = -0.3, and
    log10(0.5) = -0.30103, so the model started at 10**-0.3 = 0.501187.
    The ledger kept mass = 0.5 untouched, so the value-only reading is
    "the derivation cannot preserve this" -- true of the mechanism, and it
    sends the user looking for a broken derivation instead of at the
    second pin they wrote.  `_relax_equation` already recorded the
    relation and both paths; this asserts the report uses them.
    """
    # Arrange -- the miss, plus the diagnostic the engine files for it.
    p = _FakeParam("star.mass", names=["BulgeTarget"], unit="solMass")
    a = _auditor(
        [p],
        {"star.BulgeTarget.mass": {"initval": 0.5}},
        ledger={"star.0.mass": 0.5},
    )
    a.system.config_manager.diagnostics = [
        {
            "severity": "error",
            "message": (
                "Over-constrained relation 'star.0.mass = "
                "10**star.0.logmass' is violated (relative error "
                "0.002369): every parameter it links was set explicitly, "
                "so no value can be adjusted to satisfy it."
            ),
            "param_paths": ["star.0.logmass", "star.0.mass"],
        }
    ]
    _with_produced(monkeypatch, [0.501187])

    # Act
    (found,) = a.check_user_starts()

    # Assert
    assert found["reason"] == "overspecified", (
        "a relation the engine itself called over-constrained was reported "
        "as an approximation, which points the user at the wrong thing"
    )
    assert "star.0.logmass" in found["detail"], (
        "the report must name the OTHER pin; 'something had to give' "
        "without saying what gave is not actionable"
    )


def test_a_derived_parameter_gets_its_own_remedy_not_approximate(monkeypatch):
    """
    Given an ``initval`` on an element whose value is an EXPRESSION,
    When the resulting miss is classified,
    Then the reason is 'derived' and the detail carries the remedy --
         not 'approximate'.

    THIS IS THE WHOLE OF REVIEW 2.3.17's RESIDUE.  A derived
    ``planet.Companion.mass`` came back at 1/1047 of the requested value
    (1.03762 -> 0.00099051), which is the size measured in job 15408150 and
    essentially the whole 12,107-nat round-trip loss.  Calling that "your
    value was kept, but the derivation reproduces it only approximately"
    understates it by three orders of magnitude AND points the user
    nowhere: there is no channel an ``initval`` on a derived quantity can
    reach, so the fix is to set the sampled parameter(s) it is computed
    from.  ``parameter.py`` already says exactly that about ``sigma: 0``.
    """
    # Arrange -- the derived planet mass, requested and produced as measured.
    p = _FakeParam(
        "planet.mass", names=["Companion"], unit="jupiterMass", derived=True
    )
    a = _auditor(
        [p],
        {"planet.Companion.mass": {"initval": 1.03762}},
        ledger={"planet.0.mass": 1.03762},
    )
    _with_produced(monkeypatch, [0.00099051])

    # Act
    (found,) = a.check_user_starts()

    # Assert -- the reason, the remedy, and the SIZE all survive.
    assert found["reason"] == "derived", (
        "a pin on a derived quantity was classified as an approximation, "
        "which is the misdirection 2.3.17 is about"
    )
    assert "only approximately" not in found["detail"]
    assert "has no effect on a derived parameter" in found["detail"]
    assert found["requested"] == pytest.approx(1.03762)
    assert found["produced"] == pytest.approx(0.00099051)
    assert found["rel"] == pytest.approx(-0.999, abs=1e-3)


def test_the_derived_remedy_is_parameter_pys_own_sentence(monkeypatch):
    """
    Given the ``sigma: 0`` warning build_pymc emits for the same mistake,
    When a derived ``initval`` miss is reported,
    Then both carry the SAME remedy sentence.

    Two spellings of one rule is how these drift: the codebase has said
    "to hold a derived quantity constant, fix the sampled parameter(s)"
    since before this check existed, and the report must not invent a
    second wording for it.  Asserting the SHARED phrase rather than a
    literal of its own is what makes this a drift guard -- reword either
    site and this fails.
    """
    from exozippy.components.parameter import derived_constraint_message

    shared = derived_constraint_message("sigma=0")
    phrase = shared.split(". ", 1)[1].rstrip(".")
    assert "sampled parameter(s)" in phrase  # the remedy, not a typo

    p = _FakeParam("star.logmass", unit="dex(solMass)", derived=True)
    a = _auditor([p], {"star.logmass": {"initval": -0.343}})
    _with_produced(monkeypatch, [-0.2])

    (found,) = a.check_user_starts()
    assert phrase in found["detail"]


def test_the_derived_remedy_names_the_sampled_parameters_it_reads(
    monkeypatch,
):
    """
    Given a derived value whose graph reads two sampled parameters,
    When the remedy is written,
    Then it NAMES them.

    "Fix the sampled parameter(s)" is the class; the names are what make it
    actionable, and they are read off the BUILT GRAPH rather than a
    manifest -- the same source the produced value itself comes from.  Both
    spellings a sampled parameter reaches the graph by are covered: its own
    Deterministic (named for the label) and the ``<label>_raw`` coordinate
    one step below it.
    """
    import pytensor.tensor as pt

    # Arrange -- a derived mass built out of one label-named node and one
    # raw coordinate, plus a sampled parameter it does not read.
    log_q = pt.dscalar("lens.log_q")
    m_raw = pt.dscalar("star.logmass_raw")
    derived_node = 1047.0 * log_q * m_raw

    p = _FakeParam("planet.mass", derived=True, value=derived_node)
    q1 = _FakeParam("lens.log_q", value=log_q)
    q2 = _FakeParam("star.logmass", value=m_raw)
    q3 = _FakeParam("mulensinstrument.flux", value=pt.dscalar("unrelated"))

    a = _auditor([p, q1, q2, q3], {"planet.mass": {"initval": 1.03762}})
    _with_produced(monkeypatch, [0.00099051])

    # Act
    (found,) = a.check_user_starts()

    # Assert
    assert found["reason"] == "derived"
    assert "lens.log_q" in found["detail"]
    assert "star.logmass" in found["detail"]
    assert "mulensinstrument.flux" not in found["detail"], (
        "the remedy named a sampled parameter the expression does not "
        "read, which sends the user to change the wrong number"
    )


def test_a_derived_pin_the_engine_also_overwrote_says_both(monkeypatch):
    """
    Given a derived element whose ledger row the engine ALSO resolved away,
    When the miss is classified,
    Then it reads 'derived' and still reports what the engine resolved it to.

    Both facts are true and they are not alternatives: the remedy is the
    derived one (there is no channel for the request), while the engine's
    value is how far the back-solve got.  Reporting only the second was the
    'overspecified' misdirection; reporting only the first would throw away
    the one number that says where the request went.
    """
    p = _FakeParam("planet.mass", names=["Companion"], derived=True)
    a = _auditor(
        [p],
        {"planet.Companion.mass": {"initval": 1.03762}},
        ledger={"planet.0.mass": 0.5},
        solved_by={"planet.0.mass": "planet.mass = q * star.mass"},
    )
    _with_produced(monkeypatch, [0.00099051])

    (found,) = a.check_user_starts()
    assert found["reason"] == "derived"
    assert "sampled parameter(s)" in found["detail"]
    assert "0.5" in found["detail"]
    assert "planet.mass = q * star.mass" in found["detail"]


def test_a_sampled_parameter_is_still_reported_the_old_way(monkeypatch):
    """
    Given the SAME size of miss on a SAMPLED element,
    When it is classified,
    Then it is 'approximate', exactly as before.

    The new reason is keyed on the element's ROLE and nothing else: leaking
    it onto sampled parameters would tell every user of a seeded start to
    go and fix a parameter that does not exist.
    """
    p = _FakeParam("mulensevent.t_E", unit="d", derived=False)
    a = _auditor(
        [p],
        {"mulensevent.t_E": {"initval": 76.9}},
        ledger={"mulensevent.t_E": 76.9},
    )
    _with_produced(monkeypatch, [74.4767])

    (found,) = a.check_user_starts()
    assert found["reason"] == "approximate"
    assert "has no effect on a derived parameter" not in found["detail"]

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
    ):
        self.label = label
        self.names = names or []
        self.unit = unit
        self._factor = factor  # internal = user * factor
        self._n = n
        self._active = active
        self.shape = () if n == 1 else (n,)
        self.value = object()  # opaque: the compiled fn is monkeypatched

    def get_display_label(self, index=0):
        parts = self.label.split(".")
        if self.names and index < len(self.names):
            return f"{parts[0]}.{self.names[index]}.{parts[-1]}"
        if self._n > 1:
            return f"{parts[0]}.{index}.{parts[-1]}"
        return self.label

    def element_is_active(self, index=0):
        return self._active

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

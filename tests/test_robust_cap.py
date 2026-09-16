"""The Hogg-mixture out_scale cap and its pile-at-cap alarm (review 8.6.3).

On DC2018 event 128 the UNCAPPED mixture fitted out_scale at 300-1000x the
median flux error and forgave 777 nats of caustic-crossing residuals,
inverting a 601-nat preference for the light curve's own rho into a 142-nat
preference for the wrong mode.  Capping out_scale at 10x the median error
per light curve recovered the blind value.  The cap is a MODELLING default,
not a validity limit, so it goes through the manifest ``options`` channel
with the elements the user bounded left alone -- the params file can tighten
OR loosen it -- and a posterior piled against it is reported as an alarm.

Three layers, deliberately light (tests/test_robust_likelihood.py is the
heavy one):

1. The Instrument hooks on a dummy instrument -- the start, the cap, the
   manifest entry, the degenerate-error fallback and the user-bound gap.
2. One real three-file RV system pinning the precedence in BOTH directions.
3. The alarm on a synthetic posterior, firing and staying silent.
"""

import logging

import numpy as np
import pytest

from conftest import _DummyConfigManager
from exozippy.components import likelihood as robust_support
from exozippy.components.instrument import Instrument
from exozippy.components.parameter import Parameter
from exozippy.diagnostics import cap_alarm_findings, log_cap_alarms


class _DummyInstrument(Instrument):
    """Minimal concrete Instrument for unit-testing the shared helpers."""

    @property
    def prefix(self):
        return "dummy"

    def register_parameters(self, system):  # pragma: no cover - not exercised
        pass

    def build_likelihood(self, model, system):  # pragma: no cover
        pass


class _RecordingConfigManager(_DummyConfigManager):
    """Captures the hints a component pushes, and carries user params."""

    def __init__(self, user_params=None):
        self.hints = {}
        self.scale_hints = {}
        self.user_params = dict(user_params or {})

    def add_hint(self, path, value, rank=None):
        self.hints[path] = value

    def add_scale_hint(self, path, scale):
        self.scale_hints[path] = scale


def _two_files(user_params=None):
    """A plain file (5 points, err 1) and a hogg file (7 points, err 3)."""
    cm = _RecordingConfigManager(user_params)
    inst = _DummyInstrument(
        [
            {"name": "plain", "file": "a.rv"},
            {"name": "robust", "file": "b.rv", "likelihood": "hogg"},
        ],
        config_manager=cm,
    )
    err = np.concatenate([np.full(5, 1.0), np.full(7, 3.0)])
    inst_map = np.repeat([0, 1], [5, 7])
    return cm, inst, err, inst_map


# ---------------------------------------------------------------------------
# 1. The Instrument hooks
# ---------------------------------------------------------------------------
def test_prepare_robust_starts_at_the_median_error_and_caps_at_ten_times_it():
    """
    Given a hogg file whose median error is 3 in internal units and a
    user_factor of 2,
    When _prepare_robust seeds the mixture scale,
    Then the start hint (and its scale hint) is 1 x the median error in USER
    units (6.0) and the recorded cap is 10 x it (60.0) -- the ratio being the
    module's own two factors, not a constant restated here.
    """
    cm, inst, err, inst_map = _two_files()

    inst._prepare_robust(err, inst_map, user_factor=2.0)

    assert cm.hints == {"dummy.1.out_scale": pytest.approx(6.0)}
    assert cm.scale_hints == {"dummy.1.out_scale": pytest.approx(6.0)}
    assert inst._robust_scale_caps == {1: pytest.approx(60.0)}
    ratio = robust_support.SCALE_CAP_FACTOR / robust_support.SCALE_START_FACTOR
    assert inst._robust_scale_caps[1] / cm.hints["dummy.1.out_scale"] == (
        pytest.approx(ratio)
    )


def test_register_robust_attaches_the_cap_as_an_option_and_flags_the_alarm():
    """
    Given the caps recorded at stage 1,
    When _register_robust declares the manifest,
    Then out_scale carries the cap as a per-element ``upper`` OPTION (NaN on
    the file that did not opt in), both hogg parameters carry a per-element
    ``cap_alarm`` flag on the opted-in file only, out_frac gets no data
    bound (its 0.5 is defaults.yaml's), and the opt-in sigma pin survives.
    """
    cm, inst, err, inst_map = _two_files()
    inst._prepare_robust(err, inst_map, user_factor=2.0)

    manifest = inst._register_robust({})

    upper = manifest["out_scale"]["upper"]
    assert np.isnan(upper[0]) and upper[1] == pytest.approx(60.0)
    assert manifest["out_scale"]["cap_alarm"] == [False, True]
    assert manifest["out_frac"]["cap_alarm"] == [False, True]
    assert "upper" not in manifest["out_frac"]
    for name in robust_support.LIKELIHOOD_PARAMS["hogg"]:
        pin = manifest[name]["overrides"]["sigma"]
        assert pin[0] == 0.0 and np.isnan(pin[1])


def test_degenerate_errors_leave_the_static_default_and_still_flag():
    """
    Given a hogg file whose errors are all zero,
    When the two hooks run,
    Then no hint and no cap are recorded, the manifest carries no ``upper``
    option at all (defaults.yaml's wide static bound stands), and the alarm
    flag is still set -- that static bound is a cap too.
    """
    cm, inst, err, inst_map = _two_files()
    err[inst_map == 1] = 0.0

    inst._prepare_robust(err, inst_map, user_factor=2.0)
    manifest = inst._register_robust({})

    assert cm.hints == {} and cm.scale_hints == {}
    assert inst._robust_scale_caps == {}
    assert "upper" not in manifest["out_scale"]
    assert manifest["out_scale"]["cap_alarm"] == [False, True]


@pytest.mark.parametrize(
    "key", ["dummy.1.out_scale", "dummy.robust.out_scale", "dummy.out_scale"]
)
def test_an_element_the_user_bounded_is_left_alone_in_every_spelling(key):
    """
    Given a params file bounding out_scale on the hogg file, in any of the
    three spellings resolve() accepts,
    When _register_robust declares the manifest,
    Then that element's cap is NaN -- and since no element is left to cap,
    no ``upper`` option is emitted, so the user's bound reaches
    ConfigManager.resolve untouched (in either direction).
    """
    cm, inst, err, inst_map = _two_files({key: {"upper": 1.0e3}})
    inst._prepare_robust(err, inst_map, user_factor=2.0)

    manifest = inst._register_robust({})

    assert inst._robust_scale_caps == {1: pytest.approx(60.0)}
    assert "upper" not in manifest["out_scale"]
    assert manifest["out_scale"]["cap_alarm"] == [False, True]


def test_a_user_initval_or_sigma_does_not_suppress_the_cap():
    """
    Given a params file that touches out_scale on the hogg file WITHOUT
    bounding it,
    When _register_robust declares the manifest,
    Then the cap is still attached: only an explicit ``upper`` stands the
    component down.
    """
    cm, inst, err, inst_map = _two_files(
        {"dummy.robust.out_scale": {"initval": 5.0, "sigma": 2.0}}
    )
    inst._prepare_robust(err, inst_map, user_factor=2.0)

    manifest = inst._register_robust({})

    assert manifest["out_scale"]["upper"][1] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# 2. Precedence end to end: the params file wins in BOTH directions
# ---------------------------------------------------------------------------
_ERR_MS = 3.0  # m/s, the fixture files' (constant) error bar


def _write_rv(path, seed, n=30):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(2455000.0, 2455400.0, n))
    rv = 30.0 * np.sin(2 * np.pi * t / 17.0) + rng.normal(0, _ERR_MS, n)
    np.savetxt(path, np.column_stack([t, rv, np.full(n, _ERR_MS)]))
    return str(path)


@pytest.fixture(scope="module")
def capped_rv_system(tmp_path_factory):
    """Three hogg RV files: uncapped by the user, tightened, and loosened."""
    from exozippy.system import System

    d = tmp_path_factory.mktemp("robust_cap_rv")
    files = [_write_rv(d / f"{n}.rv", s) for n, s in zip("abc", (31, 32, 33))]
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "rvinstrument": [
            {"name": "default_cap", "file": files[0], "likelihood": "hogg"},
            {"name": "tightened", "file": files[1], "likelihood": "hogg"},
            {"name": "loosened", "file": files[2], "likelihood": "hogg"},
        ],
    }
    params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.05},
        "orbit.b.logP": {"initval": np.log10(17.0)},
        "orbit.b.tc": {"initval": 2455010.0},
        # Below the component's 10x cap (tighten) and above it (loosen).
        "rvinstrument.tightened.out_scale": {"upper": 5.0 * _ERR_MS},
        "rvinstrument.loosened.out_scale": {"upper": 100.0 * _ERR_MS},
    }
    system = System(config, params)
    system.prepare()
    model = system.build_model()
    return system, model


def test_the_cap_and_start_are_the_median_error_multiples_in_user_units(
    capped_rv_system,
):
    """
    Given the built system,
    When the untouched file's out_scale is read back in USER units,
    Then its upper is 10 x the median error and its start 1 x it -- both
    converted through the Parameter's own from_internal, never a
    hand-written factor (CLAUDE.md's reciprocal-factor rule).
    """
    system, _ = capped_rv_system
    p = system.rvinstrument.out_scale

    upper = [float(p.from_internal(p.upper[i], index=i)) for i in range(3)]
    start = [float(p.from_internal(p.initval[i], index=i)) for i in range(3)]

    cap = robust_support.SCALE_CAP_FACTOR * _ERR_MS
    assert upper[0] == pytest.approx(cap, rel=1e-9)
    assert start[0] == pytest.approx(
        robust_support.SCALE_START_FACTOR * _ERR_MS, rel=1e-9
    )
    assert list(np.atleast_1d(p.cap_alarm)) == [True, True, True]


def test_the_params_file_wins_over_the_cap_in_both_directions(
    capped_rv_system,
):
    """
    Given a user upper below the cap on one file and above it on another,
    When the model is built,
    Then each file's resolved upper is the USER's value: the cap neither
    clips the loosened bound (as the ``overrides`` channel would) nor
    replaces the tightened one (as a bare option would), and the model still
    starts inside every bound.
    """
    system, model = capped_rv_system
    p = system.rvinstrument.out_scale
    upper = [float(p.from_internal(p.upper[i], index=i)) for i in range(3)]

    assert upper[1] == pytest.approx(5.0 * _ERR_MS, rel=1e-9)
    assert upper[2] == pytest.approx(100.0 * _ERR_MS, rel=1e-9)
    assert np.isfinite(model.compile_logp()(model.initial_point()))


# ---------------------------------------------------------------------------
# 3. The pile-at-cap alarm
# ---------------------------------------------------------------------------
class _StubSystem:
    def __init__(self, params):
        self._params = params

    def get_all_parameters(self):
        return self._params


def _alarm_param(cap_alarm, n_draws=400):
    """A two-element bounded Parameter with a posterior: element 0 piles in
    the top 5% of its [0, 1] support, element 1 spreads over the middle."""
    p = Parameter(
        label="dummy.out_scale",
        names=["piled", "spread"],
        initval=[0.5, 0.5],
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        cap_alarm=cap_alarm,
    )
    rng = np.random.default_rng(7)
    piled = rng.uniform(0.96, 1.0, n_draws)
    spread = rng.uniform(0.1, 0.9, n_draws)
    p.posterior = np.vstack([piled, spread])  # sample axis LAST, as az.extract
    return p


def test_the_alarm_fires_on_a_posterior_piled_at_a_flagged_cap(caplog):
    """
    Given a flagged element whose draws all sit in the top 5% of its range,
    When the wrap-up check runs,
    Then it reports that element only -- with the cap, the fraction and the
    parameter's user path -- and the WARNING names the remedy.
    """
    p = _alarm_param(cap_alarm=[True, True])

    findings = cap_alarm_findings(_StubSystem([p]))

    assert [f["display"] for f in findings] == ["dummy.piled.out_scale"]
    assert findings[0]["cap"] == pytest.approx(1.0)
    assert findings[0]["frac"] == pytest.approx(1.0)

    with caplog.at_level(logging.WARNING):
        log_cap_alarms(findings, logging.getLogger("test_robust_cap"))
    text = caplog.text
    assert "dummy.piled.out_scale" in text
    assert "dummy.piled.out_scale: {upper: ...}" in text
    assert "wants more freedom than the cap allows" in text


def test_the_alarm_stays_silent_without_a_pile_or_without_a_flag(caplog):
    """
    Given (a) a piled element that is NOT flagged and (b) a flagged element
    whose posterior is spread across its support,
    When the check runs,
    Then neither is reported and nothing is logged.
    """
    unflagged = _alarm_param(cap_alarm=[False, True])
    none_flagged = _alarm_param(cap_alarm=None)

    findings = cap_alarm_findings(_StubSystem([unflagged, none_flagged]))

    assert findings == []
    with caplog.at_level(logging.WARNING):
        log_cap_alarms(findings, logging.getLogger("test_robust_cap"))
    assert caplog.text == ""


def test_the_alarm_needs_a_majority_in_the_top_five_percent():
    """
    Given a flagged element with 40% of its draws in the top 5% of the range,
    When the check runs with the default thresholds,
    Then it is silent (the rule is MORE than half the draws), and a 60%
    pile trips it.
    """
    rng = np.random.default_rng(11)

    def _param(frac_top, n=500):
        p = Parameter(
            label="dummy.out_frac",
            names=["x"],
            initval=[0.1],
            lower=[0.0],
            upper=[0.5],
            cap_alarm=[True],
        )
        k = int(round(frac_top * n))
        top = rng.uniform(0.476, 0.5, k)
        rest = rng.uniform(0.0, 0.47, n - k)
        p.posterior = np.concatenate([top, rest])[None, :]
        return p

    assert cap_alarm_findings(_StubSystem([_param(0.4)])) == []
    hit = cap_alarm_findings(_StubSystem([_param(0.6)]))
    assert len(hit) == 1 and hit[0]["frac"] == pytest.approx(0.6)

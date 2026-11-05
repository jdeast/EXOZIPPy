"""The direct-angle switch's pieces that need no model (exozippy.angles).

The System-level behavior -- the switch firing on a tight angle and not on a
loose one, the overrides, the likelihood at equal physical values, the fold,
the seeds and the gradients -- is in tests/test_direct_angles.py (slow: it
builds models).  These pin the bookkeeping those rest on.
"""

import json

import numpy as np
import pytest
import yaml

from exozippy import angles
from exozippy.components.parameter import Parameter
from exozippy.mkparam import write_param_file

_DEG = np.pi / 180.0


def _angle_param(window, **kw):
    fields = dict(
        label="orbit.bigomega",
        shape=(3,),
        unit="deg",
        internal_unit="rad",
        initval=[10.0, 20.0, 30.0],
        direct_window=window,
    )
    fields.update(kw)
    return Parameter(**fields)


def test_the_window_is_written_in_internal_units_on_direct_elements_only():
    """
    Given a three-element angle in degrees whose element 1 is direct, with its
      window stated in radians,
    When the Parameter is constructed,
    Then element 1's initval/lower/upper/init_scale are the window's, in
      radians, while elements 0 and 2 keep their converted start and no bound
      -- they are still derived from their latents.
    """
    nan = float("nan")
    window = {
        "center": [nan, 2.0, nan],
        "halfwidth": [nan, np.pi, nan],
        "scale": [nan, 0.01, nan],
    }

    par = _angle_param(window)

    assert par.initval[1] == pytest.approx(2.0)
    assert par.lower[1] == pytest.approx(2.0 - np.pi)
    assert par.upper[1] == pytest.approx(2.0 + np.pi)
    assert par.init_scale[1] == pytest.approx(0.01)
    assert par.initval[0] == pytest.approx(10.0 * _DEG)
    assert np.isnan(par.lower[0]) and np.isnan(par.upper[2])


@pytest.mark.parametrize(
    "window, match",
    [
        ({"center": [1.0], "halfwidth": [1.0], "scale": [1.0]}, "entries"),
        (
            {
                "center": [1.0, np.nan, np.nan],
                "halfwidth": [0.0, np.nan, np.nan],
                "scale": [0.1, np.nan, np.nan],
            },
            "positive",
        ),
    ],
)
def test_a_malformed_window_raises(window, match):
    """
    Given a window sized for the wrong element count, or a zero half-width,
    When the Parameter is constructed,
    Then it raises -- the angle layer sizes the window per element, so either
      is an upstream bug, not something to repair.
    """
    with pytest.raises(ValueError, match=match):
        _angle_param(window)


@pytest.mark.parametrize(
    "spec",
    [
        {"pair": ["x"], "expression": "default"},
        {"pair": ["x", "y"]},
        "xbigomega",
    ],
)
def test_a_malformed_declaration_raises(spec):
    """
    Given a defaults.yaml `direction_latents:` that is not {pair: [x, y],
      expression: <block>},
    When it is read,
    Then it raises naming the parameter.
    """
    base = {"orbit": {"bigomega": {"direction_latents": spec}}}
    with pytest.raises(ValueError, match="orbit.bigomega"):
        angles.direction_latents(base, "orbit", "bigomega")


def test_the_shipped_declarations_name_their_latents():
    """
    Given the shipped defaults.yaml files,
    When the three declared angles are read,
    Then each names the latent pair its expression block consumes -- the
      declaration and the expression cannot drift apart unnoticed.
    """
    from exozippy.config import ConfigManager

    base = ConfigManager({}, {}).base_defaults
    for comp, angle in (
        ("orbit", "bigomega"),
        ("orbit", "omega"),
        ("lens", "alpha"),
    ):
        decl = angles.direction_latents(base, comp, angle)
        block = base[comp][angle]["expressions"][decl["expression"]]
        assert block["deps"] == decl["pair"], (comp, angle)


def test_wrap_moves_by_whole_turns_into_the_window():
    """
    Given an angle and a window center,
    When the angle is wrapped,
    Then it lands in [center - pi, center + pi) and differs from the input
      by a whole number of turns.
    """
    center = 3.0
    for a in (-10.0, -3.0, 0.0, 3.0, 6.2, 40.0):
        w = float(angles._wrap(a, center))
        assert center - np.pi <= w < center + np.pi
        turns = (w - a) / (2.0 * np.pi)
        assert turns == pytest.approx(round(turns), abs=1e-12)


def test_the_trace_record_round_trips_exactly():
    """
    Given a choice record with full-precision floats,
    When it is written as the trace stamp and read back,
    Then every center/half-width/scale is bit-identical -- a raw draw decodes
      through exactly these bounds.
    """
    choices = {
        "orbit.bigomega": {
            "elements": [0],
            "center": [2.3782909876543211],
            "halfwidth": [np.pi],
            "scale": [0.0022853],
            "sigma": [0.0022853],
            "forced": [False],
        }
    }
    back = angles.choices_from_json(angles.choices_to_json(choices), "t")
    assert back == choices
    assert (
        back["orbit.bigomega"]["center"][0]
        == choices["orbit.bigomega"]["center"][0]
    )


def test_a_malformed_trace_record_raises():
    """
    Given a stamp missing a field or with fields of different lengths,
    When it is read,
    Then it raises naming the trace -- a guessed window would decode every
      draw wrongly with no visible symptom.
    """
    bad = {"orbit.bigomega": {"elements": [0], "center": [1.0]}}
    with pytest.raises(ValueError, match="lacks"):
        angles.choices_from_json(json.dumps(bad), "trace.nc")
    bad = {
        "orbit.bigomega": {
            "elements": [0],
            "center": [1.0, 2.0],
            "halfwidth": [1.0],
            "scale": [1.0],
        }
    }
    with pytest.raises(ValueError, match="different lengths"):
        angles.choices_from_json(json.dumps(bad), "trace.nc")


def test_a_direct_angle_restarts_as_its_angle_with_no_window(tmp_path):
    """
    Given a trace in which bigomega was SAMPLED directly (it has a raw
      companion; the latents do not exist), and a params file seeding it,
    When mkparam writes the restart file,
    Then the entry is bigomega's MAP initval, with no lower/upper -- the
      window is the component's, re-derived by the next run -- and no
      latent entries.
    """
    from test_mkparam import _make_idata

    (tmp_path / "bh.params.yaml").write_text(
        yaml.dump({"orbit.BH.bigomega": {"initval": 136.236}})
    )
    trace = _make_idata({"orbit.bigomega": 136.2701}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/bh",
        "parameter_file": "bh.params.yaml",
        "orbit": [{"name": "BH"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    result = yaml.safe_load(open(out))
    assert result["orbit.BH.bigomega"] == {"initval": pytest.approx(136.2701)}
    assert not any("xbigomega" in k or "ybigomega" in k for k in result)


def test_periodic_recentering_leaves_a_direct_posterior_alone():
    """
    Given a direct angle's draws, contiguous inside a window centered near the
      (-180, 180] cut (here 178 deg +/- 3, spanning it),
    When the periodic recentering runs before summarizing,
    Then no draw moves: the window already holds the posterior in one piece,
      so the recentering's whole-turn shifts have nothing to do.
    """
    par = Parameter(
        label="orbit.bigomega",
        shape=(1,),
        unit="deg",
        internal_unit="rad",
        initval=[178.0],
        periodic={"value": 360.0, "unit": "deg"},
    )
    rng = np.random.default_rng(3)
    draws = 178.0 + 3.0 * rng.standard_normal((2, 500, 1))
    par.posterior = draws.copy()

    par.recenter_posterior()

    np.testing.assert_array_equal(np.asarray(par.posterior), draws)


def test_a_half_turn_window_halves_the_threshold():
    """
    Given the two window sizes the components ask for,
    When the switch threshold is computed,
    Then a full turn gets DIRECT_ANGLE_MAX_SIGMA and a half turn half of it:
      the seam is half as far away, so the angle must be twice as tight for
      the seam to sit the same number of sigma clear.
    """
    assert angles.threshold(np.pi) == pytest.approx(0.1)
    assert angles.threshold(0.5 * np.pi) == pytest.approx(0.05)


def test_a_lens_companion_reads_its_fitalpha_and_a_bad_value_raises():
    """
    Given lens body entries with 'fitalpha: true' on the companion,
    When the Lens reads its direct-angle requests,
    Then element 1 (the companion) asks for the direct coordinate and the
      primary asks nothing; a non-boolean value raises naming the body.
    """
    from test_mulens_body_config import _DummyConfigManager, _system_config

    from exozippy.components.mulensing.lens import Lens

    lens = [{"body": "star.Lens"}, {"body": "planet.b", "fitalpha": True}]
    cm = _DummyConfigManager(_system_config(lens=lens))

    flag, requests = Lens(lens, cm).direct_angle_requests("alpha")

    assert flag == "fitalpha"
    assert requests == [None, True]
    lens[1]["fitalpha"] = "yes"
    with pytest.raises(ValueError, match="fitalpha"):
        Lens(lens, cm).direct_angle_requests("alpha")

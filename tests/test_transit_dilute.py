"""
Tests for the transit dilution parameter (EXOFASTv2's DILUTE):
  - ``transit.<name>.dilute`` exists only where a file sets fitdilute or
    seddeblend, and scales every deviation from the baseline by 1 - dilute
  - seddeblend constrains it with a Gaussian centered on the SED-predicted
    flux fraction of the other stars, whose width propagates the
    seddeblend_floor fractional flux error (exofast_multised.pro)
  - both are opt-in, as in EXOFASTv2: several stars and an SED alone dilute
    nothing
  - misconfigurations raise
"""

import numpy as np
import pytest

from exozippy.components.transit.transit import Transit
from exozippy.system import System

_T0 = 2459634.3
_PERIOD = 2.99


def _write_lc(path, n=120):
    rng = np.random.default_rng(42)
    t = np.linspace(_T0 - 0.2, _T0 + 0.2, n)
    flux = 1.0 + rng.normal(0.0, 1e-3, n)
    np.savetxt(path, np.column_stack([t, flux, np.full(n, 1e-3)]))
    return str(path)


def _two_star(d, transit_extra=None, ntransit=1):
    """Two identical stars (SED dilution 0.5) and ``ntransit`` V-band files,
    each with seddeblend: true unless ``transit_extra`` says otherwise."""
    sed_file = d / "two_star.sed"
    sed_file.write_text("model: NextGen\nfilters: []\n")
    transits = []
    for i in range(ntransit):
        t = {
            "name": f"inst{i}",
            "file": _write_lc(d / f"lc{i}.dat"),
            "band": "V",
            "seddeblend": True,
        }
        t.update((transit_extra or {}).get(i, {}))
        transits.append(t)
    config = {
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [
            {"name": "V", "filter": "V", "ld_law": "quadratic", "star_ndx": 0}
        ],
        "transit": transits,
        "sed": {"file": str(sed_file)},
    }
    params = {
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _T0},
        "orbit.0.cosi": {"initval": 0.05},
        "planet.0.radius": {"initval": 1.7},
    }
    for s in ("A", "B"):
        params[f"star.{s}.radius"] = {"initval": 1.0, "sigma": 0.05}
        params[f"star.{s}.mass"] = {"initval": 1.0, "sigma": 0.05}
        params[f"star.{s}.teff"] = {"initval": 5800, "sigma": 100}
        params[f"star.{s}.feh"] = {"initval": 0.0, "sigma": 0.08}
    return config, params


def _one_star(d, transit_extra=None):
    t = {"name": "inst0", "file": _write_lc(d / "lc.dat"), "band": "TESS"}
    t.update(transit_extra or {})
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "TESS", "filter": "TESS", "ld_law": "quadratic"}],
        "transit": [t],
    }
    params = {
        "star.0.radius": {"initval": 1.0, "sigma": 0.05},
        "star.0.mass": {"initval": 1.0, "sigma": 0.05},
        "star.0.teff": {"initval": 5800, "sigma": 100},
        "star.0.feh": {"initval": 0.0, "sigma": 0.08},
        "orbit.0.period": {"initval": _PERIOD},
        "orbit.0.tc": {"initval": _T0},
        "orbit.0.cosi": {"initval": 0.05},
        "planet.0.radius": {"initval": 1.7},
    }
    return config, params


def _build(config, params):
    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()
    return system, model


def _eval(model, node):
    """``node`` at the model's start point."""
    import pytensor

    ip = model.initial_point()
    fn = pytensor.function(
        [],
        node,
        givens=[
            (rv, np.asarray(ip[rv.name]))
            for rv in model.free_RVs
            if rv.name in ip
        ],
        on_unused_input="ignore",
        mode="FAST_COMPILE",
    )
    return np.asarray(fn())


def _value(system, model, param, i):
    """Element ``i`` of ``param`` at the start, in its user unit."""
    with model:
        point = system.get_internal_point(model, system.get_raw_start(model))
    return system.transit._point_value(point, param, i)


def test_seddeblend_starts_at_the_sed_with_the_exofast_floor(tmp_path):
    """
    Given two identical stars, an SED and a transit with seddeblend: true,
    When the model is built,
    Then dilute is a sampled parameter STARTED at the starting SED's
      prediction (0.5, no user initval), transit.dilute_sed is 0.5, and the
      seddeblend potential there is exactly -log(sigma) with EXOFASTv2's
      sigma = sqrt(2) * 0.02 * A_D * (1 - A_D).
    """
    config, params = _two_star(tmp_path)
    system, model = _build(config, params)

    assert "dilute" in system.transit.manifest
    assert _value(system, model, system.transit.dilute, 0) == pytest.approx(
        0.5, abs=1e-6
    )
    assert system.transit.deblend_stars == [[0, 1]]
    assert _eval(model, model.named_vars["transit.dilute_sed"]) == (
        pytest.approx([0.5], abs=1e-6)
    )
    sigma = np.sqrt(2.0) * 0.02 * 0.25
    pot = float(_eval(model, model.named_vars["transit.seddeblend"]))
    assert pot == pytest.approx(-np.log(sigma), rel=1e-5)
    assert "SED" in system.transit.dilute.get_prior_str(latex=False)


def test_a_user_initval_overrides_the_sed_start(tmp_path):
    """
    Given the default seddeblend file and a user initval of 0.3,
    When the model is built,
    Then dilute starts at the user's 0.3, not the SED's 0.5.
    """
    config, params = _two_star(tmp_path)
    params["transit.0.dilute"] = {"initval": 0.3}
    system, model = _build(config, params)
    assert _value(system, model, system.transit.dilute, 0) == pytest.approx(
        0.3
    )


def test_seddeblend_floor_sets_the_width(tmp_path):
    """
    Given seddeblend_floor: 0.1 and dilute started 0.01 off the SED's 0.5,
    When the potential is evaluated,
    Then it is the normalized Gaussian with sigma = sqrt(2)*0.1*0.25.
    """
    config, params = _two_star(tmp_path, {0: {"seddeblend_floor": 0.1}})
    params["transit.0.dilute"] = {"initval": 0.51}
    _, model = _build(config, params)
    sigma = np.sqrt(2.0) * 0.1 * 0.25
    expected = -0.5 * (0.01 / sigma) ** 2 - np.log(sigma)
    pot = float(_eval(model, model.named_vars["transit.seddeblend"]))
    assert pot == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("extra", [{"seddeblend": False}, {}])
def test_without_seddeblend_the_file_is_undiluted(tmp_path, extra):
    """
    Given two stars and an SED but seddeblend: false or unstated (the
      default, as in EXOFASTv2: a resolved neighbor, or a light curve
      already deblended upstream, must not be diluted again),
    When the model is built,
    Then there is no dilute parameter and no SED dilution term at all.
    """
    config, params = _two_star(tmp_path)
    config["transit"][0].pop("seddeblend")
    config["transit"][0].update(extra)
    system, model = _build(config, params)
    assert "dilute" not in system.transit.manifest
    assert "transit.dilute_sed" not in model.named_vars
    assert "transit.seddeblend" not in model.named_vars


def test_fitdilute_without_seddeblend_is_free_and_scales_the_depth(tmp_path):
    """
    Given one star, no SED, and fitdilute: true with dilute pinned at 0.3,
    When the light-curve model is evaluated,
    Then every deviation from the baseline is 0.7 x the undiluted one.
    """
    config, params = _one_star(tmp_path, {"fitdilute": True})
    s0, m0 = _build(config, dict(params))
    assert "dilute" in s0.transit.manifest
    assert "transit.seddeblend" not in m0.named_vars
    base = _eval(m0, s0.transit._model_flux_node)

    params["transit.0.dilute"] = {"initval": 0.3, "sigma": 0.0}
    s1, m1 = _build(config, params)
    diluted = _eval(m1, s1.transit._model_flux_node)

    f0 = _value(s0, m0, s0.transit.baseline, 0)
    assert np.ptp(base) > 1e-4  # there is a transit to dilute
    np.testing.assert_allclose(diluted - f0, 0.7 * (base - f0), atol=1e-9)


def test_a_hard_link_ties_two_files_dilutions(tmp_path):
    """
    Given two files with fitdilute (no SED deblending) and EXOFASTv2's
      'dilute_1 dilute_0 0' spelled as a hard link,
    When the model is built,
    Then only one dilution is sampled and the second tracks it.
    """
    config, params = _two_star(
        tmp_path,
        {
            0: {"fitdilute": True, "seddeblend": False},
            1: {"fitdilute": True, "seddeblend": False},
        },
        ntransit=2,
    )
    params["transit.inst0.dilute"] = {"initval": 0.2, "mu": 0.0, "sigma": 0.5}
    params["transit.inst1.dilute"] = {
        "initval": "transit.inst0.dilute",
        "sigma": 0,
    }
    system, model = _build(config, params)
    d0 = _value(system, model, system.transit.dilute, 0)
    d1 = _value(system, model, system.transit.dilute, 1)
    assert d0 == pytest.approx(0.2)
    assert d1 == pytest.approx(d0)


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"fitdilute": 1}, "fitdilute must be true or false"),
        ({"seddeblend": "B"}, "seddeblend must be true, false or a list"),
        ({"seddeblend_floor": 0}, "seddeblend_floor must be a positive"),
        ({"seddeblend": ["C"]}, "unknown star"),
        ({"seddeblend": ["A"]}, "only the host star"),
    ],
)
def test_bad_dilution_config_raises(tmp_path, extra, match):
    config, params = _two_star(tmp_path, {0: extra})
    with pytest.raises(ValueError, match=match):
        _build(config, params)


def test_seddeblend_with_no_sed_raises(tmp_path):
    config, params = _one_star(tmp_path, {"seddeblend": True})
    with pytest.raises(ValueError, match="no sed: block"):
        _build(config, params)


def test_sed_reads_dilution_from_raw_configs():
    assert not Transit.sed_reads_dilution([{}], 2)
    assert Transit.sed_reads_dilution([{"seddeblend": True}], 2)
    assert not Transit.sed_reads_dilution([{"seddeblend": True}], 1)
    assert not Transit.sed_reads_dilution([{"seddeblend": False}], 2)
    assert Transit.sed_reads_dilution(
        [{"seddeblend": False}, {"seddeblend": ["B"]}], 2
    )

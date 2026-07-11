"""
Tests for transit limb darkening living on the Band component:
  - the built model samples band LD (band.q1/q2), not transit-local LD
  - per-instrument band references map observations to the right band
  - a transit instrument without a valid band reference is an error
"""

import numpy as np
import pytest

from exozippy.system import System


def _write_lc(path, t0=2459634.3, period=2.99, n=120):
    """Synthetic flat light curve bracketing one transit window."""
    rng = np.random.default_rng(42)
    t = np.linspace(t0 - 0.2, t0 + 0.2, n)
    flux = 1.0 + rng.normal(0.0, 1e-3, n)
    err = np.full(n, 1e-3)
    np.savetxt(path, np.column_stack([t, flux, err]))
    return str(path)


def _config(lc_files, bands=("TESS",), transit_bands=None):
    transit_bands = transit_bands or ["TESS"] * len(lc_files)
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [
            {"name": b, "filter": "TESS", "ld_law": "quadratic"} for b in bands
        ],
        "transit": [
            {"name": f"inst{i}", "file": f, "band": bname}
            for i, (f, bname) in enumerate(zip(lc_files, transit_bands))
        ],
    }


def _params():
    return {
        "star.0.radius": {"initval": 1.61, "sigma": 0.05},
        "star.0.mass": {"initval": 1.204, "sigma": 0.05},
        "star.0.teff": {"initval": 6207, "sigma": 100},
        "star.0.feh": {"initval": -0.116, "sigma": 0.08},
        "orbit.0.period": {"initval": 2.99},
        "orbit.0.tc": {"initval": 2459634.3},
        "orbit.0.cosi": {"initval": 0.05},
        "planet.0.radius": {"initval": 1.7},
    }


@pytest.fixture(scope="module")
def transit_system(tmp_path_factory):
    """Given a one-instrument transit config referencing a TESS band,
    when the system is prepared and built, provide (system, model)."""
    d = tmp_path_factory.mktemp("transit_band")
    lc = _write_lc(d / "lc.dat")
    system = System(_config([lc]), user_params=_params())
    system.prepare()
    model = system.build_model()
    return system, model


def test_model_samples_band_ld_not_transit_ld(transit_system):
    """
    Given a transit instrument referencing a quadratic-law band,
    When the model is built,
    Then band.q1/band.q2 exist as model variables and no transit-local
    limb-darkening variables remain.
    """
    system, model = transit_system
    names = list(model.named_vars)
    assert any("band.q1" in n for n in names)
    assert any("band.q2" in n for n in names)
    assert not any("transit.q1" in n for n in names)
    assert not any("transit.u1" in n for n in names)


def test_band_map_points_each_instrument_at_its_band(tmp_path_factory):
    """
    Given two transit instruments referencing two different bands,
    When register_parameters runs,
    Then band_map holds each instrument's band index and obs_band_map
    tags every observation with its instrument's band.
    """
    d = tmp_path_factory.mktemp("transit_two_bands")
    lc1 = _write_lc(d / "lc1.dat")
    lc2 = _write_lc(d / "lc2.dat")
    config = _config([lc1, lc2], bands=("TESS", "V"),
                     transit_bands=["V", "TESS"])
    system = System(config, user_params=_params())
    system.prepare()

    tr = system.transit
    band_names = list(system.band.names)
    np.testing.assert_array_equal(
        tr.band_map, [band_names.index("V"), band_names.index("TESS")])
    np.testing.assert_array_equal(
        tr.obs_band_map, tr.band_map[tr.inst_map])


def test_missing_band_reference_raises(tmp_path_factory):
    """
    Given a transit instrument whose band: name matches no band block,
    When the system is prepared,
    Then a ValueError naming the instrument is raised.
    """
    d = tmp_path_factory.mktemp("transit_bad_band")
    lc = _write_lc(d / "lc.dat")
    config = _config([lc], transit_bands=["NoSuchBand"])
    system = System(config, user_params=_params())
    with pytest.raises(ValueError, match="NoSuchBand"):
        system.prepare()


def test_model_logp_is_finite(transit_system):
    """
    Given the built transit model with band-based limb darkening,
    When the initial-point log probability is evaluated,
    Then it is finite.
    """
    system, model = transit_system
    ip = model.initial_point()
    logp = model.compile_logp()(ip)
    assert np.isfinite(logp)


def test_sed_deblending_dilutes_transit_depth(tmp_path_factory):
    """
    Given a two-star system with an SED (no catalog rows; the band's V
    filter feeds the BC grid) and a transit instrument on star A,
    When the model is built,
    Then a transit.dilution Deterministic exists and, with two
    identical stars, evaluates to 0.5 at the initial point.
    """
    d = tmp_path_factory.mktemp("transit_deblend")
    lc = _write_lc(d / "lc.dat")
    sed_file = d / "two_star.sed"
    sed_file.write_text("model: NextGen\nfilters: []\n")

    config = {
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "V", "filter": "V", "ld_law": "quadratic",
                  "star_ndx": 0}],
        "transit": [{"name": "inst0", "file": lc, "band": "V"}],
        "sed": {"file": str(sed_file)},
    }
    params = _params()
    for s in ("A", "B"):
        params[f"star.{s}.radius"] = {"initval": 1.0, "sigma": 0.05}
        params[f"star.{s}.mass"] = {"initval": 1.0, "sigma": 0.05}
        params[f"star.{s}.teff"] = {"initval": 5800, "sigma": 100}
        params[f"star.{s}.feh"] = {"initval": 0.0, "sigma": 0.08}
    params.pop("star.0.radius"); params.pop("star.0.mass")
    params.pop("star.0.teff"); params.pop("star.0.feh")

    system = System(config, user_params=params)
    system.prepare()
    model = system.build_model()

    assert "transit.dilution" in model.named_vars

    ip = model.initial_point()
    logp = model.compile_logp()(ip)
    assert np.isfinite(logp)

    import pytensor
    dil_fn = pytensor.function(
        [], model.named_vars["transit.dilution"],
        givens=[(rv, np.asarray(ip[rv.name]))
                for rv in model.free_RVs if rv.name in ip],
        on_unused_input="ignore", mode="FAST_COMPILE")
    dil = dil_fn()
    assert dil.shape == (1,)
    assert dil[0] == pytest.approx(0.5, abs=1e-6)

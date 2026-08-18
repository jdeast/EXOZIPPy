"""
Tests for the SED-based flux constraints (issue #18):
  - mulensinstrument: per-lightcurve zeropoint tying f_source to the
    SED-predicted source magnitude (KMT-2019-BLG-1806 example as the
    end-to-end case)
  - astrometryinstrument: SED-derived photocenter flux fraction
    (fluxfrac_sed) replacing the sampled fluxfrac

The zeropoint is a DERIVED Parameter, so its Deterministic is the single
vector ``mulensinstrument.zeropoint`` (one element per light curve, in
``system.mulensinstrument.names`` order) and its Gaussian prior is the one
``gaussian_prior.mulensinstrument.zeropoint`` potential Parameter.build_pymc
emits for a derived-with-sigma parameter.  It used to be a hand-built
per-instrument ``mulensinstrument.<name>.zeropoint`` Deterministic plus a
matching ``.zeropoint_prior`` potential; the per-light-curve content is
identical, only the node layout changed.  See
tests/test_data_derived_provenance.py for why it became a Parameter.
"""

from pathlib import Path

import numpy as np
import pytensor
import pytest
import yaml

from exozippy.system import System

_KMT_DIR = Path(__file__).parent.parent / "examples" / "KMT-2019-BLG-1806"


# ---------------------------------------------------------------------------
# Mulensing zeropoint (stage 6)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kmt_system(monkeypatch_module_cwd=None):
    """Given the KMT-2019-BLG-1806 example (2L1S, three I-band KMTNet
    light curves, sed block with no catalog rows), when the system is
    prepared and built, provide (system, model, initial point)."""
    import os

    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")

    cwd = os.getcwd()
    os.chdir(_KMT_DIR)
    try:
        with open("KMT-2019-BLG-1806.yaml") as f:
            config = yaml.safe_load(f)
        with open(config["parameter_file"]) as f:
            user_params = yaml.safe_load(f)
        for k in ("run", "prefix", "parameter_file", "sampler"):
            config.pop(k, None)

        system = System(config, user_params=user_params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model, model.initial_point()


def _eval(model, node, point):
    """Evaluate a model node at ``point``.

    ``replace_rvs_by_values`` is not optional.  A Deterministic's graph is
    written over the RandomVariables, so compiling it against value_vars
    leaves the RVs in place and the function DRAWS FROM THE PRIOR instead of
    reading the point.  Within one process that is invisible -- pytensor.function
    does not advance the shared rng, so every call returns the same draw and
    two nodes compared against each other still agree -- but the value has
    nothing to do with ``point``, and two different models draw differently.
    """
    (node,) = model.replace_rvs_by_values([node])
    f = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    return f(*[point[v.name] for v in model.value_vars])


def _zp_index(system, inst):
    """Vector element of ``mulensinstrument.zeropoint`` for one light curve."""
    return list(system.mulensinstrument.names).index(inst)


def test_zeropoint_deterministic_and_prior_per_lightcurve(kmt_system):
    """
    Given three mulensing instruments with an I band and a sed block,
    When the model is built,
    Then the zeropoint Deterministic carries one element per instrument,
    each with its own value, and a Gaussian prior potential is applied to
    it.
    """
    system, model, point = kmt_system
    assert "mulensinstrument.zeropoint" in model.named_vars
    assert "gaussian_prior.mulensinstrument.zeropoint" in [
        p.name for p in model.potentials
    ]

    zp = np.atleast_1d(
        _eval(model, model["mulensinstrument.zeropoint"], point)
    )
    assert zp.shape == (3,)
    assert np.all(np.isfinite(zp))
    # "its own zeropoint": three different light curves, three values.
    per_inst = {
        inst: float(zp[_zp_index(system, inst)])
        for inst in ("KMTC04", "KMTS04", "KMTA04")
    }
    assert len(set(per_inst.values())) == 3, per_inst


def test_zeropoint_value_matches_manual_computation(kmt_system):
    """
    Given the built KMT model,
    When the KMTC04 zeropoint element is evaluated at the initial point,
    Then it equals m_SED(source, Cousins I) + 2.5*log10(f_source[0]),
    computed independently from the SED prediction node.
    """
    system, model, point = kmt_system
    source_idx = int(system.lens.source_map[0])
    i = _zp_index(system, "KMTC04")

    m_pred = system.sed.predict_star_appmag(source_idx, "Cousins_I", system)
    fs = system.mulensinstrument.f_source.value[i]
    expected = _eval(model, m_pred + 2.5 * pytensor.tensor.log10(fs), point)

    zp = np.atleast_1d(
        _eval(model, model["mulensinstrument.zeropoint"], point)
    )
    assert float(zp[i]) == pytest.approx(float(expected), rel=1e-10)


def test_zeropoint_prior_penalty_scales_with_sigma(kmt_system):
    """
    Given the default 0 +/- 0.2 mag zeropoint prior,
    When the zeropoint prior potential is evaluated at the initial point,
    Then it equals the sum over light curves of -0.5*(zp_i/0.2)^2 -- the
    same per-light-curve penalty the three separate potentials carried.
    """
    system, model, point = kmt_system
    zp = np.atleast_1d(
        _eval(model, model["mulensinstrument.zeropoint"], point)
    )
    pot = [
        p
        for p in model.potentials
        if p.name == "gaussian_prior.mulensinstrument.zeropoint"
    ][0]
    val = _eval(model, pot, point)
    expected = float(np.sum(-0.5 * (zp / 0.2) ** 2))
    assert float(val) == pytest.approx(expected, rel=1e-8)
    # every light curve contributes -- not just the first
    for inst in ("KMTC04", "KMTS04", "KMTA04"):
        assert zp[_zp_index(system, inst)] != 0.0


def test_kmt_model_logp_is_finite(kmt_system):
    """
    Given the full KMT model with the SED flux constraint,
    When the initial-point log probability is evaluated,
    Then it is finite.
    """
    _, model, point = kmt_system
    logp = model.compile_logp()(point)
    assert np.isfinite(logp)


def test_zeropoint_is_unchanged_by_a_flux_format_light_curve(
    kmt_system, tmp_path
):
    """
    Given the same event with every light curve rewritten in flux format
    (F = 10**(-0.4 m), sigma_F = F*sigma_m*ln(10)/2.5),
    When the model is built,
    Then every zeropoint element, and the whole logp, match the
    magnitude-format build to round-off.

    The microlensing likelihood is Gaussian in flux, and f_source lives in the
    data file's own flux system -- which for a magnitude file is the system in
    which F = 10**(-0.4 m).  The zeropoint is exactly what absorbs that
    arbitrary offset, so the SED tie must not notice which format the file was
    written in.  If it ever does, the zp prior (0 +/- 0.2 mag) is silently
    constraining a different quantity than it claims to.
    """
    import os
    import shutil

    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")

    mag_system, mag_model, mag_point = kmt_system

    work = tmp_path / "flux_twin"
    shutil.copytree(_KMT_DIR, work)

    cwd = os.getcwd()
    os.chdir(work)
    try:
        with open("KMT-2019-BLG-1806.yaml") as f:
            config = yaml.safe_load(f)
        with open(config["parameter_file"]) as f:
            user_params = yaml.safe_load(f)
        for k in ("run", "prefix", "parameter_file", "sampler"):
            config.pop(k, None)

        for entry in config["mulensinstrument"]:
            rows = np.loadtxt(entry["file"])
            flux = 10.0 ** (-0.4 * rows[:, 1])
            err = flux * rows[:, 2] * np.log(10.0) / 2.5
            np.savetxt(entry["file"], np.column_stack([rows[:, 0], flux, err]))
            entry["data_format"] = "flux"

        flux_system = System(config, user_params=user_params)
        flux_system.prepare()
        flux_model = flux_system.build_model()
        flux_point = flux_model.initial_point()
    finally:
        os.chdir(cwd)

    name = "mulensinstrument.zeropoint"
    zp_mag = np.atleast_1d(_eval(mag_model, mag_model[name], mag_point))
    zp_flux = np.atleast_1d(_eval(flux_model, flux_model[name], flux_point))
    for inst in ("KMTC04", "KMTS04", "KMTA04"):
        i = _zp_index(mag_system, inst)
        assert _zp_index(flux_system, inst) == i, inst
        assert float(zp_flux[i]) == pytest.approx(
            float(zp_mag[i]), rel=1e-9
        ), inst

    lp_mag = float(np.asarray(mag_model.compile_logp()(mag_point)))
    lp_flux = float(np.asarray(flux_model.compile_logp()(flux_point)))
    assert np.isfinite(lp_mag)
    assert lp_flux == pytest.approx(lp_mag, rel=1e-9)


def test_zeropoint_sigma_zero_raises():
    """
    Given a user override fixing a zeropoint exactly (sigma: 0),
    When the model is built,
    Then a ValueError explains that an exact zeropoint is not allowed.
    """
    import os

    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")

    cwd = os.getcwd()
    os.chdir(_KMT_DIR)
    try:
        with open("KMT-2019-BLG-1806.yaml") as f:
            config = yaml.safe_load(f)
        with open(config["parameter_file"]) as f:
            user_params = yaml.safe_load(f)
        for k in ("run", "prefix", "parameter_file", "sampler"):
            config.pop(k, None)
        user_params["mulensinstrument.KMTC04.zeropoint"] = {
            "initval": 0.0,
            "sigma": 0.0,
        }

        system = System(config, user_params=user_params)
        system.prepare()
        with pytest.raises(ValueError, match="sigma=0"):
            system.build_model()
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# Astrometry SED fluxfrac (stage 6)
# ---------------------------------------------------------------------------


def test_astrometry_fluxfrac_derived_from_sed(tmp_path):
    """
    Given a two-star system (host + luminous companion star) with a sed
    block, a Gaia band, and a gaia-mode instrument configured with
    band + companion_star_ndx,
    When the model is built,
    Then a fluxfrac_sed Deterministic exists, the sampled fluxfrac is
    fixed (not a free variable), and the derived value equals
    F_companion/(F_companion + F_host) computed from the per-star SED
    magnitude predictions.
    """
    from test_astrometry import _TRUTH, _simulate

    tc, epoch = _simulate(tmp_path)
    T = _TRUTH

    sed_file = tmp_path / "empty.sed"
    sed_file.write_text("model: NextGen\nfilters: []\n")

    config = {
        "star": [{"name": "A", "mist": False}, {"name": "C", "mist": False}],
        "planet": [{"name": "BH"}],
        "orbit": [{"name": "BH"}],
        "band": [{"name": "GaiaG", "filter": "GAIA2r.G", "ld_law": "linear"}],
        "astrometryinstrument": [
            {
                "name": "GaiaSim",
                "file": str(tmp_path / "sim.gaia.astrom"),
                "mode": "gaia",
                "observer_location": "earth",
                "epoch": epoch,
                "band": "GaiaG",
                "companion_star_ndx": 1,
            },
        ],
        "sed": {"file": str(sed_file)},
    }
    user_params = {
        "star.A.ra": {"initval": T["ra0"]},
        "star.A.dec": {"initval": T["dec0"]},
        "star.A.pm_ra": {"initval": T["pmra"]},
        "star.A.pm_dec": {"initval": T["pmdec"]},
        "star.C.ra": {"initval": T["ra0"]},
        "star.C.dec": {"initval": T["dec0"]},
        "star.C.pm_ra": {"initval": T["pmra"]},
        "star.C.pm_dec": {"initval": T["pmdec"]},
        "planet.BH.mass": {"initval": T["mcomp"] * 1047.5655},
        "planet.BH.radius": {"initval": 1.0, "sigma": 0},
        "orbit.BH.period": {"initval": T["P"]},
        "orbit.BH.tc": {"initval": tc},
        "orbit.BH.secosw": {"initval": np.sqrt(T["ecc"]) * np.cos(T["w"])},
        "orbit.BH.sesinw": {"initval": np.sqrt(T["ecc"]) * np.sin(T["w"])},
        "orbit.BH.bigomega": {"initval": np.degrees(T["bigom"])},
        "orbit.BH.cosi": {"initval": np.cos(T["inc"])},
    }
    for s in ("A", "C"):
        user_params[f"star.{s}.mass"] = {"initval": 1.0, "sigma": 0.05}
        user_params[f"star.{s}.radius"] = {"initval": 1.0, "sigma": 0.1}
        user_params[f"star.{s}.teff"] = {"initval": 5800, "sigma": 100}
        user_params[f"star.{s}.feh"] = {"initval": 0.0, "sigma": 0.1}
        user_params[f"star.{s}.distance"] = {"initval": 1000.0 / T["plx"]}

    system = System(config, user_params=user_params)
    system.prepare()
    model = system.build_model()
    point = model.initial_point()

    assert "astrometryinstrument.GaiaSim.fluxfrac_sed" in model.named_vars
    # the sampled fluxfrac was fixed (sigma 0 injected), so it is not free
    free_names = [rv.name for rv in model.free_RVs]
    assert "astrometryinstrument.fluxfrac" not in free_names

    beta = _eval(
        model, model["astrometryinstrument.GaiaSim.fluxfrac_sed"], point
    )
    # independently recompute from the per-star mag predictions
    # (host = star_ndx default 0, companion = companion_star_ndx 1)
    m_h = _eval(
        model, system.sed.predict_star_appmag(0, "GAIA2r.G", system), point
    )
    m_c = _eval(
        model, system.sed.predict_star_appmag(1, "GAIA2r.G", system), point
    )
    F_h, F_c = 10 ** (-0.4 * float(m_h)), 10 ** (-0.4 * float(m_c))
    assert float(beta) == pytest.approx(F_c / (F_c + F_h), rel=1e-10)
    assert 0.0 < float(beta) < 1.0

    logp = model.compile_logp()(point)
    assert np.isfinite(logp)

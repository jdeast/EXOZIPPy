"""``data_format: dia`` and the SED zeropoint tie (review 2.2.22a).

The dia reader puts the total flux in UNITS OF THE REFERENCE,
``F = (ref - dflux)/ref``, and records ``zp_ref = zp - 2.5*log10(ref)``:
the instrumental magnitude of a flux ``F`` is ``zp_ref - 2.5*log10(F)``,
not ``-2.5*log10(F)`` as for a magnitude file.  The SED tie used to read
``-2.5*log10(f_source)`` for both, so a dia light curve's zeropoint carried
an extra ``+zp_ref`` (+16 / +19 mag on KMT-2021-BLG-1122L) and the
source-flux seeding read the source ``zp_ref`` mag too bright.

The vehicle is ``examples/KMT-2019-BLG-1806`` (three calibrated Cousins-I
magnitude files and an SED block) written a second time as native pySIS
difference files of the SAME photometry -- ``mag`` column = the magnitude,
pySIS zp 28, the fixed pySIS reference 20.0 mag -- so the expected answer is
"identical": the same zeropoint, the same zeropoint prior term, the same
source seed, and a start logp that differs only by the flux UNIT (the
Gaussian likelihood's normalization and the unit-bearing flux priors).
"""

import copy
import os
import shutil
from pathlib import Path

import numpy as np
import pytensor
import pytest
import yaml

from exozippy.system import System

pytestmark = pytest.mark.slow

_KMT_DIR = Path(__file__).parent.parent / "examples" / "KMT-2019-BLG-1806"
_ZP_PYSIS = 28.0
_REF_MAG = 20.0  # the pySIS pipeline's fixed reference, 1584.9 at zp 28
_ZP_REF = _ZP_PYSIS - 2.5 * np.log10(10 ** (-0.4 * (_REF_MAG - _ZP_PYSIS)))


def _write_dia_twin(path_in, path_out):
    """Native five-column pySIS file of the same photometry as ``path_in``.

    ``mag = 28 - 2.5*log10(ref - dflux)`` holds exactly, and the dflux error
    is the magnitude error propagated at the observed flux -- the same
    propagation the magnitude reader applies -- so the two files are the
    same data in two flux units.
    """
    d = np.loadtxt(path_in)
    t, m, sm = d[:, 0], d[:, 1], d[:, 2]
    f_native = 10 ** (-0.4 * (m - _ZP_PYSIS))
    ref = 10 ** (-0.4 * (_REF_MAG - _ZP_PYSIS))
    np.savetxt(
        path_out,
        np.column_stack(
            [t, ref - f_native, np.log(10.0) / 2.5 * f_native * sm, m, sm]
        ),
        fmt="%.17g",
    )
    return d.shape[0]


def _build(workdir, config, user_params):
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        system = System(copy.deepcopy(config), copy.deepcopy(user_params))
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model


def _eval(model, node, point):
    (node,) = model.replace_rvs_by_values([node])
    f = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    return np.atleast_1d(f(*[point[v.name] for v in model.value_vars]))


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    """Given KMT-2019-BLG-1806 twice: as shipped (magnitude files, each
    zeropoint tied at 0 +/- 0.07, Zang+2023's OGLE-III calibration) and with
    every file replaced by its native pySIS dia twin, read with an explicit
    reference_mag 20 (zp_ref = 20)."""
    if not _KMT_DIR.is_dir():
        pytest.skip("KMT-2019-BLG-1806 example not present")
    workdir = tmp_path_factory.mktemp("dia_zp") / "KMT"
    shutil.copytree(_KMT_DIR, workdir)
    with open(workdir / "KMT-2019-BLG-1806.yaml") as f:
        config = yaml.safe_load(f)
    with open(workdir / config["parameter_file"]) as f:
        user_params = yaml.safe_load(f)
    for k in ("run", "prefix", "parameter_file", "sampler"):
        config.pop(k, None)

    dia = copy.deepcopy(config)
    n_obs = 0
    for entry in dia["mulensinstrument"]:
        out = entry["file"] + ".dia"
        n_obs += _write_dia_twin(workdir / entry["file"], workdir / out)
        entry["file"] = out
        entry["data_format"] = "dia"
        entry["reference_mag"] = _REF_MAG

    mag_sys, mag_model = _build(workdir, config, user_params)
    dia_sys, dia_model = _build(workdir, dia, user_params)
    return mag_sys, mag_model, dia_sys, dia_model, n_obs


def test_the_dia_reader_records_the_reference_offset(twins):
    """Given the dia twin, the flux system's zeropoint is zp_ref = 20 on
    every light curve, and 0 for the magnitude files."""
    mag_sys, _, dia_sys, _, _ = twins
    np.testing.assert_allclose(
        dia_sys.mulensinstrument.flux_system_zp, _ZP_REF, rtol=0, atol=1e-12
    )
    assert _ZP_REF == pytest.approx(20.0)
    assert np.all(mag_sys.mulensinstrument.flux_system_zp == 0.0)


def test_dia_and_magnitude_files_give_the_same_zeropoint(twins):
    """Given the twins, when the zeropoint is evaluated at each model's
    start, then the two agree -- the zeropoint is the calibration of the
    PHOTOMETRY, not of the flux unit it was read in.  Before the fix the dia
    zeropoint was larger by zp_ref = 20 mag, which put the zeropoint prior
    (0 +/- 0.07) tens of thousands of nats off."""
    _, mag_model, _, dia_model, _ = twins
    zp_mag = _eval(
        mag_model,
        mag_model["mulensinstrument.zeropoint"],
        mag_model.initial_point(),
    )
    zp_dia = _eval(
        dia_model,
        dia_model["mulensinstrument.zeropoint"],
        dia_model.initial_point(),
    )
    assert zp_mag.shape == (3,)
    np.testing.assert_allclose(zp_dia, zp_mag, rtol=0, atol=1e-9)


def test_dia_and_magnitude_files_seed_the_same_source(twins):
    """The source-flux seeding reads the source's instrumental magnitude
    through the zeropoint mu; with zp_ref applied both twins seed the same
    star (before the fix the dia source read 20 mag too bright, above the
    whole dwarf locus, and was not seeded at all)."""
    mag_sys, _, dia_sys, _, _ = twins
    src = int(mag_sys.source.star_map[0])
    for param in ("logmass", "teff", "radius"):
        key = f"star.{src}.{param}"
        assert key in mag_sys.config_manager.hints, key
        assert dia_sys.config_manager.hints[key] == pytest.approx(
            mag_sys.config_manager.hints[key], rel=1e-9
        )


def test_start_logp_differs_only_by_the_flux_unit(twins):
    """Given the twins, the start logp's terms agree except where the flux
    UNIT enters: the Gaussian likelihood's normalization (each point's
    sigma is 10**(0.4*zp_ref) larger in the dia unit, -ln of that per
    point) and the priors stated in flux (log_f_total's uniform, f_source's
    soft lower bound).  The zeropoint prior term -- the one this fix is
    about -- is identical."""
    _, mag_model, _, dia_model, n_obs = twins
    mag_terms = mag_model.point_logps(mag_model.initial_point())
    dia_terms = dia_model.point_logps(dia_model.initial_point())
    assert set(mag_terms) == set(dia_terms)
    unit_bearing = {
        "mulensinstrument.model",
        "logit_uniform_prior.mulensinstrument.log_f_total",
        "low_bound.mulensinstrument.f_source",
    }
    zp_term = "gaussian_prior.mulensinstrument.zeropoint"
    assert zp_term in mag_terms
    for name in mag_terms:
        if name in unit_bearing:
            continue
        assert dia_terms[name] == pytest.approx(
            mag_terms[name], rel=1e-9, abs=1e-6
        ), name

    # The likelihood moves by exactly the normalization, i.e. the residuals
    # are the same.  point_logps rounds to 2 decimals, so compare there.
    expected = -n_obs * np.log(10 ** (0.4 * _ZP_REF))
    got = (
        dia_terms["mulensinstrument.model"]
        - mag_terms["mulensinstrument.model"]
    )
    assert got == pytest.approx(expected, abs=0.05)

"""A star position given as a LINK reaches every stage-1 reader as the
linked value, not the defaults.yaml placeholder (review 1.1.10).

Links are applied at stage 4, so ``ConfigManager.resolve()`` answered a
linked ``star.Source.ra: {initval: star.Lens.ra, sigma: 0}`` with the
placeholder (180, 0) deg -- and with ``user_modified`` True, because the
link's ``sigma: 0`` stays in ``user_params``.  Five readers took it
silently: the astrometry tangent point (an abs dataset ~3e8 mas off), the
microlensing line of sight (geocentric frame and Earth-velocity
conversion: KMT-2023-BLG-1454's t_E read 6.394 for 6.438), the
no-microlensing-data line of sight, the galactic-model proper-motion
seeds and -- with the link reversed -- the BJD_TDB conversion.  Pinned
here: a linked config equals its numeric twin in each of those it
exercises, and in the light-curve logp, for both link directions;
and a target with no position at all raises instead of using the
placeholder.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from test_astrometric_microlensing import (
    DEC_DEG,
    RA_DEG,
    T0,
    _base_params,
    _mulens_config,
    _theta_E_truth,
    _write_abs_astrometry,
    _write_photometry,
)

from exozippy.system import System

SOURCE_NDX = 1  # _STARS_CONFIG = [Lens, Source]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("linkedradec")
    t_lc = np.linspace(T0 - 150.0, T0 + 150.0, 400)
    _write_photometry(tmp / "lc.dat", t_lc, _theta_E_truth(), 5.0)
    t_a = np.array([T0 - 100.0, T0, T0 + 100.0, T0 + 400.0])
    z = np.zeros_like(t_a)
    _write_abs_astrometry(tmp / "src.abs", t_a, z, z, 0.1)
    return tmp


def _config(tmp, convert_times=True):
    cfg = _mulens_config()
    # time_scale/time_frame exercise the BJD_TDB conversion's line of sight
    times = {"time_scale": "utc", "time_frame": "hjd"} if convert_times else {}
    cfg["mulensinstrument"] = [
        {"name": "KMTC", "file": str(tmp / "lc.dat"), **times}
    ]
    cfg["astrometryinstrument"] = [
        {
            "name": "HST",
            "file": str(tmp / "src.abs"),
            "mode": "abs",
            "observer_location": "earth",
            "epoch": T0,
            "star_ndx": SOURCE_NDX,
        }
    ]
    return cfg


def _params(link=None):
    """Numeric coordinates on both stars, or one star linked to the other."""
    params = _base_params()
    if link is not None:
        linked, target = link
        for c in ("ra", "dec"):
            params[f"star.{linked}.{c}"] = {
                "initval": f"star.{target}.{c}",
                "sigma": 0,
            }
    return params


def _build(tmp, params):
    system = System(_config(tmp), user_params=params)
    system.prepare()
    model = system.build_model()
    return system, model


def _lc_logp(model):
    obs = [v for v in model.observed_RVs if "mulensinstrument" in v.name]
    assert obs
    return float(model.compile_logp(vars=obs, sum=True)(model.initial_point()))


def _stage1_readings(system, model):
    mi = system.mulensinstrument
    ds = system.astrometryinstrument.datasets[0]
    return {
        "line_of_sight": np.asarray(mi._source_radec_rad),
        "earth_vperp": np.asarray(system.mulensevent._earth_vperp_en(system)),
        "lc_time": np.asarray(mi.time, dtype=float),
        "astro_tangent": np.array([ds["ra_ref"], ds["dec_ref"]]),
        "astro_dE": np.asarray(ds["dE_obs"]),
        "astro_dN": np.asarray(ds["dN_obs"]),
        "lc_logp": np.array([_lc_logp(model)]),
    }


@pytest.fixture(scope="module")
def numeric(files):
    return _stage1_readings(*_build(files, _params()))


@pytest.mark.parametrize(
    "link",
    [("Source", "Lens"), ("Lens", "Source")],
    ids=["src->lens", "lens->src"],
)
def test_linked_position_equals_its_numeric_twin(files, numeric, link):
    linked = _stage1_readings(*_build(files, _params(link)))
    # the twin is at the target, not at the (180, 0) placeholder
    np.testing.assert_allclose(
        np.degrees(numeric["line_of_sight"]), [RA_DEG, DEC_DEG], atol=1e-9
    )
    for key, want in numeric.items():
        np.testing.assert_allclose(
            linked[key], want, rtol=0, atol=1e-9, err_msg=key
        )


@pytest.mark.parametrize("convert_times", [True, False])
def test_a_target_with_no_position_raises(files, convert_times):
    """Without the BJD conversion (which always refused a placeholder), the
    microlensing line of sight used to run silently at (180, 0) deg."""
    params = _base_params()
    for star in ("Source", "Lens"):
        for c in ("ra", "dec"):
            del params[f"star.{star}.{c}"]
    system = System(_config(files, convert_times), user_params=params)
    with pytest.raises(ValueError, match="sky position"):
        system.prepare()

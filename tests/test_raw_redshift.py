"""rvinstrument ``data_format: raw_redshift``: the barycentric correction in
the model (``src/exozippy/components/rvinstrument/rvinstrument.md``, "Raw
redshifts"; ``redshift.py``).

Pinned here:

* the correction itself against barycorrpy (Kanodia & Wright 2018, the
  reference implementation of Wright & Eastman 2014) on four sites x four
  stars x 40 epochs, Barnard's star 25 yr from its epoch among them, from a
  committed fixture (``tests/fixtures/raw_redshift/``; barycorrpy is not a
  dependency);
* the forward model's own consistency: barycorrpy, given the redshifts this
  model predicts for a companion-free star, recovers a CONSTANT velocity;
* the size of what the mode exists to model, on a Gaia-4-like system;
* the spelling and its refusals;
* on the simulated example (``examples/raw_redshift_sim``): the likelihood's
  graph equals the numpy forward model at the truth, its gradients are
  finite on both backends, the plots draw reflex + residual, and a MAP fit
  recovers the injected planet and zero points.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from exozippy.components.rvinstrument.redshift import (
    barycentric_redshift,
    observer_terms,
    predicted_cz,
)
from exozippy.constants import C_MPS

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "raw_redshift_sim"
FIXTURE = ROOT / "tests" / "fixtures" / "raw_redshift" / "barycorrpy_zb.txt"


def _load_make_data():
    spec = importlib.util.spec_from_file_location(
        "raw_redshift_make_data", EXAMPLE / "make_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MD = _load_make_data()


def _read_fixture():
    stars, sites, rows = [], [], []
    for line in FIXTURE.read_text().splitlines():
        if line.startswith("# STAR ") and "name" not in line:
            f = line.split()[2:]
            stars.append((f[0], *map(float, f[1:])))
        elif line.startswith("# SITE "):
            f = line.split()[2:]
            sites.append((f[0], *map(float, f[1:])))
        elif not line.startswith("#"):
            a, b, t, v = line.split()
            rows.append((int(a), int(b), float(t), float(v)))
    return stars, sites, np.array(rows)


def _observer(site, jd_utc, ephemeris="builtin"):
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    _, lon, lat, alt = site
    loc = EarthLocation.from_geodetic(lon, lat, alt)
    t_tdb = Time(jd_utc, format="jd", scale="utc", location=loc).tdb.jd
    return t_tdb, observer_terms(t_tdb, f"{lon},{lat},{alt}", ephemeris)


def _internal(star):
    _, ra, dec, pmra, pmdec, px, rv, ep = star
    return (np.radians(ra), np.radians(dec), pmra, pmdec, px, rv), ep


# ---------------------------------------------------------------------------
# 1. The correction against barycorrpy
# ---------------------------------------------------------------------------


def test_barycentric_redshift_matches_barycorrpy():
    """
    Given barycorrpy's c z_B (DE430) for four sites, four stars (Barnard's
      star at J2016 and at J2000, tau Ceti, a Gaia-4-like star) and 40
      epochs over 2015-2026,
    When redshift.barycentric_redshift computes the same with the builtin
      (erfa) ephemeris -- no download,
    Then every epoch agrees to < 1 cm/s (measured: 0.27 cm/s worst, nearly
      all of it the builtin ephemeris -- with DE430 on both sides it is
      0.12 mm/s above the horizon; the remainder is three refinements this
      model has and barycorrpy lacks, the X(t_B) one barycorrpy's error per
      PINT; rvinstrument.md, "Validation").
    """
    stars, sites, rows = _read_fixture()
    worst = 0.0
    for a, site in enumerate(sites):
        sel = rows[:, 0] == a
        jd = np.unique(rows[sel, 2])
        t_tdb, obs = _observer(site, jd)
        for b, star in enumerate(stars):
            s = sel & (rows[:, 1] == b)
            assert np.array_equal(rows[s, 2], jd)
            st, ep = _internal(star)
            mine = C_MPS * barycentric_redshift(t_tdb, obs, st, ep)
            worst = max(worst, float(np.max(np.abs(mine - rows[s, 3]))))
    assert len(stars) == 4 and len(sites) == 4
    assert worst < 0.01, f"worst |c z_B - barycorrpy| = {worst * 100:.3f} cm/s"


def test_barycorrpy_recovers_a_constant_velocity_from_the_forward_model():
    """
    Given the c z_meas this model PREDICTS for a companion-free star at each
      fixture epoch (the forward direction the likelihood uses),
    When barycorrpy's own correction is applied to it,
      v = c[(1 + z_B)(1 + z_meas) - 1],
    Then v is the star's constant velocity along its reference direction,
      c[gamma_* (1 + rv/c) - 1], to < 1 cm/s at every epoch: the forward
      model and WE14's inverse are one relation, perspective acceleration,
      cross terms and all.
    """
    stars, sites, rows = _read_fixture()
    worst = 0.0
    for a, site in enumerate(sites):
        sel = rows[:, 0] == a
        jd = np.unique(rows[sel, 2])
        t_tdb, obs = _observer(site, jd)
        for b, star in enumerate(stars):
            s = sel & (rows[:, 1] == b)
            st, ep = _internal(star)
            cz, _ = predicted_cz(t_tdb, obs, st, ep)
            v = C_MPS * ((1.0 + rows[s, 3] / C_MPS) * (1.0 + cz / C_MPS) - 1.0)
            worst = max(worst, float(np.ptp(v)))
    assert worst < 0.01, (
        f"peak-to-peak of the recovered v: {worst * 100:.3f} cm/s"
    )


PINT_FIXTURE = FIXTURE.parent / "pint_zb.txt"


@pytest.mark.network
def test_barycentric_redshift_matches_pint_the_regression_reference():
    """
    Given PINT's c z_B (pint-pulsar 1.1.8, DE432s, IAU 2006/2000A with polar
      motion; tests/fixtures/raw_redshift/pint_zb.txt and the script beside
      it, which records the method and every setting) for four sites x
      three stars (tau Ceti, a Gaia-4-like star, Barnard's star) x six
      epochs 2023-2025,
    When redshift.barycentric_redshift computes the same with the same
      kernel,
    Then per (site, star) it differs by a constant plus <= 0.05 mm/s
      (measured 0.032 mm/s worst), and the constant is the site's: -0.08
      to -0.22 mm/s, the same for every star at one site to 0.03 mm/s.
      The constant is a convention, not an error: PINT counts the
      observer's clock in TT (on the geoid), this model in the observer's
      own proper time in the full potential, which differs by
      (U_site - W0)/c -- -0.09 mm/s at McDonald, -0.21 at Keck's 4160 m --
      and any instrument zero point absorbs it.  The residual is the two
      models' genuine differences (the Shapiro form, the Earth-orientation
      interpolation), 100x the float64 time floor (0.0004 mm/s RMS,
      docs/timing_code_crosscheck.md), so the tolerance sits just above
      what was measured: a regression of 0.02 mm/s in any term trips it.

    Needs the DE432s kernel (10 MB; astropy fetches and caches it by name):
    marked network, and skipped -- never run on the builtin ephemeris,
    which differs by mm/s -- when it cannot be loaded.
    """
    import importlib.util

    from astropy.coordinates import solar_system_ephemeris

    try:
        with solar_system_ephemeris.set("de432s"):
            pass
    except Exception as exc:  # no network, no cache
        pytest.skip(f"DE432s kernel unavailable: {exc}")
    spec = importlib.util.spec_from_file_location(
        "make_pint_fixture", PINT_FIXTURE.parent / "make_pint_fixture.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = np.loadtxt(PINT_FIXTURE)
    worst, means = 0.0, {}
    for a, (name, lon, lat, alt) in enumerate(mod.SITES):
        for b, star in enumerate(mod.STARS):
            sel = (rows[:, 0] == a) & (rows[:, 1] == b)
            assert sel.sum() >= 4, (name, star[0])
            t_tdb, obs = _observer(
                (name, lon, lat, alt), rows[sel, 2] + 2400000.5, "de432s"
            )
            st, ep = _internal(star)
            r = C_MPS * barycentric_redshift(t_tdb, obs, st, ep) - rows[sel, 3]
            means.setdefault(a, []).append(float(np.mean(r)))
            worst = max(worst, float(np.max(np.abs(r - np.mean(r)))))
    assert worst < 5e-5, (
        f"worst |EXOZIPPy - PINT - const| = {worst * 1e3:.4f} mm/s"
    )
    for a, m in means.items():
        assert np.ptp(m) < 3e-5, (mod.SITES[a][0], m)
        assert -2.5e-4 < np.mean(m) < -5e-5, (mod.SITES[a][0], m)


def test_a_spacecraft_velocity_is_its_ephemeris_derivative():
    """
    Given the shipped Gaia ephemeris (positions only, 1-day grid),
    When get_observer_posvel reads a velocity off it,
    Then it is the spline's derivative: it matches a central difference of
      the positions to < 1 mm/s, and is the ~30 km/s of an L2 orbiter.
    """
    from exozippy.ephemeris import get_observer_position, get_observer_posvel

    t = np.linspace(2457000.0, 2459000.0, 7)
    _, v = get_observer_posvel(t, "gaia")
    h = 0.01
    dx = get_observer_position(t + h, "gaia") - get_observer_position(
        t - h, "gaia"
    )
    v_fd = dx / (2 * h) * (149597870700.0 / 86400.0)
    assert np.max(np.abs(v - v_fd)) < 1e-3
    assert np.all(np.abs(np.linalg.norm(v, axis=1) - 3.0e4) < 2.0e3)


# ---------------------------------------------------------------------------
# 1b. The photon-weighted exposure average (Tronsgaard et al. 2019)
# ---------------------------------------------------------------------------

EXPOSURE_FIXTURE = FIXTURE.parent / "barycorrpy_exposure.txt"


def test_the_exposure_rules_reproduce_the_photon_moments():
    """
    Given a skewed, cloud-dipped exposure-meter record (500 samples),
    When the three exposure rules are built,
    Then the meter's 3-node Gauss rule reproduces its photon-weighted mean
      of t^k for k <= 5 (relative 1e-12), with positive weights inside the
      record; the 3-node uniform rule integrates t^k exactly for k <= 5
      over the exposure; and the 2-node moment rule has exactly the given
      mean (0), RMS and third central moment.
    """
    from exozippy.components.rvinstrument.redshift import (
        moment_rule,
        uniform_exposure_rule,
        weighted_exposure_rule,
    )

    rng = np.random.default_rng(5)
    t = np.sort(rng.uniform(0.0, 1.0, 500))
    counts = (1.2 - t) * (1.0 - 0.8 * np.exp(-0.5 * ((t - 0.4) / 0.1) ** 2))
    nodes, w = weighted_exposure_rule(t, counts, 3)
    assert w.size == 3 and np.all(w > 0) and np.sum(w) == pytest.approx(1.0)
    assert np.all((nodes > t.min()) & (nodes < t.max()))
    p = counts / counts.sum()
    for k in range(6):
        np.testing.assert_allclose(
            np.sum(w * nodes**k), np.sum(p * t**k), rtol=1e-12
        )
    x, wu = uniform_exposure_rule(3)
    for k in range(6):
        exact = (0.5 ** (k + 1) - (-0.5) ** (k + 1)) / (k + 1)
        assert np.sum(wu * x**k) == pytest.approx(exact, abs=1e-15)
    x, wm = moment_rule(300.0, 2.0e7)
    assert np.sum(wm) == pytest.approx(1.0)
    assert np.sum(wm * x) == pytest.approx(0.0, abs=1e-9)
    assert np.sum(wm * x**2) == pytest.approx(300.0**2)
    assert np.sum(wm * x**3) == pytest.approx(2.0e7)


def _read_exposure_fixture():
    rows = []
    for line in EXPOSURE_FIXTURE.read_text().splitlines():
        if line.startswith("#"):
            continue
        a, b, shape, start, T, tmid, bmid, beff = line.split()
        rows.append(
            (int(a), int(b), shape, *map(float, (start, T, tmid, bmid, beff)))
        )
    return rows


def _effective_cz_b(site, star, jd_utc, w):
    """c z_B,eff = c (1/<1/(1 + z_B)> - 1) over nodes ``jd_utc`` (UTC)
    with weights ``w``: the correction the model's photon-weighted mean
    of 1 + z_meas implies for a companion-free star."""
    t_tdb, obs = _observer(site, np.asarray(jd_utc, dtype=float))
    st, ep = _internal(star)
    zb = barycentric_redshift(t_tdb, obs, st, ep)
    return C_MPS * (1.0 / np.sum(w / (1.0 + zb)) - 1.0)


def test_the_exposure_average_matches_barycorrpy_integrated_densely():
    """
    Given barycorrpy's c z_B integrated over whole exposures (1001-point
      Simpson; tests/fixtures/raw_redshift/barycorrpy_exposure.txt): two
      sites x two stars x hour angles -5..+5 h x 10/30/60 min, for a
      uniform exposure and for one whose flux ramps linearly to zero,
    When the model's rules average the same correction --
      the 3-node Gauss-Legendre rule (an exptime column) and the 2-node
      moment rule (t_rms) for the uniform exposures; the exposure-meter
      rule (2-s bins, 3 nodes) and the moment rule with and without the
      third moment for the ramps --
    Then the averaging SHIFT (exposure mean minus midpoint, up to 1.13 m/s
      here) agrees with barycorrpy's to (measured): 0.0012 mm/s for
      Gauss-Legendre; 0.034 mm/s for the meter (the 2-s bins' box
      assumption, ~dt^2); 0.43 mm/s at 60 min for the uniform moment rule
      and 0.26 mm/s with t_skew3 on the ramps (the unknown 4th moment,
      ~T^4); and the third moment is NOT negligible: without it the ramps
      are off by 9.1 mm/s at 60 min (1.1 mm/s at 30 min).  The absolute
      correction agrees to < 1 mm/s (the builtin ephemeris vs DE430).
    """
    from exozippy.components.rvinstrument.redshift import (
        moment_rule,
        uniform_exposure_rule,
        weighted_exposure_rule,
    )

    stars, sites, _ = _read_fixture()
    worst = {}
    biggest_shift = 0.0
    for a, b, shape, start, T, tmid, bmid, beff in _read_exposure_fixture():
        site, star = sites[a], stars[b]
        d = T / 86400.0
        mid = _effective_cz_b(site, star, [tmid], np.array([1.0]))
        if shape == "uniform":
            x, w = uniform_exposure_rule(3)
            rules = {"gl3": (tmid + x * d, w)}
            x, w = moment_rule(T / np.sqrt(12.0))
            rules["moments"] = (tmid + x / 86400.0, w)
        else:
            nb = int(T / 2.0)
            c = (np.arange(nb) + 0.5) / nb
            x3, w3 = uniform_exposure_rule(3)
            sub = (c[:, None] + x3[None, :] / nb).ravel()
            cnt = ((1.0 - c)[:, None] * w3[None, :]).ravel()
            rules = {"meter": weighted_exposure_rule(start + sub * d, cnt, 3)}
            x, w = moment_rule(T / np.sqrt(18.0), T**3 / 135.0)
            rules["moments_skew"] = (tmid + x / 86400.0, w)
            x, w = moment_rule(T / np.sqrt(18.0), 0.0)
            rules["moments_noskew"] = (tmid + x / 86400.0, w)
        biggest_shift = max(biggest_shift, abs(beff - bmid))
        for name, (tt, w) in rules.items():
            e = _effective_cz_b(site, star, tt, w)
            shift_err = abs((e - mid) - (beff - bmid))
            key = (name, T)
            worst[key] = max(worst.get(key, 0.0), shift_err)
            if name != "moments_noskew":
                assert abs(e - beff) < 1e-3, (name, a, b, T, e - beff)
    assert biggest_shift > 1.0
    for T in (600.0, 1800.0, 3600.0):
        assert worst[("gl3", T)] < 1e-5
        assert worst[("meter", T)] < 5e-5
        assert worst[("moments", T)] < 6e-4
        assert worst[("moments_skew", T)] < 4e-4
    assert worst[("moments_noskew", 3600.0)] > 5e-3


def test_three_nodes_converge_with_a_fast_binary_on_top():
    """
    Given a K = 30 km/s, P = 2 d reflex on top of the Earth's rotation,
      1-hour exposures 5 h from the meridian,
    When the 3-node Gauss-Legendre average is compared with a 2001-point
      Simpson average of the same instantaneous model,
    Then they agree to < 0.05 mm/s (measured: at the float64 time
      resolution floor, ~0.04 mm/s for this K), where the midpoint model
      alone is off by meters per second.
    """
    from exozippy.components.rvinstrument.redshift import uniform_exposure_rule

    stars, sites, _ = _read_fixture()
    st, ep = _internal(stars[1])
    K, P, t0 = 3.0e4, 2.0, 2460400.5

    def refl(tb):
        return (K * np.sin(2.0 * np.pi * (tb - t0) / P), None, None)

    def mean_cz(jd_utc, w):
        t_tdb, obs = _observer(sites[0], jd_utc)
        return np.sum(w * predicted_cz(t_tdb, obs, st, ep, v_reflex=refl)[0])

    x_dense = np.linspace(-0.5, 0.5, 2001)
    w_dense = np.ones(2001)
    w_dense[1:-1:2], w_dense[2:-1:2] = 4.0, 2.0
    w_dense /= w_dense.sum()
    x3, w3 = uniform_exposure_rule(3)
    T = 3600.0 / 86400.0
    for mid in (2460400.41, 2460400.83, 2460401.27):
        ref = mean_cz(mid + x_dense * T, w_dense)
        assert abs(mean_cz(mid + x3 * T, w3) - ref) < 5e-5
        assert abs(mean_cz(np.array([mid]), np.array([1.0])) - ref) > 0.5


# ---------------------------------------------------------------------------
# 2. What the mode models that a corrected-RV fit does not: Gaia-4-like
# ---------------------------------------------------------------------------


def _gaia4_like_terms():
    """The example's epochs (three sites, +/-23 km/s of barycentric
    velocity), its truth, and three versions of the 'corrected' RVs."""
    T = MD.TRUTH
    star = (
        np.radians(T["ra"]),
        np.radians(T["dec"]),
        T["pm_ra"],
        T["pm_dec"],
        T["parallax"],
        T["rv"],
    )
    frozen = star[:2] + (0.0, 0.0) + star[4:]
    out = {k: [] for k in ("inst", "tb", "cz", "cz0", "zb", "zb_frozen")}
    for k, name in enumerate(MD.SITES):
        jd = np.loadtxt(EXAMPLE / f"Gaia4like.{name}.rv")[:, 0]
        cz, tb, t_tdb, obs = MD.predict(name, jd)
        cz0, _, _, _ = MD.predict(name, jd, reflex=False)
        out["inst"].append(np.full(jd.size, k))
        out["tb"].append(tb)
        out["cz"].append(cz)
        out["cz0"].append(cz0)
        out["zb"].append(barycentric_redshift(t_tdb, obs, star, T["epoch"]))
        out["zb_frozen"].append(
            barycentric_redshift(t_tdb, obs, frozen, T["epoch"])
        )
    return {k: np.concatenate(v) for k, v in out.items()}


def _postfit(R, d):
    """RMS of ``d`` left after fitting what a corrected-RV fit would
    absorb anyway: one offset per instrument, a rescaled Keplerian and a
    linear trend."""
    shape = MD.keplerian_rv(R["tb"])
    cols = [(R["inst"] == k).astype(float) for k in range(3)]
    A = np.column_stack(cols + [shape, R["tb"] - R["tb"].mean()])
    x, *_ = np.linalg.lstsq(A, d, rcond=None)
    return float(np.sqrt(np.mean((d - A @ x) ** 2)))


def test_gaia4_like_cross_term_and_pm_consistency_magnitudes():
    """
    Given a Gaia-4-like system (K = 330 m/s, three ground sites, barycentric
      velocities of +/-23 km/s) and the example's epochs,
    When the conventional corrections are compared with the full model,
    Then (rvinstrument.md, "What a corrected-RV fit misses"):
      * applying the correction ADDITIVELY (v = c z_meas + c z_B) instead of
        multiplicatively drops the planet cross term K v_B / c: 1.1 cm/s
        RMS, 1.7 cm/s peak, and essentially none of it is absorbed by a
        refit;
      * computing the correction toward the star's catalog position frozen
        at its epoch (proper motion not applied) is 14 cm/s RMS, 2.9 cm/s
        RMS after a refit absorbs its offset and slope;
      * even the EXACT multiplicative correction with the same, known star
        followed by a Keplerian leaves 2.6 mm/s RMS after the refit: the
        companion's gravitational redshift at the star (1.7-3.5 cm/s, mostly
        absorbed into gamma/K) and the star's own light travel (0.4 mm/s),
        which no corrected-RV fit models.
    """
    R = _gaia4_like_terms()

    def v_true(cz, zb):
        return C_MPS * ((1.0 + zb) * (1.0 + cz / C_MPS) - 1.0)

    berv = C_MPS * R["zb"]
    assert berv.min() < -20e3 and berv.max() > 20e3
    mult = v_true(R["cz"], R["zb"])
    add = R["cz"] + berv
    planet = (mult - add) - (v_true(R["cz0"], R["zb"]) - (R["cz0"] + berv))
    rms = float(np.sqrt(np.mean(planet**2)))
    assert 0.008 < rms < 0.015, rms
    assert 0.012 < np.max(np.abs(planet)) < 0.025
    assert _postfit(R, planet) > 0.8 * rms  # the refit cannot absorb it

    frozen = v_true(R["cz"], R["zb_frozen"]) - mult
    assert 0.10 < np.sqrt(np.mean(frozen**2)) < 0.18
    assert 0.02 < _postfit(R, frozen) < 0.04

    exact = mult - MD.keplerian_rv(R["tb"])
    assert 1e-3 < _postfit(R, exact) < 5e-3


# ---------------------------------------------------------------------------
# 3. The spelling
# ---------------------------------------------------------------------------


def _raw_entry(**over):
    e = {
        "name": "HPF",
        "file": str(EXAMPLE / "Gaia4like.HPF.rv"),
        "data_format": "raw_redshift",
        "observer_location": MD.SITES["HPF"],
        "time_frame": "jd",
        "time_scale": "utc",
        "epoch": MD.TRUTH["epoch"],
        "columns": {"time": 0, "rv": 1, "err": 2, "exptime": 3},
    }
    e.update(over)
    return {k: v for k, v in e.items() if v is not None}


def _cfg(entry):
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "rvinstrument": [entry],
    }


@pytest.mark.parametrize(
    "over, match",
    [
        ({"data_format": "rawz"}, "not one of"),
        ({"observer_location": None}, "needs observer_location"),
        ({"observer_location": "earth"}, "not an observatory"),
        ({"time_location": "-104.0,30.7"}, "second spelling"),
        ({"time_frame": None}, "time_frame: said explicitly"),
        ({"epoch": "J2016"}, "reference epoch"),
        ({"data_format": None}, "belongs to data_format: raw_redshift"),
        ({"ninterp": 0}, "number of Gauss nodes"),
        ({"exposure_meter": 3}, "path of the exposure-meter"),
    ],
)
def test_the_raw_keys_are_validated_at_construction(over, match):
    """
    Given an rvinstrument entry with one raw-redshift key wrong (or a raw
      key on a corrected file),
    When the System is constructed,
    Then it raises naming the key, before any data are read.
    """
    from exozippy.system import System

    with pytest.raises(ValueError, match=match):
        System(_cfg(_raw_entry(**over)), user_params={})


def _prepare(entry, params):
    from exozippy.system import System

    p = {
        "star.A.mass": {"initval": 0.64, "sigma": 0.03},
        "star.A.radius": {"initval": 0.62, "sigma": 0.03},
        "star.A.ra": {"initval": MD.TRUTH["ra"]},
        "star.A.dec": {"initval": MD.TRUTH["dec"]},
        "orbit.b.period": {"initval": MD.TRUTH["period"]},
        "orbit.b.tc": {"initval": MD.TRUTH["tc"]},
    }
    p.update(params)
    system = System(_cfg(entry), user_params=p)
    system.prepare()
    return system


def test_an_exposure_meter_bin_must_find_exactly_one_exposure(tmp_path):
    """
    Given an exposure_meter: file whose t_mid is not any data-file tag (off
      by 0.9 s, ten times the matching tolerance),
    When the system is prepared,
    Then it raises naming the t_mid -- a bin that found no exposure would
      otherwise be silently dropped and its spectrum modeled without it.
    """
    tag = np.loadtxt(EXAMPLE / "Gaia4like.HPF.rv")[0, 0]
    meter = tmp_path / "bad.expmeter"
    meter.write_text(f"{tag + 1e-5:.8f} {tag:.8f} 30.0 100.0\n")
    with pytest.raises(ValueError, match="matches no data-file time tags"):
        _prepare(
            _raw_entry(exposure_meter=str(meter)),
            {"star.A.rv": {"mu": -2.0e4, "sigma": 500.0}},
        )


def test_star_rv_needs_a_prior():
    """
    Given a raw-redshift file and no prior on star.rv,
    When the system is prepared,
    Then it raises: the redshifts constrain only (1 + z0)(1 + rv/c) per
      instrument, and the message names star.rv and a Gaia DR3 prior.
    """
    with pytest.raises(ValueError, match=r"needs a prior on\s+star.0.rv"):
        _prepare(_raw_entry(), {})


def test_the_reference_epoch_is_required_without_astrometry():
    """
    Given a raw-redshift file without epoch: and no absolute astrometry,
    When the system is prepared,
    Then it raises naming epoch: -- star.ra/dec/pm at an unstated epoch
      would be propagated from the wrong instant.
    """
    with pytest.raises(ValueError, match="needs epoch:"):
        _prepare(
            _raw_entry(epoch=None),
            {"star.A.rv": {"mu": -2.0e4, "sigma": 500.0}},
        )


def test_a_raw_file_declares_the_star_kinematics_and_seeds_from_corrected_rvs():
    """
    Given a prepared system with one raw-redshift file,
    When its manifest and seeds are read,
    Then the star carries ra, dec, pm_ra, pm_dec, distance (parallax) and
      rv; the plotted/seed times are BJD_TDB (minutes from the UTC input);
      the seed velocities are the start-corrected ones (within the planet's
      +/-K of the systemic velocity, not the +/-23 km/s raw spread), and
      gamma's hint is their mean.
    """
    s = _prepare(_raw_entry(), {"star.A.rv": {"mu": -2.0e4, "sigma": 500.0}})
    for p in ("ra", "dec", "pm_ra", "pm_dec", "distance", "parallax", "rv"):
        assert p in s.star.manifest
    rvi = s.rvinstrument
    raw_t = np.loadtxt(EXAMPLE / "Gaia4like.HPF.rv")[:, 0]
    assert np.all(np.abs(rvi.time - raw_t) * 86400.0 < 600.0)
    seed = rvi.rv_seed * rvi._rv_factor()
    assert np.ptp(rvi.rv * rvi._rv_factor()) > 10e3
    assert np.all(np.abs(seed - MD.TRUTH["rv"]) < 1.5 * MD.TRUTH["K"] + 500)
    assert rvi.gamma_init[0] == pytest.approx(np.mean(seed))


def test_a_corrected_file_is_untouched():
    """
    Given the same file read as today's corrected RVs (no data_format),
    When the system is prepared,
    Then the seeds read the data array itself (rv_seed IS rv) and the star
      declares none of the raw-mode kinematics -- the byte-identity of
      every shipped RV fit rests on this (rvinstrument.md).
    """
    entry = {
        "name": "HPF",
        "file": str(EXAMPLE / "Gaia4like.HPF.rv"),
        "columns": {"time": 0, "rv": 1, "err": 2},
    }
    s = _prepare(entry, {})
    assert s.rvinstrument.rv_seed is s.rvinstrument.rv
    assert not any(s.rvinstrument.raw)
    for p in ("ra", "pm_ra", "rv", "parallax"):
        assert p not in s.star.manifest


# ---------------------------------------------------------------------------
# 4. The likelihood, on the simulated example (slow: one build)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def example():
    """The example system with every start at the injected truth."""
    import os

    import yaml

    from exozippy.system import System

    cwd = os.getcwd()
    os.chdir(EXAMPLE)
    try:
        cfg = yaml.safe_load(open("raw_redshift_sim.yaml"))
        params = yaml.safe_load(open(cfg["parameter_file"]))
        for name in MD.SITES:
            params[f"rvinstrument.{name}.gamma"] = {
                "initval": MD.gamma_truth(name)
            }
        system = System(cfg, user_params=params)
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    start = system.get_raw_start(model)
    return system, model, start


def _at(system, model, point, node):
    import pytensor

    fn = pytensor.function(model.free_RVs, node, on_unused_input="ignore")
    return np.asarray(fn(*[point[v.name] for v in model.free_RVs]))


@pytest.mark.slow
def test_the_likelihood_is_the_forward_model_at_the_truth(example):
    """
    Given the example built with its starts at the injected truth,
    When the likelihood's model node is evaluated there,
    Then it equals the numpy forward model that simulated the data (before
      noise) to < 1 mm/s -- the residual is the start's logit round trip --
      with every spectrum the photon-weighted mean over its exposure
      (make_data.py integrates each 30-s flux step on its own 9-point
      grid; the model reads an exptime column for HPF, t_rms/t_skew3 for
      NEID and the exposure meter for FIES).
      The numpy model is handed the graph's TARGET-frame conjunction: a raw
      file retards the orbit, so the fitted `tc` is the OBSERVED (virtual
      transit) time, tc_target plus the planet's light travel
      (Orbit._ltt_reporting_mask), hundreds of seconds here.
    """
    system, model, start = example
    rvi = system.rvinstrument
    mod = _at(system, model, start, rvi._rv_model_data_node) * rvi._rv_factor()
    tc_target = float(
        _at(system, model, start, system.orbit.tc_target.value)[0]
    )
    assert abs(tc_target - MD.TRUTH["tc"]) * 86400.0 > 100.0
    truth, instant = [], []
    for name in MD.SITES:
        jd = np.loadtxt(EXAMPLE / f"Gaia4like.{name}.rv")[:, 0]
        exps = MD.exposures(name, jd)
        truth.append(MD.predict(name, jd, tc=tc_target, exps=exps)[0])
        instant.append(MD.predict(name, jd, tc=tc_target)[0])
    truth, instant = np.concatenate(truth), np.concatenate(instant)
    assert np.max(np.abs(mod - truth)) < 1e-3
    # ... and the exposure average is not vacuous: each spectrum's model
    # differs from the instantaneous one at its midpoint by up to 0.2 m/s.
    assert np.max(np.abs(instant - truth)) > 0.1


@pytest.mark.slow
def test_gradients_are_finite_on_both_backends(example):
    """
    Given the example at its start,
    When dlogp is evaluated by the C backend and by JAX,
    Then both are finite in every coordinate -- ra/dec/pm/distance/rv and
      the orbit included -- and agree.
    """
    import pytensor
    from pymc.sampling.jax import get_jaxified_graph

    system, model, _ = example
    ip = model.initial_point()
    grads = pytensor.grad(model.logp(), model.value_vars)
    vals = [ip[v.name] for v in model.value_vars]
    g_c = pytensor.function(model.value_vars, grads)(*vals)
    g_j = get_jaxified_graph(inputs=model.value_vars, outputs=grads)(*vals)
    names = [v.name for v in model.value_vars]
    for want in (
        "star.ra",
        "star.pm_ra",
        "star.distance",
        "star.rv",
        "orbit.tc",
    ):
        assert any(n.startswith(want) for n in names), want
    scale = max(float(np.max(np.abs(g))) for g in g_c)
    for name, gc, gj in zip(names, g_c, g_j):
        gj = np.asarray(gj)
        assert np.all(np.isfinite(gc)) and np.all(np.isfinite(gj)), name
        np.testing.assert_allclose(
            gj, gc, rtol=1e-6, atol=1e-9 * scale, err_msg=name
        )


@pytest.mark.slow
def test_the_plots_draw_reflex_plus_residual(example):
    """
    Given the example at the truth,
    When the plotted observations are formed,
    Then they are gamma + the Keplerian + the likelihood residual: within
      the noise (10 m/s) of gamma + reflex, where the raw data sit +/-23
      km/s away.
    """
    system, model, start = example
    rvi = system.rvinstrument
    with model:
        point = system.get_internal_point(model, start)
    plotted = rvi._plotted_rv(system, point) * rvi._rv_factor()
    gammas = np.array([MD.gamma_truth(n) for n in MD.SITES])[rvi.inst_map]
    expected = gammas + MD.keplerian_rv(rvi.time)
    assert np.std(plotted - expected) < 15.0
    assert np.ptp(rvi.rv * rvi._rv_factor() - expected) > 10e3


@pytest.mark.slow
def test_a_map_fit_recovers_the_injected_planet_and_zero_points(example):
    """
    Given the simulated raw redshifts (K = 330 m/s, 75 spectra at 10 m/s)
      and a start moved off the truth in K and tc,
    When the posterior is maximized,
    Then K and every instrument's gamma come back within ~3 sigma of the
      injection, P within its (loose, 1.3-orbit) constraint, and the star's rv stays on its prior (the data cannot move
      it: rvinstrument.md, "What the data constrain").
    """
    from scipy.optimize import minimize

    system, model, start = example
    rvs = model.free_RVs
    names = [v.name for v in model.value_vars]
    sizes = [np.size(start[v.name]) for v in rvs]
    logp = model.compile_logp()
    dlogp = model.compile_dlogp()

    def unpack(x):
        out, k = {}, 0
        for v, n in zip(rvs, sizes):
            out[v.name] = np.reshape(x[k : k + n], np.shape(start[v.name]))
            k += n
        return out

    x0 = np.concatenate([np.ravel(start[v.name]) for v in rvs]).astype(float)
    k_msini = names.index("planet.msini_raw")
    k_tc = names.index("orbit.tc_raw")
    off = np.cumsum([0] + sizes)
    x0[off[k_msini]] += 0.05
    x0[off[k_tc]] += 0.05
    res = minimize(
        lambda x: -logp(unpack(x)),
        x0,
        jac=lambda x: -np.asarray(dlogp(unpack(x)), dtype=float),
        method="L-BFGS-B",
        options={"maxiter": 5000},
    )
    best = unpack(res.x)
    rvi = system.rvinstrument
    K = (
        float(_at(system, model, best, system.orbit.K.value)[0])
        * rvi._rv_factor()
    )
    P = float(_at(system, model, best, system.orbit.period.value)[0])
    g = _at(system, model, best, rvi.gamma.value) * rvi._rv_factor()
    rv = float(_at(system, model, best, system.star.rv.value)[0])
    print(
        f"MAP: K = {K:.2f} (330), P = {P:.2f} (571), rv = {rv:.0f} "
        f"(-20000), gamma - truth = "
        f"{[round(float(gi - MD.gamma_truth(n)), 2) for n, gi in zip(MD.SITES, g)]}"
    )
    assert abs(K - MD.TRUTH["K"]) < 6.0, K
    # 75 spectra over 750 d cover 1.3 orbits: P is the loosest element (the
    # likelihood is nearly flat over 571-579 d on this noise realization,
    # logp 293.9 at 574.7 d and 295.5 at 578.8 d), K and gamma are not.
    assert abs(P - MD.TRUTH["period"]) < 15.0, P
    for name, gi in zip(MD.SITES, g):
        assert abs(gi - MD.gamma_truth(name)) < 10.0, (name, gi)
    assert abs(rv - MD.TRUTH["rv"]) < 1500.0

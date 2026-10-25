"""The microlensing lens's sky position, derived from the source's (review
8.10.4; mulensing.md "The lens's position is derived from the source's").

The light curve fixes the lens-minus-source separation at every instant
(C9), so a lens star carrying absolute astrometry gets DERIVED ra/dec --
the source's plus mulensevent.lens_offset_e/n -- and the resolved-pair
observable gets its own mode, 'lensrel'.  Pinned here:

1. The flip happens exactly where the lens position is measured, and frees
   the source's position (which the lens now follows).
2. The derived lens track minus the source track IS theta_E times the
   light curve's own trajectory, at epochs from before the event to a
   decade after it (the exactness argument in physics.lens_offset_en).
3. 'lensrel' equals the lens track minus the source track minus the
   source's centroid shift, and its correlated-error likelihood is the
   bivariate normal.
4. The binary-lens host offset is C15's primary relative to the center of
   mass, and a binary builds through the `binary` expression.
5. The galactic model's sight line reads the source when anchored on a
   derived lens.
"""

import logging

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from test_astrometric_microlensing import (
    T0,
    _base_params,
    _compile_at_start,
    _mulens_config,
    _theta_E_truth,
    _write_abs_astrometry,
    _write_photometry,
)

from exozippy.system import System

EPOCH = T0 + 2500.0
# before, at, and well after the event (a decade at most)
T_ASTRO = np.array([T0 - 300.0, T0, T0 + 700.0, T0 + 2500.0, T0 + 4000.0])
CORR = -0.6


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lenstrack")
    t_lc = np.linspace(T0 - 150.0, T0 + 150.0, 400)
    _write_photometry(tmp / "lc.dat", t_lc, _theta_E_truth(), 5.0)
    z = np.zeros_like(T_ASTRO)
    _write_abs_astrometry(tmp / "src.abs", T_ASTRO, z, z, 0.1)
    _write_abs_astrometry(tmp / "lens.abs", T_ASTRO, z, z, 0.1)
    np.savetxt(
        tmp / "pair.dat",
        np.column_stack(
            [
                T_ASTRO,
                z,
                z,
                np.full_like(z, 0.1),
                np.full_like(z, 0.2),
                np.full_like(z, CORR),
            ]
        ),
    )
    return tmp


def _config(tmp, lens_astrometry=True, pair=False):
    cfg = _mulens_config()
    cfg["mulensinstrument"] = [{"name": "OGLE", "file": str(tmp / "lc.dat")}]
    astro = [
        {
            "name": "HSTs",
            "file": str(tmp / "src.abs"),
            "mode": "abs",
            "observer_location": "earth",
            "epoch": EPOCH,
            "star_ndx": 1,
            "microlensing": False,
        }
    ]
    if lens_astrometry:
        astro.append(
            {
                "name": "HSTl",
                "file": str(tmp / "lens.abs"),
                "mode": "abs",
                "observer_location": "earth",
                "epoch": EPOCH,
                "star_ndx": 0,
            }
        )
    if pair:
        astro.append(
            {
                "name": "pair",
                "file": str(tmp / "pair.dat"),
                "mode": "lensrel",
                "observer_location": "earth",
            }
        )
    cfg["astrometryinstrument"] = astro
    return cfg


def _params():
    # Relative pm (4, 6) mas/yr, NOT along an axis, so a swapped or
    # mirrored component shows.  No lens ra/dec: it is derived.
    p = _base_params(mu_lens=(3.0, 4.0), mu_source=(-1.0, -2.0))
    del p["star.Lens.ra"], p["star.Lens.dec"]
    return p


def _build(cfg, params=None):
    system = System(cfg, user_params=params or _params())
    system.prepare()
    return system, system.build_model()


@pytest.fixture(scope="module")
def pair_system(files):
    return _build(_config(files, pair=True))


def test_lens_position_is_derived_only_where_it_is_measured(
    files, pair_system
):
    """
    Given a lens star with an abs dataset,
    When the system is built,
    Then the lens's ra/dec are DERIVED, the source's are SAMPLED (released
      from the microlensing pin, because the lens follows them),
      mulensevent declares lens_offset_e/n, and the lens dataset is
      projected about the source's position (the lens has no position entry
      of its own).  Without lens astrometry none of this happens.
    """
    system, _model = pair_system
    star, ev = system.star, system.mulensevent
    assert ev.lens_track_star(system) == 0
    assert star.ra.element_is_derived(0) and star.dec.element_is_derived(0)
    assert star.ra.element_is_sampled(1) and star.dec.element_is_sampled(1)
    assert "lens_offset_e" in ev.manifest and "lens_offset_n" in ev.manifest
    assert star.position_root(0) == 1 and star.position_root(1) == 1
    d_src, d_lens = system.astrometryinstrument.datasets[:2]
    assert d_lens["ra_ref"] == d_src["ra_ref"]
    assert d_lens["dec_ref"] == d_src["dec_ref"]

    system2, _ = _build(_config(files, lens_astrometry=False))
    assert system2.mulensevent.lens_track_star(system2) is None
    assert "lens_offset_e" not in system2.mulensevent.manifest
    assert not system2.star.ra.element_is_derived(0)


def test_lens_track_is_theta_E_times_the_trajectory(pair_system):
    """
    Given the derived lens track and the (unlensed) source track,
    When both are evaluated at epochs from 300 d before the event to 11 yr
      after it, with the observer's own Skowron deviations,
    Then lens - source == theta_E * dtheta(t) of C9 (the light curve's
      trajectory at the observer) to 1e-6 mas, and at t_0 the separation is
      theta_E * u_0 -- the derivation is the trajectory, not an
      approximation to it.
    """
    system, model = pair_system
    inst, ev = system.astrometryinstrument, system.mulensevent
    with model:
        nodes = []
        for i in (0, 1):
            d = inst.datasets[i]
            beta = inst._sed_beta_node(system, i)
            nodes += list(inst._absolute_model(system, d, d["time"], beta))
        d = inst.datasets[1]
        dev = ev.skowron_deviations(d["time"], d["xyz"])
        tau, beta = ev.get_trajectory(d["time"], dev, system, 0)
        nodes += list(ev._shift_to_sky(tau, beta, ev.theta_E.value[0]))
        nodes += [ev.theta_E.value[0]]
    sE, sN, lE, lN, trN, trE, theta_e = _compile_at_start(model, nodes)
    np.testing.assert_allclose(lE - sE, trE, rtol=0, atol=1e-6)
    np.testing.assert_allclose(lN - sN, trN, rtol=0, atol=1e-6)
    # a decade out the separation is tens of mas, so the test has teeth
    assert np.hypot(trE[-1], trN[-1]) > 50.0
    at_t0 = np.hypot(lE[1] - sE[1], lN[1] - sN[1])
    assert at_t0 == pytest.approx(float(theta_e) * 0.23, rel=1e-3)


def test_lensrel_is_the_pair_difference_minus_the_centroid_shift(
    pair_system,
):
    """
    Given a 'lensrel' dataset at the same epochs as the two abs datasets,
    When its model is evaluated,
    Then it equals lens track - source track - the source's centroid shift
      (the source is measured through its images; the lens is not lensed),
      and the dataset's log-likelihood is the bivariate normal with the
      file's correlation, summed over epochs.
    """
    from scipy.stats import multivariate_normal

    system, model = pair_system
    inst, ev = system.astrometryinstrument, system.mulensevent
    mi = system.mulensinstrument
    with model:
        nodes = []
        for i in (0, 1):
            d = inst.datasets[i]
            beta = inst._sed_beta_node(system, i)
            nodes += list(inst._absolute_model(system, d, d["time"], beta))
        d = inst.datasets[2]
        dev = ev.skowron_deviations(d["time"], d["xyz"])
        nodes += list(inst._lensrel_model(system, 2, d["time"], dev))
        _A, cN, cE = ev.get_astrometric_terms(
            d["time"],
            dev,
            system,
            0,
            *mi._finite_source_limb_darkening(system),
        )
        nodes += [cE, cN]
    sE, sN, lE, lN, rE, rN, cE, cN = _compile_at_start(model, nodes)
    np.testing.assert_allclose(rE, lE - sE - cE, rtol=0, atol=1e-6)
    np.testing.assert_allclose(rN, lN - sN - cN, rtol=0, atol=1e-6)

    obs = [v for v in model.observed_RVs if "model_pair" in v.name]
    assert len(obs) == 2
    lp = float(model.compile_logp(vars=obs, sum=True)(model.initial_point()))
    cov = [[0.01, CORR * 0.02], [CORR * 0.02, 0.04]]  # jitter starts at 0
    ref = sum(
        multivariate_normal(mean=[e, n], cov=cov).logpdf([0.0, 0.0])
        for e, n in zip(rE, rN)
    )
    assert lp == pytest.approx(ref, rel=1e-10)
    assert np.all(np.isfinite(model.compile_dlogp()(model.initial_point())))


def test_host_offset_is_C15_primary_about_the_center_of_mass():
    """
    Given a companion at s (cos alpha, -sin alpha) from the primary on C9's
      (tau_hat, beta_hat) axes (C15),
    When host_offset_from_com is evaluated,
    Then the primary sits at -q/(1+q) of that vector from the center of
      mass -- equivalently, at (-s q/(1+q), 0) in the lens-axis frame the
      Op uses (C12), rotated back by alpha.
    """
    from exozippy.components.mulensing.physics import host_offset_from_com

    s, q = 1.3, 0.25
    for alpha in (0.0, 0.7, 2.5, -1.9):
        # float64 inputs: plain Python floats autocast to float32 in pytensor
        hx, hy = (
            float(v.eval())
            for v in host_offset_from_com(
                np.float64(s), np.float64(alpha), np.float64(q)
            )
        )
        w = q / (1.0 + q)
        np.testing.assert_allclose(
            [hx, hy], [-w * s * np.cos(alpha), w * s * np.sin(alpha)]
        )
        # lens-axis frame (op.py: x = v_tau cos a - v_beta sin a, ...)
        x = hx * np.cos(alpha) - hy * np.sin(alpha)
        y = hx * np.sin(alpha) + hy * np.cos(alpha)
        np.testing.assert_allclose([x, y], [-w * s, 0.0], atol=1e-15)


def test_binary_lens_builds_through_the_binary_expression(files):
    """
    Given a planetary lens whose host carries abs astrometry,
    When the system is built,
    Then lens_offset_e/n use the `binary` expression (the host's offset
      from the center of mass enters) and the model's lens track minus the
      single-lens trajectory is exactly theta_E times that offset.
    """
    cfg = _config(files)
    cfg["planet"] = [{"name": "Lb"}]
    cfg["lens"].append({"body": "planet.Lb", "name": "Lb"})
    params = _params()
    params["planet.Lb.mass"] = {"initval": 10.0}
    params["lens.Lb.s"] = {"initval": 1.2}
    params["lens.Lb.alpha"] = {"initval": 30.0}
    system, model = _build(cfg, params)
    ev, inst = system.mulensevent, system.astrometryinstrument
    assert ev.manifest["lens_offset_e"]["expr_key"] == "binary"
    with model:
        nodes = []
        for i in (0, 1):
            d = inst.datasets[i]
            beta = inst._sed_beta_node(system, i)
            nodes += list(inst._absolute_model(system, d, d["time"], beta))
        d = inst.datasets[1]
        dev = ev.skowron_deviations(d["time"], d["xyz"])
        tau, beta = ev.get_trajectory(d["time"], dev, system, 0)
        hx, hy = ev.host_offset(system)
        nodes += list(
            ev._shift_to_sky(tau + hx, beta + hy, ev.theta_E.value[0])
        )
        nodes += [hx, hy]
    sE, sN, lE, lN, trN, trE, hx, hy = _compile_at_start(model, nodes)
    assert abs(float(hx)) > 1e-3  # q ~ 0.02: the offset is really there
    np.testing.assert_allclose(lE - sE, trE, rtol=0, atol=1e-6)
    np.testing.assert_allclose(lN - sN, trN, rtol=0, atol=1e-6)


def test_galactic_anchor_on_a_derived_lens_reads_the_source(files, caplog):
    """
    Given a galacticmodel anchored on the lens (anchor_idx 0) whose position
      is derived,
    When the system is built,
    Then it builds (the derived element's placeholder start is never read as
      a position) and says that the sight line is the source's.
    """
    cfg = _config(files)
    cfg["galacticmodel"] = [{"anchor_idx": 0}]
    caplog.set_level(logging.INFO)
    system, model = _build(cfg)
    assert any(
        "line of sight is read from star 1" in r.getMessage()
        for r in caplog.records
    )
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_lensrel_corr_out_of_range_raises(files, tmp_path):
    t = np.array([T0 + 2000.0])
    np.savetxt(
        tmp_path / "bad.dat",
        np.column_stack([t, [1.0], [1.0], [0.1], [0.1], [1.0]]),
    )
    cfg = _config(files, lens_astrometry=False)
    cfg["astrometryinstrument"].append(
        {"name": "bad", "file": str(tmp_path / "bad.dat"), "mode": "lensrel"}
    )
    with pytest.raises(ValueError, match="corr"):
        System(cfg, user_params=_params()).prepare()

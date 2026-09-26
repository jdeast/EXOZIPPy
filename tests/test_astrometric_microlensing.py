"""Astrometric microlensing: the point-lens centroid shift inside the
astrometric model (conventions.md C30; review 8.10.1 stage 1).

Three layers, from physics to wiring:

1. The closed form against a DIRECT image solve of the lens equation
   (numpy, no System) -- the formula the code uses is exact.
2. HANDEDNESS: ``MulensEvent.get_centroid_shift`` on a System with a
   lens moving due North and passing East of the source pushes the
   centroid WEST at closest approach and SOUTH afterwards, with the
   closed-form magnitude.  Fails under a mirrored ``beta_hat`` or a
   lens-ward sign.
3. Wiring: theta_E is CONSUMED by the astrometric likelihood (and not by
   an opted-out dataset), the dilution/drag algebra is what
   ``_apply_lens`` says, a binary lens raises and finite source warns,
   and a self-consistent synthetic dataset comes back with chi2 ~ N, a
   finite gradient, and a much worse chi2 at the wrong theta_E.
"""

import logging

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

pytestmark = pytest.mark.slow

from exozippy.system import System

RAD2MAS = 180.0 / np.pi * 3600.0 * 1000.0
RA_DEG, DEC_DEG = 266.4168, -29.0078
T0 = 2460025.0
U0 = 0.23


# ----------------------------------------------------------------------
# 1. The closed form, from the lens equation
# ----------------------------------------------------------------------
def test_point_lens_centroid_closed_form_matches_image_solve():
    """
    Given a point lens and an unresolved source at separation u (Einstein
      units),
    When the two images are solved from the lens equation, each weighted by
      its own magnification, and the flux-weighted centroid is formed,
    Then centroid - source == u/(u^2 + 2) (C30) and centroid - lens ==
      u (u^2 + 3)/(u^2 + 2) (VBM's astrox), and the peak of the shift sits
      at u = sqrt(2) with 0.3536.  The direct centroid-minus-source is a
      catastrophic cancellation at small u (notes/missing_mulens_physics.txt
      3a), so the shift is compared at 1e-10 from u = 0.05 up; the
      centroid itself at 1e-12.
    """
    u = np.concatenate([np.logspace(np.log10(0.05), 1, 200), [np.sqrt(2.0)]])
    y_plus = 0.5 * (u + np.sqrt(u**2 + 4.0))
    y_minus = 0.5 * (u - np.sqrt(u**2 + 4.0))
    a_plus = 1.0 / np.abs(1.0 - y_plus**-4)
    a_minus = 1.0 / np.abs(1.0 - y_minus**-4)
    centroid_from_lens = (a_plus * y_plus + a_minus * y_minus) / (
        a_plus + a_minus
    )
    shift = centroid_from_lens - u

    np.testing.assert_allclose(shift, u / (u**2 + 2.0), rtol=1e-10, atol=0)
    np.testing.assert_allclose(
        centroid_from_lens, u * (u**2 + 3.0) / (u**2 + 2.0), rtol=1e-12
    )
    # the total magnification is Paczynski's, as a sanity check on A_+-
    np.testing.assert_allclose(
        a_plus + a_minus, (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0)), rtol=1e-12
    )
    assert shift[-1] == pytest.approx(np.sqrt(2.0) / 4.0, rel=1e-12)
    assert np.argmax(shift) == len(u) - 1  # the peak IS u = sqrt(2)


# ----------------------------------------------------------------------
# System fixtures
# ----------------------------------------------------------------------
_STARS_CONFIG = [{"name": "Lens"}, {"name": "Source"}]


def _mulens_config(**event_extra):
    return {
        "name": "astromulens",
        "star": _STARS_CONFIG,
        "mulensevent": [{"mmexofast": False, **event_extra}],
        "lens": [{"body": "star.Lens", "name": "Lens"}],
        "source": [{"body": "star.Source", "name": "Source"}],
    }


def _base_params(mu_lens=(0.0, 5.0), mu_source=(0.0, 0.0)):
    """Lens 0.5 Msun at 4 kpc, source at 8 kpc: theta_E = 0.7135 mas.

    Default relative proper motion is DUE NORTH (lens pm_dec = +5 mas/yr,
    source at rest), so tau_hat = North and beta_hat = East (C9): with
    u_0 > 0 the lens passes EAST of the source.
    """
    return {
        "source.Source.t_0": {"initval": T0},
        "source.Source.u_0": {"initval": U0},
        "star.Lens.distance": {"initval": 4000.0},
        "star.Source.distance": {"initval": 8000.0},
        "star.Lens.mass": {"initval": 0.5},
        "star.Lens.pm_ra": {"initval": mu_lens[0]},
        "star.Lens.pm_dec": {"initval": mu_lens[1]},
        "star.Source.pm_ra": {"initval": mu_source[0]},
        "star.Source.pm_dec": {"initval": mu_source[1]},
        "star.Source.ra": {"initval": RA_DEG},
        "star.Source.dec": {"initval": DEC_DEG},
        "star.Lens.ra": {"initval": RA_DEG},
        "star.Lens.dec": {"initval": DEC_DEG},
    }


def _compile_at_start(model, nodes):
    """Evaluate symbolic ``nodes`` at the model's initial point."""
    if not isinstance(nodes, (list, tuple)):
        nodes = [nodes]
    fn = pytensor.function(
        model.free_RVs, list(nodes), on_unused_input="ignore"
    )
    ip = model.initial_point()
    return fn(*[ip[v.name] for v in model.free_RVs])


def _paczynski(u):
    return (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))


def _write_photometry(
    path, t, theta_e_mas, mu_rel_mas_yr, f_s=1.0e-8, f_b=0.5e-8
):
    """A PSPL light curve in magnitudes with the truth's t_0/u_0/t_E and no
    parallax; the NNLS flux decomposition recovers f_s/f_b from it."""
    t_e = theta_e_mas / (mu_rel_mas_yr / 365.25)
    u = np.hypot((t - T0) / t_e, U0)
    flux = f_s * _paczynski(u) + f_b
    mag = -2.5 * np.log10(flux)
    err = np.full_like(t, 0.01)
    np.savetxt(path, np.column_stack([t, mag, err]))


def _write_abs_astrometry(path, t, dE_mas, dN_mas, err_mas):
    ra_obs = (
        RA_DEG + dE_mas / RAD2MAS / np.cos(np.radians(DEC_DEG)) * 180 / np.pi
    )
    dec_obs = DEC_DEG + dN_mas / RAD2MAS * 180 / np.pi
    np.savetxt(
        path,
        np.column_stack(
            [
                t,
                ra_obs,
                dec_obs,
                np.full_like(t, err_mas),
                np.full_like(t, err_mas),
            ]
        ),
    )


def _theta_E_truth():
    from exozippy.constants import KAPPA

    pi_rel = 1000.0 / 4000.0 - 1000.0 / 8000.0
    return float(np.sqrt(KAPPA * 0.5 * pi_rel))


@pytest.fixture(scope="module")
def pspl_system():
    """Photometry-free PSPL event (the handedness test needs no data)."""
    system = System(_mulens_config(), user_params=_base_params())
    system.prepare()
    model = system.build_model()
    return system, model


# ----------------------------------------------------------------------
# 2. Handedness
# ----------------------------------------------------------------------
def test_centroid_shift_points_away_from_the_lens(pspl_system):
    """
    Given a lens moving due North that passes EAST of the source (u_0 > 0
      under C9), evaluated with zero observer deviations (no parallax),
    When get_centroid_shift is evaluated at closest approach and well after,
    Then at t_0 the shift is WEST (delta_E < 0, delta_N ~ 0) with magnitude
      theta_E u_0/(u_0^2 + 2), and after t_0 -- lens now NORTH of the source
      -- the shift has a SOUTHward component (delta_N < 0); and the vector
      equals -theta_E dtheta/(u^2+2) built from C9's basis to 1e-12.
    """
    system, model = pspl_system
    event = system.mulensevent
    t = np.array([T0, T0 + 40.0, T0 - 40.0, T0 + 400.0])
    zero_dev = np.zeros((len(t), 3))

    with model:
        dN_node, dE_node = event.get_centroid_shift(
            t, zero_dev, system, index=0
        )
        A_node = event.get_magnification(t, zero_dev, system, index=0)
        nodes = [
            dN_node,
            dE_node,
            A_node,
            event.theta_E.value[0],
            event.t_E.value[0],
            event.mu_ra_rel_geo.value[0],
            event.mu_dec_rel_geo.value[0],
        ]
    dN, dE, A, theta_e, t_e, mu_e, mu_n = _compile_at_start(model, nodes)
    theta_e, t_e = float(theta_e), float(t_e)

    # the lens is 0.5 Msun at 4 kpc in front of an 8 kpc source
    assert theta_e == pytest.approx(_theta_E_truth(), rel=1e-9)

    # closest approach: lens due East of the source -> centroid pushed WEST
    assert dE[0] < 0.0
    assert abs(dN[0]) < 1e-9 * theta_e
    assert abs(dE[0]) == pytest.approx(theta_e * U0 / (U0**2 + 2.0), rel=1e-9)
    # later: lens has moved North of the source -> a SOUTHward component
    assert dN[1] < 0.0 and dN[3] < 0.0
    # earlier: lens was South -> Northward
    assert dN[2] > 0.0
    # far from the event the shift decays as theta_E/u, not as A - 1
    u_far = np.hypot((t[3] - T0) / t_e, U0)
    assert np.hypot(dE[3], dN[3]) == pytest.approx(
        theta_e * u_far / (u_far**2 + 2.0), rel=1e-9
    )
    assert A[3] - 1.0 < 1e-3 and np.hypot(dE[3], dN[3]) > 0.05 * theta_e

    # the full vector from C9's basis, mu_rel,geo taken from the model
    mu_mag = np.hypot(mu_e, mu_n)
    tau_n, tau_e = mu_n / mu_mag, mu_e / mu_mag
    beta_n, beta_e = -tau_e, tau_n
    tau = (t - T0) / t_e
    dth_n = U0 * beta_n + tau * tau_n
    dth_e = U0 * beta_e + tau * tau_e
    u2 = dth_n**2 + dth_e**2
    np.testing.assert_allclose(dN, -theta_e * dth_n / (u2 + 2.0), atol=1e-12)
    np.testing.assert_allclose(dE, -theta_e * dth_e / (u2 + 2.0), atol=1e-12)
    # and the magnification is Paczynski's on the same |u|
    np.testing.assert_allclose(A, _paczynski(np.sqrt(u2)), rtol=1e-12)


def test_centroid_shift_rotates_with_the_relative_proper_motion():
    """
    Given the same event with mu_rel pointing due EAST instead of North,
    When the shift is evaluated at closest approach,
    Then it is due NORTH -- beta_hat = tau_hat rotated +90 deg North through
      East puts the lens SOUTH of the source for u_0 > 0, so the centroid
      is pushed North.  (Pins the rotation sense; a mirrored beta_hat gives
      South.)
    """
    system = System(
        _mulens_config(), user_params=_base_params(mu_lens=(5.0, 0.0))
    )
    system.prepare()
    model = system.build_model()
    t = np.array([T0])
    with model:
        dN_node, dE_node = system.mulensevent.get_centroid_shift(
            t, np.zeros((1, 3)), system, index=0
        )
    dN, dE = _compile_at_start(model, [dN_node, dE_node])
    assert dN[0] > 0.0
    assert abs(dE[0]) < 1e-9 * _theta_E_truth()


# ----------------------------------------------------------------------
# 3. Wiring through the astrometry instrument
# ----------------------------------------------------------------------
def _full_config(
    tmp_dir, astro_extra=None, event_extra=None, with_photometry=True
):
    cfg = _mulens_config(**(event_extra or {}))
    if with_photometry:
        cfg["mulensinstrument"] = [
            {"name": "OGLE", "file": str(tmp_dir / "lc.dat")}
        ]
    cfg["astrometryinstrument"] = [
        {
            "name": "Roman",
            "file": str(tmp_dir / "astrom.abs"),
            "mode": "abs",
            "observer_location": "earth",
            "epoch": T0,
            "star_ndx": 1,
            **(astro_extra or {}),
        }
    ]
    return cfg


def _astrometric_times():
    # two "seasons" straddling the event plus a late baseline
    rng = np.random.default_rng(3)
    return np.sort(
        np.concatenate(
            [
                rng.uniform(T0 - 120.0, T0 + 120.0, 60),
                rng.uniform(T0 + 300.0, T0 + 420.0, 20),
            ]
        )
    )


@pytest.fixture(scope="module")
def lensed_files(tmp_path_factory):
    """Light curve + a PLACEHOLDER astrometry file (zero offsets) at the
    truth's geometry; the recovery test overwrites the astrometry with the
    model's own prediction plus noise."""
    tmp_dir = tmp_path_factory.mktemp("astromulens")
    t_lc = np.linspace(T0 - 150.0, T0 + 150.0, 400)
    _write_photometry(tmp_dir / "lc.dat", t_lc, _theta_E_truth(), 5.0)
    t_a = _astrometric_times()
    _write_abs_astrometry(
        tmp_dir / "astrom.abs",
        t_a,
        np.zeros_like(t_a),
        np.zeros_like(t_a),
        0.1,
    )
    return tmp_dir, t_a


def _build(cfg, params=None):
    system = System(cfg, user_params=params or _base_params())
    system.prepare()
    model = system.build_model()
    return system, model


def _astro_logp(model, point, name="Roman"):
    obs = [
        v
        for v in model.observed_RVs
        if v.name
        in (
            f"astrometryinstrument.model_{name}_E",
            f"astrometryinstrument.model_{name}_N",
        )
    ]
    assert len(obs) == 2
    return float(model.compile_logp(vars=obs, sum=True)(point))


def test_lensed_dataset_registers_blend_offsets_and_consumes_theta_E(
    lensed_files,
):
    """
    Given an abs dataset of the source star with a mulensevent present,
    When the system is built with the default (microlensing on),
    Then the dataset is lensed (blend_dE/blend_dN exist, pinned at 0), the
      astrometric log-likelihood CHANGES when the lens mass -- hence
      theta_E -- changes (a consumer of theta_E now exists, which no
      likelihood term was before), and its gradient is finite.
    """
    tmp_dir, _ = lensed_files
    system, model = _build(_full_config(tmp_dir))
    inst = system.astrometryinstrument
    assert inst._lens_source == [0]
    assert inst._lens_phot == [0]
    assert "blend_dE" in inst.manifest and "blend_dN" in inst.manifest
    assert hasattr(inst, "blend_dE") and hasattr(inst, "blend_dN")

    ip = model.initial_point()
    lp0 = _astro_logp(model, ip)
    # perturb the lens mass through its raw variable
    mass_raw = [v for v in model.free_RVs if v.name.startswith("star.logmass")]
    assert mass_raw, [v.name for v in model.free_RVs]
    ip2 = dict(ip)
    raw = np.array(ip2[mass_raw[0].name], dtype=float)
    raw[0] += 0.5  # the LENS is star 0
    ip2[mass_raw[0].name] = raw
    lp1 = _astro_logp(model, ip2)
    assert lp0 != lp1, "theta_E does not reach the astrometric likelihood"

    dlogp = model.compile_dlogp()(ip)
    assert np.all(np.isfinite(dlogp))


def test_opted_out_dataset_ignores_the_lens(lensed_files):
    """
    Given the same dataset with microlensing: false,
    When the system is built,
    Then no blend offsets are declared, the dataset is unlensed, and the
      astrometric log-likelihood is INSENSITIVE to the lens mass.
    """
    tmp_dir, _ = lensed_files
    system, model = _build(_full_config(tmp_dir, {"microlensing": False}))
    inst = system.astrometryinstrument
    assert inst._lens_source == [None]
    assert "blend_dE" not in inst.manifest
    ip = model.initial_point()
    lp0 = _astro_logp(model, ip)
    mass_raw = [
        v for v in model.free_RVs if v.name.startswith("star.logmass")
    ][0]
    ip2 = dict(ip)
    raw = np.array(ip2[mass_raw.name], dtype=float)
    raw[0] += 0.5
    ip2[mass_raw.name] = raw
    assert _astro_logp(model, ip2) == lp0


def test_dilution_and_blend_drag_algebra(lensed_files):
    """
    Given the lensed dataset with its photometry (g = f_b/f_s > 0),
    When the full absolute model is evaluated on the data epochs alongside
      the pieces it is built from (the linear terms, A, the raw shift, g),
    Then full - lin == w_s * shift + w_b * (x_b - lin) with w_s = A/(A+g),
      w_b = g/(A+g) and x_b the (pinned, zero) blend offset -- _apply_lens
      does what C30 says -- and the raw shift is theta_E u/(u^2+2) on the
      SAME |u| the magnification uses.

    One system, not two: a lensed dataset without a light curve is
    refused outright (see test_lensed_dataset_without_photometry_raises),
    and the reason is visible here -- without the light curve's t0_par
    anchor the geocentric proper-motion correction (MulensEvent.
    _earth_vperp_en) falls back to heliocentric and A differs at the
    percent level.
    """
    tmp_dir, t = lensed_files
    system, model = _build(_full_config(tmp_dir))
    inst = system.astrometryinstrument
    d = inst.datasets[0]
    with model:
        beta = inst._sed_beta_node(system, 0)
        dev = system.mulensevent.skowron_deviations(t, d["xyz"])
        dE, dN = inst._absolute_model(system, d, t, beta, dev=dev)
        A = system.mulensevent.get_magnification(t, dev, system, index=0)
        sN, sE = system.mulensevent.get_centroid_shift(t, dev, system, index=0)
        mi = system.mulensinstrument
        g = mi.f_blend.value[0] / mi.f_source.value[0]
        theta_e = system.mulensevent.theta_E.value[0]
        plot_nodes = [p.value for p in system.plot_params]
    dE, dN, A, sN, sE, g, theta_e, *pvals = _compile_at_start(
        model, [dE, dN, A, sN, sE, g, theta_e] + plot_nodes
    )
    point = {p.label: np.asarray(v) for p, v in zip(system.plot_params, pvals)}
    lE, lN = inst._linear_terms(d, t, point, system)

    assert float(g) > 0.0
    w_s = A / (A + g)
    w_b = g / (A + g)
    np.testing.assert_allclose(
        dE - lE, w_s * sE + w_b * (0.0 - lE), rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        dN - lN, w_s * sN + w_b * (0.0 - lN), rtol=1e-9, atol=1e-12
    )
    # the raw shift is the closed form on the magnification's own |u|
    u = np.array([_paczynski_inverse(a) for a in A])
    np.testing.assert_allclose(
        np.hypot(sE, sN), float(theta_e) * u / (u**2 + 2.0), rtol=1e-6
    )
    # and the dilution really bites: near peak w_s > baseline w_s
    assert w_s[np.argmax(A)] > w_s[np.argmin(A)]


def _paczynski_inverse(A):
    """|u| from A for a point lens: u^2 = 2 (A/sqrt(A^2-1) - 1)."""
    return float(np.sqrt(2.0 * (A / np.sqrt(A**2 - 1.0) - 1.0)))


def test_binary_and_finite_source_events_build_through_the_op(
    lensed_files, caplog
):
    """
    Given a lensed astrometric dataset,
    When the event has a lens companion, or is finite_source,
    Then the system builds (stage 2: the centroid comes from
      VBMicrolensing's astrox accumulators through VBMDirectMagOp), the
      astrometric log-likelihood is finite, no point-source validity
      warning is issued, and the event declares the gradient-free sampler
      requirement the Op path always carried.
    """
    tmp_dir, _ = lensed_files
    caplog.set_level(logging.WARNING)
    binary = _full_config(tmp_dir)
    binary["planet"] = [{"name": "Lb"}]
    binary["lens"].append({"body": "planet.Lb", "name": "Lb"})
    params = _base_params()
    params["planet.Lb.mass"] = {"initval": 1.0}
    params["lens.Lb.s"] = {"initval": 1.2}
    params["lens.Lb.alpha"] = {"initval": 30.0}
    system, model = _build(binary, params)
    assert system.mulensevent.uses_op()
    assert np.isfinite(_astro_logp(model, model.initial_point()))
    assert "nuts" in system.mulensevent.sampler_requirements().get(
        "incompatible", set()
    )

    fs = _full_config(tmp_dir, event_extra={"finite_source": True})
    params = _base_params()
    params["source.Source.rho"] = {"initval": 1e-3}
    system, model = _build(fs, params)
    assert system.mulensevent.uses_op()
    assert np.isfinite(_astro_logp(model, model.initial_point()))
    assert not any(
        "POINT-SOURCE formula" in rec.getMessage() for rec in caplog.records
    )


def test_lensed_dataset_without_photometry_raises(lensed_files):
    """
    Given an abs dataset of the source star and a mulensevent but NO
      microlensing light curve,
    When the system is prepared,
    Then it raises naming both the missing blend fraction and the opt-out:
      astrometry of a lensed source cannot be fit without its photometry
      (the blend fraction and the geocentric frame anchor both come from
      the light curve).
    """
    tmp_dir, _ = lensed_files
    cfg = _full_config(tmp_dir, with_photometry=False)
    with pytest.raises(
        ValueError, match="cannot be fit without its photometry"
    ):
        system = System(cfg, user_params=_base_params())
        system.prepare()


def test_explicit_true_on_a_non_source_star_raises(lensed_files):
    tmp_dir, _ = lensed_files
    cfg = _full_config(tmp_dir, {"microlensing": True, "star_ndx": 0})
    with pytest.raises(ValueError, match="not a microlensing source"):
        system = System(cfg, user_params=_base_params())
        system.prepare()


def test_synthetic_recovery_chi2_and_theta_E_sensitivity(
    lensed_files, tmp_path
):
    """
    Given astrometry generated from the model's OWN full prediction at the
      truth (linear terms + centroid shift + blend drag), plus 0.1 mas noise,
    When the system is rebuilt on that file,
    Then chi2 of the two astrometric channels is ~ 2N at the truth, the
      gradient is finite, and doubling the lens mass (theta_E x sqrt 2) is
      rejected by many sigma -- the amplitude is measured, not decorative.
    """
    tmp_dir, t = lensed_files
    sys0, model0 = _build(_full_config(tmp_dir))
    inst = sys0.astrometryinstrument
    inst.compile_plotters(model0, sys0)
    vals = _compile_at_start(model0, [p.value for p in sys0.plot_params])
    vals = [np.asarray(v) if np.ndim(v) else float(v) for v in vals]
    dE_true, dN_true = inst._eval_lensed(sys0, 0, t, vals)
    # the injected shift is a real signal at 0.1 mas precision
    point = {p.label: v for p, v in zip(sys0.plot_params, vals)}
    lE, lN = inst._linear_terms(inst.datasets[0], t, point, sys0)
    assert np.max(np.hypot(dE_true - lE, dN_true - lN)) > 0.15

    rng = np.random.default_rng(11)
    err = 0.1
    new_dir = tmp_path
    import shutil

    shutil.copy(tmp_dir / "lc.dat", new_dir / "lc.dat")
    _write_abs_astrometry(
        new_dir / "astrom.abs",
        t,
        dE_true + rng.normal(0, err, len(t)),
        dN_true + rng.normal(0, err, len(t)),
        err,
    )
    system, model = _build(_full_config(new_dir))
    ip = model.initial_point()
    n = 2 * len(t)
    lp = _astro_logp(model, ip)
    chi2 = -2.0 * lp - n * np.log(2 * np.pi) - 2.0 * n * np.log(err)
    assert abs(chi2 - n) < 5.0 * np.sqrt(2.0 * n), chi2
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))

    # The WRONG theta_E: rebuild on the same data with the lens mass
    # doubled (theta_E x sqrt 2, i.e. the shift ~0.1 mas off at 0.1 mas
    # precision over 80 epochs).  A rebuild rather than a raw-variable
    # poke: the whitened raw scales are the probe's, so a fixed raw step
    # is a fraction of a percent in mass and means nothing physically.
    wrong = _base_params()
    wrong["star.Lens.mass"] = {"initval": 1.0}
    system2, model2 = _build(_full_config(new_dir), wrong)
    ip2 = model2.initial_point()
    theta_fn = pytensor.function(
        model2.free_RVs,
        system2.mulensevent.theta_E.value[0],
        on_unused_input="ignore",
    )
    theta1 = float(theta_fn(*[ip2[v.name] for v in model2.free_RVs]))
    assert theta1 == pytest.approx(np.sqrt(2.0) * _theta_E_truth(), rel=1e-6)
    lp2 = _astro_logp(model2, ip2)
    chi2_wrong = -2.0 * lp2 - n * np.log(2 * np.pi) - 2.0 * n * np.log(err)
    assert chi2_wrong - chi2 > 25.0, (chi2, chi2_wrong, theta1)

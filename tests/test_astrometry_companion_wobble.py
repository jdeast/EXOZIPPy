"""
The gaia/abs/catalog photocenter wobble of a star in an orbit's COMPANION
group (review 8.10.6).

Before, `_photocenter_terms` summed only the orbits whose PRIMARY group held
the target and WARNED for the rest, so a Gaia DR4-style fit of a binary's
companion star had no orbit at all -- and since PR #498 tied its distance
and proper motion to the orbit anchor, it was silently a system-pm fit.

Conventions (skyframe.md, "The photocenter of an unresolved pair"): with
x_rel = x_companion_group - x_primary_group, x_P = -(M_c/M) x_rel and
x_C = +(M_p/M) x_rel, and the PSF's photocenter is (1 - b) x_target +
b x_other, b the OTHER side's light fraction.  So a primary target wobbles
as x_rel (b - M_c/M) and a companion target as x_rel (M_p/M - b).

The reference here is independent of the model: explicit 3-D rotations of
the relative orbit (test_astrometry._sky_pos_reference) and the two-body
mass split written out by hand.
"""

import numpy as np
import pymc as pm
import pytensor
import pytest
from test_astrometry import _sky_pos_reference, _tp_from_tc

from exozippy.system import System

pytestmark = pytest.mark.slow

RSUN_AU = 0.004650467260962157

# A nearby G + K binary (the K star the Gaia target), and a KELT-4-like
# triple for the nesting.
_T = dict(
    ra0=150.0,
    dec0=20.0,
    plx=20.0,
    pmra=30.0,
    pmdec=-15.0,
    P=400.0,
    tc=2457100.0,
    ecc=0.3,
    w=np.radians(40.0),
    bigom=np.radians(60.0),
    inc=np.radians(60.0),
    mA=1.0,
    mB=0.5,
)
_EPOCH = 2457400.0


def _write_gaia(path, t, w=None, err=0.1, seed=3):
    rng = np.random.default_rng(seed)
    psi = np.degrees(rng.uniform(0, 2 * np.pi, len(t)))
    if w is None:
        w = rng.normal(0, err, len(t))
    np.savetxt(path, np.column_stack([t, w, np.full_like(t, err), psi]))
    return psi


def _times(n=40, seed=5):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(2456900.0, 2457900.0, n))


def _star_params(names, masses, sigma_mass=0.0):
    up = {}
    for s, m in zip(names, masses):
        # logmass is the sampled coordinate (mass is derived from it), so
        # it is the one a `sigma: 0` pins.
        up[f"star.{s}.logmass"] = {"initval": np.log10(m), "sigma": sigma_mass}
        up[f"star.{s}.radius"] = {"initval": 1.0, "sigma": 0.1}
        up[f"star.{s}.teff"] = {"initval": 5500, "sigma": 100}
        up[f"star.{s}.feh"] = {"initval": 0.0, "sigma": 0.1}
        up[f"star.{s}.ra"] = {"initval": _T["ra0"]}
        up[f"star.{s}.dec"] = {"initval": _T["dec0"]}
        up[f"star.{s}.pm_ra"] = {"initval": _T["pmra"]}
        up[f"star.{s}.pm_dec"] = {"initval": _T["pmdec"]}
        up[f"star.{s}.distance"] = {"initval": 1000.0 / _T["plx"]}
    return up


def _orbit_params(name, P, tc, ecc, w, bigom, inc):
    return {
        f"orbit.{name}.period": {"initval": P},
        f"orbit.{name}.tc": {"initval": tc},
        f"orbit.{name}.secosw": {"initval": np.sqrt(ecc) * np.cos(w)},
        f"orbit.{name}.sesinw": {"initval": np.sqrt(ecc) * np.sin(w)},
        f"orbit.{name}.bigomega": {"initval": np.degrees(bigom)},
        f"orbit.{name}.cosi": {"initval": np.cos(inc)},
    }


def _binary(tmp_dir, datasets, masses=(None, None), extra=None, sed=False):
    """A two-star A-B orbit (A primary) with one gaia dataset per entry of
    ``datasets`` = [(name, star_ndx, fluxfrac or None, extra keys)]."""
    mA = _T["mA"] if masses[0] is None else masses[0]
    mB = _T["mB"] if masses[1] is None else masses[1]
    t = _times()
    insts, up = [], _star_params(["A", "B"], [mA, mB])
    for name, star_ndx, beta, keys in datasets:
        path = tmp_dir / f"{name}.astrom"
        _write_gaia(path, t)
        insts.append(
            {
                "name": name,
                "file": str(path),
                "mode": "gaia",
                "observer_location": "earth",
                "epoch": _EPOCH,
                "star_ndx": star_ndx,
                **(keys or {}),
            }
        )
        if beta is not None:
            up[f"astrometryinstrument.{name}.fluxfrac"] = {"initval": beta}
    up.update(
        _orbit_params(
            "AB", _T["P"], _T["tc"], _T["ecc"], _T["w"], _T["bigom"], _T["inc"]
        )
    )
    up.update(extra or {})
    config = {
        "name": "bin",
        "star": [{"name": "A", "mist": False}, {"name": "B", "mist": False}],
        "orbit": [{"name": "AB", "primary": ["A"], "companion": ["B"]}],
        "astrometryinstrument": insts,
    }
    if sed:
        sed_file = tmp_dir / "empty.sed"
        sed_file.write_text("model: NextGen\nfilters: []\n")
        config["sed"] = {"file": str(sed_file)}
        config["band"] = [
            {"name": "GaiaG", "filter": "GAIA2r.G", "ld_law": "linear"}
        ]
    return config, up, t


def _start(system, model):
    """The sampler's raw start (run.py's), NOT model.initial_point()."""
    return system.get_raw_starts(model)[0][0]


def _at(model, nodes, point):
    """``nodes`` evaluated at a raw-space ``point``, with every RV replaced
    by its value variable (evaluated on the RVs, a node draws from the
    prior)."""
    nodes = model.replace_rvs_by_values(nodes)
    fn = pytensor.function(model.value_vars, nodes, on_unused_input="ignore")
    return fn(*[point[v.name] for v in model.value_vars])


def _built(config, up):
    system = System(config, user_params=up)
    system.prepare()
    model = system.build_model()
    start = _start(system, model)
    point = system.get_internal_point(model, start)
    system.compile_plotter_functions(model)
    # The masses, semimajor axes and parallax the MODEL holds at the start
    # (a derived mass need not keep a params-file initval).
    m, a, plx = _at(
        model,
        [
            system.star.mass.value,
            system.orbit.a.value,
            system.star.parallax.value,
        ],
        start,
    )
    system._test_start = (
        np.atleast_1d(m).astype(float),
        np.atleast_1d(a).astype(float),
        np.atleast_1d(plx).astype(float),
    )
    system._test_model = model
    return system, model, point


def _wobbles(system, point, t):
    """Each dataset's modeled photocenter wobble (dE, dN) in mas: the
    LIKELIHOOD graph's own term (``_photocenter_terms``), evaluated at the
    model's start.  ``point`` is unused; the plotted track reads the plot
    point, which holds a derived mass at its params-file initval."""
    import pytensor.tensor as pt

    model = system._test_model
    inst = system.astrometryinstrument
    tt = pt.as_tensor_variable(np.asarray(t, dtype=float))
    nodes = []
    for i in range(inst.n_elements):
        beta = inst._sed_beta_node(system, i)
        a_phot, omap = inst._photocenter_terms(system, i, beta)
        dE, dN = system.orbit.get_sky_position(tt, a_phot, omap)
        nodes += [pt.sum(dE, axis=1), pt.sum(dN, axis=1)]
    out = [np.asarray(x) for x in _at(model, nodes, _start(system, model))]
    return [(out[2 * k], out[2 * k + 1]) for k in range(inst.n_elements)]


def _resolved_masses(system):
    return system._test_start[0]


def _resolved_plx(system):
    """The (shared, orbit-linked) parallax at the start, mas."""
    return float(system._test_start[2][0])


def _resolved_a(system):
    """Each orbit's a_rel in solar radii, at the start."""
    return system._test_start[1]


def _x_rel(t, a_rel_mas, P, tc, ecc, w, bigom, inc):
    """x_rel = companion group minus primary group, (dE, dN) in mas.

    _sky_pos_reference traces the PRIMARY's own orbit (argument of
    periastron omega_*), i.e. the direction of -x_rel."""
    tp = _tp_from_tc(tc, P, ecc, w)
    E, N = _sky_pos_reference(t, P, tp, ecc, w, bigom, inc, a_rel_mas)
    return -E, -N


def _a_rel_mas(m_tot, P):
    return (m_tot * (P / 365.25) ** 2) ** (1.0 / 3.0) * _T["plx"]


@pytest.fixture(scope="module")
def mirror_case(tmp_path_factory):
    """A luminous companion B (b = 0.2 of the light is A's) as the target,
    and A as a target with b = 0.1 of the light B's, in one fit."""
    tmp = tmp_path_factory.mktemp("mirror")
    config, up, t = _binary(tmp, [("GB", 1, 0.2, None), ("GA", 0, 0.1, None)])
    system, model, point = _built(config, up)
    return system, model, point, t


def test_companion_target_wobble_matches_the_two_body_photocenter(
    mirror_case,
):
    """
    Given a binary's COMPANION star as the gaia target, with 20% of the
      PSF light from the primary (fluxfrac = 0.2)
    When the modeled photocenter wobble is evaluated at the start
    Then it is x_rel (M_p/M - b) to 1e-9 relative, x_rel from an
      independent rotation-matrix Kepler orbit -- the term that used to be
      dropped with a warning.
    """
    # Arrange
    system, _model, point, t = mirror_case
    mA, mB = _resolved_masses(system)
    m_tot = mA + mB
    a = float(_resolved_a(system)[0]) * RSUN_AU * _resolved_plx(system)
    ex, nx = _x_rel(
        t, a, _T["P"], _T["tc"], _T["ecc"], _T["w"], _T["bigom"], _T["inc"]
    )

    # Act
    (dE, dN), _ = _wobbles(system, point, t)

    # Assert: a real signal, of the predicted size and sign
    k = mA / m_tot - 0.2
    assert np.ptp(dE) > 5.0
    scale = np.max(np.abs(np.r_[ex, nx])) * k
    np.testing.assert_allclose(dE, k * ex, rtol=0, atol=1e-9 * scale)
    np.testing.assert_allclose(dN, k * nx, rtol=0, atol=1e-9 * scale)


def test_primary_and_companion_targets_wobble_in_opposite_directions(
    mirror_case,
):
    """
    Given the same orbit seen from the primary's data (b_c = 0.1, the
      companion's light) and from the companion's (b_p = 0.2, the
      primary's)
    When the two wobbles are compared
    Then they point in OPPOSITE directions with amplitude ratio
      (M_p/M - b_p) / (b_c - M_c/M), exactly.
    """
    # Arrange
    system, _model, point, t = mirror_case
    mA, mB = _resolved_masses(system)
    m_tot = mA + mB

    # Act
    (dE_B, dN_B), (dE_A, dN_A) = _wobbles(system, point, t)

    # Assert
    ratio = (mA / m_tot - 0.2) / (0.1 - mB / m_tot)
    assert ratio < 0.0
    np.testing.assert_allclose(dE_B, ratio * dE_A, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(dN_B, ratio * dN_A, rtol=1e-9, atol=1e-11)


def test_one_unresolved_pair_has_one_photocenter_whichever_star_is_named(
    tmp_path,
):
    """
    Given one unresolved pair named once from each star, with CONSISTENT
      fractions (b_p = 1 - b_c: the same PSF, the same light)
    When the two wobbles are compared
    Then they are the same track -- the photocenter of a PSF does not depend
      on which of its stars the catalog calls the source.
    """
    # Arrange
    config, up, t = _binary(
        tmp_path, [("GA", 0, 0.3, None), ("GB", 1, 0.7, None)]
    )
    system, _model, point = _built(config, up)

    # Act
    (dE_A, dN_A), (dE_B, dN_B) = _wobbles(system, point, t)

    # Assert
    assert np.ptp(dE_A) > 1.0
    np.testing.assert_allclose(dE_B, dE_A, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(dN_B, dN_A, rtol=1e-9, atol=1e-11)


def test_equal_mass_equal_light_pair_has_no_wobble(tmp_path):
    """
    Given an equal-mass pair with half the light from each star
    When the wobble is evaluated from either star's data
    Then it vanishes on both sides: the photocenter IS the barycenter.
    """
    # Arrange
    config, up, t = _binary(
        tmp_path,
        [("GA", 0, 0.5, None), ("GB", 1, 0.5, None)],
        masses=(0.8, 0.8),
    )
    system, _model, point = _built(config, up)

    # Act
    tracks = _wobbles(system, point, t)

    # Assert: zero against the scale of a dark companion's wobble
    scale = _a_rel_mas(1.6, _T["P"]) * 0.5
    assert scale > 10.0
    for dE, dN in tracks:
        assert np.max(np.abs(dE)) < 1e-12 * scale
        assert np.max(np.abs(dN)) < 1e-12 * scale


def test_companion_target_logp_gradient_agrees_between_c_and_jax(
    mirror_case,
):
    """
    Given the companion-target model
    When dlogp is evaluated through the C backend and through JAX
    Then both are finite and agree (the JAX where-trap check, CLAUDE.md).
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from pymc.sampling.jax import get_jaxified_logp

    # Arrange
    system, model, _point, _t = mirror_case
    ip = _start(system, model)
    vals = [ip[v.name] for v in model.value_vars]

    # Act
    c_grad = model.compile_dlogp()(ip)
    logp_fn = get_jaxified_logp(model)
    j_grads = jax.grad(lambda vs: logp_fn(vs))(vals)
    j_grad = np.concatenate([np.ravel(np.asarray(g)) for g in j_grads])

    # Assert
    assert np.all(np.isfinite(c_grad))
    assert np.all(np.isfinite(j_grad))
    np.testing.assert_allclose(j_grad, c_grad, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# SED path: a luminous PRIMARY dragging a companion target
# ---------------------------------------------------------------------------


def test_sed_fraction_of_a_luminous_primary_drags_the_companion_target(
    tmp_path,
):
    """
    Given the companion B as the gaia target, with `band:` +
      `companion_star_ndx: A` and a sed: block
    When the model is built
    Then fluxfrac_sed is the PRIMARY's light F_A/(F_A + F_B) -- "the other
      star in the PSF" -- and the wobble is x_rel (M_p/M - b) with that b.
    """
    # Arrange
    config, up, t = _binary(
        tmp_path,
        [("GB", 1, None, {"band": "GaiaG", "companion_star_ndx": 0})],
        extra={
            "star.A.teff": {"initval": 5800, "sigma": 100},
            "star.A.radius": {"initval": 1.0, "sigma": 0.1},
            "star.B.teff": {"initval": 4500, "sigma": 100},
            "star.B.radius": {"initval": 0.7, "sigma": 0.1},
        },
        sed=True,
    )
    system, model, point = _built(config, up)

    # Act
    beta, m_a, m_b = (
        float(x)
        for x in _at(
            model,
            [
                model["astrometryinstrument.GB.fluxfrac_sed"],
                system.sed.predict_star_appmag(0, "GAIA2r.G", system),
                system.sed.predict_star_appmag(1, "GAIA2r.G", system),
            ],
            _start(system, model),
        )
    )
    ((dE, dN),) = _wobbles(system, point, t)

    # Assert: the fraction is the primary's
    F_a, F_b = 10 ** (-0.4 * m_a), 10 ** (-0.4 * m_b)
    assert beta == pytest.approx(F_a / (F_a + F_b), rel=1e-10)
    assert 0.5 < beta < 1.0  # the hotter, larger primary dominates
    # ... and it drags the companion's photocenter by M_p/M - b
    mA, mB = _resolved_masses(system)
    m_tot = mA + mB
    a = float(_resolved_a(system)[0]) * RSUN_AU * _resolved_plx(system)
    ex, nx = _x_rel(
        t, a, _T["P"], _T["tc"], _T["ecc"], _T["w"], _T["bigom"], _T["inc"]
    )
    k = mA / m_tot - beta
    scale = np.max(np.abs(np.r_[ex, nx])) * abs(k)
    np.testing.assert_allclose(dE, k * ex, rtol=0, atol=1e-9 * scale)
    np.testing.assert_allclose(dN, k * nx, rtol=0, atol=1e-9 * scale)


# ---------------------------------------------------------------------------
# Hierarchy: KELT-4 C, in the BC orbit and in the companion group of A-BC
# ---------------------------------------------------------------------------

_INNER = dict(
    P=300.0,
    tc=2457050.0,
    ecc=0.2,
    w=np.radians(100.0),
    bigom=np.radians(30.0),
    inc=np.radians(50.0),
)
_OUTER = dict(
    P=4000.0,
    tc=2457300.0,
    ecc=0.4,
    w=np.radians(-70.0),
    bigom=np.radians(140.0),
    inc=np.radians(110.0),
)
_M3 = dict(A=1.2, B=0.6, C=0.7)


def _triple(tmp_dir, keys=None, beta=None, sed=False):
    """A + (C, B): orbit BC (primary C, companion B) nested in the
    companion group of A-BC; one gaia dataset on C."""
    t = _times(60, seed=9)
    path = tmp_dir / "GC.astrom"
    _write_gaia(path, t)
    up = _star_params(["A", "B", "C"], [_M3["A"], _M3["B"], _M3["C"]])
    up.update(_orbit_params("BC", **_INNER))
    up.update(_orbit_params("ABC", **_OUTER))
    if beta is not None:
        up["astrometryinstrument.GC.fluxfrac"] = {"initval": beta}
    config = {
        "name": "triple",
        "star": [{"name": s, "mist": False} for s in ("A", "B", "C")],
        "orbit": [
            {"name": "BC", "primary": ["C"], "companion": ["B"]},
            {"name": "ABC", "primary": ["A"], "companion": ["B", "C"]},
        ],
        "astrometryinstrument": [
            {
                "name": "GC",
                "file": str(path),
                "mode": "gaia",
                "observer_location": "earth",
                "epoch": _EPOCH,
                "star_ndx": 2,
                **(keys or {}),
            }
        ],
    }
    if sed:
        sed_file = tmp_dir / "empty.sed"
        sed_file.write_text("model: NextGen\nfilters: []\n")
        config["sed"] = {"file": str(sed_file)}
        config["band"] = [
            {"name": "GaiaG", "filter": "GAIA2r.G", "ld_law": "linear"}
        ]
    return config, up, t


def _triple_reference(system, t, b_B):
    """Photocenter of a PSF holding C (weight 1 - b_B) and B (b_B), about
    the system barycenter, from each star's own two-body positions."""
    m = dict(zip(("A", "B", "C"), _resolved_masses(system)))
    a = _resolved_a(system) * RSUN_AU
    plx = _resolved_plx(system)
    m_bc = m["B"] + m["C"]
    m_tot = m_bc + m["A"]
    xi = np.array(_x_rel(t, a[0] * plx, **_INNER))  # B - C
    xo = np.array(_x_rel(t, a[1] * plx, **_OUTER))  # (B,C) - A
    x_bc = (m["A"] / m_tot) * xo  # the BC barycenter
    x_c = x_bc - (m["B"] / m_bc) * xi
    x_b = x_bc + (m["C"] / m_bc) * xi
    return (1.0 - b_B) * x_c + b_B * x_b


def test_nested_companion_target_gets_the_wobble_of_every_orbit(tmp_path):
    """
    Given KELT-4 C as the gaia target -- primary of BC, inside the
      companion group of A-BC -- with B's light in the PSF (SED path,
      companion_star_ndx: B) and A resolved out
    When the wobble is evaluated
    Then it is the flux-weighted photocenter of C and B built from each
      star's own two-body position in BOTH orbits, to 1e-9.
    """
    # Arrange
    config, up, t = _triple(
        tmp_path, keys={"band": "GaiaG", "companion_star_ndx": 1}, sed=True
    )
    system, model, point = _built(config, up)
    (b_B,) = _at(
        model,
        [model["astrometryinstrument.GC.fluxfrac_sed"]],
        _start(system, model),
    )
    b_B = float(b_B)

    # Act
    ((dE, dN),) = _wobbles(system, point, t)

    # Assert
    eE, eN = _triple_reference(system, t, b_B)
    assert 0.0 < b_B < 1.0
    assert sorted(
        system.astrometryinstrument._photocenter_orbits(system, 0)
    ) == [
        (0, "primary", "companion"),
        (1, "companion", "companion"),
    ]
    scale = np.max(np.abs(np.r_[eE, eN]))
    assert scale > 10.0
    np.testing.assert_allclose(dE, eE, rtol=0, atol=1e-9 * scale)
    np.testing.assert_allclose(dN, eN, rtol=0, atol=1e-9 * scale)


def test_nested_dark_target_moves_with_both_orbits(tmp_path):
    """
    Given the same C target with the dark-companion default (fluxfrac 0)
    When the wobble is evaluated
    Then it is C's own barycentric motion through both orbits -- the
      sampled path is unambiguous when there is no other light to place.
    """
    # Arrange
    config, up, t = _triple(tmp_path)
    system, _model, point = _built(config, up)

    # Act
    ((dE, dN),) = _wobbles(system, point, t)

    # Assert
    eE, eN = _triple_reference(system, t, 0.0)
    scale = np.max(np.abs(np.r_[eE, eN]))
    np.testing.assert_allclose(dE, eE, rtol=0, atol=1e-9 * scale)
    np.testing.assert_allclose(dN, eN, rtol=0, atol=1e-9 * scale)


# ---------------------------------------------------------------------------
# What a sampled fluxfrac cannot describe RAISES
# ---------------------------------------------------------------------------


def test_sampled_fluxfrac_on_a_target_in_two_orbits_raises(tmp_path):
    """
    Given the nested C target with a NONZERO sampled fluxfrac
    When the model is built
    Then it raises, naming both orbits: one number cannot say whether the
      light is B's (the BC orbit's other side) or A's (A-BC's).
    """
    config, up, _t = _triple(tmp_path, beta=0.2)
    system = System(config, user_params=up)
    system.prepare()
    with pytest.raises(ValueError, match=r"GC.*\['BC', 'ABC'\].*which"):
        system.build_model()


def test_sampled_fluxfrac_with_a_nested_other_side_raises(tmp_path):
    """
    Given A as the target, its one orbit's other group (B, C) holding the BC
      orbit, and a nonzero sampled fluxfrac
    When the model is built
    Then it raises: the B+C light wobbles inside its group by an amount a
      single fraction cannot split.
    """
    config, up, _t = _triple(tmp_path, beta=0.2)
    config["astrometryinstrument"][0]["star_ndx"] = 0
    system = System(config, user_params=up)
    system.prepare()
    with pytest.raises(ValueError, match=r"ABC.*holds.*\['BC'\]"):
        system.build_model()


def test_sed_companion_sharing_no_orbit_with_the_target_raises(tmp_path):
    """
    Given companion_star_ndx naming a star in NO orbit with the target
    When parameters are registered
    Then it raises: that light is a blend on its own track, not a companion
      riding the target's orbit (which is how it used to be read).
    """
    config, up, _t = _binary(
        tmp_path,
        [("GA", 0, None, {"band": "GaiaG", "companion_star_ndx": 2})],
        sed=True,
    )
    config["star"].append({"name": "N", "mist": False})
    up.update(_star_params(["N"], [0.5]))
    with pytest.raises(ValueError, match="shares no Keplerian orbit"):
        System(config, user_params=up).prepare()


def test_sed_companion_equal_to_the_target_raises(tmp_path):
    """companion_star_ndx naming the target itself is refused."""
    config, up, _t = _binary(
        tmp_path,
        [("GA", 0, None, {"band": "GaiaG", "companion_star_ndx": 0})],
        sed=True,
    )
    with pytest.raises(ValueError, match="is the target star itself"):
        System(config, user_params=up).prepare()


# ---------------------------------------------------------------------------
# Maximum-likelihood recovery of a companion-target orbit
# ---------------------------------------------------------------------------


def test_companion_target_orbit_is_recovered_from_a_displaced_start(
    tmp_path,
):
    """
    Given gaia epochs of the COMPANION B generated from the model at the
      truth (0.05 mas noise, 60 epochs), b = 0.2
    When the fit starts with Omega, cos i, omega and tc displaced and is
      maximized
    Then Omega, cos i, e and P come back to the truth -- the orbit is
      measured from the companion's data alone, which before review 8.10.6
      carried no orbital term at all.
    """
    # Arrange: the truth model's own prediction, as data
    t = _times(60, seed=21)
    err = 0.05
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    config, up, _ = _binary(truth_dir, [("GB", 1, 0.2, None)])
    config["astrometryinstrument"][0]["file"] = str(truth_dir / "GB.astrom")
    psi = _write_gaia(truth_dir / "GB.astrom", t, err=err, seed=4)
    system, model, _point = _built(config, up)
    inst = system.astrometryinstrument
    d = inst.datasets[0]
    dE, dN = inst._absolute_model(system, d, d["time"], inst.fluxfrac.value[0])
    (w_true,) = _at(
        model, [dE * d["sin_psi"] + dN * d["cos_psi"]], _start(system, model)
    )
    rng = np.random.default_rng(8)
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    data = fit_dir / "GB.astrom"
    np.savetxt(
        data,
        np.column_stack(
            [t, w_true + rng.normal(0, err, len(t)), np.full_like(t, err), psi]
        ),
    )
    config["astrometryinstrument"][0]["file"] = str(data)
    start = dict(up)
    # No jitter: a free one absorbs a half-converged orbit's misfit.
    start["astrometryinstrument.GB.jitter_variance"] = {
        "initval": 0.0,
        "sigma": 0.0,
    }
    start.update(
        _orbit_params(
            "AB",
            _T["P"] * 1.003,
            _T["tc"] + 5.0,
            0.25,
            _T["w"] + 0.15,
            _T["bigom"] + np.radians(8.0),
            _T["inc"] + np.radians(5.0),
        )
    )
    fit_sys = System(config, user_params=start)
    fit_sys.prepare()
    fit_model = fit_sys.build_model()

    # Act
    orb = fit_sys.orbit
    nodes = [
        orb.bigomega.value[0],
        orb.cosi.value[0],
        orb.ecc.value[0],
        orb.period.value[0],
    ]
    start = _start(fit_sys, fit_model)
    bigom0 = float(_at(fit_model, nodes, start)[0])
    with fit_model:
        best = pm.find_MAP(
            start=start,
            method="BFGS",  # L-BFGS-B stalls on this ridge (Omega, omega, tc)
            progressbar=False,
            maxeval=20000,
            tol=1e-12,
        )
    bigom, cosi, ecc, period = (float(x) for x in _at(fit_model, nodes, best))

    # Assert: the start really was displaced, and the fit came back.
    # Omega is compared modulo 180 deg -- without RVs (Omega, omega) and
    # (Omega + 180, omega + 180) are the same sky orbit (orbit.md).
    def dnode(x):
        return (np.degrees(x - _T["bigom"]) + 90.0) % 180.0 - 90.0

    got = dict(bigom=np.degrees(bigom), cosi=cosi, ecc=ecc, period=period)
    assert abs(dnode(bigom0)) > 5.0
    assert abs(dnode(bigom)) < 1.0, got
    assert abs(cosi) == pytest.approx(abs(np.cos(_T["inc"])), abs=0.02)
    assert ecc == pytest.approx(_T["ecc"], abs=0.02)
    assert period == pytest.approx(_T["P"], rel=2e-3)

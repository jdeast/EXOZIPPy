"""The SED's sight-line dust profile (review 8.9.8; components/sed/sed.md,
"The sight-line dust profile"): the profile arithmetic, the bin-count rule,
the near/far R_V mixing, the gradient safety of every piece, and the roles,
raises and seeding a System gives it.

Real NextGen tables for the table-path systems (the conftest prewarm fetches
them); the split cases use test_sed_fit_rv_dust's blackbody runtime grid.
"""

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from test_redclump import DEC, RA, I, V, _config, _rows
from test_sed_fit_rv_dust import _bb_grid

from exozippy.components.sed import dust_profile as dp
from exozippy.components.sed.extinction import ccm89_ab
from exozippy.models.NextGen import runtime_spectra as rs
from exozippy.system import System

SINB = dp.sinb_of(-3.9)
H = dp.DEFAULT_SCALE_HEIGHT_PC


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_groups, k", [(1, 3), (2, 3), (3, 3), (4, 4), (6, 6), (9, 6)]
)
def test_one_bin_per_object_between_three_and_six(n_groups, k):
    """
    Given n SED objects at distinct distances,
    When the default bin count is chosen,
    Then it is n, clipped to [3, 6] (JDE 2026-11-09, ruling 14).
    """
    assert dp.default_bins(n_groups) == k


def test_the_prior_center_is_the_smooth_layer_exactly():
    """
    Given every z_k at its prior center 0,
    When C(d) is evaluated anywhere,
    Then it is K times the exponential layer's fraction s(d) -- exactly the
      layer, not a piecewise-linear approximation (the bins are equal shares
      of its dust).
    """
    d = np.geomspace(1.0, 3e4, 500)
    for k in (1, 3, 6):
        c = dp.cumulative(d, np.zeros(k), SINB, H, k, xp=np)
        np.testing.assert_allclose(
            c, k * dp.layer_fraction(d, SINB, H, xp=np), rtol=1e-12
        )


def test_the_profile_is_monotone_continuous_and_a_normalized_cumulative_sum():
    """
    Given 1000 random shapes z,
    When A(d) / A_anchor is evaluated on a dense grid and at every edge,
    Then it never decreases, it is continuous across each edge, and it is
      the cumulative sum of the bins' dust normalized at the anchor.
    """
    rng = np.random.default_rng(1)
    k = 5
    d = np.geomspace(1.0, 3e4, 2000)
    edges = dp.bin_edges_pc(k, SINB, H)
    for _ in range(1000):
        z = rng.normal(0.0, 1.5, k)
        c = dp.cumulative(d, z, SINB, H, k, xp=np)
        assert np.all(np.diff(c) >= 0.0)
        lo = dp.cumulative(edges * (1 - 1e-12), z, SINB, H, k, xp=np)
        hi = dp.cumulative(edges * (1 + 1e-12), z, SINB, H, k, xp=np)
        # a 2e-12 relative step in d moves C by its slope there; measured
        # against the whole column, a jump would be O(1)
        np.testing.assert_allclose(
            lo, hi, rtol=0, atol=1e-8 * dp.total(z, xp=np)
        )
        # bin j's dust is 10**z_j, all of it in front of edge j
        np.testing.assert_allclose(
            dp.cumulative(edges, z, SINB, H, k, xp=np),
            np.cumsum(10.0**z)[:-1],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            dp.cumulative(np.array([1e9]), z, SINB, H, k, xp=np),
            dp.total(z, xp=np),
            rtol=1e-12,
        )


def test_one_bin_is_the_bennett_smooth_layer_normalized_at_the_anchor():
    """
    Given dust_bins: 1 (the smooth-layer mode) and any z,
    When the profile is normalized at a source distance,
    Then A_L / A_S is Bennett et al. (2015)'s [1 - exp(-D_L sin|b|/h)] /
      [1 - exp(-D_S sin|b|/h)] to 1e-12 (examples/MOA-2008-BLG-379's former
      hand link, h = 100 pc).
    """
    rng = np.random.default_rng(2)
    sinb = dp.sinb_of(-3.1129)
    for _ in range(50):
        dl, ds = rng.uniform(500, 8000), rng.uniform(6000, 11000)
        z = rng.normal(size=1)
        c = dp.cumulative(np.array([dl, ds]), z, sinb, 100.0, 1, xp=np)
        bennett = (1 - np.exp(-dl * sinb / 100.0)) / (
            1 - np.exp(-ds * sinb / 100.0)
        )
        np.testing.assert_allclose(c[0] / c[1], bennett, rtol=1e-12)


def test_ccm_mixing_is_exact_in_inverse_rv():
    """
    Given dust in front of d from two populations with R_near and R_far,
    When the per-wavelength extinction sum_k dA_k (a + b/R_k) is compared
      with the mixed curve A (a + b/R_eff),
    Then they agree to 1e-12 at every wavelength: CCM is linear in 1/R_V.
    """
    rng = np.random.default_rng(3)
    lam = rng.uniform(0.35, 3.0, 50)
    a, b = ccm89_ab(lam)
    for _ in range(100):
        a_near, a_far = rng.uniform(0.01, 3.0, 2)
        r_near, r_far = rng.uniform(1.6, 5.5, 2)
        direct = a_near * (a + b / r_near) + a_far * (a + b / r_far)
        f = a_near / (a_near + a_far)
        r_eff = dp.mixed_rv(f, r_near, r_far)
        np.testing.assert_allclose(
            direct, (a_near + a_far) * (a + b / r_eff), rtol=1e-12
        )


def _graph():
    d = pt.dvector("d")
    z = pt.dvector("z")
    db = pt.dscalar("db")
    a = 2.0 * dp.cumulative(d, z, SINB, H, 4) / dp.total(z)
    f = dp.near_fraction(d, db, z, SINB, H, 4)
    r = dp.mixed_rv(f, 2.9, 2.0)
    out = pt.sum(pt.log(a)) + pt.sum(r)
    return (d, z, db), out


@pytest.mark.parametrize("mode", ["C", "JAX"])
def test_value_and_gradient_are_finite_at_edges_floors_and_extremes(mode):
    """
    Given distances exactly on every bin edge, at the 0.001 pc floor of a
      star's distance, far beyond the layer, and the boundary on an edge,
    When log A(d) + R_eff(d) and its gradient are compiled in the C and the
      JAX backends, with z at +/-6 dex (its bounds),
    Then every value and every gradient element is finite, and the two
      backends agree just off the edges (no pt.where anywhere; the
      where-trap, CLAUDE.md).  ON an edge two ramps meet at a kink, where
      the backends pick different (finite) subgradients.
    """
    if mode == "JAX":
        pytest.importorskip("jax")
    (d, z, db), out = _graph()
    grads = pytensor.grad(out, [d, z, db])
    fn = pytensor.function([d, z, db], [out] + grads, mode=mode)
    edges = dp.bin_edges_pc(4, SINB, H)
    dd = np.concatenate([edges, [1e-3, 1.0, 5e4, 1e9]])
    ref_fn = pytensor.function([d, z, db], [out] + grads)
    off = np.concatenate([edges * 1.001, [1e-3, 1.0, 5e4, 1e9]])
    for zz in (
        np.zeros(4),
        np.full(4, 6.0),
        np.full(4, -6.0),
        np.array([6.0, -6.0, 6.0, -6.0]),
    ):
        for b in (float(edges[1]), 4500.0):
            vals = fn(dd, zz, b)
            for v in vals:
                assert np.all(np.isfinite(v)), (zz, b, v)
            if mode == "JAX":
                for v, r in zip(fn(off, zz, 4500.0), ref_fn(off, zz, 4500.0)):
                    np.testing.assert_allclose(v, r, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# Systems: a lens, a source and the red clump on Baade's window
# ---------------------------------------------------------------------------
def _lens_rows():
    return _rows() + [
        {"name": I, "mag": 19.5, "err": 0.05, "photType": {"pos": ["Lens"]}},
    ]


def _lens_params(**extra):
    p = {
        f"star.{s}.{c}": {"initval": v, "sigma": 0}
        for s in ("Source", "Lens")
        for c, v in (("ra", RA), ("dec", DEC))
    }
    p["star.Lens.distance"] = {"initval": 3000.0}
    p["star.Source.distance"] = {"initval": 8000.0}
    p.update(extra)
    return p


def _system(tmp_path, sed=None, params=None, rows=None, clump=None):
    cfg = _config(
        tmp_path,
        rows=_lens_rows() if rows is None else rows,
        stars=("Source", "Lens"),
        sed=sed,
        clump=clump,
    )
    system = System(cfg, _lens_params() if params is None else params)
    system.prepare()
    return system


@pytest.fixture(scope="module")
def profile_system(tmp_path_factory):
    system = _system(tmp_path_factory.mktemp("dust"))
    return system, system.build_model()


def _at(model, outs, point):
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values(outs),
        on_unused_input="ignore",
    )
    return fn(*[point[v.name] for v in model.value_vars])


def test_three_objects_turn_the_profile_on_and_derive_every_av(
    profile_system,
):
    """
    Given a lens, a source and the clump the SED reads, at three distances,
    When the system is built,
    Then the profile is on with K = 3 bins anchored at the clump; every
      star's av and the clump's are DERIVED (A(d) (1 + av_frac)), each
      object samples its own av_frac, the SED samples the anchor and the
      shape, and the start's logp and gradient are finite.
    """
    system, model = profile_system
    plan = system.sed._dust_plan
    assert plan.on and plan.n_bins == 3 and plan.clump
    assert not plan.split  # the table path has no R_V
    av = system.star.av
    assert av.element_is_derived(0) and av.element_is_derived(1)
    free = {v.name for v in model.free_RVs}
    assert {
        "star.av_frac_raw",
        "redclump.av_frac_raw",
        "sed.dust_anchor_raw",
        "sed.dust_z_raw",
    } <= free
    assert "star.av_raw" not in free and "redclump.av_raw" not in free
    assert "redclump.av_ratio" not in system.redclump.manifest
    ip = model.initial_point()
    assert np.isfinite(model.compile_logp()(ip))
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))


def test_the_clump_is_the_anchor_and_the_lens_sees_less_dust(
    profile_system,
):
    """
    Given the same system at its start (z = 0, av_frac = 0),
    When the stars' and the clump's A_V are evaluated,
    Then the clump's IS the anchor, and each star's is the anchor times the
      smooth layer's s(d)/s(d_clump): a 3 kpc lens below an 8 kpc source.
    """
    system, model = profile_system
    ip = model.initial_point()
    av, d, rc_av, rc_d, anchor = _at(
        model,
        [
            system.star.av.value,
            system.star.distance.value,
            system.redclump.av.value,
            system.redclump.distance.value,
            system.sed.dust_anchor.value,
        ],
        ip,
    )
    np.testing.assert_allclose(rc_av, anchor, rtol=1e-12)
    # ...seeded from the clump's calibrated (V-I)_RC row (2.60)
    sed, rc = system.sed, system.redclump
    t, g, z = (
        float(np.atleast_1d(getattr(rc, p).initval)[0])
        for p in ("teff", "logg", "feh")
    )
    a0 = float(np.atleast_1d(rc_av)[0])
    assert sed.bc_np(I, t, g, z, a0) - sed.bc_np(V, t, g, z, a0) == (
        pytest.approx(2.60, abs=1e-4)
    )
    s = dp.layer_fraction(np.asarray(d), SINB_RC(system), H, xp=np)
    s_rc = dp.layer_fraction(np.asarray(rc_d), SINB_RC(system), H, xp=np)
    np.testing.assert_allclose(av, anchor * s / s_rc, rtol=1e-10)
    assert av[1] < av[0]


def SINB_RC(system):
    return system.sed._dust_plan.sinb


def test_distance_constrains_av_false_is_the_free_av_model(tmp_path):
    """
    Given the same system with sed.distance_constrains_av: false,
    When it is built,
    Then every star samples its own av, the clump keeps the interim av_ratio
      tie, and no profile parameter exists (the flag off is PR 2's model).
    """
    system = _system(tmp_path, sed={"distance_constrains_av": False})
    model = system.build_model()
    free = {v.name for v in model.free_RVs}
    assert "star.av_raw" in free and "redclump.av_raw" in free
    assert not any("dust" in n or "av_frac" in n for n in free)
    assert not system.sed._dust_plan.on
    assert "av_frac" not in system.star.manifest


def test_one_distance_leaves_the_profile_off(tmp_path):
    """
    Given one star the SED reads and no clump (one distance),
    When the plan is asked,
    Then it is off and the star's A_V stays free.
    """
    import yaml

    rows = [
        {"name": I, "mag": 18.0, "err": 0.05, "photType": {"pos": ["Source"]}}
    ]
    path = tmp_path / "s.sed"
    path.write_text(yaml.safe_dump({"model": "NextGen", "filters": rows}))
    cfg = {
        "star": [{"name": "Source", "mist": False}],
        "sed": {"file": str(path)},
        "galacticmodel": [{"anchor_idx": 0}],
    }
    system = System(cfg, _lens_params_one())
    system.prepare()
    assert not system.sed._dust_plan.on
    assert "av_frac" not in system.star.manifest


def _lens_params_one():
    return {
        "star.Source.ra": {"initval": RA, "sigma": 0},
        "star.Source.dec": {"initval": DEC, "sigma": 0},
    }


def test_explicit_bins_and_scale_height(tmp_path):
    """dust_bins and dust_scale_height are honored, and K = 1 is legal."""
    system = _system(
        tmp_path, sed={"dust_bins": 1, "dust_scale_height": 100.0}
    )
    plan = system.sed._dust_plan
    assert plan.n_bins == 1 and plan.h_pc == 100.0


@pytest.mark.parametrize(
    "sed, params, match",
    [
        ({"distance_constrains_av": "yes"}, None, "YAML boolean"),
        ({"dust_bins": 0}, None, "positive integer"),
        ({"dust_scale_height": -1}, None, "positive number"),
        (
            {"distance_constrains_av": False, "dust_bins": 4},
            None,
            "describe nothing",
        ),
        ({"fit_rv_dust_near": True}, None, "needs sed.fit_rv_dust"),
        (
            None,
            _lens_params(**{"star.Source.av": {"initval": 2.0, "sigma": 0}}),
            "pins a DERIVED element",
        ),
        (
            None,
            _lens_params(
                **{
                    "star.Lens.av": {
                        "initval": "star.Source.av * 0.5",
                        "sigma": 0,
                    }
                }
            ),
            "is a link",
        ),
    ],
)
def test_the_profile_raises(tmp_path, sed, params, match):
    with pytest.raises(ValueError, match=match):
        _system(tmp_path, sed=sed, params=params)


def test_an_explicit_clump_tie_under_the_profile_raises(tmp_path):
    clump = {
        "name": "RedClump",
        "bands": ["I"],
        "colors": [["V", "I"]],
        "clump_constrains_source_av": True,
    }
    with pytest.raises(ValueError, match="interim tie"):
        _system(tmp_path, clump=clump)


def test_a_users_source_av_start_is_produced_through_the_anchor(tmp_path):
    """
    Given star.Source.av: {initval: 3.0} under the profile (av derived),
    When the model is built,
    Then the anchor is seeded so the model's Source A_V starts at 3.0
      (the user-start contract, through a component hint).
    """
    system = _system(
        tmp_path, params=_lens_params(**{"star.Source.av": {"initval": 3.0}})
    )
    model = system.build_model()
    start = model.initial_point()
    start.update(system.get_raw_start(model))
    (av,) = _at(model, [system.star.av.value], start)
    np.testing.assert_allclose(av[0], 3.0, rtol=1e-6)


def test_an_anchor_band_inverts_the_band_extinction_on_the_reference(
    tmp_path,
):
    """
    Given dust_anchor_band: 2MASS Ks (a NIR map's band),
    When sed.av_anchor is evaluated at the start,
    Then the reference spectrum's A_Ks at that A_V is the sampled anchor to
      1e-5: the band-integrated A_Ks(A_V) is inverted on the SED's own grid,
      with no law conversion by hand.
    """
    system = _system(tmp_path, sed={"dust_anchor_band": "2MASS/2MASS.Ks"})
    model = system.build_model()
    ip = model.initial_point()
    a_ks, av = _at(
        model, [system.sed.dust_anchor.value, system.sed.av_anchor.value], ip
    )
    got = system.sed.reference_band_extinction_np(
        "2MASS/2MASS.Ks", float(av[0]), system
    )
    np.testing.assert_allclose(got, float(a_ks[0]), rtol=1e-5)


def _split_system(tmp_path, monkeypatch, **sed):
    monkeypatch.setattr(
        rs, "load_runtime_grid", lambda model_root=None: _bb_grid()
    )
    return _system(tmp_path, sed={"fit_rv_dust": True, **sed})


def test_the_split_gives_each_object_its_own_rv(tmp_path, monkeypatch):
    """
    Given fit_rv_dust and fit_rv_dust_near: true on the lens/source/clump
      sight line,
    When the model is built at its start,
    Then rv_dust_near and the boundary are sampled; each star's R_V is the
      mix in front of it -- between R_near and R_far, nearer R_near for the
      3 kpc lens than for the 8 kpc source -- and a table row; logp and
      gradient are finite.
    """
    system = _split_system(tmp_path, monkeypatch, fit_rv_dust_near=True)
    plan = system.sed._dust_plan
    assert plan.split
    model = system.build_model()
    free = {v.name for v in model.free_RVs}
    assert {"sed.rv_dust_near_raw", "sed.dust_boundary_raw"} <= free
    ip = model.initial_point()
    rv, rn, rf, rc = _at(
        model,
        [
            system.star.rv_dust.value,
            system.sed.rv_dust_near.value,
            system.sed.rv_dust.value,
            system.redclump.rv_dust.value,
        ],
        ip,
    )
    lo, hi = sorted((float(rn[0]), float(rf[0])))
    assert np.all((rv >= lo) & (rv <= hi)) and lo <= rc[0] <= hi
    assert abs(rv[1] - rn[0]) < abs(rv[0] - rn[0])
    assert system.star.rv_dust.print_to_table
    assert np.isfinite(model.compile_logp()(ip))
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))


def test_the_split_is_off_with_no_selected_source_and_one_rv(
    tmp_path, monkeypatch
):
    """
    Given fit_rv_dust and no microlensing event (nothing selects a source,
      so no object is known to be nearer than the far group),
    When the plan is asked with fit_rv_dust_near unset,
    Then there is no split: one sight-line R_V, star.rv_dust from_sightline.
    """
    system = _split_system(tmp_path, monkeypatch)
    assert not system.sed._dust_plan.split
    assert "rv_dust_near" not in system.sed.manifest


def test_a_cone_of_field_stars_shares_one_profile(tmp_path):
    """
    Given the "all stars in a cone" use: four field stars within 60 arcsec
      of one another, each with its own SED rows and a Gaia-like parallax
      prior, no galactic model, and the bin ceiling raised (dust_bins: 6),
    When the system is built,
    Then the profile is on with six bins, every star's A_V is derived from
      it at its own distance, and the start's logp and gradient are finite.
      (The 60 arcsec sight-line tolerance is the limit of this use: a wider
      cone raises, naming the stars.)
    """
    import yaml

    stars = {"F1": 1.0, "F2": 0.5, "F3": 0.25, "F4": 0.12}  # parallax, mas
    rows = []
    for n in stars:
        for f, m in ((V, 15.5), (I, 14.5), ("2MASS/2MASS.H", 13.5)):
            rows.append(
                {"name": f, "mag": m, "err": 0.03, "photType": {"pos": [n]}}
            )
    path = tmp_path / "cone.sed"
    path.write_text(yaml.safe_dump({"model": "NextGen", "filters": rows}))
    cfg = {
        "star": [{"name": n, "mist": False} for n in stars],
        "sed": {"file": str(path), "dust_bins": 6},
    }
    params = {}
    for k, (n, plx) in enumerate(stars.items()):
        params.update(
            {
                f"star.{n}.ra": {"initval": RA + 0.003 * k, "sigma": 0},
                f"star.{n}.dec": {"initval": DEC, "sigma": 0},
                f"star.{n}.parallax": {"mu": plx, "sigma": 0.03},
                f"star.{n}.distance": {"initval": 1000.0 / plx},
            }
        )
    system = System(cfg, params)
    system.prepare()
    plan = system.sed._dust_plan
    assert plan.on and plan.n_bins == 6 and not plan.clump
    model = system.build_model()
    assert all(system.star.av.element_is_derived(i) for i in range(4))
    ip = model.initial_point()
    assert np.isfinite(model.compile_logp()(ip))
    assert np.all(np.isfinite(model.compile_dlogp()(ip)))


def test_no_stated_position_leaves_the_profile_off_unless_asked(tmp_path):
    """
    Given two read stars with no stated sky position and no galactic model
      (a transit blend), so no Galactic latitude for the dust layer,
    When the plan is asked,
    Then the profile is off (each A_V free), and distance_constrains_av:
      true raises naming the missing position.
    """
    import yaml

    rows = [
        {"name": I, "mag": 12.0, "err": 0.05, "photType": {"pos": [n]}}
        for n in ("A", "B")
    ]
    path = tmp_path / "ab.sed"
    path.write_text(yaml.safe_dump({"model": "NextGen", "filters": rows}))

    def build(sed):
        cfg = {
            "star": [{"name": n, "mist": False} for n in ("A", "B")],
            "sed": {"file": str(path), **sed},
        }
        system = System(cfg, {})
        system.prepare()
        return system

    assert not build({}).sed._dust_plan.on
    with pytest.raises(ValueError, match="sky position"):
        build({"distance_constrains_av": True})

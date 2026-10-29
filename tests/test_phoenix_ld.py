"""The PHOENIX limb-darkening tables: the fit, the hole filling, the
loader and the band tie's second source (``ld_source: phoenix``).

Everything runs on synthetic data -- a profile that IS a quadratic law,
a profile with a cliff, a table whose coefficients are linear in every
axis (so the 4-D linear interpolation reproduces them exactly and the
asserted values need no constant of their own) -- never on the generated
tables, which are not in the tree.
"""

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest

from exozippy.components.band import ldgrid
from exozippy.components.band.band import Band
from exozippy.components.band.physics import (
    calc_u1_from_kipping,
    calc_u2_from_kipping,
)
from exozippy.components.sed.bc_grid import write_bc_table
from exozippy.models.PHOENIX import generate_PHOENIX_LD_Tables as gen
from exozippy.system import System

# --------------------------------------------------------------------------
# 1. The fit
# --------------------------------------------------------------------------


def _mu():
    return np.linspace(0.02, 1.0, 78)


def test_the_fit_recovers_a_quadratic_law_exactly():
    """
    Given a profile that IS the quadratic law 1 - u1(1-mu) - u2(1-mu)^2
      on a plane-parallel-like mu grid (no cliff),
    When it is fitted,
    Then (u1, u2) come back to a few 1e-4 (the trapezoid flux of a
      discretely sampled profile), and the linear law's u is the
      flux-conserving one, 3 - 6F.
    """
    mu = _mu()
    u1, u2 = 0.4, 0.25
    p = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2

    fits = gen.fit_profile(mu, p)

    np.testing.assert_allclose(fits["quadratic"], (u1, u2), atol=3e-3)
    F = gen._profile_flux(*gen.remap_quasi_spherical(mu, p)[:2])
    np.testing.assert_allclose(fits["linear"], (3 - 6 * F,), rtol=1e-12)
    # the law's own flux: 1/2 - u1/6 - u2/12
    np.testing.assert_allclose(F, 0.5 - u1 / 6 - u2 / 12, atol=2e-4)


def test_the_limb_moves_to_the_cliff_and_the_fit_ignores_it():
    """
    Given the same quadratic profile with a spherical-model cliff
      (intensity collapsing below mu = 0.1),
    When the quasi-spherical remap runs,
    Then the limb lands at the cliff's steepest point and the quadratic
      coefficients fitted on the remapped disk match the cliff-free law
      to a few percent, where a naive full-range fit would not.
    """
    mu = _mu()
    u1, u2 = 0.4, 0.25
    p = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
    cliff = p.copy()
    cliff[mu < 0.1] *= (mu[mu < 0.1] / 0.1) ** 3

    mu_p, p_p, limb = gen.remap_quasi_spherical(mu, cliff)

    assert 0.05 < limb < 0.11
    assert mu_p[0] == 0.0 and mu_p[-1] == 1.0
    fitted = gen.fit_quadratic(mu_p, p_p)
    np.testing.assert_allclose(fitted, (u1, u2), atol=0.03)


def test_the_remap_matches_the_analytic_limb_and_the_radius_rescaling():
    """
    Given a quadratic law times a smooth logistic cliff
      I(mu) = law(mu) / (1 + exp(-(mu - 0.08) / 0.004)), sampled on a fine
      mu grid (step h ~ 5e-4) like a spherical PHOENIX profile,
    When the quasi-spherical remap runs,
    Then (a) the limb is the mu of steepest descent of the ANALYTIC profile
      (its dI/dmu maximized by a bounded scalar solve on the closed-form
      derivative), to within one grid step -- the resolution of a sampled
      profile; (b) exactly the points at or above the limb are kept, with
      their intensities untouched; and (c) every kept point obeys the
      radius rescaling r' = r / r_limb of Claret, Hauschildt & Witte (2012),
      written here in r rather than mu: sqrt(1 - mu'^2) * r_limb ==
      sqrt(1 - mu^2), to 1e-12.

    test_the_limb_moves_to_the_cliff_and_the_fit_ignores_it accepts any
    limb in 0.05-0.11, a window ~100 grid steps wide here; a limb read one
    feature off, or mu' built from mu / mu_limb instead of the radius,
    passes it.
    """
    # ARRANGE
    from scipy.optimize import minimize_scalar

    u1, u2, mc, w = 0.4, 0.25, 0.08, 0.004
    mu = np.linspace(0.02, 1.0, 1961)
    h = mu[1] - mu[0]

    def law(m):
        return 1 - u1 * (1 - m) - u2 * (1 - m) ** 2

    def dlaw(m):
        return u1 + 2 * u2 * (1 - m)

    def sig(m):
        return 1.0 / (1.0 + np.exp(-(m - mc) / w))

    def dprofile(m):
        s = sig(m)
        return dlaw(m) * s + law(m) * s * (1 - s) / w

    profile = law(mu) * sig(mu)
    profile = profile / profile[-1]
    mu_star = minimize_scalar(
        lambda m: -dprofile(m),
        bounds=(mc - 5 * w, mc + 5 * w),
        method="bounded",
        options={"xatol": 1e-12},
    ).x

    # ACT
    mu_p, p_p, limb = gen.remap_quasi_spherical(mu, profile)

    # ASSERT -- (a) the limb
    assert mu_star < gen.MU_LIMB_SEARCH_MAX
    assert abs(limb - mu_star) <= h
    # (b) the kept points
    keep = mu >= limb
    np.testing.assert_array_equal(p_p, profile[keep])
    assert mu_p[0] == 0.0 and mu_p[-1] == 1.0
    # (c) the radius rescaling, on every kept point
    r_limb = np.sqrt(1.0 - limb**2)
    np.testing.assert_allclose(
        np.sqrt(1.0 - mu_p**2) * r_limb,
        np.sqrt(1.0 - mu[keep] ** 2),
        rtol=0,
        atol=1e-12,
    )


def test_hole_filling_interpolates_the_intensities_along_teff():
    """
    Given band intensities at 5000 K and 5200 K on DIFFERENT mu grids and
      nothing at 5100 K,
    When the holes are filled,
    Then 5100 K gets the mean of the two profiles resampled onto the lower
      neighbour's mu grid, carries that grid, and is recorded as padded
      from both; a node with only one neighbour is left alone.
    """
    mu_a = np.array([0.1, 0.5, 0.8, 1.0])
    mu_b = np.array([0.2, 0.4, 0.9, 1.0])
    a = np.broadcast_to(mu_a, (2, 1, 4)).astype(np.float32)  # p = mu
    b = np.broadcast_to(3 * mu_b, (2, 1, 4)).astype(np.float32)  # p = 3 mu
    store = {(5000.0, 4.5, "X"): a, (5200.0, 4.5, "X"): b}
    mus = {(5000.0, 4.5): mu_a, (5200.0, 4.5): mu_b}

    records = gen.fill_holes_along_teff(
        store, mus, [4900.0, 5000.0, 5100.0, 5200.0], [4.5], ["X"]
    )

    filled = store[(5100.0, 4.5, "X")][0, 0]
    # inside both grids the mean of p = mu and p = 3 mu is 2 mu exactly
    np.testing.assert_allclose(filled[1:], 2 * mu_a[1:], rtol=1e-6)
    # below the upper neighbour's first angle its profile is clamped
    # (np.interp), so the first point is the mean of 0.1 and 3 * 0.2
    np.testing.assert_allclose(filled[0], 0.5 * (0.1 + 0.6), rtol=1e-6)
    np.testing.assert_array_equal(mus[(5100.0, 4.5)], mu_a)
    assert (4900.0, 4.5, "X") not in store
    assert records == [
        {"teff": 5100.0, "logg": 4.5, "axis": "teff", "from": [5000.0, 5200.0]}
    ]


def test_the_intermediate_keeps_every_nodes_own_mu(tmp_path):
    """
    Given two nodes whose emergent angles differ (as PHOENIX's do),
    When their intensities are saved and read back,
    Then each node comes back with ITS mu array and its intensities --
      the first pass of the generator kept one array per [Fe/H] and
      fitted every node on it, which put the solar V-band limb twice too
      dark.
    """
    mu1 = np.linspace(0.02, 1, 5)
    mu2 = np.linspace(0.05, 1, 5)
    store = {
        (5800.0, 4.5, "F"): np.ones((2, 1, 5), np.float32),
        (3000.0, 5.0, "F"): np.full((2, 1, 5), 0.5, np.float32),
    }
    mus = {(5800.0, 4.5): mu1, (3000.0, 5.0): mu2}

    gen.save_intensities(tmp_path, 0.0, store, mus, [0.0, 1.0], {"F": ["X"]})
    back, mus_back, filters = gen.load_intensities(tmp_path, 0.0)

    np.testing.assert_array_equal(mus_back[(5800.0, 4.5)], mu1)
    np.testing.assert_array_equal(mus_back[(3000.0, 5.0)], mu2)
    np.testing.assert_array_equal(back[(3000.0, 5.0, "F")], 0.5)
    assert filters == {"F": ["X"]}


def test_coefficient_columns_are_named_by_law():
    assert gen.coefficient_column("TESS", "linear", "u") == "TESS"
    assert gen.coefficient_column("TESS", "quadratic", "u1") == "TESS__u1"
    assert gen.coefficient_column("TESS", "quadratic", "u2") == "TESS__u2"


# --------------------------------------------------------------------------
# 2. A synthetic table and its loader
# --------------------------------------------------------------------------

SYNTHETIC = {
    "u1": lambda t, g, z, a: (
        0.40 - 2.0e-5 * (t - 5000.0) + 0.01 * (g - 4.0) + 0.02 * z - 0.005 * a
    ),
    "u2": lambda t, g, z, a: (
        0.25 + 1.0e-5 * (t - 5000.0) - 0.01 * (g - 4.0) - 0.01 * z + 0.002 * a
    ),
    "u": lambda t, g, z, a: (
        0.55 - 3.0e-5 * (t - 5000.0) + 0.01 * (g - 4.0) + 0.01 * z - 0.004 * a
    ),
}
AXES = {
    "teff": np.array([4000.0, 5000.0, 6000.0, 7000.0]),
    "logg": np.array([3.5, 4.0, 4.5, 5.0, 5.5]),
    "feh": np.array([-1.0, -0.5, 0.0, 0.5]),
    "av": np.array([0.0, 0.5, 1.0, 5.0]),
}


def write_synthetic_phoenix_tables(root, facility="TESS", mist="TESS"):
    """Linear-in-every-axis tables for one filter, both laws, in the
    published layout under ``root``; returns ``root``."""
    rows = []
    for t in AXES["teff"]:
        for g in AXES["logg"]:
            for z in AXES["feh"]:
                for a in AXES["av"]:
                    rows.append(
                        {
                            "teff": t,
                            "logg": g,
                            "feh": z,
                            "alpha": 0.0,
                            "Av": a,
                            "Rv": 3.1,
                            f"{mist}__u1": SYNTHETIC["u1"](t, g, z, a),
                            f"{mist}__u2": SYNTHETIC["u2"](t, g, z, a),
                            mist: SYNTHETIC["u"](t, g, z, a),
                        }
                    )
    df = pd.DataFrame(rows)
    meta = {"svo_id": "TESS/TESS.Red", "intensities": "synthetic", "law": ""}
    quad = df.drop(columns=[mist])
    write_bc_table(
        quad,
        gen.ld_table_path(root, "quadratic", facility),
        {
            f"{mist}__u1": {**meta, "coefficient": "u1"},
            f"{mist}__u2": {**meta, "coefficient": "u2"},
        },
        table_meta={
            "facility": facility,
            "model": "PHOENIX",
            "law": "quadratic",
        },
        allow_new_nodes=True,
    )
    lin = df.drop(columns=[f"{mist}__u1", f"{mist}__u2"])
    write_bc_table(
        lin,
        gen.ld_table_path(root, "linear", facility),
        {mist: {**meta, "coefficient": "u"}},
        table_meta={"facility": facility, "model": "PHOENIX", "law": "linear"},
        allow_new_nodes=True,
    )
    return root


@pytest.fixture
def synthetic_phoenix_tables(tmp_path, monkeypatch):
    root = write_synthetic_phoenix_tables(tmp_path / "ld")
    monkeypatch.setattr(ldgrid, "DEFAULT_TABLE_ROOT", root)
    return root


def test_the_loader_reshapes_the_table_and_interpolates_exactly(
    synthetic_phoenix_tables,
):
    """
    Given the synthetic TESS tables,
    When each law is loaded and interpolated at an interior point,
    Then the box is the written axes, the values are exact (linear table,
      linear interpolation), and the graph interpolation agrees with numpy.
    """
    quad = ldgrid.load_ld_table("TESS", "quadratic")
    lin = ldgrid.load_ld_table("TESS", "linear")

    assert quad.columns == ("u1", "u2") and lin.columns == ("u",)
    assert quad.values.shape == (4, 5, 4, 4, 2)
    for ax in ldgrid.AXIS_ORDER:
        np.testing.assert_array_equal(quad.axes[ax], AXES[ax])
    assert quad.box["av"] == (0.0, 5.0) and quad.box["logg"] == (3.5, 5.5)

    point = (5432.0, 4.3, -0.2, 0.7)
    np.testing.assert_allclose(
        ldgrid.interpolate(quad, *point),
        [SYNTHETIC["u1"](*point), SYNTHETIC["u2"](*point)],
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        ldgrid.interpolate(lin, *point), [SYNTHETIC["u"](*point)], rtol=1e-10
    )
    scal = [pt.dscalar(n) for n in ldgrid.AXIS_ORDER]
    f = pytensor.function(scal, ldgrid.interpolate_node(quad, *scal))
    np.testing.assert_allclose(
        f(*point), ldgrid.interpolate(quad, *point), rtol=1e-10
    )


def test_lookup_names_every_way_of_having_no_table(synthetic_phoenix_tables):
    """
    Given the synthetic tables (TESS only),
    When filters and laws with no table are looked up,
    Then each answer names the reason, and load_ld_table raises with it.
    """
    assert ldgrid.ld_table_lookup("TESS", "quadratic") == ("TESS", None)
    name, why = ldgrid.ld_table_lookup("V", "quadratic")
    assert name is None and "Generic" in why and "not present" in why
    name, why = ldgrid.ld_table_lookup("TESS", "power2")
    assert name is None and "power2" in why
    name, why = ldgrid.ld_table_lookup("not-a-filter", "linear")
    assert name is None and "alias table" in why
    with pytest.raises(ValueError, match="not present"):
        ldgrid.load_ld_table("V", "linear")


# --------------------------------------------------------------------------
# 3. The band tie with ld_source: phoenix (the transit harness of
#    tests/test_claret_ld_tie.py)
# --------------------------------------------------------------------------


def _write_lc(path, t0=2459634.3, n=120):
    rng = np.random.default_rng(42)
    t = np.linspace(t0 - 0.2, t0 + 0.2, n)
    flux = 1.0 + rng.normal(0.0, 1e-3, n)
    np.savetxt(path, np.column_stack([t, flux, np.full(n, 1e-3)]))
    return str(path)


def _params():
    return {
        "star.0.radius": {"initval": 1.61, "sigma": 0.05},
        "star.0.mass": {"initval": 1.204, "sigma": 0.05},
        "star.0.teff": {"initval": 6207.0, "sigma": 100.0},
        "star.0.feh": {"initval": -0.116, "sigma": 0.08},
        "orbit.0.period": {"initval": 2.99},
        "orbit.0.tc": {"initval": 2459634.3},
        "orbit.0.cosi": {"initval": 0.05},
        "planet.0.radius": {"initval": 1.7},
    }


def _prepared(tmp_path, band):
    lc = _write_lc(tmp_path / "lc.dat")
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [band],
        "transit": [{"name": "inst0", "file": lc, "band": band["name"]}],
    }
    system = System(config, user_params=_params())
    system.prepare()
    return system


_TESS_PHOENIX = {
    "name": "TESS",
    "filter": "TESS",
    "star_constrains_ld": True,
    "ld_source": "phoenix",
}


def test_a_phoenix_tied_band_bounds_its_star_to_the_tables_box(
    tmp_path, synthetic_phoenix_tables
):
    """
    Given a TESS band tied through the PHOENIX tables and a star with no
      SED (so no A_V),
    When the system is prepared,
    Then du1/du2 are declared, the star is bounded to the table's box on
      teff/logg/feh and NOT on av (it has none), the bounds name their
      source, and the Claret readership is unchanged.
    """
    system = _prepared(tmp_path, _TESS_PHOENIX)
    band = system.band

    assert band.ld_sources == ["phoenix"]
    assert "du1" in band.manifest and "du2" in band.manifest
    overrides = system.config_manager.param_overrides
    for ax in ("teff", "logg", "feh"):
        lo, hi = AXES[ax].min(), AXES[ax].max()
        assert overrides[f"star.0.{ax}"] == {"lower": lo, "upper": hi}
        assert band.grid_bound_paths()[f"star.0.{ax}"]["source"] == (
            "PHOENIX limb-darkening table"
        )
    assert "star.0.av" not in overrides
    assert not band._ld_av_available
    read = {
        (c.param, c.star)
        for c in system.star.structure_consumers(system)
        if c.label == Band.CLARET_CONSUMER_LABEL
    }
    assert read == {("teff", 0), ("feh", 0), ("radius", 0)}


def test_the_phoenix_seed_and_offset_use_the_table_at_av_zero(
    tmp_path, synthetic_phoenix_tables
):
    """
    Given the same system,
    When the start is seeded and the model built,
    Then q1/q2 resolve to the Kipping inversion of the table at the probed
      (teff, logg, feh) and A_V = 0, and in the graph du = u - table at
      the same point, with finite logp and dlogp.
    """
    system = _prepared(tmp_path, _TESS_PHOENIX)
    cm = system.config_manager
    probed = cm.probe_start(["star.0.teff", "star.0.logg", "star.0.feh"])
    point = [
        probed[f"star.0.{ax}"].value for ax in ("teff", "logg", "feh")
    ] + [0.0]

    cfg = {
        p: cm.resolve("band", p, shape=(1,), names=system.band.names)[
            "initval"
        ]
        for p in ("q1", "q2")
    }
    q1, q2 = (
        float(np.atleast_1d(cfg["q1"])[0]),
        float(np.atleast_1d(cfg["q2"])[0]),
    )
    seeded = (
        float(calc_u1_from_kipping(q1, q2).eval()),
        float(calc_u2_from_kipping(q1, q2).eval()),
    )
    np.testing.assert_allclose(
        seeded, [SYNTHETIC["u1"](*point), SYNTHETIC["u2"](*point)], rtol=1e-10
    )

    model = system.build_model()
    band, star = system.band, system.star
    start = system.get_raw_start(model)
    fn = pytensor.function(
        model.value_vars,
        [
            band.u1.value,
            band.u2.value,
            band.du1.value,
            band.du2.value,
            star.teff.value,
            star.logg.value,
            star.feh.value,
        ],
        on_unused_input="ignore",
    )
    u1, u2, du1, du2, teff, logg, feh = fn(
        **{v.name: start[v.name] for v in model.value_vars}
    )
    at = (teff[0], logg[0], feh[0], 0.0)
    np.testing.assert_allclose(du1, u1 - SYNTHETIC["u1"](*at), rtol=1e-10)
    np.testing.assert_allclose(du2, u2 - SYNTHETIC["u2"](*at), rtol=1e-10)
    assert np.isfinite(model.compile_logp()(start))
    assert np.all(np.isfinite(model.compile_dlogp()(start)))


def test_an_unknown_ld_source_raises(tmp_path):
    bad = {**_TESS_PHOENIX, "ld_source": "atlas9"}
    with pytest.raises(ValueError, match="unknown ld_source"):
        _prepared(tmp_path, bad)


# --------------------------------------------------------------------------
# 4. The fetch owner (models/PHOENIX/ld_tables.py): the record's flat names
#    land in the per-law tree, and a local edit is never overwritten.
# --------------------------------------------------------------------------

_PAYLOAD = b"not really a parquet file, but bytes with a pinned md5\n" * 5
_RECORD_NAME = "Fake.quadratic.ld.parquet"


def _owner(monkeypatch, tmp_path):
    import hashlib

    from exozippy.models.PHOENIX import ld_tables
    from exozippy.utilities import zenodo

    files = {
        _RECORD_NAME: {
            "size": len(_PAYLOAD),
            "md5": hashlib.md5(_PAYLOAD).hexdigest(),
        }
    }
    monkeypatch.setattr(ld_tables, "_TABLE_FILES", files)
    monkeypatch.setattr(ld_tables, "_SUPERSEDED", {})
    monkeypatch.setattr(ld_tables, "ZENODO_RECORD", 123)
    monkeypatch.setattr(ld_tables, "MODEL_ROOT", tmp_path / "models")
    monkeypatch.setattr(ld_tables, "_verified", set())
    monkeypatch.setattr(zenodo.time, "sleep", lambda *a: None)
    calls = []

    def serve(url, dest):
        calls.append(url)
        dest.write_bytes(_PAYLOAD)

    monkeypatch.setattr(zenodo, "_urlretrieve", serve)
    return ld_tables, calls


def test_the_owner_fetches_a_flat_record_name_into_the_per_law_tree(
    monkeypatch, tmp_path
):
    """
    Given a pinned record file Fake.quadratic.ld.parquet and an empty tree,
    When the owner ensures it,
    Then it is downloaded once and lands at LDs/quadratic/Fake.ld.parquet,
      a second call downloads nothing, and the loader's lookup resolves the
      facility/law pair through the same names.
    """
    ld_tables, calls = _owner(monkeypatch, tmp_path)

    paths = ld_tables.ensure_tables([_RECORD_NAME])

    assert paths == [
        tmp_path
        / "models"
        / "PHOENIX"
        / "LDs"
        / "quadratic"
        / "Fake.ld.parquet"
    ]
    assert paths[0].read_bytes() == _PAYLOAD
    assert len(calls) == 1
    assert ld_tables.ensure_tables([_RECORD_NAME]) == paths
    assert len(calls) == 1
    assert ld_tables.is_published("Fake", "quadratic")
    assert not ld_tables.is_published("Fake", "linear")
    assert ld_tables.record_name("Fake", "quadratic") == _RECORD_NAME


def test_the_owner_never_overwrites_a_local_edit(monkeypatch, tmp_path):
    """
    Given a table on disk that matches no published version,
    When the owner ensures it,
    Then it raises naming the file and leaves it untouched; a name the
      record does not publish is a KeyError naming it.
    """
    ld_tables, calls = _owner(monkeypatch, tmp_path)
    path = ld_tables.table_path(_RECORD_NAME)
    path.parent.mkdir(parents=True)
    edited = _PAYLOAD.replace(b"bytes", b"BYTES")
    path.write_bytes(edited)

    with pytest.raises(RuntimeError, match="not the published"):
        ld_tables.ensure_tables([_RECORD_NAME])
    assert path.read_bytes() == edited and calls == []
    with pytest.raises(KeyError, match="not published"):
        ld_tables.ensure_tables(["Nope.linear.ld.parquet"])

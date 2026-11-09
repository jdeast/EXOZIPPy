"""The Claret limb-darkening tie (band.star_constrains_ld).

A tied band gets the derived offsets du1 = u1 - u1_Claret (and du2 on a
quadratic band) whose defaults.yaml Gaussian N(0, 0.05) IS the tie -- the
mulensinstrument.zeropoint pattern -- with the table interpolated at the
band's star's (teff, logg, feh), once per band.  The star is held inside the
tables' box: hard bounds on its sampled teff/feh, a soft barrier on its
derived logg.  The rulings are in components/sed/sed.md ("The Claret
limb-darkening tie").

Every System-level test runs against SYNTHETIC tables
(conftest.write_synthetic_claret_tables): the real ones are git-ignored and
will be fetched from Zenodo.  The synthetic coefficients are linear in every
axis, so the trilinear interpolation reproduces them exactly and the values
asserted below need no constant of their own.
"""

import logging

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from conftest import SYNTHETIC_CLARET, _DummyConfigManager
from exozippy.components.band.band import Band, LDConsumer
from exozippy.components.band.claret import (
    claret_band_name,
    interpolate,
    interpolate_node,
    kipping_from_quadratic,
)
from exozippy.components.band.physics import (
    calc_u1_from_kipping,
    calc_u2_from_kipping,
)
from exozippy.models.Claret import claret_tables as ct
from exozippy.system import System


# --------------------------------------------------------------------------
# Fixtures: the transit harness of test_transit_band.py, one or two bands.
# --------------------------------------------------------------------------
def _write_lc(path, t0=2459634.3, n=120):
    rng = np.random.default_rng(42)
    t = np.linspace(t0 - 0.2, t0 + 0.2, n)
    flux = 1.0 + rng.normal(0.0, 1e-3, n)
    np.savetxt(path, np.column_stack([t, flux, np.full(n, 1e-3)]))
    return str(path)


def _config(lc, bands, transit_bands):
    return {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": bands,
        "transit": [
            {"name": f"inst{i}", "file": lc, "band": b}
            for i, b in enumerate(transit_bands)
        ],
    }


def _params(teff=6207.0, teff_sigma=100.0):
    teff_entry = {"initval": teff}
    if teff_sigma is not None:
        teff_entry["sigma"] = teff_sigma
    return {
        "star.0.radius": {"initval": 1.61, "sigma": 0.05},
        "star.0.mass": {"initval": 1.204, "sigma": 0.05},
        "star.0.teff": teff_entry,
        "star.0.feh": {"initval": -0.116, "sigma": 0.08},
        "orbit.0.period": {"initval": 2.99},
        "orbit.0.tc": {"initval": 2459634.3},
        "orbit.0.cosi": {"initval": 0.05},
        "planet.0.radius": {"initval": 1.7},
    }


def _prepared(tmp_path, bands, transit_bands, params=None):
    lc = _write_lc(tmp_path / "lc.dat")
    system = System(
        _config(lc, bands, transit_bands), user_params=params or _params()
    )
    system.prepare()
    return system


_TESS_TIED = {"name": "TESS", "filter": "TESS", "star_constrains_ld": True}
_V_UNTIED = {"name": "V", "filter": "V", "star_constrains_ld": False}


# --------------------------------------------------------------------------
# 1. The table side: names, interpolation, the Kipping inversion.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label, claret",
    [
        ("TESS", "TESS"),
        ("TESS/TESS.Red", "TESS"),
        ("I", "I"),  # bare I -> Cousins I (AMBIGUOUS_FILTER_ALIASES)
        ("Generic/Cousins.R", "R"),
        ("Sloani", "Sloani"),
        ("2MASS/2MASS.Ks", "K"),
        # Claret & Bloemen's U B V are the Johnson-Cousins system.
        ("Generic/Johnson.U", "U"),
        ("Generic/Johnson.B", "B"),
        ("Generic/Johnson.V", "V"),
    ],
)
def test_filter_labels_resolve_to_claret_tables(label, claret):
    """
    Given a band filter label in any spelling the alias table knows,
    When its Claret table is looked up,
    Then it is the alias table's Claret column for that filter.
    """
    assert claret_band_name(label, "test") == claret


@pytest.mark.parametrize(
    "label",
    [
        "GAIA/GAIA2r.G",
        "NoSuchFilter",
        # Johnson R/I/J are not Claret's Cousins R/I or 2MASS-like J.
        "Generic/Johnson.R",
        "Generic/Johnson.I",
        "Generic/Johnson.J",
    ],
)
def test_a_filter_with_no_claret_table_raises(label):
    """
    Given a filter the alias table marks Unsupported, or does not know,
    When its Claret table is looked up,
    Then it RAISES, naming the way out -- never handing back the label as
    bc_grid.resolve_filter_name does, which would go looking for a "Gaia"
    Claret table.
    """
    with pytest.raises(ValueError, match="star_constrains_ld: false"):
        claret_band_name(label, "test")


def test_interpolation_is_exact_on_a_linear_table_and_matches_the_graph(
    synthetic_claret_tables,
):
    """
    Given a table whose coefficients are linear in teff, logg and feh,
    When it is interpolated off-node, in numpy and as a PyTensor graph,
    Then both reproduce the linear function -- the seed and the model
    compute the same number.
    """
    table = ct.load_table("TESS", "quadratic")
    point = (6123.0, 4.37, 0.13)
    expected = [SYNTHETIC_CLARET[c](*point) for c in ("a", "b")]

    np.testing.assert_allclose(
        interpolate(table, *point), expected, rtol=1e-12
    )

    t, g, z = pt.dscalar(), pt.dscalar(), pt.dscalar()
    fn = pytensor.function([t, g, z], interpolate_node(table, t, g, z))
    np.testing.assert_allclose(fn(*point), expected, rtol=1e-12)


def test_a_missing_table_is_a_runtime_error_and_a_nonexistent_one_a_value_error(
    tmp_path,
):
    """
    Given an empty, explicitly named table directory,
    When a table that should exist is read, and one that cannot exist,
    Then the first raises RuntimeError naming the directory (never
    FileNotFoundError, the BC-table fetch's rule; and an explicit directory
    is never fetched into) and the second ValueError -- there is no linear
    TESS table.
    """
    with pytest.raises(RuntimeError, match="explicitly given table directory"):
        ct.load_table("V", "quadratic", tmp_path)
    with pytest.raises(ValueError, match="no linear Claret table for 'TESS'"):
        ct.load_table("TESS", "linear", tmp_path)


@pytest.mark.parametrize("u1, u2", [(0.4, 0.25), (0.1, 0.6), (0.9, -0.2)])
def test_kipping_inversion_round_trips(u1, u2):
    """
    Given a quadratic (u1, u2) inside the physical triangle,
    When it is mapped to Kipping (q1, q2) and back through band/physics.py,
    Then it comes back unchanged -- the seed lands on the table value.
    """
    q1, q2 = kipping_from_quadratic(u1, u2)
    back = (
        float(calc_u1_from_kipping(q1, q2).eval()),
        float(calc_u2_from_kipping(q1, q2).eval()),
    )
    np.testing.assert_allclose(back, (u1, u2), rtol=1e-12)


@pytest.mark.parametrize("q1, q2", [(0.1, 0.2), (0.5, 0.5), (0.9, 0.8)])
def test_the_kipping_map_has_unit_jacobian(q1, q2):
    """
    Given the Kipping map (q1, q2) -> (u1, u2),
    When its Jacobian determinant is evaluated,
    Then |det| = 1 everywhere: uniform (q1, q2) IS uniform (u1, u2) over the
    triangle, so a Gaussian on (u1, u2) times the q1/q2 sampling is exactly
    EXOFASTv2's truncated Gaussian and the tie owes no Jacobian term.
    """
    a, b = pt.dscalar(), pt.dscalar()
    u = pt.stack([calc_u1_from_kipping(a, b), calc_u2_from_kipping(a, b)])
    jac = pytensor.function([a, b], pt.jacobian(u, [a, b]))(q1, q2)
    det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]
    assert abs(det) == pytest.approx(1.0, rel=1e-12)


# --------------------------------------------------------------------------
# 2. The flag.
# --------------------------------------------------------------------------
def test_star_constrains_ld_must_be_a_yaml_boolean():
    """
    Given a band with star_constrains_ld: "false" (quoted),
    When the band loads,
    Then it RAISES: bool("false") is True, so accepting it would turn the
    tie -- and bounds on the star -- ON for a user who asked for OFF.
    """
    band = Band(
        [{"name": "V", "filter": "V", "star_constrains_ld": "false"}],
        _DummyConfigManager(),
    )
    with pytest.raises(ValueError, match="YAML boolean"):
        band.load_data(system=None)


def _write_pspl_lc(path, t0=2460025.0, te=30.0, u0=0.1, n=60):
    t = np.linspace(t0 - 2 * te, t0 + 2 * te, n)
    u = np.sqrt(u0**2 + ((t - t0) / te) ** 2)
    amp = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    mag = 18.0 - 2.5 * np.log10(amp)
    np.savetxt(path, np.column_stack([t, mag, np.full(n, 0.01)]))
    return str(path)


def _prepared_fspl(tmp_path, band, galactic=False):
    """A finite-source PSPL whose only limb-darkening reader is the
    microlensing light curve, with a lens (star 0) and a source (star 1)
    whose Teff/[Fe/H]/radius the params file states."""
    lc = _write_pspl_lc(tmp_path / "lc.dat")
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "mulensevent": [
            {"finite_source": True, "t0_par": 2460025.0, "use_op": False}
        ],
        "lens": [{"body": "star.Lens"}],
        "source": [{"body": "star.Source"}],
        "band": [band],
        "mulensinstrument": [{"name": "OGLE", "file": lc, "band": "I"}],
    }
    if galactic:
        config["galacticmodel"] = [{"name": "gm", "anchor_idx": 1}]
    params = {
        "source.Source.t_0": {"initval": 2460025.0},
        "source.Source.u_0": {"initval": 0.1},
        "source.Source.rho": {"initval": 1.0e-3},
        "mulensevent.t_E": {"initval": 30.0},
        "star.Source.teff": {"initval": 4800.0, "sigma": 200.0},
        "star.Source.feh": {"initval": 0.1, "sigma": 0.2},
        "star.Source.radius": {"initval": 1.2, "sigma": 0.1},
    }
    for nm in ("Lens", "Source"):
        params[f"star.{nm}.ra"] = {"initval": 264.0, "sigma": 0}
        params[f"star.{nm}.dec"] = {"initval": -27.0, "sigma": 0}
    system = System(config, user_params=params)
    system.prepare()
    return system


@pytest.mark.parametrize(
    "law, offsets", [("linear", ("du1",)), ("quadratic", ("du1", "du2"))]
)
def test_a_microlensing_band_is_tied_to_the_source_star(
    tmp_path, synthetic_claret_tables, law, offsets
):
    """
    Given a band read only by a finite-source microlensing light curve,
    with star_constrains_ld unset,
    When the system is prepared and built,
    Then the band is tied exactly as a transit band would be: the tie is a
    property of the band (its filter, its star), never of what reads it.
    Its star is the SOURCE (star 1), which is bounded to the tables' box,
    read by the tie, and whose start seeds the limb darkening.
    """
    system = _prepared_fspl(
        tmp_path, {"name": "I", "filter": "I", "ld_law": law}
    )
    band = system.band
    assert band.claret_tie_stars(system) == {0: 1}
    for name in offsets:
        assert band.manifest[name] == {"expr_key": "default"}
    assert ("du2" in band.manifest) == (law == "quadratic")

    overrides = system.config_manager.param_overrides
    for ax in ("teff", "logg", "feh"):
        assert overrides[f"star.1.{ax}"] == {
            "lower": ct.BOX[ax][0],
            "upper": ct.BOX[ax][1],
        }
        assert f"star.0.{ax}" not in overrides

    read = {
        (c.param, c.star)
        for c in system.star.structure_consumers(system)
        if c.label == Band.CLARET_CONSUMER_LABEL
    }
    assert read == {("teff", 1), ("feh", 1), ("radius", 1)}

    model = system.build_model()
    assert "band.du1" in model.named_vars


@pytest.mark.parametrize("tied", [True, False])
def test_a_tied_microlensing_source_keeps_its_mass_free(
    tmp_path, synthetic_claret_tables, tied
):
    """
    Given a finite-source event with a galactic model, whose source star's
    mass is otherwise pinned (a microlensing-only source: Star's tier-2
    "overrides" pin, since only the mass prior reads it),
    When its band is tied to the Claret tables,
    Then the source's logmass is NOT pinned: the tie reads its logg =
    f(logmass, radius), and a pinned mass would leave the tables' logg
    barrier pressing on the radius -- on rho -- against an arbitrary fixed
    mass (examples/ob161045 started 466 nats into it).  Untied, the pin
    stands, so the test sees the switch.
    """
    system = _prepared_fspl(
        tmp_path,
        {
            "name": "I",
            "filter": "I",
            "ld_law": "linear",
            "star_constrains_ld": tied,
        },
        galactic=True,
    )
    # Pinned = a sigma of 0 on the source in the overrides channel; with no
    # element pinned the entry carries no sigma override at all.
    entry = system.star.manifest["logmass"]
    overrides = entry.get("overrides", {}) if isinstance(entry, dict) else {}
    sigma = overrides.get("sigma")
    pinned = sigma is not None and sigma[1] == 0.0
    assert pinned == (not tied)


def test_an_explicit_false_leaves_a_microlensing_band_untied(
    tmp_path, synthetic_claret_tables
):
    """
    Given the same microlensing band with star_constrains_ld: false,
    When the system is prepared,
    Then nothing is tied and the source star is not bounded.
    """
    system = _prepared_fspl(
        tmp_path,
        {
            "name": "I",
            "filter": "I",
            "ld_law": "linear",
            "star_constrains_ld": False,
        },
    )
    assert system.band.claret_tie_stars(system) == {}
    assert "du1" not in system.band.manifest
    assert "star.1.teff" not in system.config_manager.param_overrides


# --------------------------------------------------------------------------
# 3. On by default; false turns it off.
# --------------------------------------------------------------------------
def test_false_turns_the_tie_off_completely(tmp_path):
    """
    Given a transit fit whose band sets star_constrains_ld: false,
    When the system is prepared,
    Then there are no offsets, no bounds on the star and no new reader of
    its Teff -- EXOFASTv2's /NOCLARET.  (No tables are needed at all, and
    none are provided.)
    """
    system = _prepared(
        tmp_path,
        [{"name": "TESS", "filter": "TESS", "star_constrains_ld": False}],
        ["TESS"],
    )

    assert "du1" not in system.band.manifest
    assert "du2" not in system.band.manifest
    assert not any(
        k.startswith("star.") for k in system.config_manager.param_overrides
    )
    assert system.band.grid_bound_paths() == {}
    labels = {c.label for c in system.star.structure_consumers(system)}
    assert Band.CLARET_CONSUMER_LABEL not in labels


def test_an_unset_band_is_tied_by_default(tmp_path, synthetic_claret_tables):
    """
    Given a transit band that does not mention star_constrains_ld,
    When the system is prepared,
    Then it carries the tie exactly as an explicit true would: offsets
    declared, the star bounded and read (EXOFASTv2's default).
    """
    system = _prepared(
        tmp_path, [{"name": "TESS", "filter": "TESS"}], ["TESS"]
    )

    assert system.band.manifest["du1"] == {"expr_key": "default"}
    assert "star.0.teff" in system.config_manager.param_overrides
    labels = {c.label for c in system.star.structure_consumers(system)}
    assert Band.CLARET_CONSUMER_LABEL in labels


def test_an_unset_band_with_no_table_raises_naming_the_remedies(
    tmp_path, synthetic_claret_tables
):
    """
    Given an unset band whose limb darkening a transit reads and whose
    filter has no Claret table (Gaia G), next to a tableable one,
    When the system is prepared,
    Then it RAISES rather than quietly leaving the band untied, naming the
    band and all three ways out: ld_source: phoenix, claret_filter: (with
    the nearest Claret band), and star_constrains_ld: false (JDE
    2026-11-06).
    """
    bands = [
        {"name": "TESS", "filter": "TESS"},
        {"name": "G", "filter": "Gaia.G"},
    ]
    with pytest.raises(ValueError) as err:
        _prepared(tmp_path, bands, ["TESS", "G"])
    msg = str(err.value)
    assert "band.G:" in msg
    for remedy in (
        "ld_source: phoenix",
        "claret_filter:",
        "star_constrains_ld: false",
        "nearest Claret band",
    ):
        assert remedy in msg, (remedy, msg)


def test_claret_filter_ties_a_band_to_the_named_table_and_says_so(
    tmp_path, synthetic_claret_tables, caplog
):
    """
    Given a Gaia G band with claret_filter: TESS,
    When the system is prepared,
    Then the band is tied to the TESS table, its filter stays Gaia G, and
    the fit warns that the limb darkening is an APPROXIMATION.
    """
    import logging

    bands = [{"name": "G", "filter": "Gaia.G", "claret_filter": "TESS"}]
    with caplog.at_level(logging.WARNING, logger="exozippy"):
        system = _prepared(tmp_path, bands, ["G"])

    assert system.band._claret_tables[0].band == "TESS"
    assert system.band.filter_names[0] == "Gaia.G"
    assert any(
        "APPROXIMATION" in r.getMessage() and "TESS" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"claret_filter": "NoSuchFilter"}, "claret_filter"),
        ({"claret_filter": "TESS", "ld_source": "phoenix"}, "ld_source"),
    ],
)
def test_a_claret_filter_that_cannot_be_honored_raises(
    tmp_path, synthetic_claret_tables, extra, match
):
    """
    Given a claret_filter naming no Claret table, or one on a band whose
    ld_source is phoenix (built through the band's own filter),
    When the system is prepared,
    Then it raises naming the key -- never ignored.
    """
    bands = [{"name": "G", "filter": "Gaia.G", **extra}]
    with pytest.raises(ValueError, match=match):
        _prepared(tmp_path, bands, ["G"])


# --------------------------------------------------------------------------
# 4. The tie, prepared.
# --------------------------------------------------------------------------
def test_a_tied_band_declares_offsets_bounds_its_star_and_reads_it(
    tmp_path, synthetic_claret_tables
):
    """
    Given a TESS band with the tie on and a V band without it,
    When the system is prepared,
    Then du1/du2 are derived on TESS and inactive on V, the star is bounded
    to the tables' box on teff/logg/feh, the bounds are published as grid
    bounds, and Star counts the tie as reading teff/feh/radius.
    """
    system = _prepared(tmp_path, [_TESS_TIED, _V_UNTIED], ["TESS", "V"])
    band = system.band

    for name in ("du1", "du2"):
        entry = band.manifest[name]
        np.testing.assert_array_equal(entry["mask"], [True, False])
        assert entry["inactive_value"] == 0.0

    overrides = system.config_manager.param_overrides
    for ax in ("teff", "logg", "feh"):
        assert overrides[f"star.0.{ax}"] == {
            "lower": ct.BOX[ax][0],
            "upper": ct.BOX[ax][1],
        }
        # Per tied star, never the broadcast label (an untied star's wall
        # is not the tables running out).
        assert band.grid_bound_paths()[f"star.0.{ax}"]["source"] == (
            "Claret limb-darkening table"
        )
        assert f"star.{ax}" not in band.grid_bound_paths()

    read = {
        (c.param, c.star)
        for c in system.star.structure_consumers(system)
        if c.label == Band.CLARET_CONSUMER_LABEL
    }
    assert read == {("teff", 0), ("feh", 0), ("radius", 0)}


def test_the_seed_is_the_table_value_at_the_stars_start(
    tmp_path, synthetic_claret_tables
):
    """
    Given a tied quadratic band and an informed star start,
    When the system is prepared,
    Then q1/q2 resolve to the Kipping inversion of the table at the probed
    (teff, logg, feh) -- so the start sits on the table (du = 0) -- while
    the untied band keeps its defaults.yaml start.
    """
    system = _prepared(tmp_path, [_TESS_TIED, _V_UNTIED], ["TESS", "V"])
    cm = system.config_manager
    probed = cm.probe_start(["star.0.teff", "star.0.logg", "star.0.feh"])
    point = [probed[f"star.0.{ax}"].value for ax in ("teff", "logg", "feh")]

    cfg = {
        p: cm.resolve("band", p, shape=(2,), names=system.band.names)[
            "initval"
        ]
        for p in ("q1", "q2")
    }
    q1, q2 = cfg["q1"][0], cfg["q2"][0]
    seeded = (
        float(calc_u1_from_kipping(q1, q2).eval()),
        float(calc_u2_from_kipping(q1, q2).eval()),
    )
    expected = [SYNTHETIC_CLARET[c](*point) for c in ("a", "b")]
    np.testing.assert_allclose(seeded, expected, rtol=1e-10)

    defaults = cm.base_defaults["band"]
    assert cfg["q1"][1] == defaults["q1"]["initval"]
    assert cfg["q2"][1] == defaults["q2"]["initval"]


def test_a_user_initval_beats_the_seed(tmp_path, synthetic_claret_tables):
    """
    Given a tied band whose q1 start the params file states,
    When the system is prepared,
    Then the user's start stands: the seed is a hint, ranked below the user.
    """
    params = _params()
    params["band.TESS.q1"] = {"initval": 0.5}
    system = _prepared(tmp_path, [_TESS_TIED], ["TESS"], params)

    q1 = system.config_manager.resolve(
        "band", "q1", shape=(1,), names=system.band.names
    )["initval"]
    assert float(np.atleast_1d(q1)[0]) == 0.5


def test_a_star_starting_below_the_tables_raises(
    tmp_path, synthetic_claret_tables
):
    """
    Given a tied band whose star starts at 3200 K, below the tables' 3500 K,
    When the system is prepared,
    Then it RAISES with both remedies -- EXOFASTv2 would need /NOCLARET.
    """
    with pytest.raises(ValueError, match="star_constrains_ld: false") as exc:
        _prepared(tmp_path, [_TESS_TIED], ["TESS"], _params(teff=3200.0))
    assert "teff = 3200" in str(exc.value)


def test_a_tied_band_nothing_reads_raises(tmp_path, synthetic_claret_tables):
    """
    Given a tied V band nothing reads the limb darkening of,
    When the system is prepared,
    Then it RAISES: there is nothing to tie.
    """
    v_tied = {"name": "V", "filter": "V", "star_constrains_ld": True}
    with pytest.raises(ValueError, match="nothing to tie"):
        _prepared(tmp_path, [_TESS_TIED, v_tied], ["TESS"])


def test_a_linear_tess_band_raises(tmp_path, synthetic_claret_tables):
    """
    Given a tied TESS band with ld_law: linear,
    When the system is prepared,
    Then it RAISES: Claret (2017) is used for the quadratic law only.
    """
    band = dict(_TESS_TIED, ld_law="linear")
    with pytest.raises(ValueError, match="no linear Claret table for TESS"):
        _prepared(tmp_path, [band], ["TESS"])


def test_a_tie_on_an_otherwise_unread_teff_warns(
    tmp_path, synthetic_claret_tables, caplog
):
    """
    Given a tied band whose star's Teff has a start but no prior, no SED,
    no track and no relation,
    When the system is prepared,
    Then it WARNS that the tie is the only thing reading that Teff -- the
    user asked for it, so it is not refused.
    """
    with caplog.at_level(
        logging.WARNING, logger="exozippy.components.band.band"
    ):
        _prepared(tmp_path, [_TESS_TIED], ["TESS"], _params(teff_sigma=None))
    assert "read by nothing but the Claret" in caplog.text


# --------------------------------------------------------------------------
# 5. The tie, built: the Gaussian is EXOFASTv2's term, once per band.
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_the_tie_is_a_gaussian_on_the_offset_from_the_table(
    tmp_path, synthetic_claret_tables
):
    """
    Given a built model with one tied TESS band read by TWO light curves,
    When du1, du2, the star's (teff, logg, feh) and the tie's potential are
    evaluated at the start,
    Then du = u - table(teff, logg, feh), and the potential is
    -0.5 * sum((du / 0.05)^2) over ONE band -- not once per light curve as
    EXOFASTv2 counts it -- and logp and dlogp are finite.
    """
    system = _prepared(tmp_path, [_TESS_TIED], ["TESS", "TESS"])
    model = system.build_model()
    band, star = system.band, system.star
    start = system.get_raw_start(model)

    potential = next(
        p for p in model.potentials if p.name == "gaussian_prior.band.du1"
    )
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
            potential,
        ],
        on_unused_input="ignore",
    )
    u1, u2, du1, du2, teff, logg, feh, pot = fn(
        **{v.name: start[v.name] for v in model.value_vars}
    )
    point = (teff[0], logg[0], feh[0])

    assert du1.shape == (1,)
    np.testing.assert_allclose(
        du1, u1 - SYNTHETIC_CLARET["a"](*point), rtol=1e-10
    )
    np.testing.assert_allclose(
        du2, u2 - SYNTHETIC_CLARET["b"](*point), rtol=1e-10
    )
    np.testing.assert_allclose(
        pot, -0.5 * np.sum((du1 / 0.05) ** 2), rtol=1e-10
    )

    assert np.isfinite(model.compile_logp()(start))
    assert np.all(np.isfinite(model.compile_dlogp()(start)))


# --------------------------------------------------------------------------
# 6. Diagnostics: two grids bounding one parameter.
# --------------------------------------------------------------------------
def test_grid_bounds_from_two_components_are_merged_not_replaced():
    """
    Given the SED and the Claret tie both declaring a grid bound on star.feh,
    When the wrap-up asks for every grid-bounded path,
    Then both sources are named and the bounds are their intersection --
    a plain dict update used to keep only the later component's claim.
    """
    from types import SimpleNamespace

    from exozippy.diagnostics import grid_bounded_paths

    def comp(paths):
        return SimpleNamespace(grid_bound_paths=lambda: paths)

    system = SimpleNamespace(
        active_components={
            "sed": comp(
                {
                    "star.feh": {
                        "lower": -4.0,
                        "upper": 0.5,
                        "source": "BC grid",
                    }
                }
            ),
            "band": comp(
                {
                    "star.feh": {
                        "lower": -2.0,
                        "upper": 1.0,
                        "source": "Claret table",
                    }
                }
            ),
        }
    )
    merged = grid_bounded_paths(system)["star.feh"]
    assert merged == {
        "lower": -2.0,
        "upper": 0.5,
        "source": "BC grid and Claret table",
    }

"""Disambiguation of bare filter labels in the shared alias table.

``src/exozippy/filters/filternames.txt`` has five naming columns and
``resolve_filter_name`` matches a user's ``filter:`` string against all of
them, taking the first row that hits.  Two labels hit twice: a bare "I" and
a bare "R" appear in the Claret column for BOTH the Bessell and the Cousins
rows (and once more in the Keivan column, for Bessell).  First-row-wins made
them Bessell purely because Bessell_I is typed above Cousins_I in the file --
an accident of file order.

The project convention is Cousins, declared in
``bc_grid.AMBIGUOUS_FILTER_ALIASES`` and applied by name rather than by row
order.  These tests pin that, pin the inventory of ambiguous labels (so a new
one cannot arrive silently), and pin the claim that made the change safe to
make: filter identity does not reach limb darkening, so re-pointing "I" moves
no LD prior.
"""

import collections

import numpy as np
import pytest

from exozippy.components.sed.bc_grid import (
    AMBIGUOUS_FILTER_ALIASES,
    _load_alias_table,
    resolve_filter_name,
)
from exozippy.system import System


@pytest.fixture(scope="module")
def alias_df():
    df = _load_alias_table()
    if df is None:
        pytest.skip("filter alias table not installed")
    return df


# ---------------------------------------------------------------- resolution


@pytest.mark.parametrize(
    "bare,mist,svo",
    [
        ("I", "Cousins_I", "Generic/Cousins.I"),
        ("R", "Cousins_R", "Generic/Cousins.R"),
    ],
)
def test_bare_i_and_r_resolve_to_cousins(alias_df, bare, mist, svo):
    """
    Given a bare 'I' or 'R', which names two rows of the alias table,
    When it is resolved to a MIST or an SVO name,
    Then it resolves to the Cousins row, not the Bessell one.
    """
    assert resolve_filter_name(bare, alias_df, alias="MIST") == mist
    assert resolve_filter_name(bare, alias_df, alias="SVO") == svo


@pytest.mark.parametrize(
    "spelled,mist,svo",
    [
        ("Bessell.I", "Bessell_I", "Generic/Bessell.I"),
        ("Generic/Bessell.R", "Bessell_R", "Generic/Bessell.R"),
        ("Cousins.I", "Cousins_I", "Generic/Cousins.I"),
        ("Generic/Cousins.R", "Cousins_R", "Generic/Cousins.R"),
    ],
)
def test_spelled_out_names_are_honored(alias_df, spelled, mist, svo):
    """
    Given a filter name that names exactly one row,
    When it is resolved,
    Then the disambiguation map does not touch it: Bessell stays Bessell.
    """
    assert resolve_filter_name(spelled, alias_df, alias="MIST") == mist
    assert resolve_filter_name(spelled, alias_df, alias="SVO") == svo


@pytest.mark.parametrize(
    "name,mist,svo",
    [
        ("V", "Bessell_V", "Generic/Bessell.V"),
        ("K", "2MASS_Ks", "2MASS/2MASS.Ks"),
        ("Sloani", "SDSS_i", "SLOAN/SDSS.i"),
        ("TESS", "TESS", "TESS/TESS.Red"),
    ],
)
def test_unambiguous_bare_names_are_unaffected(alias_df, name, mist, svo):
    """
    Given a bare label that names exactly one row,
    When it is resolved,
    Then it resolves exactly as before the disambiguation map existed.
    """
    assert resolve_filter_name(name, alias_df, alias="MIST") == mist
    assert resolve_filter_name(name, alias_df, alias="SVO") == svo


def test_disambiguation_map_keys_pick_exactly_one_row(alias_df):
    """
    Given the AMBIGUOUS_FILTER_ALIASES map,
    When each key and each target is looked up in the table,
    Then every key really is ambiguous and every target names exactly one
    row -- the map cannot silently point at nothing.
    """
    for bare, mist in AMBIGUOUS_FILTER_ALIASES.items():
        hits = int(alias_df.eq(bare).any(axis=1).sum())
        assert hits > 1, f"{bare!r} is not ambiguous; drop it from the map"
        rows = alias_df[alias_df["MIST"] == mist]
        assert len(rows) == 1, f"{mist!r} does not name exactly one row"


# ---------------------------------------------------------------- inventory


def test_alias_table_ambiguities_are_the_declared_ones(alias_df):
    """
    Given every naming column of the alias table,
    When labels appearing on more than one row are collected,
    Then they are exactly the known set: 'I' and 'R' (Claret, Bessell vs
    Cousins), which the disambiguation map resolves, plus 'iPS' (Keivan,
    PS1.i vs PS1.w), which is left alone and recorded here.

    A new duplicate must fail this test rather than quietly resolve by file
    order, which is the bug this module exists to prevent.
    """
    # Arrange
    known_resolved = set(AMBIGUOUS_FILTER_ALIASES)
    # Ambiguous, deliberately NOT in the map: 'iPS' is Keivan's label for
    # PAN-STARRS i, and the second row it hits is PS1.w (the wide filter),
    # whose own Keivan cell should probably read 'wPS'.  That is a table
    # fix, not a resolution policy, and no shipped config uses either name.
    known_unresolved = {"iPS"}

    # Act
    found = set()
    for col in alias_df.columns:
        counts = collections.Counter(
            v for v in alias_df[col].dropna() if v != "Unsupported"
        )
        found |= {k for k, n in counts.items() if n > 1}

    # Assert
    assert found == known_resolved | known_unresolved


# ---------------------------------------------------------- LD independence


def _lc(path, t0=2459634.3, n=80):
    rng = np.random.default_rng(7)
    t = np.linspace(t0 - 0.2, t0 + 0.2, n)
    np.savetxt(
        path,
        np.column_stack([t, 1.0 + rng.normal(0.0, 1e-3, n), np.full(n, 1e-3)]),
    )
    return str(path)


def _transit_system(lc_file, filter_name, build=False):
    config = {
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b"}],
        "band": [{"name": "B", "filter": filter_name}],
        "transit": [{"name": "t0", "file": lc_file, "band": "B"}],
    }
    params = {
        "star.0.radius": {"initval": 1.61, "sigma": 0.05},
        "star.0.mass": {"initval": 1.204, "sigma": 0.05},
        "star.0.teff": {"initval": 6207, "sigma": 100},
        "star.0.feh": {"initval": -0.116, "sigma": 0.08},
        "orbit.0.period": {"initval": 2.99},
        "orbit.0.tc": {"initval": 2459634.3},
        "orbit.0.cosi": {"initval": 0.05},
        "planet.0.radius": {"initval": 1.7},
    }
    system = System(config, user_params=params)
    system.prepare()
    if build:
        # Parameter objects (and so their resolved priors) are materialized
        # at stage 5, inside build_model().
        system.build_model()
    return system


def _ld_fields(system):
    out = {}
    for name in ("q1", "q2", "u1", "u2"):
        p = getattr(system.band, name, None)
        if p is None:
            continue
        out[name] = {
            f: np.array(getattr(p, f), dtype=float).tolist()
            for f in ("initval", "lower", "upper", "sigma")
            if getattr(p, f) is not None
        }
    return out


def test_limb_darkening_priors_do_not_depend_on_the_filter(tmp_path):
    """
    Given two otherwise identical transit systems whose band names the
    Bessell I and the Cousins I filter,
    When each is prepared,
    Then their band limb-darkening parameters (q1/q2/u1/u2 initvals, bounds
    and sigmas) are identical.

    This is the claim that made re-pointing a bare 'I' safe: there is no
    limb-darkening table keyed on filter identity, so the affected bands get
    the same wide priors either way and no shipped fit's numbers move.  If a
    filter-dependent LD prior is ever added, this test fails and the
    Bessell -> Cousins change stops being a no-op.
    """
    # Arrange
    lc = _lc(tmp_path / "lc.dat")

    # Act
    bessell = _ld_fields(_transit_system(lc, "Generic/Bessell.I", build=True))
    cousins = _ld_fields(_transit_system(lc, "Generic/Cousins.I", build=True))

    # Assert
    assert bessell  # the band really does carry LD parameters
    assert bessell == cousins


def test_bare_i_band_resolves_to_cousins_in_a_prepared_system(tmp_path):
    """
    Given a band whose config says `filter: "I"`,
    When the system is prepared,
    Then the band's canonical MIST/SVO identities are the Cousins ones.
    """
    # Arrange / Act
    system = _transit_system(_lc(tmp_path / "lc.dat"), "I")

    # Assert
    assert system.band.filter_mist == ["Cousins_I"]
    assert system.band.filter_svo == ["Generic/Cousins.I"]

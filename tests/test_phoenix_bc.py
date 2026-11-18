"""The PHOENIX bolometric-correction tables (`model: PHOENIX` in the SED
file): the generator's physics, the pending Zenodo record and its fetch,
and the SED's selection of the model and its filter coverage.

Nothing here reads the PHOENIX archive or Zenodo: the generator is driven
with synthetic intensities and a synthetic flux spectrum, the record is
replaced by a fake one whose "download" is a local copy, and the SED reads
a synthetic PHOENIX tree under an explicit ``model_root:``.  The design is
in src/exozippy/components/sed/sed.md, "Choosing the atmospheres".
"""

import hashlib
import shutil
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from astropy import constants as const
from astropy import units as u
from astropy.io import fits

from exozippy.components.sed import bc_grid
from exozippy.components.sed.bc_grid import (
    BC_PARAM_COLS,
    DEFAULT_MODEL_ROOT,
    write_bc_table,
)
from exozippy.components.sed.magsys import read_magsys_table
from exozippy.models.PHOENIX import bc_tables, specint
from exozippy.models.PHOENIX import generate_PHOENIX_LD_Tables as gen
from exozippy.utilities import zenodo_assets
from exozippy.utilities.zenodo_assets import ZenodoFile, ZenodoRecord

_PHOENIX_BCS = DEFAULT_MODEL_ROOT / "PHOENIX" / "BCs"
_NEXTGEN_BCS = DEFAULT_MODEL_ROOT / "NextGen" / "BCs"
# The NextGen columns the PHOENIX intensities (500-26000 A) cannot carry.
_UNCOVERED = {
    "WISE_W1",
    "WISE_W2",
    "WISE_W3",
    "WISE_W4",
    "IRAC_3.6",
    "Johnson_M",
}

# --------------------------------------------------------------------------
# 1. The generator's physics
# --------------------------------------------------------------------------


def test_the_band_flux_is_2pi_int_i_mu_dmu_per_angstrom():
    """
    Given band intensities that do not depend on mu (a uniform disk) on a
      mu grid from 0 to 1,
    When the band flux is formed,
    Then it is pi times the intensity, times 1e-8 to go from the archive's
      per-centimetre intensities to NextGen's per-Angstrom flux.
    """
    mu = np.linspace(0.0, 1.0, 101)
    raw = np.full((2, 3, mu.size), 7.0)

    flux = gen.band_fluxes(raw, mu)

    np.testing.assert_allclose(flux, np.pi * 7.0 * 1e-8, rtol=1e-12)
    assert flux.shape == (2, 3)


def test_the_bc_is_nextgens_formula_with_the_models_own_bolometric_flux():
    """
    Given a band flux and a filter normalization,
    When the BC is formed with fbol = sigma Teff^4,
    Then it is NextGen's BolometricCorrection formula exactly (L0 at 10 pc
      over sigma Teff^4, times the band flux over ZP int S); and a model
      whose bolometric flux is half that -- a spherical model's flux at its
      outer radius -- is 2.5 log10(2) mag BRIGHTER in every band, because
      only the ratio of band to bolometric flux is physical.
    """
    teff = 5800.0
    sig_t4_cgs = const.sigma_sb.cgs.value * teff**4
    fdata = {"filter_zero_flux": np.array([3.0e-9, 1.2e-9])}
    band = np.array([[2.0e-3, 5.0e-4]])

    bc = gen.bc_from_fluxes(band, sig_t4_cgs, fdata)
    bc_half = gen.bc_from_fluxes(band, 0.5 * sig_t4_cgs, fdata)

    # models/NextGen/bolometric_correction.py BolometricCorrection._calc_BC
    term1 = 3.0128e28 / (
        100 * u.pc.to(u.m) ** 2 * 4 * np.pi * const.sigma_sb.value * teff**4
    )
    expected = 2.5 * np.log10(term1 * band / fdata["filter_zero_flux"])
    np.testing.assert_allclose(bc, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(bc_half - bc, 2.5 * np.log10(2.0), atol=1e-12)


def _write_flux_fits(path, flux_per_cm, teff=5800.0, logg=4.5, feh=0.0):
    hdr = fits.Header()
    hdr["PHXTEFF"] = teff
    hdr["PHXLOGG"] = logg
    hdr["PHXM_H"] = feh
    fits.PrimaryHDU(np.asarray(flux_per_cm, np.float32), hdr).writeto(path)


def test_the_bolometric_flux_adds_a_rayleigh_jeans_tail(tmp_path):
    """
    Given a flux spectrum that is constant at 2e8 erg/s/cm^2/cm (2
      per Angstrom) from 500 to 55000 A,
    When its bolometric flux is read,
    Then it is the integral over the file's range plus the tail
      F(lam0) lam0 / 3 beyond it, and the tail fraction is reported.
    """
    wave = np.linspace(500.0, 55000.0, 5001)
    path = tmp_path / "flux.fits"
    _write_flux_fits(path, np.full(wave.size, 2.0e8))

    info = specint.bolometric_flux(path, wave)

    inside = 2.0 * (55000.0 - 500.0)
    tail = 2.0 * 55000.0 / 3.0
    assert info["fbol"] == pytest.approx(inside + tail, rel=1e-6)
    assert info["tail_fraction"] == pytest.approx(
        tail / (inside + tail), rel=1e-6
    )
    assert (info["teff"], info["logg"], info["feh"]) == (5800.0, 4.5, 0.0)


def test_a_flux_spectrum_on_another_wavelength_grid_raises(tmp_path):
    path = tmp_path / "flux.fits"
    _write_flux_fits(path, np.ones(10))
    with pytest.raises(ValueError, match="does not match"):
        specint.bolometric_flux(path, np.linspace(500.0, 1000.0, 11))


def test_bc_holes_are_filled_along_teff_and_recorded():
    """
    Given BCs at 5000 K and 5200 K and nothing at 5100 K or 4900 K,
    When the holes are filled,
    Then 5100 K is the mean of its neighbours and recorded; 4900 K, with
      only one neighbour, stays absent.
    """
    bcs = {
        (5000.0, 4.5, "X"): np.array([[0.1, 1.0]]),
        (5200.0, 4.5, "X"): np.array([[0.3, 2.0]]),
    }

    records = gen.fill_bc_holes_along_teff(
        bcs, [4900.0, 5000.0, 5100.0, 5200.0], [4.5], ["X"]
    )

    np.testing.assert_allclose(bcs[(5100.0, 4.5, "X")], [[0.2, 1.5]])
    assert (4900.0, 4.5, "X") not in bcs
    assert records == [
        {"teff": 5100.0, "logg": 4.5, "axis": "teff", "from": [5000.0, 5200.0]}
    ]


def test_the_intermediate_keeps_the_band_fluxes_and_fbol(tmp_path):
    """
    Given one node streamed for the BCs and one (a filled hole) not,
    When the intermediate is saved and read back,
    Then the first node's band fluxes and bolometric flux come back and
      the second carries none -- and an intermediate written before the BCs
      existed (no band_flux column) reads as having none at all, so a run
      re-streams those nodes rather than failing or inventing a flux.
    """
    store = {
        (5800.0, 4.5, "F"): np.ones((2, 1, 3), np.float32),
        (5900.0, 4.5, "F"): np.ones((2, 1, 3), np.float32),
    }
    mus = {(5800.0, 4.5): np.linspace(0.1, 1, 3)}
    mus[(5900.0, 4.5)] = mus[(5800.0, 4.5)]
    fluxes = {(5800.0, 4.5, "F"): np.array([[3.0], [2.0]])}
    fbols = {(5800.0, 4.5): {"fbol": 6.4e10, "tail_fraction": 0.01}}

    gen.save_intensities(
        tmp_path, 0.0, store, mus, [0.0, 1.0], {"F": ["X"]}, fluxes, fbols
    )
    back_fluxes, back_fbols = gen.load_band_fluxes(tmp_path, 0.0)

    np.testing.assert_array_equal(
        back_fluxes[(5800.0, 4.5, "F")], [[3.0], [2.0]]
    )
    assert back_fbols == {(5800.0, 4.5): fbols[(5800.0, 4.5)]}
    assert (5900.0, 4.5, "F") not in back_fluxes

    # An LD-era intermediate.
    gen.save_intensities(tmp_path, 0.5, store, mus, [0.0, 1.0], {"F": ["X"]})
    path = gen.intensities_path(tmp_path, 0.5)
    df = pd.read_parquet(path)
    old = df.drop(columns=["band_flux", "fbol", "fbol_tail_fraction"])
    old.attrs = df.attrs
    old.to_parquet(path, index=False)
    assert gen.load_band_fluxes(tmp_path, 0.5) == ({}, {})


def test_an_unknown_product_raises():
    with pytest.raises(ValueError, match="products"):
        gen.generate_ld_tables(products=("sed",))


def test_the_bc_and_ld_grids_are_one_grid():
    """One stream of the archive makes both table sets, so the BC yaml's
    axes must be the LD yaml's (the generator reads the LD one)."""
    ld = yaml.safe_load(gen.GRID_YAML.read_text())["grid"]
    bc = yaml.safe_load((_PHOENIX_BCS / "PHOENIX.grid.yaml").read_text())
    assert bc["model"] == "PHOENIX"
    assert bc["grid"] == ld
    assert "2.6 um" in bc["coverage"]


def test_the_magsys_record_is_nextgens_minus_the_uncovered_filters():
    """
    Given the tracked PHOENIX and NextGen magnitude-system records,
    Then PHOENIX has every NextGen column but WISE, IRAC and Johnson M (the
      filters past 2.6 um), and each shared row is identical: the offset
      depends on the filter profile, weighting and zeropoint only.
    """
    phx = read_magsys_table(DEFAULT_MODEL_ROOT, "PHOENIX")
    ng = read_magsys_table(DEFAULT_MODEL_ROOT, "NextGen")

    assert set(ng.index) - set(phx.index) == _UNCOVERED
    assert set(phx.index) <= set(ng.index)
    pd.testing.assert_frame_equal(phx, ng.loc[phx.index])


# --------------------------------------------------------------------------
# 2. The Zenodo record: pending now, a normal record once pinned
# --------------------------------------------------------------------------


def test_the_record_is_pending_and_nothing_prefetches_it():
    assert bc_tables.RECORD_NAME in zenodo_assets.PENDING
    assert bc_tables.RECORD_NAME not in zenodo_assets.RECORDS
    assert bc_tables.RECORD_NAME in zenodo_assets.ON_DEMAND
    assert bc_tables.RECORD_NAME not in zenodo_assets.OWNERS
    with pytest.raises(zenodo_assets.RecordPendingError, match="not pub"):
        zenodo_assets.record(bc_tables.RECORD_NAME)


def test_no_record_is_both_pending_and_published():
    """Pinning a pending record (zenodo_publish.py pin --name) must be
    followed by deleting its PENDING line and naming its owner."""
    assert not set(zenodo_assets.PENDING) & set(zenodo_assets.RECORDS)


def test_reading_the_default_roots_phoenix_tables_while_pending_raises():
    """
    Given the record is pending,
    When a reader asks for a PHOENIX table under the DEFAULT root,
    Then it raises RuntimeError -- never FileNotFoundError, which the SED
      reads as "this model has no tables" and carries on -- naming both
      ways out: NextGen, or a model_root: holding locally generated tables.
    """
    with pytest.raises(RuntimeError) as err:
        bc_grid.ensure_bc_tables(DEFAULT_MODEL_ROOT, "PHOENIX", ["TESS"])
    msg = str(err.value)
    assert "not published yet" in msg
    assert "model: NextGen" in msg and "model_root:" in msg


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_record(tmp_path, monkeypatch):
    """The record as it will be once published: one table, pinned to a
    local file, whose 'download' is a copy (no network)."""
    src = tmp_path / "zenodo" / "TESS.bc.parquet"
    src.parent.mkdir()
    src.write_bytes(b"pretend parquet bytes")
    rec = ZenodoRecord(
        record_id=1,
        concept_record_id=2,
        title="PHOENIX BC tables (fake)",
        creators=("Test",),
        files=MappingProxyType(
            {
                "TESS.bc.parquet": ZenodoFile(
                    size=src.stat().st_size, md5=_md5(src)
                ),
                "MANIFEST.txt": ZenodoFile(size=1, md5="0" * 32),
            }
        ),
        citation_key=None,
        superseded=MappingProxyType({"TESS.bc.parquet": ("f" * 32,)}),
    )
    monkeypatch.setattr(
        zenodo_assets,
        "RECORDS",
        MappingProxyType({**zenodo_assets.RECORDS, "phoenix_bc_tables": rec}),
    )
    monkeypatch.setattr(zenodo_assets, "PENDING", MappingProxyType({}))
    fetched = []

    def fake_fetch(assets, dest_dir):
        for name, meta in assets.items():
            fetched.append((name, meta["url"]))
            shutil.copy(src.parent / name, dest_dir / name)

    monkeypatch.setattr(bc_tables.zenodo, "fetch_assets", fake_fetch)
    monkeypatch.setattr(bc_tables, "_verified", set())
    return SimpleNamespace(src=src, fetched=fetched)


def test_a_published_table_is_fetched_and_verified(tmp_path, fake_record):
    """
    Given the record is published (one table plus its MANIFEST),
    When the tables are ensured into an empty directory,
    Then only the table is fetched, from the record's URL, and a second
      call fetches nothing.
    """
    dest = tmp_path / "dest"
    dest.mkdir()

    assert bc_tables.published_tables() == ["TESS.bc.parquet"]
    paths = bc_tables.ensure_tables(None, dest_dir=dest)
    bc_tables.ensure_tables(None, dest_dir=dest)

    assert paths == [dest / "TESS.bc.parquet"]
    assert fake_record.fetched == [
        (
            "TESS.bc.parquet",
            "https://zenodo.org/records/1/files/TESS.bc.parquet",
        )
    ]


def test_a_locally_changed_table_raises_and_is_kept(tmp_path, fake_record):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "TESS.bc.parquet").write_bytes(b"regenerated locally")

    with pytest.raises(RuntimeError, match="NOT overwritten"):
        bc_tables.ensure_tables(["TESS.bc.parquet"], dest_dir=dest)
    assert (dest / "TESS.bc.parquet").read_bytes() == b"regenerated locally"
    # the generator's own merges are allowed
    bc_tables.ensure_tables(
        ["TESS.bc.parquet"], dest_dir=dest, allow_local_changes=True
    )


def test_the_default_root_dispatches_to_the_phoenix_module(
    monkeypatch, fake_record
):
    """bc_grid fetches a PHOENIX facility through models/PHOENIX/
    bc_tables.py, and only the published facilities."""
    asked = []
    monkeypatch.setattr(
        bc_tables, "ensure_tables", lambda names, **kw: asked.append(names)
    )

    bc_grid.ensure_bc_tables(DEFAULT_MODEL_ROOT, "PHOENIX", ["TESS", "WISE"])

    assert asked == [["TESS.bc.parquet"]]


# --------------------------------------------------------------------------
# 3. The SED: selecting the model, and its filter coverage
# --------------------------------------------------------------------------

# A synthetic PHOENIX tree: BC linear in every axis, so the 4-D linear
# interpolation reproduces it exactly at any point.
_AXES = {
    "teff": [4000.0, 5000.0, 6000.0, 7000.0],
    "logg": [3.5, 4.0, 4.5, 5.0],
    "feh": [-0.5, 0.0, 0.5],
    "Av": [0.0, 0.5, 1.0, 2.0],
}
_COEF = {
    "Bessell_V": (-0.5, 0.1, 0.02, -0.03, -1.0),
    "Cousins_I": (0.3, 0.05, 0.01, -0.02, -0.5),
}


def _bc(col, teff, logg, feh, av):
    a, b, c, d, e = _COEF[col]
    return a + b * teff / 1000.0 + c * logg + d * feh + e * av


def _phoenix_tree(root):
    bcs = root / "PHOENIX" / "BCs"
    bcs.mkdir(parents=True)
    for name in ("PHOENIX.grid.yaml", "PHOENIX.magsys.csv"):
        shutil.copy(_PHOENIX_BCS / name, bcs / name)
    record = read_magsys_table(DEFAULT_MODEL_ROOT, "PHOENIX")
    grid = np.array(np.meshgrid(*_AXES.values(), indexing="ij"))
    t, g, f, a = (x.ravel() for x in grid)
    df = pd.DataFrame(
        {"teff": t, "logg": g, "feh": f, "alpha": 0.0, "Av": a, "Rv": 3.1}
    )
    for col in _COEF:
        df[col] = _bc(col, t, g, f, a)
    meta = {
        col: {
            "svo_id": record.loc[col, "svo_id"],
            "zeropoint_Fl_Vega": float(record.loc[col, "zeropoint_Fl_Vega"]),
            "flux_weighting": record.loc[col, "flux_weighting"],
        }
        for col in _COEF
    }
    write_bc_table(
        df[BC_PARAM_COLS + list(_COEF)],
        bcs / "Generic.bc.parquet",
        meta,
        table_meta={"model": "PHOENIX", "mag_system": "Vega"},
    )
    return root


def _sed(tmp_path, rows, model="PHOENIX", model_root=None):
    from exozippy.components.sed.sed import SED
    from exozippy.config import ConfigManager

    path = tmp_path / "phx.sed"
    path.write_text(yaml.safe_dump({"model": model, "filters": rows}))
    cm = ConfigManager(
        {
            "star.teffsed": {"initval": 5500.0},
            "star.feh": {"initval": 0.0},
            "star.av": {"initval": 0.2},
        }
    )
    cfg = {"file": str(path)}
    if model_root is not None:
        cfg["model_root"] = str(model_root)
    return SED(cfg, cm)


def _load(sed):
    star = SimpleNamespace(names=["A"], n_elements=1)
    sed.load_data(SimpleNamespace(star=star, active_components={}))
    return sed


def _row(name, mag):
    return {"name": name, "mag": mag, "err": 0.02, "magsys": "Vega"}


def test_model_phoenix_reads_the_phoenix_tables(tmp_path):
    """
    Given an SED file saying `model: PHOENIX` and a PHOENIX tree,
    When the SED loads,
    Then its grid is the PHOENIX tables' -- the BC at a point between the
      nodes is the synthetic tables' value -- and its grid bounds are named
      after the PHOENIX grid.
    """
    root = _phoenix_tree(tmp_path / "models")
    sed = _load(
        _sed(
            tmp_path,
            [_row("Generic/Bessell.V", 10.0), _row("Generic/Cousins.I", 9.3)],
            model_root=root,
        )
    )

    point = (5321.0, 4.21, 0.13, 0.77)
    assert sed.sedmodel == "PHOENIX"
    assert sed.bc_np("Generic/Bessell.V", *point) == pytest.approx(
        _bc("Bessell_V", *point), abs=1e-12
    )
    assert sed.bc_np("Generic/Cousins.I", *point) == pytest.approx(
        _bc("Cousins_I", *point), abs=1e-12
    )
    bounds = sed.grid_bound_paths()["star.teffsed"]
    assert (bounds["lower"], bounds["upper"]) == (4000.0, 7000.0)
    assert bounds["source"] == "PHOENIX bolometric-correction grid"


def test_a_filter_past_2p6_um_is_refused_naming_the_coverage(tmp_path):
    """
    Given `model: PHOENIX` and a WISE W1 row,
    When the SED loads,
    Then the grid build refuses the row (the missing-column policy for a
      .sed row), saying why PHOENIX has no WISE column and how to fit it.
    """
    root = _phoenix_tree(tmp_path / "models")
    sed = _sed(
        tmp_path,
        [_row("Generic/Bessell.V", 10.0), _row("WISE/WISE.W1", 8.1)],
        model_root=root,
    )

    with pytest.raises(NotImplementedError) as err:
        _load(sed)
    msg = str(err.value)
    assert "WISE" in msg and "2.6 um" in msg
    assert "model: NextGen" in msg


@pytest.mark.parametrize("spelling", ["phoenix", "Phoenix", "nextgen"])
def test_a_case_variant_of_a_model_name_raises(tmp_path, spelling):
    """Model names are case-sensitive: a variant raises naming the right
    spelling, it is never folded."""
    right = "PHOENIX" if spelling.lower() == "phoenix" else "NextGen"
    with pytest.raises(ValueError, match=f"did you mean '{right}'"):
        _sed(tmp_path, [_row("Generic/Bessell.V", 10.0)], model=spelling)


def test_an_unpublished_model_under_the_default_root_raises(tmp_path):
    with pytest.raises(ValueError, match="no published"):
        _sed(tmp_path, [_row("Generic/Bessell.V", 10.0)], model="Kurucz")


def test_phoenix_under_the_default_root_says_it_is_not_published(tmp_path):
    """
    Given `model: PHOENIX`, no model_root:, and the record still pending,
    When the SED is constructed,
    Then it raises naming the way out instead of fitting without grid
      bounds.
    """
    with pytest.raises(RuntimeError, match="not published yet"):
        _sed(tmp_path, [_row("Generic/Bessell.V", 10.0)])


def test_an_absent_model_key_is_nextgen(tmp_path):
    from exozippy.components.sed.sed import SED

    sed = SED.__new__(SED)
    sed.sedfile = "x.sed"
    sed.model_root = DEFAULT_MODEL_ROOT
    assert sed._resolve_sed_model(None) == "NextGen"


def test_a_phoenix_fit_says_which_atmospheres_in_the_draft():
    from exozippy.components.sed.sed import SED
    from exozippy.outputs.prose import get_collector

    sed = SED.__new__(SED)
    system = SimpleNamespace()
    for model, expected in (("NextGen", False), ("PHOENIX", True)):
        sed.sedmodel = model
        sed._add_model_prose(system)
        assert ("sed.bc_model" in get_collector(system)) is expected
    text = " ".join(s.text for s in get_collector(system).sentences())
    assert r"\citep{Husser:2013}" in text

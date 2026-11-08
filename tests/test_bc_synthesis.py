"""Only the band filters whose SED flux is READ need a BC column, and a
read one the published tables lack is synthesized from the full-resolution
spectra into a per-user cache (sed.md, "Which band filters need a BC
column"; models/NextGen/bc_synthesis.py).

No test here touches the network or the 3.3 GB spectra record: the bundle
is a few fake processed-spectra parquets on the real R = 20000 wavelength
grid, the filter is a profile the package ships (2MASS J), and the fetch
is replaced at its seam (``bc_synthesis.fetch_spectra``).
"""

import logging
import threading
import time

import numpy as np
import pandas as pd
import pytest

from exozippy.components.sed import bc_grid
from exozippy.components.sed.bc_grid import (
    BC_PARAM_COLS,
    bc_table_path,
    build_bc_grid,
    locate_bc_column,
    read_bc_meta,
    read_bc_table,
    write_bc_table,
)
from exozippy.components.sed.magsys import (
    MAGSYS_COLUMNS,
    magsys_table_path,
    read_magsys_table,
)
from exozippy.components.sed.sed import SED
from exozippy.models.NextGen import bc_synthesis
from exozippy.models.NextGen import generate_NextGen_BC_Tables as gen
from exozippy.models.NextGen.bolometric_correction import BolometricCorrection
from exozippy.models.NextGen.nextgen_spectra import WAVELENGTH_PTS

_H = 6.62607015e-27  # erg s
_C = 2.99792458e10  # cm/s
_KB = 1.380649e-16  # erg/K

TINY_GRID = {
    "teff": [5000.0, 5100.0, 5200.0],
    "logg": [4.5, 5.0],
    "feh": [-0.5, 0.0],
    "av": [0.0, 0.5],
}
# A node the fake bundle lacks (an interior Teff hole), which step 2 pads.
HOLE = (5100.0, 5.0, 0.0)


def _blackbody_flambda(wave_ang, teff):
    lam = wave_ang * 1e-8
    b = 2 * _H * _C**2 / lam**5 / np.expm1(_H * _C / (lam * _KB * teff))
    return np.pi * b * 1e-8


def _write_bundle(path):
    """Fake step-1 parquets (the published bundle's format) on TINY_GRID,
    float32 flux like the published ones, missing HOLE."""
    path.mkdir(parents=True, exist_ok=True)
    for feh in TINY_GRID["feh"]:
        rows = [
            (t, g)
            for g in TINY_GRID["logg"]
            for t in TINY_GRID["teff"]
            if (t, g, feh) != HOLE
        ]
        df = pd.DataFrame(
            {
                "teff": [t for t, _ in rows],
                "logg": [g for _, g in rows],
                "feh": feh,
                "alpha": 0.0,
                "flux": [
                    _blackbody_flambda(WAVELENGTH_PTS, t).astype(np.float32)
                    for t, _ in rows
                ],
            }
        )
        df.attrs["meta"] = {"n_wavelength": len(WAVELENGTH_PTS)}
        df.to_parquet(path / gen._processed_filename(feh))


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """The machine cache in tmp_path, the target grid tiny, and the fetch
    replaced by a fake bundle.  Returns (synthesized root, bundle dir)."""
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(bc_synthesis, "target_grid", lambda: dict(TINY_GRID))
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    fetched = []

    def fake_fetch(fehs):
        fetched.append(list(fehs))
        return bundle

    monkeypatch.setattr(bc_synthesis, "fetch_spectra", fake_fetch)
    root = bc_grid.synthesized_model_root(
        bc_grid.DEFAULT_MODEL_ROOT, "NextGen"
    )
    assert root == tmp_path / "cache" / "bc_synthesized"
    return root, bundle, fetched


# ---------------------------------------------------------------------------
# The synthesis itself: a real step-2 run on the fake bundle
# ---------------------------------------------------------------------------


def test_a_synthesized_column_round_trips_through_the_cache(cache_env):
    """
    Given a fake processed bundle and the shipped 2MASS J profile,
    When the column is synthesized,
    Then it lands in the per-user cache (never the package tree) with the
      published pipeline's provenance, equals a direct BolometricCorrection,
      pads the node the bundle lacks, records its native system and
      m_AB - m_Vega beside it, counts as current, and a second request
      computes nothing.
    """
    root, bundle, fetched = cache_env

    # ACT
    path = bc_synthesis.synthesize_column("2MASS/2MASS.J", "Vega", "test")

    # ASSERT -- where and with what provenance
    assert path == bc_table_path(root, "NextGen", "2MASS")
    assert fetched == [sorted(TINY_GRID["feh"])]
    meta = read_bc_meta(path)
    fmeta = meta["filters"]["2MASS_J"]
    assert fmeta["generator"] == gen.GENERATOR
    assert fmeta["spectra"] == gen.SPECTRA_TAG
    assert meta["mag_system"] == "Vega"
    assert bc_synthesis.cached_column_is_current(path, "2MASS_J")

    # the values are step 2's, node for node
    df = read_bc_table(path)
    row = df[(df.teff == 5200.0) & (df.logg == 4.5) & (df.feh == -0.5)]
    direct = BolometricCorrection(
        ["2MASS/2MASS.J"],
        {"teff": 5200.0, "logg": 4.5, "feh": -0.5, "av": row["Av"].values},
        spectrum=(0.0, _blackbody_flambda(WAVELENGTH_PTS, 5200.0)),
    )
    np.testing.assert_allclose(
        row["2MASS_J"].values, direct.BC_by_av[:, 0], atol=1e-5
    )
    assert len(df) == np.prod([len(v) for v in TINY_GRID.values()])

    # the hole is padded from its Teff neighbours, and recorded
    pads = meta["padding"]
    assert [(p["teff"], p["logg"], p["feh"]) for p in pads] == [HOLE]
    assert pads[0]["from_teff"] == [5000.0, 5200.0]
    assert pads[0]["columns"] == ["2MASS_J"]

    # the magnitude-system record beside it
    record = read_magsys_table(root, "NextGen")
    assert record.loc["2MASS_J", "native_system"] == "Vega"
    assert record.loc["2MASS_J", "flux_weighting"] == fmeta["flux_weighting"]
    assert np.isfinite(record.loc["2MASS_J", "ab_minus_vega"])

    # nothing was left behind in the cache directory but the table, the
    # record and the lock files
    leftovers = [
        p.name for p in path.parent.iterdir() if not p.name.endswith(".lock")
    ]
    assert sorted(leftovers) == ["2MASS.bc.parquet", "NextGen.magsys.csv"]

    # a second request reads the cache and computes nothing
    calls = []
    real = bc_synthesis._generate
    bc_synthesis._generate = lambda *a, **k: calls.append(a) or real(*a, **k)
    try:
        assert (
            bc_synthesis.synthesize_column("2MASS/2MASS.J", "Vega", "again")
            == path
        )
    finally:
        bc_synthesis._generate = real
    assert calls == []


def test_a_stale_cached_column_is_not_current(cache_env):
    """
    Given a cached column written by another pipeline (no GENERATOR tag),
    When its currency is asked,
    Then it is not current, so it is recomputed rather than read.
    """
    root, _, _ = cache_env
    path = bc_table_path(root, "NextGen", "2MASS")
    _write_tiny_column(path, "2MASS_J", 1.0, generator="make_bc.py")
    assert not bc_synthesis.cached_column_is_current(path, "2MASS_J")


def test_two_builders_of_one_column_are_serialized(cache_env, monkeypatch):
    """
    Given two fits that both need the same missing column,
    When they synthesize it at the same time,
    Then the (facility, filter) lock lets one compute while the other waits,
      and the second finds the column current and computes nothing.
    """
    root, _, _ = cache_env
    calls = []

    def slow_generate(filter_sets, processed, work, unavailable, natives):
        calls.append(time.monotonic())
        time.sleep(1.0)
        _write_tiny_column(
            bc_table_path(work, "NextGen", "2MASS"),
            "2MASS_J",
            1.0,
            svo="2MASS/2MASS.J",
        )

    monkeypatch.setattr(bc_synthesis, "_generate", slow_generate)
    results, errors = [], []

    def build():
        try:
            results.append(
                bc_synthesis.synthesize_column("2MASS/2MASS.J", "Vega", "t")
            )
        except Exception as e:  # noqa: BLE001 -- reported by the assert
            errors.append(e)

    threads = [threading.Thread(target=build) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == []
    assert len(calls) == 1
    assert results == [bc_table_path(root, "NextGen", "2MASS")] * 2


def test_without_the_machine_cache_synthesis_raises_naming_the_switch(
    monkeypatch,
):
    """
    Given the machine cache switched off (EXOZIPPY_CACHE_DIR=""),
    When a column is synthesized,
    Then it raises naming EXOZIPPY_CACHE_DIR -- it never writes into the
      package's published tables.
    """
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", "")
    assert (
        bc_grid.synthesized_model_root(bc_grid.DEFAULT_MODEL_ROOT, "NextGen")
        is None
    )
    with pytest.raises(RuntimeError, match="EXOZIPPY_CACHE_DIR"):
        bc_synthesis.synthesize_column("2MASS/2MASS.J", "Vega", "t")


# ---------------------------------------------------------------------------
# The SED's choice: which band filters go in, and which are synthesized
# ---------------------------------------------------------------------------


class _StubConfigManager:
    def __init__(self, bands):
        self.system_config = {"band": bands}


def _sed_stub(bands, model_root, filters=()):
    sed = object.__new__(SED)
    sed.config_manager = _StubConfigManager(bands)
    sed.filters = list(filters)
    sed.sedmodel = "NextGen"
    sed.model_root = model_root
    return sed


def test_an_unread_band_without_a_column_is_left_out_with_info(
    tmp_path, monkeypatch, caplog
):
    """
    Given a band whose filter has no BC column (CFHT ships none) and whose
      SED flux nothing reads,
    When the band filters are collected,
    Then it is left out of the grid with one INFO line saying nothing reads
      it, and nothing is queued for synthesis.
    """
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(tmp_path / "cache"))
    sed = _sed_stub(
        [{"name": "g", "filter": "CFHT/MegaCam.g"}], bc_grid.DEFAULT_MODEL_ROOT
    )
    with caplog.at_level(logging.DEBUG, logger="exozippy"):
        extra = sed._collect_band_filters({})
    assert extra == []
    assert sed._to_synthesize == {}
    assert "no component reads its SED flux" in caplog.text


def test_a_read_band_without_a_column_needs_a_mag_system(
    tmp_path, monkeypatch
):
    """
    Given a band whose filter has no BC column and whose SED flux a
      component reads, with no `mag_system:`,
    When the band filters are collected,
    Then it raises naming the band and the key -- Vega is not guessed.
    """
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(tmp_path / "cache"))
    sed = _sed_stub(
        [{"name": "g_band", "filter": "CFHT/MegaCam.g"}],
        bc_grid.DEFAULT_MODEL_ROOT,
    )
    with pytest.raises(ValueError) as exc:
        sed._collect_band_filters({"g_band": ["mulensinstrument"]})
    msg = str(exc.value)
    assert "g_band" in msg and "mag_system" in msg
    assert "mulensinstrument" in msg


def test_mag_system_is_case_sensitive(tmp_path, monkeypatch):
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(tmp_path / "cache"))
    sed = _sed_stub(
        [{"name": "g", "filter": "CFHT/MegaCam.g", "mag_system": "ab"}],
        bc_grid.DEFAULT_MODEL_ROOT,
    )
    with pytest.raises(ValueError, match="Did you mean 'AB'"):
        sed._collect_band_filters({"g": ["transit"]})


def test_a_read_band_is_queued_for_synthesis_with_its_system(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(tmp_path / "cache"))
    sed = _sed_stub(
        [{"name": "g", "filter": "CFHT/MegaCam.g", "mag_system": "AB"}],
        bc_grid.DEFAULT_MODEL_ROOT,
    )
    extra = sed._collect_band_filters({"g": ["transit"]})
    assert extra == ["CFHT/MegaCam.g"]
    [(svo, native, where)] = sed._to_synthesize.values()
    assert (svo, native) == ("CFHT/MegaCam.g", "AB")
    assert "transit" in where


def test_every_component_is_asked_which_bands_it_reads():
    """
    Given components each declaring the bands whose SED flux they read,
    When the SED collects them,
    Then it gets each band with its readers -- the SED itself knows no
      component by name.
    """

    class _Reader:
        def __init__(self, prefix, bands):
            self.prefix, self._bands = prefix, bands

        def sed_flux_bands(self, system):
            return set(self._bands)

    class _System:
        active_components = {
            "a": _Reader("transit", ["V"]),
            "b": _Reader("mulensinstrument", ["V", "I"]),
            "c": _Reader("star", []),
        }

    sed = object.__new__(SED)
    got = sed.consumed_sed_bands(_System())
    assert {k: sorted(v) for k, v in got.items()} == {
        "V": ["mulensinstrument", "transit"],
        "I": ["mulensinstrument"],
    }


def test_mulensinstrument_reads_only_tied_light_curves():
    """
    Given three light curves -- one with a stated zeropoint mu and sigma,
      one with only mu, one with sed_constrains_blend --
    When asked which bands it reads,
    Then it names the tied and the blend-tied bands, not the mu-only one.
    """
    from exozippy.components.mulensing.mulensinstrument import (
        MulensInstrument,
    )

    class _CM:
        user_params = {
            "mulensinstrument.A.zeropoint": {"mu": 22.0, "sigma": 0.1},
            "mulensinstrument.B.zeropoint": {"mu": 22.0},
        }
        links = {}

    mi = object.__new__(MulensInstrument)
    mi.config = [
        {"name": "A", "band": "I"},
        {"name": "B", "band": "V"},
        {"name": "C", "band": "R", "sed_constrains_blend": True},
        {"name": "D"},
    ]
    mi.names = ["A", "B", "C", "D"]
    mi.n_elements = 4
    mi.config_manager = _CM()
    assert mi.sed_flux_bands(None) == {"I", "R"}


# ---------------------------------------------------------------------------
# The grid: published and synthesized columns, read seamlessly
# ---------------------------------------------------------------------------


def _tiny_nodes():
    return bc_grid.grid_nodes(TINY_GRID)


def _write_tiny_column(path, col, value, generator=gen.GENERATOR, svo=None):
    df = _tiny_nodes()
    df["alpha"] = 0.0
    df["Rv"] = 3.1
    df[col] = value + 0.001 * df["teff"]
    meta = {
        "svo_id": svo or f"X/{col}",
        "zeropoint_Fl_Vega": 1.0e-9,
        "flux_weighting": "energy",
        "spectra": gen.SPECTRA_TAG,
        "generator": generator,
    }
    write_bc_table(
        df[BC_PARAM_COLS + [col]],
        path,
        {col: meta},
        table_meta={"mag_system": "Vega", "model": "NextGen"},
        allow_new_nodes=True,
    )
    return meta


def _write_magsys(root, rows):
    path = magsys_table_path(root, "NextGen")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(MAGSYS_COLUMNS)).to_csv(path, index=False)


@pytest.fixture
def fake_published(tmp_path, monkeypatch):
    """A tiny DEFAULT model root whose FAKE facility publishes Cam.a only,
    with the machine cache in tmp_path."""
    root = tmp_path / "models"
    tables = root / "NextGen" / "BCs"
    tables.mkdir(parents=True)
    import yaml

    (tables / "NextGen.grid.yaml").write_text(
        yaml.safe_dump({"model": "NextGen", "grid": TINY_GRID})
    )
    meta = _write_tiny_column(
        bc_table_path(root, "NextGen", "FAKE"), "Cam_a", 1.0, svo="FAKE/Cam.a"
    )
    _write_magsys(
        root,
        [
            [
                "Cam_a",
                "FAKE/Cam.a",
                "Vega",
                meta["zeropoint_Fl_Vega"],
                meta["flux_weighting"],
                0.25,
            ]
        ],
    )
    monkeypatch.setattr(bc_grid, "DEFAULT_MODEL_ROOT", root)
    monkeypatch.setenv("EXOZIPPY_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(bc_synthesis, "target_grid", lambda: dict(TINY_GRID))
    return root


def test_a_read_band_is_synthesized_and_the_grid_then_holds_it(
    fake_published, monkeypatch
):
    """
    Given a facility that publishes Cam.a but not Cam.b, a band on each,
      and a component that reads band B's SED flux,
    When the SED collects its band filters, synthesizes what is missing
      (the synthesis mocked to write a cached column) and builds its grid,
    Then both filters are in the grid -- Cam.a from the published table,
      Cam.b from the per-user cache, read alike -- the SED knows Cam.b is
      synthesized, and its magnitude system comes from the cache's record.
    """
    root = fake_published
    synth_root = bc_grid.synthesized_model_root(root, "NextGen")
    synthesized = []

    def fake_synthesize(svo, native, where):
        synthesized.append((svo, native))
        meta = _write_tiny_column(
            bc_table_path(synth_root, "NextGen", "FAKE"),
            "Cam_b",
            2.0,
            svo=svo,
        )
        _write_magsys(
            synth_root,
            [
                [
                    "Cam_b",
                    svo,
                    native,
                    meta["zeropoint_Fl_Vega"],
                    meta["flux_weighting"],
                    0.5,
                ]
            ],
        )
        return bc_table_path(synth_root, "NextGen", "FAKE")

    monkeypatch.setattr(bc_synthesis, "synthesize_column", fake_synthesize)
    bands = [
        {"name": "A", "filter": "FAKE/Cam.a"},
        {"name": "B", "filter": "FAKE/Cam.b", "mag_system": "AB"},
    ]
    sed = _sed_stub(bands, root)

    # ACT
    sed.all_filters = sed._collect_band_filters({"B": ["background"]})
    sed._synthesize_missing_columns()
    grid = build_bc_grid(sed.all_filters, model="NextGen", model_root=root)

    # ASSERT
    assert synthesized == [("FAKE/Cam.b", "AB")]
    assert grid["filter_order"] == ["Cam_a", "Cam_b"]
    nodes = _tiny_nodes()
    t = grid["teff_pts"]
    np.testing.assert_allclose(
        grid["bc_values"][:, 0, 0, 0, 0], 1.0 + 0.001 * t
    )
    np.testing.assert_allclose(
        grid["bc_values"][:, 0, 0, 0, 1], 2.0 + 0.001 * t
    )
    assert len(nodes) == np.prod(grid["bc_values"].shape[:4])
    assert locate_bc_column(root, "NextGen", "FAKE", "Cam_b") == (
        bc_table_path(synth_root, "NextGen", "FAKE")
    )

    sed.mist_filters = grid["filter_order"]
    sed._bc_table_meta = {}
    sed._to_synthesize = {}
    sed._locate_grid_columns()
    assert sed.synthesized_filters == ["FAKE/Cam.b"]
    assert sed._column_magsys(1, None, "w") == ("AB", 0.5)
    assert sed._column_magsys(0, None, "w") == ("Vega", 0.25)
    sed._check_band_mag_systems()  # B states AB, and the record says AB


def test_a_mag_system_contradicting_the_record_raises(fake_published):
    """
    Given a band stating `mag_system: AB` on a published Vega column,
    When the grid's columns are checked,
    Then it raises naming the band and the key rather than ignoring it.
    """
    root = fake_published
    sed = _sed_stub(
        [{"name": "A", "filter": "FAKE/Cam.a", "mag_system": "AB"}], root
    )
    sed.all_filters = sed._collect_band_filters({})
    sed.mist_filters = ["Cam_a"]
    sed._locate_grid_columns()
    with pytest.raises(ValueError, match=r"'A' mag_system: AB contradicts"):
        sed._check_band_mag_systems()


# ---------------------------------------------------------------------------
# The record is on demand: no prewarm, no CI cache
# ---------------------------------------------------------------------------


def test_the_prewarm_skips_the_on_demand_record(monkeypatch):
    """
    Given a registry with one ordinary and one on-demand record,
    When fetch_every_record runs (the test prewarm and CI's cache step),
    Then only the ordinary record is fetched -- and the real spectra record
      is registered as on demand, owned, and kept out of CI's cache entry.
    """
    import sys
    import types

    from exozippy.utilities import zenodo, zenodo_assets

    called = []
    for name, tag in (("fake_owner_a", "a"), ("fake_owner_big", "big")):
        module = types.ModuleType(name)
        module.fetch_all = lambda tag=tag: called.append(tag)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        zenodo_assets, "RECORDS", {"rec_a": object(), "rec_big": object()}
    )
    monkeypatch.setattr(
        zenodo_assets,
        "OWNERS",
        {"rec_a": "fake_owner_a", "rec_big": "fake_owner_big"},
    )
    monkeypatch.setattr(zenodo_assets, "ON_DEMAND", frozenset({"rec_big"}))

    zenodo.fetch_every_record()

    assert called == ["a"]


def test_the_spectra_record_is_on_demand_and_out_of_the_ci_cache():
    import importlib.util
    from pathlib import Path

    from exozippy.utilities import zenodo_assets

    assert bc_synthesis.RECORD in zenodo_assets.ON_DEMAND
    assert zenodo_assets.OWNERS[bc_synthesis.RECORD] == bc_synthesis.__name__
    spec = importlib.util.spec_from_file_location(
        "zenodo_ci_cache",
        Path(__file__).resolve().parents[1] / "scripts" / "zenodo_ci_cache.py",
    )
    ci = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci)
    names = ci.current_entry_names()
    rec = zenodo_assets.record(bc_synthesis.RECORD)
    assert not any(pin.md5 in n for pin in rec.files.values() for n in names)
    assert any(n.endswith("2MASS.bc.parquet") for n in names)

"""The NextGen BCs are normalized by each spectrum's OWN bolometric flux
(review 8.9.13), not sigma Teff^4.

BT-NextGen spectra are fluxes at the model's outer radius, so a giant's
integrates to well under sigma Teff^4 (0.766 at 4000 K, log g 0.5).  A BC
is a band flux over the bolometric flux of the same spectrum; these tests
pin that rule in the generator, the one-time table conversion, the
per-node table reader and the provenance check that retires stale columns.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy import constants as const
from astropy import units as u

from exozippy.components.sed.bc_grid import read_bc_table, write_bc_table
from exozippy.constants import SIGMA_SB_CGS
from exozippy.models.NextGen import bc_tables
from exozippy.models.NextGen import generate_NextGen_BC_Tables as gen
from exozippy.models.NextGen.bolometric_correction import (
    L0,
    NORMALIZATION,
    BolometricCorrection,
    bolometric_flux,
)
from exozippy.models.NextGen.nextgen_spectra import WAVELENGTH_PTS

FILTERS = ["Generic/Bessell.B", "Generic/Bessell.V", "2MASS/2MASS.Ks"]
REPO = Path(__file__).resolve().parents[1]


def _bb(teff):
    h, c, k = 6.62607015e-27, 2.99792458e10, 1.380649e-16
    lam = np.asarray(WAVELENGTH_PTS, float) * 1e-8
    with np.errstate(over="ignore"):
        return (
            np.pi * 2 * h * c**2 / lam**5 / np.expm1(h * c / (lam * k * teff))
        ) * 1e-8


def _bc(teff, flux, av=(0.0, 3.0)):
    return BolometricCorrection(
        FILTERS,
        {"teff": teff, "logg": 2.0, "feh": 0.0, "av": np.asarray(av)},
        spectrum=(0.0, flux),
    )


def test_bolometric_flux_of_a_blackbody_is_sigma_t4():
    """The R = 20000 integral plus its Rayleigh-Jeans tail recovers
    sigma T^4 for a blackbody to 1e-4 (the tail is exact in the RJ limit;
    at 2600 K and 30 micron x = 0.18, so it is not quite)."""
    for t in (2600.0, 4800.0, 10000.0):
        got = bolometric_flux(WAVELENGTH_PTS, _bb(t))
        assert got / (SIGMA_SB_CGS * t**4) == pytest.approx(1.0, abs=1e-4)


def test_bc_is_invariant_to_the_spectrum_normalization():
    """
    Given a spectrum and the same spectrum diluted by 0.766 (a 4000 K,
    log g 0.5 outer-radius flux),
    When BCs are computed from each,
    Then they are IDENTICAL: the BC is a ratio of band flux to the same
      spectrum's F_bol.  Under the old sigma Teff^4 normalization the
      diluted spectrum's BCs were 2.5 log10(0.766) = -0.29 mag too faint.
    """
    f = _bb(4000.0)
    a, b = _bc(4000.0, f), _bc(4000.0, 0.766 * f)
    np.testing.assert_allclose(b.BC_by_av, a.BC_by_av, atol=1e-12, rtol=0)


def test_new_bc_is_old_bc_plus_one_constant_per_node():
    """BC_new - BC_old = 2.5 log10(sigma Teff^4 / F_bol) for EVERY filter
    and A_V: the one scalar per node scripts/renormalize_nextgen_bc_tables
    applies to the published tables."""
    t = 4500.0
    f = 0.93 * _bb(t)  # an outer-radius-like dilution
    bc = _bc(t, f)
    # the sigma Teff^4 formula the generator used before review 8.9.13
    k = L0 / (100 * u.pc.to(u.m) ** 2 * 4 * np.pi * const.sigma_sb.value)
    old = 2.5 * np.log10(
        k / t**4 * bc.flux_in_filter / bc.filter_zero_flux
    ).reshape(bc.BC_by_av.shape)
    delta = 2.5 * np.log10(SIGMA_SB_CGS * t**4 / bc.fbol)
    np.testing.assert_allclose(old + delta, bc.BC_by_av, atol=1e-10)
    assert delta == pytest.approx(-2.5 * np.log10(0.93), abs=1e-4)


def test_stale_normalization_is_not_reusable():
    """A column (published or a synthesized cache column) without the
    own-F_bol normalization in its provenance is recomputed, not reused."""
    meta = {"generator": gen.GENERATOR, "spectra": gen.SPECTRA_TAG}
    assert not gen._reusable(meta)
    assert gen._reusable({**meta, "normalization": NORMALIZATION})


def _fbol_table(path, teffs=(4000.0, 4100.0), loggs=(0.5, 1.0), fehs=(0.0,)):
    rows = []
    for t in teffs:
        for g in loggs:
            for z in fehs:
                r = 0.8 + 0.1 * g
                rows.append((t, g, z, 0.0, r * SIGMA_SB_CGS * t**4, r, "own"))
    df = pd.DataFrame(rows, columns=gen.FBOL_COLUMNS)
    df.to_parquet(path, index=False)
    return df


def test_read_fbol_grid_shapes_and_raises_on_a_missing_node(tmp_path):
    p = tmp_path / "NextGen.fbol.parquet"
    df = _fbol_table(p)
    axes = {"teff": [4000.0, 4100.0], "logg": [0.5, 1.0], "feh": [0.0]}
    got = bc_tables.read_fbol_grid(p, axes)
    assert got.shape == (2, 2, 1)
    assert got[1, 0, 0] == pytest.approx(np.log10(df.fbol[2]))
    with pytest.raises(KeyError, match="teff=4200"):
        bc_tables.read_fbol_grid(p, {**axes, "teff": [4000.0, 4200.0]})


def _renorm_module():
    spec = importlib.util.spec_from_file_location(
        "renorm", REPO / "scripts" / "renormalize_nextgen_bc_tables.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_renormalize_adds_the_node_constant_once(tmp_path):
    """
    Given a sigma Teff^4-normalized table and a per-node F_bol table,
    When the one-time conversion runs,
    Then every filter cell moves by -2.5 log10(F_bol / sigma Teff^4) of
      its node, the provenance says so, and a second run refuses.
    """
    fb = _fbol_table(tmp_path / "NextGen.fbol.parquet")
    rows = []
    for t, g, z in zip(fb.teff, fb.logg, fb.feh):
        for av in (0.0, 1.0):
            rows.append(
                {"teff": t, "logg": g, "feh": z, "alpha": 0.0, "Av": av,
                 "Rv": 3.1, "X_a": -0.5 - av, "X_b": 1.0 + av}
            )  # fmt: skip
    src = tmp_path / "src" / "X.bc.parquet"
    src.parent.mkdir()
    fm = {c: {"svo_id": f"X/X.{c}"} for c in ("X_a", "X_b")}
    write_bc_table(pd.DataFrame(rows), src, fm, table_meta={"facility": "X"})
    mod = _renorm_module()
    out = tmp_path / "out" / "X.bc.parquet"
    out.parent.mkdir()
    mod.renormalize_table(src, out, fb)
    old, new = read_bc_table(src), read_bc_table(out)
    r = dict(zip(zip(fb.teff, fb.logg), fb.fbol_over_sigmaT4))
    d = np.array(
        [-2.5 * np.log10(r[(t, g)]) for t, g in zip(old.teff, old.logg)]
    )
    for c in ("X_a", "X_b"):
        np.testing.assert_allclose(new[c] - old[c], d, atol=1e-12)
    meta = new.attrs["meta"]
    assert meta["normalization"] == NORMALIZATION
    assert meta["filters"]["X_a"]["normalization"] == NORMALIZATION
    with pytest.raises(ValueError, match="twice"):
        mod.renormalize_table(out, tmp_path / "again.bc.parquet", fb)


def test_unpinned_fbol_table_raises_naming_the_record(monkeypatch):
    """A pinned record version without the F_bol table predates the
    own-F_bol normalization; the runtime path refuses it rather than mix
    normalizations."""
    files = {
        k: v
        for k, v in bc_tables._BC_TABLE_FILES.items()
        if k != bc_tables.FBOL_TABLE
    }
    monkeypatch.setattr(bc_tables, "_BC_TABLE_FILES", files)
    with pytest.raises(RuntimeError, match="8.9.13"):
        bc_tables.ensure_fbol_table()

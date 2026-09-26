"""
Tests for the full-resolution NextGen BC pipeline (models/NextGen/):
  - nextgen_spectra.py            : raw filenames and the alpha fallback
  - bolometric_correction.py      : weighting, extinction units, Av arrays
  - generate_NextGen_BC_Tables.py : step 1 (raw -> processed parquet) and
                                    step 2 (processed -> BC table), end to
                                    end on blackbody "raw spectra" in
                                    tmp_path, so no external drive is needed
"""

import numpy as np
import pytest
import yaml

from exozippy.components.sed.bc_grid import build_bc_grid, read_bc_table
from exozippy.models.NextGen import generate_NextGen_BC_Tables as gen
from exozippy.models.NextGen.bolometric_correction import BolometricCorrection
from exozippy.models.NextGen.nextgen_spectra import (
    WAVELENGTH_PTS,
    find_spectrum_file,
    get_NextGen2009_filename,
)

_H = 6.62607015e-27  # erg s
_C = 2.99792458e10  # cm/s
_KB = 1.380649e-16  # erg/K


def _blackbody_flambda(wave_ang, teff):
    """Surface flux pi*B_lambda in erg/s/cm^2/A."""
    lam = wave_ang * 1e-8
    b = 2 * _H * _C**2 / lam**5 / np.expm1(_H * _C / (lam * _KB * teff))
    return np.pi * b * 1e-8


def _write_raw_spectrum(path, teff):
    """A raw-format (SVO ASCII) blackbody spectrum."""
    wave = np.geomspace(100.0, 1.0e6, 4000)
    header = "# fake BT-NextGen\n# column 1: WAVELENGTH\n# column 2: FLUX\n"
    flux = _blackbody_flambda(wave, teff)
    body = "\n".join(f"{w:.4f} {f:.6e}" for w, f in zip(wave, flux))
    path.write_text(header + body + "\n")


# ---------------------------------------------------------------------------
# Section 1 -- raw spectra
# ---------------------------------------------------------------------------


def test_get_nextgen2009_filename_matches_the_svo_naming():
    """
    Given the solar grid point teff=5800, logg=4.5, feh=0, alpha=0,
    When get_NextGen2009_filename is called,
    Then it returns the SVO file name of that spectrum.
    """
    # ACT
    name = get_NextGen2009_filename(
        {"teff": 5800.0, "logg": 4.5, "feh": 0.0, "alpha": 0.0}
    )

    # ASSERT
    assert name == "lte058-4.5-0.0a+0.0.BT-NextGen.7.dat.txt"


def test_find_spectrum_file_falls_back_to_the_next_alpha(tmp_path):
    """
    Given a grid point whose only spectrum on disk is alpha = +0.4,
    When find_spectrum_file is called,
    Then it returns alpha 0.4 and that file.
    """
    # ARRANGE
    pt = {"teff": 3000.0, "logg": 5.0, "feh": -1.0, "alpha": 0.4}
    target = tmp_path / get_NextGen2009_filename(pt)
    target.write_text("")

    # ACT
    alpha, path = find_spectrum_file(3000.0, 5.0, -1.0, tmp_path)

    # ASSERT
    assert alpha == 0.4
    assert path == target


# ---------------------------------------------------------------------------
# Section 2 -- BolometricCorrection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def solar_blackbody():
    return 0.0, _blackbody_flambda(WAVELENGTH_PTS, 5800.0)


def test_detector_weighting_follows_svo_detector_type(solar_blackbody):
    """
    Given 2MASS J (an SVO photon counter) and WISE W3 (an energy counter),
    When BolometricCorrection runs with the default weighting,
    Then J is photon-weighted and W3 energy-weighted, each identical to
    forcing that weighting explicitly.
    """
    # ARRANGE
    filters = ["2MASS/2MASS.J", "WISE/WISE.W3"]
    star = {"teff": 5800.0, "logg": 4.5, "feh": 0.0, "av": 0.0}

    # ACT
    det = BolometricCorrection(filters, star, spectrum=solar_blackbody)
    pho = BolometricCorrection(
        filters, star, spectrum=solar_blackbody, weighting="photon"
    )
    ene = BolometricCorrection(
        filters, star, spectrum=solar_blackbody, weighting="energy"
    )

    # ASSERT
    assert det.filter_weightings == ["photon", "energy"]
    assert det.BC[0] == pytest.approx(pho.BC[0], abs=1e-12)
    assert det.BC[1] == pytest.approx(ene.BC[1], abs=1e-12)
    # and the choice matters: W3 is broad enough to move by > 0.1 mag
    assert abs(pho.BC[1] - ene.BC[1]) > 0.1


def test_extinction_slope_uses_the_law_in_microns(solar_blackbody):
    """
    Given a solar blackbody through 2MASS Ks,
    When BCs are computed at Av = 0 and Av = 1,
    Then BC drops by A_Ks/Av ~ 0.125 (ext(2.16 um)/ext(0.55 um)) -- the
    notebook's Angstrom lookup extrapolated the law off its end and gave
    a slope several times too large.
    """
    # ARRANGE
    av = np.array([0.0, 1.0])
    star = {"teff": 5800.0, "logg": 4.5, "feh": 0.0, "av": av}

    # ACT
    bc = BolometricCorrection(
        ["2MASS/2MASS.Ks"], star, spectrum=solar_blackbody
    )
    slope = bc.BC_by_av[1, 0] - bc.BC_by_av[0, 0]

    # ASSERT
    assert -0.14 < slope < -0.11


def test_av_array_matches_scalar_av(solar_blackbody):
    """
    Given the same spectrum and filters,
    When BCs are computed once for an Av array and once per scalar Av,
    Then the rows of BC_by_av equal the scalar results.
    """
    # ARRANGE
    filters = ["2MASS/2MASS.J", "GAIA/GAIA2r.G", "TESS/TESS.Red"]
    av = np.array([0.0, 0.4, 6.0])

    # ACT
    vec = BolometricCorrection(
        filters,
        {"teff": 5800.0, "logg": 4.5, "feh": 0.0, "av": av},
        spectrum=solar_blackbody,
    )
    scalars = [
        BolometricCorrection(
            filters,
            {"teff": 5800.0, "logg": 4.5, "feh": 0.0, "av": a},
            spectrum=solar_blackbody,
        ).BC
        for a in av
    ]

    # ASSERT
    assert vec.BC_by_av.shape == (3, 3)
    np.testing.assert_allclose(vec.BC_by_av, np.array(scalars), atol=1e-12)


# ---------------------------------------------------------------------------
# Section 3 -- the pipeline, end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """
    Blackbody raw spectra on a 2 x 2 x 1 (teff, logg, feh) grid -- one
    node available only at alpha = +0.2 -- run through step 1 and step 2
    into a private model root.
    """
    tmp = tmp_path_factory.mktemp("pipeline")
    raw, processed, root = tmp / "raw", tmp / "processed", tmp / "models"
    raw.mkdir()
    grid = {
        "teff": [5000.0, 6000.0],
        "logg": [4.0, 4.5],
        "feh": [0.0],
        "av": [0.0, 1.0, 3.0],
    }
    (root / "NextGen" / "BCs").mkdir(parents=True)
    with open(root / "NextGen" / "BCs" / "NextGen.grid.yaml", "w") as f:
        yaml.safe_dump({"model": "NextGen", "grid": grid}, f)
    for teff in grid["teff"]:
        for logg in grid["logg"]:
            alpha = 0.2 if (teff, logg) == (6000.0, 4.0) else 0.0
            pt = {"teff": teff, "logg": logg, "feh": 0.0, "alpha": alpha}
            _write_raw_spectrum(raw / get_NextGen2009_filename(pt), teff)

    gen.process_raw_spectra_for_feh(
        0.0,
        grid["teff"],
        grid["logg"],
        raw_path=raw,
        processed_path=processed,
        n_workers=1,
    )
    filter_sets = {"2MASS": ["2MASS/2MASS.J"], "WISE": ["WISE/WISE.W3"]}
    gen.generate_bc_tables(
        filter_sets=filter_sets, processed_path=processed, model_root=root
    )
    return root, grid, processed


def test_pipeline_table_values_equal_a_direct_calculation(pipeline_run):
    """
    Given the pipeline's 2MASS table,
    When a node is recomputed directly from its step-1 spectrum,
    Then the table holds exactly that node's BCs at every Av, and the
    step-1 spectrum is the raw blackbody resampled (to 1e-4 relative --
    the raw file is a coarse 4000-point tabulation).
    """
    # ARRANGE
    root, grid, processed = pipeline_run
    df = read_bc_table(root / "NextGen" / "BCs" / "2MASS.bc.parquet")
    node = df[(df.teff == 5000.0) & (df.logg == 4.5)].sort_values("Av")
    spec = gen.load_processed_spectra(0.0, processed)
    spec = spec.set_index(["teff", "logg"])
    flux = np.asarray(spec.loc[(5000.0, 4.5), "flux"])

    # ACT
    direct = BolometricCorrection(
        ["2MASS/2MASS.J"],
        {"teff": 5000.0, "logg": 4.5, "feh": 0.0, "av": np.array(grid["av"])},
        spectrum=(0.0, flux),
    )

    # ASSERT
    np.testing.assert_allclose(
        node["2MASS_J"].values, direct.BC_by_av[:, 0], atol=1e-12
    )
    band = (WAVELENGTH_PTS > 1.0e4) & (WAVELENGTH_PTS < 1.5e4)
    np.testing.assert_allclose(
        flux[band], _blackbody_flambda(WAVELENGTH_PTS[band], 5000.0), rtol=1e-4
    )


def test_pipeline_records_alpha_fallback_and_weighting(pipeline_run):
    """
    Given one grid node whose only raw spectrum is alpha = +0.2,
    When the pipeline has run,
    Then that node's rows carry alpha 0.2, the others 0.0, and each
    column's metadata records the weighting its detector type implies.
    """
    # ARRANGE
    root, _, _ = pipeline_run

    # ACT
    df = read_bc_table(root / "NextGen" / "BCs" / "WISE.bc.parquet")
    meta = df.attrs["meta"]["filters"]["WISE_W3"]

    # ASSERT
    fallback = (df.teff == 6000.0) & (df.logg == 4.0)
    assert (df.loc[fallback, "alpha"] == 0.2).all()
    assert (df.loc[~fallback, "alpha"] == 0.0).all()
    assert meta["svo_id"] == "WISE/WISE.W3"
    assert meta["flux_weighting"] == "energy"


def test_pipeline_tables_load_through_build_bc_grid(pipeline_run):
    """
    Given the pipeline's 2MASS and WISE tables,
    When build_bc_grid assembles one filter from each,
    Then the grid spans the yaml axes and is fully populated.
    """
    # ARRANGE
    root, grid, _ = pipeline_run

    # ACT
    out = build_bc_grid(
        ["2MASS/2MASS.J", "WISE/WISE.W3"], model="NextGen", model_root=root
    )

    # ASSERT
    assert out["bc_values"].shape == (2, 2, 1, 3, 2)
    np.testing.assert_array_equal(out["av_pts"], grid["av"])
    assert not np.isnan(out["bc_values"]).any()

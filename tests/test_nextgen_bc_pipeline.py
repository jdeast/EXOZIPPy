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


# Magnitudes per unit optical depth, 2.5 / ln(10) = 1.0857362...: A_lam =
# MAG_PER_TAU * tau_lam exactly. The generator divides by the rounded 1.086
# instead, so its tau is A / 1.086 and every A it applies is short by
# 1 - MAG_PER_TAU / 1.086 = 2.43e-4 relative. See the xfail test below.
_MAG_PER_TAU = 2.5 / np.log(10.0)
_GENERATOR_TAU_DIVISOR = 1.086


def _independent_band_mean(filter_id, teff, av, power):
    """
    The band-averaged F_lambda of a Teff blackbody behind Av magnitudes of
    extinction, written from scratch for the exact checks below:

        <F> = int(F_lam 10^(-0.4 Av k(lam)/k(V)) S lam^power dlam)
              / int(S lam^power dlam)

    F_lam is the Planck surface flux evaluated analytically on the filter
    profile's own grid, k is ``models/extinction_law.ascii`` read with
    np.loadtxt and interpolated linearly in MICRONS, k(V) at 0.55 um,
    power = 0 is energy weighting and 1 photon weighting. Nothing from
    bolometric_correction.py is used, only the SVO filter profile.
    """
    from exozippy.components.sed.bc_grid import DEFAULT_MODEL_ROOT
    from exozippy.filters.filter import Filter

    wave, profile = Filter(filter_id).ProcessedFilterCurve
    law = np.loadtxt(DEFAULT_MODEL_ROOT / "extinction_law.ascii")
    k = np.interp(wave * 1e-4, law[:, 0], law[:, 1])
    k_v = np.interp(0.55, law[:, 0], law[:, 1])
    weight = profile * wave**power
    flux = _blackbody_flambda(wave, teff) * 10.0 ** (-0.4 * av * k / k_v)
    return np.trapezoid(flux * weight, wave) / np.trapezoid(weight, wave)


def _independent_extinction(filter_id, teff, av, power):
    """A_band = -2.5 log10(<F>(Av) / <F>(0)) from the integral above."""
    return -2.5 * np.log10(
        _independent_band_mean(filter_id, teff, av, power)
        / _independent_band_mean(filter_id, teff, 0.0, power)
    )


_EXTINCTION_CASES = [
    ("2MASS/2MASS.Ks", "photon"),
    ("2MASS/2MASS.J", "photon"),
    ("WISE/WISE.W3", "energy"),
]


@pytest.mark.parametrize("filter_id, weighting", _EXTINCTION_CASES)
def test_band_extinction_matches_an_independent_integral_up_to_the_rounded_1086(
    solar_blackbody, filter_id, weighting
):
    """
    Given a solar blackbody through 2MASS Ks / J (photon counters) and
    WISE W3 (energy counter),
    When BolometricCorrection computes BC at Av = 0, 1 and 6,
    Then BC(0) - BC(Av) equals an independent Planck x profile x weight x
    10^(-0.4 Av k/k_V) integral to 1e-12 mag -- once Av is scaled by
    2.5/ln(10)/1.086, the one known departure (the rounded optical-depth
    divisor; see the strict xfail below). This is the exact form of the
    margin test above: microns vs Angstrom, the wrong k_V, the wrong
    weighting power or the wrong law file all miss by >> 1e-12. When the
    1.086 is fixed, drop the Av scaling here.
    """
    # ARRANGE
    av = np.array([0.0, 1.0, 6.0])
    star = {"teff": 5800.0, "logg": 4.5, "feh": 0.0, "av": av}
    power = {"energy": 0, "photon": 1}[weighting]
    av_applied = av * _MAG_PER_TAU / _GENERATOR_TAU_DIVISOR

    # ACT
    bc = BolometricCorrection([filter_id], star, spectrum=solar_blackbody)
    code = bc.BC_by_av[0, 0] - bc.BC_by_av[:, 0]
    ref = np.array(
        [
            _independent_extinction(filter_id, 5800.0, a, power)
            for a in av_applied
        ]
    )

    # ASSERT
    assert bc.filter_weightings == [weighting]
    np.testing.assert_allclose(code, ref, rtol=0, atol=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BC generator rounds 2.5/ln(10) to 1.086 in tau = A k/k_V / 1.086 "
        "(bolometric_correction.py and make_bc.py): every applied extinction "
        "is short by 2.43e-4 relative -- measured A_Ks(Av=1) 3.1e-5 mag, "
        "A_J(Av=6) 4.4e-4 mag, A_W3(Av=6) 6.5e-5 mag below the exact law."
    ),
)
@pytest.mark.parametrize("filter_id, weighting", _EXTINCTION_CASES)
def test_band_extinction_matches_the_exact_magnitude_law(
    solar_blackbody, filter_id, weighting
):
    """
    Given a solar blackbody through 2MASS Ks / J and WISE W3,
    When BolometricCorrection computes BC at Av = 0, 1 and 6,
    Then BC(0) - BC(Av) equals the independent integral with the exact
    magnitude law 10^(-0.4 A) (A = 2.5 log10(e) tau) to 1e-12 mag.
    """
    # ARRANGE
    av = np.array([0.0, 1.0, 6.0])
    star = {"teff": 5800.0, "logg": 4.5, "feh": 0.0, "av": av}
    power = {"energy": 0, "photon": 1}[weighting]

    # ACT
    bc = BolometricCorrection([filter_id], star, spectrum=solar_blackbody)
    code = bc.BC_by_av[0, 0] - bc.BC_by_av[:, 0]
    ref = np.array(
        [_independent_extinction(filter_id, 5800.0, a, power) for a in av]
    )

    # ASSERT
    np.testing.assert_allclose(code, ref, rtol=0, atol=1e-12)


def test_w3_photon_minus_energy_bc_matches_an_independent_integral(
    solar_blackbody,
):
    """
    Given a solar blackbody through WISE W3 at Av = 0,
    When BCs are computed photon-weighted and energy-weighted,
    Then BC_photon - BC_energy = 2.5 log10(<F>_photon / <F>_energy) from
    the independent integral to 1e-12 mag (the Vega zeropoint and the
    bolometric term cancel in the difference). Exact form of the
    "> 0.1 mag" check in the detector-weighting test: a wrong power of
    lambda (lam^2, 1/lam) or a weight left out of the normalization
    misses by > 0.1 mag.
    """
    # ARRANGE
    filters = ["WISE/WISE.W3"]
    star = {"teff": 5800.0, "logg": 4.5, "feh": 0.0, "av": 0.0}

    # ACT
    pho = BolometricCorrection(
        filters, star, spectrum=solar_blackbody, weighting="photon"
    ).BC
    ene = BolometricCorrection(
        filters, star, spectrum=solar_blackbody, weighting="energy"
    ).BC
    ref = 2.5 * np.log10(
        _independent_band_mean("WISE/WISE.W3", 5800.0, 0.0, 1)
        / _independent_band_mean("WISE/WISE.W3", 5800.0, 0.0, 0)
    )

    # ASSERT
    assert float(pho - ene) == pytest.approx(ref, abs=1e-12)
    assert abs(ref) > 0.1  # the check has teeth: W3 is broad


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


# ---------------------------------------------------------------------------
# Section 4 -- incremental runs: nothing already on disk is recomputed
# ---------------------------------------------------------------------------


def _write_grid_yaml(root, grid):
    with open(root / "NextGen" / "BCs" / "NextGen.grid.yaml", "w") as f:
        yaml.safe_dump({"model": "NextGen", "grid": grid}, f)


def _count_bc_calls(monkeypatch):
    """Record (filters, av) of every BolometricCorrection the generator
    constructs, still computing the real thing."""
    calls = []

    class Counting(BolometricCorrection):
        def __init__(self, filters, star_dict, **kwargs):
            av = list(np.atleast_1d(star_dict["av"]))
            calls.append((list(filters), av))
            super().__init__(filters, star_dict, **kwargs)

    monkeypatch.setattr(gen, "BolometricCorrection", Counting)
    return calls


@pytest.fixture
def small_run(tmp_path):
    """Blackbody raw spectra on a 2 x 2 x 1 grid, step 1 done, and step 2
    done for 2MASS J on Av = 0, 1 only."""
    raw, processed, root = tmp_path / "raw", tmp_path / "proc", tmp_path / "m"
    raw.mkdir()
    (root / "NextGen" / "BCs").mkdir(parents=True)
    grid = {
        "teff": [5000.0, 6000.0],
        "logg": [4.0, 4.5],
        "feh": [0.0],
        "av": [0.0, 1.0],
    }
    _write_grid_yaml(root, grid)
    for teff in (5000.0, 6000.0, 7000.0):
        for logg in grid["logg"]:
            pt = {"teff": teff, "logg": logg, "feh": 0.0, "alpha": 0.0}
            _write_raw_spectrum(raw / get_NextGen2009_filename(pt), teff)
    gen.process_missing_spectra(
        grid, raw_path=raw, processed_path=processed, n_workers=1
    )
    gen.generate_bc_tables(
        filter_sets={"2MASS": ["2MASS/2MASS.J"]},
        processed_path=processed,
        model_root=root,
    )
    return root, grid, raw, processed


def test_extending_av_computes_only_the_new_av_values(small_run, monkeypatch):
    """
    Given a 2MASS table computed on Av = 0, 1,
    When the grid yaml gains Av = 3 and step 2 is re-run,
    Then every BolometricCorrection is asked for Av = 3 alone, the Av = 0,
    1 cells are unchanged, and the Av = 3 cells equal a direct calculation.
    """
    # ARRANGE
    root, grid, _, processed = small_run
    path = root / "NextGen" / "BCs" / "2MASS.bc.parquet"
    before = read_bc_table(path)
    _write_grid_yaml(root, {**grid, "av": [0.0, 1.0, 3.0]})
    calls = _count_bc_calls(monkeypatch)

    # ACT
    gen.generate_bc_tables(
        filter_sets={"2MASS": ["2MASS/2MASS.J"]},
        processed_path=processed,
        model_root=root,
    )
    after = read_bc_table(path)

    # ASSERT
    assert len(calls) == 4  # one per (teff, logg) node
    assert all(av == [3.0] for _, av in calls)
    key = ["teff", "logg", "feh", "Av"]
    old = after.merge(before, on=key, suffixes=("", "_before"))
    assert len(old) == len(before)
    np.testing.assert_array_equal(old["2MASS_J"], old["2MASS_J_before"])
    spec = gen.load_processed_spectra(0.0, processed).set_index(
        ["teff", "logg"]
    )
    direct = BolometricCorrection(
        ["2MASS/2MASS.J"],
        {"teff": 6000.0, "logg": 4.0, "feh": 0.0, "av": 3.0},
        spectrum=(0.0, np.asarray(spec.loc[(6000.0, 4.0), "flux"])),
    )
    node = (after.Av == 3.0) & (after.teff == 6000.0) & (after.logg == 4.0)
    assert after.loc[node, "2MASS_J"].iloc[0] == pytest.approx(
        float(direct.BC_by_av.ravel()[0]), abs=1e-12
    )


def test_adding_a_filter_computes_only_that_filter(small_run, monkeypatch):
    """
    Given a 2MASS table holding J,
    When H is added to the facility's filter list and step 2 is re-run,
    Then BolometricCorrection is asked for H alone, and J is unchanged.
    """
    # ARRANGE
    root, _, _, processed = small_run
    path = root / "NextGen" / "BCs" / "2MASS.bc.parquet"
    before = read_bc_table(path).sort_values(["teff", "logg", "Av"])
    calls = _count_bc_calls(monkeypatch)

    # ACT
    gen.generate_bc_tables(
        filter_sets={"2MASS": ["2MASS/2MASS.J", "2MASS/2MASS.H"]},
        processed_path=processed,
        model_root=root,
    )
    after = read_bc_table(path).sort_values(["teff", "logg", "Av"])

    # ASSERT
    assert len(calls) == 4
    assert all(filters == ["2MASS/2MASS.H"] for filters, _ in calls)
    np.testing.assert_array_equal(
        after["2MASS_J"].values, before["2MASS_J"].values
    )
    assert after["2MASS_H"].notna().all()


def test_rerunning_a_complete_table_computes_nothing(small_run, monkeypatch):
    """
    Given a table already complete on the grid yaml,
    When step 2 is re-run unchanged,
    Then no BolometricCorrection is constructed and no spectra are loaded.
    """
    # ARRANGE
    root, _, _, processed = small_run
    calls = _count_bc_calls(monkeypatch)
    loads = []
    monkeypatch.setattr(
        gen, "load_processed_spectra", lambda *a, **k: loads.append(a)
    )

    # ACT
    out = gen.generate_bc_tables(
        filter_sets={"2MASS": ["2MASS/2MASS.J"]},
        processed_path=processed,
        model_root=root,
    )

    # ASSERT
    assert calls == [] and loads == [] and out == {}


def test_step1_processes_only_the_missing_teff_nodes(small_run, monkeypatch):
    """
    Given processed spectra for teff = 5000, 6000,
    When the grid gains teff = 7000 and step 1 is re-run,
    Then only the two new (teff, logg) nodes are read from the raw files,
    and the parquet then holds all six.
    """
    # ARRANGE
    _, grid, raw, processed = small_run
    read = []
    real = gen.process_spectrum
    monkeypatch.setattr(
        gen, "process_spectrum", lambda p: read.append(p) or real(p)
    )
    grid = {**grid, "teff": [5000.0, 6000.0, 7000.0]}

    # ACT
    done = gen.process_missing_spectra(
        grid, raw_path=raw, processed_path=processed, n_workers=1
    )
    df = gen.load_processed_spectra(0.0, processed)

    # ASSERT
    assert done == {0.0: 2}
    assert len(read) == 2 and all("lte070" in p.name for p in read)
    assert sorted(zip(df.teff, df.logg)) == [
        (t, g) for t in grid["teff"] for g in grid["logg"]
    ]

"""exozippy.utilities.exofast2exozippy: the EXOFASTv2 driver conventions it
must reproduce.

Each test converts a small synthetic driver end to end with ``convert`` and
reads the emitted YAML back:

  - the path idiom every EXOFASTv2 example driver uses (getenv, filepath,
    string concatenation, an ``if n_elements(x) eq 0`` default) and arrays
    of path globs;
  - EXOFASTv2's numbering: transit files in sorted-filename order, bands by
    SORTED unique name, an unsuffixed prior as instance 0;
  - circular=, mistsedfile=, the MIST tracks, the SED floors, and an
    absolute prefix kept out of the EXOFASTv2 fit's own directory;
  - /fitslope and /fitquad as a Taylor orbit (orbit type linear/quadratic),
    and a priors.final's section headers re-pointing priors written for a
    different set of data files.
"""

import pytest
import yaml

from exozippy.utilities import exofast2exozippy as e2z

_DRIVER = """\
pro fit, maxsteps=maxsteps, outpath=outpath
path = filepath('', root_dir=getenv('E2Z_TEST_ROOT'), subdir=['data'])
if n_elements(outpath) eq 0 then outpath = '/abs/elsewhere/'
exofastv2, nplanets=1, tranpath=[path+'n2020*.dat', path+'n2019*.dat'], $
   rvpath=path+'*.rv', priorfile='fit.priors', mistsedfile=path+'star.sed', $
   prefix=outpath+'run.', circular=[1], teffsedfloor=0.03, $
   fehsedfloor=0.08, maxsteps=maxsteps
end
"""

_PRIORS = """\
teff 5700 100
u1 0.31 0.1
u1_2 0.22 0.1
"""


@pytest.fixture
def converted(tmp_path, monkeypatch):
    """Convert _DRIVER: three transits (two bands on one FLWO night, listed
    by globs in reverse date order), one RV file and a MIST sed file, all
    under $E2Z_TEST_ROOT/data. Returns (config, params, warnings, infos)."""
    return _convert(tmp_path, monkeypatch, _DRIVER)


def _convert(
    tmp_path, monkeypatch, driver, priors=_PRIORS, priorfile_override=None
):
    """Convert ``driver`` against _DRIVER's data files and ``priors``."""
    data = tmp_path / "data"
    data.mkdir()
    for name in (
        "n20200101.TESS.TESS.dat",
        "n20190101.Sloanz.FLWO.dat",
        "n20190101.Sloang.FLWO.dat",
    ):
        (data / name).write_text("2458484.5 1.0 0.001\n")
    (data / "HIRES.rv").write_text("2458484.5 0.0 5.0\n")
    (data / "star.sed").write_text("J2M 10.0 0.02 0.02\n")
    (tmp_path / "fit.pro").write_text(driver)
    (tmp_path / "fit.priors").write_text(priors)
    monkeypatch.setenv("E2Z_TEST_ROOT", str(tmp_path))
    monkeypatch.setattr(e2z, "WARNINGS", [])
    monkeypatch.setattr(e2z, "INFOS", [])

    out = tmp_path / "out"
    out.mkdir()
    e2z.convert(
        tmp_path / "fit.pro", out, "t", priorfile_override=priorfile_override
    )

    config = yaml.safe_load((out / "t.yaml").read_text())
    params = yaml.safe_load((out / "t.params.yaml").read_text())
    return config, params, list(e2z.WARNINGS), list(e2z.INFOS)


def test_the_example_driver_path_idiom_finds_every_data_file(converted):
    """
    Given a driver that builds its paths with filepath(getenv(...)) and
    string concatenation, and lists its transits as an array of globs,
    When it is converted,
    Then every data file is found and nothing is left unevaluated.
    """
    # ARRANGE
    config, _, warnings, _ = converted

    # ACT
    files = [t["file"] for t in config["transit"]]

    # ASSERT
    assert len(files) == 3
    assert config["rvinstrument"][0]["file"] == "HIRES.rv"
    assert config["sed"]["file"] == "t.sed.yaml"
    assert not [w for w in warnings if "could not evaluate" in w]


def test_transits_are_numbered_in_sorted_filename_order(converted):
    """
    Given transit globs listed in reverse date order,
    When the driver is converted,
    Then the transits come out in sorted-filename order (exofastv2's
    numbering), and two bands on one night are told apart by band.
    """
    # ARRANGE
    config, _, _, _ = converted

    # ACT
    names = [t["name"] for t in config["transit"]]

    # ASSERT
    assert names == [
        "FLWO_UT20190101_Sloang",
        "FLWO_UT20190101_Sloanz",
        "TESS_UT20200101",
    ]


def test_limb_darkening_priors_index_bands_by_sorted_name(converted):
    """
    Given priors u1 (no suffix) and u1_2 over bands Sloang, Sloanz, TESS,
    When the driver is converted,
    Then u1 lands on band 0 and u1_2 on band 2 of the SORTED band list --
    an unsuffixed prior is instance 0, never "every band".
    """
    # ARRANGE
    _, params, _, _ = converted

    # ACT
    u1 = {k: v["mu"] for k, v in params.items() if k.endswith(".u1")}

    # ASSERT
    assert u1 == {"band.Sloang.u1": 0.31, "band.TESS.u1": 0.22}


def test_circular_pins_the_eccentricity_pair_at_zero(converted):
    """
    Given circular=[1],
    When the driver is converted,
    Then orbit.b.secosw and orbit.b.sesinw are both pinned at zero.
    """
    # ARRANGE
    _, params, _, _ = converted

    # ACT
    pins = {k: params[k] for k in ("orbit.b.secosw", "orbit.b.sesinw")}

    # ASSERT
    for fields in pins.values():
        assert fields == {"initval": 0.0, "sigma": 0.0}


def test_mist_driver_gets_an_evolutionary_model_and_sed_floors(converted):
    """
    Given a MIST fit (no /nomist) with mistsedfile=, teffsedfloor= and
    fehsedfloor=,
    When the driver is converted,
    Then the star is mist: True with an evolutionarymodel block, the
    teffsed floor is carried into the sed: block, and fehsedfloor is
    reported as having no EXOZIPPy equivalent.
    """
    # ARRANGE
    config, _, warnings, _ = converted

    # ACT
    evol = config["evolutionarymodel"]

    # ASSERT
    assert config["star"][0]["mist"] is True
    assert evol == [
        {"star": "A", "constrain": ["feh", "radius", "teff", "age"]}
    ]
    assert config["sed"]["teffsedfloor"] == 0.03
    assert "fbolsedfloor" not in config["sed"]
    assert any("fehsedfloor" in w for w in warnings)


def test_an_absolute_prefix_is_kept_out_of_the_exofast_fit_directory(
    converted,
):
    """
    Given a prefix built from the driver's absolute outpath default,
    When the driver is converted,
    Then the EXOZIPPy prefix keeps only the file stem under fitresults/.
    """
    # ARRANGE
    config, _, _, infos = converted

    # ACT
    prefix = config["prefix"]

    # ASSERT
    assert prefix == "fitresults/run"
    assert any("absolute prefix" in i for i in infos)


def test_an_unset_driver_keyword_is_never_concatenated_into_a_path():
    """
    Given a driver keyword with no default (a run-time pass-through),
    When an expression concatenates it with a string,
    Then the expression is reported unevaluated rather than turned into a
    path that starts with the keyword's name.
    """
    # ARRANGE
    variables = {"outpath": e2z._PassThrough("outpath")}
    before = len(e2z.WARNINGS)

    # ACT
    val = e2z._parse_idl_value("outpath+'run.'", variables)

    # ASSERT
    assert val == "outpath+'run.'"
    assert "could not evaluate" in e2z.WARNINGS[-1]
    del e2z.WARNINGS[before:]


def _driver_with(extra):
    """_DRIVER with ``extra`` keywords appended to the exofastv2 call."""
    tail = "fehsedfloor=0.08, maxsteps=maxsteps"
    assert _DRIVER.count(tail) == 1
    return _DRIVER.replace(tail, f"{tail}, {extra}")


def test_fitspline_array_is_per_transit_in_sorted_filename_order(
    tmp_path, monkeypatch
):
    """
    Given fitspline=[0,1,0] and splinespace=0.5 (mkss.pro: per-transit
    arrays, in the sorted-filename order the transits are numbered in),
    When the driver is converted,
    Then only the second file in sorted order (n20190101.Sloanz) gets
    fitspline: true and splinespace: 0.5, and fitspline is no longer
    reported as untranslated.
    """
    # ARRANGE / ACT
    config, _, warnings, _ = _convert(
        tmp_path,
        monkeypatch,
        _driver_with("fitspline=[0,1,0], splinespace=0.5"),
    )

    # ASSERT
    by_file = {t["file"]: t for t in config["transit"]}
    assert by_file["n20190101.Sloanz.FLWO.dat"]["fitspline"] is True
    assert by_file["n20190101.Sloanz.FLWO.dat"]["splinespace"] == 0.5
    for name in ("n20190101.Sloang.FLWO.dat", "n20200101.TESS.TESS.dat"):
        assert "fitspline" not in by_file[name]
        assert "splinespace" not in by_file[name]
    assert not any("fitspline" in w or "splinespace" in w for w in warnings)


def test_fitspline_flag_applies_to_every_transit(tmp_path, monkeypatch):
    """
    Given the scalar flag /fitspline and no splinespace,
    When the driver is converted,
    Then every transit gets fitspline: true and no splinespace key (the
    component's default is EXOFASTv2's 0.75 d).
    """
    config, _, _, _ = _convert(
        tmp_path, monkeypatch, _driver_with("/fitspline")
    )
    assert [t.get("fitspline") for t in config["transit"]] == [True] * 3
    assert not any("splinespace" in t for t in config["transit"])


def test_splinespace_without_fitspline_is_moot(tmp_path, monkeypatch):
    """
    Given splinespace=0.5 but no fitspline,
    When the driver is converted,
    Then no transit gets either key and the conversion says it was moot.
    """
    config, _, _, infos = _convert(
        tmp_path, monkeypatch, _driver_with("splinespace=0.5")
    )
    assert not any(
        "fitspline" in t or "splinespace" in t for t in config["transit"]
    )
    assert any("splinespace=0.5" in i and "moot" in i for i in infos)


@pytest.mark.parametrize(
    "extra, key",
    [
        ("fitspline=dofit", "fitspline"),
        ("fitspline=[0,dofit,0]", "fitspline"),
        ("/fitspline, splinespace=ss", "splinespace"),
    ],
)
def test_unevaluable_fitspline_value_raises(tmp_path, monkeypatch, extra, key):
    """
    Given a fitspline or splinespace the converter cannot evaluate (an
    undefined IDL variable, kept as a string),
    When the driver is converted,
    Then the conversion raises naming the keyword, instead of a truthy
    string silently turning the spline on (fitspline) or a bare float()
    ValueError (splinespace).
    """
    with pytest.raises(ValueError, match=rf"^{key}="):
        _convert(tmp_path, monkeypatch, _driver_with(extra))


def test_fitspline_length_mismatch_raises(tmp_path, monkeypatch):
    """
    Given fitspline=[0,1] for three transit files,
    When the driver is converted,
    Then the conversion raises naming the keyword and both counts.
    """
    with pytest.raises(ValueError, match=r"fitspline=.*2 entries for 3"):
        _convert(tmp_path, monkeypatch, _driver_with("fitspline=[0,1]"))


# ---------------------------------------------------------------------------
# /fitslope, /fitquad -> a Taylor orbit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag, otype", [("/fitslope", "linear"), ("/fitquad", "quadratic")]
)
def test_an_rv_trend_becomes_a_taylor_orbit(
    tmp_path, monkeypatch, flag, otype
):
    """
    Given /fitslope (or /fitquad), an RVEPOCH, and slope/quad priors,
    When the driver is converted,
    Then a `trend` orbit of type linear (quadratic) is added with the star as
      its primary and RVEPOCH as its epoch, the slope prior lands on
      orbit.trend.gammadot unchanged, and the quad prior on gammaddot at
      TWICE its value -- EXOFASTv2's QUAD is the coefficient of (t - t0)^2,
      gammaddot the second derivative.  Nothing is warned about.
    """
    priors = _PRIORS + "slope -1.5\nquad 0.002 0.001\n"
    config, params, warnings, _ = _convert(
        tmp_path,
        monkeypatch,
        _driver_with(f"{flag}, rvepoch=2458500.0"),
        priors=priors,
    )
    trend = config["orbit"][-1]
    assert trend == {
        "name": "trend",
        "type": otype,
        "primary": ["A"],
        "epoch": 2458500.0,
    }
    assert params["orbit.trend.gammadot"] == {"initval": -1.5}
    assert params["orbit.trend.gammaddot"] == {
        "initval": 0.004,
        "mu": 0.004,
        "sigma": 0.002,
    }
    assert not [w for w in warnings if "slope" in w or "quad" in w]


def test_an_rv_trend_without_rvs_is_moot(tmp_path, monkeypatch):
    """
    Given /fitslope but no RV files,
    Then no trend orbit is emitted, and the note says why.
    """
    driver = _driver_with("/fitslope").replace("rvpath=path+'*.rv', ", "")
    config, _, _, infos = _convert(tmp_path, monkeypatch, driver)
    assert [o["name"] for o in config["orbit"]] == ["b"]
    assert any("moot" in i and "fitslope" in i for i in infos)


# ---------------------------------------------------------------------------
# A priors.final's section headers outrank a stale _N index
# ---------------------------------------------------------------------------

# Written by a run on ONLY the TESS file and one FLWO band: its transit 0 is
# TESS and its band 0 is TESS, while this conversion numbers the TESS file 2
# and the TESS band 2 (Sloang, Sloanz, TESS sorted).
_STALE_PRIORS = """\
teff 5700 100
# TESS
u1_0 0.31 0.1
# TESS UT 2020-01-01 (TESS)
variance_0 1e-6
# FLWO UT 2019-01-02 (z')
f0_1 1.001
"""


def test_section_headers_re_point_stale_indices(tmp_path, monkeypatch):
    """
    Given a priors.final whose _N indices were written for another file
      set, with EXOFASTv2's own section headers,
    When it is converted,
    Then each prior lands on the instance its header NAMES (the TESS band,
      the TESS transit) -- not on whatever this conversion numbers _N --
      and a prior whose header names a dataset this conversion lacks is
      dropped with a warning instead of landing on an unrelated file.
    """
    _, params, warnings, infos = _convert(
        tmp_path, monkeypatch, _DRIVER, priors=_STALE_PRIORS
    )
    assert params["band.TESS.u1"]["mu"] == 0.31
    assert "band.Sloang.u1" not in params
    assert "transit.TESS_UT20200101.jitter_variance" in params
    assert not any(k.endswith(".baseline") for k in params)
    assert any("re-pointed" in i for i in infos)
    assert any("names no transit" in w for w in warnings)


def test_priorfile_override_replaces_the_drivers(tmp_path, monkeypatch):
    """
    Given --priorfile naming another file,
    When the driver is converted,
    Then that file's priors are used, and the note says which file won.
    """
    other = tmp_path / "run.priors.final"
    other.write_text("teff 6100 80\n")
    _, params, _, infos = _convert(
        tmp_path, monkeypatch, _DRIVER, priorfile_override=other
    )
    assert params["star.A.teff"]["mu"] == 6100
    assert any("--priorfile" in i for i in infos)


def test_multi_planet_conversions_point_each_planet_at_its_orbit(
    tmp_path, monkeypatch
):
    """
    Given nplanets=2,
    Then each planet block carries its own orbit_ndx -- without it every
      planet would read orbit 0's geometry.
    """
    driver = _DRIVER.replace("nplanets=1", "nplanets=2").replace(
        "circular=[1]", "circular=[1,0]"
    )
    config, _, _, _ = _convert(tmp_path, monkeypatch, driver)
    assert config["planet"] == [
        {"name": "b", "orbit_ndx": 0},
        {"name": "c", "orbit_ndx": 1},
    ]

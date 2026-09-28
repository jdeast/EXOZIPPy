"""scripts/exofast2exozippy.py: the EXOFASTv2 driver conventions it must
reproduce.

Each test converts a small synthetic driver end to end with ``convert`` and
reads the emitted YAML back:

  - the path idiom every EXOFASTv2 example driver uses (getenv, filepath,
    string concatenation, an ``if n_elements(x) eq 0`` default) and arrays
    of path globs;
  - EXOFASTv2's numbering: transit files in sorted-filename order, bands by
    SORTED unique name, an unsuffixed prior as instance 0;
  - circular=, mistsedfile=, the MIST tracks, the SED floors, and an
    absolute prefix kept out of the EXOFASTv2 fit's own directory.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NAME = "exofast2exozippy_under_test"
if _NAME in sys.modules:
    e2z = sys.modules[_NAME]
else:
    _SPEC = importlib.util.spec_from_file_location(
        _NAME, _REPO_ROOT / "scripts" / "exofast2exozippy.py"
    )
    e2z = importlib.util.module_from_spec(_SPEC)
    sys.modules[_NAME] = e2z
    _SPEC.loader.exec_module(e2z)


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
    (tmp_path / "fit.pro").write_text(_DRIVER)
    (tmp_path / "fit.priors").write_text(_PRIORS)
    monkeypatch.setenv("E2Z_TEST_ROOT", str(tmp_path))
    monkeypatch.setattr(e2z, "WARNINGS", [])
    monkeypatch.setattr(e2z, "INFOS", [])

    out = tmp_path / "out"
    out.mkdir()
    e2z.convert(tmp_path / "fit.pro", out, "t")

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

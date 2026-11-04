"""exozippy-getdata's epoch-astrometry downloads (utilities/getastrometry.py):
the Hipparcos 1997 IAD -> a mode-gaia file, and Gaia DR4 epoch astrometry
-> the RAW VOTable `data_format: gaia_epoch` reads.

No network except the one test marked `network`: the Hipparcos page is the
copy examples/HIP1349 ships, the Gaia table the DR4 pre-release
examples/gaia4 ships, and `http_get` is replaced where a request is made.
"""

import io
import urllib.error
import zipfile
from pathlib import Path

import numpy as np
import pytest

from exozippy.components.astrometryinstrument.astrometryinstrument import (
    read_gaia_epoch_astrometry,
)
from exozippy.utilities import getastrometry as ga
from exozippy.utilities import getdata

EXAMPLES = Path(__file__).parent.parent / "examples"
HIP1349_IAD = EXAMPLES / "HIP1349" / "HIP1349.iad.txt"
HIP1349_ASTROM = EXAMPLES / "HIP1349" / "HIP1349.Hipparcos.astrom"
DR4_PRE = EXAMPLES / "gaia4" / "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml"
HD114762 = 3937211745905473024


# ----------------------------------------------------------------------
# Hipparcos IAD
# ----------------------------------------------------------------------
def test_reconstruction_reproduces_the_hip1349_file(tmp_path):
    """
    Given the HIP 1349 IAD page examples/HIP1349 ships,
    When it is parsed, reconstructed, checked and written,
    Then the file equals the shipped HIP1349.Hipparcos.astrom value for
      value (the example's validated reconstruction), the reference point
      is the IH3/IH4 catalog position, and the parallax factors match IA5
      to 1e-3.
    """
    header, rec = ga.parse_hipparcos_iad(HIP1349_IAD.read_text())
    ab = ga.reconstruct_hipparcos_abscissae(header, rec)
    assert ga.check_hipparcos_parallax_factors(ab) < 1e-3
    assert (ab["ra0"], ab["dec0"]) == (4.22329974, -52.65159230)
    out = tmp_path / "x.astrom"
    ga.write_hipparcos_astrom(out, ab)
    np.testing.assert_array_equal(np.loadtxt(out), np.loadtxt(HIP1349_ASTROM))


def test_merged_consortia_are_the_minimum_variance_combination():
    """
    Given one orbit observed by FAST and NDAC with errors 2 and 3 mas and
      correlation 0.5, and one orbit observed by FAST only,
    When the consortia are merged,
    Then the pair becomes one abscissa equal to the generalized
      least-squares mean of the two under their covariance, with its
      variance, and the single record is kept unchanged.
    """
    rec = {
        "orbit": np.array([10, 10, 11]),
        "consortium": np.array(["F", "N", "F"]),
        "IA3": np.array([0.6, 0.6, 0.1]),
        "IA4": np.array([0.8, 0.8, 0.99]),
        "IA5": np.array([0.3, 0.3, 0.2]),
        "IA6": np.array([0.1, 0.1, 0.2]),
        "IA7": np.array([0.2, 0.2, 0.3]),
        "IA8": np.array([1.0, -2.0, 4.0]),
        "IA9": np.array([2.0, 3.0, 1.5]),
        "IA10": np.array([0.5, 0.5, np.nan]),
    }
    m = ga.merge_consortia(rec)
    C = np.array([[4.0, 0.5 * 6.0], [0.5 * 6.0, 9.0]])
    Ci = np.linalg.inv(C)
    one = np.ones(2)
    gls = (one @ Ci @ np.array([1.0, -2.0])) / (one @ Ci @ one)
    assert m["IA8"][0] == pytest.approx(gls, rel=1e-12)
    assert m["IA9"][0] == pytest.approx(
        np.sqrt(1.0 / (one @ Ci @ one)), rel=1e-12
    )
    assert list(m["consortium"]) == ["M", "F"]
    assert m["IA8"][1] == 4.0 and m["IA9"][1] == 1.5


def test_unsupported_solution_and_inconsistent_epochs_raise():
    header, rec = ga.parse_hipparcos_iad(HIP1349_IAD.read_text())
    with pytest.raises(ValueError, match="solution type '7'"):
        ga.reconstruct_hipparcos_abscissae({**header, "IH8": "7"}, rec)
    bad = {**rec, "IA6": rec["IA6"] + 0.5}
    with pytest.raises(ValueError, match="disagree"):
        ga.reconstruct_hipparcos_abscissae(header, bad)
    with pytest.raises(ValueError, match="no Hipparcos Intermediate"):
        ga.parse_hipparcos_iad("<html>no such star</html>")


def test_names_resolve_without_the_network():
    assert ga.hip_number("HIP 64426") == 64426
    assert ga.hip_number("hip1349") == 1349
    assert ga.hip_number("HD 114762") is None
    assert ga.gaia_number("Gaia DR3 3937211745905473024") == HD114762
    assert ga.gaia_number("HIP 64426") is None
    assert ga.resolve_source_id("anything", source_id=42) == 42


def test_a_hip_name_resolves_through_the_cross_match(monkeypatch):
    from astropy.table import Table

    seen = []

    def tap(adql, timeout=120):
        seen.append(adql)
        return Table({"source_id": [HD114762]})

    monkeypatch.setattr(ga, "gaia_tap", tap)
    assert ga.resolve_source_id("HIP 64426") == HD114762
    assert "original_ext_source_id = 64426" in seen[0]


# ----------------------------------------------------------------------
# Gaia DR4 epoch astrometry
# ----------------------------------------------------------------------
def test_selected_source_reads_exactly_as_from_the_full_table(tmp_path):
    """
    Given the twelve-source DR4 pre-release table,
    When HD 114762's rows are extracted by source_id,
    Then data_format: gaia_epoch reads the extracted file (no source_id
      needed) to exactly the records and reference point it reads from the
      full table; a source not in the table raises.
    """
    out = tmp_path / "one.xml"
    n = ga.select_source_votable(DR4_PRE.read_bytes(), HD114762, out)
    full, meta_full = read_gaia_epoch_astrometry(DR4_PRE, HD114762)
    one, meta_one = read_gaia_epoch_astrometry(out)
    assert n == meta_full["n_transits"] == 90
    np.testing.assert_array_equal(one.values, full.values)
    assert meta_one == meta_full
    with pytest.raises(ValueError, match="is not in the epoch-astrometry"):
        ga.select_source_votable(DR4_PRE.read_bytes(), 42, tmp_path / "x.xml")


def _datalink_error(message):
    body = (
        "<html><body><h2>SERVICE ERROR</h2><p>Context: DataRetrieval "
        f"Message: {message} Cannot process request</p></body></html>"
    ).encode()
    return urllib.error.HTTPError(
        ga.GAIA_DATALINK_URL, 500, "500", {}, io.BytesIO(body)
    )


def test_dr4_not_yet_in_the_archive_says_so(monkeypatch):
    """
    Given the archive's answer before DR4 (measured 2026-11-03: HTTP 500,
      "Unknown retrieval type: 'EPOCH_ASTROMETRY'"),
    When DR4 epoch astrometry is requested,
    Then GaiaReleaseNotAvailable names the public date and the pre-release
      route; an unrelated server error propagates unchanged.
    """

    def refuse(url, data=None, timeout=120):
        raise _datalink_error("Unknown retrieval type: 'EPOCH_ASTROMETRY'")

    monkeypatch.setattr(ga, "http_get", refuse)
    with pytest.raises(ga.GaiaReleaseNotAvailable, match="--gaia-prerelease"):
        ga.fetch_gaia_epoch_astrometry(HD114762)

    def broken(url, data=None, timeout=120):
        raise _datalink_error("disk full")

    monkeypatch.setattr(ga, "http_get", broken)
    with pytest.raises(urllib.error.HTTPError):
        ga.fetch_gaia_epoch_astrometry(HD114762)


def test_datalink_bundle_is_unpacked_to_its_votable(monkeypatch):
    """The DataLink answer is a zip holding one VOTable; that is what is
    returned, and the request names DR4, EPOCH_ASTROMETRY and RAW."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"EPOCH_ASTROMETRY-Gaia DR4 {HD114762}.xml", b"<VOTABLE/>")
    sent = {}

    def ok(url, data=None, timeout=120):
        sent.update(data)
        return buf.getvalue()

    monkeypatch.setattr(ga, "http_get", ok)
    assert ga.fetch_gaia_epoch_astrometry(HD114762) == b"<VOTABLE/>"
    assert sent["RELEASE"] == "Gaia DR4"
    assert sent["RETRIEVAL_TYPE"] == "EPOCH_ASTROMETRY"
    assert sent["DATA_STRUCTURE"] == "RAW"


# ----------------------------------------------------------------------
# The CLI
# ----------------------------------------------------------------------
def test_cli_writes_both_products_and_skips_light_curves(
    monkeypatch, tmp_path
):
    """
    Given the getdata CLI with --hipparcos-iad and --gaia-epoch-astrometry
      --gaia-prerelease-file (no network: the IAD page and the pre-release
      table are the shipped copies),
    When it runs for HIP 1349 with --source-id naming HD 114762's source,
    Then it writes HIP1349.Hipparcos.astrom and the extracted VOTable and
      never starts the light-curve download.
    """
    monkeypatch.setattr(
        ga, "fetch_hipparcos_iad", lambda hip: HIP1349_IAD.read_text()
    )
    monkeypatch.setattr(
        getdata, "run", lambda args: pytest.fail("light curves were fetched")
    )
    getdata.main(
        [
            "HIP 1349",
            "--hipparcos-iad",
            "--gaia-epoch-astrometry",
            "--gaia-prerelease-file",
            str(DR4_PRE),
            "--source-id",
            str(HD114762),
            "-p",
            str(tmp_path),
        ]
    )
    assert (tmp_path / "HIP1349.Hipparcos.astrom").is_file()
    xml = tmp_path / f"Gaia{HD114762}.DR4prerelease.epoch_astrometry.xml"
    df, meta = read_gaia_epoch_astrometry(xml)
    assert meta["source_id"] == HD114762


@pytest.mark.parametrize(
    "argv, match",
    [
        (
            ["X", "--gaia-prerelease", "--hipparcos-iad"],
            "--gaia-epoch-astrometry",
        ),
        (
            ["X", "--hip-merge-consortia", "--gaia-epoch-astrometry"],
            "--hipparcos-iad",
        ),
    ],
)
def test_cli_modifier_without_its_product_raises(argv, match):
    with pytest.raises(SystemExit, match=match):
        getdata.main(argv)


def test_the_light_curve_default_is_unchanged(monkeypatch):
    called = []
    monkeypatch.setattr(getdata, "run", lambda args: called.append(args.id))
    monkeypatch.setattr(
        getdata, "run_astrometry", lambda args: pytest.fail("astrometry ran")
    )
    getdata.main(["TOI-1234"])
    assert called == ["TOI-1234"]


# ----------------------------------------------------------------------
# The live services (optional)
# ----------------------------------------------------------------------
@pytest.mark.network
def test_live_hipparcos_iad_matches_the_shipped_copy(tmp_path):
    """The ESA service still serves HIP 1349's IAD as examples/HIP1349
    cached it (skipped when unreachable)."""
    try:
        body = ga.fetch_hipparcos_iad(1349, timeout=30)
    except (urllib.error.URLError, TimeoutError) as e:
        pytest.skip(f"no network ({e})")
    h1, r1 = ga.parse_hipparcos_iad(body)
    h2, r2 = ga.parse_hipparcos_iad(HIP1349_IAD.read_text())
    assert h1["IH3"] == h2["IH3"]
    np.testing.assert_array_equal(r1["IA8"], r2["IA8"])

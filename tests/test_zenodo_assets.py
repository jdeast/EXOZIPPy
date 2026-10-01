"""Tests for utilities/zenodo_assets.py, the one table of Zenodo pins.

Everything except the last test is offline. The last one re-reads every
registered record's API (https://zenodo.org/api/records/<id>) and compares
it with the table. It is marked ``network`` and, like the other real-Zenodo
tests (tests/test_bc_tables_zenodo.py), runs in the ordinary suite and
skips only when Zenodo cannot be reached: an HTTP 4xx means a pinned id is
wrong, which is exactly what it is for, so that fails.
"""

from __future__ import annotations

import dataclasses
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from exozippy.utilities import zenodo_assets
from exozippy.utilities.zenodo_assets import RECORDS

_SRC = Path(zenodo_assets.__file__).resolve().parents[1]

# The literals each consumer carried before review 4.9.2 moved them here,
# copied verbatim from the deleted code (components/sed/make_bc.py,
# models/MIST/eep_grid.py, models/NextGen/bc_tables.py). Migration must not
# change a byte.
_OLD_LITERALS = {
    "nextgen_spectra": (
        20547997,
        {
            "NextGen.spectra.csv": (
                259149813,
                "7a2b81333f6a5bfccd4cbc07bdea6648",
            ),
            "NextGen.wavelength.csv": (
                60943,
                "29ae520da3a5b7b3c407688abba7abf2",
            ),
        },
    ),
    "mist_eep_grids": (
        21893308,
        {
            "afe_p0_vvcrit0.0.grid.parquet": (
                127992558,
                "30545e0087ddd7dd79e87f294f4d8d58",
            ),
        },
    ),
    "nextgen_bc_tables": (
        23074951,
        {
            "2MASS.bc.parquet": (4253006, "41ff8f2f6e9a1b88f3f085f128dfcc2a"),
            "Euclid.bc.parquet": (4252914, "44925baf29ad0d1d4d3cedd4d45abc07"),
            "GAIA.bc.parquet": (8464605, "073dcf7acbd6195478954cc13a9fd742"),
            "GALEX.bc.parquet": (2849599, "e5f774a917144a0c942ea65e2a613b11"),
            "Gemini.bc.parquet": (2849819, "73e689d4edafe76052f29f16e5ee9d52"),
            "Generic.bc.parquet": (
                25307518,
                "4e067a46d646edd2991141f5a3cdd728",
            ),
            "Keck.bc.parquet": (7060295, "7b0be59aef0304dcc68b03a7831b7b85"),
            "Kepler.bc.parquet": (1445932, "9b21575a77dd737b6f0fa4d43c00b52d"),
            "PAN-STARRS.bc.parquet": (
                5656266,
                "8f5d5a4c24ad21a93028b52e4414e1ab",
            ),
            "Roman.bc.parquet": (11270909, "5e431ef09b0a65a4471f6598f7f7cdae"),
            "SLOAN.bc.parquet": (7059882, "95c997b768f763faf51ed41305ff2dcb"),
            "TESS.bc.parquet": (1445767, "af9cabbe4018da49e57ccf88c48072ca"),
            "TYCHO.bc.parquet": (8464106, "6cbe366d4c185da85d9ab399d609b517"),
            "WISE.bc.parquet": (5656474, "7738d148d3dd9e0a37c84720d3737614"),
        },
    ),
}


def test_the_migrated_pins_are_bit_identical_to_the_old_literals():
    """
    Given the size/md5 literals the consumers carried before the move,
    When they are compared with the registry,
    Then every record id, file name, size and md5 is unchanged, and no
    record was added or dropped on the way.
    """
    assert set(RECORDS) == set(_OLD_LITERALS)
    for name, (record_id, files) in _OLD_LITERALS.items():
        rec = zenodo_assets.record(name)
        assert rec.record_id == record_id, name
        got = {f: (p.size, p.md5) for f, p in rec.files.items()}
        assert got == files, name


def test_the_consumers_read_their_pins_from_the_registry():
    """
    Given the loaders that fetch from Zenodo,
    When their asset tables are inspected,
    Then each is exactly what the registry holds for its record.
    """
    from exozippy.components.sed import make_bc
    from exozippy.models.MIST import eep_grid
    from exozippy.models.NextGen import bc_tables

    assert make_bc._MODEL_DATA == {
        "NextGen": zenodo_assets.assets("nextgen_spectra")
    }
    assert eep_grid._EEP_GRID_ASSETS == zenodo_assets.assets("mist_eep_grids")

    bc = zenodo_assets.record("nextgen_bc_tables")
    assert bc_tables.ZENODO_RECORD == bc.record_id
    assert bc_tables.ZENODO_CONCEPT_RECORD == bc.concept_record_id
    assert {
        name: (meta["size"], meta["md5"])
        for name, meta in bc_tables._BC_TABLE_FILES.items()
    } == {name: (p.size, p.md5) for name, p in bc.files.items()}
    assert bc_tables._assets(bc_tables.published_tables()) == (
        zenodo_assets.assets("nextgen_bc_tables")
    )


def test_the_registry_is_internally_consistent():
    """
    Given every registered record,
    When its fields and derived URLs are checked,
    Then ids are distinct, DOIs and URLs are built from the right id, each
    file pin is a positive size and a 32-hex md5, and every citation key
    names an entry in latex/references.bib.
    """
    ids = [r.record_id for r in RECORDS.values()]
    concepts = [r.concept_record_id for r in RECORDS.values()]
    assert len(set(ids)) == len(ids)
    assert len(set(concepts)) == len(concepts)
    assert not set(ids) & set(concepts)
    bib = (_SRC / "latex" / "references.bib").read_text(encoding="utf-8")

    for name, rec in RECORDS.items():
        assert re.fullmatch(r"[a-z0-9_]+", name), name
        assert rec.title and rec.creators, name
        assert rec.doi == f"10.5281/zenodo.{rec.record_id}"
        assert rec.concept_doi == f"10.5281/zenodo.{rec.concept_record_id}"
        assert rec.api_url == (
            f"https://zenodo.org/api/records/{rec.record_id}"
        )
        if rec.citation_key is not None:
            assert re.search(
                r"@\w+\{" + re.escape(rec.citation_key) + ",", bib
            ), f"{name}: no {rec.citation_key} entry in references.bib"
            assert rec.doi in bib, f"{name}: references.bib cites another DOI"
        assert rec.files, f"{name} pins no files"
        for filename, pin in rec.files.items():
            assert "/" not in filename and filename.strip() == filename
            assert isinstance(pin.size, int) and pin.size > 0, filename
            assert re.fullmatch(r"[0-9a-f]{32}", pin.md5), filename
            meta = zenodo_assets.file_pin(name, filename)
            assert meta == {
                "url": (
                    f"https://zenodo.org/records/{rec.record_id}/files/"
                    f"{filename}"
                ),
                "size": pin.size,
                "md5": pin.md5,
            }
        assert sorted(zenodo_assets.assets(name)) == sorted(rec.files)

    # No file is pinned twice (two records claiming one md5 would make the
    # md5-keyed machine cache serve one record's file for the other's).
    md5s = [p.md5 for r in RECORDS.values() for p in r.files.values()]
    assert len(set(md5s)) == len(md5s)


def test_the_registry_cannot_be_mutated_through_a_lookup():
    """
    Given the registry,
    When a caller mutates what a lookup returned, or the registry itself,
    Then the registry is unchanged or the write is refused.
    """
    meta = zenodo_assets.file_pin("nextgen_spectra", "NextGen.spectra.csv")
    meta["md5"] = "0" * 32
    assert (
        zenodo_assets.file_pin("nextgen_spectra", "NextGen.spectra.csv")["md5"]
        == "7a2b81333f6a5bfccd4cbc07bdea6648"
    )
    with pytest.raises(TypeError):
        RECORDS["x"] = None  # type: ignore[index]
    rec = zenodo_assets.record("nextgen_spectra")
    with pytest.raises(TypeError):
        rec.files["x"] = None  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.record_id = 1  # type: ignore[misc]


def test_an_unknown_record_or_file_raises_naming_it():
    """
    Given a record name or a file name that is not registered,
    When it is looked up,
    Then KeyError is raised naming the offender (no fallback).
    """
    with pytest.raises(KeyError, match="no_such_record"):
        zenodo_assets.record("no_such_record")
    with pytest.raises(KeyError, match="no_such_file.csv"):
        zenodo_assets.file_pin("nextgen_spectra", "no_such_file.csv")
    with pytest.raises(KeyError, match="no_such_file.csv"):
        zenodo_assets.assets("mist_eep_grids", ["no_such_file.csv"])


def test_no_other_module_carries_a_zenodo_pin():
    """
    Given the source tree,
    When every .py file other than zenodo_assets.py is scanned,
    Then none spells out a Zenodo record URL with a literal id, and none
    carries a registered record id or md5 -- the pins live in ONE place
    (review 4.9.2).
    """
    pattern = re.compile(r"zenodo\.org/(?:api/)?records/\d")
    needles = {str(r.record_id) for r in RECORDS.values()}
    needles |= {str(r.concept_record_id) for r in RECORDS.values()}
    needles |= {p.md5 for r in RECORDS.values() for p in r.files.values()}
    offenders = []
    for path in _SRC.rglob("*.py"):
        if path.name == "zenodo_assets.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if pattern.search(text) or any(n in text for n in needles):
            offenders.append(str(path.relative_to(_SRC)))
    assert not offenders, (
        f"Zenodo pins outside utilities/zenodo_assets.py: {offenders}"
    )


def _live_record_or_skip(url: str) -> dict:
    """A record's API JSON, or skip if Zenodo cannot be reached.

    Only a TRANSPORT failure skips (no network, DNS, a 5xx): an HTTP 4xx
    means the pinned record id is wrong, so it fails.
    """
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code < 500:
            raise
        pytest.skip(f"Zenodo unavailable ({e}); cannot check {url}")
    except (urllib.error.URLError, TimeoutError) as e:
        pytest.skip(f"no network ({e}); cannot reach {url}")


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(RECORDS))
def test_the_pins_match_the_live_zenodo_record(name):
    """
    Given a registered record,
    When its API (https://zenodo.org/api/records/<id>) is read,
    Then its id, concept id, DOIs and the full file list with every size
    and md5 match the registry exactly -- no file missing, none extra, so
    a re-uploaded or added file is caught here, not in a fit.
    """
    rec = zenodo_assets.record(name)
    live = _live_record_or_skip(rec.api_url)

    assert int(live["id"]) == rec.record_id
    assert int(live["conceptrecid"]) == rec.concept_record_id
    assert live["doi"] == rec.doi
    assert live["conceptdoi"] == rec.concept_doi
    live_files = {}
    for f in live["files"]:
        algo, digest = f["checksum"].split(":", 1)
        assert algo == "md5", f["key"]
        live_files[f["key"]] = (int(f["size"]), digest)
    pinned = {f: (p.size, p.md5) for f, p in rec.files.items()}
    assert pinned == live_files

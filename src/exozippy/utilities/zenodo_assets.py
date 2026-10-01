"""Every Zenodo record EXOZIPPy downloads from, pinned in one place.

This is the ONE table of what the code fetches from Zenodo: for each record,
its id, its concept record (the DOI that names "this dataset, any version"),
what to cite, and for each file its name, size and md5. The download
machinery is :mod:`exozippy.utilities.zenodo`; the loaders that need a file
(``components/sed/make_bc.py``, ``models/MIST/eep_grid.py``, ...) ask this
module for its pin and hand the result to ``zenodo.fetch_assets``. Nothing
else in the tree may carry a Zenodo size, md5 or record id (review 4.9.2,
JDE ruling 2026-10-01).

The records stay SEPARATE, one per dataset, so each is versioned and cited
on its own; they are grouped for humans by the Zenodo community
https://zenodo.org/communities/exozippy.

Publishing a new version of a record
------------------------------------
1. On Zenodo, open the record and choose "New version"; upload the files
   and publish. Zenodo assigns a new record id under the same concept id.
2. Update that record's ONE entry below: ``record_id``, and every changed
   file's ``size`` and ``md5`` (from ``https://zenodo.org/api/records/<id>``,
   where ``files[*].checksum`` reads ``md5:<hex>``). The concept id does not
   change.
3. Run the network test, which re-reads every record's API and compares it
   with this table::

       pytest tests/test_zenodo_assets.py -n0

Every cache (the destination directory and the machine cache behind
``fetch_assets``) is keyed by size and md5, so the old version's copies are
simply never matched again.

There is deliberately no logic here beyond lookup, and no fallback: an
unknown record or file is a bookkeeping error in the caller and raises
KeyError naming it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

# The community every record below is published into.
COMMUNITY_URL = "https://zenodo.org/communities/exozippy"

_DOI_PREFIX = "10.5281/zenodo."


@dataclass(frozen=True)
class ZenodoFile:
    """One pinned file: its byte count and md5, as the record's API reports."""

    size: int
    md5: str


@dataclass(frozen=True)
class ZenodoRecord:
    """One published Zenodo record version and the files EXOZIPPy reads.

    ``record_id`` is the published VERSION the pins describe;
    ``concept_record_id`` groups every version of the dataset. ``files``
    lists every file in the record, and the network test asserts it is the
    whole record, not a subset. ``citation_key`` is the record's entry in
    ``latex/references.bib`` -- None for a record no generated paper cites
    yet (cite it by ``doi``); a test checks every key named here exists.
    """

    record_id: int
    concept_record_id: int
    title: str
    creators: tuple[str, ...]
    files: Mapping[str, ZenodoFile]
    citation_key: str | None

    @property
    def doi(self) -> str:
        """DOI of this version (what a paper using these exact files cites)."""
        return f"{_DOI_PREFIX}{self.record_id}"

    @property
    def concept_doi(self) -> str:
        """DOI of the dataset, resolving to its latest version."""
        return f"{_DOI_PREFIX}{self.concept_record_id}"

    @property
    def record_url(self) -> str:
        return f"https://zenodo.org/records/{self.record_id}"

    @property
    def api_url(self) -> str:
        return f"https://zenodo.org/api/records/{self.record_id}"

    def file_url(self, filename: str) -> str:
        """Download URL of one file of this record."""
        self.file(filename)
        return f"{self.record_url}/files/{filename}"

    def file(self, filename: str) -> ZenodoFile:
        """The pin of one file; raises KeyError naming it if unknown."""
        if filename not in self.files:
            raise KeyError(
                f"'{filename}' is not a file of Zenodo record "
                f"{self.record_id} ({self.title}); its files are "
                f"{sorted(self.files)}."
            )
        return self.files[filename]


def _pins(table: Mapping[str, tuple[int, str]]) -> Mapping[str, ZenodoFile]:
    """Freeze a {filename: (size, md5)} table."""
    return MappingProxyType(
        {name: ZenodoFile(size=s, md5=m) for name, (s, m) in table.items()}
    )


RECORDS: Mapping[str, ZenodoRecord] = MappingProxyType(
    {
        # NextGen model spectra, resampled to R=150. Consumer:
        # components/sed/make_bc.py (synthesizing BCs for a filter with no
        # published table; SED plotting).
        "nextgen_spectra": ZenodoRecord(
            record_id=20547997,
            concept_record_id=20547996,
            title=(
                "NextGen Model spectra resampled to R=150 for EXOZIPPy SED "
                "plotting"
            ),
            creators=("Sandhaus, Phoebe", "Eastman, Jason"),
            citation_key=None,
            files=_pins(
                {
                    "NextGen.spectra.csv": (
                        259149813,
                        "7a2b81333f6a5bfccd4cbc07bdea6648",
                    ),
                    "NextGen.wavelength.csv": (
                        60943,
                        "29ae520da3a5b7b3c407688abba7abf2",
                    ),
                }
            ),
        ),
        # The per-facility NextGen bolometric-correction tables. Consumer:
        # models/NextGen/bc_tables.py (fetched per facility on first use,
        # via components/sed/bc_grid.py).
        "nextgen_bc_tables": ZenodoRecord(
            record_id=23074951,
            concept_record_id=23074950,
            title="Bolometric Correction Grids for EXOZIPPy",
            creators=("Eastman, Jason", "Sandhaus, Phoebe"),
            citation_key="Eastman:2026bc",
            files=_pins(
                {
                    "2MASS.bc.parquet": (
                        4253006,
                        "41ff8f2f6e9a1b88f3f085f128dfcc2a",
                    ),
                    "Euclid.bc.parquet": (
                        4252914,
                        "44925baf29ad0d1d4d3cedd4d45abc07",
                    ),
                    "GAIA.bc.parquet": (
                        8464605,
                        "073dcf7acbd6195478954cc13a9fd742",
                    ),
                    "GALEX.bc.parquet": (
                        2849599,
                        "e5f774a917144a0c942ea65e2a613b11",
                    ),
                    "Gemini.bc.parquet": (
                        2849819,
                        "73e689d4edafe76052f29f16e5ee9d52",
                    ),
                    "Generic.bc.parquet": (
                        25307518,
                        "4e067a46d646edd2991141f5a3cdd728",
                    ),
                    "Keck.bc.parquet": (
                        7060295,
                        "7b0be59aef0304dcc68b03a7831b7b85",
                    ),
                    "Kepler.bc.parquet": (
                        1445932,
                        "9b21575a77dd737b6f0fa4d43c00b52d",
                    ),
                    "PAN-STARRS.bc.parquet": (
                        5656266,
                        "8f5d5a4c24ad21a93028b52e4414e1ab",
                    ),
                    "Roman.bc.parquet": (
                        11270909,
                        "5e431ef09b0a65a4471f6598f7f7cdae",
                    ),
                    "SLOAN.bc.parquet": (
                        7059882,
                        "95c997b768f763faf51ed41305ff2dcb",
                    ),
                    "TESS.bc.parquet": (
                        1445767,
                        "af9cabbe4018da49e57ccf88c48072ca",
                    ),
                    "TYCHO.bc.parquet": (
                        8464106,
                        "6cbe366d4c185da85d9ab399d609b517",
                    ),
                    "WISE.bc.parquet": (
                        5656474,
                        "7738d148d3dd9e0a37c84720d3737614",
                    ),
                }
            ),
        ),
        # The processed MISTv2.5 EEP track grids, keyed by the
        # {afe}_{vvcrit} stem eep_grid.eep_grid_filename builds. Consumer:
        # models/MIST/eep_grid.py.
        "mist_eep_grids": ZenodoRecord(
            record_id=21893308,
            concept_record_id=21893307,
            title="MISTv2.5 model grids for EXOZIPPy",
            creators=("Sandhaus, Phoebe", "Eastman, Jason"),
            citation_key=None,
            files=_pins(
                {
                    "afe_p0_vvcrit0.0.grid.parquet": (
                        127992558,
                        "30545e0087ddd7dd79e87f294f4d8d58",
                    ),
                }
            ),
        ),
    }
)


def record(name: str) -> ZenodoRecord:
    """The record registered as `name`; raises KeyError naming it if unknown."""
    if name not in RECORDS:
        raise KeyError(
            f"'{name}' is not a registered Zenodo record; registered: "
            f"{sorted(RECORDS)} (utilities/zenodo_assets.py)."
        )
    return RECORDS[name]


def file_pin(name: str, filename: str) -> dict:
    """``{"url", "size", "md5"}`` for one file, as ``fetch_assets`` takes it.

    A fresh dict on every call, so a caller can never mutate the registry.
    """
    rec = record(name)
    pin = rec.file(filename)
    return {"url": rec.file_url(filename), "size": pin.size, "md5": pin.md5}


def assets(name: str, filenames: Iterable[str] | None = None) -> dict:
    """``{filename: file_pin(...)}`` for `filenames` (default: every file)."""
    rec = record(name)
    names = sorted(rec.files) if filenames is None else list(filenames)
    return {f: file_pin(name, f) for f in names}

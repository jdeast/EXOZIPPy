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
   change. ``scripts/zenodo_publish.py pin --record <id> --update-registry``
   does this step (and the upload before it; see ``docs/zenodo.md``).
3. Run the network test, which re-reads every record's API and compares it
   with this table::

       pytest tests/test_zenodo_assets.py -n0

Every cache (the destination directory and the machine cache behind
``fetch_assets``, which CI keeps between runs keyed on this file's hash) is
keyed by size and md5, so the old version's copies are simply never matched
again.

Adding a new record
-------------------
Add its entry to RECORDS and name its loader module in OWNERS; that module
must expose ``fetch_all()``. Nothing else is needed for the test suite's
prewarm and CI's download cache to cover it.

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
    # {filename: md5s of that file in EARLIER published versions}.  An owner
    # that finds a file on disk matching one of these has a checkout that
    # predates the pin, not a local edit: it re-fetches the pinned version
    # (bc_tables._check_present) instead of raising "regenerated locally".
    # Filled when a record is re-versioned; empty for a first version.
    superseded: Mapping[str, tuple[str, ...]] = MappingProxyType({})

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
            record_id=23522276,
            concept_record_id=23074950,
            title="Bolometric Correction Grids for EXOZIPPy",
            creators=("Eastman, Jason", "Sandhaus, Phoebe"),
            citation_key="Eastman:2026bc",
            files=_pins(
                {
                    "2MASS.bc.parquet": (
                        4566065,
                        "df84dc4426a1b29e7eceebb55d798c08",
                    ),
                    "Euclid.bc.parquet": (
                        4565957,
                        "cc8ab258863d077a7b9aec4b57fbf927",
                    ),
                    "GAIA.bc.parquet": (
                        8975523,
                        "0c9654322c0d31ea8fa79a8dee7a3093",
                    ),
                    "GALEX.bc.parquet": (
                        3096721,
                        "d2e5d6ff3d9684d5a9bcf95ff3e56221",
                    ),
                    "Gemini.bc.parquet": (
                        3096953,
                        "efae58a2bd2ec304f096279749bdec37",
                    ),
                    "Generic.bc.parquet": (
                        26609685,
                        "a652a7f764e5e59666cdc04fc8cfe866",
                    ),
                    "Keck.bc.parquet": (
                        7505242,
                        "73ac7988b7b485955190ce43f62151af",
                    ),
                    "Kepler.bc.parquet": (
                        1627111,
                        "6e506f9f70df6480522cc84842a55773",
                    ),
                    "MANIFEST.txt": (
                        820,
                        "17928a49ba335de1754341dae2ab9b74",
                    ),
                    "PAN-STARRS.bc.parquet": (
                        6035220,
                        "99051cdc9a0aedbdcdfb73ed97026916",
                    ),
                    "Roman.bc.parquet": (
                        11913657,
                        "e573360c53ad4d8bfd2df39973e208df",
                    ),
                    "SLOAN.bc.parquet": (
                        7504799,
                        "de6cccbd1aa0d3369d917c1169e6212d",
                    ),
                    "TESS.bc.parquet": (
                        1626938,
                        "a5d1b015f0defe7fd36162cd23834c71",
                    ),
                    "TYCHO.bc.parquet": (
                        8974993,
                        "5fe4ed31af26dc765d84d5dfef8ed4cd",
                    ),
                    "WISE.bc.parquet": (
                        6035458,
                        "8d48b1c02b733f10c0e77d49f246372e",
                    ),
                }
            ),
            superseded=MappingProxyType(
                {
                    "2MASS.bc.parquet": ("41ff8f2f6e9a1b88f3f085f128dfcc2a",),
                    "Euclid.bc.parquet": ("44925baf29ad0d1d4d3cedd4d45abc07",),
                    "GAIA.bc.parquet": ("073dcf7acbd6195478954cc13a9fd742",),
                    "GALEX.bc.parquet": ("e5f774a917144a0c942ea65e2a613b11",),
                    "Gemini.bc.parquet": ("73e689d4edafe76052f29f16e5ee9d52",),
                    "Generic.bc.parquet": (
                        "4e067a46d646edd2991141f5a3cdd728",
                    ),
                    "Keck.bc.parquet": ("7b0be59aef0304dcc68b03a7831b7b85",),
                    "Kepler.bc.parquet": ("9b21575a77dd737b6f0fa4d43c00b52d",),
                    "PAN-STARRS.bc.parquet": (
                        "8f5d5a4c24ad21a93028b52e4414e1ab",
                    ),
                    "Roman.bc.parquet": ("5e431ef09b0a65a4471f6598f7f7cdae",),
                    "SLOAN.bc.parquet": ("95c997b768f763faf51ed41305ff2dcb",),
                    "TESS.bc.parquet": ("af9cabbe4018da49e57ccf88c48072ca",),
                    "TYCHO.bc.parquet": ("6cbe366d4c185da85d9ab399d609b517",),
                    "WISE.bc.parquet": ("7738d148d3dd9e0a37c84720d3737614",),
                }
            ),
        ),
        # The processed MISTv2.5 EEP track grids, keyed by the
        # {afe}_{vvcrit} stem eep_grid.eep_grid_filename builds. Consumer:
        # models/MIST/eep_grid.py. Version 2 (record 23802906) adds the
        # present-day star_mass and the per-track final_star_mass/final_EEP
        # columns; version 1's eleven columns are carried over bit for bit
        # (components/evolutionarymodel/evolutionarymodel.md).
        "mist_eep_grids": ZenodoRecord(
            record_id=23802906,
            concept_record_id=21893307,
            title="MISTv2.5 model grids for EXOZIPPy",
            creators=("Sandhaus, Phoebe", "Eastman, Jason"),
            citation_key=None,
            files=_pins(
                {
                    "MANIFEST.txt": (
                        73,
                        "4481988eebaf5cdceaed5f4dbbfbd801",
                    ),
                    "afe_p0_vvcrit0.0.grid.parquet": (
                        144797120,
                        "050509775cee55c1ee67232575d83433",
                    ),
                }
            ),
        ),
        # The Claret limb-darkening tables (one parquet per band and law,
        # built by models/Claret/generate_claret_tables.py). Consumer:
        # models/Claret/claret_tables.py (fetched per table on first use, via
        # the band's star_constrains_ld tie).
        "claret_ld_tables": ZenodoRecord(
            record_id=23384502,
            concept_record_id=23384501,
            title="Claret Limb-Darkening Tables for EXOZIPPy",
            creators=("Yee, Samuel", "Eastman, Jason"),
            citation_key="Yee:2026claret",
            files=_pins(
                {
                    "MANIFEST.txt": (
                        3676,
                        "4fa87eff980d7e0f2147c18314d59286",
                    ),
                    "README.md": (
                        8779,
                        "b7ce5a9284abee6e42e601b9f56c60a2",
                    ),
                    "claret_ATLAS_L_vt2.B.linear.parquet": (
                        21079,
                        "7e67bd3642d81bf12e441d1522e2c0fc",
                    ),
                    "claret_ATLAS_L_vt2.B.quad.parquet": (
                        38132,
                        "3145a808c435def2261e2078de287404",
                    ),
                    "claret_ATLAS_L_vt2.CoRoT.linear.parquet": (
                        21670,
                        "32b61fb4157b3e3f23d61c4108301f01",
                    ),
                    "claret_ATLAS_L_vt2.CoRoT.quad.parquet": (
                        35943,
                        "004a00dd71de4a3916795e8447b20535",
                    ),
                    "claret_ATLAS_L_vt2.H.linear.parquet": (
                        19677,
                        "7ef59bce9433fe55e32a6acbe6deff08",
                    ),
                    "claret_ATLAS_L_vt2.H.quad.parquet": (
                        30712,
                        "511bc193f7b714577dcf740b93881747",
                    ),
                    "claret_ATLAS_L_vt2.I.linear.parquet": (
                        20800,
                        "0ffcbd7dd6d01a24dae7589932b27e4e",
                    ),
                    "claret_ATLAS_L_vt2.I.quad.parquet": (
                        33312,
                        "7d150fb1f68456acd5b0dc0666bf0ec9",
                    ),
                    "claret_ATLAS_L_vt2.J.linear.parquet": (
                        19777,
                        "6c36fd7349c5890b30b8df99779b922d",
                    ),
                    "claret_ATLAS_L_vt2.J.quad.parquet": (
                        31353,
                        "a057b3d4b5eeb2560b612ad5ea2b3edc",
                    ),
                    "claret_ATLAS_L_vt2.K.linear.parquet": (
                        19164,
                        "dd7750c63497c3afa3270f6153832ac5",
                    ),
                    "claret_ATLAS_L_vt2.K.quad.parquet": (
                        30028,
                        "eab895a702fcbcfc4c9eb134a8f12eab",
                    ),
                    "claret_ATLAS_L_vt2.Kepler.linear.parquet": (
                        20936,
                        "9fa4cfbd3909e9c4b9cbf9af4d32abb3",
                    ),
                    "claret_ATLAS_L_vt2.Kepler.quad.parquet": (
                        35269,
                        "cf6427a769393b6e83d2594d944e3f7c",
                    ),
                    "claret_ATLAS_L_vt2.R.linear.parquet": (
                        20988,
                        "8797c960acdc2571b73a10893ea3f1f8",
                    ),
                    "claret_ATLAS_L_vt2.R.quad.parquet": (
                        34225,
                        "c6ff95843b4898c0261f0e870943ad70",
                    ),
                    "claret_ATLAS_L_vt2.Sloang.linear.parquet": (
                        21109,
                        "883ad6143efa1afb917d44b52cd72e9a",
                    ),
                    "claret_ATLAS_L_vt2.Sloang.quad.parquet": (
                        37219,
                        "abf849749bdd118703584e0f4f835f92",
                    ),
                    "claret_ATLAS_L_vt2.Sloani.linear.parquet": (
                        20860,
                        "22af70136ab90c86e95f9f617b86591b",
                    ),
                    "claret_ATLAS_L_vt2.Sloani.quad.parquet": (
                        34178,
                        "57fdbe93a0979ae4cf3f02d4b781754c",
                    ),
                    "claret_ATLAS_L_vt2.Sloanr.linear.parquet": (
                        21237,
                        "1aa4d2f234a9dedb23377c2f0354022f",
                    ),
                    "claret_ATLAS_L_vt2.Sloanr.quad.parquet": (
                        35290,
                        "e553ebbdc6b2412750775442f02af43b",
                    ),
                    "claret_ATLAS_L_vt2.Sloanu.linear.parquet": (
                        23297,
                        "413be1e75b87a182dab7d3f0877c63f8",
                    ),
                    "claret_ATLAS_L_vt2.Sloanu.quad.parquet": (
                        37401,
                        "b7630f6f6ed1b0f2cc584b1d9a35080f",
                    ),
                    "claret_ATLAS_L_vt2.Sloanz.linear.parquet": (
                        21441,
                        "4083e8e6c32ac056096e21cb7785f941",
                    ),
                    "claret_ATLAS_L_vt2.Sloanz.quad.parquet": (
                        33180,
                        "92db2da28df2a266a133ef1a93875f4c",
                    ),
                    "claret_ATLAS_L_vt2.Spit36.linear.parquet": (
                        18957,
                        "7dd0b8fb3302f1f165d0ae4ce816d909",
                    ),
                    "claret_ATLAS_L_vt2.Spit36.quad.parquet": (
                        28704,
                        "0f7345d232615b8582cf841907d9a144",
                    ),
                    "claret_ATLAS_L_vt2.Spit45.linear.parquet": (
                        18014,
                        "76029a22d90c9103a5f7b250b05b6d91",
                    ),
                    "claret_ATLAS_L_vt2.Spit45.quad.parquet": (
                        27137,
                        "6190645e69249a2c7946103851592dda",
                    ),
                    "claret_ATLAS_L_vt2.Spit58.linear.parquet": (
                        17111,
                        "bf3102bea1a41416f1f4d6ec07025bd0",
                    ),
                    "claret_ATLAS_L_vt2.Spit58.quad.parquet": (
                        26461,
                        "58b7536177ca0600a9f0944bc300ff0c",
                    ),
                    "claret_ATLAS_L_vt2.Spit80.linear.parquet": (
                        16919,
                        "00e7b3747cc8b290a6031eda52fa1aa3",
                    ),
                    "claret_ATLAS_L_vt2.Spit80.quad.parquet": (
                        25149,
                        "df0715fd90b9626afed3f9f570212f3f",
                    ),
                    "claret_ATLAS_L_vt2.Stromb.linear.parquet": (
                        21367,
                        "b7b49cb501fce62fffd824c11d324295",
                    ),
                    "claret_ATLAS_L_vt2.Stromb.quad.parquet": (
                        37714,
                        "2efe58b0419f731ced3fa5b12e8a08c8",
                    ),
                    "claret_ATLAS_L_vt2.Stromu.linear.parquet": (
                        21793,
                        "848b92367dae30c46d90876914683bf4",
                    ),
                    "claret_ATLAS_L_vt2.Stromu.quad.parquet": (
                        37329,
                        "0db44414be73e83ee07daba60cdcf540",
                    ),
                    "claret_ATLAS_L_vt2.Stromv.linear.parquet": (
                        21455,
                        "db3e958e35e783667982ef14058fea4c",
                    ),
                    "claret_ATLAS_L_vt2.Stromv.quad.parquet": (
                        39690,
                        "abbaedde7b2910d6e185f600cc599fdc",
                    ),
                    "claret_ATLAS_L_vt2.Stromy.linear.parquet": (
                        20790,
                        "42333bf361c33c5ddec463909b2f5a43",
                    ),
                    "claret_ATLAS_L_vt2.Stromy.quad.parquet": (
                        35789,
                        "9d064f461c14ff4185ee7f9643bf60fa",
                    ),
                    "claret_ATLAS_L_vt2.TESS.quad.parquet": (
                        34449,
                        "360b9c3f80ccf40e149aaea96883b669",
                    ),
                    "claret_ATLAS_L_vt2.U.linear.parquet": (
                        21924,
                        "eac142cb90356d1447ccbde892a9e66c",
                    ),
                    "claret_ATLAS_L_vt2.U.quad.parquet": (
                        38363,
                        "23e5393095ca700fbdee1f650f137a2d",
                    ),
                    "claret_ATLAS_L_vt2.V.linear.parquet": (
                        21067,
                        "23be6b7f4c8eb076472c690f115693d8",
                    ),
                    "claret_ATLAS_L_vt2.V.quad.parquet": (
                        37707,
                        "16d552971099ca2ef27edb4f37d2024b",
                    ),
                }
            ),
        ),
        # The limb-darkening tables EXOZIPPy generates from the PHOENIX-ACES
        # specific intensities (models/PHOENIX/generate_PHOENIX_LD_Tables.py),
        # one file per facility and law, flattened as {FACILITY}.{law}.ld.parquet.
        # Consumer: models/PHOENIX/ld_tables.py (fetched per facility on first
        # use, via components/band/ldgrid.py).
        "phoenix_ld_tables": ZenodoRecord(
            record_id=23522329,
            concept_record_id=23522328,
            title=(
                "Limb-Darkening Coefficient Grids for EXOZIPPy from PHOENIX "
                "specific intensities"
            ),
            creators=("Eastman, Jason", "Sandhaus, Phoebe"),
            citation_key="Eastman:2026ld",
            files=_pins(
                {
                    "2MASS.linear.ld.parquet": (
                        4379980,
                        "21f4e2538bf4bbce2928aec56e917741",
                    ),
                    "2MASS.quadratic.ld.parquet": (
                        8588726,
                        "296aa5cb0d281444eb28275048abfb89",
                    ),
                    "Euclid.linear.ld.parquet": (
                        4379856,
                        "5fb73f507c65bd971bd0c2dc67079150",
                    ),
                    "Euclid.quadratic.ld.parquet": (
                        8588489,
                        "7ff5482bd898193d6a176a5e082efc84",
                    ),
                    "GAIA.linear.ld.parquet": (
                        8588853,
                        "deca6a3b4053d7ccfd37fe1973a285de",
                    ),
                    "GAIA.quadratic.ld.parquet": (
                        17007304,
                        "713c3ebed2d42b20657b8c3f77f6359a",
                    ),
                    "GALEX.linear.ld.parquet": (
                        2977439,
                        "71831d82908afb447ba4e33a3e3bbf24",
                    ),
                    "GALEX.quadratic.ld.parquet": (
                        5783416,
                        "3965b4110a68026dac0cfab7859df74f",
                    ),
                    "Gemini.linear.ld.parquet": (
                        2977659,
                        "0f66ea266b3ca2765c7fe87d807a25e1",
                    ),
                    "Gemini.quadratic.ld.parquet": (
                        5783855,
                        "9e99d2f2a0f9c2c84ec2b5a22e4e6519",
                    ),
                    "Generic.linear.ld.parquet": (
                        24018332,
                        "7a55091e866c777bfd97e79ab2231495",
                    ),
                    "Generic.quadratic.ld.parquet": (
                        47869168,
                        "189d6572238d56da02fa9488b39ceecc",
                    ),
                    "Keck.linear.ld.parquet": (
                        7185462,
                        "2dfc6816257fcd08a26d4a70e47341a0",
                    ),
                    "Keck.quadratic.ld.parquet": (
                        14200232,
                        "68c814123f9dac4daab53e594c233454",
                    ),
                    "Kepler.linear.ld.parquet": (
                        1574677,
                        "1e796c4856682617f4b573ad967c6b6f",
                    ),
                    "Kepler.quadratic.ld.parquet": (
                        2977736,
                        "397160f55c09ed885f68a08c8fe6f7bc",
                    ),
                    "MANIFEST.txt": (
                        1752,
                        "e6ea1b84b2bffc130a221ce5f5828ffe",
                    ),
                    "PAN-STARRS.linear.ld.parquet": (
                        5782300,
                        "483b69c42f674b3f32e7ad527aa0fb8f",
                    ),
                    "PAN-STARRS.quadratic.ld.parquet": (
                        11393637,
                        "269333e4d72efec6163d66bc95dd23ad",
                    ),
                    "Roman.linear.ld.parquet": (
                        11393383,
                        "fcec5bb65614f95349c6e0518df062aa",
                    ),
                    "Roman.quadratic.ld.parquet": (
                        22616905,
                        "4d85c928a14197b2975f05018f241244",
                    ),
                    "SLOAN.linear.ld.parquet": (
                        7185032,
                        "7106885ec766b6f9b157fe337e5e6f56",
                    ),
                    "SLOAN.quadratic.ld.parquet": (
                        14199381,
                        "cee9b7a234d1326f98965cab6700c94f",
                    ),
                    "TESS.linear.ld.parquet": (
                        1574509,
                        "28adb44377b3591c36138f024ab84bfe",
                    ),
                    "TESS.quadratic.ld.parquet": (
                        2977405,
                        "7788e5f2e2902fbac2745eeb25aa3f4d",
                    ),
                    "TYCHO.linear.ld.parquet": (
                        8588367,
                        "d345f87bee6732f903d3cd79c7192512",
                    ),
                    "TYCHO.quadratic.ld.parquet": (
                        17006318,
                        "be724e28358cc2fe78b26035391677bd",
                    ),
                }
            ),
        ),
    }
)

# The module that OWNS each record: the one whose loader reads its files, and
# whose module-level ``fetch_all()`` puts every file that loader reads into
# its default destination. ``zenodo.fetch_every_record`` drives the pre-fetch
# (the test suite's prewarm, CI's machine-cache step, an offline machine)
# from this table, so the set of records fetched is the set registered here
# by construction -- a record added to RECORDS without an owner raises,
# naming it, before anything is downloaded.
OWNERS: Mapping[str, str] = MappingProxyType(
    {
        "nextgen_spectra": "exozippy.components.sed.make_bc",
        "nextgen_bc_tables": "exozippy.models.NextGen.bc_tables",
        "mist_eep_grids": "exozippy.models.MIST.eep_grid",
        "claret_ld_tables": "exozippy.models.Claret.claret_tables",
        "phoenix_ld_tables": "exozippy.models.PHOENIX.ld_tables",
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

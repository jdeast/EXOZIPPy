"""Fetch the Theophylline dataset, rather than redistributing it.

READ README.md IN THIS DIRECTORY FIRST.

WHY A FETCHER
-------------

EXOZIPPy is BSD-3-Clause.  The Theophylline table reaches most people through
R's ``datasets`` package, which is GPL-2, and shipping GPL-2 material inside a
BSD distribution is a licence-compatibility problem for THIS project even
though GPL-2 plainly permits redistribution -- it is copyleft, not a ban.

Whether the table is copyrightable at all is a separate and weaker question:
it is measured factual data, and under *Feist* facts are not copyrightable in
the US (the EU's sui generis database right is the residual concern).  The R
Foundation is in any case not its originator.

So the honest summary is that bundling it would probably be fine and is not
obviously fine, which is exactly the kind of judgement call not worth making
on a user's behalf.  Downloading it at the user's request is what
``sklearn.datasets.fetch_*`` and astroquery do, it removes the question, and
it keeps the repository small.

PROVENANCE
----------

Theophylline pharmacokinetics: 12 subjects given a single oral dose, 11 serum
concentrations each over about 25 hours, with body weight recorded.

    Boeckmann, A. J., Sheiner, L. B., and Beal, S. L. (1994),
    *NONMEM Users Guide: Part V*, NONMEM Project Group,
    University of California, San Francisco.

It is also Table 1 of Upton's theophylline study as reproduced in Pinheiro &
Bates, *Mixed-Effects Models in S and S-PLUS* (2000), and ships with R as
``datasets::Theoph``.  The copy fetched here is the Rdatasets CSV mirror,
pinned by md5 so a re-upload or a truncated download is caught rather than
silently used.

COLUMNS, AND THEIR UNITS
------------------------

``Subject`` 1-12, ``Wt`` kg, ``Dose`` mg/kg, ``Time`` hr, ``conc`` mg/L.
Note ``Dose`` is PER KILOGRAM, which is why the example config carries both a
weight and ``dose_unit: "mg/kg"``; reading it as an absolute mg dose would be
wrong by a factor of about 70.
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

FILENAME = "theoph.csv"

# Pinned by content, not just by URL.  size+md5 are what utilities.zenodo
# verifies after download; a mismatch raises rather than handing back a file
# that is not the one this component was written against.
THEOPH_ASSET = {
    "url": (
        "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/"
        "Theoph.csv"
    ),
    "size": 3147,
    "md5": "d392bc5fe83339988c0c26265c79158a",
}

# What the pinned file must contain, checked after fetching.  The md5 already
# guarantees the bytes; this is here so that REPOINTING the fetcher at another
# mirror cannot silently change the shape of the data underneath the example.
EXPECTED_COLUMNS = ["rownames", "Subject", "Wt", "Dose", "Time", "conc"]
EXPECTED_ROWS = 132
EXPECTED_SUBJECTS = 12


def fetch(dest_dir=".", force=False):
    """Ensure ``theoph.csv`` is present in ``dest_dir``.  Returns its path.

    Cached: ``utilities.zenodo.fetch_assets`` keeps a shared copy under the
    user's cache directory, so a second example directory costs no download.
    """
    dest_dir = Path(dest_dir)
    target = dest_dir / FILENAME

    if target.exists() and not force:
        logger.info("[theoph] %s already present; not re-fetching.", target)
        return target

    from ...utilities.zenodo import fetch_assets

    dest_dir.mkdir(parents=True, exist_ok=True)
    fetch_assets(
        {FILENAME: THEOPH_ASSET},
        dest_dir,
        on_fetch=lambda name: logger.warning(
            "[theoph] downloading %s from %s -- see fetch_theoph.py for its "
            "provenance and licence.",
            name,
            THEOPH_ASSET["url"],
        ),
    )
    _verify(target)
    return target


def _verify(path):
    """Check the fetched file has the shape the example was written against."""
    import pandas as pd

    frame = pd.read_csv(path)
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"[theoph] {path} has columns {list(frame.columns)}, expected "
            f"{EXPECTED_COLUMNS}. The pinned source has changed shape; the "
            f"example config's 'columns:' mapping would read the wrong data."
        )
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"[theoph] {path} has {len(frame)} rows, expected {EXPECTED_ROWS}."
        )
    n_subjects = frame["Subject"].nunique()
    if n_subjects != EXPECTED_SUBJECTS:
        raise ValueError(
            f"[theoph] {path} has {n_subjects} subjects, expected "
            f"{EXPECTED_SUBJECTS}."
        )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="exozippy-fetch-theoph",
        description=(
            "Download the Theophylline pharmacokinetic dataset (Boeckmann, "
            "Sheiner & Beal 1994) used by examples/theophylline. The data are "
            "not redistributed with EXOZIPPy; see fetch_theoph.py for why."
        ),
    )
    parser.add_argument(
        "--dest",
        default=".",
        help="Directory to write theoph.csv into (default: current).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if the file is already present.",
    )
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    path = fetch(args.dest, force=args.force)
    print(f"Theophylline data ready at {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

"""Write this example's light curves from the NASA Exoplanet Archive tables.

The three light curves are the archive's contributed data set for
MOA-2008-BLG-379 (Suzuki et al. 2014, ApJ 780, 123), downloaded as
UID_0302515_PLC_00{1,2,3}.tbl from

  https://exoplanetarchive.ipac.caltech.edu/data/ExoData/0302/0302515/data/

(listed in the archive's bulk-download script wget_MICROLENSING.bat).
Run it on a directory holding those three files:

  python prepare_data.py /path/to/downloaded/tables

It copies the (HJD, mag, err) columns verbatim, prefixed by the archive
header as '#' comments, so every number in the shipped files is the
archive's.  Nothing is trimmed, rescaled or converted here; the time
system (HJD, UTC) is declared in the config instead.
"""

import sys
from pathlib import Path

FILES = {
    "UID_0302515_PLC_001.tbl": "OGLE_I.dat",
    "UID_0302515_PLC_002.tbl": "OGLE_V.dat",
    "UID_0302515_PLC_003.tbl": "MOA_R.dat",
}


def convert(src: Path, dst: Path) -> int:
    header, rows = [], []
    for line in src.read_text().splitlines():
        if line.startswith("\\"):
            header.append("# " + line[1:].strip())
        elif line.startswith("|"):
            continue
        elif line.strip():
            t, m, e = line.split()
            rows.append(f"{t} {m} {e}")
    text = [
        "# MOA-2008-BLG-379 light curve from the NASA Exoplanet Archive,",
        f"# {src.name} (converted by prepare_data.py; values verbatim).",
        "# Columns: HJD (UTC)  magnitude  magnitude uncertainty",
        "#",
        *header,
        *rows,
    ]
    dst.write_text("\n".join(text) + "\n")
    return len(rows)


def main():
    src_dir = Path(sys.argv[1])
    out_dir = Path(__file__).resolve().parent
    for src, dst in FILES.items():
        n = convert(src_dir / src, out_dir / dst)
        print(f"{dst}: {n} points")


if __name__ == "__main__":
    main()

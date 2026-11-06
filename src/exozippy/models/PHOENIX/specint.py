"""
PHOENIX-ACES specific-intensity spectra (Husser et al. 2013): where they
are, how one is fetched, and how it is read.

The archive (Goettingen, mirrored at ftp.gwdg.de) serves one FITS per
(teff, logg, [Fe/H]) node: the primary HDU is I(lambda, mu) in
erg/s/cm^2/cm/sr on 25500 wavelengths (500-26000 A at 1 A; CRVAL1 /
CDELT1) by 78 emergent angles, the MU extension holds the angles
(ascending, mu = 1 last).  16 MB each, ~7000 nodes: streamed to a
temporary file per node and deleted, never rehosted (the derived tables
are what EXOZIPPy publishes).  Only the alpha = 0 grid is used.
"""

from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np

BASE_URL = (
    "https://ftp.gwdg.de/pub/misc/phoenix/SpecIntFITS/"
    "PHOENIX-ACES-AGSS-COND-SPECINT-2011/"
)
SUFFIX = ".PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits"
FILE_BYTES = 15926400  # every file in the grid has this size

Node = Tuple[float, float, float]


def feh_dir(feh: float) -> str:
    """The archive's metallicity folder: ``Z-0.0``, ``Z-0.5``, ``Z+0.5``."""
    return f"Z{'+' if feh > 0 else '-'}{abs(float(feh)):.1f}"


def specint_filename(teff: float, logg: float, feh: float) -> str:
    """``lte05800-4.50-0.0`` + SUFFIX; logg carries its sign, feh ``-0.0``
    for solar, as the archive spells them."""
    z = f"{'+' if feh > 0 else '-'}{abs(float(feh)):.1f}"
    return f"lte{int(round(teff)):05d}-{float(logg):.2f}{z}{SUFFIX}"


def specint_url(teff: float, logg: float, feh: float) -> str:
    return BASE_URL + feh_dir(feh) + "/" + specint_filename(teff, logg, feh)


def list_available(
    feh: float, cache_dir: Optional[Path] = None, timeout: float = 120.0
) -> Set[Tuple[float, float]]:
    """The (teff, logg) nodes the archive holds at one [Fe/H], from the
    folder listing (cached as text under ``cache_dir`` when given)."""
    cache = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"specint_listing_{feh_dir(feh)}.txt"
    if cache is not None and cache.is_file():
        text = cache.read_text()
    else:
        url = BASE_URL + feh_dir(feh).replace("+", "%2B") + "/"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        if cache is not None:
            cache.write_text(text)
    nodes = set()
    for m in re.finditer(
        r'href="(lte\d{5}-\d\.\d\d[-+%0-9B.]+?' + re.escape(SUFFIX) + ')"',
        text,
    ):
        name = m.group(1).replace("%2B", "+")
        mm = re.match(r"lte(\d{5})-(\d\.\d\d)([-+]\d\.\d)", name)
        if mm:
            nodes.add((float(mm.group(1)), float(mm.group(2))))
    return nodes


def fetch_specint(
    teff: float,
    logg: float,
    feh: float,
    dest_dir: Path | str,
    timeout: float = 300.0,
    retries: int = 4,
) -> Path:
    """Download one node's FITS into ``dest_dir`` (atomic, size-checked,
    retried), returning its path.  A complete file already there is kept."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / specint_filename(teff, logg, feh)
    if dest.is_file() and dest.stat().st_size == FILE_BYTES:
        return dest
    part = dest.with_suffix(".fits.part")
    url = specint_url(teff, logg, feh)
    last = None
    for attempt in range(retries):
        try:
            with (
                urllib.request.urlopen(url, timeout=timeout) as r,
                part.open("wb") as fh,
            ):
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            if part.stat().st_size != FILE_BYTES:
                raise OSError(
                    f"{part.stat().st_size} B downloaded, {FILE_BYTES} expected"
                )
            part.replace(dest)
            return dest
        except OSError as exc:
            last = exc
            time.sleep(min(60.0, 5.0 * 2**attempt))
    raise RuntimeError(f"could not fetch {url}: {last}")


def read_specint(path: Path | str) -> Dict[str, np.ndarray]:
    """``{"wave": (n_lambda,) Angstrom, "mu": (n_mu,) ascending,
    "intensity": (n_mu, n_lambda), "teff", "logg", "feh"}``.

    Intensities are returned as float64 in the file's own units; only
    RATIOS of them are ever used (I(mu)/I(1) per band), so the unit
    does not matter and is not converted.
    """
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header
        data = np.asarray(hdul[0].data, dtype=np.float64)
        mu = np.asarray(hdul["MU"].data, dtype=np.float64)
    n_lambda = int(hdr["NAXIS1"])
    wave = float(hdr["CRVAL1"]) + float(hdr["CDELT1"]) * np.arange(n_lambda)
    if data.shape != (mu.size, n_lambda):
        raise ValueError(
            f"{path}: intensity array {data.shape} does not match "
            f"(n_mu={mu.size}, n_lambda={n_lambda})"
        )
    order = np.argsort(mu)
    return {
        "wave": wave,
        "mu": mu[order],
        "intensity": data[order],
        "teff": float(hdr["PHXTEFF"]),
        "logg": float(hdr["PHXLOGG"]),
        "feh": float(hdr["PHXM_H"]),
    }

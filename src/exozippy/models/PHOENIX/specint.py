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

The bolometric corrections need one more number per node: the model's own
BOLOMETRIC flux.  The specific intensities stop at 2.6 um, and their flux
2 pi int I mu dmu is the flux at the model's OUTER radius, not at the
radius that defines Teff -- so it is NOT sigma Teff^4: measured 2026-11-18,
int F dlambda over 500 A - 5.5 um is 0.985 sigma Teff^4 at (5800, 4.5),
0.93 at (4500, 2.0) and 0.78 at (4000, 0.5), with only ~1-2 percent of a
Planck curve beyond 5.5 um.  The archive's flux spectra (HiResFITS, one
per node, 6 MB, F_lambda in erg/s/cm^2/cm on 500 A - 5.5 um) are that same
flux -- 2 pi int I mu dmu of the specific intensities reproduces them to
1e-4 at every node checked -- and reach far enough that a Rayleigh-Jeans
tail beyond 5.5 um is a percent-level correction to a percent-level term.
``fetch_flux`` / ``bolometric_flux`` read them for that number only.
"""

from __future__ import annotations

import os
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

# The flux spectra (same models, same nodes): one F_lambda per node on the
# shared wavelength array WAVE_FILE.
FLUX_BASE_URL = (
    "https://ftp.gwdg.de/pub/misc/phoenix/HiResFITS/"
    "PHOENIX-ACES-AGSS-COND-2011/"
)
FLUX_WAVE_URL = (
    "https://ftp.gwdg.de/pub/misc/phoenix/HiResFITS/"
    "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
)
FLUX_SUFFIX = ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
FLUX_FILE_BYTES = 6287040  # every flux file in the grid has this size
FLUX_WAVE_BYTES = 12556800


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


def flux_filename(teff: float, logg: float, feh: float) -> str:
    """The flux spectrum of a node: the specific-intensity name with the
    flux grid's suffix."""
    return specint_filename(teff, logg, feh)[: -len(SUFFIX)] + FLUX_SUFFIX


def flux_url(teff: float, logg: float, feh: float) -> str:
    return FLUX_BASE_URL + feh_dir(feh) + "/" + flux_filename(teff, logg, feh)


# product -> (folder base url, file suffix, listing-cache tag)
_PRODUCTS = {
    "specint": (BASE_URL, SUFFIX, "specint"),
    "flux": (FLUX_BASE_URL, FLUX_SUFFIX, "flux"),
}


def list_available(
    feh: float,
    cache_dir: Optional[Path] = None,
    timeout: float = 120.0,
    product: str = "specint",
) -> Set[Tuple[float, float]]:
    """The (teff, logg) nodes the archive holds at one [Fe/H], from the
    folder listing (cached as text under ``cache_dir`` when given).
    ``product`` is ``"specint"`` (the specific intensities) or ``"flux"``
    (the flux spectra, ``fetch_flux``)."""
    if product not in _PRODUCTS:
        raise ValueError(
            f"product must be one of {sorted(_PRODUCTS)}, not {product!r}"
        )
    base, suffix, tag = _PRODUCTS[product]
    cache = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"{tag}_listing_{feh_dir(feh)}.txt"
    if cache is not None and cache.is_file():
        text = cache.read_text()
    else:
        url = base + feh_dir(feh).replace("+", "%2B") + "/"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        if cache is not None:
            cache.write_text(text)
    nodes = set()
    for m in re.finditer(
        r'href="(lte\d{5}-\d\.\d\d[-+%0-9B.]+?' + re.escape(suffix) + ')"',
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
    return _fetch(
        specint_url(teff, logg, feh),
        Path(dest_dir) / specint_filename(teff, logg, feh),
        FILE_BYTES,
        timeout,
        retries,
    )


def fetch_flux(
    teff: float,
    logg: float,
    feh: float,
    dest_dir: Path | str,
    timeout: float = 300.0,
    retries: int = 4,
) -> Path:
    """Download one node's flux spectrum (fetch_specint's rules)."""
    return _fetch(
        flux_url(teff, logg, feh),
        Path(dest_dir) / flux_filename(teff, logg, feh),
        FLUX_FILE_BYTES,
        timeout,
        retries,
    )


def fetch_flux_wavelength(
    dest_dir: Path | str, timeout: float = 300.0, retries: int = 4
) -> Path:
    """The flux spectra's shared wavelength array (fetched once, kept)."""
    return _fetch(
        FLUX_WAVE_URL,
        Path(dest_dir) / FLUX_WAVE_URL.rsplit("/", 1)[1],
        FLUX_WAVE_BYTES,
        timeout,
        retries,
    )


def _fetch(
    url: str, dest: Path, nbytes: int, timeout: float, retries: int
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == nbytes:
        return dest
    # Per process: two workers may fetch the shared wavelength file at once.
    part = dest.with_name(f"{dest.name}.{os.getpid()}.part")
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
            if part.stat().st_size != nbytes:
                raise OSError(
                    f"{part.stat().st_size} B downloaded, {nbytes} expected"
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


def read_flux_wavelength(path: Path | str) -> np.ndarray:
    """The flux spectra's wavelength array, Angstrom (vacuum)."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)


# Above this wavelength (Angstrom) the flux spectra stop; the bolometric
# flux beyond it is a Rayleigh-Jeans tail F(lam) = F(lam0) (lam0/lam)^4,
# whose integral is F(lam0) lam0 / 3, with F(lam0) the mean flux over the
# last TAIL_FIT_FRACTION of the wavelength range (one pixel would carry a
# line).  The tail is 0.1-3 percent of the total from 12000 K to 2300 K.
TAIL_FIT_FRACTION = 0.02


def bolometric_flux(path: Path | str, wave: np.ndarray) -> Dict[str, float]:
    """The model's bolometric flux, erg/s/cm^2, at the radius its spectra
    are given at (the outer boundary of the model; see the module
    docstring): ``{"fbol", "tail_fraction", "teff", "logg", "feh"}``.

    int F dlambda over the file's range (500 A - 5.5 um) plus the
    Rayleigh-Jeans tail beyond it.  Flux below 500 A is not added: it is
    below 1e-5 of the total at the grid's hottest node.
    """
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header
        flux = np.asarray(hdul[0].data, dtype=np.float64) * 1e-8  # per A
    if flux.shape != wave.shape:
        raise ValueError(
            f"{path}: flux array {flux.shape} does not match the wavelength "
            f"array {wave.shape}"
        )
    inside = float(np.trapezoid(flux, wave))
    lam0 = float(wave[-1])
    edge = wave >= wave[0] + (1.0 - TAIL_FIT_FRACTION) * (lam0 - wave[0])
    tail = float(np.mean(flux[edge])) * lam0 / 3.0
    total = inside + tail
    if not (np.isfinite(total) and total > 0):
        raise ValueError(f"{path}: bolometric flux {total} is not positive")
    return {
        "fbol": total,
        "tail_fraction": tail / total,
        "teff": float(hdr["PHXTEFF"]),
        "logg": float(hdr["PHXLOGG"]),
        "feh": float(hdr["PHXM_H"]),
    }

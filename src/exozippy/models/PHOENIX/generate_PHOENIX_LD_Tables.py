"""
Build the limb-darkening tables -- and the bolometric-correction tables --
from the PHOENIX-ACES specific intensities (Husser et al. 2013), for the
filters the NextGen BC tables ship -- the analog of
models/NextGen/generate_NextGen_BC_Tables.py.  ``products`` selects which
(``ld``, ``bc``, or both); both come from ONE stream of the archive.

Per (teff, logg, feh) node the intensity file I(lambda, mu) is streamed
from the archive (specint.py), integrated through every filter profile
with the SAME weighting and extinction law as the bolometric corrections
(bolometric_correction._filter_set_data: SVO DetectorType weighting,
models/extinction_law.ascii at R_V = 3.1, the Av axis of the BC tables)
to I_band(mu; Av), and each profile is fitted with the linear and the
quadratic law.  One table per facility and law,
``models/PHOENIX/LDs/{law}/{FACILITY}.ld.parquet``, in the BC tables'
format (bc_grid.write_bc_table: teff logg feh alpha Av Rv + one column
per coefficient, metadata per column), so the incremental merge, the
complete-box loader and the Zenodo publish path all transfer.

The band-integrated intensities themselves are kept as the pipeline's
intermediate, ``LDs/intensities/feh{+X.X}.intensities.parquet`` (float32,
~0.6 MB per node): a different law or fit recipe can be refitted from
them without streaming the archive again, and interior holes of the
archive grid are filled from them (below).

The fit (notes 2026-08-04, design locked):

* QUASI-SPHERICAL REMAP.  PHOENIX atmospheres are spherical, so I(mu)
  falls off a cliff over the last few percent of the geometric disk,
  where rays graze only tenuous upper atmosphere.  The limb a transit or
  finite-source code assumes is at the TOP of that cliff.  Per profile
  the limb is placed at the mu of steepest descent (max dI/dmu, searched
  at mu < MU_LIMB_SEARCH_MAX) and mu is remapped so it sits at mu' = 0
  (Claret, Hauschildt & Witte 2012/2013).
* FLUX-CONSERVING FIT.  The law conserves the remapped profile's flux
  F = int_0^1 I(mu') mu' dmu' exactly -- which fixes the linear law's u
  outright -- and the quadratic law's remaining degree of freedom is the
  mu'-weighted least-squares residual.  The mu weight sends the limb's
  contribution to zero, so the fit is insensitive to any residual cliff.

The mu grid is PER NODE.  PHOENIX's 78 emergent angles follow each
model's own atmospheric structure, so no two nodes share them exactly;
the first pass of this generator fitted a whole [Fe/H] plane on one
node's angles and put the solar V-band limb a factor of two too dark.
Every node's angles therefore travel with its intensities -- through the
intermediate and into the fit -- and a hole's neighbours are resampled
onto one grid before they are interpolated.

Holes.  The archive is not a rectangle: hot stars are absent at the
lowest gravities and a few interior nodes are missing (5000 K at solar
metallicity).  An interior node is filled by LINEAR INTERPOLATION ALONG
TEFF of the band-integrated intensities of its two nearest available
neighbours at the same (logg, feh), then fitted like any other; the
table's ``meta["padding"]`` names every such node.  Edge planes with no
such neighbours stay absent and the loader's complete box trims them.

The bolometric corrections (``products`` containing ``bc``; review 8.9.12)
are NextGen's (models/NextGen/bolometric_correction.py) with two inputs
replaced.  The band flux is the specific intensities' own flux,
2 pi int I_band(mu) mu dmu, from the SAME unnormalized band intensities the
limb darkening is fitted to -- one atmosphere for the limb darkening and
the flux of a star.  And the bolometric term is the MODEL's bolometric
flux, not sigma Teff^4:

    BC_X = 2.5 log10[ L0 / (4 pi (10 pc)^2 F_bol)
                      * int(F_lam e^-tau S dlam) / (ZP_X int(S dlam)) ]

because a PHOENIX flux is given at the model's outer radius, where it is
smaller than sigma Teff^4 by (R_Teff / R_out)^2 -- 2 percent for a dwarf,
22 percent at (4000 K, logg 0.5) (specint.py; measured 2026-11-18).  Only
the RATIO of a band flux to the bolometric flux at one radius is physical,
and with F_bol the 500-26000 A range of the intensities limits which
filters get a column (COVERAGE_MIN: GALEX through 2MASS Ks; no WISE,
Spitzer or Johnson M) but not their accuracy.  F_bol comes from the
archive's flux spectrum of the same node (500 A - 5.5 um plus a
Rayleigh-Jeans tail, specint.bolometric_flux), which equals the
intensities' flux where both exist.  The tables are written in the NextGen
format, ``models/PHOENIX/BCs/{FACILITY}.bc.parquet`` (Vega-referenced, the
same SVO zeropoints and DetectorType weighting), with the magnitude-system
record ``PHOENIX.magsys.csv`` beside them, and the per-node unnormalized
band fluxes and F_bol are kept in the intensities intermediate, so either
can be re-derived without the archive.  A node missing from the archive
inside a (logg, [Fe/H]) row is filled by linear interpolation of the BCs
along Teff and recorded in ``meta["padding"]``, as for the limb darkening.

Run (hours of streaming; a compute node with network, not a laptop):

    python -m exozippy.models.PHOENIX.generate_PHOENIX_LD_Tables --workers 32
    python -m exozippy.models.PHOENIX.generate_PHOENIX_LD_Tables --workers 32 --products bc
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from exozippy.components.sed.bc_grid import (
    BC_PARAM_COLS,
    DEFAULT_MODEL_ROOT,
    GRID_KEY_COLS,
    _load_alias_table,
    bc_table_filter_columns,
    bc_table_path,
    read_bc_meta,
    read_bc_table,
    resolve_filter_name,
    write_bc_table,
)
from exozippy.models.NextGen.bolometric_correction import (
    L0,
    _filter_set_data,
)
from exozippy.models.NextGen.generate_NextGen_BC_Tables import (
    FILTER_SETS,
    write_magsys_table,
)
from exozippy.models.PHOENIX import specint

logger = logging.getLogger(__name__)

MODEL = "PHOENIX"
LAWS: Tuple[str, ...] = ("linear", "quadratic")
COEFFICIENTS = {"linear": ("u",), "quadratic": ("u1", "u2")}
RV = 3.10
FLUX_WEIGHTING = "detector"
GENERATOR = "models/PHOENIX/generate_PHOENIX_LD_Tables.py"
INTENSITIES_TAG = (
    "PHOENIX-ACES-AGSS-COND-SPECINT-2011 (Husser+2013), 78 mu angles, "
    "500-26000 A at 1 A"
)
METHOD = (
    "quasi-spherical: limb at the mu of max dI/dmu, remapped to mu' = 0; "
    "flux-conserving fit, mu'-weighted least squares on the residual"
)
# The limb is searched for below this mu: the cliff of a spherical model
# sits in the last few percent of the disk, and a concave profile's
# derivative is largest at small mu anyway.
MU_LIMB_SEARCH_MAX = 0.5
# A filter is tabulated only if this fraction of its weighted profile's
# integral lies inside the intensity spectra's wavelength range (WISE and
# Spitzer, past 2.6 um, are not).
COVERAGE_MIN = 0.99
# Where a worker streams a node's FITS to (deleted after use).
WORK_DIR_DEFAULT = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))

GRID_YAML = Path(__file__).parent / "LDs" / "PHOENIX.grid.yaml"

# What a run writes: the limb-darkening tables, the bolometric-correction
# tables, or both (one stream of the archive either way).
PRODUCTS: Tuple[str, ...] = ("ld", "bc")
# The BC tables' column provenance (filter_meta), the NextGen pipeline's
# keys: what the band flux is and what it is divided by.
MAG_SYSTEM = "Vega"
SPECTRA_TAG = (
    "PHOENIX-ACES-AGSS-COND-SPECINT-2011 (Husser+2013): flux 2 pi int I mu "
    "dmu of the specific intensities, 500-26000 A, resampled to R=20000"
)
BOLOMETRIC = (
    "the model's own bolometric flux at the radius its spectra are given "
    "at: PHOENIX-ACES-AGSS-COND-2011 flux spectrum, 500 A - 5.5 um, plus a "
    "Rayleigh-Jeans tail (not sigma Teff^4)"
)
# 10 pc in metres, squared, times 4 pi: L0 / FOUR_PI_D10_SQ is the
# bolometric flux of an M_bol = 0 star at 10 pc, W/m^2.
FOUR_PI_D10_SQ = 4.0 * np.pi * (10.0 * 3.0856775814913673e16) ** 2
# erg/s/cm^2 -> W/m^2
CGS_FLUX_TO_SI = 1e-3


# -------------------------------------------------------------------
# Paths and grid
# -------------------------------------------------------------------


def ld_table_path(
    model_root: Path | str, law: str, facility: str, model: str = MODEL
) -> Path:
    """``{model_root}/{model}/LDs/{law}/{facility}.ld.parquet``."""
    return Path(model_root) / model / "LDs" / law / f"{facility}.ld.parquet"


def intensities_path(model_root: Path | str, feh: float) -> Path:
    return (
        Path(model_root)
        / MODEL
        / "LDs"
        / "intensities"
        / f"feh{float(feh):+.1f}.intensities.parquet"
    )


def read_grid_yaml(model_root: Path | str = DEFAULT_MODEL_ROOT) -> dict:
    path = Path(model_root) / MODEL / "LDs" / "PHOENIX.grid.yaml"
    if not path.is_file():
        path = GRID_YAML
    with open(path) as fh:
        grid = yaml.safe_load(fh)["grid"]
    return {k: np.asarray(v, dtype=float) for k, v in grid.items()}


def coefficient_column(mist_name: str, law: str, coef: str) -> str:
    """The table column of one coefficient: the MIST filter name for the
    linear law's single u, ``{name}__u1`` / ``{name}__u2`` for the
    quadratic law."""
    return mist_name if law == "linear" else f"{mist_name}__{coef}"


# -------------------------------------------------------------------
# Physics: band intensities, the remap, the fits
# -------------------------------------------------------------------


def _trapz(x, y):
    dx = x[..., 1:] - x[..., :-1]
    return np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, axis=-1)


def filter_coverage(
    fdata: dict, wave_min: float, wave_max: float
) -> np.ndarray:
    """Fraction of each weighted profile's integral inside [wave_min,
    wave_max] (Angstrom)."""
    x = fdata["filter_wavelengths"]
    s = fdata["filter_profiles"]
    inside = (x >= wave_min) & (x <= wave_max)
    total = _trapz(x, s)
    part = _trapz(x, np.where(inside, s, 0.0))
    return part / np.where(total > 0, total, np.nan)


def interpolate_intensities(spec: dict) -> np.ndarray:
    """I(mu, lambda) on the filters' common wavelength grid
    (filters/filter.py's WAVELENGTH_PTS, R = 20000), zero outside the
    spectra's range -- done ONCE per node and sliced per filter set by its
    ``wave_mask``.  Returns (n_mu, n_wavelength_pts)."""
    from exozippy.filters.filter import Filter

    x = np.asarray(Filter.WAVELENGTH_PTS, dtype=float)
    wave, inten = spec["wave"], spec["intensity"]
    inside = (x >= wave[0]) & (x <= wave[-1])
    out = np.zeros((inten.shape[0], x.size))
    xi = x[inside]
    for k in range(inten.shape[0]):
        out[k, inside] = np.interp(xi, wave, inten[k])
    return out


def band_intensities(
    i_full: np.ndarray, fdata: dict, av_pts: Sequence[float]
) -> np.ndarray:
    """I_band(mu; Av) for one node and one filter set, normalized so that
    mu = 1 is 1 per (Av, filter): only the shape of the profile matters
    for the fit.  (n_av, n_filters, n_mu); see band_intensities_raw."""
    return normalize_band_intensities(
        band_intensities_raw(i_full, fdata, av_pts)
    )


def normalize_band_intensities(ib: np.ndarray) -> np.ndarray:
    """``ib`` divided by its mu = 1 value (the last; mu ascending)."""
    center = ib[:, :, -1][:, :, None]
    return ib / np.where(center > 0, center, np.nan)


def band_intensities_raw(
    i_full: np.ndarray, fdata: dict, av_pts: Sequence[float]
) -> np.ndarray:
    """int I(lambda, mu) exp(-tau) S_w dlambda for one node and one filter
    set, in the file's units times Angstrom -- NOT normalized.

    ``i_full`` is interpolate_intensities' array; ``fdata`` the filter
    set's _filter_set_data (profiles already carry the detector
    weighting, ``wave_mask`` its slice of the common grid).  Returns
    (n_av, n_filters, n_mu).  One matrix product per Av: the trapezoid
    weights fold into the profiles.
    """
    x = fdata["filter_wavelengths"][0]  # every row shares the grid
    inten = i_full[:, fdata["wave_mask"]]  # (n_mu, n_wave)
    av = np.asarray(av_pts, dtype=float)
    tau_unit = fdata["extinction_modeled"] / fdata["V_band_extinction"] / 1.086
    s = fdata["filter_profiles"]  # (n_filt, n_wave)
    dx = np.diff(x)
    trap = np.zeros(x.size)
    trap[:-1] += 0.5 * dx
    trap[1:] += 0.5 * dx
    s_trap = s * trap[None, :]
    n_mu = inten.shape[0]
    ib = np.empty((av.size, s.shape[0], n_mu))
    for i, a in enumerate(av):
        w = s_trap * np.exp(-tau_unit * a)[None, :]  # (n_filt, n_wave)
        ib[i] = (inten @ w.T).T  # (n_filt, n_mu)
    return ib


def band_fluxes(ib_raw: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """The band flux int F_lam exp(-tau) S_w dlam, erg/s/cm^2 (times the
    profile's units: Angstrom for a photon counter), from the unnormalized
    band intensities: F = 2 pi int I mu dmu over the node's own angles, and
    the intensities' erg/s/cm^2/cm/sr per CENTIMETRE of wavelength times
    1e-8 per Angstrom -- the units of NextGen's flux_in_filter, so the BC
    divides it by the same filter_zero_flux.  (n_av, n_filters)."""
    mu = np.asarray(mu, dtype=float)
    return 2.0 * np.pi * 1e-8 * np.trapezoid(ib_raw * mu, mu, axis=-1)


def bc_from_fluxes(
    band_flux: np.ndarray, fbol: float, fdata: dict
) -> np.ndarray:
    """BC_X(Av) for one node and one filter set (the module docstring's
    formula): ``band_flux`` (n_av, n_filters) from band_fluxes, ``fbol``
    the model's bolometric flux at the same radius, erg/s/cm^2 (specint.
    bolometric_flux), ``fdata`` the filter set's _filter_set_data."""
    term1 = L0 / FOUR_PI_D10_SQ / (float(fbol) * CGS_FLUX_TO_SI)
    return 2.5 * np.log10(term1 * band_flux / fdata["filter_zero_flux"])


def remap_quasi_spherical(mu: np.ndarray, p: np.ndarray):
    """``(mu_prime, p_prime, mu_limb)``: the profile with its limb -- the
    mu of steepest descent below MU_LIMB_SEARCH_MAX -- moved to mu' = 0.

    ``mu`` ascending with mu[-1] = 1, ``p`` the normalized profile.  Points
    inside the cliff (mu < mu_limb) are dropped; r' = r / r_limb and
    mu' = sqrt(1 - r'^2), so mu' = sqrt((mu^2 - mu_limb^2) / (1 -
    mu_limb^2)).  A profile with no cliff (the derivative largest at the
    first point) is returned unchanged but for the first point moving to
    mu' = 0.
    """
    mu = np.asarray(mu, dtype=float)
    p = np.asarray(p, dtype=float)
    dp = np.gradient(p, mu)
    search = mu < MU_LIMB_SEARCH_MAX
    if not search.any():
        search = np.ones_like(mu, dtype=bool)
    k = int(np.argmax(np.where(search, dp, -np.inf)))
    mu_limb = mu[k]
    keep = mu >= mu_limb
    mu_k, p_k = mu[keep], p[keep]
    denom = 1.0 - mu_limb**2
    mu_prime = np.sqrt(np.clip((mu_k**2 - mu_limb**2) / denom, 0.0, 1.0))
    mu_prime[-1] = 1.0
    return mu_prime, p_k, float(mu_limb)


def _profile_flux(mu_prime: np.ndarray, p_prime: np.ndarray) -> float:
    """F = int_0^1 I(mu') mu' dmu' / I(1) by the trapezoid rule, with the
    end points (0, p(0)) and (1, 1) in place."""
    m = mu_prime
    f = p_prime * m
    if m[0] > 0:  # the limb point sits at mu' = 0 by construction
        m = np.concatenate([[0.0], m])
        f = np.concatenate([[0.0], f])
    return float(np.trapezoid(f, m))


def fit_linear(mu_prime: np.ndarray, p_prime: np.ndarray) -> Tuple[float]:
    """The linear law's u from flux conservation alone:
    int (1 - u (1 - mu)) mu dmu = 1/2 - u/6 = F  ->  u = 3 - 6 F."""
    F = _profile_flux(mu_prime, p_prime)
    return (3.0 - 6.0 * F,)


def fit_quadratic(
    mu_prime: np.ndarray, p_prime: np.ndarray
) -> Tuple[float, float]:
    """(u1, u2): flux conserved exactly (1/2 - u1/6 - u2/12 = F) and the
    mu'-weighted residual minimized over the remaining degree of freedom.

    Substituting u1 = (3 - 6F) - u2/2, the residual is linear in u2:
    r_i + u2 a_i with r_i = p_i - 1 + (3 - 6F)(1 - mu_i) and
    a_i = (1 - mu_i)^2 - (1 - mu_i)/2, so u2 = -sum(w a r) / sum(w a^2),
    w the trapezoid weights times mu' (the flux measure).
    """
    F = _profile_flux(mu_prime, p_prime)
    m, p = mu_prime, p_prime
    w = np.zeros_like(m)
    dm = np.diff(m)
    w[:-1] += 0.5 * dm
    w[1:] += 0.5 * dm
    w *= m
    one_m = 1.0 - m
    r = p - 1.0 + (3.0 - 6.0 * F) * one_m
    a = one_m**2 - 0.5 * one_m
    denom = float(np.sum(w * a * a))
    u2 = -float(np.sum(w * a * r)) / denom if denom > 0 else 0.0
    u1 = (3.0 - 6.0 * F) - 0.5 * u2
    return (u1, u2)


_FITS = {"linear": fit_linear, "quadratic": fit_quadratic}


def fit_profile(mu: np.ndarray, p: np.ndarray) -> Dict[str, Tuple[float, ...]]:
    """Both laws' coefficients for one normalized profile."""
    mu_p, p_p, _limb = remap_quasi_spherical(mu, p)
    return {law: _FITS[law](mu_p, p_p) for law in LAWS}


def coefficients_from_intensities(
    mu: np.ndarray, ib: np.ndarray
) -> Dict[str, np.ndarray]:
    """``{law: (n_av, n_filters, n_coef)}`` from band_intensities' array."""
    n_av, n_f, _ = ib.shape
    out = {
        law: np.full((n_av, n_f, len(COEFFICIENTS[law])), np.nan)
        for law in LAWS
    }
    for i in range(n_av):
        for j in range(n_f):
            prof = ib[i, j]
            if not np.all(np.isfinite(prof)):
                continue
            fits = fit_profile(mu, prof)
            for law in LAWS:
                out[law][i, j] = fits[law]
    return out


# -------------------------------------------------------------------
# One node, in a worker
# -------------------------------------------------------------------


def _facility_filters(filter_sets: Dict[str, List[str]]):
    """{facility: (svo ids tuple, mist names list)} through the alias table."""
    alias_df = _load_alias_table()
    out = {}
    for fac, filters in filter_sets.items():
        svo = tuple(
            resolve_filter_name(f, alias_df, alias="SVO") for f in filters
        )
        mist = [
            resolve_filter_name(f, alias_df, alias="MIST") for f in filters
        ]
        out[fac] = (svo, mist)
    return out


def covered_filters(filter_sets: Dict[str, List[str]]) -> Dict[str, List[int]]:
    """{facility: indices of the filters inside the intensity spectra's
    wavelength range (COVERAGE_MIN)} -- the columns the tables get."""
    out = {}
    for fac, (svo, _mist) in _facility_filters(filter_sets).items():
        fdata = _filter_set_data(svo, FLUX_WEIGHTING)
        cov = filter_coverage(fdata, 500.0, 26000.0)
        keep = [
            i
            for i, c in enumerate(cov)
            if np.isfinite(c) and c >= COVERAGE_MIN
        ]
        out[fac] = keep
    return out


_FLUX_WAVE: Dict[str, np.ndarray] = {}


def _flux_wavelength(work_dir: Path) -> np.ndarray:
    """The flux spectra's wavelength array, fetched into ``work_dir`` once
    and kept (read once per worker process)."""
    key = str(work_dir)
    if key not in _FLUX_WAVE:
        path = specint.fetch_flux_wavelength(work_dir)
        _FLUX_WAVE[key] = specint.read_flux_wavelength(path)
    return _FLUX_WAVE[key]


def node_bolometric_flux(node, work_dir: Path) -> Dict[str, float]:
    """specint.bolometric_flux of one node's flux spectrum, streamed into
    ``work_dir`` and deleted; raises if the file is not that node's."""
    teff, logg, feh = node
    wave = _flux_wavelength(work_dir)
    path = specint.fetch_flux(teff, logg, feh, work_dir)
    try:
        info = specint.bolometric_flux(path, wave)
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    got = (info["teff"], info["logg"], info["feh"])
    if not np.allclose(got, node, atol=1e-6):
        raise ValueError(
            f"the flux spectrum fetched for node {node} is labelled {got}"
        )
    return info


def covered_filter_sets(
    filter_sets: Dict[str, List[str]] = FILTER_SETS,
) -> Dict[str, List[str]]:
    """``filter_sets`` restricted to the filters inside the intensity
    spectra's wavelength range (covered_filters): the columns both the
    limb-darkening and the bolometric-correction tables get.  A facility
    with none left is dropped."""
    keep = covered_filters(filter_sets)
    return {
        fac: [filter_sets[fac][i] for i in keep[fac]]
        for fac in filter_sets
        if keep[fac]
    }


def write_bc_magsys_table(
    model_root: Path | str = DEFAULT_MODEL_ROOT,
    filter_sets: Dict[str, List[str]] = FILTER_SETS,
) -> Path:
    """``PHOENIX/BCs/PHOENIX.magsys.csv``: each BC column's native
    magnitude system and m_AB - m_Vega, computed exactly as the NextGen
    record is (it depends on the filter profile, weighting and Vega
    zeropoint only, so the two records agree column for column)."""
    return write_magsys_table(
        covered_filter_sets(filter_sets),
        Path(model_root),
        model=MODEL,
        generator=GENERATOR,
    )


def process_node(
    node: Tuple[float, float, float],
    filter_sets: Dict[str, List[str]],
    av_pts: Sequence[float],
    work_dir: Path,
    with_bc: bool = False,
):
    """Fetch, read and band-integrate one node.  Returns (node,
    {facility: (n_av, n_filters, n_mu) float32 normalized intensities},
    mu, {facility: (n_av, n_filters) band fluxes}, bolometric record) --
    the last two empty / None unless ``with_bc``."""
    teff, logg, feh = node
    path = specint.fetch_specint(teff, logg, feh, work_dir)
    try:
        spec = specint.read_specint(path)
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    i_full = interpolate_intensities(spec)
    out, fluxes = {}, {}
    for fac, (svo, _mist) in _facility_filters(filter_sets).items():
        fdata = _filter_set_data(svo, FLUX_WEIGHTING)
        raw = band_intensities_raw(i_full, fdata, av_pts)
        out[fac] = normalize_band_intensities(raw).astype(np.float32)
        if with_bc:
            fluxes[fac] = band_fluxes(raw, spec["mu"])
    fbol = node_bolometric_flux(node, work_dir) if with_bc else None
    return node, out, spec["mu"], fluxes, fbol


# -------------------------------------------------------------------
# The intensities intermediate
# -------------------------------------------------------------------


def _intensities_frame(rows, av_pts):
    df = pd.DataFrame(rows)
    df.attrs["meta"] = {
        "intensities": INTENSITIES_TAG,
        "av": [float(a) for a in av_pts],
        "mu": "per row: the node's own emergent angles, ascending, 1 last",
        "shape": "(n_av, n_filters, n_mu), flattened, float32, normalized to mu = 1",
        "flux_weighting": FLUX_WEIGHTING,
        "Rv": RV,
    }
    return df


def load_intensities(model_root: Path | str, feh: float):
    """``(store, mus, filters)`` of one feh's intermediate: {(teff, logg,
    facility): (n_av, n_f, n_mu)}, {(teff, logg): that node's mu array}
    and {facility: filter names}; all empty when it does not exist."""
    path = intensities_path(model_root, feh)
    if not path.is_file():
        return {}, {}, {}
    df = pd.read_parquet(path)
    meta = df.attrs.get("meta", {})
    n_av = len(meta["av"])
    out, mus, filters = {}, {}, {}
    for r in df.itertuples(index=False):
        mu = np.asarray(r.mu, dtype=float)
        arr = np.asarray(r.intensity, dtype=np.float32)
        n_f = arr.size // (n_av * mu.size)
        key = (float(r.teff), float(r.logg))
        out[key + (r.facility,)] = arr.reshape(n_av, n_f, mu.size)
        mus[key] = mu
        filters[r.facility] = list(r.filters)
    return out, mus, filters


def load_band_fluxes(model_root: Path | str, feh: float):
    """``(fluxes, fbols)`` of one feh's intermediate, save_intensities'
    shapes; only the nodes that carry them (an intermediate written before
    the bolometric corrections existed carries none)."""
    path = intensities_path(model_root, feh)
    if not path.is_file():
        return {}, {}
    df = pd.read_parquet(path)
    if "band_flux" not in df.columns:
        return {}, {}
    n_av = len(df.attrs["meta"]["av"])
    fluxes, fbols = {}, {}
    for r in df.itertuples(index=False):
        if r.band_flux is None or not np.isfinite(r.fbol):
            continue
        key = (float(r.teff), float(r.logg))
        arr = np.asarray(r.band_flux, dtype=np.float64)
        fluxes[key + (r.facility,)] = arr.reshape(n_av, arr.size // n_av)
        fbols[key] = {
            "fbol": float(r.fbol),
            "tail_fraction": float(r.fbol_tail_fraction),
        }
    return fluxes, fbols


def save_intensities(
    model_root,
    feh,
    store,
    mus,
    av_pts,
    filters_by_fac,
    fluxes=None,
    fbols=None,
):
    """One row per (node, facility), the node's mu array with it -- and,
    for a node streamed for the bolometric corrections, its unnormalized
    band fluxes (``fluxes``: {(teff, logg, facility): (n_av, n_f)}) and its
    bolometric flux (``fbols``: {(teff, logg): specint.bolometric_flux
    record}).  A node without them (streamed for the limb darkening only,
    or a filled hole) stores None / NaN there."""
    fluxes = fluxes or {}
    fbols = fbols or {}
    rows = []
    for (teff, logg, fac), arr in sorted(store.items()):
        flux = fluxes.get((teff, logg, fac))
        fb = fbols.get((teff, logg))
        rows.append(
            {
                "teff": teff,
                "logg": logg,
                "feh": float(feh),
                "facility": fac,
                "filters": list(filters_by_fac[fac]),
                "mu": np.asarray(mus[(teff, logg)], dtype=float),
                "intensity": np.asarray(arr, dtype=np.float32).ravel(),
                "band_flux": (
                    None
                    if flux is None
                    else np.asarray(flux, dtype=np.float64).ravel()
                ),
                "fbol": np.nan if fb is None else float(fb["fbol"]),
                "fbol_tail_fraction": (
                    np.nan if fb is None else float(fb["tail_fraction"])
                ),
            }
        )
    path = intensities_path(model_root, feh)
    path.parent.mkdir(parents=True, exist_ok=True)
    _intensities_frame(rows, av_pts).to_parquet(
        path, compression="snappy", index=False
    )
    return path


# -------------------------------------------------------------------
# Holes: fill along teff from the two nearest available neighbours
# -------------------------------------------------------------------


def _resample_mu(arr, mu_from, mu_to):
    """``arr`` (n_av, n_f, n_mu_from) linearly resampled onto ``mu_to``."""
    out = np.empty(arr.shape[:2] + (mu_to.size,), dtype=np.float64)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            out[i, j] = np.interp(mu_to, mu_from, arr[i, j])
    return out


def fill_holes_along_teff(store, mus, teff_pts, logg_pts, facilities):
    """Add (teff, logg, fac) entries for the grid nodes `store` lacks
    whose two nearest TEFF neighbours at the same logg it holds, by linear
    interpolation of the intensities -- on the LOWER neighbour's mu grid,
    which the hole then carries in ``mus``, since no two nodes share
    their emergent angles.  Returns the padding records."""
    records = []
    teffs = sorted(float(t) for t in teff_pts)
    for logg in logg_pts:
        logg = float(logg)
        for k, teff in enumerate(teffs):
            if all((teff, logg, fac) in store for fac in facilities):
                continue
            lo = next(
                (
                    t
                    for t in reversed(teffs[:k])
                    if all((t, logg, f) in store for f in facilities)
                ),
                None,
            )
            hi = next(
                (
                    t
                    for t in teffs[k + 1 :]
                    if all((t, logg, f) in store for f in facilities)
                ),
                None,
            )
            if lo is None or hi is None:
                continue
            wgt = (teff - lo) / (hi - lo)
            mu = np.asarray(mus[(lo, logg)], dtype=float)
            for fac in facilities:
                a = np.asarray(store[(lo, logg, fac)], dtype=np.float64)
                b = _resample_mu(
                    store[(hi, logg, fac)], np.asarray(mus[(hi, logg)]), mu
                )
                store[(teff, logg, fac)] = ((1 - wgt) * a + wgt * b).astype(
                    np.float32
                )
            mus[(teff, logg)] = mu
            records.append(
                {"teff": teff, "logg": logg, "axis": "teff", "from": [lo, hi]}
            )
    return records


# -------------------------------------------------------------------
# The tables
# -------------------------------------------------------------------


def _existing_nodes(model_root, law, fac):
    path = ld_table_path(model_root, law, fac)
    if not path.is_file():
        return set()
    df = pd.read_parquet(path, columns=GRID_KEY_COLS)
    return {
        (float(t), float(g), float(z))
        for t, g, z in zip(df.teff, df.logg, df.feh)
    }


def _existing_bc_nodes(model_root, fac):
    path = bc_table_path(model_root, MODEL, fac)
    if not path.is_file():
        return set()
    df = pd.read_parquet(path, columns=GRID_KEY_COLS)
    return {
        (float(t), float(g), float(z))
        for t, g, z in zip(df.teff, df.logg, df.feh)
    }


def fill_bc_holes_along_teff(bcs, teff_pts, logg_pts, facilities):
    """The BC analog of fill_holes_along_teff: ``bcs`` {(teff, logg, fac):
    (n_av, n_f) BC} gains every grid node whose two nearest TEFF
    neighbours at the same logg it holds (all facilities), by linear
    interpolation of the BCs; returns the padding records."""
    records = []
    teffs = sorted(float(t) for t in teff_pts)

    def have(t, g):
        return all((t, g, f) in bcs for f in facilities)

    for logg in logg_pts:
        logg = float(logg)
        for k, teff in enumerate(teffs):
            if have(teff, logg):
                continue
            lo = next((t for t in reversed(teffs[:k]) if have(t, logg)), None)
            hi = next((t for t in teffs[k + 1 :] if have(t, logg)), None)
            if lo is None or hi is None:
                continue
            wgt = (teff - lo) / (hi - lo)
            for fac in facilities:
                bcs[(teff, logg, fac)] = (1 - wgt) * bcs[
                    (lo, logg, fac)
                ] + wgt * bcs[(hi, logg, fac)]
            records.append(
                {"teff": teff, "logg": logg, "axis": "teff", "from": [lo, hi]}
            )
    return records


def _write_bc_tables(
    model_root,
    feh,
    sets,
    svo_of,
    mist_of,
    av_pts,
    grid,
    fluxes,
    fbols,
    generated,
):
    """One [Fe/H]'s BC rows, every facility: BCs at every node holding a
    band flux and a bolometric flux, interior holes filled along Teff,
    merged into ``PHOENIX/BCs/{FACILITY}.bc.parquet``.  Returns the number
    of holes filled."""
    bcs = {}
    for (teff, logg, fac), flux in fluxes.items():
        if fac not in sets:
            continue
        fdata = _filter_set_data(tuple(svo_of[fac]), FLUX_WEIGHTING)
        bcs[(teff, logg, fac)] = bc_from_fluxes(
            flux, fbols[(teff, logg)]["fbol"], fdata
        )
    padding = fill_bc_holes_along_teff(
        bcs, grid["teff"], grid["logg"], list(sets)
    )
    for fac in sets:
        fdata = _filter_set_data(tuple(svo_of[fac]), FLUX_WEIGHTING)
        recs = []
        for (teff, logg, f2), bc in sorted(bcs.items()):
            if f2 != fac:
                continue
            for i, av in enumerate(av_pts):
                recs.append(
                    {
                        "teff": teff,
                        "logg": logg,
                        "feh": feh,
                        "alpha": 0.0,
                        "Av": av,
                        "Rv": RV,
                        **dict(zip(mist_of[fac], bc[i])),
                    }
                )
        if not recs:
            continue
        fmeta = {
            col: {
                "svo_id": svo_of[fac][j],
                "zeropoint_Fl_Vega": float(fdata["filter_zero_pts"][j]),
                "flux_weighting": fdata["filter_weightings"][j],
                "spectra": SPECTRA_TAG,
                "bolometric": BOLOMETRIC,
                "generator": GENERATOR,
                "generated": generated,
            }
            for j, col in enumerate(mist_of[fac])
        }
        path = bc_table_path(model_root, MODEL, fac)
        old_pad = (
            read_bc_meta(path).get("padding", []) if path.is_file() else []
        )
        new_pad = [
            {**p, "feh": feh}
            for p in padding
            if not any(
                q.get("teff") == p["teff"]
                and q.get("logg") == p["logg"]
                and q.get("feh") == feh
                for q in old_pad
            )
        ]
        df = pd.DataFrame(recs)
        write_bc_table(
            df[BC_PARAM_COLS + list(mist_of[fac])],
            path,
            fmeta,
            table_meta={
                "facility": fac,
                "mag_system": MAG_SYSTEM,
                "Rv": RV,
                "L0_W": L0,
                "model": MODEL,
                "bolometric": BOLOMETRIC,
                "padding": old_pad + new_pad,
            },
            allow_new_nodes=True,
        )
    return len(padding)


def generate_ld_tables(
    filter_sets: Dict[str, List[str]] = FILTER_SETS,
    model_root: Path | str = DEFAULT_MODEL_ROOT,
    grid: Optional[dict] = None,
    work_dir: Path | str = WORK_DIR_DEFAULT,
    n_workers: int = 4,
    fehs: Optional[Sequence[float]] = None,
    laws: Sequence[str] = LAWS,
    products: Sequence[str] = PRODUCTS,
) -> Dict[str, int]:
    """Stream the archive node by node and write the tables, one [Fe/H] at
    a time (checkpointed), computing only the nodes the tables lack.

    ``products``: ``"ld"`` writes the limb-darkening tables, ``"bc"`` the
    bolometric-correction tables (``PHOENIX/BCs/``) and their
    magnitude-system record; a node is streamed once for whichever of them
    lacks it.  A "bc" node also needs the archive's flux spectrum (its
    bolometric flux), so only the nodes BOTH archives hold get a BC row.

    Returns {feh: nodes processed}.
    """
    products = tuple(products)
    unknown = [p for p in products if p not in PRODUCTS]
    if unknown or not products:
        raise ValueError(
            f"products must be a subset of {PRODUCTS}: {products}"
        )
    model_root = Path(model_root)
    grid = read_grid_yaml(model_root) if grid is None else grid
    av_pts = [float(a) for a in grid["av"]]
    keep = covered_filters(filter_sets)
    sets = covered_filter_sets(filter_sets)
    dropped = {
        fac: [
            filter_sets[fac][i]
            for i in range(len(filter_sets[fac]))
            if i not in keep[fac]
        ]
        for fac in filter_sets
    }
    dropped = {k: v for k, v in dropped.items() if v}
    if dropped:
        print(
            f"filters outside the intensity spectra's 500-26000 A (no column): {dropped}"
        )
    alias_df = _load_alias_table()
    mist_of = {
        fac: [resolve_filter_name(f, alias_df, alias="MIST") for f in fs]
        for fac, fs in sets.items()
    }
    svo_of = {
        fac: [resolve_filter_name(f, alias_df, alias="SVO") for f in fs]
        for fac, fs in sets.items()
    }
    generated = datetime.date.today().isoformat()
    work_dir = Path(work_dir) / "phoenix_specint"
    listing_cache = model_root / MODEL / "LDs" / "cache"
    done = {}
    for feh in fehs if fehs is not None else grid["feh"]:
        feh = float(feh)
        available = specint.list_available(feh, cache_dir=listing_cache)
        wanted = [
            (float(t), float(g), feh)
            for g in grid["logg"]
            for t in grid["teff"]
            if (float(t), float(g)) in available
        ]
        store, mus, _f = load_intensities(model_root, feh)
        fluxes, fbols = load_band_fluxes(model_root, feh)
        todo_ld, todo_bc = [], []
        if "ld" in products:
            have = (
                set.intersection(
                    *[
                        _existing_nodes(model_root, law, fac)
                        for law in laws
                        for fac in sets
                    ]
                )
                if sets
                else set()
            )
            todo_ld = [
                n
                for n in wanted
                if n not in have
                and not all((n[0], n[1], fac) in store for fac in sets)
            ]
        if "bc" in products:
            with_flux = specint.list_available(
                feh, cache_dir=listing_cache, product="flux"
            )
            have_bc = (
                set.intersection(
                    *[_existing_bc_nodes(model_root, fac) for fac in sets]
                )
                if sets
                else set()
            )
            todo_bc = [
                n
                for n in wanted
                if (n[0], n[1]) in with_flux
                and n not in have_bc
                and not (
                    (n[0], n[1]) in fbols
                    and all((n[0], n[1], fac) in fluxes for fac in sets)
                )
            ]
        bc_set = set(todo_bc)
        todo = sorted(set(todo_ld) | bc_set)
        print(
            f"feh={feh:+.1f}: {len(available)} nodes in the archive, "
            f"{len(wanted)} on the grid, {len(todo)} to stream "
            f"({len(todo_ld)} for ld, {len(todo_bc)} for bc)"
        )
        if todo:
            with ProcessPoolExecutor(max_workers=max(1, n_workers)) as pool:
                futures = {
                    pool.submit(
                        process_node, n, sets, av_pts, work_dir, n in bc_set
                    ): n
                    for n in todo
                }
                for k, fut in enumerate(as_completed(futures), 1):
                    node, arrs, mu_node, flux_node, fbol = fut.result()
                    key = (node[0], node[1])
                    mus[key] = np.asarray(mu_node, dtype=float)
                    for fac, arr in arrs.items():
                        store[key + (fac,)] = arr
                    for fac, arr in flux_node.items():
                        fluxes[key + (fac,)] = arr
                    if fbol is not None:
                        fbols[key] = fbol
                    if k % 50 == 0 or k == len(todo):
                        print(
                            f"  feh={feh:+.1f}: {k}/{len(todo)} nodes",
                            flush=True,
                        )
        if not store:
            done[feh] = 0
            continue
        padding = fill_holes_along_teff(
            store, mus, grid["teff"], grid["logg"], list(sets)
        )
        save_intensities(
            model_root, feh, store, mus, av_pts, mist_of, fluxes, fbols
        )
        if "bc" in products and fluxes:
            n_pad = _write_bc_tables(
                model_root,
                feh,
                sets,
                svo_of,
                mist_of,
                av_pts,
                grid,
                fluxes,
                fbols,
                generated,
            )
            print(
                f"feh={feh:+.1f}: wrote {len(sets)} BC tables; {n_pad} hole(s) filled"
            )
        if "ld" not in products:
            done[feh] = len(todo)
            continue
        # fit and write every facility and law for this feh
        for fac in sets:
            n_f = len(mist_of[fac])
            recs = {law: [] for law in laws}
            for (teff, logg, f2), arr in store.items():
                if f2 != fac:
                    continue
                coefs = coefficients_from_intensities(mus[(teff, logg)], arr)
                for law in laws:
                    for i, av in enumerate(av_pts):
                        row = {
                            "teff": teff,
                            "logg": logg,
                            "feh": feh,
                            "alpha": 0.0,
                            "Av": av,
                            "Rv": RV,
                        }
                        for j in range(n_f):
                            for c, name in enumerate(COEFFICIENTS[law]):
                                row[
                                    coefficient_column(
                                        mist_of[fac][j], law, name
                                    )
                                ] = float(coefs[law][i, j, c])
                        recs[law].append(row)
            for law in laws:
                if not recs[law]:
                    continue
                df = pd.DataFrame(recs[law])
                cols = [c for c in df.columns if c not in BC_PARAM_COLS]
                fmeta = {}
                for j in range(n_f):
                    for name in COEFFICIENTS[law]:
                        fmeta[
                            coefficient_column(mist_of[fac][j], law, name)
                        ] = {
                            "svo_id": svo_of[fac][j],
                            "coefficient": name,
                            "law": law,
                            "flux_weighting": FLUX_WEIGHTING,
                            "intensities": INTENSITIES_TAG,
                            "method": METHOD,
                            "generator": GENERATOR,
                            "generated": generated,
                        }
                path = ld_table_path(model_root, law, fac)
                old_pad = (
                    read_bc_meta(path).get("padding", [])
                    if path.is_file()
                    else []
                )
                new_pad = [
                    {**p, "feh": feh}
                    for p in padding
                    if not any(
                        q.get("teff") == p["teff"]
                        and q.get("logg") == p["logg"]
                        and q.get("feh") == feh
                        for q in old_pad
                    )
                ]
                write_bc_table(
                    df[BC_PARAM_COLS + cols],
                    path,
                    fmeta,
                    table_meta={
                        "facility": fac,
                        "model": MODEL,
                        "law": law,
                        "Rv": RV,
                        "method": METHOD,
                        "intensities": INTENSITIES_TAG,
                        "padding": old_pad + new_pad,
                    },
                    allow_new_nodes=True,
                )
        done[feh] = len(todo)
        print(
            f"feh={feh:+.1f}: wrote {len(sets)} facilities x {len(laws)} laws; {len(padding)} hole(s) filled"
        )
    if "bc" in products:
        # A property of the columns' filter profiles, weighting and
        # zeropoints only, rewritten with them (the NextGen rule).
        write_bc_magsys_table(model_root, filter_sets)
    return done


def main(argv=None):
    p = argparse.ArgumentParser(
        description="PHOENIX limb-darkening and bolometric-correction tables"
    )
    p.add_argument(
        "--workers", type=int, default=int(os.environ.get("NSLOTS", 4))
    )
    p.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    p.add_argument("--work-dir", default=str(WORK_DIR_DEFAULT))
    p.add_argument("--feh", type=float, nargs="*", default=None)
    p.add_argument(
        "--products",
        nargs="+",
        choices=PRODUCTS,
        default=list(PRODUCTS),
        help="which tables to write (default: both)",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    done = generate_ld_tables(
        model_root=args.model_root,
        work_dir=args.work_dir,
        n_workers=args.workers,
        fehs=args.feh,
        products=args.products,
    )
    print(done)


if __name__ == "__main__":
    main()

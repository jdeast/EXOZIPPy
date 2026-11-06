"""
Build the limb-darkening tables from the PHOENIX-ACES specific
intensities (Husser et al. 2013), for the filters the NextGen BC tables
ship -- the LD analog of models/NextGen/generate_NextGen_BC_Tables.py.

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

Run (hours of streaming; a compute node with network, not a laptop):

    python -m exozippy.models.PHOENIX.generate_PHOENIX_LD_Tables --workers 32
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
    read_bc_meta,
    read_bc_table,
    resolve_filter_name,
    write_bc_table,
)
from exozippy.models.NextGen.bolometric_correction import _filter_set_data
from exozippy.models.NextGen.generate_NextGen_BC_Tables import FILTER_SETS
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
    """I_band(mu; Av) for one node and one filter set.

    ``i_full`` is interpolate_intensities' array; ``fdata`` the filter
    set's _filter_set_data (profiles already carry the detector
    weighting, ``wave_mask`` its slice of the common grid).  Returns
    (n_av, n_filters, n_mu), normalized so that mu = 1 is 1 per (Av,
    filter): only the shape of the profile matters for the fit.  One
    matrix product per Av: the trapezoid weights fold into the profiles.
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
    center = ib[:, :, -1][:, :, None]  # mu = 1 is last (ascending)
    return ib / np.where(center > 0, center, np.nan)


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


def process_node(
    node: Tuple[float, float, float],
    filter_sets: Dict[str, List[str]],
    av_pts: Sequence[float],
    work_dir: Path,
) -> Tuple[Tuple[float, float, float], Dict[str, np.ndarray], np.ndarray]:
    """Fetch, read and band-integrate one node.  Returns (node,
    {facility: (n_av, n_filters, n_mu) float32}, mu)."""
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
    out = {}
    for fac, (svo, _mist) in _facility_filters(filter_sets).items():
        fdata = _filter_set_data(svo, FLUX_WEIGHTING)
        out[fac] = band_intensities(i_full, fdata, av_pts).astype(np.float32)
    return node, out, spec["mu"]


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


def save_intensities(model_root, feh, store, mus, av_pts, filters_by_fac):
    """One row per (node, facility), the node's mu array with it."""
    rows = []
    for (teff, logg, fac), arr in sorted(store.items()):
        rows.append(
            {
                "teff": teff,
                "logg": logg,
                "feh": float(feh),
                "facility": fac,
                "filters": list(filters_by_fac[fac]),
                "mu": np.asarray(mus[(teff, logg)], dtype=float),
                "intensity": np.asarray(arr, dtype=np.float32).ravel(),
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


def generate_ld_tables(
    filter_sets: Dict[str, List[str]] = FILTER_SETS,
    model_root: Path | str = DEFAULT_MODEL_ROOT,
    grid: Optional[dict] = None,
    work_dir: Path | str = WORK_DIR_DEFAULT,
    n_workers: int = 4,
    fehs: Optional[Sequence[float]] = None,
    laws: Sequence[str] = LAWS,
) -> Dict[str, int]:
    """Stream the archive node by node and write the tables, one [Fe/H] at
    a time (checkpointed), computing only the nodes the tables lack.

    Returns {feh: nodes processed}.
    """
    model_root = Path(model_root)
    grid = read_grid_yaml(model_root) if grid is None else grid
    av_pts = [float(a) for a in grid["av"]]
    keep = covered_filters(filter_sets)
    fac_filters = _facility_filters(filter_sets)
    sets = {
        fac: [filter_sets[fac][i] for i in keep[fac]]
        for fac in filter_sets
        if keep[fac]
    }
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
            f"filters outside the intensity spectra's 500-26000 A (no LD column): {dropped}"
        )
    mist_of = {
        fac: [
            resolve_filter_name(f, _load_alias_table(), alias="MIST")
            for f in fs
        ]
        for fac, fs in sets.items()
    }
    svo_of = {
        fac: [
            resolve_filter_name(f, _load_alias_table(), alias="SVO")
            for f in fs
        ]
        for fac, fs in sets.items()
    }
    generated = datetime.date.today().isoformat()
    work_dir = Path(work_dir) / "phoenix_specint"
    done = {}
    for feh in fehs if fehs is not None else grid["feh"]:
        feh = float(feh)
        available = specint.list_available(
            feh, cache_dir=model_root / MODEL / "LDs" / "cache"
        )
        wanted = [
            (float(t), float(g), feh)
            for g in grid["logg"]
            for t in grid["teff"]
            if (float(t), float(g)) in available
        ]
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
        store, mus, _f = load_intensities(model_root, feh)
        todo = [
            n
            for n in wanted
            if n not in have
            and not all((n[0], n[1], fac) in store for fac in sets)
        ]
        print(
            f"feh={feh:+.1f}: {len(available)} nodes in the archive, {len(wanted)} on the grid, {len(todo)} to stream"
        )
        if todo:
            with ProcessPoolExecutor(max_workers=max(1, n_workers)) as pool:
                futures = {
                    pool.submit(process_node, n, sets, av_pts, work_dir): n
                    for n in todo
                }
                for k, fut in enumerate(as_completed(futures), 1):
                    node, arrs, mu_node = fut.result()
                    mus[(node[0], node[1])] = np.asarray(mu_node, dtype=float)
                    for fac, arr in arrs.items():
                        store[(node[0], node[1], fac)] = arr
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
        save_intensities(model_root, feh, store, mus, av_pts, mist_of)
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
    return done


def main(argv=None):
    p = argparse.ArgumentParser(description="PHOENIX limb-darkening tables")
    p.add_argument(
        "--workers", type=int, default=int(os.environ.get("NSLOTS", 4))
    )
    p.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    p.add_argument("--work-dir", default=str(WORK_DIR_DEFAULT))
    p.add_argument("--feh", type=float, nargs="*", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    done = generate_ld_tables(
        model_root=args.model_root,
        work_dir=args.work_dir,
        n_workers=args.workers,
        fehs=args.feh,
    )
    print(done)


if __name__ == "__main__":
    main()

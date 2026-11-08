"""The R = 3000 BT-NextGen grid the SED's spectral mode reads at runtime.

The SED's default path interpolates per-band bolometric corrections in
(Teff, logg, [Fe/H], A_V), tabulated at one extinction law (R_V = 3.1).
Its spectral mode (``components/sed/spectral_bc.py``) instead interpolates
the SPECTRUM in (Teff, logg, [Fe/H]), reddens it with any A_V and R_V, and
integrates the bands at every evaluation.  This module builds and reads the
grid it interpolates.

What the grid is.  Every node of ``NextGen.grid.yaml`` (Teff 2600-10000 K,
logg 0-5.5, [Fe/H] -4..+0.5), resampled from the R = 20000 step-1 spectra
(``generate_NextGen_BC_Tables.py``) onto log-uniform bins at R = 3000 over
the same 0.03-30 micron range: each bin holds the flux-conserving MEAN
F_lambda over the bin (the trapezoid integral of the R = 20000 spectrum
between the bin edges, divided by the bin width), stored as log10 in
float32.  A band flux is then sum_b Fbar_b exp(-tau(lambda_b)) Sint_fb, with
Sint_fb the integral of the filter's weighted profile over bin b; against
the R = 20000 integral that is within 0.43 mmag for every one of the 93
shipped filters at A_V 0-20, and within 1.4 mmag at R = 1000 (the
feasibility study, ``models/NextGen/README.md``, "The runtime spectral
grid").

Why log flux.  The SED interpolates log10 F trilinearly.  Interpolating
the linear flux is 2-3x worse than the BC tables for M dwarfs (off-node,
against held-out grid nodes); the log flux matches them.

Padding.  The grid nodes SVO serves no spectrum for are filled exactly as
step 2 fills them for a BC column (``generate_NextGen_BC_Tables.
_pad_spectrum``): an interior hole from the spectrum interpolated along
Teff, the logg 5.5 hot end from the nearest lower-gravity spectrum.  They
are listed in the metadata's ``padding``.  A node SVO lists but step 1 has
not processed raises.

Files: ``NextGen.R3000.logflux.npy`` (float32, shape (n_teff, n_logg,
n_feh, n_bin)) and ``NextGen.R3000.meta.json`` (axes, bin edges in
Angstrom, padding, provenance).  The .npy is memory-mapped, so a fit
touches only the wavelengths its bands cover and PTDE's workers share the
pages.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np  # before pyarrow, which the step-1 reader imports

from exozippy.models.NextGen.nextgen_spectra import WAVELENGTH_PTS

RESOLUTION = 3000
GRID_FILE = "NextGen.R3000.logflux.npy"
META_FILE = "NextGen.R3000.meta.json"
LOG10_FLOOR = -300.0  # log10 F where the R = 20000 flux is exactly 0


def bin_edges(resolution: int = RESOLUTION) -> np.ndarray:
    """Log-uniform bin edges (Angstrom) at `resolution` over the step-1
    wavelength grid, the last edge clipped to its end."""
    w = np.asarray(WAVELENGTH_PTS, dtype=float)
    k0 = np.ceil(np.log(w[0]) * resolution)
    k1 = np.floor(np.log(w[-1]) * resolution)
    return np.exp(np.arange(k0, k1 + 1) / resolution)


def _cumulative(w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral of y(w) along the last axis, from 0."""
    seg = 0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(w)
    out = np.zeros(y.shape[:-1] + (y.shape[-1],), dtype=float)
    np.cumsum(seg, axis=-1, out=out[..., 1:])
    return out


def bin_integrals(
    w: np.ndarray, y: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """The trapezoid integral of y(w) over each bin [edges[i], edges[i+1]]
    (exact for the piecewise-linear y the trapezoid rule assumes at the
    bin interior; linear interpolation of the cumulative at the edges)."""
    c = _cumulative(w, y)
    if c.ndim == 1:
        return np.diff(np.interp(edges, w, c))
    return np.diff(np.stack([np.interp(edges, w, ci) for ci in c]), axis=-1)


def bin_spectrum(flux: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Flux-conserving bin means of one R = 20000 spectrum."""
    w = np.asarray(WAVELENGTH_PTS, dtype=float)
    return bin_integrals(w, np.asarray(flux, dtype=float), edges) / np.diff(
        edges
    )


def build_runtime_grid(
    processed_path: Path,
    raw_path: Path,
    out_dir: Path,
    resolution: int = RESOLUTION,
) -> Path:
    """Write the runtime grid from the step-1 spectra; returns the .npy."""
    from exozippy.models.NextGen import generate_NextGen_BC_Tables as gen
    from exozippy.utilities import zenodo_assets

    grid = gen.read_grid_yaml()
    teffs = [float(t) for t in grid["teff"]]
    loggs = [float(g) for g in grid["logg"]]
    fehs = [float(z) for z in grid["feh"]]
    edges = bin_edges(resolution)
    unavailable = {
        tuple(round(x, 6) for x in n)
        for n in gen.unavailable_grid_nodes(grid, raw_path)
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / GRID_FILE
    arr = np.lib.format.open_memmap(
        npy,
        mode="w+",
        dtype=np.float32,
        shape=(len(teffs), len(loggs), len(fehs), len(edges) - 1),
    )
    padding, alphas = [], {}
    for k, feh in enumerate(fehs):
        spec = gen._index_spectra(
            gen.load_processed_spectra(feh, processed_path)
        )
        for i, teff in enumerate(teffs):
            for j, logg in enumerate(loggs):
                if (teff, logg) in spec.index:
                    row = spec.loc[(teff, logg)]
                    alpha = float(row["alpha"])
                    flux = np.asarray(row["flux"], dtype=float)
                else:
                    if (
                        round(teff, 6),
                        round(logg, 6),
                        round(feh, 6),
                    ) not in unavailable:
                        raise FileNotFoundError(
                            f"No step-1 spectrum for teff={teff}, logg={logg}, "
                            f"feh={feh} in {processed_path}, and SVO lists one; "
                            f"finish step 1 first."
                        )
                    r, record = gen._pad_spectrum(
                        spec, grid, teff, logg, feh, processed_path
                    )
                    alpha, flux = (
                        r["alpha"],
                        np.asarray(r["flux"], dtype=float),
                    )
                    padding.append(record)
                fbar = bin_spectrum(flux, edges)
                with np.errstate(divide="ignore"):
                    lg = np.where(fbar > 0, np.log10(fbar), LOG10_FLOOR)
                arr[i, j, k] = lg.astype(np.float32)
                alphas[f"{teff:g},{logg:g},{feh:g}"] = alpha
        print(
            f"feh={feh:+.1f}: {len(teffs) * len(loggs)} nodes binned",
            flush=True,
        )
        del spec
    arr.flush()
    del arr
    meta = {
        "model": "BT-NextGen (AGSS2009)",
        "resolution": resolution,
        "quantity": "log10 of the bin-mean F_lambda, erg/s/cm^2/A",
        "log10_floor": LOG10_FLOOR,
        "axes": {"teff": teffs, "logg": loggs, "feh": fehs},
        "bin_edges_A": edges.tolist(),
        "padding": padding,
        "alpha": alphas,
        "source": "R = 20000 step-1 spectra, Zenodo "
        + zenodo_assets.record("nextgen_spectra_r20000").doi,
        "generator": "exozippy.models.NextGen.runtime_spectra.build_runtime_grid",
        "generated": datetime.date.today().isoformat(),
    }
    (out_dir / META_FILE).write_text(json.dumps(meta))
    return npy


# ---------------------------------------------------------------- reading

RECORD = "nextgen_runtime_spectra"  # utilities/zenodo_assets.py
SUBDIR = Path("NextGen") / "runtime"


def runtime_dir(model_root=None) -> Path:
    from exozippy.components.sed.bc_grid import DEFAULT_MODEL_ROOT

    root = DEFAULT_MODEL_ROOT if model_root is None else Path(model_root)
    return root / SUBDIR


def _assets():
    from exozippy.utilities import zenodo_assets

    return {
        name: meta
        for name, meta in zenodo_assets.assets(RECORD).items()
        if name in (GRID_FILE, META_FILE)
    }


def ensure_runtime_grid(model_root=None) -> Path:
    """The directory holding the runtime grid, fetched on first use.

    Only into the DEFAULT model root (the BC tables' rule): an explicit
    ``model_root`` is the caller's own tree, used as given and never
    fetched into, so a grid missing there raises naming the path.
    """
    from exozippy.components.sed.bc_grid import DEFAULT_MODEL_ROOT
    from exozippy.utilities.zenodo import fetch_assets

    d = runtime_dir(model_root)
    if model_root is not None and Path(model_root) != DEFAULT_MODEL_ROOT:
        missing = [f for f in (GRID_FILE, META_FILE) if not (d / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"The SED's fitted-R_V mode reads the R = {RESOLUTION} "
                f"runtime grid, and {d} lacks {missing}. An explicit "
                f"model_root is used as given; copy the published grid "
                f"(record '{RECORD}') there, or drop model_root to fetch it."
            )
        return d
    fetch_assets(_assets(), d)
    return d


def fetch_all() -> list[Path]:
    """Owner hook for the record (``zenodo_assets.OWNERS``)."""
    d = ensure_runtime_grid()
    return [d / GRID_FILE, d / META_FILE]


class RuntimeGrid:
    """The runtime grid, memory-mapped: ``axes`` (teff, logg, feh lists),
    ``edges`` (bin edges, Angstrom), ``logflux`` (n_teff, n_logg, n_feh,
    n_bin float32 memmap) and the full ``meta``."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / META_FILE).read_text())
        self.axes = {
            k: np.asarray(v, float) for k, v in self.meta["axes"].items()
        }
        self.edges = np.asarray(self.meta["bin_edges_A"], float)
        self.logflux = np.load(self.directory / GRID_FILE, mmap_mode="r")
        want = (
            len(self.axes["teff"]),
            len(self.axes["logg"]),
            len(self.axes["feh"]),
            len(self.edges) - 1,
        )
        if self.logflux.shape != want:
            raise ValueError(
                f"{self.directory / GRID_FILE} has shape {self.logflux.shape}, "
                f"but its metadata describes {want}."
            )


_GRIDS: dict = {}


def load_runtime_grid(model_root=None) -> RuntimeGrid:
    """The runtime grid under ``model_root`` (fetched into the default
    root on first use), cached per directory."""
    d = ensure_runtime_grid(model_root)
    if d not in _GRIDS:
        _GRIDS[d] = RuntimeGrid(d)
    return _GRIDS[d]

"""
Bolometric-correction table generator.

Builds BC tables for arbitrary SVO filters by integrating the shipped
model spectra (e.g. NextGen, R=150, on the common wavelength grid)
through SVO filter profiles, and writes them in the same per-feh file
format the shipped 2MASS/GAIA/WISE tables use, so bc_grid.py loads them
transparently. This resolves the "future implementation will automate
this step" TODO in bc_grid._collect_facility_files.

Conventions
-----------
* BC_X = M_bol - M_X with M_bol from sigma*Teff^4 and the IAU 2015
  bolometric zero point (L0 = 3.0128e35 erg/s), matching how the SED
  component consumes the BC (star.physics.calc_luminosity uses
  sigma*T^4).
* Band-averaged flux density is photon-weighted:
  <f> = int(f S lambda dlam) / int(S lambda dlam).
* Vega zeropoints from SVO (specified value when quoted, else SVO's
  calculated one), via the Filter class.
* Extinction IS applied along the Av axis:
  tau(lam) = ext(lam)/ext(0.55um) * Av / 1.086 (models/extinction_law.ascii),
  so BC(Av) = M_bol(unextincted) - M_X(extincted). NOTE: the shipped
  2MASS/GAIA/WISE tables do NOT vary with Av (extinction appears to be
  missing there); tables generated here do.

Accuracy caveat: the shipped R=150 spectra reproduce the original
2MASS/GAIA BC tables only to ~0.01-0.04 mag (those were evidently
computed from full-resolution spectra). Fine for broad-band flux
constraints (e.g. the mulensing zeropoint prior is 0.2 mag); revisit if
percent-level absolute calibration is needed.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .bc_grid import (
    DEFAULT_BC_ROOT,
    peek_grid_axes,
    resolve_filter_name,
    facility_from_svo_name,
    _load_alias_table,
    _read_single_bc_file,
)
from .filters.filter import Filter

logger = logging.getLogger(__name__)

try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()

SIGMA_SB = 5.670374419e-5        # erg s^-1 cm^-2 K^-4
L0 = 3.0128e35                   # IAU 2015 resolution B2, erg/s
PC_CM = 3.0856775814913673e18
F0_10PC = L0 / (4.0 * np.pi * (10.0 * PC_CM) ** 2)   # erg s^-1 cm^-2
V_BAND_MICRON = 0.55

# Alpha-abundance fallback order when a grid point has no alpha=0
# spectrum (mirrors models/NextGen/plot.py ALPHA_GRID_PTS).
ALPHA_FALLBACK = (0.0, 0.2, -0.2, 0.4, 0.6)

_MODEL_DATA_URLS = {
    "NextGen": {
        "NextGen.spectra.csv":    "https://zenodo.org/records/20547997/files/NextGen.spectra.csv?download=1",
        "NextGen.wavelength.csv": "https://zenodo.org/records/20547997/files/NextGen.wavelength.csv?download=1",
    }
}


def ensure_model_data(model: str, bc_root: Path | str = DEFAULT_BC_ROOT):
    """Download large model data files from Zenodo if not present locally."""
    urls = _MODEL_DATA_URLS.get(model, {})
    model_dir = Path(bc_root) / model
    for filename, url in urls.items():
        dest = model_dir / filename
        if not dest.exists():
            logger.info(f"Downloading {filename} from Zenodo...")
            urllib.request.urlretrieve(url, dest)
            logger.info(f"Saved {filename} to {dest}")


def _load_spectra(model: str, bc_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load the model spectra table and its wavelength grid (Angstrom).
    The flux column is kept as JSON strings; _select_spectrum parses
    only the rows actually used (a large fraction of the file may be
    alternate-alpha rows that are never touched).
    """
    model_dir = Path(bc_root) / model
    df_spec = pd.read_csv(model_dir / f"{model}.spectra.csv")
    df_wave = pd.read_csv(model_dir / f"{model}.wavelength.csv")
    return df_spec, df_wave["wavelength_angstrom"].values.astype(float)


def _unit_optical_depth(wave_ang: np.ndarray) -> np.ndarray:
    """Optical depth per magnitude of Av on the spectra wavelength grid."""
    ext = pd.read_csv(
        current_dir / "models" / "extinction_law.ascii",
        names=["wavelength", "extinction"], delimiter=" ",
        index_col=False, skipinitialspace=True,
    )
    from scipy import interpolate
    f = interpolate.interp1d(
        ext["wavelength"], ext["extinction"], fill_value="extrapolate")
    wave_micron = wave_ang * 1e-4
    return (f(wave_micron) / f(V_BAND_MICRON)) / 1.086


def _select_spectrum(df_spec, teff, logg, feh):
    """Spectrum at a grid node, with the alpha fallback order."""
    for alpha in ALPHA_FALLBACK:
        rows = df_spec[
            (df_spec.teff == teff) & (df_spec.logg == logg)
            & (df_spec.feh == feh) & (df_spec.alpha == alpha)
        ]
        if len(rows) > 0:
            flux = rows.iloc[0].flux
            if isinstance(flux, str):
                flux = np.array(json.loads(flux))
            return flux
    return None


def _vega_zeropoint(filt: Filter) -> float:
    """Vega F_lambda zeropoint: specified when quoted, else calculated."""
    zp = getattr(filt, "Zp_Spec_Fl_Vega", None) or getattr(filt, "Zp_Calc_Fl_Vega", None)
    if zp is None:
        raise ValueError(
            f"No Vega F_lambda zeropoint available for {filt.filterID}.")
    return float(zp)


def make_bc_tables(
    svo_filter_ids: Sequence[str],
    model: str = "NextGen",
    bc_root: Path | str = DEFAULT_BC_ROOT,
) -> List[Path]:
    """
    Generate BC tables for the given SVO filter IDs (grouped per facility)
    on exactly the (teff, logg, feh, Av) axes of the shipped tables, and
    write them under {bc_root}/{model}/BCs/{FACILITY}/feh*_afe+0.0.{FACILITY}.

    Returns the list of files written.
    """
    bc_root = Path(bc_root)
    ensure_model_data(model, bc_root)

    axes = peek_grid_axes(model=model, bc_root=bc_root)
    teff_pts = axes["teff_pts"]
    logg_pts = axes["logg_pts"]
    feh_pts = axes["feh_pts"]
    av_pts = axes["av_pts"]

    df_spec, wave_ang = _load_spectra(model, bc_root)
    tau_unit = _unit_optical_depth(wave_ang)
    # (n_av, n_wave) attenuation factors
    atten = np.exp(-np.outer(av_pts, tau_unit))

    alias_df = _load_alias_table()

    # group by facility, keep the BC-table column names (MIST convention).
    # resolve_filter_name synthesizes a column name for filters with no
    # alias-table entry, matching what build_bc_grid looks up later.
    by_facility: Dict[str, List[tuple[str, str]]] = {}
    for svo_id in svo_filter_ids:
        fac = facility_from_svo_name(svo_id)
        col = resolve_filter_name(svo_id, alias_df, alias="MIST")
        by_facility.setdefault(fac, []).append((svo_id, col))

    written: List[Path] = []
    for fac, items in by_facility.items():
        # filter transmissions on the spectra grid + zeropoints
        S = []
        zps = []
        for svo_id, _ in items:
            filt = Filter(svo_id)
            wf, tf = filt.ProcessedFilterCurve
            S.append(np.interp(wave_ang, wf, tf, left=0.0, right=0.0))
            zps.append(_vega_zeropoint(filt))
        S = np.array(S)                     # (n_filt, n_wave)
        zps = np.array(zps)                 # (n_filt,)
        # photon-weighted band normalization: int(S lambda dlam)
        S_norm = np.trapezoid(S * wave_ang, wave_ang, axis=1)

        out_dir = bc_root / model / "BCs" / fac
        out_dir.mkdir(parents=True, exist_ok=True)

        new_cols = [c for _, c in items]
        for feh in feh_pts:
            recs = []
            for teff in teff_pts:
                mbol_term = SIGMA_SB * teff ** 4 / F0_10PC
                for logg in logg_pts:
                    spec = _select_spectrum(df_spec, teff, logg, feh)
                    if spec is None:
                        raise ValueError(
                            f"No {model} spectrum for teff={teff}, "
                            f"logg={logg}, feh={feh} (any alpha)."
                        )
                    # (n_av, n_filt) band-averaged flux densities
                    fmean = np.trapezoid(
                        (atten * spec)[:, None, :] * (S * wave_ang)[None, :, :],
                        wave_ang, axis=2,
                    ) / S_norm[None, :]
                    # BC = M_bol - M_X ; the (R/d)^2 factor cancels
                    bc = 2.5 * np.log10(fmean / zps[None, :] / mbol_term)
                    for i_av, av in enumerate(av_pts):
                        recs.append((float(teff), float(logg), float(av),
                                     *bc[i_av]))
            df_new = pd.DataFrame(recs, columns=["teff", "logg", "Av"] + new_cols)

            fname = f"feh{feh:+.1f}_afe+0.0.{fac}"
            path = out_dir / fname

            # Merge into an existing facility file WITHOUT touching its
            # other columns (they may come from a different pipeline,
            # e.g. the original full-resolution BC computation).
            keep_old_cols: List[str] = []
            if path.exists():
                df_old, old_cols = _read_single_bc_file(path)
                keep_old_cols = [c for c in old_cols if c not in new_cols]
                if keep_old_cols:
                    df_new = df_new.merge(
                        df_old[["teff", "logg", "Av"] + keep_old_cols],
                        on=["teff", "logg", "Av"], how="left",
                    )
                    if df_new[keep_old_cols].isna().any().any():
                        raise ValueError(
                            f"Grid-axis mismatch while merging new BC "
                            f"columns into existing {path}."
                        )

            out_cols = keep_old_cols + new_cols
            col_hdr = "".join(f"{c:>21s}" for c in out_cols)
            n_spectra = len(teff_pts) * len(logg_pts)
            with open(path, "w") as f:
                f.write(f"# {model}\n")
                f.write(f"# {fac} (Vega)\n")
                f.write("#  filters spectra  num Av  num Rv version\n")
                f.write(f"#       {len(out_cols):2d}   {n_spectra:4d}     "
                        f"{len(av_pts):3d}       1       1\n")
                f.write(f"# lgTef  logg  Fe_H a_Fe   Av   Rv{col_hdr}\n")
                # plain arrays: itertuples would mangle column names that
                # start with a digit (e.g. 2MASS_J)
                keys = df_new[["teff", "logg", "Av"]].values
                vals = df_new[out_cols].values
                for (teff_r, logg_r, av_r), bcs in zip(keys, vals):
                    bc_str = "".join(f"{b:21.4f}" for b in bcs)
                    f.write(f"{np.log10(teff_r):.5f} {logg_r:5.2f} "
                            f"{feh:5.2f} {0.0:4.1f} "
                            f"{av_r:4.2f} {3.10:4.2f}{bc_str}\n")
            written.append(path)
            logger.info(f"Wrote {path}")

    return written


def generate_missing_facility(
    facility: str,
    svo_names: Sequence[str],
    model: str,
    bc_root: Path | str,
) -> bool:
    """
    Auto-generation hook used by bc_grid.build_bc_grid when a facility's
    BC directory is missing: build tables for the requested SVO filters.
    Returns True on success.
    """
    wanted = [s for s in svo_names if facility_from_svo_name(s) == facility]
    if not wanted:
        return False
    logger.warning(
        f"BC tables for facility '{facility}' not found; generating them "
        f"now from the {model} spectra for {wanted} (one-time cost)."
    )
    try:
        make_bc_tables(wanted, model=model, bc_root=bc_root)
        return True
    except Exception as e:
        logger.error(f"BC auto-generation for '{facility}' failed: {e}")
        return False

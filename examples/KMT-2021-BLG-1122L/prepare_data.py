"""Convert the KMTNet pySIS light curves of KMT-2021-BLG-1122 into the
five-column (HJD, dflux, dflux_err, mag, mag_err) files the config reads with
`data_format: dia`.

Source: https://kmtnet.kasi.re.kr/~ulens/event/2021/view.php?event=KMT-2021-BLG-1122
(field BLG14; files data/KB211122/pysis/KMT{A,C,S}14_{I,V}.pysis, downloaded
2026-10-06).  I files: HJD-2450000, Delta_flux, flux_err, mag, mag_err, fwhm,
sky, secz.  V files carry no mag columns: HJD-2450000, Delta_flux, flux_err,
fwhm, sky, secz.

WHY DIFFERENCE FLUX, NOT THE MAG COLUMN.  pySIS's mag column satisfies
mag = 28.0 - 2.5*log10(1584.9 - Delta_flux) at every site to 3e-5 mag:
the reference flux 1584.9 is 20.0 mag at zp 28, a FIXED pipeline reference,
not this star's baseline.  A magnitude file built from it therefore has a
fictitious ~20.0 mag baseline; the source flux (set by the size of the
brightening) is real, but the total flux and the blend are not, and the
q_source <= 2 bound is applied against the fictitious total (the first
acceptance fit sat at q_source 1.1-1.8).  So the files keep the native
difference flux, and the config gives each instrument an explicit
`reference_mag` large enough that the bound never binds; the reference
only shifts the blend, which nothing reads (sed_constrains_blend is off).

CUTS: rows whose fwhm <= 0 (pySIS's flag for a failed PSF fit -- nearly all
of the far-from-event points that sat 1-9 mag above baseline, one at 11.4
mag with sigma 0.16), and rows with a non-finite or non-positive error.
Difference flux keeps the faint epochs that the magnitude conversion had to
drop (negative total flux).  Han et al. 2023 renormalized the error bars
(Yee et al. 2012); we leave that to the fit's err_scale.
"""

import numpy as np

# The reference each instrument is read against (config: reference_mag).
REF_MAG = {"I": 16.0, "V": 19.0}

for band in ("I", "V"):
    for site in ("A", "C", "S"):
        src = f"KMT{site}14_{band}.pysis"
        dat = np.loadtxt(src, comments="#")
        hjd, dflux, ferr = dat[:, 0], dat[:, 1], dat[:, 2]
        fwhm = dat[:, 5] if band == "I" else dat[:, 3]
        ok = np.isfinite(dflux) & np.isfinite(ferr) & (ferr > 0) & (fwhm > 0)
        # Five columns, as the dia reader requires even when the config gives
        # an explicit reference (a reader bug, filed): mag and mag_err are
        # computed from THAT reference (REF_MAG below, the same value the
        # config states), so the file is self-consistent and the reader,
        # which uses the override and never reads mag, sees no conflict.
        ref = 10.0 ** (-0.4 * (REF_MAG[band] - 28.0))
        tot = ref - dflux[ok]
        mag = 28.0 - 2.5 * np.log10(tot)
        merr = 2.5 / np.log(10.0) * ferr[ok] / tot
        out = np.column_stack(
            [hjd[ok] + 2450000.0, dflux[ok], ferr[ok], mag, merr]
        )
        dest = f"n20210603.{band}.KMT{site}14.dia"
        np.savetxt(dest, out, fmt="%.5f %.4f %.4f %.6f %.6f")
        print(
            f"{src}: {len(dat)} rows -> {dest}: {ok.sum()} kept "
            f"({(fwhm <= 0).sum()} failed-PSF rows dropped); "
            f"min dflux {out[:, 1].min():.0f} (brightest), median err {np.median(out[:, 2]):.1f}"
        )

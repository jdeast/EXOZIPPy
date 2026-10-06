"""Convert the KMTNet pySIS light curves of KMT-2021-BLG-1122 into the
three-column (HJD, I, sigma_I) files the config reads.

Source: https://kmtnet.kasi.re.kr/~ulens/event/2021/view.php?event=KMT-2021-BLG-1122
(field BLG14, star 015732; files data/KB211122/pysis/KMT{A,C,S}14_I.pysis,
downloaded 2026-10-06).  pySIS columns: HJD-2450000, Delta_flux, flux_err,
mag, mag_err, fwhm, sky, secz.  We keep (HJD, mag, mag_err) with the full
HJD, as examples/KMT-2019-BLG-1806 does, and DROP rows whose magnitude is
undefined (negative flux in the pipeline's own conversion) or whose
mag_err is not finite.  Han et al. 2023 used the I-band pySIS curves with
error bars renormalized per Yee et al. 2012; we leave the renormalization
to the fit's own err_scale (the factors are not printed in the paper).
"""

import numpy as np

for site in ("A", "C", "S"):
    src = f"KMT{site}14_I.pysis"
    dat = np.loadtxt(src, comments="#")
    hjd, dflux, ferr, mag, merr = (
        dat[:, 0],
        dat[:, 1],
        dat[:, 2],
        dat[:, 3],
        dat[:, 4],
    )
    ok = np.isfinite(mag) & np.isfinite(merr) & (merr > 0) & (merr < 5)
    out = np.column_stack([hjd[ok] + 2450000.0, mag[ok], merr[ok]])
    dest = f"n20210603.I.KMT{site}14.pys"
    np.savetxt(dest, out, fmt="%.5f %.4f %.4f")
    print(
        f"{src}: {len(dat)} rows -> {dest}: {ok.sum()} kept ({(~ok).sum()} dropped); "
        f"HJD {out[:, 0].min():.2f}-{out[:, 0].max():.2f}, I median {np.median(out[:, 1]):.2f}, "
        f"sigma median {np.median(out[:, 2]):.3f}"
    )

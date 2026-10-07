"""Generate the EXOZIPPy input files for OGLE-2016-BLG-1045.

The four `n20*.OB161045.txt` files are copied verbatim from MMEXOFAST
(`source/mmexofast/data/OB161045/`) and kept here as the provenance record.
They need three fixes before EXOZIPPy can read them, and only the third is
an approximation.

1. THE KMT FLUX SIGN IS NATIVE, NOT A MISTAKE
---------------------------------------------
KMTNet pySIS reports a DIFFERENCE flux that becomes MORE NEGATIVE as the
star brightens.  That is the pipeline's own convention, not something done
to these files: MulensModel ships an independently sourced native KMT set
(`data/photometry_files/KB180003/*.pysis`, five columns
`HJD' dflux dflux_err mag mag_err`) whose brightest epoch, at mag 11.579,
carries dflux = -3,699,974.  Both events also share a baseline dflux near
-337, which is the pipeline's zero offset.

So the sign is flipped here only to put brightening in the positive
direction that EXOZIPPy expects.  Nothing is being corrected.

2. THE SCALE
------------
Raw DIA counts put f_source near 1.8e4, against f_source's default bound of
[0, 1000] in `components/mulensing/defaults.yaml` -- a start ~150,000 nats
inside the wall.  Each file is divided by a round power of ten.  A
microlensing flux unit is arbitrary with no `sed:` block to tie it to a
magnitude, so this costs nothing; DC2018_128 ships normalized flux for the
same reason.

3. DIFFERENCE FLUX -> TOTAL FLUX: THE APPROXIMATION
---------------------------------------------------
This is the one to read before using any fit from this example.

EXOZIPPy's flux model is

    f_total  = 10**log_f_total            (strictly positive)
    f_source = f_total * q_source
    f_blend  = f_total * (1 - q_source)   with q_source bounded [0, 2]

which describes a TOTAL-flux light curve.  A difference-flux curve has its
baseline at zero, so f_total = f_source + f_blend ~ 0 while f_source is
large: at the published trajectory KMTC needs f_source = 1.784 and
f_blend = -1.774, hence q_source = 181 against an upper bound of 2.  The
bound is not the problem and must not be raised -- it encodes "the blend is
at most as negative as the total flux", which is true of the total-flux
curves it was written for, and widening it would loosen the prior on every
other microlensing fit for the sake of this one file.

The right fix is to add the DIA reference flux back per site, recovering
the true total flux.  THAT NUMBER IS NOT AVAILABLE HERE.  It is recoverable
exactly from a native KMT file, because `mag` and `dflux` together solve

    mag = zp - 2.5*log10(ref - dflux)

for `ref` (verified on KB180003: ref = 1584.9 at zp = 28.000 for all three
sites, residual rms 2.8e-5 mag).  But these files are a three-column
reprocessing with the `mag` column stripped, so the information is gone,
and the constant is event-specific -- KB180003's reference happens to sit
at exactly mag 20.000, while OB161045 needs roughly 17,500.

So each curve is instead OFFSET TO ZERO BLENDING: the linear flux problem
is solved at the published trajectory, and -f_blend is added, which places
the baseline at the source flux and sets the blend to zero by construction.

WHAT THAT COSTS, explicitly:

  - The blending is IMPOSED, not fitted.  Any blend flux this example
    reports is an artifact of this offset.
  - Shin et al. 2018 derive theta_* from a CMD analysis that depends on the
    source flux, so a fit here will NOT reproduce their theta_E = 0.244 mas
    or M_L = 0.08 M_sun.  The trajectory (t_0, u_0, t_E, rho) is the part
    that remains meaningful.
  - This example is therefore a finite-source DEMONSTRATION, not a
    reproduction of the published result.

TO REMOVE THE APPROXIMATION: obtain the native five-column KMT pySIS files
for OB161045 (and the Auckland reference flux), set REFERENCE_FLUX below,
and the offset step turns itself off.

Run from this directory:

    python convert_data.py
"""

import numpy as np

# Published trajectory -- Shin et al. 2018, ApJ 863, 23, Table 2, (-, +).
PUBLISHED = {
    "t_0": 2457559.201,
    "u_0": -0.01308,
    "t_E": 11.981,
    "rho": 0.03186,
}

# Per-site DIA reference flux, in each file's RAW units, if it ever becomes
# known.  Setting a value here switches that file from the zero-blending
# offset to the exact conversion total = ref - dflux.
REFERENCE_FLUX = {
    "KMTC": None,
    "KMTS": None,
    "KMTA": None,
    "Auckland": None,
}

# (source file, output file, site, flux sign, divisor)
#   sign -1 for KMT: pySIS difference flux is negative for brightening.
#   sign +1 for Auckland: microFUN photometry is already positive.
FILES = [
    (
        "n20160221.I.KMTC.OB161045.txt",
        "KMTC.I.OB161045.dat",
        "KMTC",
        -1.0,
        1e4,
    ),
    (
        "n20160222.I.KMTS.OB161045.txt",
        "KMTS.I.OB161045.dat",
        "KMTS",
        -1.0,
        1e4,
    ),
    (
        "n20160221.I.KMTA.OB161045.txt",
        "KMTA.I.OB161045.dat",
        "KMTA",
        -1.0,
        1e4,
    ),
    (
        "n20160619.R.Auckland.OB161045.txt",
        "Auckland.R.OB161045.dat",
        "Auckland",
        +1.0,
        1e2,
    ),
]

HEADER = """OGLE-2016-BLG-1045 -- {site}
Generated by convert_data.py from {src}
flux sign {sign:+.0f}; divided by {div:g}; {offset_note}
Columns: HJD (full Julian Date), flux, flux error.

{warning}"""

ZERO_BLEND_WARNING = """WARNING -- BLENDING IS IMPOSED, NOT FITTED.
No DIA reference flux is available for this site, so this curve was offset
so that its blend flux is zero at the published trajectory.  A fit to it
will NOT reproduce Shin et al. 2018's theta_E or lens mass, because their
theta_* comes from a CMD analysis that depends on the source flux.  The
trajectory (t_0, u_0, t_E, rho) remains meaningful.
SOLUTION: supply the native five-column KMT pySIS file (which carries the
mag column, from which the reference flux follows exactly) and set
REFERENCE_FLUX in convert_data.py."""


def _magnification(t):
    """FSPL magnification at the published trajectory."""
    import MulensModel as mm

    model = mm.Model(PUBLISHED)
    model.set_magnification_methods(
        [
            PUBLISHED["t_0"] - 1.0,
            "finite_source_uniform_Gould94",
            PUBLISHED["t_0"] + 1.0,
        ]
    )
    return model.get_magnification(t)


def _linear_flux(flux, err, amp):
    """Weighted least squares for (f_source, f_blend) in F = f_s*A + f_b."""
    w = 1.0 / err**2
    m = np.vstack([amp, np.ones_like(amp)]).T
    return np.linalg.solve((m.T * w) @ m, (m.T * w) @ flux)


def main():
    warned = False
    for src, out, site, sign, div in FILES:
        d = np.loadtxt(src)
        order = np.argsort(d[:, 0])
        t, f, e = d[order, 0], sign * d[order, 1], d[order, 2]

        ref = REFERENCE_FLUX.get(site)
        if ref is not None:
            # Exact: pySIS stores (reference - total), so with the sign
            # already flipped above, total = ref + f.
            f = f + ref
            note = "reference flux %g added (EXACT)" % ref
            warning = "Total flux, reconstructed from the DIA reference flux."
        else:
            amp = _magnification(t)
            f_s, f_b = _linear_flux(f, e, amp)
            f = f - f_b
            note = "offset %+.1f to ZERO BLENDING (approximate)" % (-f_b)
            warning = ZERO_BLEND_WARNING
            warned = True

        f, e = f / div, e / div
        np.savetxt(
            out,
            np.column_stack([t, f, e]),
            fmt=["%.6f", "%.8f", "%.8f"],
            header=HEADER.format(
                site=site,
                src=src,
                sign=sign,
                div=div,
                offset_note=note,
                warning=warning,
            ),
        )
        print("%-30s -> %-26s N=%4d  %s" % (src, out, len(t), note))

    if warned:
        print("")
        print("*" * 72)
        print("WARNING: at least one light curve was offset to ZERO BLENDING.")
        print("Its blending is imposed, not fitted.  A fit will not reproduce")
        print("Shin+18's theta_E or lens mass.  SOLUTION: supply the native")
        print("five-column KMT pySIS files and set REFERENCE_FLUX above.")
        print("*" * 72)


if __name__ == "__main__":
    main()

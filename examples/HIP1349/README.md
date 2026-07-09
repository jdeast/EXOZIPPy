# HIP 1349: real Hipparcos epoch astrometry

HIP 1349 (HD 1273, G8V, 23 pc) has an orbital (O-type) solution in the
Hipparcos Double and Multiple Systems Annex: a 411-day photocenter orbit
with a0 = 19.9 mas.  This example fits the REAL 1997-consortium
Intermediate Astrometric Data (IAD) -- 1-D along-scan abscissae with scan
angles, the same data structure as Gaia epoch astrometry -- using the
`astrometryinstrument` component in `mode: gaia`.

```
poetry run python get_hipparcos_iad.py    # (re)download + reconstruct
poetry run exozippy hip1349.yaml
```

`get_hipparcos_iad.py` downloads the IAD from the ESA Hipparcos tools
service, reconstructs the full along-scan coordinate relative to the
J1991.25 catalog position, and validates:

- epoch self-consistency of the proper-motion partials;
- our Earth-ephemeris along-scan parallax factors against the
  consortium's (agree to < 1e-3);
- that the published DMSA/O orbit, projected along-scan with EXOZIPPy's
  Thiele-Innes conventions, fully explains the abscissa residuals
  (chi2/N: 33 raw -> 1.1 after subtracting the orbit).  This confirms the
  DMSA/O (omega, Omega, i) convention matches the EXOFASTv2 convention
  implemented here.

DMSA/O solution (ESA 1997) to compare against: P = 411.4 d,
a0 = 19.94 mas, e = 0.567, omega = 4.7 deg, i = 80.5 deg,
Omega = 352.6 deg, parallax = 43.45 mas.

Caveats (demo, not publication): FAST and NDAC abscissae of the same
orbit are correlated (IA10) but treated as independent; the companion is
assumed dark, so the fitted "companion mass" describes the photocenter
orbit.  With photocenter astrometry only (no RVs), (Omega, omega) and
(Omega+180, omega+180) are exactly degenerate, so EXOZIPPy restricts
Omega to [0, 180] deg (and notes this in the LaTeX table).  The DMSA/O
solution quotes Omega = 352.6 deg, which lies in the other mode: the fit
automatically remaps the starting point and should recover
Omega ~ 172.6 deg, omega ~ 184.7 deg (the degenerate partner of the
published values).

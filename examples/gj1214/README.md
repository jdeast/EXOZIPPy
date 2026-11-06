This fit recreates the EXOFASTv2 results in Mahajan & Eastman, 2024
https://ui.adsabs.harvard.edu/abs/2024ApJ...963L..37M/abstract

It uses JWST transits and occultations to constrain the eccentricity
and stellar density, combines that with the Mann mass-Ks relation to
constrain the radius, combines that with the SED to constrain the
bolometric flux/temperature.

## The loggsed barrier (history)

GJ 1214 is an M dwarf whose mass and radius give logg = 5.022. Until
2026-10-20 the NextGen bolometric-correction grid stopped at logg 5.0, so
this example shipped `star.A.loggsed: {bound_scale: 25.0}` in its params
file: it widened the SED's soft loggsed barrier from its measured width
(~2e-4 dex, which clamped the fit to the grid edge at a cost of 434 nats)
to half a NextGen logg cell (transition width 0.25 dex), and the fit was
scored on extrapolated bolometric corrections. The tables now reach logg
5.5 (`src/exozippy/components/sed/sed.md`), the star sits inside the grid,
and the entry was removed (#508): the measured barrier applies again, and
the start logp moved by +2.2e-4 nats.

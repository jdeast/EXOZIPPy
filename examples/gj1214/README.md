This fit recreates the EXOFASTv2 results in Mahajan & Eastman, 2024
https://ui.adsabs.harvard.edu/abs/2024ApJ...963L..37M/abstract

It uses JWST transits and occultations to constrain the eccentricity
and stellar density, combines that with the Mann mass-Ks relation to
constrain the radius, combines that with the SED to constrain the
bolometric flux/temperature.

## The softened loggsed barrier

`gj1214.params.yaml` carries `star.A.loggsed: {bound_scale: 25.0}`. GJ 1214
is an M dwarf whose mass and radius give logg = 5.022, just past what was
the 5.0 ceiling of the NextGen bolometric-correction grid's logg axis
until 2026-10-20 (the tables now reach 5.5; see
`src/exozippy/components/sed/sed.md`), so its bolometric corrections were
extrapolated off the grid's edge cell. A modestly extrapolated correction
beats not fitting the star at all, as long as it is explicit, and the SED
component warns by name and value whenever a star is off the grid.

The entry widens the soft bound the SED puts on `loggsed` from its
measured width (~2e-4 dex, which clamped the fit to the grid edge and
cost that start 434 nats) to half a NextGen logg cell: `bound_scale` 25
dex -> transition width 0.25 dex, slope 17.6 nats/dex; a penalty of 0.91
nats at the 5.022 start, 4.41 at logg 5.25 (half a cell past the old
edge) and 8.80 at 5.5 (a full cell). `mkparam` carries the entry into its
restart file verbatim (with no start value added, since `loggsed` is
derived).


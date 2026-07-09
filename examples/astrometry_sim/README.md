# Simulated G+M binary: absolute + relative astrometry

A fully synthetic nearby binary (0.80 + 0.25 Msun at 20 pc, P = 3 yr)
exercising the two ground-based modes of the `astrometryinstrument`
component:

- `mode: abs` -- 2-D absolute astrometry (time, ra, dec) of the
  photocenter, including parallax (50 mas), proper motion, and the
  23 mas photocenter orbit.  The M dwarf contributes 2% of the light
  (`fluxfrac: 0.02`), shrinking the photocenter orbit relative to the
  dark-companion case.
- `mode: rel` -- relative astrometry (time, separation, position angle)
  of the companion with respect to the primary (a_rel = 106 mas).

There are no RVs, so the (Omega, omega) -> (Omega+180, omega+180)
degeneracy is exact -- it is a reflection through the sky plane,
invisible to all astrometry (absolute AND relative); only RVs identify
which node is ascending.  EXOZIPPy therefore restricts Omega to
[0, 180] deg (noted in the LaTeX table), and the injected mode
(Omega = 210, omega = 55) is automatically remapped at initialization to
its degenerate partner: the fit should recover Omega ~ 30 deg,
omega ~ 235 deg.

```
poetry run python simulate_astrometry.py   # regenerate the data
poetry run exozippy simbinary.yaml
```

Injected truth is documented in `simulate_astrometry.py` and mirrored in
`simbinary.params.yaml`.

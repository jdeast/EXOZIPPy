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

There are no RVs, but the relative astrometry observes which side of the
primary the companion is on, so the (Omega, omega) -> (Omega+180,
omega+180) degeneracy of photocenter-only fits does not apply here and
Omega is sampled over the full circle.

```
poetry run python simulate_astrometry.py   # regenerate the data
poetry run exozippy simbinary.yaml
```

Injected truth is documented in `simulate_astrometry.py` and mirrored in
`simbinary.params.yaml`.

# HAT-P-3 b

A hot Jupiter on a 2.9-day orbit around a K dwarf, with four TESS sectors
(16, 22, 49, 76), HIRES radial velocities and broadband photometry
(`hat3_sed.yaml`, built with `scripts/mkticsed.py`). The configs build the
fit up piece by piece:

| config | params | what it fits |
|---|---|---|
| `hat3_staronly.yaml` | `hat3_staronly.params.yaml` | the star alone: SED + parallax |
| `hat3_mist.yaml` | `hat3_mist.params.yaml` | the star alone, with the MIST evolutionary tracks |
| `hat3_transitonly.yaml` | `hat3_transitonly.params.yaml` | TESS transits + SED + Torres relations |
| `hat3.yaml` | `hat3.params.yaml` | transits + RVs + SED + Torres relations |
| `hat3_mist_claret.yaml` | `hat3_mist_claret.params.yaml` | transits + RVs + SED + MIST, with the Claret limb-darkening tie |

```
cd examples/hat3 && poetry run exozippy hat3.yaml
```

## The parameter files

`hat3.params.yaml` carries the MAP of a converged fit of the same config
(4 TESS sectors, NUTS, 500 tune + 1000 draws x 4 chains), so the short
smoke-test run in `hat3.yaml` starts in the right place; the priors,
bounds and citations are the example's own. The seed is one point, so
the chains are scattered around it (`overdisperse: true`).

**The mass.** Only `hat3_staronly` (SED alone, no mass relation) puts a
Gaussian prior on the stellar mass (0.92 +/- 0.05). `hat3` and
`hat3_transitonly` constrain it with the Torres relations, and
`hat3_mist`/`hat3_mist_claret` with the MIST tracks' radius, teff and age
penalties, so in those the mass is a start value only (the hand prior
was removed from `hat3` and `hat3_transitonly` in #508).

**MIST coordinates.** `star.A.initfeh` is the INITIAL (protostellar)
metallicity -- the grid's third axis, distinct from `star.feh`, the
present-day SURFACE value. MIST evolves one away from the other over a
track (diffusion, dredge-up), and the two are tied only by the feh
penalty, never equated. It is seeded slightly above the measured surface
value because diffusion depletes a K dwarf's surface over its lifetime;
a start value, not a prior. Its defaults.yaml range [-4.0, 0.5] is a
placeholder: the component narrows it to the loaded grid's own axis
extent ([-2.5, 0.5] for the shipped grid after trimming).

`star.A.eep` is deliberately not seeded: the component searches the grid
track nearest (logmass, initfeh) for the EEP minimizing the same objective
the fit uses, and hints that -- better than a guess, and an explicit
initval would override it (a user value outranks a data-derived hint).
Seed it only if you know the evolutionary phase. `star.A.age` likewise
keeps its defaults.yaml start; the tracks constrain it, and HAT-P-3's age
is not independently known.

**The Claret tie.** The tie is on by default, so it applies to every
config here with a transit (`hat3`, `hat3_transitonly`, `hat3_mist_claret`;
the last writes `star_constrains_ld: true` out explicitly). None of their
params files gives a `band.TESS` limb-darkening start: the tie seeds q1/q2
from the Claret table at the star's starting teff/logg/feh, so the start
sits exactly on the table (`band.TESS.du1 = du2 = 0`), and a start written
in the params file would win over that seed. `hat3.params.yaml` used to
carry `u1`/`u2` initvals with their own 0.05 Gaussians; the derived u1/u2
could not be started there (the model started on the table anyway), and
the Gaussians duplicated the tie. The tie's width is `band.TESS.du1`/`du2`'s
sigma (default 0.05, the tables' systematic floor, applied once per band).

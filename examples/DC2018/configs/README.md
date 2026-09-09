# DC2018-128 configuration arms

Working configs behind the DC2018 event-128 investigation (review 8.6.7 and
its children).  Their traces are the evidence base for the conclusion that
the event's `theta_E` error is entirely a `mu_rel` error.

## Ported to the 8.6.17 component split (PR #246)

These build and are verified against the current code:

| config | what it isolates |
|---|---|
| `DC2018_128_severed_v3.yaml` | the severed baseline |
| `DC2018_128_severed_v4.yaml` | v3 + `star.Lens.radius` unpinned |
| `DC2018_128_severed_v5.yaml` | v4 + `teff` unpinned, `feh` given N(0, 0.5) |
| `DC2018_128_severed_v6.yaml` | v5 + a 3.0 cap on `sed.sed.errscale` |
| `DC2018_128_severed_v7.yaml` | v5 + `star_constrains_rho` back ON (the relink) |
| `DC2018_128_tightpriors.yaml` | the observable NS arm with 8.2.2's bounds |

Verification: v3, v6 and v7 reproduce their pre-refactor start logp
BIT-IDENTICALLY (80519.222 / 80507.376 / 81373.201).  v4 and v5 build with
the expected element counts (39 and 41) but had no pre-refactor baseline
recorded.  `tightpriors` is discussed below.

## NOT ported -- pre-#246, will not load

Everything else here (`DC2018_128_mulens*.yaml`, `*_severed_forensic.yaml`,
`*mulens+sed*.yaml`) still uses the pre-split surface: a single `lens:`
entry carrying `lenses:`/`sources:` lists.  They are kept for PROVENANCE --
they document exactly what produced the traces the 8.6.7 conclusions rest
on -- and they are cheap to port when one is next needed:

  * event-shared keys (`finite_source`, `t0_par`, `fitmurel`, `fitpirel`,
    `fitthetae`, `mmexofast`) move to a `mulensevent:` block;
  * each lens body gets its own `lens: - body: star.X` entry, and
    `orbital_motion`/`orbit` go on the COMPANION's entry;
  * each source gets `source: - body: star.Y`, carrying `fitu0te` and
    `star_constrains_rho`;
  * params keys move by role: `t_0`/`u_0`/`rho` to `source.<Src>.*`,
    `s`/`alpha`/`q` to `lens.<Companion>.*`, `t_E`/`theta_E`/`pi_rel`/
    `mu_*_rel` to `mulensevent.*`.

`components/mulensing/bodies.py` is the authority on key homes, and it
raises a "this key now lives in X" error rather than failing silently.

## A note on the `tightpriors` baseline

Do not trust the 80865.224 start logp recorded for it on 2026-09-04.  That
number was measured while the t_0 bound was keyed `lens.DC2018_128.t_0` --
the RUN name, which matches no instance -- so the tightening was silently
absent (review 2.3.16).  Measured since:

    pre-#246 code + corrected config : -14,122,931.6
    post-#246 code + ported config   :     +82,400.5

The port did not regress anything; the baseline describes a config state
that no longer exists.  The pre-#246 figure is worth someone's attention on
its own -- applying the tightened t_0 to the OLD code produced a start the
engine could not resolve sanely -- but it does not affect nested sampling,
which draws live points from the prior and never uses that start.

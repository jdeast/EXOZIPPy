# DC2018-128 configuration arms

Working configs behind the DC2018 event-128 investigation.

## Retired: the severed arms (v3..v7)

`DC2018_128_severed_v3.yaml` ... `_v7.yaml` (and their
`DC2018_128_sed_severed_v*.params.yaml`) cut `rho` loose from the stellar
chain (`star_constrains_rho: false`) to find why every fit put `theta_E`
~6x above the answer key, then varied the lens pins, an SED error-scale cap
and the relink.  They were removed because they all fit a STATIC lens, and
event 128's lens orbits (P = 1.27 yr): with linear lens orbital motion the
prior-free light curve gains 458 in chi2 over the best static fit and
recovers the key's `s`, `q` and `rho` to 0.1%, 0.1% and 1%, while the static
model's `s` and `q` are 40 and 21 sigma off (`../dc18_orbital_motion_signal.py`,
#318).  The arms were diagnosing a symptom of fitting a moving lens static.
Their lasting lessons -- an uncapped noise scale absorbs any disagreement
between parts of the model, and a pin inherited from a no-SED config can
force a solar-type lens -- are already in the code (the error-scale caps)
and in the configs here.  They also stopped building when the relaxation
engine was rewritten (#455: no start for `log_pi_rel` with the `rho` tie
cut and no `pi_E` seed).  The scripts that read their traces went with
them.  Recover any of them from git history (removed in the PR that added
this section).  The fit that matters for 128 is one with lens orbital
motion.

## Ported to the 8.6.17 component split (PR #246)

| config | what it isolates |
|---|---|
| `DC2018_128_tightpriors.yaml` | the observable NS arm with 8.2.2's bounds |

## NOT ported -- pre-#246, will not load

**These are shipped deliberately broken.**  JDE's ruling 2026-09-09: keep
them with this README, and DELETE THEM WHEN 8.6.7 LANDS if they turn out to
have no further value.  So this section is expected to shrink to nothing,
and nobody should invest in porting one unless they are actually about to
re-run it.

Everything else here (`DC2018_128_mulens*.yaml`, `*_severed_forensic.yaml`,
`*mulens+sed*.yaml`) still uses the pre-split surface: a single `lens:`
entry carrying `lenses:`/`sources:` lists.  They are kept for PROVENANCE --
they document exactly what produced the traces the 8.6.7 conclusions rest
on -- and they are cheap to port when one is next needed:

  * event-shared keys (`finite_source`, `t0_par`, `fitmurel`, `fitpirel`,
    `fitthetae`) move to a `mulensevent:` block (a `mmexofast:` key
    is gone: run `../convert_mmexofast_json.py` on the ported config);
  * each lens body gets its own `lens: - body: star.X` entry, and
    `orbit` goes on the COMPANION's entry (the orbit's `type:` sets the motion);
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

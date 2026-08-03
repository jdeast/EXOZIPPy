# KELT-17 b -- Rossiter-McLaughlin example

Demonstrates the Rossiter-McLaughlin (RM) hook in `rvinstrument` on a real
misaligned hot Jupiter (KELT-17 b; Zhou et al. 2016). Data converted from the
EXOFASTv2 `examples/kelt17/RM` example.

The star is a fast rotator (vsini ~ 44 km/s) and the orbit is retrograde
(lambda ~ -116 deg). The out-of-transit orbital RVs and the two in-transit
RM sequences are all in a single TRES file (`KELT-17.TRES.rv`); the instrument
opts into the RM model with `rm: b`. The RM anomaly is ~0 outside transit, so
the same tag safely covers the out-of-transit points that constrain the orbit.

## What it shows
- The `rm:` / `rm_band:` wiring end to end: the RM anomaly is added to the
  in-transit RV mean, using the sqrt(vsini)cos/sin(lambda) reparameterization
  (`svcoslam`, `svsinlam`) on `orbit` -- the spin-orbit analog of the
  secosw/sesinw -> ecc/omega idiom. The line-broadening terms
  (vmacro/vbeta/vmicro) live on `star`.
- A finite initial logp, the RM S-curve drawn in the RV panels, and recovery of
  the retrograde spin-orbit angle.

## Run
```bash
cd examples/kelt17
exozippy kelt17.yaml
```

## Notes
- The stellar radius/Teff/[Fe/H] are priored directly (no SED/MIST here); vsini
  and lambda come out of the in-transit RM signal via `svcoslam`/`svsinlam`.
- With the standard `numpyro` NUTS config (diagonal mass, `init: adapt_diag`,
  `target_accept: 0.95`) the chains converge and recover a retrograde lambda
  consistent with the published value. `check_curvatures: False` is set because
  `quad_solution_vector` (limb-darkened blocked flux) has no second derivative,
  as in the `kelt4` example.
- This is a minimal, ground-based-transit demonstration of the RM wiring;
  publication-quality obliquity uses the fuller constraint set (SED, more
  transits, a tighter b/cosi prior) -- i.e. the full EXOFASTv2 KELT-17 setup.

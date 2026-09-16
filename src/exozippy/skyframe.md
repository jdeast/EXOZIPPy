# The sky-plane frame

`src/exozippy/skyframe.py`: the left-handed astrometric frame the whole codebase uses, and
the one owner of the projection onto it.

Read this before writing anything that projects a 3-D position onto the sky, before
"fixing" a sign in orbit, astrometry or microlensing geometry, and before assuming two
call sites disagree about a convention.

For the *microlensing* consequences of this frame -- the trajectory and parallax signs
written out, `alpha`'s frame, `q > 1`, and the mappings onto other codes' and papers'
conventions -- see `src/exozippy/components/mulensing/conventions.md`, whose paper-facing
twin is `src/exozippy/latex/convention.tex`.

## The sky-plane frame (`skyframe.py`)

EXOZIPPy uses the century-old **left-handed** astrometric frame everywhere: **+X = North** (plotted up), **+Y = East** (plotted left), **+Z = distance** (growing away from the observer). It is left-handed as a set of physical directions (`X x Y = -Z`), and that is the point, not an oversight to tidy up. It is what simultaneously satisfies the textbook definitions of the Keplerian elements *while* preserving the standard Euler application `Rz(bigomega) Rx(inc) Rz(omega)`: `bigomega` comes out as the position angle of the ascending node measured **East of North** (at `omega + f = 0` the body sits at `PA = bigomega` exactly), `omega` is the argument of periastron of the orbit it names (here the **primary's**, `omega_*`), and `dZ/dt` is the RV with the right sign (positive = receding = redshift) because `+Z` is distance. Any right-handed relabelling breaks one of the three; this is the convention repeatedly mangled in the exoplanet literature. `Orbit.get_sky_position` / `Orbit.get_radial_velocity` implement it directly, and the primary-transit anomaly `f = pi/2 - omega_*` (`calc_tp`, `calc_b`) puts the **star** at `Z > 0`, i.e. the planet in front -- which is what fixes the primary/secondary sense.

**`skyframe.py` owns the projection onto that frame**, in three functions, and nothing may re-derive the algebra at a call site. `sky_basis(ra, dec)` returns the spherical tangent basis `(e_hat, n_hat)`; the two projections of a 3-D barycentric observer position (`ephemeris.get_observer_position`, ICRS/J2000 equatorial AU) are `observer_sky_offset` and `parallax_factors`, and **they differ by a sign because they are different QUANTITIES, not different conventions**:

- `observer_sky_offset(xyz, ra, dec, xp=np)` -> `(delta_e, delta_n)`, the **observer's own** offset projected on the sky. What the microlensing trajectory consumes. Gould 2004 writes it `s(t)`; that name is deliberately **not** used here, because `lens.s` is the binary separation in Einstein radii.
- `parallax_factors(...)` -> `(P_E, P_N)`, the apparent displacement **of the source** per unit parallax, consumed as `+ plx * P_E`. An observer displaced by `R` sees a source at distance `d` shifted by `-R/d`, so this is *defined* as `-observer_sky_offset` rather than written out a second time.

That sign relation is exactly what review 4.6.3 mis-read as astrometry carrying "the OPPOSITE sign convention". It does not; it carries the other quantity. All seven former copies (`lens.py` x2, `op.py`, `mulensinstrument.py` x2, `astrometryinstrument.py` x2, in three spellings) projected onto the **same** `(e_hat, n_hat)`. Backend-agnostic via `xp=` (pass `pytensor.tensor` for the symbolic likelihood path).

**Microlensing already agrees with this convention** -- verified, not assumed. Building the apparent lens-source separation from raw 3-D geometry reproduces `Lens.get_magnification`'s trajectory to 1.8e-15 in `|u|`, while the mirrored-beta convention is wrong by 0.09; since both the `tau` and the `beta` sign are pinned by that, it is a **handedness** test. `beta_hat` is `tau_hat` rotated +90 degrees in the East-of-North sense -- the same orientation convention as `get_sky_position`'s PA. So **no sign needs unifying and no parity with MulensModel/VBM is broken**; `tests/test_pspl_symbolic_vs_op.py` already pins symbolic == MulensModel with real annual parallax. Likewise the galactic-prior interface: `star.pm_ra` is `mu_alpha cos(delta)` (East-positive) throughout, `calc_pi_E_E` is proportional to `mu_ra_rel` in that same basis, and `calc_mu_ra_rel_geo = mu_ra_rel - pi_rel * earth_vperp_e` reproduces Gould 2004's helio->geo conversion sign exactly (it falls out of substituting the linear reference trajectory into the geometry above).

Two things `skyframe` deliberately does **not** own. `op.py`'s basis used to be built as MulensModel `Coordinates` builds it (`east = normalize(z x direction)`, `north = direction x east`); that is the same basis to **1 ulp**, not bit-identical, so sharing the definition moved the Op path's line of sight in its last bit and nothing else (all three shipped examples exercised -- ob08092 symbolic, GaiaBH1 astrometry, ob161003 binary-lens Op -- are bit-identical in start logp *and* full initial point). And **`alpha`**, the binary-lens orientation, is a genuinely microlensing-only convention with no universal-frame analogue: `op.py` puts companion *j* at `s(cos alpha, -sin alpha)` in a frame where the source sits at `(-tau, -u)`. It is pinned against MulensModel by `tests/test_vbm_direct_vs_mulensmodel.py` and has a known 180-degree offset against some published values (`alpha_MM = 180 - alpha_paper` for ob161003). Do not fold it into this section's argument.

Tests: `tests/test_skyframe.py` (helper contract, plus the two first-principles physics tests -- the Keplerian left-handedness claims and the microlensing handedness check -- each of which fails loudly under a mirrored convention).


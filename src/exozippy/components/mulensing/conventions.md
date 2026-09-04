# Microlensing conventions

`src/exozippy/components/mulensing/`: the sky frame, the origins, the parallax signs, and
the binary geometry -- stated once, each claim tied to the code that implements it and the
test that pins it, plus the mappings onto the conventions used by other codes and papers.

Read this before comparing an EXOZIPPy microlensing parameter to a published one, before
"fixing" a sign, and before writing a converter between EXOZIPPy and any other modelling
code. Related: `src/exozippy/skyframe.md` (the frame itself, and the one owner of the
projection onto it), `src/exozippy/components/mulensing/mulensing.md` (the flux likelihood,
MMEXOFAST seeding, and the lens/source body rules).

## The two artifacts, and which one is normative

This file is the **normative** list. `src/exozippy/latex/convention.tex` is a drop-in
section for the EXOZIPPy microlensing paper carrying the *same* claim list in the paper's
register, with the same identifiers `C1`...`C23`. The identifiers are the anti-drift
device: a claim may be reworded in either file, but a `C`-number must mean the same thing
in both, and a claim added to one must be added to the other under the same number. There
is no generator and no test enforcing that -- keep them in one commit.

The split of labour is: **this file names the code**, `convention.tex` names the
**literature**. Nothing in the paper section should assert a convention that is not a
`C`-number here, and nothing here should be a bare claim without a file or a test beside
it.

## Notation used below

`N` and `E` are the North and East components of a sky-plane vector. `mu_rel` is always
the LENS relative to the SOURCE (C7). `theta_E` is always the angular Einstein radius of
the TOTAL lens mass (C13). `phi_pi = atan2(pi_E_E, pi_E_N)` is the position angle of the
relative motion, East of North.

---

## 1. The sky-plane frame

### C1 -- the axes

    +X = North      (plotted up)
    +Y = East       (plotted left)
    +Z = distance   (growing away from the observer)

This is **left-handed as a set of physical directions** (`X x Y = -Z`), deliberately. It is
the only labelling that simultaneously satisfies the textbook definitions of the Keplerian
elements and preserves the standard Euler application `Rz(bigomega) Rx(inc) Rz(omega)`:
`bigomega` comes out as the position angle of the ascending node measured East of North,
`omega` is the argument of periastron of the orbit it names, and `dZ/dt` is the radial
velocity with positive = receding. A right-handed relabelling breaks one of the three.

- Implemented in: `src/exozippy/skyframe.py` (module docstring), `Orbit.get_sky_position`,
  `Orbit.get_radial_velocity`.
- Pinned by: `tests/test_skyframe.py::test_keplerian_sky_and_rv_are_left_handed`, which
  builds the orbit from the raw Euler rotation with the axes labelled `(N, E, Z)` and
  checks all three claims; and
  `tests/test_skyframe.py::test_primary_transit_puts_the_planet_in_front`, which fixes the
  sign of `+Z`.

### C2 -- the positive rotation sense is North through East

Every angle on the sky in this codebase -- position angles, `bigomega`, `phi_pi`, `alpha`
(C15) -- increases from North toward East. On a sky plot drawn with **North up and East to
the left** that is counterclockwise, which is why the astrometry sky panels invert the
horizontal axis (`astrometryinstrument.py`, `meta["x_inverted"] = True  # East to the
left`).

Skowron+2011 Section A.6 states the same thing as "all coordinate systems are
right-handed, either in two dimensions [(N,E), ...]": `(N, E)` is a right-handed 2-D pair,
so `N -> E` is the positive sense. C1's left-handedness and C2's right-handed `(N, E)` pair
are not in conflict -- C1 is about the 3-D triple including the line of sight, C2 about the
2-D sky plane alone.

- Pinned by: the position-angle assertion inside
  `tests/test_skyframe.py::test_keplerian_sky_and_rv_are_left_handed` (at
  `omega + f = 0` the body sits at `PA = bigomega` exactly).

### C3 -- the basis

`skyframe.sky_basis(ra, dec)` returns the spherical tangent basis in ICRS/J2000 equatorial
XYZ, with `ra`/`dec` in radians:

    u_hat = ( cos dec cos ra,  cos dec sin ra,  sin dec)
    e_hat = (-sin ra,          cos ra,          0      )
    n_hat = (-sin dec cos ra, -sin dec sin ra,  cos dec)

This is the same basis MulensModel's `Coordinates` builds as
`east = normalize(z_hat x u_hat)`, `north = u_hat x east`, to within 1 ulp -- so the Op path
and the symbolic path share one line of sight by construction, not by coincidence.

- Pinned by: `tests/test_skyframe.py::test_sky_basis_is_the_spherical_tangent_basis`,
  `::test_cross_product_construction_agrees`.

ASCII picture of the sky plane as EXOZIPPy plots it (only the POSITIVE half-axes carry
arrowheads):

                              N (+X)
                                ^
                                |
                                |
            E (+Y) <------------o
                                |
                                |

    +Z points AWAY from the reader, into the page: the observer is in
    front of the page and the source behind it.  So the triple is
    left-handed, X x Y = -Z.

    Positive angles (PA, bigomega, phi_pi, alpha) rotate N -> E, i.e.
    counterclockwise AS DRAWN.

---

## 2. Origins and reference epochs

### C4 -- times are BJD_TDB

Every epoch entering the microlensing model -- data times, `t_0`, `t0_par` -- is BJD_TDB.
Files in another time system are converted at load by the shared `Instrument._read_data`
machinery (`time_scale:`, `time_frame:`, `time_offset:`); see
`src/exozippy/components/instrument.md`. Microlensing has one extra rule:
`MulensInstrument._reject_time_spec_with_mmexofast` hard-errors when a time spec and an
active MMEXOFAST seeding run are combined, because MMEXOFAST reads the raw files itself
and would see the unconverted times.

### C5 -- the geocentric frame, anchored at `t0_par`

Parameters live in the Skowron+2011 geocentric inertial frame: the frame moving with
Earth's **position and velocity** at the fiducial epoch `t0_par`. `t0_par` is a
configuration choice, never a fitted parameter. `MulensInstrument._resolve_t0_par_final`
picks it in the order explicit `lens: t0_par:` > user `lens.0.t_0` initval > MMEXOFAST seed
`t_0` > median data time.

`t_E` and the DIRECTION of `pi_E` are geocentric quantities and therefore depend on
`t0_par`; `|pi_E|`, `theta_E`, `s`, `q` and `rho` do not. The star components' `pm_ra` /
`pm_dec` are barycentric observables, so `physics.calc_mu_ra_rel_geo` /
`calc_mu_dec_rel_geo` apply the Gould (2004) conversion
`mu_geo = mu_helio - pi_rel * v_earth_perp(t0_par) / AU` before `t_E` and `pi_E` are formed.

- Pinned by: `tests/test_mu_rel_geo.py::test_mu_geo_sign_matches_first_principles_trajectory`
  (the flipped sign changes `t_E` by tens of percent at `pi_rel ~ 0.35 mas`).
- Reported by: the shared `table_note` on `t_E` / `mu_*_rel_geo` in `Lens.register_parameters`.

### C6 -- observer positions arrive as geocentric DEVIATIONS

`MulensInstrument._abs_to_delta` converts each observer's absolute barycentric position to

    delta(t) = xyz_obs(t) - [xyz_earth(t0_par) + v_earth(t0_par) * (t - t0_par)]

in AU, ICRS/J2000 equatorial. **Both** magnification backends -- the symbolic PSPL formula
and the VBMicrolensing / MulensModel Ops -- consume exactly this array, so the two are
interchangeable on this input, and a satellite is "just another observatory"
(Yee et al. 2015 Section 3): its deviation is simply large (~1-2 AU for Spitzer) where
Earth's is small (annual parallax). Rows of zeros mean no parallax.

- Pinned by: `tests/test_pspl_symbolic_vs_op.py::test_pspl_symbolic_vs_op_with_annual_parallax`
  and `::test_pspl_symbolic_vs_op_with_satellite_offset` -- each asserts symbolic == Op to
  `1e-6` AND that the magnification actually responds to the deviations (the second half is
  the regression for the 2026-08-08 bug in which the Op subtracted the actual ephemeris and
  so deleted annual parallax entirely).

### C7 -- all relative motion is LENS relative to SOURCE

    mu_ra_rel  = lens_pm_ra  - source_pm_ra
    mu_dec_rel = lens_pm_dec - source_pm_dec
    pi_rel     = 1000/d_lens - 1000/d_source     (mas, with d in pc)

- Implemented in: `mulensing/symbolic_physics.py` `RELATIONS`.
- This is Skowron+2011 Section A.6's rule verbatim: "all relative motion conventions are defined by
  the motion of the lens (with the source thought of as fixed)".

---

## 3. The trajectory, and the parallax sign

### C8 -- the parallax vector

    pi_E = (pi_E_N, pi_E_E) = (pi_rel / theta_E) * mu_hat_rel,geo
    |pi_E| = pi_rel / theta_E
    phi_pi = atan2(pi_E_E, pi_E_N)  = the position angle of the lens's motion
                                      relative to the source, East of North

- Implemented in: `mulensing/physics.py` `calc_pi_E_N` (takes `mu_dec_rel_geo`),
  `calc_pi_E_E` (takes `mu_ra_rel_geo`); the `_geo` inputs are C5's conversion.
- `star.pm_ra` is `mu_alpha cos(delta)` (East-positive) throughout, so `pi_E_E` really is
  the East component and no `cos(dec)` factor is missing anywhere.

### C9 -- the trajectory, from first principles

Write `tau_hat = mu_hat_rel,geo` and let `beta_hat` be `tau_hat` rotated **+90 degrees in
the East-of-North sense** (C2). Then the LENS-minus-SOURCE angular separation, in units of
`theta_E`, is

    dtheta(t) = u_0 * beta_hat
              + [(t - t_0)/t_E] * tau_hat
              - (pi_rel/theta_E) * (delta_N, delta_E)

-- everything in `(N, E)` components, with
`(delta_N, delta_E) = observer_sky_offset(...)` in AU and `pi_rel` in mas per AU of
observer displacement (the standard microlensing shorthand; the code writes exactly
`-(pi_rel/theta_E) * delta_n` and likewise for E). Rotating that last term onto
`(tau_hat, beta_hat)` is what gives C10's two scalar equations. Two consequences worth
stating plainly:

- `u_0` is the coefficient of `beta_hat`, so **`u_0 > 0` means the lens passes the source on
  the lens's right** -- exactly Skowron+2011 Section A.6's sign rule.
- `A(u)` depends only on `|u|`, so the sign of `u_0` is observable only through parallax;
  see C23.

- Pinned by: `tests/test_skyframe.py::test_microlensing_trajectory_matches_3d_geometry`,
  which rebuilds `dtheta(t)` from raw 3-D geometry with a real Earth ephemeris and matches
  `Lens.get_magnification`'s `|u(t)|` to `1e-12`, while the MIRRORED `beta` convention is
  wrong by `O(0.1)`. Both the `tau` sign and the `beta` sign are pinned by that one
  assertion, which makes it a **handedness** test rather than a magnitude test, and it is
  the only absolute pin on the sense of `u_0` (the MulensModel parity tests pass `u_0`
  straight through to MulensModel and so cannot see an agreed-upon flip).

### C10 -- the parallax signs, written out BOTH ways

This is the single most-confused item in this file, and the confusion is not about the
physics: it is about **which offset the symbol `delta` names.** There are two, they are
exact negatives, and both spellings appear in the codebase because both appear in the
literature.

With `(delta_N, delta_E) = skyframe.observer_sky_offset(...)`, the **observer's own**
projected offset (what `Lens.get_magnification` uses):

    tau(t) = (t - t_0)/t_E - delta_N * pi_E_N - delta_E * pi_E_E
    u(t)   = u_0           + delta_N * pi_E_E - delta_E * pi_E_N

With `(D_N, D_E) = skyframe.parallax_factors(...) = -(delta_N, delta_E)`, the apparent
displacement **of the source** per unit parallax (what MulensModel calls `delta` and what
`op.VBMDirectMagOp._deltas` caches):

    tau(t) = (t - t_0)/t_E + D_N * pi_E_N + D_E * pi_E_E
    u(t)   = u_0           - D_N * pi_E_E + D_E * pi_E_N

The two displays are the same equations. **"Minus on the East terms" is not a
convention-free statement** and should not be repeated without saying which offset is
meant: in the first display `tau` carries a minus on BOTH N and E, in the second it carries
a plus on both. The shorthand "minus on the East terms, per Gould 2004 / Yee+2014" has
circulated in this project's notes and in review text; retire it -- it names neither offset
and is not true of `tau` under either reading. Quote one of the two displays above instead.

- Implemented in: `Lens.get_magnification` (symbolic, first display) and
  `op.VBMDirectMagOp._compute` + `_deltas` (numeric, second display).
- Byte-for-byte the same as MulensModel `Trajectory._project_delta`
  (`delta_tau = dN*pi_E_N + dE*pi_E_E`, `delta_beta = -dN*pi_E_E + dE*pi_E_N`) with
  MulensModel's `_get_delta_annual` / `_get_delta_satellite`, both of which negate the
  observer position on projection. MMEXOFAST calls MulensModel, so published `pi_E_N`,
  `pi_E_E` values are calibrated to this convention and drop straight in.
- Pinned by: `tests/test_pspl_symbolic_vs_op.py` (all three tests) and
  `tests/test_vbm_direct_vs_mulensmodel.py`.
- The stakes are not academic. Earth's annual deviation is `~0.003 AU`, so a sign error
  there barely moves a ground-only fit; Spitzer's projected deviation is `~0.125 AU`, where
  the same error gave `A = 1.91` against `1.77` near peak -- an 8% error, against
  photometry good to ~0.1%. So the sign is effectively untestable on ground data alone and
  immediately fatal with a satellite, which is why this is pinned by a test rather than by
  a fit looking healthy.

### C11 -- the two offsets differ by a sign because they are DIFFERENT QUANTITIES

`observer_sky_offset` and `parallax_factors` do not carry different conventions.
`observer_sky_offset` is where the OBSERVER moved;
`parallax_factors` is where the SOURCE appears to move, which for an observer displaced by
`R` looking at a source at distance `d` is `-R/d`. `skyframe.parallax_factors` is therefore
*defined* as the negative of the other rather than written out a second time. Microlensing
consumes the first, `astrometryinstrument` consumes the second (as `+ plx * P_E`), and that
is the whole of the apparent "sign disagreement" between the two components.

- Pinned by: `tests/test_skyframe.py::test_parallax_factors_are_the_negated_offset` (the
  exact-negative relation) and `::test_parallax_factors_match_first_principles_displacement`
  (that `parallax_factors` really is the apparent source displacement, computed in 3-D).

---

## 4. Binary and multiple lenses

### C12 -- the origin is the lens CENTER OF MASS

`t_0` and `u_0` are the time and separation of the source's closest approach to the lens
**center of mass**, for any number of lens bodies. `op.VBMDirectMagOp._magnify` builds the
lens positions with the primary at the origin and then shifts by `pos -= m @ pos`; the
source coordinates are not shifted, so they are COM-referenced. The single-companion branch
hands `(s, q, x, y)` to VBMicrolensing, which is COM-centred by the same convention.

Skowron+2011 Section A.6 requires the "system center" to be stated explicitly; this is ours.

### C13 -- lengths are in Einstein radii of the TOTAL lens mass

`theta_E**2 = kappa * M_lens,total * pi_rel`, and `t_E`, `rho`, `s` and `pi_E` all inherit
that normalization. For a multi-body lens `Lens.register_parameters` swaps `mlens_total`
(the sum of every lens body's mass) into `theta_E`'s dependency chain in place of the
primary's mass. This matches the published parameterization.

`s` itself is derived: `log_s` is the sampled coordinate and `s = 10**log_s`, which makes
the close/wide degeneracy the exact reflection `log_s -> -log_s` (`|J| = 1`).

### C14 -- `q = m_companion / m_primary`, and `q > 1 is LEGAL`

"Primary" is the **first entry of the lens block's `lenses:` list** -- a slot label, not a
mass ordering. `physics.calc_q` divides each companion's mass by the primary's, and
`lens.q`'s bounds are `[1e-8, 100]` with the defaults.yaml comment saying why: "q > 1 is
legal -- it is the same geometry with the two bodies relabelled."

Consequences when reading a published value:

- Most papers adopt the convention that the primary is the **more massive** body, so their
  `q <= 1`. If EXOZIPPy's `lenses:` list puts the lighter body first, the published `q`
  must be inverted before it is used as a seed -- and `alpha` shifted by 180 degrees (C19).
- `examples/ob161003` is the shipped case: `q = 1.188`, i.e. the "companion" (`LensB`) is
  the more massive body, matching Jung et al. (2017) Table 1 as published.
- The magnification backends do not care. MulensModel and VBMicrolensing put the body of
  mass `1/(1+q)` at negative `x` and the body of mass `q/(1+q)` at positive `x`, whichever
  is heavier.

### C15 -- `alpha`

**`alpha_j` is the angle from the primary-to-companion-j axis to the direction of
lens-source relative motion (`tau_hat`), measured counterclockwise (North through East).**

Operationally, in the trajectory frame whose `+x` is `tau_hat` and whose `+y` is `beta_hat`
(C9), with the origin at the lens center of mass:

    source:       ( -tau, -u )
    companion j:  s_j * ( cos alpha_j, -sin alpha_j )   relative to the primary

`op.VBMDirectMagOp` uses that layout directly for `n_companions >= 2`; for a single
companion it rotates by `alpha` into the lens-axis frame instead
(`x = -tau*cos a + u*sin a`, `y = -tau*sin a - u*cos a`), which is algebraically identical
and is character-for-character MulensModel's `Trajectory._get_xy`.

Note that the source *moves* along `-tau_hat`: `tau_hat` is the direction the LENS moves
relative to the source. Measuring `alpha` to the source's own direction of travel instead
is the single commonest source of a 180-degree offset in the literature (C21).

- Every companion's `alpha_j` is measured from the same `tau_hat`, so an N-body lens has one
  `alpha` per companion and no chained angles.
- Pinned by: `tests/test_vbm_direct_vs_mulensmodel.py::test_vbm_direct_matches_mulensmodel_binary_with_parallax_and_ld`
  and `::test_vbm_direct_matches_mulensmodel_binary_point_source` (the direct path against
  MulensModel, over randomized draws -- "any convention drift ... shows up here as a
  magnification mismatch far above floating-point noise"), plus
  `::test_multi_lens_frame_reduces_to_binary`, which is what pins the N-body layout against
  the binary rotation.

### C16 -- `alpha` is sampled as an unconstrained pair

`alpha = arctan2(yalpha, xalpha)` (`physics.calc_alpha`), with `xalpha` and `yalpha` the
sampled coordinates. The angle therefore never wraps and needs no periodic boundary. A user
seeding `lens.<name>.alpha` in degrees is propagated to the pair by the relaxation engine.
`alpha`'s user unit is degrees and its internal unit is radians; both magnification
backends take degrees, which is what `Lens._alpha_deg` converts to.

### C24 -- lens orbital motion: `ds_dt`, `dalpha_dt`, and the gamma identities

**The linear mode is definitional.** Per companion `j`, with `t0_par` the same anchor the
parallax uses (C5; Skowron Eq. A17 recommends `t_0,kep = t_0,par`, which is what makes the
orbital and parallax terms composable):

    s_j(t)     = s_j0     + ds_dt_j     * (t - t0_par)/365.25
    alpha_j(t) = alpha_j0 + dalpha_dt_j * (t - t0_par)/365.25

`ds_dt` is in Einstein radii of the TOTAL mass (C13) per YEAR; `dalpha_dt`'s internal unit
is rad/yr and its user unit deg/yr -- the identity mapping to MulensModel's `ds_dt` /
`dalpha_dt` (C18 extends to both), and Skowron's own Section 3.3.1 parameterization up to
the sign vocabulary below.

**Skowron's gamma vocabulary maps onto these with ONE minus sign, and it is not optional:**

    gamma_par     = ds_dt / s_0
    gamma_perp    = -dalpha_dt        [rad/yr]
    d(PA_axis)/dt = +gamma_perp = -dalpha_dt

The identity `gamma_perp = -dalpha/dt` is Skowron+2011 Appendix A.4 verbatim, and their
Section 3.3.1 writes the linear expansion as `alpha(t) = alpha_0 - gamma_perp (t - t0_par)`.
It is also a one-line consequence of C15/C20: `alpha = phi_pi - PA(axis)` with `tau_hat`
fixed, so `alpha` runs OPPOSITE to the axis's own position angle.  `gamma_perp` is therefore
the axis's physical sky rotation rate, positive in the C2 sense (North through East) --
`(gamma_par, gamma_perp)` is "right-handed ... just like (N, E)" (A.4), and no handedness
correction relates it to our frame: C1's left-handed triple and Skowron's right-handed one
differ only in which way the THIRD axis points, and that flip cancels the handedness label
flip identically in the sky plane, the only plane `alpha` and `gamma_perp` live in.

**Never quote a sign for `dalpha_dt` or `gamma_perp` without stating `sign(u_0)`.**  The
orbiting-binary ecliptic degeneracy (Skowron Eq. A16, the C23 mirror extended to orbital
motion) reverses all four together:

    (u_0, alpha, pi_E_perp, gamma_perp) -> -(u_0, alpha, pi_E_perp, gamma_perp)

For OGLE-2009-BLG-020 itself (their Eq. 16, near-equinox peak) `pi_E_perp ~ pi_E_N`.  A
mirror seed must flip all four or it is a different (and usually terrible) model, not the
degenerate partner.

**The keplerian mode projects an orbit through the same definitions**: with
`delta_j(t) = (dE, dN)` the companion's offset from the primary in Einstein radii
(`Orbit.state_vectors` scaled by `a / (D_L * theta_E)`),

    s_j(t)      = |delta_j(t)|
    PA_axis(t)  = atan2(dE, dN)
    alpha_j(t)  = alpha_j0 - [PA_axis(t) - PA_axis(t0_par)]

The MINUS is the same `d(PA_axis)/dt = -dalpha/dt` rule; getting it wrong does not raise
chi2 -- it silently reports the wrong inclination branch (the rotation sense of the binary
axis is the ONLY thing that measures `sign(cos i)` here; Skowron Section 5.2:
`gamma_perp -> -gamma_perp` is `Omega_node -> -Omega_node, i -> pi - i`).  That is why the
sign is pinned by a synthetic orbit with KNOWN inclination rather than by a chi2
improvement.

- Implemented in: `Lens` (config keys `orbital_motion: linear|keplerian`, `orbit:`),
  `mulensing/physics.py`, `op.VBMDirectMagOp` (per-epoch `s_t`/`alpha_t` inputs).
- Pinned by: `tests/test_lens_orbital_motion.py` -- the known-inclination sign test, the
  MulensModel linear-motion parity test, and the A16 mirror test.

### C25 -- source orbital motion (xallarap): the SOURCE's offset at the parallax slot

Parallax is the OBSERVER's offset; xallarap is the SOURCE's own.  They enter the
trajectory at exactly the same slot, with the same sign discipline (C8/C9/C11's "which
offset does the symbol name" lesson applies verbatim).  With `dsigma = (dsigma_N,
dsigma_E)` the luminous source's sky offset from its own barycentric motion, in Einstein
radii and ANCHORED AT `t0_par` (the offset vanishes there, so `t_0`/`u_0` keep their C5
meaning -- one anchor for the parallax, the lens orbital motion and the source orbital
motion):

    dtau = -(dsigma_N * tau_hat_N + dsigma_E * tau_hat_E)
    du   = +(dsigma_N * tau_hat_E - dsigma_E * tau_hat_N)

added to C10's first display.  Both components are `-(dsigma . basis_vector)`, and the
MINUS is C7: the trajectory is LENS minus SOURCE, so a source displaced by `+dsigma`
moves the relative position by `-dsigma`; `tau_hat = mu_hat_rel,geo = (tau_hat_N,
tau_hat_E)` and `beta_hat = (-tau_hat_E, +tau_hat_N)` -- the +90 deg North-through-East
rotation -- are C9's basis, the SAME pair the parallax terms project onto (C10 display 1's
`u` term is `-|pi_E| (delta . beta_hat)` with exactly this `beta_hat`).  HISTORY (review
2.6.13, fixed 2026-09): the `du` line above carried a leading minus, i.e. `beta_hat =
(tau_hat_E, -tau_hat_N)`, the MINUS-90 rotation -- inconsistent with the parallax slot it
claims to share -- and the xi_* mapping below had been tuned against that inverted
projection.  The two errors cancelled in the track-level parity test and inverted the
shift actually applied to the light curve: on examples/ob170114 the built magnification
peaked at A = 8.6 where MulensModel (and the photometry, at chi2/N = 1.5) give 3.3.

The offset itself is the orbit component's primary-body track in Einstein units:
`dsigma(t) = [r_1(t) - r_1(t0_par)]`, `r_1` from `a_1 = a * m_companion / m_total`
projected through the SAME kernel and Thiele-Innes owner as everything else, scaled by
`a_1 / (D_S * theta_E)`.  NO new sampled parameters: the orbit's period, eccentricity,
orientation and the companion mass (through the barycentric scale) are the physical
coordinates, which is Mroz et al. (2026)'s "ET" philosophy expressed through the component
graph rather than bolted on.  A LINEAR source drift is deliberately not offered: it is
EXACTLY degenerate with `(t_E, t_0, u_0, alpha)` in the light curve alone
(notes/orbital_motion_and_nbody.txt section 2); it becomes meaningful only where an
external dataset constrains the source's proper motion over a baseline `>> t_E`.

A xallarap orbit measures `bigomega` (the track's sky orientation enters the trajectory)
but remains NODE-DEGENERATE -- a sky track of any kind is invariant under the sky-plane
reflection `(bigomega, omega) -> (bigomega + 180, omega + 180)` -- unlike the LENS
keplerian case (C24), where the caustic's rotation SENSE breaks it.

**The mapping to MulensModel's `xi_*` elements (Zhai et al. 2024; the parameterization of
Mroz et al. 2026), VERIFIED at the LIGHT-CURVE level** -- the direct Op reproduces
MulensModel's full 2L1S + xallarap magnification to 4e-16 at mapped elements, and the
shift tracks agree to ~1e-9 over random draws
(`tests/test_xallarap.py::test_mm_xi_closed_form_mapping`,
`::test_binary_op_matches_mulensmodel_xallarap_lightcurve`).  Their frame's reference
direction is `tau_hat`; operationally:

    bigomega = phi_pi + xi_Omega_node + 180 deg
    i        = xi_inclination
    omega_*  = xi_omega_periapsis                  (the SOURCE's own orbit: no 180 flip)
    e        = xi_eccentricity,   P = xi_period,   t_0_xi = t0_par
    nu(t_0_xi) = xi_u - xi_omega_periapsis  ->  tp  (the standard anomaly chain)
    xi_semimajor_axis = a_1 / (D_S * theta_E)      (a_1 = a * m_companion / m_total)

A published `xi_*` solution therefore drops into an EXOZIPPy config through these six
lines; `examples/ob170114` is the shipped worked case (Mroz et al. 2026 Table B.1).

- Implemented in: `Lens._source_offset_series` (config keys
  `source_orbital_motion: keplerian`, `source_orbit:`),
  `mulensing/physics.source_offset_from_orbit` / `xallarap_trajectory_shift`,
  `Lens.get_magnification` (symbolic) and `op.VBMDirectMagOp(source_motion=True)`.
- Pinned by: `tests/test_xallarap.py` -- the C9 reconstruction with a displaced
  source, the t0_par anchor, symbolic-vs-Op equality, the xi_* track contract
  (`test_mm_xi_closed_form_mapping`), and -- the pin the track contract cannot
  provide -- LIGHT-CURVE parity of the composed 2L1S + xallarap magnification
  against MulensModel (`test_binary_op_matches_mulensmodel_xallarap_lightcurve`,
  review 2.6.13's regression test).

---

## 5. Mappings to other conventions

### C17 -- Skowron et al. (2011) Appendix A: IDENTICAL, as Section A.6 defines it

Section A.6 states three rules, and EXOZIPPy satisfies all three unchanged:

| Skowron Section A.6 | EXOZIPPy |
|---|---|
| "the sign of `u_0` is positive if the lens passes the source on its right" | C9 |
| "`phi_pi = atan2(pi_E_E, pi_E_N)` is the angle of lens motion, measured counter-clockwise relative to North" | C8 |
| "`alpha_0` is the angle of the lens motion, measured counter-clockwise relative to the primary-secondary axis" | C15 |

plus `q = m2/m1` (C14), `s` in Einstein radii (C13), and the geocentric `t0_par` framework
(C5, An et al. 2002; Gould 2004). So a solution reported in the Skowron reference system --
which is what that appendix exists to provide -- transfers with no transformation at all.

### C18 -- MulensModel: IDENTICAL for every trajectory parameter

`t_0`, `u_0`, `t_E`, `rho`, `pi_E_N`, `pi_E_E`, `t_0_par`, `s`, `q` and `alpha` all mean
the same thing. This is not an inference: the symbolic path is checked against
MulensModel's Op with real parallax (`tests/test_pspl_symbolic_vs_op.py`) and the direct
VBMicrolensing path against the MulensModel-backed binary Op
(`tests/test_vbm_direct_vs_mulensmodel.py`), and `op.py`'s trajectory rotation is
MulensModel's `Trajectory._get_xy` line for line.

**One caveat for anyone chasing a 180-degree offset.** MulensModel's `Trajectory` class
docstring says it "follows the conventions defined in Appendix A of Skowron et al. (2011)
except the definition of *alpha*, which is shifted by 180 deg", and its `_get_xy` carries
the same comment. That note does not hold. Skowron Section A.6 (arXiv:1101.3312, p. 34)
is quoted here verbatim, checked against the paper itself on 2026-08-18 rather than
paraphrased:

> All relative motion conventions are defined by the motion of the lens (with the source
> thought of as fixed). Thus, first, the sign of `u_0` is positive if the lens passes the
> source on its right. Second, `phi_pi = atan2(pi_E,E, pi_E,N)` is the angle of lens motion,
> measured counter-clockwise relative to North. And third, **`alpha_0` is the angle of the
> lens motion, measured counter-clockwise relative to the primary-secondary axis.**

and Section A.4 says the same thing with the ambiguity closed by hand: "the direction of
lens-source relative motion (i.e., **lens motion relative to the source**) with respect to
the binary axis (which points from primary toward secondary) ... (The angle `alpha_0` is
counter-clockwise.)". Against that definition: with
`alpha = 0` MulensModel's source runs from `x = +tau_E` to `x = -tau_E`, so the lens moves
relative to the source in the `+x` direction, which is primary-toward-secondary, which is
`alpha = 0` in Skowron's sense. (Verified numerically against the installed MulensModel,
and algebraically above.) The likeliest origin is reading Skowron's phrase "the direction
of lens-source relative motion" as the SOURCE's motion -- which is exactly the reading
Skowron closes off with the parenthetical quoted above. **EXOZIPPy follows Skowron
as written, which is also what MulensModel computes;** do not apply a 180-degree shift on
the strength of that docstring.

### C19 -- relabelling which body is the primary

Swapping the two entries of a binary `lenses:` list is a pure relabelling. The center of
mass, the trajectory and the light curve are unchanged, so

    q     -> 1/q
    alpha -> alpha + 180        (the axis reverses)
    s, t_0, u_0, t_E, rho, pi_E : unchanged

This is the transformation to apply when a paper reports `q <= 1` for a system whose
EXOZIPPy `lenses:` list is in the other order, and vice versa (C14).

### C20 -- codes that define `alpha` as a SKY POSITION ANGLE

BAGLE (Bhadra et al. 2026) defines its `alpha` as "the angle between North and the binary
axis", incrementing eastwards of North -- an absolute position angle, not an angle relative
to the trajectory. Converting requires the relative-motion direction as well:

    alpha_EXOZIPPy = phi_pi - PA(primary -> companion axis)    (mod 360)

with `phi_pi = atan2(pi_E_E, pi_E_N)` (C8). **Check which end of the axis the other code
means before using this**: BAGLE's Figure 7 caption says "the binary axis" while its
Appendix C says "the binary axis is a vector from M2 to M1", i.e. secondary-to-primary,
which would put another 180 degrees in. The structural point stands regardless: a
position-angle `alpha` and a trajectory-relative `alpha` are not related by any fixed
offset -- the conversion depends on `phi_pi`, i.e. on the fit.

### C21 -- papers that measure `alpha` to the SOURCE trajectory

Measuring the angle to the direction the SOURCE moves, rather than to the lens's motion
relative to the source, flips it by 180 degrees. Composing that with the opposite `u_0` sign
branch (C23) gives the reflection-plus-shift `alpha -> 180 - alpha`.

`examples/ob161003` is the shipped worked example. Jung et al. (2017) report
`alpha = 48.243 deg` for OGLE-2016-BLG-1003; the params file carries `alpha = 131.757 deg`
with `u_0 > 0`, i.e. `180 - alpha_paper`, and the file records that it was established by a
chi2 scan over `{+/-alpha, 180 +/- alpha} x sign(u_0)`, not by assumption. Its mirror
partner `(u_0 < 0, alpha = 180 + alpha_paper = 228.243 deg)` is the same physical solution
by C23 and would serve equally well; the fit takes `u_0 > 0` by Skowron's recommendation
that negative `u_0` be reserved for solutions including parallax.

**There is no general rule here.** `180 - alpha_paper` is a fact about that paper's
convention, established empirically for that event, and must not be applied blind to
another paper. The reliable procedure is the one the example used: scan the eight
`(+/-u_0) x (alpha, -alpha, 180 +/- alpha)` combinations at the published values of
everything else and keep the one the light curve prefers.

### C22 -- the 2018 Roman (WFIRST) Data Challenge answer key has NO mappable `alpha`

**Resolved; do not reopen.** The master file's `alpha` cannot be mapped onto the fitted
convention by any global transformation. This was measured, not assumed: for each of the 44
events, `alpha` was scanned in the MulensModel convention at the truth values of `t_0`,
`u_0`, `t_E`, `rho`, `s` and `q` with the fluxes fit linearly, giving the `alpha` the light
curve itself prefers, and every candidate transformation scattered like noise against it --

| hypothesis | circular `R` (1.0 = it IS the rule) |
|---|---|
| `fit - alpha_DC` | 0.09 |
| `fit + alpha_DC` (reflection) | 0.10 |
| either, with the galactic -> equatorial PA removed | 0.11 - 0.19 |
| either, with `PA(mu_rel)` removed (a position-angle `alpha`, C20) | 0.03 - 0.19 |

Restricting to the twelve events where the anomaly pins `alpha` hardest does not help
(`R = 0.22 / 0.41`, against `~0.29` expected from twelve random angles), so this is a
property of the answer key, not of a weak constraint or of the wrong sign branch. The
identity mapping DOES hold between MMEXOFAST and EXOZIPPy; it is the challenge's truth
table that stands apart from both.

Consequences, already implemented in `examples/DC2018/dc18_common.py`
(`ALPHA_IS_UNMAPPABLE`): the comparison table reports the fitted `alpha` with **no truth
value and no pull**, and `u_0` is compared in absolute value (the truth table carries a
trajectory-side sign the fits do not, and with `|pi_E| ~ 0.02` these events have negligible
parallax, so the sign is degenerate with `alpha`'s anyway -- event 128 shows
`(+0.1418, 308.15)` and `(-0.1418, 51.85)` giving identical chi2 to every digit). The old
sign/offset search was deleted rather than improved, because it always returned its closest
candidate and so could not fail visibly: on event 128 it printed a 2034-sigma `alpha` pull
while the fitted `alpha` (307.686) sat 0.3 degrees from the light curve's own optimum
(308.0).

### C23 -- the discrete degeneracies, and what they do to the signs

- **`(u_0, alpha) -> -(u_0, alpha)`** is EXACT for a static binary with no parallax
  (Skowron Eq. A12). This is why `alpha` and the sign of `u_0` can only ever be established
  together.
- **The ecliptic degeneracy**: `(u_0, pi_E_perp) -> -(u_0, pi_E_perp)` for a point lens
  (Skowron Eq. A6) and `(u_0, alpha, pi_E_perp) -> -(u_0, alpha, pi_E_perp)` for a static binary
  (Skowron Eq. A13). Approximate, but strong for bulge sources, which is why the point-lens
  `examples/ob140939` seeds all four Yee et al. (2015) basins rather than one.
- **Close/wide**: `log_s -> -log_s` (C13). Sampling `log_s` makes it an exact reflection of
  the domain onto itself.
- `U_0_FLOOR` is applied to `|u_0|` with the sign kept
  (`physics.apply_u_0_floor` / `floor_u_0_value`), precisely so the first of these stays
  exact.

---

## 6. Disagreements found while writing this, for the record

- **MulensModel's own docstring** contradicts Skowron A.6 on `alpha` (C18). EXOZIPPy's code
  is consistent with both Skowron-as-written and MulensModel-as-implemented; only
  MulensModel's prose is the odd one out.
- **`examples/DC2018/README.md`** described the deleted sign/offset search ("the comparison
  maps truth through a sign/offset search ... treat alpha pulls as indicative"), which
  `dc18_common.py` had already replaced with "no truth, no pull". Corrected in the same
  commit as this file.
- **RESOLVED 2026-08-18.** This document originally reported that `dc18_common.py` cited
  `examples/DC2018/dc18_alpha_convention.py` for the C22 measurement while no such file
  existed on any branch. PR #190 shipped it, at `scripts/dc18_alpha_convention.py`. That
  PR updated one of the two citations in `dc18_common.py` and left the other pointing at
  the old path; both now name the shipped location.
- **"Yee+2014"** in `MulensInstrument._abs_to_delta`'s docstring and in review item 3.6.1
  is Yee et al. **2015**, ApJ 802, 76 -- cited by the year of its 2014 arXiv posting
  (arXiv:1410.5429). The bib key added for this document is `Yee:2015`.
- **MulensModel 3.11.0's KEPLERIAN lens orbital motion contradicts its own linear mode**
  (found 2026-08-27, measured on the installed package; reproduction in the private notes
  repo, `mm_keplerian_sign_check.py`).  With `dalpha_dt = +40 deg/yr`, the linear branch of
  `ModelParameters.get_alpha` returns `d(alpha)/dt = +40` at `t_0_kep` (definitional,
  correct) while every keplerian variant returns `-40` -- the composition
  `alpha + atan2(y, x)` should be `alpha - atan2(y, x)` by C24's
  `d(PA_axis)/dt = -dalpha/dt` rule.  The circular-from-`s_z` variant additionally corrupts
  the reference geometry (`alpha(t_0_kep)` off by ~19 deg, `ds/dt` sign flipped, plus an
  IEEE `-0.0` `atan2` edge case).  NOT contaminated, and still good references: the
  `gamma_perp` property (`-dalpha_dt`, docstring and body -- the C24 identity), the LINEAR
  orbital-motion branch, and every static/parallax convention C18's parity tests pin.
  Consequence: EXOZIPPy's keplerian mode is validated against first-principles synthetic
  orbits, never against MulensModel's keplerian mode.
- **Skowron+2011 Eq. B9 displays its two rotation matrices under swapped labels**: the
  matrix printed as `R_x(beta)` rotates about the THIRD axis and the one printed as
  `R_z(beta)` about the FIRST, while the text says "around the first and third axes
  respectively" and every use (B8, B15-B17, and `cos i = R_33` in B27) requires the
  standard z-x-z Euler composition.  A typographic swap with no propagated consequence in
  the paper -- recorded so nobody transcribes B9 literally.

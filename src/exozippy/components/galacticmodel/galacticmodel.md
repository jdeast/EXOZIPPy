# galacticmodel: the frame and how it is tested

The galactic model's mass prior (IMF, FFP mass function, and the present-day mass function -- `present_day: true`, the DEFAULT since review 1.7.2 -- with its MIST turnoff and WD/NS/BH remnants, `pdmf.py`), its density and velocity mixture, linked stars and the evidence normalizer are documented in `src/exozippy/components/star/star.md` ("Stellar mass priors", "Linked stars", "The stellar distance prior"). This file holds what is specific to the coordinates: which frame the kinematic prior works in, why, and the tests that pin it. Read it before touching `physics.py`'s `line_of_sight_basis`, `galactic_xyz` or `polar_velocity`, the bar angle, or any sign in the velocity transform.

## One frame for positions and velocities

The kinematic prior is a Gaussian in galactocentric polar velocity `(v_r, v_phi, v_z)`, so it needs a position `(x, y)` (for the polar angle) and a velocity `(v_x, v_y)` in the SAME frame. The model frame is:

- **astropy's `Galactocentric`** as configured in `physics.GALACTOCENTRIC_FRAME`: origin at Sgr A*, `R0 = 8.16 kpc`, `z_sun = 25 pc`, solar motion `(10, 243, 7) km/s` (genulens's values; constants.py),
- **with x mirrored** (`physics.MODEL_FRAME_MIRROR = diag(-1, 1, 1)`), applied to positions and velocities alike.

The mirror puts the frame in genulens's orientation (Koshimoto, Baba & Bennett 2021, `Dlb2xyz`): Sun at `x ~ +R0`, `y` toward `l = 90 deg`, `z` toward the NGP, and the Sun's velocity `(vx, vy, vz)sun = (-10, 243, 7) km/s` -- exactly what genulens prints. Three things follow, and they are why this frame and not astropy's unmirrored one:

1. The density model (`branch_log_number_densities`: the bar angle `x_bar = x cos + y sin`, the disk profiles) is written in genulens's orientation, so it is unchanged.
2. The Sun at `+x` moving `+y` makes `x v_y - y v_x` positive for co-rotation, so `polar_velocity`'s `v_phi` is positive for Galactic rotation everywhere, as every branch mean assumes.
3. The origin is Sgr A*, which is genulens's default (`CenSgrA = 1`); genulens translates to it where astropy rotates to it, a difference below 15 pc over the bulge (`test_positions_track_genulens_dlb2xyz`).

`line_of_sight_basis(ra, dec)` returns `(M_rot, v0, sun_xyz, los_hat)`, all from one astropy transform: the velocity map `v = v0 + M_rot @ (K pm_ra d, K pm_dec d, rv)` and the position `sun_xyz + d * los_hat` are both exactly affine, so a linearization at 1 kpc is exact. `M_rot` is orthogonal with `det = -1` (the mirror); only `|det|` enters the velocity Jacobian. `galactic_xyz(d, *basis[2:])` is the position; it is pure arithmetic and serves numpy (hints, the position normalizer) and pytensor (the likelihood).

**Do not build a position or a velocity by hand.** Until 2026-10 (review 1.7.1) the position was built by hand in genulens's orientation (`x = R0 - d cos l cos b`) while the velocity was astropy's unmirrored one, so `v_x` entered `v_r` and `v_phi` with the wrong sign: an exactly circular orbit at 4 kpc toward ob140939 got `v_r = +47 km/s`, at `l = 90 deg`, 4 kpc `+172 km/s`, and the Sun itself moved outward at `+10 km/s`. The mismatch was born when the model was integrated (it kept the reference script's positions and took astropy's velocities); the 2026-08 fix that moved the rotation onto `v_phi` and the later matching of the two frames' `R0`/`z_sun`/`v_sun` both left the sign of `x` alone, because the tests of the time compared log-density margins through the mixture, whose hotter branches absorbed the error.

## Tests

`tests/test_galactic_frame.py` checks computed QUANTITIES against astropy, not margins:

| test | what it pins | on the two-frame code |
|---|---|---|
| circular orbits (grid l = 0, +/-5, +/-30, +/-90, 180; b = 0, +/-3; d = 0.5-12 kpc) | `v_r = v_z = 0`, `v_phi = speed` to 1e-6 km/s through the model's own path, disk speed and bulge `Omega r` | `v_r` up to 172 km/s |
| prior-mean proper motion | `expected_proper_motion` (the start hint) is a circular orbit in astropy, every branch | `v_r` 2.6-215 km/s |
| positions | astropy's Galactocentric with x mirrored, 1e-12 kpc; GC and Sun where they belong | within 2 pc (the positions were nearly right; the velocities' x was not) |
| the Sun | a star at d = 0 at rest has `(-10, 243, 7)`, `v_phi = 243`, `v_r = -10` | `v_r = +10` |
| velocity map | equals astropy's full transform (1e-9 km/s) and inverts to machine precision | x component sign-flipped |
| mirror symmetry | with the bar off, a star at `(R, phi, z)` and its twin at `(R, -phi, z)` with the same polar velocity have identical kinematic logp (1e-9 nats) | 1.03 nats apart |
| rotation sense | a co-rotating bulge star has `v_phi = +Omega r`, its counter-rotating twin `-Omega r` | 32.16 for 34.62 |

`tests/test_galactic_model.py`'s co-rotation tests (`..._prefers_corotation_over_radial_plunge`, `..._corotating_star_pays_no_velocity_penalty`) stay as physics sanity checks of the mixture. Their margins at the 2.5 kpc ob140939-field star: co-rotation beats a radial plunge by 16.19 nats (16.49 with the bug), counter-rotation by 33.24 (33.10), +/-3 sigma azimuthal by 4.44/3.94 (4.45/3.91) -- which is exactly why they could not see it.

## Fidelity to genulens

`tests/test_galactic_genulens.py` compares the prior's rate-weighted event distributions with genulens 2.0.0a4 on two sight lines (ob140939's, and Baade's window), with genulens's own source distances and IMF fed in; `scripts/make_genulens_fixture.py` records `tests/fixtures/genulens.json` and says how to build genulens. KS distances, median EXOZIPPy / genulens:

| sight line | D_L | mu_rel (heliocentric) | t_E |
|---|---|---|---|
| ob140939 (l = 6.65, b = 3.61) | 0.106 (6.17 / 6.41 kpc) | 0.014 (5.90 / 5.86 mas/yr) | 0.035 (18.6 / 19.9 d) |
| Baade (l = 1, b = -3.9) | 0.207 (6.54 / 7.17 kpc) | 0.065 (5.73 / 6.18 mas/yr) | 0.086 (18.2 / 15.4 d) |

It is a fidelity guard, not an exactness test: the 1.7.1 fix moved these by less than their noise, because the frame error lived mostly in near-disk lenses, which the event rate down-weights. It does catch gross kinematic errors (counter-rotating branches: mu_rel KS ~0.4; no rotation: 0.106 on ob140939). The largest gap, too few lenses in the near disk (D_L q10 4.1 vs 3.0 kpc toward ob140939), is a property of the simplified density model and predates 1.7.1.

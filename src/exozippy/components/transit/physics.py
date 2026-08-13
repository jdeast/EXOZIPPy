import numpy as np
import pytensor.tensor as pt
from exoplanet_core.pymc import ops as ops

# Limb-darkening physics lives on the Band component
# (calc_u1_from_kipping / calc_u2_from_kipping in band/physics.py).
# The reported jitter (calc_jitter) belongs to the shared additive noise model
# and lives on components/instrument.py, next to the jitter-variance floor
# that makes its signed square root meaningful.


def calc_planet_visible(b_p, Z_p, r_p):
    """Fraction of the planet's disk not occulted by the star: 1.0 away
    from secondary eclipse, dropping toward 0.0 at eclipse center.

    Shared by Transit.build_likelihood (the actual likelihood) and
    Transit.compile_plotters (plotting), so the two can't drift apart --
    see PR 1.a (fitthermal).

    Same occultation primitive used for the primary transit
    (quad_solution_vector), but with the planet and star roles swapped:
    the planet (radius 1, in its own units) is the occulted disk, and the
    star (radius 1/r_p, in planet radii) is the occulter, with zero limb
    darkening (uniform planetary disk: c0=1, c1=c2=0, ld_norm=pi). This is
    exofast_tran.pro's `exofast_occultquad_cel, z/abs(p), 0, 0, 1/p, mu1`
    (line 105) -- the impact parameter is rescaled by r_p too, not just
    the radius ratio.

    In front of the star (primary transit, Z_p > 0), the planet's disk is
    never occulted by the star, whatever the swapped geometry computes at
    that same sky-projected separation (b_p is small near *both*
    conjunctions; only Z's sign tells them apart). Matches
    exofast_tran.pro:103-107's planetvisible defaulting to 1 and only
    being overwritten for `secondary`.

    b_p, Z_p: same-shape tensors of any dimensionality (plotting passes
    (N,), build_likelihood (n_group, ninterp)), sky-plane separation and
    line-of-sight coordinate in units of R_*, as computed for the primary
    transit. r_p: scalar R_p/R_*.
    """
    b_swap = b_p / r_p
    r_swap = 1.0 / r_p
    sol_swap = ops.quad_solution_vector(b_swap, r_swap + pt.zeros_like(b_swap))
    # s0/pi is the visible flux fraction of the uniform planetary disk
    # (1 in the clear, 0 fully occulted).
    visible_frac = sol_swap[..., 0] / np.pi
    return pt.where(Z_p > 0.0, 1.0, visible_frac)


# --- BEER (PR 1.b): phase-curve terms, all referenced from tc (time of
# conjunction/primary transit), never from periastron (tp) -- matching
# exofast_tran.pro:128,143,151, which all use (transitbjd - tc0); tp only
# ever appears there as a fallback to *compute* tc0 when tc isn't already
# known (exofast_tran.pro:120-123), which EXOZIPPy doesn't need since
# orbit.tc is already a first-class parameter.
#
# Shared by Transit.build_likelihood and Transit.compile_plotters, same
# reasoning as calc_planet_visible above: one implementation per term, so
# a bug can't hide in the plotting path while the fitted likelihood is
# wrong (or vice versa) -- see the PR 1.a mutation-test writeup.


def calc_reflect_term(time, tc, period, reflect, planetvisible):
    """Reflected-light phase curve, in flux units (not yet diluted).

    0 at primary transit (t=tc), full `reflect` ppm at secondary eclipse
    (t=tc+P/2), and gated by the same `planetvisible` as thermal --
    reflected light comes off the planet's disk, so it's blocked the same
    way thermal emission is. exofast_tran.pro:128, algebraically expanded:
    -(reflect/2)*(cos(x)-1) == reflect*0.5*(1-cos(x)).
    """
    phase_angle = 2.0 * np.pi * (time - tc) / period
    return 1e-6 * reflect * 0.5 * (1.0 - pt.cos(phase_angle)) * planetvisible


def calc_ellipsoidal_factor(time, tc, period, ellipsoidal):
    """Multiplicative ellipsoidal-variation factor (not additive, and not
    gated by planetvisible -- this is the star's own tidally-distorted
    shape, present regardless of whether the planet's disk is occulted).

    Half the orbital period (exofast_tran.pro:143): dimmest at both
    conjunctions (t=tc, t=tc+P/2; the star's tidal bulge points along our
    line of sight, presenting its narrowest silhouette), brightest at both
    quadratures (t=tc+-P/4; bulge viewed broadside).
    """
    phase_angle = 2.0 * np.pi * (time - tc) / (period / 2.0)
    return 1.0 - 1e-6 * ellipsoidal * pt.cos(phase_angle)


def calc_beam_term(time, tc, period, beam):
    """Doppler beaming phase curve, in flux units. Not gated by
    planetvisible -- a relativistic boost of the star's own light from its
    reflex RV motion, unrelated to the planet's occultation state.

    An odd (sine) function of phase (exofast_tran.pro:151), unlike
    reflect/ellipsoidal's even (cosine) forms: it tracks the star's
    directional line-of-sight velocity, zero at both conjunctions
    (t=tc, t=tc+P/2, where RV crosses zero) and extremal with opposite
    sign at the two quadratures.
    """
    phase_angle = 2.0 * np.pi * (time - tc) / period
    return 1e-6 * beam * pt.sin(phase_angle)

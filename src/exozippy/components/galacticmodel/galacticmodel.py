import logging

import astropy.units as u
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from astropy.coordinates import CartesianDifferential, Galactocentric, SkyCoord
from scipy.special import erf, erfc

from exozippy.components.component import Component
from exozippy.constants import (
    BULGE_BAR_ANGLE,
    BULGE_CENTRAL_NUMBER_DENSITY,
    BULGE_DENSITY_X_0,
    BULGE_DENSITY_Y_0,
    BULGE_DENSITY_Z_0,
    BULGE_GAMMA,
    BULGE_RC,
    BULGE_RC_WIDTH,
    BULGE_ROTATION_ANGULAR_VELOCITY,
    BULGE_VELOCITY_SIGMA_1,
    BULGE_VELOCITY_SIGMA_2,
    BULGE_VELOCITY_SIGMA_3,
    DISK_LOCAL_NUMBER_DENSITY,
    DISK_RDBREAK,
    DISK_ROTATION_VELOCITY,
    DISK_SCALE_HEIGHT,
    DISK_SCALE_LENGTH,
    DISK_VELOCITY_SIGMA_U,
    DISK_VELOCITY_SIGMA_V,
    DISK_VELOCITY_SIGMA_W,
    K_VEL_CONVERSION,
    SALPETER_IMF_SLOPE,
    SUN_GALCEN_V,
    SUN_GC_DISTANCE,
    SUN_Z_OFFSET,
    THICK_DISK_LOCAL_NUMBER_DENSITY,
    THICK_DISK_ROTATION_VELOCITY,
    THICK_DISK_SCALE_HEIGHT,
    THICK_DISK_SCALE_LENGTH,
    THICK_DISK_VELOCITY_SIGMA_U,
    THICK_DISK_VELOCITY_SIGMA_V,
    THICK_DISK_VELOCITY_SIGMA_W,
)

"""
This class implements a mixture model of the thin disk, thick disk, and
bulge/bar, enforcing the kinematics and density of the galaxy.
It's a simplified version of the model defined in https://ui.adsabs.harvard.edu/abs/2021ApJ...917...78K/abstract.
and codified here https://github.com/nkoshimoto/genulens
This class distills genulens's core functionality into a component for EXOZIPPy

Fidelity notes (vs genulens, reviewed 2026-08): the thin disk collapses
genulens's 7 age bins into one exponential layer with its B14disk-mode
(Bennett et al. 2014) kinematics instead of the Shu DF; the bar keeps a
plain-ellipsoid exp(-r_s/2) profile (not the K21 super-ellipsoid + X-shape)
but carries genulens's outer cylindrical cutoff and VVV-box-budget
normalization; NSD, stellar halo, and the bar's spatial sigma gradients /
streaming motion are omitted.  All three branch weights are genulens's
NUMBER-density channel on one absolute scale, so the logsumexp mixture
weight is physical.  The event-rate selection factor (theta_E * mu_rel) is
deliberately NOT here -- it lives in the lens component, so this prior
stays microlensing-agnostic.
"""

logger = logging.getLogger(__name__)

# One consistent galactocentric frame for the velocity transform, matching
# the density grid's R0/z_sun (genulens: R0 = 8160 pc, zsun = 25 pc,
# vsun = (10, 243, 7) km/s toward-GC/rotation/up).  Astropy's default
# Galactocentric (R0 = 8.122 kpc) would put the velocities in a slightly
# different frame than the densities.
GALACTOCENTRIC_FRAME = Galactocentric(
    galcen_distance=SUN_GC_DISTANCE * u.kpc,
    z_sun=SUN_Z_OFFSET * u.kpc,
    galcen_v_sun=CartesianDifferential(list(SUN_GALCEN_V) * (u.km / u.s)),
)


def _sampled_bounds(param):
    """(lower, upper) of a Parameter's hard support as float arrays.

    Returns None when the bounds are missing, non-finite, or symbolic, which
    the IMF normalizers treat as "leave this prior unnormalized" -- a
    constant offset never changes the sampling, so a bound the component
    cannot read is not worth failing a fit over.
    """
    try:
        # atleast_1d: a scalar bound must still broadcast against the
        # (n_star,) logmass vector, and np.select wants real arrays.
        lower = np.atleast_1d(np.asarray(param.lower, dtype=float))
        upper = np.atleast_1d(np.asarray(param.upper, dtype=float))
    except (AttributeError, TypeError, ValueError):
        return None

    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        return None
    return lower, upper


def _unnormalized_warning():
    logger.warning(
        "[galacticmodel] Cannot read finite log10-mass bounds; the IMF "
        "prior is left unnormalized (harmless for sampling, but its logp "
        "is offset by an unknown constant and is not comparable across "
        "IMF choices)."
    )
    return 0.0


def _power_law_log_norm(k, param):
    """log of the normalizer of p(x) ~ 10^(k x) over x in [lower, upper].

    ``param`` is the sampled log10-mass Parameter; its hard bounds are the
    support, so

        Z = int_lo^hi 10^(k x) dx = (10^(k hi) - 10^(k lo)) / (k ln10)

    which is finite for any k as long as both bounds are.  Evaluated as
    logdiffexp(., .) - log(|k| ln10) so the exponentials never overflow
    (10^(k*lo) is 1e12 already at the defaults.yaml floor of -9 dex).
    """
    bounds = _sampled_bounds(param)
    if bounds is None:
        return _unnormalized_warning()
    lower, upper = bounds

    if k == 0.0:  # uniform in log10 M
        return np.log(upper - lower)

    ln10 = np.log(10.0)
    a, b = k * upper * ln10, k * lower * ln10
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    return hi + np.log1p(-np.exp(lo - hi)) - np.log(abs(k) * ln10)


def _lognormal_log_norm(mu, sigma, param):
    """log of the normalizer of p(x) ~ exp(-0.5((x-mu)/sigma)^2) over the
    Parameter's hard support [lower, upper].

    With u = (x - mu)/sigma,

        Z = int_lo^hi exp(-u^2/2) sigma du
          = sigma sqrt(2 pi) [Phi(u_hi) - Phi(u_lo)]

    so log Z = log(sigma) + 0.5 log(2 pi) + log(Phi(u_hi) - Phi(u_lo)).

    The bracket is computed from erf/erfc rather than as a difference of
    CDFs, choosing the branch whose two terms cannot cancel:
      - both bounds above the mean -> erfc(z_lo) - erfc(z_hi), two small
        positive numbers (a Phi difference would be 1-eps minus 1-eps',
        which throws away every significant digit once the bounds are a few
        sigma out; at 5 and 6 sigma only ~7 digits survive in float64);
      - both below -> the mirrored erfc form;
      - straddling -> erf(z_hi) - erf(z_lo), whose terms have opposite
        signs, so subtracting them is exact.
    All three are evaluated and selected elementwise; none can overflow
    (erf is bounded, erfc saturates at 2 or underflows to 0).
    """
    bounds = _sampled_bounds(param)
    if bounds is None:
        return _unnormalized_warning()
    lower, upper = bounds

    # z = u / sqrt(2), so Phi(u) = 0.5 erfc(-z)
    z_lo = (lower - mu) / (sigma * np.sqrt(2.0))
    z_hi = (upper - mu) / (sigma * np.sqrt(2.0))

    mass = 0.5 * np.select(
        [z_lo >= 0.0, z_hi <= 0.0],
        [
            erfc(z_lo) - erfc(z_hi),
            erfc(-z_hi) - erfc(-z_lo),
        ],
        default=erf(z_hi) - erf(z_lo),
    )

    if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
        # Support so far into a tail that its probability mass underflows.
        return _unnormalized_warning()

    return np.log(sigma) + 0.5 * np.log(2.0 * np.pi) + np.log(mass)


class GalacticModel(Component):
    # The only mass function implemented below.  ``IMF:`` is validated
    # against this in __init__ rather than silently ignored: the key selected
    # KROUPA_IMF_SLOPE / SALPETER_IMF_SLOPE for a power-law prior that had
    # not been applied since the Chabrier lognormal replaced it, so every
    # `IMF: Salpeter` in a config was a no-op.  Kroupa is still unsupported:
    # it is a BROKEN power law (alpha = 1.3 below 0.5 Msun, 2.3 above) and
    # needs a piecewise, continuity-matched density, not a single slope --
    # KROUPA_IMF_SLOPE is only its low-mass segment.
    SUPPORTED_IMFS = ("chabrier", "salpeter")

    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Galactic Prior"
        if self.n_elements != 1:
            # Only config[0] is ever read (imf, anchor_idx), and the extra
            # blocks used to leak into the likelihood as a broadcast: with
            # 2 blocks and 1 star the whole kinematic prior was counted
            # TWICE, and with 2 blocks and 3 stars it raised a bare shape
            # error.  One galactic prior describes one line of sight.
            raise ValueError(
                f"galacticmodel takes exactly one config block, got "
                f"{self.n_elements}.  It is a single prior on the line of "
                f"sight shared by every star; use 'anchor_idx' to choose "
                f"which star's (ra, dec) defines it."
            )
        imf = str(self.config[0].get("IMF", "chabrier")).lower()
        if imf not in self.SUPPORTED_IMFS:
            raise ValueError(
                f"galacticmodel IMF '{self.config[0].get('IMF')}' is not "
                f"implemented.  Supported: {', '.join(self.SUPPORTED_IMFS)} "
                f"('chabrier' = Chabrier 2003 system IMF, a lognormal in "
                f"log10 mass; 'salpeter' = Salpeter 1955 power law).  "
                f"'Kroupa' was accepted but silently ignored before 2026-08."
            )
        self.imf = imf
        self.anchor_idx = self.config[0].get("anchor_idx", 0)

    @property
    def prefix(self):
        return "galacticmodel"

    @classmethod
    def config_schema(cls):
        return [
            {
                "key": "IMF",
                "kind": "option",
                "accepts": list(cls.SUPPORTED_IMFS),
                "required": False,
                "doc": (
                    "Initial mass function for the stellar mass prior "
                    "(default 'chabrier'): 'chabrier' is the Chabrier 2003 "
                    "system IMF, a lognormal in log10 mass; 'salpeter' is "
                    "the Salpeter 1955 power law dN/dM ~ M^-2.35.  Both are "
                    "normalized over the sampled logmass bounds, so their "
                    "logp values are directly comparable.  Anything else "
                    "raises."
                ),
            },
            {
                "key": "anchor_idx",
                "kind": "option",
                "accepts": None,
                "required": False,
                "doc": (
                    "Index of the star whose (ra, dec) defines the line of "
                    "sight for the density/kinematic prior (default 0). One "
                    "galacticmodel block describes one sight line."
                ),
            },
        ]

    def register_parameters(self, system):
        """No parameters to sample! Just an empty manifest."""
        self.manifest = {}

    def build_likelihood(self, model, system):
        stars = system.star

        # 1. Pre-compute the transformation matrix using Astropy, ONCE, from
        # the anchor star's initial RA/Dec.  This is a property of the line
        # of sight, not of a component element: the loop this replaced ran
        # over self.n_elements (the number of galacticmodel BLOCKS) while
        # indexing self.anchor_idx every time, so it stacked n_elements
        # identical copies and then relied on those copies broadcasting
        # against the (n_star,) sampled vectors -- which silently doubled
        # the whole prior for 2 blocks and 1 star, and raised a bare shape
        # error whenever n_blocks and n_stars disagreed and neither was 1.
        # Keeping the matrix 2-D (and the direction cosines scalar) lets it
        # broadcast over any number of stars by construction.
        ra_rad = float(np.atleast_1d(stars.ra.initval)[self.anchor_idx])
        dec_rad = float(np.atleast_1d(stars.dec.initval)[self.anchor_idx])

        sc = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad)
        d = 1.0  # kpc, arbitrary distance for velocity basis projection
        pm_1 = 1.0 / (K_VEL_CONVERSION * d)  # mas/yr

        def _basis(pm_ra_cosdec, pm_dec, rv):
            return SkyCoord(
                ra=sc.ra,
                dec=sc.dec,
                distance=d * u.kpc,
                pm_ra_cosdec=pm_ra_cosdec * u.mas / u.yr,
                pm_dec=pm_dec * u.mas / u.yr,
                radial_velocity=rv * u.km / u.s,
            ).transform_to(GALACTOCENTRIC_FRAME)

        gal0 = _basis(0, 0, 0)
        gal1 = _basis(pm_1, 0, 0)
        gal2 = _basis(0, pm_1, 0)
        gal3 = _basis(0, 0, 1)

        v0_arr = np.array([gal0.v_x.value, gal0.v_y.value, gal0.v_z.value])
        v1 = (
            np.array([gal1.v_x.value, gal1.v_y.value, gal1.v_z.value]) - v0_arr
        )
        v2 = (
            np.array([gal2.v_x.value, gal2.v_y.value, gal2.v_z.value]) - v0_arr
        )
        v3 = (
            np.array([gal3.v_x.value, gal3.v_y.value, gal3.v_z.value]) - v0_arr
        )

        l_rad = sc.galactic.l.rad
        b_rad = sc.galactic.b.rad

        # Convert to tensors for graph injection
        M_rot = pt.as_tensor_variable(np.column_stack([v1, v2, v3]))  # (3, 3)
        v0 = pt.as_tensor_variable(v0_arr)  # (3,)
        cosl_cosb = pt.as_tensor_variable(np.cos(l_rad) * np.cos(b_rad))
        sinl_cosb = pt.as_tensor_variable(np.sin(l_rad) * np.cos(b_rad))
        sinb = pt.as_tensor_variable(np.sin(b_rad))

        # 2. PyTensor Math Helpers
        def get_galactocentric_velocity(dist_kpc, pm_ra, pm_dec, rv_ms):
            v_alpha_kms = K_VEL_CONVERSION * pm_ra * dist_kpc
            v_delta_kms = K_VEL_CONVERSION * pm_dec * dist_kpc
            v_rad_kms = rv_ms / 1e3
            v_icrs = pt.stack(
                [v_alpha_kms, v_delta_kms, v_rad_kms], axis=1
            )  # (N, 3)
            v_gal_offset = (M_rot @ v_icrs[:, :, None]).squeeze(-1)
            return v0 + v_gal_offset

        def get_galactic_xyz(dist):
            # genulens Dlb2xyz: the Sun sits SUN_Z_OFFSET above the plane,
            # handled as a small rotation of z by bsun = z_sun/R0 (so a
            # star at d=0 lands at z = +z_sun).  x and y keep the flat
            # convention (Sun at x = +R0, GC at the origin).
            x = SUN_GC_DISTANCE - dist * cosl_cosb
            y = dist * sinl_cosb
            bsun = SUN_Z_OFFSET / SUN_GC_DISTANCE
            z = dist * sinb * np.cos(bsun) + x * np.sin(bsun)
            return x, y, z

        def get_polar_velocity(x, y, r, v_x, v_y):
            cos_phi = x / r  # unitless
            sin_phi = y / r  # unitless
            v_r = v_y * sin_phi + v_x * cos_phi
            v_phi = v_y * cos_phi - v_x * sin_phi
            return v_r, v_phi

        # match the IMF.  The key is validated in __init__, so every branch
        # here is a supported option.  BOTH are densities in the SAMPLED
        # coordinate x = log10(M), not in M, and both are summed over every
        # modeled star (lens and source alike).
        if self.imf == "salpeter":
            # Salpeter (1955): dN/dM ~ M^-alpha, alpha = 2.35.  Change of
            # variables to x = log10(M)  (M = 10^x, dM/dx = M ln10):
            #   dN/dx = (dN/dM)(dM/dx) = M^-alpha * M ln10
            #         = ln10 * M^(1-alpha) = ln10 * 10^((1-alpha)x)
            #   log p(x) = (1 - alpha) * ln10 * x + const
            # SALPETER_IMF_SLOPE is the SIGNED MASS-SPACE exponent (-alpha),
            # NOT an already-converted log-space slope, so the slope in the
            # sampled coordinate is SALPETER_IMF_SLOPE + 1 = -1.35, i.e.
            # -1.35*ln10 = -3.108 nats per dex of mass.  A linear tilt is
            # fine for NUTS now that the bounded-coordinate transform is
            # sound, and the support IS bounded (star.logmass carries hard
            # lower/upper), so the density is proper and normalizable.
            k = SALPETER_IMF_SLOPE + 1.0
            imf_logp = k * np.log(10.0) * stars.logmass.value - (
                _power_law_log_norm(k, stars.logmass)
            )
        else:
            ### Chabrier 2003 System IMF parameters
            log_Mc = np.log10(0.22)
            sigma_imf = 0.57

            # This provides beautiful, constant curvature (-1 / sigma^2) for
            # NUTS.  For high mass ( > 1 M_sun), you smoothly match it to a
            # Salpeter tail but the low-mass end is usually where the
            # unconstrained NUTS particles fall into the abyss.
            #
            # Normalized over the SAME star.logmass support the power law
            # uses, so both IMFs are proper densities and switching between
            # them moves logp by a meaningful amount rather than by an
            # arbitrary offset.  The truncated-lognormal constant is
            #   log(sigma) + 0.5 log(2 pi) + log(Phi(u_hi) - Phi(u_lo)),
            # +0.3568 nats per star at the defaults.yaml bounds (so every
            # archived logp for the default config shifts down by that much
            # times the number of stars -- see the re-pinned baselines in
            # tests/test_runaway_logp_regression.py).
            imf_logp = -0.5 * pt.sqr(
                (stars.logmass.value - log_Mc) / sigma_imf
            ) - _lognormal_log_norm(log_Mc, sigma_imf, stars.logmass)

        pm.Potential(f"{self.prefix}.imf_prior", pt.sum(imf_logp))
        ######

        # even though non-physical values will be rejected
        # we still need to be able to calculate a likelihood,
        distance = pt.maximum(stars.distance.value, 1e-3) / 1e3  # kpc
        x, y, z = get_galactic_xyz(distance)
        z_smooth = pt.sqrt(z**2 + 1e-6)
        r = pt.sqrt(x**2 + y**2 + 1e-6)

        vel = get_galactocentric_velocity(
            distance, stars.pm_ra.value, stars.pm_dec.value, stars.rv.value
        )  # km/s
        v_x, v_y, v_z = vel[:, 0], vel[:, 1], vel[:, 2]
        v_r, v_phi = get_polar_velocity(x, y, r, v_x, v_y)

        # match the density distribution of the galaxy
        cos_bar = np.cos(BULGE_BAR_ANGLE)
        sin_bar = np.sin(BULGE_BAR_ANGLE)
        x_bar = x * cos_bar + y * sin_bar
        y_bar = -x * sin_bar + y * cos_bar

        def hinge(t):
            # Smooth max(t, 0) (C-infinity), transition width ~50 pc in
            # kpc units -- keeps the plateau/cutoff kinks differentiable.
            return 0.5 * (t + pt.sqrt(t * t + 0.0025))

        # Each mixture branch must carry its own normalization: constants
        # cancel within a single Potential but NOT across the logsumexp.
        # Velocity Gaussians contribute -log(sigma1*sigma2*sigma3) (the
        # shared (2*pi)^(3/2) is identical in all branches and dropped);
        # the density anchors (stars/pc^3, genulens's number-density
        # channel, see constants.py) set the physical population ratios.

        # 1. Thin disk (Spatial + Kinematic)
        # Radial profile: exponential with Rd = 2.6 kpc, held FLAT inside
        # R = DISK_RDBREAK (genulens's DISK=2 inner plateau) -- a plain
        # exponential over-weights near-GC disk lenses by ~3x at R = 1
        # kpc.  Anchored to the local (R0, midplane) number density.
        # Galactic rotation is AZIMUTHAL: the circular-speed offset belongs
        # on v_phi (paired with the azimuthal dispersion sigma_V) and the
        # radial component v_r is zero-centered (sigma_U). These centers
        # were swapped until 2026-08: the prior penalized an EXACTLY
        # circular co-rotating orbit by ~49 nats and was maximized by a
        # star plunging radially at the rotation speed (pinned by
        # tests/test_galactic_model.py). On examples/ob140939 the swap
        # gave the anti-rotation parallax solution +5 nats and flipped the
        # mode weights to 0.98/0.02 AGAINST the Yee et al. 2015 proper-
        # motion-preferred solution. Sign convention: v_phi as computed by
        # get_polar_velocity is positive for co-rotation at every position
        # (verified against astropy Galactocentric), so the center is
        # +DISK_ROTATION_VELOCITY.
        r_beyond_sun = SUN_GC_DISTANCE - DISK_RDBREAK
        log_dens_thin = (
            np.log(DISK_LOCAL_NUMBER_DENSITY)
            - (hinge(r - DISK_RDBREAK) - r_beyond_sun) / DISK_SCALE_LENGTH
            - z_smooth / DISK_SCALE_HEIGHT
        )
        log_vel_thin = (
            (-0.5 / DISK_VELOCITY_SIGMA_U**2) * v_r**2
            + (-0.5 / DISK_VELOCITY_SIGMA_V**2)
            * (v_phi - DISK_ROTATION_VELOCITY) ** 2
            + (-0.5 / DISK_VELOCITY_SIGMA_W**2) * v_z**2
        )
        L_thin = (
            log_dens_thin
            + log_vel_thin
            - np.log(
                DISK_VELOCITY_SIGMA_U
                * DISK_VELOCITY_SIGMA_V
                * DISK_VELOCITY_SIGMA_W
            )
        )

        # 2. Thick disk (Spatial + Kinematic) -- same plateau, its own
        # scale length/height; kinematics include the asymmetric drift
        # (mean rotation 170 km/s) and hotter dispersions.
        log_dens_thick = (
            np.log(THICK_DISK_LOCAL_NUMBER_DENSITY)
            - (hinge(r - DISK_RDBREAK) - r_beyond_sun)
            / THICK_DISK_SCALE_LENGTH
            - z_smooth / THICK_DISK_SCALE_HEIGHT
        )
        log_vel_thick = (
            (-0.5 / THICK_DISK_VELOCITY_SIGMA_U**2) * v_r**2
            + (-0.5 / THICK_DISK_VELOCITY_SIGMA_V**2)
            * (v_phi - THICK_DISK_ROTATION_VELOCITY) ** 2
            + (-0.5 / THICK_DISK_VELOCITY_SIGMA_W**2) * v_z**2
        )
        L_thick = (
            log_dens_thick
            + log_vel_thick
            - np.log(
                THICK_DISK_VELOCITY_SIGMA_U
                * THICK_DISK_VELOCITY_SIGMA_V
                * THICK_DISK_VELOCITY_SIGMA_W
            )
        )

        # 3. Bulge/bar (Spatial + Kinematic), anchored at its center.
        # The Gaussian cylindrical cutoff beyond BULGE_RC (genulens Rc/
        # srob) is what confines the bar: without it the shallow
        # exp(-r_s/2) profile is still 9% of central at a 4-kpc lens
        # distance (vs ~4e-6 in genulens), flooding the disk range with
        # spurious bulge stars.
        r_bulge_coord = pt.sqrt(
            (x_bar / BULGE_DENSITY_X_0) ** 2
            + (y_bar / BULGE_DENSITY_Y_0) ** 2
            + (z / BULGE_DENSITY_Z_0) ** 2
        )
        log_dens_bulge = (
            np.log(BULGE_CENTRAL_NUMBER_DENSITY)
            - 0.5 * r_bulge_coord
            - 0.5 * (hinge(r - BULGE_RC) / BULGE_RC_WIDTH) ** 2
        )
        # Bulge cylindrical rotation is azimuthal too (same fix as above).
        bulge_rot = BULGE_ROTATION_ANGULAR_VELOCITY * r
        log_vel_bulge = (
            (-0.5 / BULGE_VELOCITY_SIGMA_1**2) * v_r**2
            + (-0.5 / BULGE_VELOCITY_SIGMA_2**2) * (v_phi - bulge_rot) ** 2
            + (-0.5 / BULGE_VELOCITY_SIGMA_3**2) * v_z**2
        )
        L_bulge = (
            log_dens_bulge
            + log_vel_bulge
            - np.log(
                BULGE_VELOCITY_SIGMA_1
                * BULGE_VELOCITY_SIGMA_2
                * BULGE_VELOCITY_SIGMA_3
            )
        )

        volume_element = 2.0 * pt.log(distance * 1000.0)

        # The velocity Gaussians are evaluated in km/s while the sampled
        # coordinates are (pm_ra, pm_dec, rv); the change of variables
        # v = M_rot @ (K*pm_ra*d, K*pm_dec*d, rv/1e3) + v0 needs
        # |det dv/d(pm_ra, pm_dec, rv)| = (K*d)^2 * 1e-3 (M_rot is a pure
        # rotation, det = 1; the constant 1e-3 from m/s -> km/s is
        # dropped).  Without the +2*log(d) the intended prior
        # rho * d^2 * f(v) * (K*d)^2 was applied as rho * d^2 * f(v),
        # under-weighting large distances by d^2.
        velocity_jacobian = 2.0 * pt.log(K_VEL_CONVERSION * distance)

        # 4. Combine them using LogSumExp
        # log(exp(L_thin) + exp(L_thick) + exp(L_bulge))
        kinematic_penalty = pt.sum(
            pm.math.logsumexp(pt.stack([L_thin, L_thick, L_bulge]), axis=0)
            + volume_element
            + velocity_jacobian
        )
        pm.Potential(f"{self.prefix}.kinematic_prior", kinematic_penalty)

        # check parameters for debugging
        # pm.Deterministic(f"{self.prefix}.gal_x", x)
        # pm.Deterministic(f"{self.prefix}.gal_y", y)
        # pm.Deterministic(f"{self.prefix}.gal_z", z)
        # pm.Deterministic(f"{self.prefix}.gal_r", r)
        # pm.Deterministic(f"{self.prefix}.v_x", v_x)
        # pm.Deterministic(f"{self.prefix}.v_y", v_y)
        # pm.Deterministic(f"{self.prefix}.v_z", v_z)
        # pm.Deterministic(f"{self.prefix}.v_r", v_r)
        # pm.Deterministic(f"{self.prefix}.v_phi", v_phi)
        # pm.Deterministic(f"{self.prefix}.L_disk", log_dens_disk + log_vel_disk)
        # pm.Deterministic(f"{self.prefix}.L_bulge", log_dens_bulge + log_vel_bulge)
        # pm.Deterministic(f"{self.prefix}.log_imf_weight", log_m_weight)

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass

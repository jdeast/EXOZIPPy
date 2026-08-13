import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import erf, erfc

from exozippy.components.component import Component

# One owner for "read a Parameter's hard support" -- star.py's volume prior
# normalizes over star.distance's bounds exactly the way the IMF branches
# below normalize over star.logmass's.
from exozippy.components.parameter import sampled_bounds as _sampled_bounds
from exozippy.constants import (
    BULGE_BAR_ANGLE,
    BULGE_CENTRAL_NUMBER_DENSITY,
    BULGE_DENSITY_X_0,
    BULGE_DENSITY_Y_0,
    BULGE_DENSITY_Z_0,
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
    SUN_GC_DISTANCE,
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

# The frame, the line-of-sight basis and the position transform live in
# physics.py (the numpy layer) so the start-value hints in
# mulensing/lens.py use exactly the code this likelihood uses.
from .physics import (  # noqa: E402
    GALACTOCENTRIC_FRAME,
    line_of_sight_basis,
)
from .physics import (  # noqa: E402
    galactic_xyz as _galactic_xyz_np,
)


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

    ``k`` may be a scalar (one slope shared by every star, as Salpeter is) or
    a per-star array (the FFP slope is per star and user-settable, and each
    star carries its own bounds).  Both branches of the k == 0 split are
    always evaluated and then selected, which is safe because neither can
    produce a NaN: upper > lower always, and the power-law branch substitutes
    k = 1 where k is zero.
    """
    bounds = _sampled_bounds(param)
    if bounds is None:
        return _unnormalized_warning()
    lower, upper = bounds

    ln10 = np.log(10.0)
    k = np.asarray(k, dtype=float)
    is_flat = k == 0.0  # uniform in log10 M
    k_safe = np.where(is_flat, 1.0, k)

    a, b = k_safe * upper * ln10, k_safe * lower * ln10
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    power_law = hi + np.log1p(-np.exp(lo - hi)) - np.log(np.abs(k_safe) * ln10)
    return np.where(is_flat, np.log(upper - lower), power_law)


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


def ffp_logmass_logp(logmass, alpha, param):
    """Free-floating-planet mass prior, as a density in x = log10(M/Msun).

    THIS IS THE SEAM.  If you have your own free-floating-planet mass
    function, replace the two lines below the comment banner and nothing else:
    every caller (``GalacticModel.build_likelihood``), the per-star selection
    (``star:``'s ``mass_function: ffp``), the floor, and the tests reach the
    functional form only through here.

    Functional form
    ---------------
    Sumi et al. 2023 (AJ 166, 108; arXiv:2303.08280), MOA-II 9-year survey:

        dN/dlog M = Z * (M / M_norm)^(-alpha)  dex^-1 star^-1

    with alpha = 0.96 (+0.47/-0.27), Z = 2.18 (+0.52/-1.40) dex^-1 star^-1 and
    M_norm = 8 M_Earth, fit over 0.33 < M/M_Earth < 6660 (1e-6 < M/Msun <
    0.02).  ``alpha`` here is that exponent, POSITIVE for the observed
    rising-toward-low-mass slope; it is per star and user-settable (``star:``
    key ``mass_function: {kind: ffp, alpha: ...}``) because this measurement
    is uncertain and Roman will revise it.

    Change of variables
    -------------------
    None -- and that is the whole point of preferring this parameterization.
    The measurement is ALREADY a density in log mass, and x = log10(M) is
    already the sampled coordinate (``star.logmass``), so

        log p(x) = -alpha * ln10 * x + const,

    with no Jacobian.  Contrast the Salpeter branch in ``build_likelihood``,
    which is quoted as dN/dM and therefore picks up |dM/dx| = M ln10, turning
    its exponent from -alpha into (1 - alpha).  If you substitute a mass
    function quoted as dN/dM, you must put that Jacobian back.

    Normalization
    -------------
    From the sampled support, not from the paper.  ``param`` is the
    ``star.logmass`` Parameter; ``_power_law_log_norm`` integrates 10^(k x)
    over its hard [lower, upper] bounds, which makes this a proper density
    directly comparable with the IMF branches over the same support.  Both Z
    and M_norm are therefore unusable here: each contributes only an additive
    constant that the normalizer cancels exactly.  Z is an abundance (a rate
    per star), and would matter only to a model that weighed the FFP and
    stellar lens populations against each other -- this prior conditions on
    the lens being an FFP, so it cannot see a rate.

    The support's lower end is a real modeling choice -- a rising density
    piles prior mass against it, 90% of it within 1/alpha dex -- and it is
    deliberately left to the user, as ``star.<name>.logmass``'s ordinary
    ``lower`` bound.  Nothing clamps it: sub-stellar masses are this
    relation's domain, so a floor imposed here would gate exactly the models
    it exists to express.  ``Star._warn_ffp_logmass_bound`` says so out loud
    when the bound is still the default.
    """
    # ---- substitute your own mass function here -------------------------
    # k is the exponent of the density in the SAMPLED coordinate, p(x) ~
    # 10^(k x); the normalizer is the integral of that over the support.
    k = -np.asarray(alpha, dtype=float)
    return k * np.log(10.0) * logmass - _power_law_log_norm(k, param)
    # ---------------------------------------------------------------------


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
                    "raises.  This is the STELLAR mass prior: a star may opt "
                    "out of it individually with the star block's "
                    "'mass_function: ffp' (the free-floating-planet mass "
                    "function), which is how an FFP lens and a stellar source "
                    "get different mass priors in the same fit."
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

    # ------------------------------------------------------------------
    # Reporting: tell the tables what these potentials are.
    #
    # Both potentials below act on parameters this component does not own
    # and does not sample -- they are `pm.Potential`s over star.logmass and
    # over (star.distance, pm_ra, pm_dec, rv).  `Parameter.get_prior_str`
    # can only see a parameter's OWN fields, so without these declarations
    # every one of them is reported as "Uniform" (a bounded element with no
    # sigma), which is exactly the prior these terms replace.  See
    # parameter.PriorContribution.
    # ------------------------------------------------------------------

    def _declare_mass_prior(self, stars, ffp_mask):
        """Describe the imf_prior potential, per star.

        Two descriptions, because the potential really is two densities: the
        stellar IMF selected by ``IMF:``, and the FFP mass function on the
        stars whose ``mass_function:`` opted out of it.  Both are normalized
        over star.logmass's own support, so both supersede the uniform.
        """
        logmass = getattr(stars, "logmass", None)
        if logmass is None:
            return

        if self.imf == "salpeter":
            imf_latex = r"Salpeter (1955) IMF in $\log_{10} M$"
            imf_text = "Salpeter (1955) IMF in log10 M"
        else:
            imf_latex = r"Chabrier (2003) IMF in $\log_{10} M$"
            imf_text = "Chabrier (2003) IMF in log10 M"

        imf_elements = np.nonzero(~ffp_mask)[0] if ffp_mask.size else None
        if imf_elements is None or len(imf_elements):
            logmass.add_prior_contribution(
                latex=imf_latex,
                text=imf_text,
                elements=imf_elements,
                supersedes_bounds=True,
            )

        # The FFP slope is per star, so a system with two differently-sloped
        # FFPs gets two contributions, each naming its own elements.
        alphas = np.atleast_1d(getattr(stars, "ffp_alpha", []))
        for i in np.nonzero(ffp_mask)[0]:
            alpha = float(alphas[i]) if i < alphas.size else float("nan")
            logmass.add_prior_contribution(
                latex=(
                    r"Sumi et al. (2023) FFP mass function "
                    rf"($\alpha = {alpha:g}$)"
                ),
                text=f"Sumi+2023 FFP mass function (alpha = {alpha:g})",
                elements=[int(i)],
                supersedes_bounds=True,
            )

    def _add_mass_prior_prose(self, system, stars, ffp_mask):
        """Modeling-draft sentences for the imf_prior potential.

        Mirrors ``_declare_mass_prior``'s per-star split: the stellar IMF
        sentence names its stars only when some star opted onto the FFP
        mass function, and each FFP star gets the Sumi slope actually
        applied.  Declared next to the potential (declare-at-site rule).
        """
        from ...outputs.prose import get_collector, join_names
        from ...outputs.texutils import latex_escape

        prose = get_collector(system)

        if getattr(stars, "logmass", None) is None:
            return
        # Read the star inventory defensively: unit tests drive this
        # potential through minimal mock star components.
        star_names = [str(nm) for nm in (getattr(stars, "names", None) or [])]
        n = max(
            int(getattr(stars, "n_elements", len(star_names))),
            len(star_names),
            int(ffp_mask.size),
        )
        if len(star_names) < n:
            star_names += [str(i) for i in range(len(star_names), n)]
        mask = ffp_mask if ffp_mask.size else np.zeros(n, dtype=bool)
        imf_cite = (
            r"\citet{Salpeter:1955} power-law"
            if self.imf == "salpeter"
            else r"\citet{Chabrier:2003} lognormal"
        )
        imf_names = [nm for nm, f in zip(star_names, mask) if not f]
        if imf_names:
            target = (
                "each modeled star"
                if len(imf_names) == n
                else (
                    ("star " if len(imf_names) == 1 else "stars ")
                    + join_names(latex_escape(nm) for nm in imf_names)
                )
            )
            prose.add(
                f"We applied the {imf_cite} initial mass function as a "
                f"prior on the mass of {target}.",
                section="priors",
                key=f"{self.prefix}.imf",
                rank=10,
            )
        alphas = np.atleast_1d(getattr(stars, "ffp_alpha", []))
        for i in np.nonzero(mask)[0]:
            alpha = float(alphas[i]) if i < alphas.size else float("nan")
            prose.add(
                f"For {latex_escape(star_names[i])}, we instead adopted "
                r"the free-floating-planet mass function of "
                r"\citet{Sumi:2023}, "
                r"$dN/d\log M \propto M^{-\alpha}$ with "
                rf"$\alpha = {alpha:g}$.",
                section="priors",
                key=f"{self.prefix}.ffp.{i}",
                rank=11,
            )

    def _declare_kinematic_prior(self, stars):
        """Describe the kinematic_prior potential.

        One joint density over distance and the velocity coordinates, so the
        same term is declared against each of them; distance additionally
        carries the ``2 log d`` volume element, which is why
        ``Star.build_likelihood`` stands down where this component exists.
        """
        distance = getattr(stars, "distance", None)
        if distance is not None:
            distance.add_prior_contribution(
                latex=(
                    r"Galactic density $\times$ kinematics "
                    r"(incl. $d^{2}$ volume element)"
                ),
                text="Galactic density x kinematics (incl. d^2 volume)",
                supersedes_bounds=True,
            )
        for name in ("pm_ra", "pm_dec", "rv"):
            param = getattr(stars, name, None)
            if param is not None:
                param.add_prior_contribution(
                    latex="Galactic kinematic model",
                    supersedes_bounds=True,
                )

    def _warn_if_anchor_coords_sampled(self, stars, ra_rad, dec_rad):
        """Warn once if the anchor star's ra/dec are SAMPLED, since the
        line-of-sight basis below is built from their start values and frozen.

        Same policy as ``Lens._frozen_op_coords_deg``: keep the freeze, say so
        -- but only when it could conceivably matter, i.e. when the sampler
        actually moves the coordinates (galacticmodel + gaia/abs astrometry).

        The freeze is safe here too, and separately measured rather than
        assumed by analogy with the microlensing Op: shifting the sight line
        by 1 arcsec moves the anchor's galactocentric velocity by 1.3e-3 km/s
        (4e-5 of a 30 km/s thin-disk dispersion) and its galactic position by
        5e-5 kpc, against kpc-scale density gradients.  It stays negligible
        out to ~1 arcmin (0.08 km/s, 3 pc) and only reaches 0.16 sigma at a
        full degree -- five orders of magnitude beyond the mas-scale posterior
        width astrometry gives ra/dec.
        """
        moving = [
            name
            for name, param in (("ra", stars.ra), ("dec", stars.dec))
            if param.element_is_sampled(self.anchor_idx)
        ]
        if not moving:
            return
        logger.warning(
            f"[{self.prefix}] star.{'/'.join(moving)} of the anchor star "
            f"{self.anchor_idx} is sampled, but the galactic model's line of "
            f"sight is FROZEN at the start value (ra="
            f"{np.degrees(ra_rad):.6f} deg, dec={np.degrees(dec_rad):.6f} "
            f"deg) for the whole fit. This is safe -- 1 arcsec of coordinate "
            f"error moves the galactocentric velocity by ~1e-3 km/s and the "
            f"galactic position by ~0.05 pc -- but the sampled ra/dec do NOT "
            f"feed the density or kinematic prior."
        )

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
        ra_rad = stars.ra.element_start(self.anchor_idx)
        dec_rad = stars.dec.element_start(self.anchor_idx)
        self._warn_if_anchor_coords_sampled(stars, ra_rad, dec_rad)

        # Shared with the start-value hints (physics.line_of_sight_basis):
        # the affine (pm_ra_cosdec, pm_dec, rv) -> galactocentric velocity map,
        # as one offset plus three unit-response columns.
        (
            m_rot_np,
            v0_arr,
            cosl_cosb_np,
            sinl_cosb_np,
            sinb_np,
        ) = line_of_sight_basis(ra_rad, dec_rad)

        # Convert to tensors for graph injection
        M_rot = pt.as_tensor_variable(m_rot_np)  # (3, 3)
        v0 = pt.as_tensor_variable(v0_arr)  # (3,)
        cosl_cosb = pt.as_tensor_variable(cosl_cosb_np)
        sinl_cosb = pt.as_tensor_variable(sinl_cosb_np)
        sinb = pt.as_tensor_variable(sinb_np)

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
            # physics.galactic_xyz is pure arithmetic, so the same function
            # serves numpy scalars (the hints) and tensors (here).
            return _galactic_xyz_np(dist, cosl_cosb, sinl_cosb, sinb)

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

        # Per-star opt-out of the stellar IMF.  A microlensing lens that is a
        # free-floating planet MUST be declared as a star (Lens._validate_
        # bodies rejects a non-star primary), and everything else that makes
        # it a star is right -- the galactic density and kinematic priors
        # below are mass-independent.  Only the mass prior is wrong: under
        # Chabrier a 3 Mjup body sits ~3.3 sigma below the peak, ~5.4 nats of
        # penalty for being what the user said it is.  The choice has to be
        # PER STAR (a stellar source with an FFP lens is the realistic
        # system), which is why it lives on the star block and not here; see
        # Star._parse_mass_functions.
        ffp_mask = getattr(stars, "ffp_mask", None)
        ffp_mask = (
            np.zeros(0, dtype=bool)
            if ffp_mask is None
            else np.asarray(ffp_mask, dtype=bool)
        )
        if ffp_mask.any():
            # Both branches are finite everywhere on the support (one linear
            # in x, one quadratic), so a plain weighted sum is safe and, being
            # arithmetic rather than pt.switch, keeps a clean gradient on
            # every backend.
            w = ffp_mask.astype(float)
            imf_logp = (
                w
                * ffp_logmass_logp(
                    stars.logmass.value, stars.ffp_alpha, stars.logmass
                )
                + (1.0 - w) * imf_logp
            )

        pm.Potential(f"{self.prefix}.imf_prior", pt.sum(imf_logp))

        self._declare_mass_prior(stars, ffp_mask)
        self._add_mass_prior_prose(system, stars, ffp_mask)
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

        self._declare_kinematic_prior(stars)
        from ...outputs.prose import get_collector

        get_collector(system).add(
            r"We imposed Galactic density and kinematic priors on the "
            r"distance, proper motion, and radial velocity of each modeled "
            r"star, following a simplified version of the parametric "
            r"Galactic model of \citet{Koshimoto:2021}.",
            section="priors",
            key=f"{self.prefix}.kinematic",
            rank=20,
        )

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass

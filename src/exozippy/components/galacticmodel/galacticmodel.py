import astropy.units as u
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from astropy.coordinates import Galactocentric, SkyCoord

from exozippy.components.component import Component
from exozippy.constants import (
    BULGE_BAR_ANGLE,
    BULGE_DENSITY_RHO0,
    BULGE_DENSITY_X_0,
    BULGE_DENSITY_Y_0,
    BULGE_DENSITY_Z_0,
    BULGE_GAMMA,
    BULGE_ROTATION_ANGULAR_VELOCITY,
    BULGE_VELOCITY_SIGMA_1,
    BULGE_VELOCITY_SIGMA_2,
    BULGE_VELOCITY_SIGMA_3,
    DISK_DENSITY_RHO0,
    DISK_ROTATION_VELOCITY,
    DISK_SCALE_HEIGHT,
    DISK_SCALE_LENGTH,
    DISK_VELOCITY_SIGMA_U,
    DISK_VELOCITY_SIGMA_V,
    DISK_VELOCITY_SIGMA_W,
    K_VEL_CONVERSION,
    KROUPA_IMF_SLOPE,
    SALPETER_IMF_SLOPE,
    SUN_GC_DISTANCE,
)

"""
This class implements a mixture model of the bulge and disk, enforcing the kinematics and density of the galaxy 
It's a simplified version of the model defined in https://ui.adsabs.harvard.edu/abs/2021ApJ...917...78K/abstract. 
and codified here https://github.com/nkoshimoto/genulens 
This class distills genulens's core functionality into a component for EXOZIPPy
"""


class GalacticModel(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Galactic Prior"
        self.imf = self.config[0].get("IMF", "Kroupa")
        self.anchor_idx = self.config[0].get("anchor_idx", 0)

    @property
    def prefix(self):
        return "galacticmodel"

    def register_parameters(self, system):
        """No parameters to sample! Just an empty manifest."""
        self.manifest = {}

    def build_likelihood(self, model, system):
        stars = system.star

        M_rot_list = []
        v0_list = []
        sinl_cosb_list = []
        cosl_cosb_list = []
        sinb_list = []

        # 1. Pre-compute transformation matrices using Astropy based on initial RA/Dec
        for i in range(self.n_elements):
            # Grab the internal initvals (which are in radians)
            ra_rad = float(np.atleast_1d(stars.ra.initval)[self.anchor_idx])
            dec_rad = float(np.atleast_1d(stars.dec.initval)[self.anchor_idx])

            sc = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad)
            d = 1.0  # kpc, arbitrary distance for velocity basis projection
            pm_1 = 1.0 / (K_VEL_CONVERSION * d)  # mas/yr

            sc0 = SkyCoord(
                ra=sc.ra,
                dec=sc.dec,
                distance=d * u.kpc,
                pm_ra_cosdec=0 * u.mas / u.yr,
                pm_dec=0 * u.mas / u.yr,
                radial_velocity=0 * u.km / u.s,
            )
            sc1 = SkyCoord(
                ra=sc.ra,
                dec=sc.dec,
                distance=d * u.kpc,
                pm_ra_cosdec=pm_1 * u.mas / u.yr,
                pm_dec=0 * u.mas / u.yr,
                radial_velocity=0 * u.km / u.s,
            )
            sc2 = SkyCoord(
                ra=sc.ra,
                dec=sc.dec,
                distance=d * u.kpc,
                pm_ra_cosdec=0 * u.mas / u.yr,
                pm_dec=pm_1 * u.mas / u.yr,
                radial_velocity=0 * u.km / u.s,
            )
            sc3 = SkyCoord(
                ra=sc.ra,
                dec=sc.dec,
                distance=d * u.kpc,
                pm_ra_cosdec=0 * u.mas / u.yr,
                pm_dec=0 * u.mas / u.yr,
                radial_velocity=1 * u.km / u.s,
            )

            gal0 = sc0.transform_to(Galactocentric())
            gal1 = sc1.transform_to(Galactocentric())
            gal2 = sc2.transform_to(Galactocentric())
            gal3 = sc3.transform_to(Galactocentric())

            v0_arr = np.array([gal0.v_x.value, gal0.v_y.value, gal0.v_z.value])
            v1 = (
                np.array([gal1.v_x.value, gal1.v_y.value, gal1.v_z.value])
                - v0_arr
            )
            v2 = (
                np.array([gal2.v_x.value, gal2.v_y.value, gal2.v_z.value])
                - v0_arr
            )
            v3 = (
                np.array([gal3.v_x.value, gal3.v_y.value, gal3.v_z.value])
                - v0_arr
            )

            M_rot_list.append(np.column_stack([v1, v2, v3]))
            v0_list.append(v0_arr)

            l_rad = sc.galactic.l.rad
            b_rad = sc.galactic.b.rad
            sinl_cosb_list.append(np.sin(l_rad) * np.cos(b_rad))
            cosl_cosb_list.append(np.cos(l_rad) * np.cos(b_rad))
            sinb_list.append(np.sin(b_rad))

        # Convert to tensors for graph injection
        M_rot = pt.as_tensor_variable(np.array(M_rot_list))
        v0 = pt.as_tensor_variable(np.array(v0_list))
        cosl_cosb = pt.as_tensor_variable(np.array(cosl_cosb_list))
        sinl_cosb = pt.as_tensor_variable(np.array(sinl_cosb_list))
        sinb = pt.as_tensor_variable(np.array(sinb_list))

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
            x = SUN_GC_DISTANCE - dist * cosl_cosb
            y = dist * sinl_cosb
            z = dist * sinb
            return x, y, z

        def get_polar_velocity(x, y, r, v_x, v_y):
            cos_phi = x / r  # unitless
            sin_phi = y / r  # unitless
            v_r = v_y * sin_phi + v_x * cos_phi
            v_phi = v_y * cos_phi - v_x * sin_phi
            return v_r, v_phi

        # match the IMF
        if self.imf == "Kroupa":
            imf_slope = KROUPA_IMF_SLOPE
        else:
            imf_slope = SALPETER_IMF_SLOPE

        # if we sample in log10 mass, there is no curvature
        # log_m_weight = pt.sum((imf_slope + 1.0) * np.log(10.0) * stars.logmass.value)

        # if we sample in mass, it's hard to explore the full dynamic range
        # log_m_weight = pt.sum(imf_slope * pt.log(stars.mass.value))
        # pm.Potential(f"{self.prefix}.imf_prior", log_m_weight)

        ### Chabrier 2003 System IMF parameters
        log_Mc = np.log10(0.22)
        sigma_imf = 0.57

        # This provides beautiful, constant curvature (-1 / sigma^2) for NUTS
        chabrier_logp = -0.5 * pt.sqr(
            (stars.logmass.value - log_Mc) / sigma_imf
        )

        # For high mass ( > 1 M_sun), you smoothly match it to a Salpeter tail
        # but the low-mass end is usually where the unconstrained NUTS particles fall into the abyss.
        pm.Potential(f"{self.prefix}.imf_prior", pt.sum(chabrier_logp))
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

        # 1. Compute Disk Likelihood (Spatial + Kinematic)
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
        log_dens_disk = (-1.0 / DISK_SCALE_LENGTH) * r + (
            -1.0 / DISK_SCALE_HEIGHT
        ) * z_smooth
        log_vel_disk = (
            (-0.5 / DISK_VELOCITY_SIGMA_U**2) * v_r**2
            + (-0.5 / DISK_VELOCITY_SIGMA_V**2)
            * (v_phi - DISK_ROTATION_VELOCITY) ** 2
            + (-0.5 / DISK_VELOCITY_SIGMA_W**2) * v_z**2
        )
        # Each mixture branch must carry its own normalization: constants
        # cancel within a single Potential but NOT across the logsumexp.
        # Velocity Gaussians contribute -log(sigma1*sigma2*sigma3) (the
        # shared (2*pi)^(3/2) is identical in both branches and dropped);
        # without it the bulge's ~(120*100*80)/(30*30*30) wider ellipsoid
        # was over-weighted by ~3.6 nats.  The density zero points
        # log(rho0) set the physical disk:bulge count ratio (both in
        # Msun/pc^3, see constants.py).
        log_norm_disk = np.log(DISK_DENSITY_RHO0) - np.log(
            DISK_VELOCITY_SIGMA_U
            * DISK_VELOCITY_SIGMA_V
            * DISK_VELOCITY_SIGMA_W
        )
        L_disk = log_dens_disk + log_vel_disk + log_norm_disk

        # 2. Compute Bulge Likelihood (Spatial + Kinematic)
        r_bulge_coord = pt.sqrt(
            (x_bar / BULGE_DENSITY_X_0) ** 2
            + (y_bar / BULGE_DENSITY_Y_0) ** 2
            + (z / BULGE_DENSITY_Z_0) ** 2
        )
        log_dens_bulge = -0.5 * r_bulge_coord
        # Bulge cylindrical rotation is azimuthal too (same fix as above).
        bulge_rot = BULGE_ROTATION_ANGULAR_VELOCITY * r
        log_vel_bulge = (
            (-0.5 / BULGE_VELOCITY_SIGMA_1**2) * v_r**2
            + (-0.5 / BULGE_VELOCITY_SIGMA_2**2) * (v_phi - bulge_rot) ** 2
            + (-0.5 / BULGE_VELOCITY_SIGMA_3**2) * v_z**2
        )
        log_norm_bulge = np.log(BULGE_DENSITY_RHO0) - np.log(
            BULGE_VELOCITY_SIGMA_1
            * BULGE_VELOCITY_SIGMA_2
            * BULGE_VELOCITY_SIGMA_3
        )
        L_bulge = log_dens_bulge + log_vel_bulge + log_norm_bulge

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

        # 3. Combine them using LogSumExp
        # This effectively does: log(exp(L_disk) + exp(L_bulge))
        kinematic_penalty = pt.sum(
            pm.math.logsumexp(pt.stack([L_disk, L_bulge]), axis=0)
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

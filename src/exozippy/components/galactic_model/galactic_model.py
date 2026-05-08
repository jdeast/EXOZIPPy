import numpy as np
import pytensor.tensor as pt
import pymc as pm
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u

from exozippy.components.component import Component
from exozippy.constants import (SUN_GC_DISTANCE, BULGE_BAR_ANGLE, BULGE_DENSITY_X_0,
                                BULGE_DENSITY_Y_0, BULGE_DENSITY_Z_0, BULGE_GAMMA,
                                DISK_SCALE_LENGTH, DISK_SCALE_HEIGHT, DISK_ROTATION_VELOCITY,
                                KROUPA_IMF_SLOPE, SALPETER_IMF_SLOPE, DISK_VELOCITY_SIGMA_U, DISK_VELOCITY_SIGMA_V,
                                DISK_VELOCITY_SIGMA_W, BULGE_VELOCITY_SIGMA_1,
                                BULGE_VELOCITY_SIGMA_2, BULGE_VELOCITY_SIGMA_3,
                                BULGE_ROTATION_ANGULAR_VELOCITY, K_VEL_CONVERSION)


class GalacticModel(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Galactic Prior"

        self.is_microlensing = []
        for c in self.config:
            self.is_microlensing.append("lens_ndx" in c and "source_ndx" in c)

        self.imf = self.config[0].get("IMF", "Kroupa")

    @property
    def prefix(self):
        return "galactic_model"

    def build_parameters(self, model):
        pass

    def load_data(self):
        pass

    def build_map(self, system):
        star_map, lens_map, source_map = [], [], []

        for c in self.config:
            if "star_ndx" in c:
                star_map.append(c["star_ndx"])
            elif "lens_ndx" in c and "source_ndx" in c:
                lens_map.append(c["lens_ndx"])
                source_map.append(c["source_ndx"])
            else:
                raise ValueError("Galactic model requires either 'star_ndx' or ('lens_ndx' and 'source_ndx').")

        # If the list has items, convert to PyTensor. Otherwise, explicitly set to None.
        self.star_map = pt.as_tensor_variable(np.array(star_map)).astype("int32") if star_map else None
        self.lens_map = pt.as_tensor_variable(np.array(lens_map)).astype("int32") if lens_map else None
        self.source_map = pt.as_tensor_variable(np.array(source_map)).astype("int32") if source_map else None

    def build_dependent_parameters(self, model, system):
        pass

    def build_likelihood(self, model, system):
        stars = system.star

        M_rot_list = []
        v0_list = []
        sinl_cosb_list = []
        cosl_cosb_list = []
        sinb_list = []

        # 1. Pre-compute transformation matrices using Astropy based on initial RA/Dec
        for i in range(self.n_elements):
            # For microlensing, the convention is to anchor the sky coordinates to the source star
            if self.is_microlensing[i]:
                anchor_idx = int(self.source_map.eval()[i])
            else:
                anchor_idx = int(self.star_map.eval()[i])

            # Grab the internal initvals (which are in radians)
            ra_rad = float(np.atleast_1d(stars.ra.initval)[anchor_idx])
            dec_rad = float(np.atleast_1d(stars.dec.initval)[anchor_idx])

            sc = SkyCoord(ra=ra_rad * u.rad, dec=dec_rad * u.rad)
            d = 1.0 # kpc, arbitrary distance for velocity basis projection
            pm_1 = 1.0 / (K_VEL_CONVERSION * d) # mas/yr

            sc0 = SkyCoord(ra=sc.ra, dec=sc.dec, distance=d*u.kpc, pm_ra_cosdec=0 * u.mas / u.yr,
                           pm_dec=0 * u.mas / u.yr, radial_velocity=0 * u.km / u.s)
            sc1 = SkyCoord(ra=sc.ra, dec=sc.dec, distance=d*u.kpc, pm_ra_cosdec=pm_1 * u.mas / u.yr,
                           pm_dec=0 * u.mas / u.yr, radial_velocity=0 * u.km / u.s)
            sc2 = SkyCoord(ra=sc.ra, dec=sc.dec, distance=d*u.kpc, pm_ra_cosdec=0 * u.mas / u.yr,
                           pm_dec=pm_1 * u.mas / u.yr, radial_velocity=0 * u.km / u.s)
            sc3 = SkyCoord(ra=sc.ra, dec=sc.dec, distance=d*u.kpc, pm_ra_cosdec=0 * u.mas / u.yr,
                           pm_dec=0 * u.mas / u.yr, radial_velocity=1 * u.km / u.s)

            gal0 = sc0.transform_to(Galactocentric())
            gal1 = sc1.transform_to(Galactocentric())
            gal2 = sc2.transform_to(Galactocentric())
            gal3 = sc3.transform_to(Galactocentric())

            v0_arr = np.array([gal0.v_x.value, gal0.v_y.value, gal0.v_z.value])
            v1 = np.array([gal1.v_x.value, gal1.v_y.value, gal1.v_z.value]) - v0_arr
            v2 = np.array([gal2.v_x.value, gal2.v_y.value, gal2.v_z.value]) - v0_arr
            v3 = np.array([gal3.v_x.value, gal3.v_y.value, gal3.v_z.value]) - v0_arr

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
            v_rad_kms = rv_ms/1e3
            v_icrs = pt.stack([v_alpha_kms, v_delta_kms, v_rad_kms], axis=1)  # (N, 3)
            v_gal_offset = (M_rot @ v_icrs[:, :, None]).squeeze(-1)
            return v0 + v_gal_offset

        def get_galactic_xyz(dist):
            x = SUN_GC_DISTANCE - dist * cosl_cosb
            y = dist * sinl_cosb
            z = dist * sinb
            return x, y, z

        def get_polar_velocity(x, y, r, v_x, v_y):
            cos_phi = x / r # unitless
            sin_phi = y / r # unitless
            v_r = v_y * sin_phi + v_x * cos_phi
            v_phi = v_y * cos_phi - v_x * sin_phi
            return v_r, v_phi

        # match the IMF
        if self.imf == "Kroupa":
            imf_slope = KROUPA_IMF_SLOPE
        else:
            imf_slope = SALPETER_IMF_SLOPE

        # if we sample in log10 mass, there is no curvature
        #log_m_weight = pt.sum((imf_slope + 1.0) * np.log(10.0) * stars.logmass.value)

        # if we sample in mass, it's hard to explore the full dynamic range
        #log_m_weight = pt.sum(imf_slope * pt.log(stars.mass.value))
        #pm.Potential(f"{self.prefix}.imf_prior", log_m_weight)

        ### Chabrier 2003 System IMF parameters
        log_Mc = np.log10(0.22)
        sigma_imf = 0.57

        # This provides beautiful, constant curvature (-1 / sigma^2) for NUTS
        chabrier_logp = -0.5 * pt.sqr((stars.logmass.value - log_Mc) / sigma_imf)

        # For high mass ( > 1 M_sun), you smoothly match it to a Salpeter tail
        # but the low-mass end is usually where the unconstrained NUTS particles fall into the abyss.
        pm.Potential(f"{self.prefix}.imf_prior", pt.sum(chabrier_logp))
        ######

        # even though non-physical values will be rejected
        # we still need to be able to calculate a likelihood,
        distance = pt.maximum(stars.distance.value, 1e-3)/1e3 # kpc
        x,y,z = get_galactic_xyz(distance)
        z_smooth = pt.sqrt(z**2 + 1e-6)
        r = pt.sqrt(x ** 2 + y ** 2 + 1e-6)

        vel = get_galactocentric_velocity(distance, stars.pm_ra.value, stars.pm_dec.value, stars.rv.value) # km/s
        v_x, v_y, v_z = vel[:, 0], vel[:, 1], vel[:, 2]
        v_r, v_phi = get_polar_velocity(x, y, r, v_x, v_y)

        # match the density distribution of the galaxy
        cos_bar = np.cos(BULGE_BAR_ANGLE)
        sin_bar = np.sin(BULGE_BAR_ANGLE)
        x_bar = x * cos_bar + y * sin_bar
        y_bar = -x * sin_bar + y * cos_bar

        # 1. Compute Disk Likelihood (Spatial + Kinematic)
        log_dens_disk = (-1.0 / DISK_SCALE_LENGTH) * r + (-1.0 / DISK_SCALE_HEIGHT) * z_smooth
        log_vel_disk = (-0.5 / DISK_VELOCITY_SIGMA_U ** 2) * (v_r - DISK_ROTATION_VELOCITY) ** 2 \
                       + (-0.5 / DISK_VELOCITY_SIGMA_V ** 2) * v_phi ** 2 \
                       + (-0.5 / DISK_VELOCITY_SIGMA_W ** 2) * v_z ** 2
        L_disk = log_dens_disk + log_vel_disk

        # 2. Compute Bulge Likelihood (Spatial + Kinematic)
        r_bulge_coord = pt.sqrt(
            (x_bar / BULGE_DENSITY_X_0) ** 2 + (y_bar / BULGE_DENSITY_Y_0) ** 2 + (z / BULGE_DENSITY_Z_0) ** 2)
        log_dens_bulge = -0.5 * r_bulge_coord
        bulge_rot = BULGE_ROTATION_ANGULAR_VELOCITY * r
        log_vel_bulge = (-0.5 / BULGE_VELOCITY_SIGMA_1 ** 2) * (v_r - bulge_rot) ** 2 \
                        + (-0.5 / BULGE_VELOCITY_SIGMA_2 ** 2) * v_phi ** 2 \
                        + (-0.5 / BULGE_VELOCITY_SIGMA_3 ** 2) * v_z** 2
        L_bulge = log_dens_bulge + log_vel_bulge

        volume_element = 2.0 * pt.log(distance * 1000.0)

        # 3. Combine them using LogSumExp
        # This effectively does: log(exp(L_disk) + exp(L_bulge))
        kinematic_penalty = pt.sum(pm.math.logsumexp(pt.stack([L_disk, L_bulge]), axis=0)+volume_element)
        pm.Potential(f"{self.prefix}.kinematic_prior", kinematic_penalty)

        # check parameters for debugging
        #pm.Deterministic(f"{self.prefix}.gal_x", x)
        #pm.Deterministic(f"{self.prefix}.gal_y", y)
        #pm.Deterministic(f"{self.prefix}.gal_z", z)
        #pm.Deterministic(f"{self.prefix}.gal_r", r)
        #pm.Deterministic(f"{self.prefix}.v_x", v_x)
        #pm.Deterministic(f"{self.prefix}.v_y", v_y)
        #pm.Deterministic(f"{self.prefix}.v_z", v_z)
        #pm.Deterministic(f"{self.prefix}.v_r", v_r)
        #pm.Deterministic(f"{self.prefix}.v_phi", v_phi)
        #pm.Deterministic(f"{self.prefix}.L_disk", log_dens_disk + log_vel_disk)
        #pm.Deterministic(f"{self.prefix}.L_bulge", log_dens_bulge + log_vel_bulge)
        #pm.Deterministic(f"{self.prefix}.log_imf_weight", log_m_weight)

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
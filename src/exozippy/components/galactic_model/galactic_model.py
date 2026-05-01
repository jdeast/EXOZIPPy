import numpy as np
import pytensor.tensor as pt
import pymc as pm
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.units as u

from exozippy.components.component import Component
from exozippy.constants import (SUN_GC_DISTANCE, BULGE_BAR_ANGLE, BULGE_DENSITY_X_0,
                                BULGE_DENSITY_Y_0, BULGE_DENSITY_Z_0, BULGE_GAMMA,
                                DISK_SCALE_LENGTH, DISK_SCALE_HEIGHT, DISK_ROTATION_VELOCITY,
                                IMF_SLOPE, DISK_VELOCITY_SIGMA_U, DISK_VELOCITY_SIGMA_V,
                                DISK_VELOCITY_SIGMA_W, BULGE_VELOCITY_SIGMA_1,
                                BULGE_VELOCITY_SIGMA_2, BULGE_VELOCITY_SIGMA_3,
                                BULGE_ROTATION_ANGULAR_VELOCITY)


class GalacticModel(Component):
    def __init__(self, config, config_manager):
        super().__init__(config, config_manager)
        self.label = "Galactic Prior"

        self.is_microlensing = []
        for c in self.config:
            self.is_microlensing.append("lens_ndx" in c and "source_ndx" in c)

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
            d = 1000.0  # Arbitrary distance for velocity basis projection
            pm_1 = 1.0 / (4.74047 * d)

            sc0 = SkyCoord(ra=sc.ra, dec=sc.dec, distance=d * u.pc, pm_ra_cosdec=0 * u.mas / u.yr,
                           pm_dec=0 * u.mas / u.yr, radial_velocity=0 * u.km / u.s)
            sc1 = SkyCoord(ra=sc.ra, dec=sc.dec, distance=d * u.pc, pm_ra_cosdec=pm_1 * u.mas / u.yr,
                           pm_dec=0 * u.mas / u.yr, radial_velocity=0 * u.km / u.s)
            sc2 = SkyCoord(ra=sc.ra, dec=sc.dec, distance=d * u.pc, pm_ra_cosdec=0 * u.mas / u.yr,
                           pm_dec=pm_1 * u.mas / u.yr, radial_velocity=0 * u.km / u.s)
            sc3 = SkyCoord(ra=sc.ra, dec=sc.dec, distance=d * u.pc, pm_ra_cosdec=0 * u.mas / u.yr,
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
        def get_galactocentric_velocity(dist, pm_ra, pm_dec, rv, idx_slice):
            v_alpha = 4.74047 * pm_ra * dist
            v_delta = 4.74047 * pm_dec * dist
            v_rad = rv / 1000.0  # internal is m/s, convert to km/s
            v_icrs = pt.stack([v_alpha, v_delta, v_rad], axis=1)  # (N, 3)
            v_gal_offset = (M_rot[idx_slice] @ v_icrs[:, :, None]).squeeze(-1)
            return v0[idx_slice] + v_gal_offset

        def get_galactic_xyz(dist, idx_slice):
            x = SUN_GC_DISTANCE - dist * cosl_cosb[idx_slice]
            y = dist * sinl_cosb[idx_slice]
            z = dist * sinb[idx_slice]
            return x, y, z

        def get_polar_velocity(x, y, r, v_x, v_y):
            cos_phi = x / r
            sin_phi = y / r
            v_r = v_y * sin_phi + v_x * cos_phi
            v_phi = v_y * cos_phi - v_x * sin_phi
            return v_r, v_phi

        # 3. Apply Penalties
        # --- A: Microlensing Kinematics (Lens & Source interacting) ---
        if self.lens_map is not None:
            idx_slice = np.where(self.is_microlensing)[0]

            d_l_raw = stars.distance.value[self.lens_map]
            d_s_raw = stars.distance.value[self.source_map]

            # even though non-physical values will be rejected (in lens.py),
            # we still need to be able to calculate a likelihood,
            # so we round to physical values
            d_l = pt.maximum(d_l_raw, 1e-6)/1e3 # kpc
            d_s = pt.maximum(d_s_raw, d_l + 1e-6)/1e3 # kpc

            v_s = get_galactocentric_velocity(d_s, stars.pm_ra.value[self.source_map],
                                              stars.pm_dec.value[self.source_map], stars.rv.value[self.source_map],
                                              idx_slice)
            x_s, y_s, z_s = get_galactic_xyz(d_s, idx_slice)
            r_s = pt.sqrt(x_s ** 2 + y_s ** 2)

            cos_bar = np.cos(BULGE_BAR_ANGLE)
            sin_bar = np.sin(BULGE_BAR_ANGLE)
            xx_s = x_s * cos_bar + y_s * sin_bar
            yy_s = -x_s * sin_bar + y_s * cos_bar
            x_1 = xx_s ** 2 / BULGE_DENSITY_X_0 ** 2
            y_1 = yy_s ** 2 / BULGE_DENSITY_Y_0 ** 2
            z_1 = z_s ** 4 / BULGE_DENSITY_Z_0 ** 4
            r_s_2 = pt.sqrt((x_1 + y_1) ** 2 + z_1)
            self.log_source_density = -0.5 * r_s_2

            v_r_s, v_phi_s = get_polar_velocity(x_s, y_s, r_s, v_s[:, 0], v_s[:, 1])
            bulge_rot = BULGE_ROTATION_ANGULAR_VELOCITY * r_s
            w_vs = (-0.5 / BULGE_VELOCITY_SIGMA_1 ** 2) * (v_r_s - bulge_rot) ** 2 \
                   + (-0.5 / BULGE_VELOCITY_SIGMA_2 ** 2) * v_phi_s ** 2 \
                   + (-0.5 / BULGE_VELOCITY_SIGMA_3 ** 2) * v_s[:, 2] ** 2
            self.log_weight_v_s = w_vs
            self.log_d_s_weight = pt.log(d_s ** BULGE_GAMMA)

            m_l = stars.mass.value[self.lens_map]
            v_l = get_galactocentric_velocity(d_l, stars.pm_ra.value[self.lens_map], stars.pm_dec.value[self.lens_map],
                                              stars.rv.value[self.lens_map], idx_slice)
            x_l, y_l, z_l = get_galactic_xyz(d_l, idx_slice)
            r_l = pt.sqrt(x_l ** 2 + y_l ** 2)

            scaled_disk = (-1.0 / DISK_SCALE_LENGTH) * r_l + (-1.0 / DISK_SCALE_HEIGHT) * pt.abs(z_l)
            self.log_lens_density = scaled_disk

            v_r_l, v_phi_l = get_polar_velocity(x_l, y_l, r_l, v_l[:, 0], v_l[:, 1])
            self.log_weight_v_l = (-0.5 / DISK_VELOCITY_SIGMA_U ** 2) * (v_r_l - DISK_ROTATION_VELOCITY) ** 2 \
                   + (-0.5 / DISK_VELOCITY_SIGMA_V ** 2) * v_phi_l ** 2 \
                   + (-0.5 / DISK_VELOCITY_SIGMA_W ** 2) * v_l[:, 2] ** 2
            self.log_m_l_weight = pt.log(m_l ** IMF_SLOPE)

            self.mu_rel = system.lens.mu_rel_mag.value[self.lens_map]
            self.theta_E = system.lens.theta_E.value[self.lens_map]

            self.log_total_weight = (
                    pt.log(self.mu_rel) +
                    pt.log(self.theta_E) +
                    self.log_weight_v_s +
                    self.log_source_density +
                    self.log_d_s_weight +
                    self.log_weight_v_l +
                    self.log_lens_density +
                    self.log_m_l_weight
            )
            pm.Potential(f"{self.prefix}.prior", self.log_total_weight)

        # --- B: Host Star Kinematics (Disk prior only) ---
        if self.star_map is not None:
            idx_slice = np.where(~np.array(self.is_microlensing))[0]

            d_l = stars.distance.value[self.star_map]
            m_l = stars.mass.value[self.star_map]
            v_l = get_galactocentric_velocity(d_l, stars.pm_ra.value[self.star_map], stars.pm_dec.value[self.star_map],
                                              stars.rv.value[self.star_map], idx_slice)
            x_l, y_l, z_l = get_galactic_xyz(d_l, idx_slice)
            r_l = pt.sqrt(x_l ** 2 + y_l ** 2)

            scaled_disk = (-1.0 / DISK_SCALE_LENGTH) * r_l + (-1.0 / DISK_SCALE_HEIGHT) * pt.abs(z_l)
            lens_density = pt.exp(scaled_disk)

            v_r_l, v_phi_l = get_polar_velocity(x_l, y_l, r_l, v_l[:, 0], v_l[:, 1])
            w_vl = (-0.5 / DISK_VELOCITY_SIGMA_U ** 2) * (v_r_l - DISK_ROTATION_VELOCITY) ** 2 \
                   + (-0.5 / DISK_VELOCITY_SIGMA_V ** 2) * v_phi_l ** 2 \
                   + (-0.5 / DISK_VELOCITY_SIGMA_W ** 2) * v_l[:, 2] ** 2
            weight_v_l = pt.exp(w_vl)
            m_l_weight = m_l ** IMF_SLOPE

            total_weight = weight_v_l * lens_density * m_l_weight
            safe_weight = pt.maximum(total_weight, 1e-300)
            pm.Potential(f"{self.prefix}.host_prior", pt.sum(pt.log(safe_weight)))

    def compile_plotters(self, model, system):
        pass

    def plot(self, system, points, filename_prefix="debug"):
        pass
import numpy as np

from ...constants import (
    KAPPA, SUN_GC_DISTANCE, BULGE_BAR_ANGLE, BULGE_DENSITY_X_0, BULGE_DENSITY_Y_0, BULGE_DENSITY_Z_0, BULGE_GAMMA
    DISK_SCALE_LENGTH, DISK_SCALE_HEIGHT, DISK_ROTATION_VELOCITY, IMF_SLOPE,
    DISK_VELOCITY_SIGMA_U, DISK_VELOCITY_SIGMA_V, DISK_VELOCITY_SIGMA_W,
    BULGE_VELOCITY_SIGMA_1, BULGE_VELOCITY_SIGMA_2, BULGE_VELOCITY_SIGMA_3, BULGE_ROTATION_ANGULAR_VELOCITY,
    SUN_VELOCITY_X, SUN_VELOCITY_Y, SUN_VELOCITY_Z)

BULGE_ROTATION_ANGULAR_VELOCITY

# NOTE: For simplicity we assume Sun at z=0.

class MicrolensingEvent(object):
    """
    A microlensing event.
    Parameters :
        l: *float*
            Galactic longitude in degrees.
        b: *float*
            Galactic latitude in degrees.
    """
    def __init__(self, l=0., b=0.):
        self._l_deg = l
        self._b_deg = b
        self._cosl_cosb = np.cos(np.radians(l)) * np.cos(np.radians(b))
        self._sinl_cosb = np.sin(np.radians(l)) * np.cos(np.radians(b))
        self._sinb = np.sin(np.radians(b))
        self._b_sign = np.sign(b)  # Defined in order to caclulate |z| without if statement.

        self._cos_bar_angle = np.cos(BULGE_BAR_ANGLE)
        self._sin_bar_angle = np.sin(BULGE_BAR_ANGLE)
        self._bulge_x_0_2 = BULGE_DENSITY_X_0**2
        self._bulge_y_0_2 = BULGE_DENSITY_Y_0**2
        self._bulge_z_0_4 = BULGE_DENSITY_Z_0**4
        self._disk_const_1 = -1. / DISK_SCALE_LENGTH
        self._disk_const_2 = -self._b_sign / DISK_SCALE_HEIGHT
        self._disk_const_velocity_1 = -0.5 / DISK_VELOCITY_SIGMA_U**2
        self._disk_const_velocity_2 = -0.5 / DISK_VELOCITY_SIGMA_V**2
        self._disk_const_velocity_3 = -0.5 / DISK_VELOCITY_SIGMA_W**2
        self._bulge_const_velocity_1 = -0.5 / BULGE_VELOCITY_SIGMA_1**2
        self._bulge_const_velocity_2 = -0.5 / BULGE_VELOCITY_SIGMA_2**2
        self._bulge_const_velocity_3 = -0.5 / BULGE_VELOCITY_SIGMA_3**2

    def _get_galactic_xyz(self, distance):
        """Calculate x,y,z coordinates for given distance."""
        x = SUN_GC_DISTANCE - distance * self._cosl_cosb
        y = SUN_GC_DISTANCE - distance * self._sinl_cosb
        z = distance * self._sinb
        return x, y, z

    def _get_GC_distance(self, x, y):
        """calculate distance from Galactic center"""
        return np.sqrt(x**2 + y**2)

    def _get_bulge_coords(self, x, y, z):
        """rotate coordinates from standard to aligned with the bulge"""
        xx = x * self._cos_bar_angle + y * self._sin_bar_angle
        yy = -x * self._sin_bar_angle + y * self._cos_bar_angle
        return (xx, yy, z)

    def _get_relative_bulge_density(self, x, y, z):
        """
        Transform given coordinates and calculate the relative bulge density.
        Currently based on Dwek+95 model.
        """
        (xx, yy, zz) = self._get_bulge_coords(x, y, z)
        x_1 = xx**2 / self._bulge_x_0_2
        y_1 = yy**2 / self._bulge_y_0_2
        z_1 = zz**4 / self._bulge_z_0_4
        r_s_2 = np.sqrt((x_1 + y_1)**2 + z_1)  # r_s**2 in Dwek+95 notation.
        return np.exp(-0.5 * r_s_2)

    def _get_relative_lens_density(self, r_l, z_l):
        """Relative lens density"""
        return self._get_relative_disk_density(r_l, z_l)

    def _get_relative_disk_density(self, radius, z):
        """Relative disk density"""
        scaled = self._disk_const_1 * radius + self._disk_const_2 * z
        return np.exp(scaled)

    def _get_theta_E(self, d_s, d_l, m_l):
        """caclulate theta_E in mas"""
        return np.sqrt(KAPPA * m_l * (1./d_l - 1./d_s))

    def _get_weight_lens_velocity(self, x_l, y_l, r_l, v_l_x, v_l_y, v_l_z):
        """
        Calculate weight of lens velocity.
        We assume lens is in the disk.
        """
        (v_r_l, v_phi_l) = self._get_polar_velocity(x_l, y_l, r_l, v_l_x, v_l_y)
        out = self._disk_const_velocity_1 * (v_r_l - DISK_ROTATION_VELOCITY)**2
        out += self._disk_const_velocity_2 * v_phi_l**2
        out += self._disk_const_velocity_3 * v_l_z**2
        return np.exp(out)

    def _get_polar_velocity(self, x, y, r, v_x, v_y):
        """transform 2D cartersian vector to polar frame"""
        cos_phi = x / r
        sin_phi = y / r
        v_r = v_y * sin_phi + v_x * cos_phi
        v_phi = v_y * cos_phi - v_x * sin_phi
        return (v_r, v_phi)

    def _get_weight_source_velocity(self, x_s, y_s, r_s, v_s_x, v_s_y, v_s_z):
        """
        Calculate weight of source velocity.
        We assume lens is in the bulge with rotation + sigma kinematics.
        """
        (v_r_s, v_phi_s) = self._get_polar_velocity(x_s, y_s, r_s, v_s_x, v_s_y)
        bulge_rotation = BULGE_ROTATION_ANGULAR_VELOCITY * r_s
        out = self._bulge_const_velocity_1 * (v_r_s - bulge_rotation)**2
        out += self._bulge_const_velocity_2 * v_phi_s**2
        out += self._bulge_const_velocity_3 * v_s_z**2
        return np.exp(out)

    def _get_source_distance_weight(self, d_s):
        """
        The further the source, the fainter it is; various scalings are used in literature.
        See Koshimoto and Bennett 2020.
        """
        return d_s**BULGE_GAMMA

    def _get_lens_mass_weight(self, m_l):
        """
        Calculate how mass function affects the number of potential lenses.
        """
        return m_l**IMF_SLOPE

    def _get_mu_rel(self, d_s, d_l, v_l_x, v_l_y, v_l_z, v_s_x, v_s_y, v_s_z):
        """
        Calculate relative lens-source proper motion in (l,b).
        """
        (mu_l_l, mu_l_b) = self._get_proper_motion(d_l, v_l_x, v_l_y, v_l_z)
        (mu_s_l, mu_s_b) = self._get_proper_motion(d_l, v_s_x, v_s_y, v_s_z)
        mu_rel_l = mu_l_l - mu_s_l
        mu_rel_b = mu_l_b - mu_s_b
        return (mu_rel_l, mu_rel_b)

    def get_relative_weight(self, d_s, d_l, m_l, v_l_x, v_l_y, v_l_z, v_s_x, v_s_y, v_s_z):
        """
        Calculate relative weight of event with given parameters.
        Parameters :
            d_s: *float*
                Source distance in kpc
            d_l: *float*
                Lens distance in kpc
            m_l: *float*
                Lens mass in M_Sun
            v_l_x, v_l_y, v_l_z: *float*
                Lens velocity in km/s
            v_s_x, v_s_y, v_s_z: *float*
                Source velocity in km/s
        """
        (x_s, y_s, z_s) = self._get_galactic_xyz(d_s)
        r_s = self._get_GC_distance(x_s, y_s)
        (x_l, y_l, z_l) = self._get_galactic_xyz(d_l)
        r_l = self._get_GC_distance(x_l, y_l)
        source_density = self._get_relative_bulge_density(x_s, y_s, z_s)
        lens_density = self._get_relative_lens_density(r_l, z_l)
        theta_E = self._get_theta_E(d_s, d_l, m_l)
        weight_v_l = self._get_weight_lens_velocity(x_l, y_l, r_l, v_l_x, v_l_y, v_l_z)
        weight_v_s = self._get_weight_source_velocity(x_s, y_s, r_s, v_s_x, v_s_y, v_s_z)
        (mu_rel_l, mu_rel_b) = self._get_mu_rel(d_s, d_l, v_l_x, v_l_y, v_l_z, v_s_x, v_s_y, v_s_z)
        mu_rel = np.sqrt(mu_rel_l**2 + mu_rel_b**2)
        m_l_weight = self._get_lens_mass_weight(m_l)
        d_s_weight = self._get_source_distance_weight(d_s)

        out = mu_rel * rel_theta_E
        out *= weight_v_s * source_density * d_s_weight
        out *= weight_v_l * lens_density * m_l_weight
        return out


if __name__ == '__main__':
    event = MicrolensingEvent(l=1.02, b=-3.92)  # i.e., Baade's window
    kwargs = dict(d_s=8, d_l=5, m_l=0.3, v_l_x=100., v_l_y=100., v_l_z=100., v_s_x=100., v_s_y=100., v_s_z=100.)
    weight = event.get_relative_weight(**kwargs)
    print(weight)

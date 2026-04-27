import numpy as np

from ...constants import (SUN_GC_DISTANCE, BULGE_BAR_ANGLE, BULGE_DENSITY_X_0, BULGE_DENSITY_Y_0, BULGE_DENSITY_Z_0, 
                         DISK_SCALE_LENGTH, DISK_SCALE_HEIGHT, DISK_ROTATION_VELOCITY,
                         SUN_VELOCITY_X, SUN_VELOCITY_Y, SUN_VELOCITY_Z)

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

    def _get_galactic_xyz(self, distance):
        """Calculate x,y,z coordinates for given distance."""
        x = SUN_GC_DISTANCE - distance * self._cosl_cosb
        y = SUN_GC_DISTANCE - distance * self._sinl_cosb
        z = distance * self._sinb
        return x, y, z

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

    def _get_relative_lens_density(self, x_l, y_l, z_l):
        """Relative lens density"""
        radius_l = np.sqrt(x_l**2 + y_l**2)
        return self._get_relative_disk_density(radius_l, z_l)

    def _get_relative_disk_density(self, radius, z):
        """Relative disk density"""
        scaled = self._disk_const_1 * radius + self._disk_const_2 * z
        return np.exp(scaled)

    def _get_unormalized_theta_E(self, d_s, d_l, m_l):
        """caclulate theta_E ignoring kappa constant"""
        return np.sqrt(m_l * (1./d_l - 1./d_s))

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
        (x_l, y_l, z_l) = self._get_galactic_xyz(d_l)
        source_density = self._get_relative_bulge_density(x_s, y_s, z_s)
        lens_density = self._get_relative_lens_density(x_l, y_l, z_l)
        rel_theta_E = self._get_unormalized_theta_E(d_s, d_l, m_l)
        weight_v_l = self._get_weight_lens_velocity(v_l_x, v_l_y, v_l_z)
        # weight_v_s = self._get_weight_source_velocity(v_s_x, v_s_y, v_s_z)
        # mu_rel = self._get_mu_rel(d_s, d_l, v_l_x, v_l_y, v_l_z, v_s_x, v_s_y, v_s_z)
        # m_l_weight = self._get_lens_mass_weight(m_l)
        # d_s_weight = self._get_source_distance_weight(d_s)

        out = mu_rel * rel_theta_E
        out *= weight_v_s * source_density * d_s_weight
        out *= weight_v_l * lens_density * m_l_weight
        return out


if __name__ == '__main__':
    event = MicrolensingEvent(l=1.02, b=-3.92)  # i.e., Baade's window
    kwargs = dict(d_s=8, d_l=5, m_l=0.3, v_l_x=100., v_l_y=100., v_l_z=100., v_s_x=100., v_s_y=100., v_s_z=100.)
    weight = event.get_relative_weight(**kwargs)
    print(weight)

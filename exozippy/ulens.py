"""
Functions and classes for converting between microlensing and physical
parameters.
"""

class Phys2UlensConverter(object):

    def __init__(self, source=None, lens=None, coords=None, t_ref=None):
        self.source = source
        self.lens = lens
        self._set_coords(coords)
        self.t_ref = t_ref

    def get_ulens_params(self):
        self.calc_thetaE()
        self.calc_rho()

        return {'t_E': self.t_E, 'rho': self.rho}



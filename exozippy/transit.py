from exozippy.parameter import Parameter


class Transit():
    def __init__(self, name, planet, star, band, ttv=False, tdeltav=False, tiv=False):

        self.f0 = Parameter("f0" + name, lower=0.0, upper=10.0, initval=1.0,
                             unit=1, latex='F_0', latex_unit='',
                             user_params=user_params)

        self.variance = Parameter("variance" + name, lower=-10, upper=10.0, initval=0.0,
                             unit=1, latex='\sigma^2', latex_unit='',
                             user_params=user_params)

        if ttv:
            self.ttv = Parameter("ttv" + name, lower=-10, upper=10.0, initval=0.0,
                                      unit=u.d, latex='TTV', latex_unit='Days',
                                      user_params=user_params)

        if tdeltav:
            self.tdeltav = Parameter("tdeltav" + name, lower=-10, upper=10.0, initval=0.0,
                                      unit=1, latex='T\deltaV', latex_unit='',
                                      user_params=user_params)

        if tiv:
            self.tiv = Parameter("tiv" + name, lower=-10, upper=10.0, initval=0.0,
                                      unit=u.deg, latex='TiV', latex_unit='$^\circ$',
                                      user_params=user_params)

        readtran(tranfile, ndx=i, ttvs=ttvs, tdeltavs=tdeltavs, tivs=tivs)
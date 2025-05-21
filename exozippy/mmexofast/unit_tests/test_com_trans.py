import unittest

import matplotlib.pyplot as plt
import numpy.testing
import MulensModel

from exozippy.mmexofast import com_trans


class TestCOMass2COMag(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_close_case(self):
        params = {'t_0': 0., 'u_0': 0.5, 't_E': 1., 's': 0.7, 'q': 0.1, 'alpha': 30.}
        new_params = com_trans.co_mass_to_co_magnif(params)
        assert new_params['t_0'] == params['t_0']
        assert new_params['u_0'] == params['u_0']

    def do_wide_test(self, alpha):
        params = {'t_0': 0., 'u_0': 0.5, 't_E': 1., 's': 20., 'q': 0.01, 'alpha': alpha}
        #delta = params['s'] * params['q'] / (1 + params['q'])
        new_params = com_trans.co_mass_to_co_magnif(params)
        new_params['t_E'] = params['t_E']
        planet = MulensModel.Model(parameters=params)
        pspl = MulensModel.Model(new_params)

        #print(planet)
        #print(pspl)
        #plt.figure(figsize=(8, 4))
        #plt.subplot(1, 2, 1)
        #planet.plot_trajectory(caustics=True)
        #
        #plt.subplot(1, 2, 2)
        #planet.plot_magnification(label='planet')
        #pspl.plot_magnification(label='PSPL')
        #plt.legend()
        #plt.minorticks_on()
        #plt.show()

        numpy.testing.assert_allclose(
            pspl.get_magnification(params['t_0']),
            planet.get_magnification(params['t_0']), rtol=0.01)

    def test_wide_case_1(self):
        self.do_wide_test(0.)

    def test_wide_case_2(self):
        self.do_wide_test(90.)

    def test_wide_case_3(self):
        self.do_wide_test(180.)

    def test_wide_case_4(self):
        self.do_wide_test(270.)

    def test_wide_case_5(self):
        self.do_wide_test(-90.)

    def test_wide_case_6(self):
        self.do_wide_test(-180.)

    def test_wide_case_7(self):
        self.do_wide_test(-270.)


class TestCOMag2COMass(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_close_case(self):
        print('tests not implemented, probably has some weird offset due to mmv2 --> v3')
        assert 1 == 2


class TestPrimary2COMass(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_close_case(self):
        print('tests not implemented, probably has some weird offset due to mmv2 --> v3')
        assert 1 == 2


class TestPrimary2COMag(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_close_case(self):
        print('tests not implemented, probably has some weird offset due to mmv2 --> v3')
        assert 1 == 2

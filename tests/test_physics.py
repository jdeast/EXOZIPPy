import numpy as np
import pymc as pm
import pytest
import astropy.units as u
import astropy.constants as const
from exozippy.components.star import Star
from exozippy.components.orbit import Orbit
from exozippy.components.planet import Planet


def test_planet_k_value():
    """
    Directly tests the Planet class's internal K parameter calculation.
    Target: 1 Mjup, 1 Msun, 365.25 days, e=0, i=90 should be ~28.43 m/s.
    """
    # 1. Setup minimal config dictionaries
    star_cfg = {"name": "Sun"}
    orbit_cfg = {"name": "Orbit1"}
    planet_cfg = {"name": "Jupiter"}

    # Mock user_params to provide standard "Sun-like" and "Jupiter-like" values
    user_params = {
        "star.Sun.mass": {"initval": 1.0},  # Change from 1.0 to a dict
        "orbit.Orbit1.logP": {"initval": np.log10(365.25)},
        "orbit.Orbit1.tc_base": {"initval": 0.0},
        "orbit.Orbit1.cosi": {"initval": 1e-7},
        "orbit.Orbit1.secosw": {"initval": 0.01},
        "orbit.Orbit1.sesinw_raw": {"initval": 0.01},
        "planet.Jupiter.mass": {"initval": 1.0},
        "planet.Jupiter.radius": {"initval": 1.0}
    }



    # 2. Instantiate actual components
    sun = Star(star_cfg, user_params)
    orbit = Orbit(orbit_cfg, user_params)
    jupiter = Planet(planet_cfg, user_params)

    # 3. Trigger the parameter building inside a PyMC model context
    with pm.Model() as model:
        # Now pm.Uniform calls inside these methods will work
        sun.build_parameters(model)
        orbit.build_parameters(model)
        jupiter.build_parameters(model)
        jupiter.build_dependent_parameters(model, sun, orbit)

        test_point = {
            sun.mass.value: 1.0,
            orbit.logP.value: np.log10(365.25),
            orbit.cosi.value: 1e-7,
            orbit.sesinw_raw.value: 0.001,
            orbit.secosw.value: 0.001,
            jupiter.mass.value: 1.0
        }

        calculated_k = jupiter.K.value.eval(test_point)

        print(calculated_k)
        #import ipdb; ipdb.set_trace()


        # 4. Extract and Evaluate the K expression
        # Use .eval() to compute the numerical result from the symbolic graph
        #calculated_k = jupiter.K.value.eval()

    print("sini = " + str(orbit.sini.value.eval()))
    print("inc = " + str(orbit.inc.value.eval()*180.0/np.pi) + " deg")
    print("e = " + str(orbit.ecc.value.eval()))
    print("omega = " + str(orbit.omega.value.eval()*180.0/np.pi) + " deg")
    print("period = " + str(orbit.period.value.eval()) + " days")
    print("planet mass = " + str(jupiter.mass.value.eval()) + " MJ")
    print("star mass = " + str(sun.mass.value.eval()) + " MJ")


    # 5. Calculate Truth using Astropy Units for 100% precision
    P_true = 365.25 * u.day
    Ms_true = 1.0 * u.Msun
    Mp_true = 1.0 * u.Mjup

    # Standard K formula: (2*pi*G/P)^(1/3) * (Mp*sin(i)) / (Mtotal^(2/3)) / sqrt(1-e^2)
    truth_k = ((2 * np.pi * const.G / P_true) ** (1 / 3) * (Mp_true) / ((Ms_true + Mp_true) ** (2 / 3))
               ).to(u.m / u.s).value

    print(f"\n[Component Test] {jupiter.name} K: {calculated_k:.6f} m/s")
    print(f"[Astropy Truth]  K: {truth_k:.6f} m/s")

    # Assert that your internal class math matches physics within 0.1%
    np.testing.assert_allclose(calculated_k, truth_k, rtol=1e-3)


if __name__ == "__main__":
    test_planet_k_value()
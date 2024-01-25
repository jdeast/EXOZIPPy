import pymc as pm
from astropy import units as u
import astropy.constants as constants

def build_model(mask=None, start=None):
    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    with pm.Model() as model:
        # Parameters for the stellar properties
        mean_flux = pm.Uniform("mean_flux", mu=0.0, sd=10.0)

        # limb darkening
        u_star = xo.QuadLimbDark("u_star")
        star = xo.LimbDarkLightCurve(u_star)

        # stellar parameters
        mstar = pm.Uniform("mstar", lower=0.0, upper=250.0)*constants.solMass # m_sun
        rstar = pm.Uniform("rstar", lower=0.0, upper=2000.0) # r_sun
        teff  = pm.Uniform("teff", lower=0.0, upper=500000.0) # K
        age = pm.Uniform("age", lower=0.0, upper=13.77) # Gyr
        feh = pm.Uniform("feh", lower=-5.0, upper=5.0) #
        initfeh = pm.Uniform("initfeh", lower=-5.0,upper=5.0)
        parallax = pm.Uniform("initfeh", lower=-5.0,upper=5.0)
        distance = pm.Uniform("initfeh", lower=-5.0,upper=5.0)

        # noinspection PyTypeChecker
        rhostar = pm.Deterministic('rhostar',mstar/rstar**3*rhosun)
        logg = pm.Deterministic('logg',np.log10(mstar/rstar**2*gsun))
        lstar = pm.Deterministic('lstar',4.0*math.pi*rstar**2*teff**4*sigmab*rsun**2/lsun)
        fbol = pm.Deterministic(lstar*lsun/(4.0*math.pi*distance**2.0))

        # Orbital parameters for the planets
        t0 = pm.Normal("t0", mu=np.array(t0s), sd=1, shape=2)
        log_m_pl = pm.Normal("log_m_pl", mu=np.log(msini.value), sd=1, shape=2)
        log_period = pm.Normal("log_period", mu=np.log(periods), sd=1, shape=2)

        # Fit in terms of transit depth (assuming b<1)
        b = pm.Uniform("b", lower=0, upper=1, shape=2)
        log_depth = pm.Normal(
            "log_depth", mu=np.log(depths), sigma=2.0, shape=2
        )
        ror = pm.Deterministic(
            "ror",
            star.get_ror_from_approx_transit_depth(
                1e-3 * tt.exp(log_depth), b
            ),
        )
        r_pl = pm.Deterministic("r_pl", ror * r_star)

        m_pl = pm.Deterministic("m_pl", tt.exp(log_m_pl))
        period = pm.Deterministic("period", tt.exp(log_period))

        ecs = pmx.UnitDisk("ecs", shape=(2, 2), testval=0.01 * np.ones((2, 2)))
        ecc = pm.Deterministic("ecc", tt.sum(ecs**2, axis=0))
        omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
        xo.eccentricity.vaneylen19(
            "ecc_prior", multi=True, shape=2, fixed=True, observed=ecc
        )

        # RV jitter & a quadratic RV trend
        log_sigma_rv = pm.Normal(
            "log_sigma_rv", mu=np.log(np.median(yerr_rv)), sd=5
        )
        trend = pm.Normal(
            "trend", mu=0, sd=10.0 ** -np.arange(3)[::-1], shape=3
        )

        # Transit jitter & GP parameters
        log_sigma_lc = pm.Normal(
            "log_sigma_lc", mu=np.log(np.std(y[mask])), sd=10
        )
        log_rho_gp = pm.Normal("log_rho_gp", mu=0.0, sd=10)
        log_sigma_gp = pm.Normal(
            "log_sigma_gp", mu=np.log(np.std(y[mask])), sd=10
        )

        # Orbit models
        orbit = xo.orbits.KeplerianOrbit(
            r_star=r_star,
            m_star=m_star,
            period=period,
            t0=t0,
            b=b,
            m_planet=xo.units.with_unit(m_pl, msini.unit),
            ecc=ecc,
            omega=omega,
        )

        # Compute the model light curve
        light_curves = (
            star.get_light_curve(orbit=orbit, r=r_pl, t=x[mask], texp=texp)
            * 1e3
        )
        light_curve = pm.math.sum(light_curves, axis=-1) + mean_flux
        resid = y[mask] - light_curve

        # GP model for the light curve
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp),
            rho=tt.exp(log_rho_gp),
            Q=1 / np.sqrt(2),
        )
        gp = GaussianProcess(kernel, t=x[mask], yerr=tt.exp(log_sigma_lc))
        gp.marginal("transit_obs", observed=resid)

        # And then include the RVs as in the RV tutorial
        x_rv_ref = 0.5 * (x_rv.min() + x_rv.max())

        def get_rv_model(t, name=""):
            # First the RVs induced by the planets
            vrad = orbit.get_radial_velocity(t)
            pm.Deterministic("vrad" + name, vrad)

            # Define the background model
            A = np.vander(t - x_rv_ref, 3)
            bkg = pm.Deterministic("bkg" + name, tt.dot(A, trend))

            # Sum over planets and add the background to get the full model
            return pm.Deterministic(
                "rv_model" + name, tt.sum(vrad, axis=-1) + bkg
            )

        # Define the model
        rv_model = get_rv_model(x_rv)
        get_rv_model(t_rv, name="_pred")

        # The likelihood for the RVs
        err = tt.sqrt(yerr_rv**2 + tt.exp(2 * log_sigma_rv))
        pm.Normal("obs", mu=rv_model, sd=err, observed=y_rv)

        # Compute and save the phased light curve models
        pm.Deterministic(
            "lc_pred",
            1e3
            * tt.stack(
                [
                    star.get_light_curve(
                        orbit=orbit, r=r_pl, t=t0[n] + phase_lc, texp=texp
                    )[..., n]
                    for n in range(2)
                ],
                axis=-1,
            ),
        )

        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
        map_soln = pmx.optimize(start=start, vars=[trend])
        map_soln = pmx.optimize(start=map_soln, vars=[log_sigma_lc])
        map_soln = pmx.optimize(start=map_soln, vars=[log_depth, b])
        map_soln = pmx.optimize(start=map_soln, vars=[log_period, t0])
        map_soln = pmx.optimize(
            start=map_soln, vars=[log_sigma_lc, log_sigma_gp]
        )
        map_soln = pmx.optimize(start=map_soln, vars=[log_rho_gp])
        map_soln = pmx.optimize(start=map_soln)

        extras = dict(
            zip(
                ["light_curves", "gp_pred"],
                pmx.eval_in_model([light_curves, gp.predict(resid)], map_soln),
            )
        )

    return model, map_soln, extras


model0, map_soln0, extras0 = build_model()


class Event:
    pass




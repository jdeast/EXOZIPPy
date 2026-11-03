"""The EXOZIPPy prior on one sight line: model, compiled logp, derived nodes.

Needs exozippy importable (PYTHONPATH=<repo>/src).
"""

import numpy as np

LENS_IDX, SOURCE_IDX = 0, 1  # star order in build_config

# What every evaluation returns besides logp: (column, Parameter getter,
# element).  t_E uses mulensevent's mu_rel_geo, which with no light curve
# (no t0_par, no Earth velocity) IS the heliocentric mu_rel; genulens is
# run with ``vEarthlb 0 0`` to match.
DERIVED = (
    ("t_E", lambda s: s.mulensevent.t_E, 0),
    ("theta_E", lambda s: s.mulensevent.theta_E, 0),
    ("mu_rel", lambda s: s.mulensevent.mu_rel_mag, 0),
    ("D_L", lambda s: s.star.distance, LENS_IDX),
    ("D_S", lambda s: s.star.distance, SOURCE_IDX),
    ("M_L", lambda s: s.star.mass, LENS_IDX),
)


def radec(l_deg, b_deg):
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic").icrs
    return float(c.ra.deg), float(c.dec.deg)


def build_config(ra, dec, *, binaries=False, d_max_pc=20000.0, pm_max=50.0):
    """Lens + source + galactic model with NO light curve.

    The bounds tighten the defaults.yaml supports (1 mpc - 100 kpc,
    +/-1e5 mas/yr, +/-c) only to keep the coordinates well scaled; they do
    not move the answer: the densities are negligible beyond 20 kpc on a
    bulge line (genulens's own grid stops at 16 kpc unless told otherwise),
    50 mas/yr at 1 kpc is 240 km/s (> 5 sigma of the thick disk), and a
    1000 km/s rv is > 7 sigma of every branch.  The lens distance starts at
    100 pc; n D^2 x theta_E mu_rel gives the nearest 100 pc a negligible
    share of the rate.

    """
    gm = {"name": "gm", "anchor_idx": 1}
    if binaries:
        gm["binaries"] = True
    config = {
        "star": [{"name": "Lens"}, {"name": "Source"}],
        "lens": [{"body": "star.Lens"}],
        "source": [{"body": "star.Source"}],
        "mulensevent": [{"finite_source": False}],
        # anchor_idx 1: the line of sight is the SOURCE's.
        "galacticmodel": [gm],
        "logger_level": "WARNING",
    }
    params = {}
    for star, d0 in (("Lens", 6000.0), ("Source", 8500.0)):
        params[f"star.{star}.ra"] = {"initval": ra, "sigma": 0.0}
        params[f"star.{star}.dec"] = {"initval": dec, "sigma": 0.0}
        params[f"star.{star}.distance"] = {
            "initval": d0,
            "lower": 100.0,
            "upper": d_max_pc,
        }
        params[f"star.{star}.pm_ra"] = {"lower": -pm_max, "upper": pm_max}
        params[f"star.{star}.pm_dec"] = {"lower": -pm_max, "upper": pm_max}
        params[f"star.{star}.rv"] = {
            "initval": 0.0,
            "lower": -1.0e6,
            "upper": 1.0e6,
        }
    params["star.Lens.logmass"] = {"initval": -0.5}
    return config, params


class Layout:
    """Pack the model's raw value variables into one flat vector."""

    def __init__(self, model):
        self.names = [v.name for v in model.value_vars]
        self.shapes = [tuple(v.type.shape) for v in model.value_vars]
        for name, shape in zip(self.names, self.shapes):
            if any(s is None for s in shape):
                raise ValueError(f"value variable {name} has no static shape")
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)]).astype(int)
        self.total = int(self.offsets[-1])

    def index(self, name, element=0):
        return int(self.offsets[self.names.index(name)] + element)

    def unpack(self, vec):
        return [
            np.asarray(vec[a:b], float).reshape(sh)
            for a, b, sh in zip(
                self.offsets[:-1], self.offsets[1:], self.shapes
            )
        ]

    def pack_trace(self, posterior):
        """(n_draws, total) array from an arviz posterior of raw variables."""
        cols = []
        for name, size in zip(self.names, self.sizes):
            if name not in posterior:
                raise KeyError(f"value variable {name} absent from the trace")
            a = np.asarray(posterior[name])
            cols.append(a.reshape(a.shape[0] * a.shape[1], size))
        return np.concatenate(cols, axis=1)


class PriorModel:
    """The compiled prior for one sight line.

    ``evaluate(vec)`` returns the model's own logp in its raw coordinates
    (the density the samplers see: every prior, potential and Jacobian)
    followed by the internal-unit values of ``DERIVED``.
    """

    def __init__(self, l_deg, b_deg, *, binaries=False):
        import pytensor

        from exozippy.system import System

        self.l, self.b, self.binaries = l_deg, b_deg, binaries
        self.ra, self.dec = radec(l_deg, b_deg)
        config, params = build_config(self.ra, self.dec, binaries=binaries)
        self.system = System(config, user_params=params)
        self.system.prepare()
        self.model = self.system.build_model()
        if int(self.system.mulensevent.lens_bodies[0][1]) != LENS_IDX:
            raise RuntimeError("the event's lens is not star 0")
        self.layout = Layout(self.model)
        self.params = [(name, get(self.system), i) for name, get, i in DERIVED]
        outputs = [self.model.logp(jacobian=True)] + [
            p.value for _, p, _ in self.params
        ]
        nodes = self.model.replace_rvs_by_values(outputs)
        self.fn = pytensor.function(
            self.model.value_vars, nodes, on_unused_input="ignore"
        )
        self.fn.trust_input = True
        # The lens log-mass raw coordinate gets its own proposal.
        self.logmass = self.system.star.logmass
        self.i_logmass = self.layout.index("star.logmass_raw", 0)

    def evaluate(self, vec):
        out = self.fn(*self.layout.unpack(vec))
        row = np.empty(1 + len(self.params))
        row[0] = float(out[0])
        for k, ((_, _, i), v) in enumerate(zip(self.params, out[1:]), start=1):
            row[k] = float(np.atleast_1d(v)[i])
        return row

    def to_user_units(self, rows):
        """Internal -> user units for the derived columns of ``rows``."""
        cols = {}
        for k, (name, p, i) in enumerate(self.params, start=1):
            cols[name] = np.asarray(
                p.from_internal(rows[:, k], index=i), float
            )
        return cols

    def logmass_raw_from_phys(self, log_m):
        return self.logmass.element_raw_from_phys(LENS_IDX, log_m)

    def pilot_nuts(self, *, draws, tune, chains, seed):
        """A short NUTS run: the starting point for the proposal ONLY.

        NUTS does not visit the black-hole lens mode (the present-day mass
        function has a ~19-dex trough near 3 Msun between the luminous
        stars and the black holes), so these draws are never used as the
        prior; the proposal built from them gets a separate log-mass
        component that covers that mode.
        """
        import pymc as pm

        with self.model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=chains,
                target_accept=0.9,
                # The pilot only seeds the proposal; a capped tree keeps a
                # chain that wanders into a stiff corner (some nuclear-disk
                # lines) from taking an hour of 1024-step trajectories.
                max_treedepth=8,
                random_seed=seed,
                progressbar=False,
                compute_convergence_checks=False,
            )
        div = int(np.asarray(idata.sample_stats["diverging"]).sum())
        return self.layout.pack_trace(idata.posterior), div

"""Simulation-based calibration (SBC) of whole exozippy models (review 7.16.4).

Talts et al. (2018, arXiv:1804.06788): draw theta* from the PRIOR, simulate
data y ~ p(y | theta*), fit, and record the rank of theta* among L thinned
posterior draws.  If the prior, the likelihood and every Jacobian agree, the
rank is uniform on {0, ..., L} for every scalar function of the parameters --
sampled or derived, whatever the posterior's shape.  A U-shaped rank
histogram means the posterior is too NARROW, a hump that it is too WIDE, a
slope that it is BIASED.

What this harness tests, and what it deliberately does not:

* The prior is the one the BUILT model applies.  The prior draws come from
  NUTS on the built model with its observed (likelihood) terms removed, so
  every logit-uniform Jacobian, Gaussian prior, component potential (Chen,
  the collision bound, the chord Jacobian, the PK ordering bound...) and soft
  barrier is in it.  Nothing here writes a second copy of the prior.
  ``pm.sample_prior_predictive`` CANNOT be used: a logit element's raw
  coordinate is a CustomDist whose forward ``random`` draws N(0, 1), which is
  not its prior (parameter.md, "The raw coordinate of a logit element is
  FLAT"); its prior lives in a potential that forward sampling ignores.
* The data are simulated by an INDEPENDENT forward model (a Keplerian, the
  uniform-disk overlap area, the closed-form one-compartment curve) from the
  physical truth, never by exozippy's own graph, so a shared forward-model bug
  cannot cancel.  Every replicate is then cross-checked: exozippy's model mean
  and sigma at the truth must agree with the independent ones, or the
  replicate is refused (a disagreement means the harness or the code has the
  physics wrong, and the ranks would measure that, not calibration).
* The prior must not depend on the data.  Every sampled element's start,
  bounds and Gaussian prior are written in the params file (so no data hint
  can move them), the design (times, error bars) is fixed, and every replicate
  ASSERTS that its build's frozen raw transform and its prior log density at
  the truth are identical to the prior build's.
* The fit is NUTS (pymc), started at the truth with pymc's jitter, and the
  whitening is NOT measured (``measure_scales: false`` in production terms):
  the measured soft-barrier steepness is data-dependent, which would make the
  prior differ between the prior build and the fit by construction.  NUTS
  adapts its own mass matrix.  Starting at the truth means SBC tests the
  MODEL, not mode-finding; a multi-modal posterior that the chains do not
  cross shows up as Rhat failures or as a hump, which is why non-converged
  replicates are counted and reported, never silently dropped.

Subcommands (run from anywhere; outputs go under --outdir):

    python sbc.py prior   --model rv_circ --n 300 --outdir OUT
    python sbc.py run     --model rv_circ --start 0 --count 10 --outdir OUT
    python sbc.py analyze --model rv_circ --outdir OUT

Models: rv_circ, rv_ecc, transit, transit_pinq, pk, mulens_pspl.  ``prior`` writes OUT/<model>/prior.npz;
``run`` writes one OUT/<model>/rows/rep_NNNN.json per replicate; ``analyze``
writes OUT/<model>/summary.json and the rank-histogram and ECDF-difference
figures (with the simultaneous band of Sailynoja, Buerkner & Vehtari 2022,
Stat. Comput. 32, 32).
"""

import argparse
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Number of posterior draws each rank is computed against.  L = 99 gives 100
# possible ranks, which bin evenly into 20 bins of 5.
L_DRAWS = 99
N_BINS = 20

T0 = 2459000.0
RJUP_IN_RSUN = 0.10276269  # informational only; never used to build truth


# ---------------------------------------------------------------------------
# Independent forward models.
# ---------------------------------------------------------------------------
def kepler_E(M, e):
    """Eccentric anomaly by Newton iteration (vectorized, to 1e-14)."""
    M = np.mod(M, 2.0 * np.pi)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(100):
        f = E - e * np.sin(E) - M
        dE = -f / (1.0 - e * np.cos(E))
        E = E + dE
        if np.max(np.abs(dE)) < 1e-14:
            break
    return E


def keplerian_rv(t, period, tc, K, e, omega_star):
    """Stellar RV (positive = receding) for a Keplerian orbit.

    ``tc`` is the time of transit (inferior conjunction of the companion),
    where the true anomaly is pi/2 - omega_star.  ``omega_star`` is the
    argument of periastron of the STAR, in radians.
    """
    nu_c = np.pi / 2.0 - omega_star
    E_c = 2.0 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu_c / 2.0))
    M_c = E_c - e * np.sin(E_c)
    tp = tc - M_c * period / (2.0 * np.pi)
    M = 2.0 * np.pi * (t - tp) / period
    E = kepler_E(M, e)
    nu = 2.0 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2.0), np.sqrt(1 - e) * np.cos(E / 2.0)
    )
    return K * (np.cos(nu + omega_star) + e * np.cos(omega_star))


def overlap_area(z, p):
    """Area of overlap of a unit disk and a disk of radius p at separation z
    (Mandel & Agol 2002 eq. 1)."""
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    full = z <= (1.0 - p)
    out[full] = np.pi * p * p
    part = (~full) & (z < 1.0 + p)
    zp = z[part]
    k0 = np.arccos(np.clip((p * p + zp * zp - 1.0) / (2.0 * p * zp), -1, 1))
    k1 = np.arccos(np.clip((1.0 - p * p + zp * zp) / (2.0 * zp), -1, 1))
    rad = np.maximum(4.0 * zp * zp - (1.0 + zp * zp - p * p) ** 2, 0.0)
    out[part] = p * p * k0 + k1 - 0.5 * np.sqrt(rad)
    return out


def circular_transit_flux(t, period, tc, ar, cosi, p):
    phase = 2.0 * np.pi * (t - tc) / period
    z = ar * np.sqrt(np.sin(phase) ** 2 + (cosi * np.cos(phase)) ** 2)
    z = np.where(np.cos(phase) >= 0.0, z, 10.0 * ar + 10.0)
    return 1.0 - overlap_area(z, p) / np.pi


def one_compartment(t, dose, ka, ke, v):
    """Oral single dose, first-order absorption, one compartment."""
    return dose * ka / (v * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))


def earth_skowron_deviations(t, t0_par):
    """Earth's offset (AU, ICRS equatorial) from its straight-line motion
    through its t0_par position and velocity (Skowron et al. 2011), from
    astropy's own ephemeris -- not exozippy's ephemeris module."""
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time

    def posvel(tt):
        p, v = get_body_barycentric_posvel("earth", Time(tt, format="jd", scale="tdb"))
        return (
            np.atleast_2d(p.xyz.to("au").value.T),
            np.atleast_2d(v.xyz.to("au/d").value.T),
        )

    pos, _ = posvel(np.asarray(t, dtype=float))
    p0, v0 = posvel(np.array([t0_par]))
    return pos - (p0 + v0 * (np.asarray(t)[:, None] - t0_par))


def pspl_parallax_mag(t, t0, u0, tE, piEN, piEE, ra, dec, t0_par):
    """Point-lens point-source magnification with annual parallax, in the
    geocentric frame: tau = (t - t0)/tE - dN piEN - dE piEE,
    beta = u0 + dN piEE - dE piEN (the MulensModel / Skowron et al. 2011
    convention; d = the observer's sky-projected offset)."""
    xyz = earth_skowron_deviations(t, t0_par)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    de = -x * np.sin(ra) + y * np.cos(ra)
    dn = -x * np.cos(ra) * np.sin(dec) - y * np.sin(ra) * np.sin(dec) + z * np.cos(dec)
    tau = (t - t0) / tE - dn * piEN - de * piEE
    beta = u0 + dn * piEE - de * piEN
    u2 = tau**2 + beta**2
    return (u2 + 2.0) / np.sqrt(u2 * (u2 + 4.0))


# ---------------------------------------------------------------------------
# Model specifications.  Each one fixes a DESIGN (times, error bars), a config,
# a params file that states every sampled element's start and prior, and an
# independent simulator from the physical truth.
# ---------------------------------------------------------------------------
class Spec:
    name = None
    # Sampler budget for the fit.
    tune = 1000
    draws = 1000
    chains = 4
    target_accept = 0.9
    # Sampler budget for the prior-only chains.
    prior_tune = 3000
    prior_chains = 8
    prior_target_accept = 0.95

    def write_data(self, workdir, values=None):
        raise NotImplementedError

    def config(self):
        raise NotImplementedError

    def params(self):
        raise NotImplementedError

    def simulate(self, truth, system, rng):
        """Return (mean, sigma, observed) in the DATA FILE's units."""
        raise NotImplementedError


def _user(system, label, internal):
    """Internal -> user units through the Parameter itself (CLAUDE.md: never
    hand-write a conversion factor)."""
    comp, pname = label.split(".", 1)
    par = getattr(system.active_components[comp], pname)
    return float(par.from_internal(float(internal), index=0))


class RVSpec(Spec):
    """One planet, one RV instrument, RV only."""

    n_epochs = 40
    err = 5.0  # m/s

    def __init__(self, eccentric):
        self.eccentric = eccentric
        self.name = "rv_ecc" if eccentric else "rv_circ"
        if eccentric:
            self.tune = 2000
            self.draws = 1000
        rng = np.random.default_rng(20261029)
        self.t = np.sort(T0 + rng.uniform(-30.0, 30.0, self.n_epochs))

    def write_data(self, workdir, values=None):
        v = np.zeros_like(self.t) if values is None else values
        np.savetxt(
            Path(workdir) / "synth.rv",
            np.column_stack([self.t, v, np.full_like(self.t, self.err)]),
            fmt="%.10f",
        )

    def config(self):
        return {
            "star": [{"name": "A", "mist": False}],
            "planet": [{"name": "b"}],
            "orbit": [{"name": "b"}],
            "rvinstrument": [{"name": "SYNTH", "file": "synth.rv"}],
        }

    def params(self):
        p = {
            # The star is not a parameter of an RV-only fit: nothing but a
            # prior would constrain it.  cos i is pinned edge-on, so the mass
            # is the minimum mass and the mass coordinate is linear.
            "star.A.logmass": {"initval": 0.0, "sigma": 0},
            "star.A.radius": {"initval": 1.0, "sigma": 0},
            "orbit.b.cosi": {"initval": 0.0, "sigma": 0},
            "orbit.b.logP": {
                "initval": float(np.log10(5.0)),
                "lower": float(np.log10(4.8)),
                "upper": float(np.log10(5.2)),
            },
            # A Gaussian, not a box: the orbit's tc window is one period
            # wide whatever the user bounds say, so a uniform tc prior would
            # wrap at its edges.  The bounds ARE written (equal to that
            # window) because a user-bounded tc is what keeps the sampled
            # epoch at the seed's: otherwise the epoch follows the data's
            # information-weighted centre, which moves with the simulated
            # data and changes the parameterization from one replicate to
            # the next (measured: tc became DERIVED from tc_sampled on one
            # of two pilot replicates).
            "orbit.b.tc": {
                "initval": T0,
                "mu": T0,
                "sigma": 0.3,
                "lower": T0 - 2.5,
                "upper": T0 + 2.5,
            },
            "planet.b.mass": {"initval": 0.6, "lower": 0.2, "upper": 1.0},
            "rvinstrument.SYNTH.gamma": {
                "initval": 0.0,
                "lower": -50.0,
                "upper": 50.0,
            },
            "rvinstrument.SYNTH.jitter_variance": {
                "initval": 25.0,
                "lower": 0.0,
                "upper": 100.0,
            },
        }
        if self.eccentric:
            p["orbit.b.secosw"] = {"initval": 0.1, "lower": -0.5, "upper": 0.5}
            p["orbit.b.sesinw"] = {"initval": 0.1, "lower": -0.5, "upper": 0.5}
        else:
            p["orbit.b.secosw"] = {"initval": 0.0, "sigma": 0}
            p["orbit.b.sesinw"] = {"initval": 0.0, "sigma": 0}
        return p

    def simulate(self, truth, system, rng):
        period = truth["orbit.period"][0]
        tc = truth["orbit.tc"][0]
        e = truth["orbit.ecc"][0] if self.eccentric else 0.0
        omega = truth["orbit.omega"][0] if self.eccentric else np.pi / 2.0
        K = _user(system, "orbit.K", truth["orbit.K"][0])
        gamma = _user(
            system, "rvinstrument.gamma", truth["rvinstrument.gamma"][0]
        )
        jv = _user(
            system,
            "rvinstrument.jitter_variance",
            truth["rvinstrument.jitter_variance"][0],
        )
        mean = gamma + keplerian_rv(self.t, period, tc, K, e, omega)
        sigma = np.sqrt(self.err**2 + jv) * np.ones_like(self.t)
        return mean, sigma, mean + sigma * rng.standard_normal(self.t.size)


class TransitSpec(Spec):
    """One circular planet, one light curve, uniform disk (u1 pinned at 0)."""

    name = "transit"
    err = 1.0e-3
    tune = 1500
    draws = 1000

    def __init__(self, pin_mass=False):
        # pin_mass: the "transit_pinq" variant.  A transit measures no mass;
        # the default samples log_q under the Chen mass-radius prior alone,
        # and that one coordinate is where the transit leg's divergences and
        # Rhat failures concentrate.  Pinning it isolates the geometry.
        self.pin_mass = pin_mass
        if pin_mass:
            self.name = "transit_pinq"
        cad = 10.0 / 1440.0
        windows = [T0 + k * 5.0 for k in range(3)]
        self.t = np.concatenate(
            [np.arange(c - 0.4, c + 0.4, cad) for c in windows]
        )

    def write_data(self, workdir, values=None):
        v = np.ones_like(self.t) if values is None else values
        np.savetxt(
            Path(workdir) / "synth.trn",
            np.column_stack([self.t, v, np.full_like(self.t, self.err)]),
            fmt="%.10f",
        )

    def config(self):
        return {
            "star": [{"name": "A", "mist": False}],
            "planet": [{"name": "b"}],
            "orbit": [
                {
                    "name": "b",
                    "primary": ["A"],
                    "companion": ["b"],
                    # Circular: the eccentricity coordinates are pinned
                    # below, and V_c/V_e is singular at e = 0.  The chord stays
                    # on: it is the coordinate a transit-only fit uses and it
                    # takes the sign of cos i from i180 instead of sampling it.
                    "fitvcve": False,
                    "fitchord": True,
                }
            ],
            "band": [
                {
                    "name": "TESS",
                    "filter": "TESS",
                    "ld_law": "linear",
                    "star_constrains_ld": False,
                }
            ],
            "transit": [
                {"name": "SYNTH", "file": "synth.trn", "band": "TESS"}
            ],
        }

    def params(self):
        p = self._params()
        if self.pin_mass:
            p["planet.b.log_q"] = {"initval": -3.0, "sigma": 0}
        return p

    def _params(self):
        return {
            "star.A.logmass": {"initval": 0.0, "sigma": 0},
            "star.A.radius": {"initval": 1.0, "sigma": 0},
            "orbit.b.secosw": {"initval": 0.0, "sigma": 0},
            "orbit.b.sesinw": {"initval": 0.0, "sigma": 0},
            "band.TESS.u1": {"initval": 0.0, "sigma": 0},
            "orbit.b.logP": {
                "initval": float(np.log10(5.0)),
                "lower": float(np.log10(4.95)),
                "upper": float(np.log10(5.05)),
            },
            # Bounds equal to the orbit's own window: see RVSpec.
            "orbit.b.tc": {
                "initval": T0,
                "mu": T0,
                "sigma": 0.02,
                "lower": T0 - 2.5,
                "upper": T0 + 2.5,
            },
            # p = Rp/R* in [0.051, 0.154].
            "planet.b.radius": {"initval": 1.0, "lower": 0.5, "upper": 1.5},
            # chord = sqrt((1+p)^2 - b^2).  Upper 1.05 < 1 + p_min, so the
            # chord can never pass the central-transit value (the region the
            # radicand floor and the chord_geometry soft bound handle);
            # lower 0.5 keeps the truth away from grazing geometries.
            "orbit.b.chord": {"initval": 0.9, "lower": 0.5, "upper": 1.05},
            "transit.SYNTH.baseline": {
                "initval": 1.0,
                "lower": 0.99,
                "upper": 1.01,
            },
            "transit.SYNTH.jitter_variance": {
                "initval": 1.0e-7,
                "lower": 0.0,
                "upper": 1.0e-6,
            },
        }

    def simulate(self, truth, system, rng):
        period = truth["orbit.period"][0]
        tc = truth["orbit.tc"][0]
        ar = truth["planet.ar"][0]
        cosi = truth["orbit.cosi"][0]
        p = truth["planet.p"][0]
        f0 = truth["transit.baseline"][0]
        jv = truth["transit.jitter_variance"][0]
        mean = f0 * circular_transit_flux(self.t, period, tc, ar, cosi, p)
        sigma = np.sqrt(self.err**2 + jv) * np.ones_like(self.t)
        return mean, sigma, mean + sigma * rng.standard_normal(self.t.size)


class PKSpec(Spec):
    """One subject, oral single dose, one compartment, combined error."""

    name = "pk"
    weight = 70.0
    dose_per_kg = 4.0
    tune = 1500
    draws = 1000

    def __init__(self):
        self.t = np.array(
            [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 24.0]
        )

    def write_data(self, workdir, values=None):
        v = np.ones_like(self.t) if values is None else values
        with open(Path(workdir) / "pk.csv", "w") as f:
            f.write("Subject,Time,conc\n")
            for ti, ci in zip(self.t, v):
                f.write(f"1,{float(ti)!r},{float(ci)!r}\n")

    def config(self):
        return {
            "subject": [
                {
                    "name": "S1",
                    "weight": self.weight,
                    "dose": self.dose_per_kg,
                    "dose_unit": "mg/kg",
                    "fitclke": True,
                }
            ],
            "assay": [
                {
                    "name": "serum",
                    "datafile": "pk.csv",
                    "columns": {
                        "subject": "Subject",
                        "time": "Time",
                        "obs": "conc",
                    },
                    "subject_prefix": "S",
                    "conc_unit": "mg/L",
                    "time_unit": "hr",
                }
            ],
        }

    def params(self):
        # Non-overlapping ka and ke ranges: the oral one-compartment curve is
        # exactly symmetric under ka <-> ke in this basis (flip-flop), and
        # disjoint priors are what make the posterior single-moded.
        return {
            "subject.S1.log_cl": {"initval": 0.45, "lower": 0.1, "upper": 0.8},
            "subject.S1.log_ka": {"initval": 0.2, "lower": -0.3, "upper": 0.7},
            "subject.S1.log_ke": {
                "initval": -1.1,
                "lower": -1.5,
                "upper": -0.8,
            },
            "assay.serum.sigma_add": {
                "initval": 0.2,
                "lower": 0.05,
                "upper": 0.5,
            },
            "assay.serum.sigma_prop": {
                "initval": 0.08,
                "lower": 0.01,
                "upper": 0.2,
            },
        }

    def simulate(self, truth, system, rng):
        ka = truth["subject.ka"][0]
        ke = truth["subject.ke"][0]
        v = truth["subject.v"][0]
        dose = self.dose_per_kg * self.weight
        sa = truth["assay.sigma_add"][0]
        sp = truth["assay.sigma_prop"][0]
        mean = one_compartment(self.t, dose, ka, ke, v)
        sigma = np.sqrt(sa**2 + (sp * mean) ** 2)
        return mean, sigma, mean + sigma * rng.standard_normal(self.t.size)


class MulensSpec(Spec):
    """1L1S point-source point-lens event, one light curve in flux.

    The physical state (both stars' masses, distances, proper motions and
    radial velocities) is sampled under the full galactic prior and the
    event-rate weight -- the defaults, deliberately: that prior is what is
    being tested.  Only the light-curve nuisances and the event's own
    observables get user boxes.  Annual parallax is always on in this
    model, so the independent simulator carries it (astropy's Earth).
    """

    name = "mulens_pspl"
    ra_deg, dec_deg = 266.87, -34.73
    err = 0.01  # flux, baseline ~1
    tune = 2000
    draws = 1000
    # Measured on the galactic prior alone (no data), default coordinates:
    # 20% of NUTS transitions diverge at target_accept 0.95 and 1.5% at
    # 0.995, where the adapted step size falls to ~2e-3.  The full
    # coordinate-swap set cut the prior's divergences from 75 to 4 in 400.
    target_accept = 0.99
    prior_target_accept = 0.99

    def __init__(self, swaps=()):
        # swaps: mulensevent coordinate switches (fitmurel, fitpirel,
        # fitthetae).  Each carries its exact Jacobian, so the prior is the
        # same; only the coordinates NUTS walks change (mulensing.md,
        # "Choosing a coordinate set").
        self.swaps = tuple(swaps)
        if self.swaps:
            self.name = "mulens_pspl_" + "_".join(
                w.replace("fit", "") for w in self.swaps
            )
        self.t = np.arange(T0 - 200.0, T0 + 200.0, 0.5)

    def write_data(self, workdir, values=None):
        v = np.ones_like(self.t) if values is None else values
        np.savetxt(
            Path(workdir) / "synth.flux",
            np.column_stack([self.t, v, np.full_like(self.t, self.err)]),
            fmt="%.10f",
        )

    def config(self):
        return {
            "star": [{"name": "Lens"}, {"name": "Source"}],
            "mulensevent": [
                {
                    "finite_source": False,
                    "t0_par": T0,
                    "use_op": False,
                    "peak_find": False,
                    **{w: True for w in self.swaps},
                }
            ],
            "lens": [{"body": "star.Lens"}],
            "source": [{"body": "star.Source"}],
            "galacticmodel": [{"name": "synth", "anchor_idx": 1}],
            "band": [{"name": "Cousins_I", "filter": "Cousins_I"}],
            "mulensinstrument": [
                {
                    "name": "SYNTH",
                    "file": "synth.flux",
                    "band": "Cousins_I",
                    "data_format": "flux",
                }
            ],
        }

    def params(self):
        return {
            "star.Lens.ra": {"initval": self.ra_deg, "sigma": 0},
            "star.Lens.dec": {"initval": self.dec_deg, "sigma": 0},
            "star.Source.ra": {"initval": self.ra_deg, "sigma": 0},
            "star.Source.dec": {"initval": self.dec_deg, "sigma": 0},
            "source.Source.t_0": {"initval": T0, "lower": T0 - 5.0, "upper": T0 + 5.0},
            # u_0 > 0: +/- u_0 are (nearly, with parallax) the same event,
            # and a chain started at the truth would see only one of them.
            "source.Source.u_0": {"initval": 0.3, "lower": 0.01, "upper": 1.0},
            "mulensinstrument.SYNTH.log_f_total": {
                "initval": 0.0,
                "lower": -0.2,
                "upper": 0.2,
            },
            "mulensinstrument.SYNTH.q_source": {
                "initval": 0.6,
                "lower": 0.2,
                "upper": 1.0,
            },
            "mulensinstrument.SYNTH.err_scale": {
                "initval": 1.0,
                "lower": 0.8,
                "upper": 1.25,
            },
        }

    def simulate(self, truth, system, rng):
        ra, dec = np.radians(self.ra_deg), np.radians(self.dec_deg)
        A = pspl_parallax_mag(
            self.t,
            truth["source.t_0"][0],
            truth["source.u_0"][0],
            truth["mulensevent.t_E"][0],
            truth["mulensevent.pi_E_N"][0],
            truth["mulensevent.pi_E_E"][0],
            ra,
            dec,
            T0,
        )
        fs = truth["mulensinstrument.f_source"][0]
        fb = truth["mulensinstrument.f_blend"][0]
        mean = fs * A + fb
        sigma = self.err * truth["mulensinstrument.err_scale"][0] * np.ones_like(self.t)
        return mean, sigma, mean + sigma * rng.standard_normal(self.t.size)


SPECS = {
    "rv_circ": lambda: RVSpec(eccentric=False),
    "rv_ecc": lambda: RVSpec(eccentric=True),
    "transit": TransitSpec,
    "transit_pinq": lambda: TransitSpec(pin_mass=True),
    "pk": PKSpec,
    "mulens_pspl": MulensSpec,
    "mulens_pspl_murel": lambda: MulensSpec(swaps=("fitmurel",)),
    "mulens_pspl_murel_pirel_thetae": lambda: MulensSpec(
        swaps=("fitmurel", "fitpirel", "fitthetae")
    ),
}


# ---------------------------------------------------------------------------
# Building, the prior-only model, and the per-build invariants.
# ---------------------------------------------------------------------------
def build(spec, workdir, values=None):
    """Write the design (and `values` as the observations), then build."""
    from exozippy.system import System

    spec.write_data(workdir, values)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        system = System(dict(spec.config()), user_params=dict(spec.params()))
        system.prepare()
        model = system.build_model()
    finally:
        os.chdir(cwd)
    return system, model


def sampled_parameters(system):
    return [
        p
        for p in system.get_all_parameters()
        if getattr(p, "_raw_transform", None) is not None
    ]


def transform_fingerprint(system):
    """Everything that defines a sampled element's prior and raw coordinate.

    Equality between two builds means the same raw point is the same physical
    point AND carries the same prior: bounds, Gaussian centres and widths,
    logit anchors and scales.  A data-dependent bound or prior centre makes
    this differ between the prior build and a replicate's.
    """
    out = {}
    for p in sampled_parameters(system):
        tf = p._raw_transform
        out[p.label] = {
            k: np.asarray(tf[k], dtype=float).tolist()
            for k in (
                "sampled_idx",
                "lowers",
                "uppers",
                "logit_q_inits",
                "init_scale_logits",
                "gaussian_mus",
                "gaussian_scales",
            )
        }
        out[p.label]["use_logit"] = np.asarray(tf["use_logit"]).tolist()
    return out


def prior_only_model(model):
    """The built model minus its observed variables -- the prior, exactly.

    Keeps every free RV, every potential (Jacobians, component priors, soft
    barriers) and every Deterministic.  Refuses a model whose likelihood is
    not purely observed RVs, because a likelihood potential would then leak
    into the "prior".
    """
    from pymc.model.fgraph import (
        ModelObservedRV,
        fgraph_from_model,
        model_from_fgraph,
    )

    if not model.observed_RVs:
        raise RuntimeError("model has no observed variables: nothing to drop")
    saved = dict(model.rvs_to_initial_values)
    try:
        # fgraph_from_model refuses non-default initvals; they only set the
        # start, which the caller supplies again.
        for k in saved:
            model.rvs_to_initial_values[k] = None
        fg, _ = fgraph_from_model(model)
    finally:
        for k, v in saved.items():
            model.rvs_to_initial_values[k] = v
    drop = [
        i
        for i, o in enumerate(fg.outputs)
        if isinstance(o.owner.op, ModelObservedRV)
    ]
    for i in reversed(drop):
        fg.remove_output(i)
    prior = model_from_fgraph(fg, mutate_fgraph=True)
    return prior


def free_raw_point(model, system, phys):
    """Raw point (keyed by free RV name) for a physical, internal-unit truth
    ``phys`` {label: full vector}, through each Parameter's frozen transform."""
    lookup = {p.label: p for p in sampled_parameters(system)}
    point = {}
    for rv in model.free_RVs:
        label = rv.name[: -len("_raw")] if rv.name.endswith("_raw") else None
        if label not in lookup:
            raise RuntimeError(
                f"free RV {rv.name!r} is not a Parameter's raw vector; the "
                "harness only knows how to place Parameter coordinates"
            )
        point[rv.name] = np.asarray(
            lookup[label].raw_from_initval(np.asarray(phys[label])),
            dtype=float,
        )
    return point


def compile_on_values(model, nodes):
    import pytensor

    repl = model.replace_rvs_by_values(nodes)
    return pytensor.function(model.value_vars, repl, on_unused_input="ignore")


def point_args(model, point):
    return [np.asarray(point[v.name]) for v in model.value_vars]


def prior_logp_fn(model):
    """log prior density (raw space): free RVs + every potential."""
    return model.compile_logp(vars=model.free_RVs + model.potentials)


def observed_mu_sigma(model):
    """The (mu, sigma) graphs of every observed Normal, concatenated."""
    import pytensor.tensor as pt

    mus, sigmas = [], []
    for obs in model.observed_RVs:
        op = obs.owner.op
        if getattr(op, "name", "") != "normal":
            raise RuntimeError(
                f"observed variable {obs.name!r} is a {op!r}, not a Normal; "
                "the cross-check only knows the Gaussian likelihood"
            )
        mu, sigma = op.dist_params(obs.owner)
        mus.append(pt.broadcast_to(mu, obs.shape).ravel())
        sigmas.append(pt.broadcast_to(sigma, obs.shape).ravel())
    return pt.concatenate(mus), pt.concatenate(sigmas)


def observed_data(model):
    out = []
    for obs in model.observed_RVs:
        val = model.rvs_to_values[obs]
        try:
            out.append(np.asarray(val.eval()).ravel())
        except Exception:  # noqa: BLE001
            out.append(np.asarray(val.data).ravel())
    return np.concatenate(out)


def tracked_quantities(trace_ds, var_names):
    """{name[i]: (chain, draw) array} for every scalar element of the given
    Deterministics."""
    out = {}
    for name in var_names:
        if name not in trace_ds:
            continue
        a = np.asarray(trace_ds[name])
        if a.ndim == 2:
            out[name] = a
            continue
        flat = a.reshape(a.shape[0], a.shape[1], -1)
        for i in range(flat.shape[-1]):
            out[f"{name}[{i}]"] = flat[..., i]
    return out


# ---------------------------------------------------------------------------
# Stage 1: prior draws from the built model.
# ---------------------------------------------------------------------------
def cmd_prior(args, spec=None):
    import arviz as az
    import pymc as pm

    spec = spec or SPECS[args.model]()
    out = Path(args.outdir) / spec.name
    out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"sbcprior_{spec.name}_"))
    system, model = build(spec, workdir)
    prior = prior_only_model(model)
    det_names = [d.name for d in prior.deterministics]

    n = int(args.n)
    chains = spec.prior_chains
    # --raw-per-draw raw draws per kept draw (default 40): beyond the prior
    # autocorrelation time measured here, which the report checks against the
    # bulk ESS (ess_per_kept_draw must exceed 1).
    per_chain = int(np.ceil(args.raw_per_draw * n / chains))
    with prior:
        idata = pm.sample(
            draws=per_chain,
            tune=spec.prior_tune,
            chains=chains,
            cores=min(chains, args.cores),
            target_accept=spec.prior_target_accept,
            random_seed=args.seed,
            progressbar=False,
            compute_convergence_checks=False,
        )
    post = idata.posterior
    q = tracked_quantities(post, det_names)
    q = {
        k: v for k, v in q.items() if np.all(np.isfinite(v)) and np.std(v) > 0
    }
    ds = az.convert_to_dataset({k: v for k, v in q.items()})
    rhat = az.rhat(ds)
    ess = az.ess(ds, method="bulk")
    diag = {k: {"rhat": float(rhat[k]), "ess_bulk": float(ess[k])} for k in q}
    n_div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    worst_rhat = max(d["rhat"] for d in diag.values())
    worst_ess = min(d["ess_bulk"] for d in diag.values())

    # Thin: n draws spread evenly over the pooled chains.
    total = chains * per_chain
    pick = np.linspace(0, total - 1, n).round().astype(int)
    phys = {}
    for name in det_names:
        a = np.asarray(post[name])
        a = a.reshape(total, *a.shape[2:])
        phys[name] = a[pick]
    raw_names = [v.name for v in prior.free_RVs]
    raw = {}
    for name in raw_names:
        a = np.asarray(post[name])
        raw[name] = a.reshape(total, *a.shape[2:])[pick]

    # Prior log density at every kept draw, in the PRIOR build's raw
    # coordinates.  Each replicate re-evaluates its own build's prior at the
    # same physical point and must reproduce this number.
    lp = prior_logp_fn(prior)
    prior_lp = np.array(
        [lp({k: raw[k][j] for k in raw_names}) for j in range(n)]
    )

    fp = transform_fingerprint(system)
    np.savez(
        out / "prior.npz",
        **{f"phys::{k}": v for k, v in phys.items()},
        **{f"raw::{k}": v for k, v in raw.items()},
        prior_lp=prior_lp,
    )
    report = {
        "model": spec.name,
        "n": n,
        "chains": chains,
        "tune": spec.prior_tune,
        "draws_per_chain": per_chain,
        "divergences": n_div,
        "worst_rhat": worst_rhat,
        "worst_ess_bulk": worst_ess,
        "ess_per_kept_draw": worst_ess / n,
        "diagnostics": diag,
        "deterministics": det_names,
        "free_rvs": raw_names,
        "potentials": [p.name for p in prior.potentials],
        "dropped_observed": [o.name for o in model.observed_RVs],
        "fingerprint": fp,
        "declared_prior_check": declared_prior_check(system, phys),
    }
    with open(out / "prior_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "model",
                    "n",
                    "divergences",
                    "worst_rhat",
                    "worst_ess_bulk",
                )
            },
            indent=1,
        )
    )
    shutil.rmtree(workdir, ignore_errors=True)


def declared_prior_check(system, phys):
    """KS of each sampled element's prior draws against its DECLARED prior.

    Informational: a parameter that also carries a component potential (Chen,
    the collision bound, a soft barrier through a derived quantity) should
    differ from its declared box, and the report names which do.  One that
    differs with no such potential is a prior the code applies but does not
    report.
    """
    from scipy import stats

    out = {}
    for p in sampled_parameters(system):
        tf = p._raw_transform
        vals = np.asarray(phys[p.label])
        for j, i in enumerate(tf["sampled_idx"]):
            x = vals[:, i]
            lo, hi = float(tf["lowers"][i]), float(tf["uppers"][i])
            if tf["use_logit"][i]:
                # A Gaussian on a logit element would be a potential; the
                # declared prior of a bare logit element is U(lo, hi).
                has_mu = p.mu is not None and np.any(
                    np.isfinite(np.atleast_1d(p.mu))
                )
                if has_mu:
                    mu = float(np.atleast_1d(p.mu)[min(i, np.size(p.mu) - 1)])
                    sg = float(
                        np.atleast_1d(p.sigma)[min(i, np.size(p.sigma) - 1)]
                    )
                    a, b = (lo - mu) / sg, (hi - mu) / sg
                    cdf = lambda v, a=a, b=b, mu=mu, sg=sg: (
                        stats.truncnorm.cdf(v, a, b, loc=mu, scale=sg)
                    )
                    kind = f"TN({mu:.6g}, {sg:.3g}; {lo:.6g}, {hi:.6g})"
                else:
                    cdf = lambda v, lo=lo, hi=hi: stats.uniform.cdf(
                        v, loc=lo, scale=hi - lo
                    )
                    kind = f"U({lo:.6g}, {hi:.6g})"
            else:
                mu = float(tf["gaussian_mus"][i])
                sg = float(tf["gaussian_scales"][i])
                if np.isfinite(lo) and np.isfinite(hi):
                    a, b = (lo - mu) / sg, (hi - mu) / sg
                    cdf = lambda v, a=a, b=b, mu=mu, sg=sg: (
                        stats.truncnorm.cdf(v, a, b, loc=mu, scale=sg)
                    )
                    kind = f"TN({mu:.6g}, {sg:.3g}; {lo:.6g}, {hi:.6g})"
                else:
                    cdf = lambda v, mu=mu, sg=sg: stats.norm.cdf(
                        v, loc=mu, scale=sg
                    )
                    kind = f"N({mu:.6g}, {sg:.3g})"
            ks = stats.kstest(x, cdf)
            out[f"{p.label}[{i}]"] = {
                "declared": kind,
                "ks_p": float(ks.pvalue),
                "n": int(x.size),
            }
    return out


# ---------------------------------------------------------------------------
# Stage 2: one replicate.
# ---------------------------------------------------------------------------
class Timeout(Exception):
    pass


def _on_term(signum, frame):  # noqa: ARG001
    raise Timeout(f"signal {signum}")


def run_replicate(spec, prior_file, prior_report, index, seed):
    import arviz as az
    import pymc as pm

    pz = np.load(prior_file)
    phys = {
        k.split("::", 1)[1]: pz[k][index]
        for k in pz.files
        if k.startswith("phys::")
    }
    rng = np.random.default_rng(seed)

    # 1. Build once on placeholder data to evaluate the independent simulator
    #    on Parameter metadata (units), then simulate and rebuild on the data.
    workdir = Path(tempfile.mkdtemp(prefix=f"sbc_{spec.name}_{index}_"))
    try:
        system0, _ = build(spec, workdir)
        mean, sigma, yobs = spec.simulate(phys, system0, rng)
        del system0
        system, model = build(spec, workdir, values=yobs)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # 2. Invariants: same prior parameterization as the prior build...
    fp = transform_fingerprint(system)
    if fp != prior_report["fingerprint"]:
        diff = sorted(
            k
            for k in set(fp) | set(prior_report["fingerprint"])
            if fp.get(k) != prior_report["fingerprint"].get(k)
        )
        raise RuntimeError(
            "DATA-DEPENDENT PRIOR: the replicate build's raw transform differs "
            f"from the prior build's for {diff}; SBC is invalid until the "
            "prior is fixed in the params file."
        )
    truth_raw = free_raw_point(model, system, phys)
    # ...and the same prior density at the truth.
    lp_prior = float(prior_logp_fn(model)(truth_raw))
    lp_ref = float(pz["prior_lp"][index])
    if not np.isclose(
        lp_prior, lp_ref, rtol=0, atol=1e-6 * max(1, abs(lp_ref))
    ):
        raise RuntimeError(
            f"DATA-DEPENDENT PRIOR: the replicate's prior logp at the truth is "
            f"{lp_prior!r}, the prior build's {lp_ref!r}."
        )

    # 3. The independent simulator must agree with exozippy's own mean and
    #    sigma at the truth (in the model's internal data units).
    mu_g, sg_g = observed_mu_sigma(model)
    f = compile_on_values(model, [mu_g, sg_g])
    mu_m, sg_m = (np.asarray(x) for x in f(*point_args(model, truth_raw)))
    data_int = observed_data(model)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = data_int / yobs
    factor = float(np.median(ratio[np.isfinite(ratio)]))
    mu_err = float(np.max(np.abs(mu_m / factor - mean)) / np.max(sigma))
    sg_err = float(np.max(np.abs(sg_m / factor - sigma) / sigma))
    # 1e-2 sigma: far below anything a rank histogram of a few hundred
    # replicates can see, far above solver round-off (the transit leg measured
    # 2.7e-4 sigma between the exact overlap area and exozippy's model).
    if mu_err > 1e-2 or sg_err > 1e-6:
        raise RuntimeError(
            "FORWARD MODELS DISAGREE at the truth: max |mu_exozippy - "
            f"mu_independent| = {mu_err:.3g} sigma, max sigma rel. diff = "
            f"{sg_err:.3g}.  Either the harness or the code has the physics "
            "wrong; the ranks would measure that, not calibration."
        )

    # 4. Fit from the truth.
    system.apply_polished_starts([truth_raw], [0])
    system.recenter_whitening_anchor(model)
    start = system.get_raw_start(model)
    t_fit = time.time()
    with model:
        idata = pm.sample(
            draws=spec.draws,
            tune=spec.tune,
            chains=spec.chains,
            cores=min(spec.chains, getattr(spec, "cores", spec.chains)),
            target_accept=spec.target_accept,
            initvals=start,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )
    t_fit = time.time() - t_fit
    post = idata.posterior
    det_names = [d.name for d in model.deterministics]
    q = tracked_quantities(post, det_names)
    truth_q = {}
    for name in det_names:
        v = np.asarray(phys[name])
        if v.ndim == 0:
            truth_q[name] = float(v)
        else:
            for i, x in enumerate(v.ravel()):
                truth_q[f"{name}[{i}]"] = float(x)
    varying = set(prior_report["diagnostics"])
    q = {k: v for k, v in q.items() if k in varying}

    ds = az.convert_to_dataset(q)
    rhat = az.rhat(ds)
    ess_b = az.ess(ds, method="bulk")
    ess_t = az.ess(ds, method="tail")
    n_div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    total = spec.chains * spec.draws
    pick = np.linspace(0, total - 1, L_DRAWS).round().astype(int)
    rows = {}
    for k, v in q.items():
        flat = v.reshape(-1)
        thin = flat[pick]
        t = truth_q[k]
        rows[k] = {
            "rank": int(np.sum(thin < t)),
            "quantile": float(np.mean(flat < t)),
            "rhat": float(rhat[k]),
            "ess_bulk": float(ess_b[k]),
            "ess_tail": float(ess_t[k]),
            "truth": t,
            "median": float(np.median(flat)),
            # The L thinned draws themselves, so a rank can be recomputed
            # against ANY truth (the smoke test's negative control ranks
            # replicate i's posterior against replicate j's truth).
            "thin": thin.tolist(),
        }
    return {
        "model": spec.name,
        "index": int(index),
        "seed": int(seed),
        "divergences": n_div,
        "fit_seconds": t_fit,
        "mu_check_sigma": mu_err,
        "sigma_check_rel": sg_err,
        "prior_lp": lp_prior,
        "L": L_DRAWS,
        "quantities": rows,
    }


def cmd_run(args, spec=None):
    spec = spec or SPECS[args.model]()
    out = Path(args.outdir) / spec.name
    rows = out / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    with open(out / "prior_report.json") as f:
        prior_report = json.load(f)
    previous = signal.signal(signal.SIGTERM, _on_term)
    try:
        _run_loop(args, spec, out, rows, prior_report)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _run_loop(args, spec, out, rows, prior_report):
    t0 = time.time()
    for index in range(args.start, args.start + args.count):
        if index >= prior_report["n"]:
            break
        dest = rows / f"rep_{index:04d}.json"
        if dest.exists():
            continue
        if args.maxtime and time.time() - t0 > args.maxtime:
            print(f"maxtime {args.maxtime}s reached before replicate {index}")
            break
        try:
            row = run_replicate(
                spec, out / "prior.npz", prior_report, index, args.seed + index
            )
        except Timeout as exc:
            print(f"replicate {index}: stopped by {exc}; no row written")
            break
        except Exception as exc:  # noqa: BLE001 -- recorded, never hidden
            row = {
                "model": spec.name,
                "index": int(index),
                "error": f"{type(exc).__name__}: {exc}",
            }
        tmp = dest.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(row, f, indent=1)
        os.replace(tmp, dest)
        print(
            f"replicate {index}: "
            + (
                row.get("error")
                or f"div={row['divergences']} fit={row['fit_seconds']:.0f}s"
            )
        )


# ---------------------------------------------------------------------------
# Stage 3: analysis.
# ---------------------------------------------------------------------------
from sbc_stats import (  # noqa: E402
    chi2_uniformity,
    ecdf_band_gamma,
    ecdf_diff,
    ks_uniformity,
    rank_hist_band,
)

CONVERGENCE = {"rhat": 1.01, "ess_bulk": 400.0, "ess_tail": 200.0}


def converged(row):
    if "error" in row:
        return False, "error"
    if row["divergences"] > 0:
        return False, "divergences"
    for q in row["quantities"].values():
        if q["rhat"] > CONVERGENCE["rhat"]:
            return False, "rhat"
        if q["ess_bulk"] < CONVERGENCE["ess_bulk"]:
            return False, "ess_bulk"
        if q["ess_tail"] < CONVERGENCE["ess_tail"]:
            return False, "ess_tail"
    return True, "ok"


def analyze_ranks(ranks_by_q, L, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    out = {}
    gammas = {}
    for name, r in ranks_by_q.items():
        r = np.asarray(r, dtype=int)
        n = r.size
        if n not in gammas:
            gammas[n] = ecdf_band_gamma(n, L, alpha=alpha, rng=rng)
        g, p_ecdf, inside = ecdf_diff(r, L, gammas[n])
        out[name] = {
            "n": int(n),
            "chi2_p": chi2_uniformity(r, L, N_BINS),
            "ks_p": ks_uniformity(r, L, rng),
            "ecdf_p": p_ecdf,
            "inside_band": bool(inside),
            "mean_rank_frac": float(np.mean(r) / L),
            "shape": rank_shape(r, L),
        }
    return out


def rank_shape(r, L):
    """A plain-words description of a rank histogram's departure.

    The fraction in the outer fifth of the ranks (expected 0.4) separates U
    (too narrow: excess) from hump (too wide: deficit); the mean fractional
    rank (expected 0.5) separates a slope (bias).  A mean below 0.5 means the
    truth sits LOW in the posterior, i.e. the posterior is biased HIGH.
    """
    u = (np.asarray(r) + 0.5) / (L + 1)
    tails = float(np.mean((u < 0.2) | (u > 0.8)))
    return {"tail_frac": tails, "mean_u": float(np.mean(u))}


def cmd_analyze(args):
    spec = SPECS[args.model]()
    out = Path(args.outdir) / spec.name
    rows = []
    for fp in sorted((out / "rows").glob("rep_*.json")):
        with open(fp) as f:
            rows.append(json.load(f))
    with open(out / "prior_report.json") as f:
        prior_report = json.load(f)
    status = {}
    good = []
    for row in rows:
        ok, why = converged(row)
        status[why] = status.get(why, 0) + 1
        if ok:
            good.append(row)
    errors = [r for r in rows if "error" in r]
    usable = [r for r in rows if "error" not in r]
    names = sorted(
        set().union(*(r["quantities"].keys() for r in usable))
        if usable
        else []
    )
    sampled = {p for p in prior_report["fingerprint"]}

    # A quantity that is constant up to round-off (e.g. planet.mass when the
    # mass coordinate and the star are pinned) has no rank: every draw TIES
    # the truth.  Measured on transit_pinq: planet.mass varied at 1e-19 in
    # the prior draws, passed a std > 0 test, and ranked 0 in every
    # replicate.  Drop such quantities, and break genuine ties at random.
    def spread(k):
        vals = [r["quantities"][k]["truth"] for r in usable if k in r["quantities"]]
        scale = max(np.max(np.abs(vals)), 1e-300)
        return np.ptp(vals) / scale

    names = [k for k in names if spread(k) > 1e-9]
    tie_rng = np.random.default_rng(7164)

    def rank_of(q):
        if "thin" not in q:
            return q["rank"]
        thin = np.asarray(q["thin"])
        below = int(np.sum(thin < q["truth"]))
        ties = int(np.sum(thin == q["truth"]))
        return below + int(tie_rng.integers(0, ties + 1))

    def ranks(sel):
        return {
            k: [rank_of(r["quantities"][k]) for r in sel if k in r["quantities"]]
            for k in names
        }

    res_conv = analyze_ranks(ranks(good), L_DRAWS)
    res_all = analyze_ranks(ranks(usable), L_DRAWS)
    n_tests = len(names)
    summary = {
        "model": spec.name,
        "n_prior_draws": prior_report["n"],
        "n_rows": len(rows),
        "n_errors": len(errors),
        "errors": sorted({r["error"] for r in errors}),
        "status_counts": status,
        "n_converged": len(good),
        "convergence_rule": CONVERGENCE | {"divergences": 0},
        "L": L_DRAWS,
        "n_bins": N_BINS,
        "bonferroni_alpha": 0.05 / max(n_tests, 1),
        "sampled_parameters": sorted(sampled),
        "converged": res_conv,
        "all_rows": res_all,
        "prior_report": {
            k: prior_report[k]
            for k in (
                "divergences",
                "worst_rhat",
                "worst_ess_bulk",
                "ess_per_kept_draw",
                "declared_prior_check",
            )
        },
        "fit_seconds_median": float(
            np.median([r["fit_seconds"] for r in usable])
        )
        if usable
        else None,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    plot_ranks(out, spec.name, ranks(good), res_conv, len(good), sampled)
    print_summary(summary)


def is_sampled_name(name, sampled):
    return name.split("[")[0] in sampled


def plot_ranks(out, model_name, ranks_by_q, res, n_good, sampled):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(
        ranks_by_q, key=lambda k: (not is_sampled_name(k, sampled), k)
    )
    if not names or n_good == 0:
        return
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    for kind in ("hist", "ecdf"):
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(4 * ncol, 2.6 * nrow), squeeze=False
        )
        for ax in axes.ravel()[len(names) :]:
            ax.axis("off")
        for ax, name in zip(axes.ravel(), names):
            r = np.asarray(ranks_by_q[name])
            st = res[name]
            if kind == "hist":
                edges = np.linspace(-0.5, L_DRAWS + 0.5, N_BINS + 1)
                lo, hi = rank_hist_band(r.size, N_BINS)
                ax.axhspan(lo, hi, color="0.85", lw=0)
                ax.axhline(r.size / N_BINS, color="0.5", lw=0.8)
                ax.hist(r, bins=edges, color="#3b6ea8", histtype="stepfilled")
                ax.set_xlim(-0.5, L_DRAWS + 0.5)
            else:
                z, d, band_lo, band_hi = ecdf_diff_curve(r, L_DRAWS, st)
                ax.fill_between(
                    z, band_lo, band_hi, color="0.85", lw=0, step="post"
                )
                ax.step(z, d, where="post", color="#3b6ea8")
                ax.axhline(0.0, color="0.5", lw=0.8)
            flag = "" if st["inside_band"] else "  OUTSIDE"
            star = "" if is_sampled_name(name, sampled) else " (derived)"
            ax.set_title(
                f"{name}{star}\nchi2 p={st['chi2_p']:.2g} KS p={st['ks_p']:.2g}"
                f"{flag}",
                fontsize=8,
            )
            ax.tick_params(labelsize=7)
        fig.suptitle(
            f"SBC {model_name}: N={n_good} converged replicates, L={L_DRAWS}"
            + (
                " -- rank histograms (grey: 99% per-bin band)"
                if kind == "hist"
                else " -- ECDF - uniform (grey: 95% simultaneous band)"
            ),
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"sbc_{model_name}_{kind}.png", dpi=110)
        plt.close(fig)


def ecdf_diff_curve(r, L, st):
    from sbc_stats import ecdf_band_gamma_cached
    from scipy import stats

    n = r.size
    g = ecdf_band_gamma_cached(n, L)
    z = np.arange(L + 1) / (L + 1)
    p = (np.arange(L + 1) + 1) / (L + 1)
    F = np.array([np.mean(r <= k) for k in range(L + 1)])
    lo = stats.binom.ppf(g / 2, n, p) / n - p
    hi = stats.binom.ppf(1 - g / 2, n, p) / n - p
    return z, F - p, lo, hi


def print_summary(s):
    print(
        f"{s['model']}: rows {s['n_rows']}, errors {s['n_errors']}, converged "
        f"{s['n_converged']} ({s['status_counts']})"
    )
    a = s["bonferroni_alpha"]
    for name, st in sorted(s["converged"].items()):
        # Flag on the Bonferroni level over every tracked quantity (they are
        # strongly correlated, so this is conservative).  The per-quantity
        # 95% band is drawn in the figures; one quantity in twenty is
        # expected to leave it by chance.
        flag = ""
        if min(st["chi2_p"], st["ks_p"], st["ecdf_p"]) < a:
            flag = "  <-- FLAG"
        st_all = s["all_rows"].get(name)
        if st_all is not None:
            flag += "  (all rows: min p=%.3f)" % min(
                st_all["chi2_p"], st_all["ks_p"], st_all["ecdf_p"]
            )
        print(
            f"  {name:40s} chi2 p={st['chi2_p']:.3f} KS p={st['ks_p']:.3f} "
            f"ECDF p={st['ecdf_p']:.3f} tails={st['shape']['tail_frac']:.2f} "
            f"mean_u={st['shape']['mean_u']:.3f}{flag}"
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prior", "run", "analyze"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", required=True, choices=sorted(SPECS))
        sp.add_argument("--outdir", required=True)
        sp.add_argument("--seed", type=int, default=7164)
        sp.add_argument("--cores", type=int, default=4)
        if name == "prior":
            sp.add_argument("--n", type=int, default=300)
            sp.add_argument("--raw-per-draw", type=int, default=40)
        if name == "run":
            sp.add_argument("--start", type=int, required=True)
            sp.add_argument("--count", type=int, default=1)
            sp.add_argument(
                "--maxtime",
                type=float,
                default=0.0,
                help="seconds; start no new replicate after this",
            )
    args = ap.parse_args()
    {"prior": cmd_prior, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()

"""Distribution-level check of the galactic prior against genulens.

genulens (Koshimoto, Baba & Bennett 2021) is the public simulator of the
Galactic model galacticmodel simplifies.  tests/fixtures/genulens.json
(scripts/make_genulens_fixture.py; genulens 2.0.0a4) stores, on two sight
lines, genulens's source distances and IMF as equal-weight draws, and the
rate-weighted quantiles of the lens distance, the heliocentric mu_rel and
t_E.  The test feeds the SAME sources and IMF to EXOZIPPy's prior -- lens
position n(D) D^2 and velocity mixture, drawn exactly as the kinematic
prior defines them, through the model's own frame and velocity map -- and
applies the event-rate weight mu_rel * theta_E that both codes use
(genulens's wtj; mulensevent.event_rate_prior here) by sampling-importance-
resampling.  The statistic is the KS distance evaluated at genulens's 19
stored quantiles.

This is a FIDELITY GUARD, not an exactness test (test_galactic_frame.py is
that): the model is a simplification of genulens and does not match it.
Measured at the recording (N = 20000 here, 40000 genulens events; the test
uses fewer draws and a fixed seed, so it is deterministic):

    sight line   quantity  KS     median EXOZIPPy / genulens
    ob140939     D_L       0.086  6.59 / 6.41 kpc
                 mu_rel    0.032  5.65 / 5.86 mas/yr
                 t_E       0.049  18.0 / 19.9 d
    Baade        D_L       0.045  7.22 / 7.17 kpc
                 mu_rel    0.116  5.40 / 6.18 mas/yr
                 t_E       0.026  16.3 / 15.4 d

(20000 draws; this test's own 8000 give 0.084/0.033/0.052 and
0.035/0.116/0.027.)  Review 1.7.3 replaced the plain-ellipsoid bar with
K21's E+E_X bar (galacticmodel.md, "The bar density"): the Baade D_L KS fell
from 0.207 to 0.035-0.045 and t_E from 0.086 to 0.027, while the Baade
mu_rel KS ROSE from 0.065 to 0.116 -- with the lenses and sources now where
genulens puts them, the bar's KINEMATICS (ours: constant dispersions and
solid-body rotation; K21: position-dependent dispersions and streaming
along the bar) is the remaining gap, the old density's spread having partly
hidden it.  Open (review 1.7.3).  The bounds sit ~1.4-2x above the values
this test's own draws give.  Measured sensitivity: making every branch
counter-rotate gives mu_rel KS 0.41/0.38, dropping rotation altogether
0.106 on ob140939 -- both fail.
"""

import json
import os

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import Galactic, SkyCoord

from exozippy.components.galacticmodel import galacticmodel as gm
from exozippy.components.galacticmodel import physics as ph
from exozippy.constants import K_VEL_CONVERSION

KAPPA = 8.144  # mas / Msun
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "genulens.json")

#: KS bounds per sight line and quantity (measured values: module docstring).
KS_BOUND = {
    "ob140939": {"D_L": 0.13, "mu_rel": 0.06, "t_E": 0.09},
    "baade": {"D_L": 0.08, "mu_rel": 0.16, "t_E": 0.06},
}


def _prior_events(l_deg, b_deg, ds, masses, rng, k=24):
    """D_L, mu_rel, t_E of events drawn from EXOZIPPy's prior (see the
    module docstring), one per source distance in ``ds``."""
    sc = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame=Galactic()).icrs
    m_rot, v0, *los = ph.line_of_sight_basis(sc.ra.rad, sc.dec.rad)

    def geometry(d):
        x, y, z = ph.galactic_xyz(d, *los)
        return x, y, z, np.sqrt(x**2 + y**2 + 1e-6)

    def branch(d):
        dens = np.stack(gm.branch_log_number_densities(*geometry(d), xp=np))
        p = np.exp(dens - dens.max(axis=0))
        c = np.cumsum(p / p.sum(axis=0), axis=0)
        return (rng.random(d.size)[None, :] > c).sum(axis=0).clip(0, 2)

    def proper_motion(d):
        x, y, _z, r = geometry(d)
        comp = branch(d)
        pm = np.empty((d.size, 2))
        for bi, name in enumerate(gm.BRANCHES):
            sel = comp == bi
            n = int(sel.sum())
            if not n:
                continue
            s = gm.BRANCH_SIGMAS[name]
            v_r = rng.normal(0.0, s[0], n)
            v_phi = ph.mean_polar_velocity(r[sel], name)[1] + rng.normal(
                0.0, s[1], n
            )
            v_z = rng.normal(0.0, s[2], n)
            c, sn = x[sel] / r[sel], y[sel] / r[sel]
            v = (
                np.stack([v_r * c - v_phi * sn, v_r * sn + v_phi * c, v_z])
                - np.asarray(v0)[:, None]
            )
            icrs = np.linalg.solve(m_rot, v)
            pm[sel] = (icrs[:2] / (K_VEL_CONVERSION * d[sel])).T
        return pm

    grid = np.linspace(1e-3, 20.0, 20001)
    dens = gm.branch_log_number_densities(*geometry(grid), xp=np)
    ln_n = np.logaddexp.reduce(np.stack(dens), axis=0)
    pdf = np.exp(ln_n - ln_n.max()) * grid**2
    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))]
    )

    n = ds.size
    src_pm = proper_motion(ds)
    cdf_at = np.interp(ds, grid, cdf)
    dl = np.empty((k, n))
    mu = np.empty((k, n))
    th = np.empty((k, n))
    for j in range(k):
        dl[j] = np.interp(rng.random(n) * cdf_at, cdf, grid)
        mu[j] = np.hypot(*(proper_motion(dl[j]) - src_pm).T)
        pirel = np.clip(1.0 / dl[j] - 1.0 / ds, 0.0, None)
        th[j] = np.sqrt(KAPPA * rng.choice(masses, n) * pirel)
    w = mu * th  # the event rate, mu_rel * theta_E
    keep = w.sum(axis=0) > 0
    cum = np.cumsum(w[:, keep], axis=0) / w[:, keep].sum(axis=0)
    pick = (rng.random(int(keep.sum()))[None, :] > cum).sum(axis=0)
    cols = np.arange(int(keep.sum()))
    dl, mu, th = (a[:, keep][pick, cols] for a in (dl, mu, th))
    return {"D_L": dl, "mu_rel": mu, "t_E": th / mu * 365.25}


@pytest.mark.parametrize("sight_line", ("ob140939", "baade"))
def test_prior_tracks_genulens_event_distributions(sight_line):
    """
    Given genulens's source distances and IMF on this sight line,
    When EXOZIPPy's galactic prior generates rate-weighted events,
    Then the lens distance, heliocentric mu_rel and t_E distributions are
      within KS_BOUND of genulens's (KS at genulens's stored quantiles).
    """
    with open(FIXTURE) as fh:
        fx = json.load(fh)
    probs = np.asarray(fx["probabilities"])
    sl = fx["sight_lines"][sight_line]
    rng = np.random.default_rng(1171)
    ds = rng.choice(np.asarray(sl["source_distance_kpc"]), 8000)
    events = _prior_events(
        sl["l_deg"], sl["b_deg"], ds, np.asarray(sl["lens_mass_msun"]), rng
    )
    report = {}
    for name, q_genulens in sl["quantiles"].items():
        x = np.sort(events[name])
        ecdf = np.searchsorted(x, q_genulens, side="right") / x.size
        report[name] = float(np.max(np.abs(ecdf - probs)))
    for name, ks in report.items():
        assert ks < KS_BOUND[sight_line][name], (sight_line, report)

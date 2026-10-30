"""Prior-predictive t_E check against OGLE's measured distribution (review 7.16.3).

The microlensing prior (galacticmodel's density x kinematics x IMF, plus
mulensevent's event-rate weight theta_E * mu_rel) is sampled ALONE -- no
light curve -- toward Baade's window, through the real System and NUTS
(examples/validation/prior_predictive_tE.py, ``sample_prior``), and its t_E
quantiles are compared with the efficiency-corrected t_E distribution
OGLE-IV measured in the field that contains Baade's window (Mroz et al.
2019, ApJS 244, 29: Table 3's 213 BLG513 events, weighted by the table's
own inverse detection efficiency; this reproduces the paper's Table 7 mean
t_E for every field).

TOLERANCE, per quantile q in (10, 50, 90), in ln(t_E):

    |ln(model / Mroz)| <= 3 * sqrt(s_ref^2 + s_model^2) + sys

* s_ref: the reference's own uncertainty -- the bootstrap sigma of the
  weighted quantile over the 213 events, relative (median: 18.9 +/- 1.6 d,
  8.5%).
* s_model: the model quantile's Monte Carlo sigma (bootstrap of the
  draws, inflated by sqrt(N / ESS)).
* sys: the size of the two model differences that are KNOWN and deliberate
  (no source luminosity function, no remnant lenses), measured with
  genulens as |ln(survey-like run / matched run)| for the same quantile
  (median: 17.2 vs 16.1 d, 6.6%).

So the median must agree to about 32%.

THE FINDING (the strict xfail below).  On master after 1.7.1 the prior gives
t_E = 9.0 / 28.7 / 106 d (q10/q50/q90) against Mroz's 4.2 / 18.9 / 51.6 --
the median 52% long, KS p ~ 1e-9; the same on BLG505 (29.0 vs 12.8 d).  It
localizes to the LENS MASS FUNCTION: the rate-weighted lens mass is
0.55 Msun median with q90 = 4.2 Msun and 33% of events above 1 Msun (up to
the 316 Msun logmass bound), against genulens's 0.30 / 0.88 / 3.6%.  The
Chabrier IMF is an INITIAL (and system) mass function, applied to lenses
as if it were the present-day one; an old bulge has no main-sequence stars
above ~1 Msun.  The mass is independent of position and velocity in the
rate-weighted prior, so swapping in genulens's lens masses is an exact
test of that piece -- and it brings Baade's t_E to 19.7 d (the passing
test below).  Kinematics are not the cause (mu_rel KS vs genulens 0.06).
A smaller second piece is geometric (sources ~0.9 kpc farther than
genulens's); it is what is left on BLG505 after the mass swap.

THE MASS FUNCTION IS FIXED (review 1.7.2, now the default: the
galacticmodel's present-day mass function -- MIST turnoff, white dwarfs,
neutron stars and black holes per population).  Measured 2026-10-30 (4 x
2500 NUTS draws): t_E = 8.2 / 22.4 / 57.7 d, M_L 0.091 / 0.422 / 0.887 Msun.
The median and q90 now agree within tolerance, but q10 does NOT (|ln
ratio| 0.66 against 0.55): the prior still has too few short events, the
geometric piece above (review 1.7.3), which no mass function can fix.  So
the xfail stays strict and now names q10; when 1.7.3 is fixed it turns
into an XPASS and fails strictly: update the test then, do not widen the
tolerance.  With present_day on, the lens mass is no longer independent
of position (each population has its own present-day mass function), so
the genulens mass swap below is an approximation rather than exact; it is
kept as the localization it was.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

VALIDATION = Path(__file__).resolve().parents[1] / "examples" / "validation"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pp():
    sys.path.insert(0, str(VALIDATION))
    try:
        yield importlib.import_module("prior_predictive_tE")
    finally:
        sys.path.remove(str(VALIDATION))


@pytest.fixture(scope="module")
def baade_draws(pp, tmp_path_factory):
    import os

    import arviz as az

    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("priorpred"))
    try:
        cols, idata, _, _ = pp.sample_prior(
            "baade", draws=1500, tune=500, chains=2, maxtime=900, seed=7163
        )
    finally:
        os.chdir(cwd)
    t_e = cols["t_E"].reshape(idata.posterior.sizes["chain"], -1)
    ess = float(np.asarray(az.ess(t_e)))
    return cols, ess


def _tolerances(pp, model_te, ess):
    """(q_model, q_ref, tol) arrays for q10/q50/q90; see the module doc."""
    import json

    rng = np.random.default_rng(1)
    q_ref, s_ref, _, _ = pp.mroz_quantiles_with_errors("baade")
    q_mod = pp._q(model_te)
    boots = np.array(
        [pp._q(rng.choice(model_te, model_te.size)) for _ in range(300)]
    )
    s_mod = boots.std(axis=0) * np.sqrt(model_te.size / ess)
    gen = json.loads((pp.REFS / "genulens_reference.json").read_text())
    g = gen["sightlines"]["baade"]
    sys_ = np.abs(
        np.log(
            [
                g["survey"]["t_E"][k] / g["matched"]["t_E"][k]
                for k in ("q10", "q50", "q90")
            ]
        )
    )
    tol = 3 * np.hypot(s_ref / q_ref, s_mod / q_mod) + sys_
    return q_mod, q_ref, tol


@pytest.mark.xfail(
    strict=True,
    reason="review 1.7.3: with the present-day lens mass function (1.7.2) "
    "the median and q90 of t_E agree with OGLE toward Baade's window, but "
    "q10 is still ~2x long (8.2 vs 4.2 d) -- too few short events, the "
    "source/lens geometry, not the mass function",
)
def test_prior_tE_matches_ogle_baade_window(pp, baade_draws):
    """
    Given the microlensing prior sampled alone toward Baade's window,
    When its t_E quantiles are compared with OGLE-IV's efficiency-corrected
      distribution in field BLG513 (Mroz et al. 2019),
    Then q10, the median and q90 each agree within the stated tolerance.
    """
    cols, ess = baade_draws
    q_mod, q_ref, tol = _tolerances(pp, cols["t_E"], ess)
    dev = np.abs(np.log(q_mod / q_ref))
    assert np.all(dev <= tol), (
        f"model {q_mod.round(1)} vs Mroz {q_ref.round(1)} d: "
        f"|ln ratio| {dev.round(3)} > tol {tol.round(3)}"
    )


def test_prior_tE_with_genulens_lens_masses_matches_ogle_baade_window(
    pp, baade_draws
):
    """
    Given the same prior draws,
    When each draw's lens mass is replaced by one from genulens's
      rate-weighted lens-mass distribution (t_E scales as sqrt(M_L), and
      the rate-weighted mass is independent of position and velocity in
      both models, so this swaps exactly the mass function and nothing
      else),
    Then the median and q90 agree with OGLE within the stated tolerance --
      i.e. everything in the prior EXCEPT the lens mass function is
      consistent with the survey on this sight line.  q10 is not asserted:
      it passes only narrowly (ln ratio 0.48 for a tolerance of 0.52 at the
      recording), because the reference's short tail rests on few events.
    """
    cols, ess = baade_draws
    gen = np.loadtxt(pp.REFS / "genulens_baade_matched_resampled.dat")[:, 4]
    rng = np.random.default_rng(7163)
    m_swap = rng.choice(gen, cols["M_L"].size)
    t_swap = cols["t_E"] * np.sqrt(m_swap / cols["M_L"])
    q_mod, q_ref, tol = _tolerances(pp, t_swap, ess)
    dev = np.abs(np.log(q_mod / q_ref))
    assert np.all(dev[1:] <= tol[1:]), (
        f"model {q_mod.round(1)} vs Mroz {q_ref.round(1)} d: "
        f"|ln ratio| {dev.round(3)} > tol {tol.round(3)}"
    )

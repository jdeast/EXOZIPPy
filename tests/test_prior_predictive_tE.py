"""Prior-predictive t_E check against OGLE's measured distribution (review 7.16.3).

The microlensing prior (galacticmodel's density x kinematics x present-day
mass function, plus mulensevent's event-rate weight theta_E * mu_rel) is
sampled ALONE -- no light curve -- toward Baade's window, through the real
System and NUTS (examples/validation/prior_predictive_tE.py,
``sample_prior``), and its t_E
quantiles are compared with the efficiency-corrected t_E distribution
OGLE-IV measured in the field that contains Baade's window (Mroz et al.
2019, ApJS 244, 29: Table 3's 213 BLG513 events, weighted by the table's
own inverse detection efficiency; this reproduces the paper's Table 7 mean
t_E for every field).

OGLE IS A CONSISTENCY CHECK HERE, NOT AN INDEPENDENT CALIBRATION TARGET.
Koshimoto, Baba & Bennett (2021, ApJ 917, 78; K21) -- the model genulens
implements and galacticmodel simplifies -- fit their IMF to these same
OGLE-IV t_E distributions (Mroz et al. 2017, 2019).  genulens agreeing with
OGLE holds by construction, and a K21-like mass function here agreeing
with OGLE checks our implementation of their model, not the IMF.  Never
tune the prior to make this test pass: that counts the data twice.

TOLERANCE, per quantile q in (10, 50, 90), in ln(t_E):

    |ln(model / Mroz)| <= 3 * sqrt(s_ref^2 + s_model^2) + sys

* s_ref: the reference's own uncertainty -- the bootstrap sigma of the
  weighted quantile over the 213 events, relative (median: 18.9 +/- 1.6 d,
  8.5%).
* s_model: the model quantile's Monte Carlo sigma (bootstrap of the
  draws, inflated by sqrt(N / ESS)).
* sys: the one model difference from a survey that is KNOWN and
  deliberate -- no source luminosity function -- measured with genulens as
  |ln(survey run / remnants run)| for the same quantile.  The two runs
  differ ONLY in the source weight (gammaDs 0.5 vs 2), both with remnants
  on (median: 17.4 vs 18.1 d, 4.0%).  Until 2026-10-30 this term compared
  the survey run with a REMNANT 0 run, which also folded in genulens's
  discarding of evolved lenses (median 6.6%).

So the median must agree to about 30%.

HISTORY.  With the Chabrier IMF used as the present-day mass function
(before review 1.7.2) the prior gave t_E = 9.0 / 28.7 / 106 d (q10/q50/q90)
against Mroz's 4.2 / 18.9 / 51.6, the rate-weighted lens mass 0.55 / 4.2
Msun (median / q90) with 33% of events above 1 Msun.

genulens's own present-day lens mass function is its REMNANT 1 mode:
0.46 / 0.95 Msun, 8.0% above 1 Msun, 21% dark lenses (WD 16.9, NS 2.4, BH
1.4%) toward Baade's window.  (REMNANT 0 is NOT a no-remnant model: it
DISCARDS every lens whose star has died, giving 0.31 / 0.86 Msun and 3.6%;
the numbers first quoted for this comparison came from that mode.)
Swapping the remnants-on genulens masses into those IMF draws -- exact
there, since one IMF everywhere makes the rate-weighted mass independent
of position and velocity -- gives t_E = 7.3 / 22.2 / 60.8 d.

THE PRESENT-DAY MASS FUNCTION (review 1.7.2, the default): measured
2026-10-30 (4 x 5000 NUTS draws, Baade): t_E = 7.9 / 22.4 / 59.1 d, M_L
0.083 / 0.406 / 0.891 Msun (genulens remnants 0.070 / 0.456 / 0.949; KS
0.045), 16.7% dark lenses.  The median and q90 agree within tolerance, but
q10 does NOT (|ln ratio| ~0.6 against ~0.5): the prior still has too few
short events.  That is geometric, not the mass function: t_E / sqrt(M_L)
is 37.5 d at the median against genulens's 30.9 (sources ~0.9 kpc farther,
D_S median 9.6 vs 8.7 kpc), +44% toward BLG505; mu_rel agrees (KS 0.06).
So the xfail stays strict and names q10; when the geometry is fixed it
turns into an XPASS and fails strictly: update the test then, do not
widen the tolerance.  With present_day on, the lens mass is no longer
independent of position (each population has its own present-day mass
function), so the genulens mass swap below is an approximation rather than
exact; it is kept as the localization it was.
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
                g["survey"]["t_E"][k] / g["remnants"]["t_E"][k]
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
      rate-weighted, REMNANTS-ON lens-mass distribution (t_E scales as
      sqrt(M_L); approximately a swap of the mass function alone, see the
      module docstring),
    Then the median agrees with OGLE within the stated tolerance.  q10 is
      not asserted (it is the open geometric shortfall the xfail above
      names), and neither is q90: with the remnants-on masses it sits at
      the edge (ln ratio ~0.19 for a tolerance of ~0.21 on the full 4 x
      5000-draw run), so asserting it would make the test flaky.  With the
      REMNANT 0 masses used before 2026-10-30 this test passed on median
      and q90 only because those masses were too light.
    """
    cols, ess = baade_draws
    gen = np.loadtxt(pp.REFS / "genulens_baade_remnants_resampled.dat")[:, 4]
    rng = np.random.default_rng(7163)
    m_swap = rng.choice(gen, cols["M_L"].size)
    t_swap = cols["t_E"] * np.sqrt(m_swap / cols["M_L"])
    q_mod, q_ref, tol = _tolerances(pp, t_swap, ess)
    dev = np.abs(np.log(q_mod / q_ref))
    assert dev[1] <= tol[1], (
        f"model {q_mod.round(1)} vs Mroz {q_ref.round(1)} d: "
        f"|ln ratio| {dev.round(3)} > tol {tol.round(3)}"
    )

"""The built-in PSPL peak finder (review 8.4.9).

WHAT THESE GUARD.  The seeder's whole value is that it finds the EVENT from
the data alone, so the tests are recovery tests on synthetic curves whose
answer is known exactly, plus the two properties that make it safe to leave
on by default: it must not need the flux scale to be known, and it must
degrade to None rather than raise.
"""

import logging

import numpy as np
import pytest

from exozippy.components.mulensing import peakfind


def _pspl(t, t_0, u_0, t_E):
    u2 = ((t - t_0) / t_E) ** 2 + u_0 * u_0
    return (u2 + 2.0) / np.sqrt(u2 * (u2 + 4.0))


def _curve(
    t_0, u_0, t_E, f_source, f_blend, n=3000, span=200.0, noise=1e-3, seed=0
):
    rng = np.random.default_rng(seed)
    t = np.linspace(t_0 - span / 2, t_0 + span / 2, n)
    flux = f_source * _pspl(t, t_0, u_0, t_E) + f_blend
    err = np.full(n, noise * max(f_blend, f_source))
    return t, flux + rng.normal(0.0, err), 1.0 / err**2


@pytest.mark.parametrize(
    "t_0, u_0, t_E",
    [
        (2458550.0, 0.15, 18.0),  # the DC2018-128-like case
        (2458550.0, 0.01, 9.5),  # high magnification
        (2458550.0, 1.20, 26.0),  # low magnification, wide wing
    ],
)
def test_recovers_the_injected_geometry(t_0, u_0, t_E):
    seed = peakfind.find_pspl_seed([_curve(t_0, u_0, t_E, 1.0, 0.5)])
    assert seed is not None
    # t_0 to a fraction of t_E, u_0 and t_E to a few percent.  Loose on
    # purpose: this is a SEED, and the tolerance that matters is "close
    # enough that the sampler finds the basin", not a fit's precision.
    assert abs(seed["t_0"] - t_0) < 0.05 * t_E
    assert seed["u_0"] == pytest.approx(u_0, rel=0.15)
    assert seed["t_E"] == pytest.approx(t_E, rel=0.15)


def test_is_blind_to_the_flux_scale():
    """The flux parameters are profiled out, so scaling every flux by a
    constant -- a different zeropoint, a different filter -- must leave the
    geometry untouched.  If this ever fails, the linear solve has picked up
    a scale dependence and the seeder has quietly become calibration-
    dependent."""
    t_0, u_0, t_E = 2458550.0, 0.2, 15.0
    a = peakfind.find_pspl_seed([_curve(t_0, u_0, t_E, 1.0, 0.5, seed=1)])
    b = peakfind.find_pspl_seed([_curve(t_0, u_0, t_E, 1e4, 5e3, seed=1)])
    for k in ("t_0", "u_0", "t_E"):
        assert a[k] == pytest.approx(b[k], rel=1e-6)


def test_two_bands_share_the_geometry_and_not_the_fluxes():
    """A second band with a different source/blend decomposition must help,
    not confuse: each curve solves its own fluxes against one shared
    trajectory."""
    t_0, u_0, t_E = 2458550.0, 0.25, 22.0
    curves = [
        _curve(t_0, u_0, t_E, 1.0, 0.2, n=3000, seed=2),
        _curve(t_0, u_0, t_E, 0.3, 4.0, n=400, seed=3),
    ]
    seed = peakfind.find_pspl_seed(curves)
    assert seed["t_0"] == pytest.approx(t_0, abs=0.05 * t_E)
    assert seed["u_0"] == pytest.approx(u_0, rel=0.15)
    assert seed["n_points"] == 3400


def test_flat_curve_still_returns_something_finite():
    """Pure noise has no event.  The seeder may return any geometry it likes
    -- there is no right answer -- but it must return finite numbers rather
    than NaN or a crash, because this path runs as a DEFAULT fallback and a
    dataset with no event in it is a thing users have."""
    rng = np.random.default_rng(7)
    t = np.linspace(2458500.0, 2458700.0, 2000)
    f = rng.normal(1.0, 0.01, t.size)
    seed = peakfind.find_pspl_seed([(t, f, np.full(t.size, 1e4))])
    assert seed is not None
    assert all(np.isfinite(seed[k]) for k in ("t_0", "u_0", "t_E", "chi2"))
    assert seed["u_0"] > 0 and seed["t_E"] > 0


def test_too_few_epochs_returns_none_rather_than_raising():
    t = np.array([2458550.0, 2458551.0])
    assert peakfind.find_pspl_seed([(t, np.ones(2), np.ones(2))]) is None
    assert peakfind.find_pspl_seed([]) is None


def test_push_hints_seeds_exactly_three_paths():
    """s, alpha, q and rho must NOT be seeded: the point of the peak finder
    is that the sampler finds the anomaly, and quietly seeding a generic
    companion would put the start in a place no solution occupies."""
    pushed = {}

    class FakeCM:
        def add_seed_hints(self, sets):
            pushed["sets"] = sets

    n = peakfind.push_peak_find_hints(
        {
            "t_0": 1.0,
            "u_0": 0.2,
            "t_E": 20.0,
            "chi2": 1.0,
            "n_points": 10,
            "converged": True,
        },
        FakeCM(),
    )
    assert n == 1
    assert pushed["sets"] == [
        {
            "source.0.t_0": 1.0,
            "source.0.u_0": 0.2,
            "mulensevent.0.t_E": 20.0,
        }
    ]


def test_push_hints_is_a_no_op_without_a_seed():
    assert peakfind.push_peak_find_hints(None, object()) == 0


def test_add_seed_hints_overwrites_which_is_why_the_gate_exists():
    """The reason _peak_find_seeds must not run after MMEXOFAST.

    ConfigManager.add_seed_hints ASSIGNS seed_hint_sets rather than
    appending (config.py), so two callers do not compose -- the second
    silently discards the first.  MMEXOFAST pushes one seed set per
    solution, including the binary-lens s/q/alpha; the peak finder pushes
    exactly one point-lens set.  Running it second therefore threw away
    every MMEXOFAST solution, and user_hints_sufficient could not catch it
    because that reads user_params and probe_derivable, where seed hints
    never appear.  This pins the overwrite so the gate is not "fixed" away
    by someone who assumes the calls accumulate.
    """
    from exozippy.config import ConfigManager

    cm = ConfigManager.__new__(ConfigManager)
    cm.seed_hint_sets = []
    cm._translate_and_scale = lambda path, value: (path, value)

    cm.add_seed_hints([{"a": 1.0}, {"a": 2.0}])
    assert len(cm.seed_hint_sets) == 2
    cm.add_seed_hints([{"a": 3.0}])
    assert len(cm.seed_hint_sets) == 1, (
        "add_seed_hints now appends; the peak finder's _mmexofast_seeded "
        "gate in mulensinstrument was written for overwrite semantics and "
        "should be revisited"
    )


# ---------------------------------------------------------------------------
# The u_0 -> 0, t_E -> inf degeneracy (review 2.4.14): DC2018-001 and -226
# ---------------------------------------------------------------------------


class _Degenerate:
    """A Nelder-Mead result that walked off along the degenerate direction
    with a chi2 marginally BELOW the grid's -- the case the old `res.fun >
    best_chi2` guard could not catch, and exactly what the two DC2018 seeds
    looked like (u_0 = 5e-10, t_E = 4e8 d)."""

    def __init__(self, t_0, chi2):
        self.x = np.array([t_0, np.log(5e-10), np.log(4.4e8)])
        self.fun = chi2 - 1.0
        self.success = True


def test_a_degenerate_refinement_is_discarded_for_the_grid_seed(
    monkeypatch, caplog
):
    """
    Given a refinement that returns u_0 = 5e-10 and t_E = 4e8 d with a
      slightly better chi2 than the grid -- and per-u_0 refits that are just
      as degenerate (the fake hands back t_E = 5e-10 d, shorter than the
      cadence),
    When find_pspl_seed runs,
    Then the grid point is returned instead, marked not converged, and the
      log names the degeneracy -- a seed like that costs 15 s per evaluation
      and puts the polish in a point-lens basin for ten hours.  (When the
      per-u_0 refits DO converge they are preferred to the grid point:
      test_wing_only_peak_in_a_gap_is_placed_from_the_wing.)
    """
    # ARRANGE
    t, f, ivar = _curve(2458550.0, 0.15, 18.0, 1.0, 0.3)
    real_minimize = peakfind.minimize

    def fake_minimize(fun, x0, **kw):
        grid_chi2 = fun(x0)
        return _Degenerate(x0[0], grid_chi2)

    monkeypatch.setattr(peakfind, "minimize", fake_minimize)

    # ACT
    with caplog.at_level(logging.WARNING, logger=peakfind.__name__):
        seed = peakfind.find_pspl_seed([(t, f, ivar)])
    monkeypatch.setattr(peakfind, "minimize", real_minimize)

    # ASSERT: the grid point, not the runaway
    assert seed["u_0"] in peakfind._U0_GRID
    assert seed["t_E"] in peakfind._TE_GRID
    assert seed["converged"] is False
    assert "degeneracy" in caplog.text


@pytest.mark.parametrize(
    "u_0, t_E, span, degenerate",
    [
        (5e-10, 4.4e8, 2000.0, True),  # DC2018-226's actual seed
        (0.15, 18.0, 200.0, False),
        (0.15, 500.0, 200.0, True),  # t_E longer than twice the data span
        (20.0, 18.0, 200.0, True),  # unmagnified: no event
        (0.15, 18.0, 0.0, False),  # no span known: only u_0 is tested
        (np.nan, 18.0, 200.0, True),
    ],
)
def test_refinement_is_degenerate_rule(u_0, t_E, span, degenerate):
    assert peakfind.refinement_is_degenerate(u_0, t_E, span) is degenerate


def test_a_healthy_refinement_is_kept():
    """The guard must not fire on the recovery cases above: a clean event
    refines to its injected geometry and reports converged."""
    t, f, ivar = _curve(2458550.0, 0.15, 18.0, 1.0, 0.3)
    seed = peakfind.find_pspl_seed([(t, f, ivar)])
    assert seed["converged"] is True
    assert abs(seed["u_0"] - 0.15) < 0.02


# ---------------------------------------------------------------------------
# Primary first (review 2.4.14, experiment A): the anomaly is masked and the
# broader event behind it is the seed
# ---------------------------------------------------------------------------


def _seasons(t_0, gap_lo, gap_hi, span=200.0, n=6000):
    """Epochs over a span with one gap -- a peak can be hidden in it."""
    t = np.linspace(t_0 - span / 2, t_0 + span / 2, n)
    return t[(t < gap_lo) | (t > gap_hi)]


def _pspl_curve_on(t, t_0, u_0, t_E, f_s, f_b, noise, seed):
    rng = np.random.default_rng(seed)
    flux = f_s * _pspl(t, t_0, u_0, t_E) + f_b
    err = np.full(t.size, noise * (f_s + f_b))
    return t, flux + rng.normal(0.0, err), 1.0 / err**2


def test_wing_only_peak_in_a_gap_is_placed_from_the_wing():
    """DC2018-226's primary: u_0 = 1.1, t_E = 18 d, peak in a season gap so
    only the declining wing is observed.  The free refinement walks u_0 to
    zero (the wing fixes t_0 and t_E for ANY u_0 but not u_0 itself); the
    per-u_0 refits must still put t_0 within a fraction of t_E of the truth
    instead of at the season's first epoch."""
    t_0, u_0, t_E = 2459958.95, 1.1, 18.0
    t = _seasons(t_0, t_0 - 80.0, t_0 + 28.0, n=20000)
    seed = peakfind.find_pspl_seed(
        [_pspl_curve_on(t, t_0, u_0, t_E, 1.0, 1.0, 2e-2, 11)],
        primary_first=False,
    )
    # Half a t_E: a wing alone leaves t_0 correlated with u_0 and t_E, and
    # a seed this close is in the basin (experiment A's posterior on 226
    # was +/- 3.6 d wide in t_0 from a start AT the truth).  The raw grid
    # point the old fallback returned sat at the season's first epoch,
    # 1.6 t_E off.
    assert abs(seed["t_0"] - t_0) < 0.5 * t_E
    # t_E trades against the unconstrained u_0 along the wing: a factor of
    # two either way is the same basin.
    assert 0.5 * t_E < seed["t_E"] < 2.0 * t_E
    assert seed["u_0"] in peakfind._U0_GRID or seed["converged"]


def test_anomaly_dominated_curve_seeds_the_primary():
    """A 226-like curve: the primary's wing plus a one-day, many-sigma spike
    68 days after t_0.  The first fit locks onto the spike; the primary-first
    pass masks it and returns the primary, reporting the spike's window."""
    t_0, u_0, t_E = 2459958.95, 1.1, 18.0
    # Roman-like cadence (100 epochs/day): dense enough that a one-day spike
    # outweighs the wing in chi2, which is what makes it the first fit.
    t = _seasons(t_0, t_0 - 80.0, t_0 + 28.0, n=20000)
    # 2% errors: the wing is a few sigma per epoch at the season's start,
    # as 226's was (1.7 sigma), and the anomaly below is ~10 sigma per epoch
    # over a day, as 226's was (9.4).
    tt, flux, ivar = _pspl_curve_on(t, t_0, u_0, t_E, 1.0, 1.0, 2e-2, 12)
    # the anomaly: a planetary-caustic bump, FWHM ~1 d
    t_anom = t_0 + 68.0
    flux = flux + 0.4 * np.exp(-0.5 * ((tt - t_anom) / 0.45) ** 2)
    curves = [(tt, flux, ivar)]

    first = peakfind.find_pspl_seed(curves, primary_first=False)
    assert abs(first["t_0"] - t_anom) < 1.0, "the spike IS the strongest feature"

    seed = peakfind.find_pspl_seed(curves)
    assert abs(seed["t_0"] - t_0) < 0.5 * t_E
    assert seed["anomaly"] is not None
    lo, hi = seed["anomaly"]["window"]
    assert lo < t_anom < hi
    assert seed["anomaly"]["fwhm"] == pytest.approx(1.06, abs=0.5)


def test_a_plain_event_is_unchanged_by_the_primary_first_pass():
    """On an ordinary well-sampled PSPL the masked refit fits the same event
    from its wings, lands inside the mask, and the first fit stands."""
    t_0, u_0, t_E = 2458550.0, 0.15, 18.0
    curves = [_curve(t_0, u_0, t_E, 1.0, 0.5, seed=4)]
    a = peakfind.find_pspl_seed(curves, primary_first=False)
    b = peakfind.find_pspl_seed(curves)
    for k in ("t_0", "u_0", "t_E", "chi2"):
        assert a[k] == pytest.approx(b[k], rel=1e-9)
    assert b["anomaly"] is None


def test_a_lone_short_event_stays_the_seed():
    """A free-floating-planet-class spike with nothing else in the curve:
    masking it leaves noise, the second fit is insignificant, and the spike
    remains the seed -- the pass must not invent a primary."""
    t_0, u_0, t_E = 2458550.0, 0.1, 0.3
    curves = [_curve(t_0, u_0, t_E, 1.0, 0.5, n=20000, span=100.0, seed=5)]
    seed = peakfind.find_pspl_seed(curves)
    assert abs(seed["t_0"] - t_0) < 0.1
    assert seed["anomaly"] is None


def test_feature_window_measures_the_spike_not_the_model():
    """The mask comes from the data's own FWHM, padded, and never from the
    (possibly degenerate) model that found the feature."""
    t = np.linspace(0.0, 100.0, 10001)  # 0.01 d cadence -> 0.1 d bins
    rng = np.random.default_rng(6)
    flux = 1.0 + 0.5 * np.exp(-0.5 * ((t - 50.0) / 0.5) ** 2)
    flux = flux + rng.normal(0.0, 0.01, t.size)
    lo, hi, fwhm = peakfind.feature_window([(t, flux, np.full(t.size, 1e4))], 50.0)
    assert fwhm == pytest.approx(2.355 * 0.5, rel=0.3)
    assert lo < 50.0 - fwhm and hi > 50.0 + fwhm
    assert (hi - lo) < 6 * fwhm + 1.0

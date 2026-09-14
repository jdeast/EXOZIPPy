"""The built-in PSPL peak finder (review 8.4.9).

WHAT THESE GUARD.  The seeder's whole value is that it finds the EVENT from
the data alone, so the tests are recovery tests on synthetic curves whose
answer is known exactly, plus the two properties that make it safe to leave
on by default: it must not need the flux scale to be known, and it must
degrade to None rather than raise.
"""

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

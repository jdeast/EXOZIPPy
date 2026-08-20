"""The DE move cannot cross modes at its own optimal scale; gamma=1 can.

`RawLayout.propose` builds x_i + gamma*(x_j - x_k) with gamma ~ 2.38/sqrt(2d)
-- 0.32 at d=27, and the gamma adapter drives it lower still.  When x_j and
x_k sit in different modes their difference IS the inter-mode vector, but
scaled by 0.32 the proposal lands a third of the way across, in the valley,
and is rejected.  So the DE kernel is a good WITHIN-mode sampler that cannot
move between modes, and every between-mode transition falls to PT round trips
-- which on DC2018 event 128 were measured at ZERO even with a ladder whose
swap acceptance had been equalized to 0.504 +/- 0.019.

ter Braak (2006) prescribes the fix: use gamma = 1 on a fraction of proposals
(conventionally 0.1), so the proposal translates by the FULL difference vector
and lands in the other mode.  Both kernels are symmetric in (x_j, x_k), so the
mixture needs no Hastings correction -- which is what makes this a ten-line
change rather than a new acceptance ratio (contrast the snooker update, whose
radial step needs an r^(d-1) Jacobian).

Two properties are pinned here: the hop is OFF by default and bit-identical
when off, and when on it actually reaches the other mode where the scaled
gamma provably does not.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from exozippy.samplers._common import RawLayout
from exozippy.samplers.ptde_async import ptde_async_sample


class _MinimalSystem:
    active_components = {}

    def get_raw_start(self, model):
        return model.initial_point()


def _bimodal_model(sep=8.0):
    """Two well-separated Gaussian modes in one dimension.

    `sep` is in units of the component sigma, so the valley between the modes
    is deep: at sep=8 the midpoint sits 4 sigma from either centre.
    """
    with pm.Model() as model:
        x = pm.Normal("x", mu=0.0, sigma=1.0)
        # An EQUAL-WEIGHT mixture: logaddexp of the two components, minus the
        # base RV's own logp so it is not double counted.  The first version
        # of this added logp(N(sep,1)) - logp(N(0,1)), which cancels the base
        # term and leaves a SINGLE mode at sep -- both runs then placed 100%
        # of their draws "in the far mode" and the test was vacuous.
        pm.Potential(
            "bimodal",
            pt.logaddexp(
                pm.logp(pm.Normal.dist(mu=0.0, sigma=1.0), x),
                pm.logp(pm.Normal.dist(mu=sep, sigma=1.0), x),
            )
            - pm.logp(pm.Normal.dist(mu=0.0, sigma=1.0), x),
        )
    return model


def test_scaled_gamma_lands_in_the_valley_and_unity_gamma_does_not():
    """
    Given two population members in different modes,
    When a proposal is built from their difference,
    Then the scaled gamma lands between the modes while gamma=1 lands ON the
      far mode.

    This is the geometric core of the whole issue, with no sampler involved.
    """
    # Arrange: mode A at 0, mode B at 10, in a 27-dim space (DC2018-shaped).
    d = 27
    start = {"x": np.zeros(d)}
    layout = RawLayout(start)
    a = np.zeros(d)
    b = np.full(d, 10.0)
    pop = np.vstack([a, b, a])  # member 0 in A, member 1 in B, member 2 in A
    gamma_scaled = 2.38 / np.sqrt(2 * d)

    # Act: propose from member 0 (in mode A).  Members 1 and 2 straddle the
    # modes, so the difference vector is +/- the inter-mode displacement.
    rng = np.random.default_rng(0)
    scaled = layout.propose(rng, pop, 0, gamma_scaled, jitter=0.0)
    rng = np.random.default_rng(0)
    unity = layout.propose(rng, pop, 0, 1.0, jitter=0.0)

    # Assert: |scaled| is a fraction of the way; |unity| is all the way.
    frac_scaled = abs(scaled[0]) / 10.0
    frac_unity = abs(unity[0]) / 10.0
    assert frac_scaled == pytest.approx(gamma_scaled, rel=1e-9)
    assert frac_scaled < 0.35, "the scaled step should fall far short"
    assert frac_unity == pytest.approx(1.0, rel=1e-9)


def test_mode_hop_is_off_by_default_and_leaves_the_rng_stream_untouched():
    """
    Given de_mode_hop unset,
    When two runs are made with the same seed, one passing 0.0 explicitly,
    Then the posteriors are bit-identical.

    The hop's rng draw is taken only when hops are enabled, precisely so that
    turning the feature off is not merely statistically equivalent but
    identical -- RawLayout documents bit-identical draw order as a property
    and every existing config must keep it.
    """
    # Arrange / Act
    out = []
    for kwargs in ({}, {"de_mode_hop": 0.0}):
        model = _bimodal_model()
        idata = ptde_async_sample(
            model,
            _MinimalSystem(),
            draws=25,
            tune=25,
            n_temps=2,
            T_max=4.0,
            n_chains=4,
            cores=1,
            seed=11,
            log_interval=1000,
            **kwargs,
        )
        out.append(np.asarray(idata.posterior["x"]))

    # Assert
    np.testing.assert_array_equal(out[0], out[1])


def test_mode_hop_rejects_an_out_of_range_probability():
    """A probability outside [0, 1) is a config error, not a clamp."""
    model = _bimodal_model()
    for bad in (-0.1, 1.0, 2.5):
        with pytest.raises(ValueError, match="de_mode_hop"):
            ptde_async_sample(
                model,
                _MinimalSystem(),
                draws=5,
                tune=5,
                n_temps=2,
                T_max=2.0,
                n_chains=4,
                cores=1,
                seed=1,
                log_interval=1000,
                de_mode_hop=bad,
            )


def test_hops_reach_the_other_basin_where_the_scaled_gamma_cannot():
    """
    Given a population STRADDLING two well-separated basins in 27 dimensions,
    When many proposals are generated with the scaled gamma and with gamma=1,
    Then only the gamma=1 proposals land inside the other basin.

    Tested at the PROPOSAL level rather than by running the sampler, for two
    reasons.  It isolates the DE kernel from PT transport, which would
    otherwise supply the crossings and mask the effect; and the dimension
    matters -- the optimal gamma is 2.38/sqrt(2d), so in 1-D it is 1.68, LARGER
    than one, and a low-dimensional toy shows no effect at all (the first
    version of this test used a 1-D target and both arms crossed freely).

    NOTE THE LIMITATION this makes explicit: a hop moves a chain between
    basins the population ALREADY OCCUPIES, because the difference vector is
    drawn from the population.  It is a mixing accelerator for known modes,
    not a mode DISCOVERER.  That is exactly the right tool for our case --
    MMEXOFAST seeds both members of the s <-> 1/s pair, and the mode weights
    were reported UNRELIABLE (N_eff = 19, 2/10 chains never switching) --
    but it will not find an unseeded third basin.
    """
    # Arrange: 27-D, two basins separated along axis 0, population split
    # between them exactly as a two-seed start would leave it.
    d, sep = 27, 30.0
    layout = RawLayout({"x": np.zeros(d)})
    rng = np.random.default_rng(3)
    n_pop = 2 * d
    pop = rng.standard_normal((n_pop, d))
    pop[n_pop // 2 :, 0] += sep  # half in basin B
    gamma_scaled = 2.38 / np.sqrt(2 * d)

    def frac_crossing(gamma):
        """Fraction of proposals from basin A that land inside basin B."""
        hits = 0
        trials = 400
        for t in range(trials):
            i = t % (n_pop // 2)  # always propose FROM basin A
            v = layout.propose(rng, pop, i, gamma, jitter=0.0)
            # "inside basin B" = within 5 units of its centre on axis 0.
            if abs(v[0] - sep) < 5.0:
                hits += 1
        return hits / trials

    # Act
    frac_scaled = frac_crossing(gamma_scaled)
    frac_unity = frac_crossing(1.0)

    # Assert
    assert frac_scaled == 0.0, (
        f"the scaled gamma reached basin B on {frac_scaled:.3f} of proposals; "
        f"it should never get there (step is {gamma_scaled:.2f} x the gap)"
    )
    assert frac_unity > 0.2, (
        f"gamma=1 reached basin B on only {frac_unity:.3f} of proposals"
    )

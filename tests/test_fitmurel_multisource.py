"""`fitmurel: true` on a TWO-source event (review 1.6.9).

Pre-split, fitmurel sampled a mu_rel pair for EVERY source while the lens
star's pm was derived from trajectory 0 only, so trajectory j >= 1's
mu_rel detached from the star kinematics: 2(N-1) spurious dimensions, the
identity mu_rel_j - mu_rel_0 = pm_s0 - pm_sj enforced nowhere.  The 8.6.17
split made mu_rel EVENT-level (ruling R1: co-moving sources, one relative
proper motion per event), so the extra dimensions are unrepresentable.
This pins it: exactly one sampled mu_rel pair, and the lens pm = pm_source
+ mu_rel identity at the start.
"""

import numpy as np
import pytensor
import pytest
from mulens_synthetic import build, mulens_config, mulens_params, write_flat_lc


def _eval(model, node, point):
    (node,) = model.replace_rvs_by_values([node])
    f = pytensor.function(model.value_vars, node, on_unused_input="ignore")
    return np.atleast_1d(f(*[point[v.name] for v in model.value_vars]))


@pytest.mark.slow
def test_two_sources_sample_one_mu_rel_and_the_lens_pm_follows(tmp_path):
    """
    Given: a 2-source event with fitmurel on the event block,
    When: the model is built,
    Then: the only sampled mu_rel coordinates are mulensevent's one-element
      mu_ra_rel / mu_dec_rel (no per-source copies), and at the start the
      lens star's pm equals source 0's pm plus that mu_rel, component by
      component, while both sources keep their own sampled pm.
    """
    # Arrange / Act
    lc = write_flat_lc(tmp_path / "lc.dat")
    config = mulens_config(
        lc, sources=(("S1", {}), ("S2", {})), event={"fitmurel": True}
    )
    system, model = build(config, mulens_params(config))
    point = model.initial_point()

    # Assert: one sampled mu_rel pair, event-level, length 1.
    murel_vars = [v.name for v in model.value_vars if "mu_" in v.name]
    murel_vars = [n for n in murel_vars if "_rel" in n]
    assert sorted(murel_vars) == [
        "mulensevent.mu_dec_rel_raw",
        "mulensevent.mu_ra_rel_raw",
    ], murel_vars
    for n in murel_vars:
        assert np.size(point[n]) == 1, (n, point[n])

    # Assert: the lens pm is DERIVED from source 0's pm + mu_rel, and the
    # sources' pm stay sampled.
    star_ndx = {n: i for i, n in enumerate(system.star.names)}
    lens, s1, s2 = star_ndx["L1"], star_ndx["S1"], star_ndx["S2"]
    for axis in ("ra", "dec"):
        pm = getattr(system.star, f"pm_{axis}")
        assert pm.element_is_derived(lens)
        assert pm.element_is_sampled(s1) and pm.element_is_sampled(s2)
        pm_v = _eval(model, pm.value, point)
        mu = _eval(
            model, getattr(system.mulensevent, f"mu_{axis}_rel").value, point
        )
        assert mu.size == 1
        assert np.isclose(pm_v[lens], pm_v[s1] + mu[0], rtol=0, atol=1e-12), (
            axis,
            pm_v,
            mu,
        )

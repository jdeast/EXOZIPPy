"""The DC2018 answer key's alpha maps onto ours (conventions.md C22).

    alpha_EXZ = alpha_key + 180 - atan2(sin(phase) cos(inc), cos(phase))

The pairs below are copied from the 2026-10-02 measurement (the private notes
repo's alpha_conventions.txt sec 4, scan JSON alpha_conventions/
dc18_alpha.json + dc18_alpha_extra.json, produced by
scripts/dc18_alpha_convention.py): the key's alpha, phase and inc for one
event, and the alpha the light curve itself prefers at the key's other
parameters (MulensModel, which is EXOZIPPy's convention, C18).  Copied, not
read, so the test needs neither the notes repo nor the DC2018 data tree.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

DC18 = Path(__file__).resolve().parents[1] / "examples" / "DC2018"

# event: (alpha_key, phase, inc, alpha_fit, P [yr]) -- all chi2 contrast >= 1000
PAIRS = {
    4: (38.6665, 270.630, 83.799, 302.95, 686.81),
    12: (274.149, 79.424, 43.007, 18.70, 10.78),
    66: (68.818, 50.322, -85.602, 243.55, 82.00),
    78: (132.215, 22.718, 17.950, 290.50, 39.65),
    163: (71.2536, 279.850, 68.674, 315.70, 31.11),
    223: (300.308, 61.191, -40.786, 64.95, 8.81),
    # The shortest-period strong event: the simulator's lens orbital motion
    # leaves the static scan 1.5 deg off the t_0 geometry, still well inside
    # the measured max.
    128: (348.357, 231.558, 50.556, 308.15, 1.27),
}
# The rule's own predictions for those events, from the same table.
PREDICTED = {
    4: 302.854,
    12: 18.472,
    66: 243.536,
    78: 290.497,
    163: 315.732,
    223: 66.301,
    128: 309.684,
}


@pytest.fixture(scope="module")
def common():
    sys.path.insert(0, str(DC18))
    try:
        yield importlib.import_module("dc18_common")
    finally:
        sys.path.remove(str(DC18))


@pytest.mark.parametrize("event", sorted(PAIRS))
def test_key_alpha_maps_onto_the_light_curves_own_alpha(common, event):
    alpha_key, phase, inc, alpha_fit, _ = PAIRS[event]
    pred = float(common.key_alpha_to_exozippy(alpha_key, phase, inc))
    assert 0.0 <= pred < 360.0
    assert pred == pytest.approx(PREDICTED[event], abs=2e-3)
    # The measurement: median 0.10 deg, max 1.53 deg over the 19 events
    # with chi2 contrast >= 1000.
    assert abs(float(common.wrap180(alpha_fit - pred))) < 1.6


def test_no_global_offset_explains_the_pairs(common):
    """Why the key was wrongly recorded as unmappable (C22 as first written):
    the correction theta_axis differs per event, so `fit - key` is not a
    constant.  If this ever concentrated, the mapping above would be
    redundant -- and the pairs would have been mis-copied."""
    off = np.array(
        [float(common.wrap180(v[3] - v[0])) for v in PAIRS.values()]
    )
    R = np.hypot(
        np.cos(np.radians(off)).mean(), np.sin(np.radians(off)).mean()
    )
    assert R < 0.7


def test_alpha_pull_is_taken_on_the_circle(common):
    """results.csv reports alpha in (-180, 180]; the mapped truth is in
    [0, 360).  Event 128's EXOZIPPy fit (-52.114 +/- 0.039) against the
    mapped key (309.684) is 1.8 deg, not 361.8 deg."""
    pull = common.sigma_pull(309.684, -52.114, 0.039, 0.039, angle=True)
    assert pull == pytest.approx((309.684 - 307.886) / 0.039, rel=1e-6)


def test_mirror_branch_truth_reports_the_tie(common):
    u0, alpha, mirrored = common.mirror_branch_truth(0.1418, 309.684, -0.14)
    assert mirrored
    assert u0 == pytest.approx(-0.1418)
    assert alpha == pytest.approx(360.0 - 309.684)
    assert common.mirror_branch_truth(0.1418, 309.684, 0.14) == (
        0.1418,
        309.684,
        False,
    )

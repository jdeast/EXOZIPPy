"""The SBC uniformity statistics (examples/validation/sbc_stats.py, review 7.16.4).

Fast and model-free: synthetic rank samples of known shape.  These are the
teeth of the slow end-to-end smoke test -- a statistic that cannot reject a U,
a hump or a slope would let that test pass on a miscalibrated model.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "examples" / "validation")
)

import sbc_stats  # noqa: E402

L = 99
N = 200


def _ranks_from_u(u):
    return np.minimum((np.asarray(u) * (L + 1)).astype(int), L)


def _stats(r, rng):
    gamma = sbc_stats.ecdf_band_gamma(r.size, L, rng=np.random.default_rng(1))
    _, p_ecdf, inside = sbc_stats.ecdf_diff(r, L, gamma)
    return (
        sbc_stats.chi2_uniformity(r, L, 20),
        sbc_stats.ks_uniformity(r, L, rng),
        p_ecdf,
        inside,
    )


def test_uniform_ranks_pass_every_statistic():
    """Given ranks drawn uniformly, when tested, then nothing rejects at 1%."""
    rng = np.random.default_rng(3)
    r = rng.integers(0, L + 1, N)
    chi2_p, ks_p, ecdf_p, inside = _stats(r, rng)
    assert chi2_p > 0.01 and ks_p > 0.01 and ecdf_p > 0.01
    assert inside


@pytest.mark.parametrize(
    "shape, u_of",
    [
        # posterior too narrow: the truth falls in the tails (U)
        ("U", lambda rng: rng.beta(0.5, 0.5, N)),
        # posterior too wide: the truth sits in the middle (hump)
        ("hump", lambda rng: rng.beta(3.0, 3.0, N)),
        # posterior biased: the truth sits low (slope)
        ("slope", lambda rng: rng.beta(1.0, 1.6, N)),
    ],
)
def test_miscalibrated_shapes_are_rejected(shape, u_of):
    """Given a U, a hump or a slope at N = 200, then the ECDF band and both
    p-values reject it."""
    rng = np.random.default_rng(5)
    r = _ranks_from_u(u_of(rng))
    chi2_p, ks_p, ecdf_p, inside = _stats(r, rng)
    assert not inside, shape
    assert ecdf_p < 0.01, shape
    assert min(chi2_p, ks_p) < 0.01, shape


def test_simultaneous_band_has_its_nominal_coverage():
    """The adjusted band holds a uniform sample everywhere ~95% of the time --
    a POINTWISE 95% band would be crossed far more often than 5%."""
    rng = np.random.default_rng(11)
    n = 100
    gamma = sbc_stats.ecdf_band_gamma(n, L, rng=np.random.default_rng(2))
    inside = [
        sbc_stats.ecdf_diff(rng.integers(0, L + 1, n), L, gamma)[2]
        for _ in range(400)
    ]
    assert 0.90 < np.mean(inside) < 0.99
    assert gamma < 0.05 / 5  # the adjustment is large for 99 evaluation points


def test_rank_bins_must_divide_evenly():
    with pytest.raises(ValueError):
        sbc_stats.chi2_uniformity(np.zeros(10, int), 100, 20)

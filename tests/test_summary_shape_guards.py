"""Posterior/summary SHAPE mismatches raise instead of printing a stand-in.

A reporting site used to repair a count mismatch the codebase itself
produced and carry on with a plausible, wrong number (CLAUDE.md "No
fallbacks for internal invariants"):

- review 2.2.17: ``Parameter.compute_mode_summaries`` read ANY disagreement
  between the posterior's sample count and the mode labels as "constant over
  the trace" and put the pooled all-draws summary under every mode.  Only a
  bare value (ndim 0) or a trailing axis of 1 (generate_posterior's fixed
  branch) is constant; anything else is a stale or foreign posterior.
"""

import numpy as np
import pytest

from exozippy.components.parameter import Parameter


def _param(shape=()):
    return Parameter(
        label="toy.x",
        latex="x",
        description="toy parameter",
        initval=10.0,
        lower=0.0,
        upper=100.0,
        shape=shape,
    )


# --- 2.2.17: compute_mode_summaries ---------------------------------------


def test_sample_count_mismatch_with_mode_labels_raises():
    """
    Given a posterior of 10 draws and 12 mode labels,
    When per-mode summaries are computed,
    Then a ValueError names the parameter and both counts -- the pooled
      summary is never published under every mode.
    """
    p = _param()
    p.posterior = np.arange(10.0)
    labels = np.array([0, 1] * 6)

    with pytest.raises(
        ValueError,
        match=r"toy\.x: posterior has 10 samples but 12 mode labels",
    ):
        p.compute_mode_summaries(labels, 2)


def test_matching_sample_count_splits_by_mode():
    """
    Given 12 draws whose two halves sit at 1 and 5, labelled by mode,
    When per-mode summaries are computed,
    Then each mode reports its own median, not the pooled one.
    """
    p = _param()
    p.posterior = np.array([1.0] * 6 + [5.0] * 6)
    labels = np.array([0] * 6 + [1] * 6)

    summaries = p.compute_mode_summaries(labels, 2)

    assert [s.median for s in summaries] == [1.0, 5.0]


@pytest.mark.parametrize(
    "posterior, shape",
    [
        (3.0, ()),  # generate_posterior's fixed scalar: no sample axis
        (np.array([[1.0], [2.0], [3.0]]), (3,)),  # fixed vector: n_el x 1
    ],
)
def test_constant_posterior_is_the_same_in_every_mode(posterior, shape):
    """
    Given a constant posterior in either legitimate spelling (a bare value,
      or elements x 1 sample),
    When per-mode summaries are computed against 12 labels,
    Then every mode carries the one constant summary and nothing raises.
    """
    p = _param(shape)
    p.posterior = posterior
    labels = np.array([0, 1] * 6)

    summaries = p.compute_mode_summaries(labels, 2)

    assert len(summaries) == 2
    assert summaries[0] == summaries[1]

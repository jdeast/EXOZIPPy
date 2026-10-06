"""Posterior/summary SHAPE mismatches raise instead of printing a stand-in.

Two reporting sites used to repair a count mismatch the codebase itself
produced and carry on with a plausible, wrong number (CLAUDE.md "No
fallbacks for internal invariants"):

- review 2.2.17: ``Parameter.compute_mode_summaries`` read ANY disagreement
  between the posterior's sample count and the mode labels as "constant over
  the trace" and put the pooled all-draws summary under every mode.  Only a
  bare value (ndim 0) or a trailing axis of 1 (generate_posterior's fixed
  branch) is constant; anything else is a stale or foreign posterior.
- review 2.11.7: ``build_csv_output``'s ``summ_at`` returned the LAST
  element's summary for every element the summary list was short of, so
  results.csv published that element's median under the missing names.
"""

import numpy as np
import pytest

from exozippy.components.parameter import Parameter, PosteriorSummary
from exozippy.outputs.latex import build_csv_output


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


# --- 2.11.7: build_csv_output's summ_at -----------------------------------


class _FakeComp:
    label = "toy"


def _fake_system(comp):
    class _Sys:
        name = "test"

        def get_all_components(self):
            return [comp]

    return _Sys()


def test_short_summary_list_raises_in_build_csv_output(tmp_path):
    """
    Given a 3-element parameter whose summary list has only 2 entries,
    When build_csv_output writes the results CSV,
    Then a ValueError names the parameter and both lengths -- element 2 is
      not published with element 1's numbers.
    """
    comp = _FakeComp()
    comp.x = _param((3,))
    comp.x.summary = [
        PosteriorSummary(median=1.0, err_minus=0.1, err_plus=0.1),
        PosteriorSummary(median=2.0, err_minus=0.1, err_plus=0.1),
    ]

    with pytest.raises(
        ValueError,
        match=r"toy\.x: summary has 2 elements but the parameter has 3",
    ):
        build_csv_output(_fake_system(comp), str(tmp_path / "results.csv"))


def test_full_summary_list_writes_one_row_per_element(tmp_path):
    """
    Given a 3-element parameter with one summary per element,
    When build_csv_output writes the results CSV,
    Then each element's row carries its own median.
    """
    comp = _FakeComp()
    comp.x = _param((3,))
    comp.x.summary = [
        PosteriorSummary(median=float(m), err_minus=0.1, err_plus=0.1)
        for m in (1, 2, 3)
    ]
    path = tmp_path / "results.csv"

    build_csv_output(_fake_system(comp), str(path))

    rows = [
        ln.split(",")
        for ln in path.read_text().splitlines()
        if not ln.startswith("#")
    ]
    assert [float(r[1]) for r in rows] == [1.0, 2.0, 3.0]

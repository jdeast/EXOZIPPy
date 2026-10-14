"""The DC2018 collector reads a summary table whose vectors are per element.

`<prefix>_summary.txt` is `az.summary(...).to_string()`, which names a
sampled VECTOR once per element -- `lens.log_s[0]`, `lens.log_s[1]` -- while
`SUMMARY_VARS` names the variable.  An exact dict lookup therefore matched
only the SCALAR parameters, so the collector's `s_rhat`, `s_ess_bulk`,
`alpha_rhat` and `alpha_ess_bulk` columns were silently absent from every row
it ever produced: no KeyError, no warning, just missing convergence numbers
for the two parameters hardest to converge in a binary-lens fit.

The table below is a real header and real rows, copied from
`/n/holylabs/jjohnson_lab/Lab/dc18_rerun/detect128/asfit/DC2018_128_summary.txt`
(arviz 1.3, which is what pyproject pins) so the test does not need the
DC2018 data tree.  Two things it is deliberately literal about: the
`# WARNING: convergence NOT reached -- max Rhat=...` banner that
`run.py:_format_summary` prepends, and the fact that `to_string()` writes NO
header token above the index column, so the header line has one FEWER field
than every data row.
"""

import importlib
import sys
from pathlib import Path

import pytest

DC18 = Path(__file__).resolve().parents[1] / "examples" / "DC2018"

SUMMARY_TEXT = """\
# burn-in discarded: 20000 draws (40% of 50000); chains kept: 78
# WARNING: convergence NOT reached -- max Rhat=1.021 (lens.log_s), min ESS=3362 (lens.log_s)
                      mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
source.t_0        2.46e+06  0.101  2.5e+06  2.5e+06     8811     7749  1.00    0.0035  0.0024
source.u_0           0.151   0.02    0.12     0.18      7804     6500  1.01     0.002  0.0014
lens.log_s[0]        0.051   0.05   -0.03     0.13      6231     5900  1.01     0.002  0.0014
lens.log_s[1]        0.049   0.05   -0.03     0.13      4007     3800  1.02     0.002  0.0014
lens.xalpha[0]        0.08   0.96    -1.0      1.0      5725     5100  1.01    0.0065  0.0043
lens.xalpha[1]        0.07   0.96    -1.0      1.0      5100     4900  1.01    0.0065  0.0043
lens.yalpha[0]        0.50   0.19    0.19     0.82      5725     5100  1.01    0.0065  0.0043
lens.yalpha[1]        0.48   0.19    0.19     0.82      3362     3100  1.02    0.0065  0.0043
planet.log_q         -3.00    2.5    -6.0     -1.0      5735     5400  1.01       6.2     4.3
"""


@pytest.fixture(scope="module")
def dc18():
    sys.path.insert(0, str(DC18))
    try:
        yield (
            importlib.import_module("dc18_common"),
            importlib.import_module("collect_results"),
        )
    finally:
        sys.path.remove(str(DC18))


@pytest.fixture
def per_var(tmp_path, dc18):
    common, _ = dc18
    path = tmp_path / "DC2018_128_summary.txt"
    path.write_text(SUMMARY_TEXT)
    overall, per_var = common.read_summary_diagnostics(path)
    # The reader itself must survive the banner and the index-column offset;
    # if it does not, the lookup test below would pass vacuously.
    assert overall["rhat_max"] == pytest.approx(1.02)
    assert overall["ess_bulk_min"] == pytest.approx(3362)
    return per_var


def test_every_reported_parameter_finds_its_vector_elements(dc18, per_var):
    """
    Given a summary table naming s and alpha per ELEMENT (`lens.log_s[0]`),
    When the collector looks up each reported parameter's diagnostics,
    Then the vector-valued ones resolve, not just the scalars.

    The regression: `[per_var[n] for n in names if n in per_var]` matched
    `source.t_0` and `planet.log_q` (scalars) and nothing for s or alpha, so
    the collector dropped those columns without saying so.
    """
    # Arrange
    _, collect = dc18

    # Act / Assert
    for p, names in collect.SUMMARY_VARS.items():
        if p == "rho":
            # Legitimately absent: these fits ran finite_source off, so rho
            # was never sampled.  Asserting it present would encode a bug.
            continue
        hits = collect.match_per_var(per_var, names)
        assert hits, (
            f"{p} (names={names}) matched nothing in a real summary table "
            f"whose keys are {sorted(per_var)}"
        )


def test_a_vector_parameter_reports_its_WORST_element(dc18, per_var):
    """
    Given `lens.log_s[0]` (r_hat 1.01, ess 6231) and `lens.log_s[1]`
      (r_hat 1.02, ess 4007),
    When s's diagnostics are reduced,
    Then the WORST element is reported -- max r_hat, min ess -- because one
      unconverged element means the parameter is unconverged.
    """
    # Arrange
    _, collect = dc18

    # Act
    entries = collect.match_per_var(per_var, collect.SUMMARY_VARS["s"])

    # Assert
    assert len(entries) == 2, "both log_s elements should be in play"
    assert max(e[0] for e in entries) == pytest.approx(1.02)
    assert min(e[1] for e in entries) == pytest.approx(4007)

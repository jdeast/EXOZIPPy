"""A second report off one live System must report the SECOND trace.

Review 3.14.7.  ``Parameter.summary`` is a cache of one set of draws, and
every consumer recomputes it only ``if p.summary is None``
(``build_csv_output``, ``to_latex_def``, ``latex._value_cells``).
``System.distribute_posterior`` overwrote ``Parameter.posterior`` and left
the summary alone, so a System that was reported twice -- ``exozippy-modes``,
the GUI's re-solve, any script that fits and then re-reports -- published the
FIRST trace's medians the second time round.  Silently: a median is a
plausible number whichever trace it came from.

This is 2.11.3's ``mode_summaries`` defect one level down, and 2.11.3's fix
does not transfer: there the cached LENGTH could be compared against the new
mode count, and a single summary has no length.  The fix is invalidation at
the write (``Parameter.posterior``'s setter).
"""

import numpy as np
import pytest

from exozippy.components.parameter import Parameter
from exozippy.outputs.latex import build_csv_output
from exozippy.system import System

_LC = dict(baseline=1.0, depth=0.01, P=3.2, tc=2459100.0, err=4.0e-4)

N_CHAIN, N_DRAW = 2, 40


def _write_lc(path, n=180):
    t = np.linspace(_LC["tc"] - 0.2, _LC["tc"] + 0.2, n)
    in_transit = np.abs(t - _LC["tc"]) < 0.04
    flux = _LC["baseline"] - _LC["depth"] * in_transit
    np.savetxt(path, np.column_stack([t, flux, np.full_like(t, _LC["err"])]))


@pytest.fixture(scope="module")
def built_system(tmp_path_factory):
    """A real, fully built System -- the thing that gets reported twice."""
    lc_path = tmp_path_factory.mktemp("tworeports") / "two.TESS.dat"
    _write_lc(lc_path)
    config = {
        "run": {"name": "tworeports"},
        "star": [{"name": "A", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
        "band": [{"name": "TESS", "filter": "TESS"}],
        "transit": [{"name": "TESS", "file": str(lc_path), "band": "TESS"}],
    }
    user_params = {
        "star.A.mass": {"initval": 1.0, "sigma": 0.05},
        "star.A.radius": {"initval": 1.0, "sigma": 0.1},
        "star.A.teff": {"initval": 5800, "sigma": 100},
        "star.A.feh": {"initval": 0.0, "sigma": 0.1},
        "orbit.b.period": {"initval": _LC["P"]},
        "orbit.b.tc": {"initval": _LC["tc"]},
    }
    system = System(config, user_params=user_params)
    system.prepare()
    system.build_model()
    return system


def _fake_trace(system, offset):
    """An InferenceData naming every Parameter of ``system``, shifted by ``offset``.

    Every label is present, so ``distribute_posterior`` takes its
    already-in-the-trace branch for all of them and no derived parameter has
    to be re-evaluated -- the point here is the CACHE, not the evaluation.
    """
    import arviz as az
    import xarray as xr

    rng = np.random.default_rng(7)
    data_vars = {}
    for label, par in system.get_parameter_lookup().items():
        n = par._n_elements()
        if n == 1:
            arr = offset + rng.normal(0.0, 0.01, size=(N_CHAIN, N_DRAW))
            dims = ["chain", "draw"]
        else:
            arr = offset + rng.normal(0.0, 0.01, size=(N_CHAIN, N_DRAW, n))
            dims = ["chain", "draw", f"{label}_dim"]
        data_vars[label] = xr.DataArray(arr, dims=dims)
    return az.from_dict({"posterior": xr.Dataset(data_vars)})


def _csv_medians(system, path):
    """{parameter name: median} from a freshly written results CSV."""
    build_csv_output(system, str(path))
    medians = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        name, med, _up, _lo = line.rstrip("\n").split(",")
        medians[name] = float(med)
    return medians


def test_two_reports_off_one_system_report_their_own_traces(
    built_system, tmp_path
):
    """
    Given one live System reported twice, with a DIFFERENT trace each time,
    When the second results CSV is written,
    Then its medians come from the second trace, not the first.

    Before 3.14.7 the two CSVs were identical: ``distribute_posterior``
    replaced ``posterior`` but ``build_csv_output`` recomputed the summary
    only when it was None, so the second report re-served the first trace's
    medians for every parameter.
    """
    # ARRANGE
    system = built_system

    # ACT
    system.distribute_posterior(_fake_trace(system, offset=1.0))
    first = _csv_medians(system, tmp_path / "first_results.csv")

    system.distribute_posterior(_fake_trace(system, offset=5.0))
    second = _csv_medians(system, tmp_path / "second_results.csv")

    # ASSERT
    shared = set(first) & set(second)
    assert shared, "the two reports named no parameter in common"
    moved = [k for k in shared if first[k] != second[k]]
    assert moved, "the second report reproduced the first report's medians"
    # The offset is 4.0 and the scatter is 0.01, so every row that carries a
    # posterior must move by about 4.  A row that did not move at all is a
    # fixed parameter reported from its initval, which has no summary.
    for key in moved:
        assert second[key] - first[key] == pytest.approx(4.0, abs=0.2), key


def test_a_summary_is_dropped_when_new_draws_arrive():
    """
    Given a Parameter whose summary has been computed,
    When a different posterior is assigned,
    Then the summary and the per-mode summaries are dropped, so the next
      reader recomputes from the new draws.

    The unit-level statement of the same rule, and the one that covers
    writers other than distribute_posterior.
    """
    # ARRANGE
    par = Parameter(label="x.y", initval=1.0, unit="", internal_unit="")
    par.posterior = np.arange(10.0)
    par.compute_summary()
    par.compute_mode_summaries(np.zeros(10, dtype=int), 1)
    assert par.summary is not None and par.mode_summaries is not None

    # ACT
    par.posterior = np.arange(10.0) + 100.0

    # ASSERT
    assert par.summary is None
    assert par.mode_summaries is None
    assert par.compute_summary().median == pytest.approx(104.5)


def test_assigning_the_same_object_back_still_invalidates():
    """
    Given a Parameter whose summary is current,
    When the SAME posterior object is assigned again,
    Then the summary is still dropped.

    Deliberate: the setter does not compare, it invalidates.  Keying the
    cache on the object it came from -- the alternative 3.14.7 offered --
    would keep the cache here and would also miss an in-place mutation of
    that object, which is the case identity cannot see at all.
    """
    par = Parameter(label="x.y", initval=1.0, unit="", internal_unit="")
    draws = np.arange(10.0)
    par.posterior = draws
    par.compute_summary()

    par.posterior = draws

    assert par.summary is None

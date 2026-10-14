"""Detrend coefficients are reported under the file that owns them.

``detrend_coeffs`` has one element per detrend COLUMN over every file, laid
out block-diagonally in file order, so element j is not instance j.  Every
report still labelled element j with instance j's name: in a transit fit
whose first three files have no detrend columns and whose later
ground-based files each detrend against airmass, results.csv put the
ground-based airmass coefficients under the three column-less files, and
the restart file seeded them under those names.

The fixture is the case a length check cannot see: three RV instruments
and three columns between them (A none, B one, C two), so the vector and
the config list have the same length and only the OWNERS tell them apart.
Params-file semantics are deliberately unchanged -- ``rvinstrument.<i>.
detrend_coeffs`` is column i -- so the pins below use the index form.
"""

import csv
import json

import arviz as az
import numpy as np
import pytest
import xarray as xr
import yaml

from exozippy.mkparam import write_param_file
from exozippy.outputs.latex import build_csv_output, build_latex_output
from exozippy.system import System
from exozippy.trace_meta import (
    OWNERS_ATTR,
    structural_metadata,
    trace_element_owners,
)

_PINS = [10.0, 20.0, 30.0]  # m/s per raw unit, element order


def _write_rv(path, n_cols, seed):
    rng = np.random.default_rng(seed)
    t = np.sort(2450000.0 + rng.uniform(0.0, 40.0, 30))
    rv = 50.0 * np.sin(2 * np.pi * (t - 2450004.0) / 8.0) + rng.normal(0, 3, t.size)
    cols = [t, rv, np.full_like(t, 3.0)]
    cols += [1.0 + 0.1 * rng.random(t.size) for _ in range(n_cols)]
    np.savetxt(path, np.column_stack(cols))


@pytest.fixture(scope="module")
def three_instruments(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("detrend_owners")
    insts = []
    for i, (name, n_cols) in enumerate((("A", 0), ("B", 1), ("C", 2))):
        path = tmp / f"{name}.rv"
        _write_rv(path, n_cols, seed=i)
        insts.append({"name": name, "file": str(path)})
    config = {
        "run": {"name": "detrend_owners"},
        "star": [{"name": "S", "mist": False}],
        "planet": [{"name": "b"}],
        "orbit": [{"name": "b", "primary": ["S"], "companion": ["b"]}],
        "rvinstrument": insts,
    }
    user_params = {
        "star.S.mass": {"initval": 1.0, "sigma": 0.05},
        "star.S.radius": {"initval": 1.0, "sigma": 0.1},
        "star.S.teff": {"initval": 5800, "sigma": 100},
        "star.S.feh": {"initval": 0.0, "sigma": 0.1},
        "orbit.b.period": {"initval": 8.0},
        "orbit.b.tc": {"initval": 2450004.0},
    }
    for i, v in enumerate(_PINS):
        user_params[f"rvinstrument.{i}.detrend_coeffs"] = {"initval": v, "sigma": 0}
    system = System(config, user_params=user_params)
    system.prepare()
    system.build_model()
    return system, tmp


def test_owners_are_the_files_whose_rows_carry_each_column(three_instruments):
    """
    Given three RV files with 0, 1 and 2 detrend columns,
    When the system is built,
    Then each coefficient's owner is the file on whose rows its column of
    the design matrix is nonzero -- read off the matrix itself, so the
    assertion carries no layout of its own to get wrong.
    """
    # ARRANGE
    system, _ = three_instruments
    comp = system.rvinstrument
    p = comp.detrend_coeffs

    # ACT
    support = [
        [comp.names[i] for i in range(comp.n_elements)
         if np.any(comp.detrend_matrix[comp.rows(i), j] != 0)]
        for j in range(comp.detrend_matrix.shape[1])
    ]

    # ASSERT
    assert support == [[o] for o in p.element_owners]
    assert list(p.element_owners) == ["B", "C", "C"]


def test_results_csv_labels_each_coefficient_by_its_file(three_instruments, tmp_path):
    """
    Given the pinned coefficients 10, 20, 30 on columns 0, 1, 2,
    When results.csv is written,
    Then each value sits under its own file and position within it, and A
    -- a file with no columns -- has no coefficient row.

    Regression: the rows were labelled A, B, C by position, so A (no
    columns) reported B's coefficient, B reported C's first and only C's
    second landed under C.
    """
    # ARRANGE
    system, _ = three_instruments
    out = tmp_path / "results.csv"

    # ACT
    build_csv_output(system, csv_filename=str(out))
    with open(out) as f:
        rows = {r[0]: r[1] for r in csv.reader(line for line in f if not line.startswith("#"))}

    # ASSERT
    coeffs = {k: float(v) for k, v in rows.items() if "detrend_coeffs" in k}
    assert coeffs == {
        "rvinstrument.B.detrend_coeffs[0]": pytest.approx(_PINS[0]),
        "rvinstrument.C.detrend_coeffs[0]": pytest.approx(_PINS[1]),
        "rvinstrument.C.detrend_coeffs[1]": pytest.approx(_PINS[2]),
    }


def test_latex_table_puts_coefficients_under_their_owner(three_instruments, tmp_path):
    """
    Given the same system,
    When the deluxetable is written,
    Then A's sub-head carries no coefficient row, B's one and C's two, the
    latter subscripted by position -- and each row cites its OWN element's
    value macro, so the numbers follow the labels.
    """
    # ARRANGE
    system, _ = three_instruments
    table = tmp_path / "table.tex"
    p = system.rvinstrument.detrend_coeffs

    # ACT
    build_latex_output(system, var_filename=str(tmp_path / "v.tex"),
                       table_filename=str(table), caption="x")
    text = table.read_text()

    # ASSERT
    blocks = {}
    for name in ("A", "B", "C"):
        start = text.index(rf"\textit{{{name}:}}")
        nxt = [text.index(rf"\textit{{{m}:}}") for m in ("A", "B", "C")
               if text.index(rf"\textit{{{m}:}}") > start]
        blocks[name] = text[start:min(nxt) if nxt else len(text)]
    rows = {n: [ln for ln in b.splitlines() if "Detrending" in ln]
            for n, b in blocks.items()}
    assert [len(rows[n]) for n in "ABC"] == [0, 1, 2]
    assert "c_{1}" in rows["C"][1] and "c_{0}" in rows["C"][0]
    for j, (owner, k) in enumerate((("B", 0), ("C", 0), ("C", 1))):
        macro = p.to_table_line_at(j).split("&")[1].strip()
        assert macro in rows[owner][k]


def test_owners_are_stamped_on_the_trace(three_instruments):
    """
    Given the built system,
    When the structural metadata a trace is stamped with is built,
    Then it carries the owners, and the reader mkparam uses returns them.
    """
    # ARRANGE
    system, _ = three_instruments
    idata = az.from_dict({"posterior": {"x": np.zeros((1, 1))}})

    # ACT
    meta = structural_metadata(system)
    idata.attrs[OWNERS_ATTR] = meta[OWNERS_ATTR]

    # ASSERT
    assert trace_element_owners(idata) == {
        "rvinstrument.detrend_coeffs": ["B", "C", "C"]
    }


def _vector_trace(tmp_path, values, owners):
    vals = np.asarray(values, dtype=float)[None, None, :]
    post = xr.Dataset({
        "rvinstrument.detrend_coeffs": xr.DataArray(vals, dims=["chain", "draw", "d"]),
        "rvinstrument.detrend_coeffs_raw": xr.DataArray(vals * 0.1, dims=["chain", "draw", "d"]),
    })
    stats = xr.Dataset({"lp": xr.DataArray(np.array([[-1.0]]), dims=["chain", "draw"])})
    idata = az.from_dict({"posterior": post, "sample_stats": stats})
    idata.attrs[OWNERS_ATTR] = json.dumps({"rvinstrument.detrend_coeffs": owners})
    path = tmp_path / "trace.nc"
    idata.to_netcdf(str(path))
    return path


def test_restart_file_writes_coefficients_in_the_index_form(tmp_path):
    """
    Given a trace whose coefficients are owned B, C, C and an earlier
      restart file that seeded element 1 under the old name form
      (`rvinstrument.B.detrend_coeffs`, which resolves to element 1) with a
      prior,
    When mkparam writes the next restart file,
    Then every coefficient is written as `rvinstrument.<i>.detrend_coeffs`
    with its own value, the old entry's prior is carried onto element 1 and
    the old key is not repeated, and the header says which file each
    index belongs to.
    """
    # ARRANGE
    trace = _vector_trace(tmp_path, [1.5, 2.5, 3.5], ["B", "C", "C"])
    old = tmp_path / "old.params.yaml"
    old.write_text("rvinstrument.B.detrend_coeffs:\n  initval: 2.0\n  sigma: 0.5\n  mu: 2.0\n")
    config = {
        "prefix": "fitresults/x",
        "parameter_file": str(old),
        "rvinstrument": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
    }

    # ACT
    out = write_param_file(config, base_dir=tmp_path, trace_path=trace)
    text = out.read_text()
    written = yaml.safe_load(text)

    # ASSERT
    coeffs = {k: v for k, v in written.items() if "detrend_coeffs" in k}
    assert set(coeffs) == {f"rvinstrument.{i}.detrend_coeffs" for i in range(3)}
    assert [coeffs[f"rvinstrument.{i}.detrend_coeffs"]["initval"] for i in range(3)] == [1.5, 2.5, 3.5]
    assert coeffs["rvinstrument.1.detrend_coeffs"]["sigma"] == 0.5
    assert "rvinstrument.1.detrend_coeffs = C, its column 0" in text
    assert "rvinstrument.2.detrend_coeffs = C, its column 1" in text

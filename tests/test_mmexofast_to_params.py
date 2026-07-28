# tests/test_mmexofast_to_params.py
"""Tests for the MMEXOFAST -> params.yaml converter.

The interface between MMEXOFAST and EXOZIPPy is a JSON file, not an import,
so these tests pin the parts of that contract the converter relies on. Per-fit
``sigmas`` are optional: MMEXOFAST omits them for solutions that are initial
estimates rather than optimized fits, and is expected to stop emitting them
altogether.
"""

import copy
import json
from pathlib import Path

import pytest
import yaml

from exozippy.utilities.mmexofast_to_params import mmexofast_to_params

MMX_PATH = (
    Path(__file__).parent.parent / "examples" / "DC2018_128" / "mmexofast.json"
)

PARAM_PATHS = [
    "lens.Lens.t_0",
    "lens.Lens.u_0",
    "lens.Lens.t_E",
    "lens.Lens.s",
    "lens.Lens.alpha",
    "lens.Lens.rho",
    "lens.Lens.q",
]


def _write_without_sigmas(tmp_path, keep=None):
    """Copy the example file, dropping every per-fit sigma except ``keep``."""
    data = json.loads(MMX_PATH.read_text())
    stripped = copy.deepcopy(data)
    for fit in stripped["fits"]:
        if keep is None:
            fit.pop("sigmas", None)
        else:
            fit["sigmas"] = {k: fit["sigmas"][k] for k in keep}

    path = tmp_path / "mmexofast_no_sigmas.json"
    path.write_text(json.dumps(stripped))

    return path


def test_sigmas_present_emit_init_scale(tmp_path):
    """Given a file with sigmas, every parameter block carries an
    init_scale alongside its initval."""
    parsed = yaml.safe_load(
        mmexofast_to_params(MMX_PATH, out_path=tmp_path / "out.yaml")
    )

    assert sorted(parsed) == sorted(PARAM_PATHS)
    for path in PARAM_PATHS:
        assert "initval" in parsed[path], path
        assert "init_scale" in parsed[path], path


def test_missing_sigmas_still_converts(tmp_path):
    """Given a file whose fits have no sigmas at all -- what MMEXOFAST emits
    when it has only initial estimates -- conversion succeeds and yields
    initvals with no init_scale, rather than raising KeyError."""
    path = _write_without_sigmas(tmp_path)

    parsed = yaml.safe_load(
        mmexofast_to_params(path, out_path=tmp_path / "out.yaml")
    )

    assert sorted(parsed) == sorted(PARAM_PATHS)
    for name, entry in parsed.items():
        assert "initval" in entry, name
        assert "init_scale" not in entry, name


def test_partial_sigmas_emit_init_scale_only_where_known(tmp_path):
    """A sigma present for some parameters and absent for others is not
    all-or-nothing: each block is decided independently."""
    path = _write_without_sigmas(tmp_path, keep=["t_0", "log_q"])

    parsed = yaml.safe_load(
        mmexofast_to_params(path, out_path=tmp_path / "out.yaml")
    )

    with_scale = {
        name for name, entry in parsed.items() if "init_scale" in entry
    }
    assert with_scale == {"lens.Lens.t_0", "lens.Lens.q"}


@pytest.mark.parametrize("solution_index", [None, 0, 1])
def test_output_is_valid_yaml_without_sigmas(tmp_path, solution_index):
    """Both the multi-seed (None) and single-solution paths stay parseable
    when sigmas are absent."""
    path = _write_without_sigmas(tmp_path)

    text = mmexofast_to_params(
        path,
        solution_index=solution_index,
        out_path=tmp_path / "out.yaml",
    )

    parsed = yaml.safe_load(text)
    assert parsed
    assert all("initval" in entry for entry in parsed.values())

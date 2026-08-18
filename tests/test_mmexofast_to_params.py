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


def test_sigmas_present_are_not_emitted(tmp_path):
    """Given a file with sigmas, no init_scale is written -- whitening
    scales are measured from the data by EXOZIPPy at startup, so the key
    would only trigger the warn-and-ignore path."""
    parsed = yaml.safe_load(
        mmexofast_to_params(MMX_PATH, out_path=tmp_path / "out.yaml")
    )

    assert sorted(parsed) == sorted(PARAM_PATHS)
    for path in PARAM_PATHS:
        assert "initval" in parsed[path], path
        assert "init_scale" not in parsed[path], path


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


def test_partial_sigmas_also_not_emitted(tmp_path):
    """Sigmas present for only some parameters still produce no init_scale
    anywhere (the field is retired)."""
    path = _write_without_sigmas(tmp_path, keep=["t_0", "log_q"])

    parsed = yaml.safe_load(
        mmexofast_to_params(path, out_path=tmp_path / "out.yaml")
    )

    with_scale = {
        name for name, entry in parsed.items() if "init_scale" in entry
    }
    assert with_scale == set()


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


# ---------------------------------------------------------------------------
# jd_offset (review 1.6.6)
# ---------------------------------------------------------------------------


def _write_with_jd_offset(tmp_path, jd_offset):
    """Copy the example file, stamping a top-level jd_offset on it."""
    data = json.loads(MMX_PATH.read_text())
    data["jd_offset"] = jd_offset
    path = tmp_path / "mmexofast_jd_offset.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("solution_index", [None, 0])
def test_t_0_has_jd_offset_subtracted(tmp_path, solution_index):
    """Given a newer JSON carrying jd_offset = 2450000, when it is converted,
    then the emitted t_0 initvals are shifted back into the data's own time
    system -- the same contract mmexofast_support.push_seed_hints keeps."""
    jd_offset = 2450000.0
    raw = json.loads(MMX_PATH.read_text())
    shifted = _write_with_jd_offset(tmp_path, jd_offset)

    parsed = yaml.safe_load(
        mmexofast_to_params(
            shifted,
            solution_index=solution_index,
            out_path=tmp_path / "out.yaml",
        )
    )

    fits = (
        raw["fits"]
        if solution_index is None
        else [raw["fits"][solution_index]]
    )
    expected = [fit["parameters"]["t_0"] - jd_offset for fit in fits]
    got = parsed["lens.Lens.t_0"]["initval"]
    got = got if isinstance(got, list) else [got]
    assert got == pytest.approx(expected, abs=1e-7)


def test_no_jd_offset_key_leaves_t_0_alone(tmp_path):
    """Given a pre-jd_offset JSON, when it is converted, then t_0 is emitted
    verbatim -- the shift must not appear out of nowhere for older files."""
    raw = json.loads(MMX_PATH.read_text())
    assert "jd_offset" not in raw

    parsed = yaml.safe_load(
        mmexofast_to_params(MMX_PATH, out_path=tmp_path / "o.yaml")
    )

    expected = [fit["parameters"]["t_0"] for fit in raw["fits"]]
    assert parsed["lens.Lens.t_0"]["initval"] == pytest.approx(
        expected, abs=1e-7
    )


def test_only_t_0_is_shifted(tmp_path):
    """Given a JSON with jd_offset, when it is converted, then only the epoch
    parameter moves -- t_E is a duration and u_0/s/q/rho/alpha are
    dimensionless, so a shift there would be a units error."""
    plain = yaml.safe_load(
        mmexofast_to_params(MMX_PATH, out_path=tmp_path / "a.yaml")
    )
    shifted_path = _write_with_jd_offset(tmp_path, 2450000.0)
    shifted = yaml.safe_load(
        mmexofast_to_params(shifted_path, out_path=tmp_path / "b.yaml")
    )

    for path in PARAM_PATHS:
        if path.endswith(".t_0"):
            continue
        assert shifted[path] == plain[path], path

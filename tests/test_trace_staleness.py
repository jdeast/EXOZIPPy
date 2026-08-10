"""
Tests for the saved-trace staleness check (src/exozippy/trace_meta.py).

A trace's raw draws only decode through the model they were sampled from.
Reloading one under an edited config (`recompute_trace: false`, or the
exozippy-modes CLI) used to succeed silently and regenerate every
posterior, LaTeX table and plot from foreign draws.  The fix stamps
evaluator.structural_hash into the trace's root attrs at save time and
verifies it on reload.

These tests pin the three load outcomes and the save-side stamping:
  * the fingerprint is written into the trace attrs and survives a real
    netCDF round trip;
  * a matching fingerprint reloads silently;
  * a mismatched fingerprint RAISES StaleTraceError, and the message names
    both what changed and the remedy;
  * a trace with NO fingerprint (every trace written before this check
    existed) reloads with an "unverifiable" warning -- never reported as
    stale -- and does not raise;
  * System.structural_fingerprint is a stable snapshot of the config +
    params the System was built from.

They follow AAA with Given/When/Then docstrings.
"""

import copy
import logging

import arviz as az
import numpy as np
import pytest

from exozippy.evaluator import structural_hash, structural_payload
from exozippy.system import System
from exozippy.trace_meta import (
    HASH_ATTR,
    PAYLOAD_ATTR,
    StaleTraceError,
    check_trace_freshness,
    stamp_structural_metadata,
)

_CONFIG = {
    "run": {"name": "staleness"},
    "star": [{"name": "A", "mist": False}],
    "planet": [{"name": "b"}],
    "orbit": [{"name": "b", "primary": ["A"], "companion": ["b"]}],
}
_PARAMS = {
    "star.A.mass": {"initval": 1.2, "sigma": 0.05},
    "orbit.b.logP": {"initval": 0.5, "lower": 0.1, "upper": 1.0},
}


class _FakeSystem:
    """Minimal stand-in exposing only System.structural_fingerprint.

    Uses the real evaluator functions, so the fingerprint is exactly what a
    System would produce; it just skips building the components.
    """

    def __init__(self, config, params=None):
        self._hash = structural_hash(config, params)
        self._payload = structural_payload(config, params)

    def structural_fingerprint(self):
        return self._hash, self._payload


def _idata():
    """A one-variable InferenceData standing in for a saved trace."""
    return az.from_dict({"posterior": {"star.mass_raw": np.zeros((2, 5))}})


# ---------------------------------------------------------------------------
# System fingerprint
# ---------------------------------------------------------------------------


def test_system_fingerprint_is_stable_for_equal_inputs():
    """
    Given the same config and params dicts,
    When two Systems are built from (deep copies of) them,
    Then their structural fingerprints agree.
    """
    sys_a = System(copy.deepcopy(_CONFIG), user_params=copy.deepcopy(_PARAMS))
    sys_b = System(copy.deepcopy(_CONFIG), user_params=copy.deepcopy(_PARAMS))

    hash_a, payload_a = sys_a.structural_fingerprint()
    hash_b, payload_b = sys_b.structural_fingerprint()

    assert hash_a == hash_b
    assert payload_a == payload_b
    assert hash_a == structural_hash(_CONFIG, _PARAMS)


def test_system_fingerprint_changes_on_a_bound_edit():
    """
    Given a System built from a params dict,
    When a bound is edited and another System is built,
    Then the structural fingerprints differ.
    """
    edited = copy.deepcopy(_PARAMS)
    edited["orbit.b.logP"]["upper"] = 2.0

    base = System(copy.deepcopy(_CONFIG), user_params=copy.deepcopy(_PARAMS))
    changed = System(copy.deepcopy(_CONFIG), user_params=edited)

    assert (
        base.structural_fingerprint()[0]
        != (changed.structural_fingerprint()[0])
    )


# ---------------------------------------------------------------------------
# Save side
# ---------------------------------------------------------------------------


def test_stamp_writes_the_fingerprint_into_trace_attrs():
    """
    Given an idata about to be saved and the System that produced it,
    When stamp_structural_metadata is called,
    Then the root attrs carry the System's hash and its payload.
    """
    system = System(copy.deepcopy(_CONFIG), user_params=copy.deepcopy(_PARAMS))
    idata = _idata()

    stamp_structural_metadata(idata, system)

    assert idata.attrs[HASH_ATTR] == system.structural_fingerprint()[0]
    assert PAYLOAD_ATTR in idata.attrs


def test_stamped_fingerprint_survives_a_netcdf_round_trip(tmp_path):
    """
    Given a stamped idata,
    When it is written to netCDF and read back,
    Then the reloaded trace still verifies against the same System.
    """
    system = _FakeSystem(_CONFIG, _PARAMS)
    idata = _idata()
    stamp_structural_metadata(idata, system)
    path = tmp_path / "round_trip_trace.nc"

    idata.to_netcdf(str(path))
    reloaded = az.from_netcdf(str(path))

    assert check_trace_freshness(reloaded, system, str(path)) == "match"


# ---------------------------------------------------------------------------
# Load side
# ---------------------------------------------------------------------------


def test_matching_fingerprint_reloads_silently(caplog):
    """
    Given a trace stamped by the same config it is being reloaded under,
    When it is checked,
    Then the check reports "match" and logs no warning.
    """
    system = _FakeSystem(_CONFIG, _PARAMS)
    idata = _idata()
    stamp_structural_metadata(idata, system)

    with caplog.at_level(logging.WARNING, logger="exozippy.trace_meta"):
        result = check_trace_freshness(idata, system, "fit_trace.nc")

    assert result == "match"
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_mismatched_fingerprint_raises_and_names_change_and_remedy():
    """
    Given a trace stamped under one config,
    When it is reloaded under a config whose bounds have been edited,
    Then StaleTraceError is raised, naming the changed parameter and the
    recompute_trace remedy.
    """
    old_system = _FakeSystem(_CONFIG, _PARAMS)
    idata = _idata()
    stamp_structural_metadata(idata, old_system)
    edited = copy.deepcopy(_PARAMS)
    edited["orbit.b.logP"]["upper"] = 2.0
    new_system = _FakeSystem(_CONFIG, edited)

    with pytest.raises(StaleTraceError) as excinfo:
        check_trace_freshness(idata, new_system, "fit_trace.nc")

    message = str(excinfo.value)
    assert "STALE TRACE" in message
    assert "fit_trace.nc" in message
    assert "orbit.b.logP" in message
    assert "recompute_trace: true" in message


def test_mismatch_names_a_removed_component():
    """
    Given a trace stamped with a two-star config,
    When it is reloaded under a config with one of the stars removed,
    Then the raised message names the changed component block.
    """
    two_stars = copy.deepcopy(_CONFIG)
    two_stars["star"] = [{"name": "A"}, {"name": "B"}]
    idata = _idata()
    stamp_structural_metadata(idata, _FakeSystem(two_stars, _PARAMS))

    with pytest.raises(StaleTraceError) as excinfo:
        check_trace_freshness(
            idata, _FakeSystem(_CONFIG, _PARAMS), "fit_trace.nc"
        )

    assert "component changed: star" in str(excinfo.value)


def test_mismatch_without_a_stored_payload_still_raises():
    """
    Given a trace carrying only a hash (its payload attr was dropped),
    When it is reloaded under a different config,
    Then it still raises, saying the specific difference cannot be shown.
    """
    idata = _idata()
    stamp_structural_metadata(idata, _FakeSystem(_CONFIG, _PARAMS))
    del idata.attrs[PAYLOAD_ATTR]
    other = copy.deepcopy(_CONFIG)
    other["star"] = [{"name": "A"}, {"name": "B"}]

    with pytest.raises(StaleTraceError) as excinfo:
        check_trace_freshness(
            idata, _FakeSystem(other, _PARAMS), "fit_trace.nc"
        )

    assert "cannot be shown" in str(excinfo.value)


def test_trace_without_a_fingerprint_is_unverifiable_not_stale(caplog):
    """
    Given a trace written before this check existed (no fingerprint attr),
    When it is reloaded,
    Then the check reports "unverifiable", warns in those words rather than
    as a mismatch, and does not raise.
    """
    system = _FakeSystem(_CONFIG, _PARAMS)
    idata = _idata()

    with caplog.at_level(logging.WARNING, logger="exozippy.trace_meta"):
        result = check_trace_freshness(idata, system, "old_trace.nc")

    assert result == "unverifiable"
    text = caplog.text
    assert "UNVERIFIABLE TRACE" in text
    assert "STALE TRACE" not in text
    assert "old_trace.nc" in text


# ---------------------------------------------------------------------------
# End to end through the exozippy-modes CLI, which regenerates the published
# LaTeX/CSV tables from a reloaded trace -- the same wrong-numbers exposure
# as run.py's `recompute_trace: false` branch.
# ---------------------------------------------------------------------------

_CLI_CONFIG = {"name": "stale_cli_test", "orbit": [{"name": "test_orbit"}]}
_CLI_PARAMS = {
    "orbit.test_orbit.logP": {"initval": 1.0, "lower": 0.1, "upper": 3.0},
    "orbit.test_orbit.tc": {"initval": 0.0},
    "orbit.test_orbit.secosw": {"initval": 0.0},
    "orbit.test_orbit.sesinw": {"initval": 0.0},
}


def _write_fit_inputs(tmp_path):
    """Write config.yaml + params.yaml; return (config_path, prefix, config,
    params) as re-read from disk (so a fingerprint taken here is over exactly
    the dicts the CLI will load)."""
    import yaml

    prefix = tmp_path / "stalefit"
    params_path = tmp_path / "params.yaml"
    config_path = tmp_path / "config.yaml"
    config = dict(_CLI_CONFIG)
    config["prefix"] = str(prefix)
    config["parameter_file"] = str(params_path)
    with open(params_path, "w") as f:
        yaml.safe_dump(_CLI_PARAMS, f)
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    with open(config_path) as f:
        config_disk = yaml.safe_load(f)
    with open(params_path) as f:
        params_disk = yaml.safe_load(f)
    return config_path, prefix, config_disk, params_disk


def _write_stamped_trace(prefix, config, params, rng):
    """A synthetic trace over the config's real free_RV names, stamped with
    the fingerprint of (config, params)."""
    system = System(copy.deepcopy(config), user_params=copy.deepcopy(params))
    system.prepare()
    model = system.build_model()
    names = [v.name for v in model.free_RVs]
    posterior = {n: rng.normal(0, 1, (2, 60)) for n in names}
    lp = rng.normal(100.0, 1.0, (2, 60))
    idata = az.from_dict({"posterior": posterior, "sample_stats": {"lp": lp}})
    stamp_structural_metadata(idata, _FakeSystem(config, params))
    trace_path = str(prefix) + "_trace.nc"
    idata.to_netcdf(trace_path)
    return trace_path


@pytest.mark.slow
def test_modes_cli_accepts_a_trace_stamped_by_its_own_config(tmp_path):
    """
    Given a trace stamped with the fingerprint of the very config the CLI
      loads from disk,
    When `exozippy-modes config.yaml` runs,
    Then it completes -- the fingerprint a System computes from the YAML
      reproduces the one written beside the draws.
    """
    from click.testing import CliRunner

    from exozippy import cli_modes

    rng = np.random.default_rng(7)
    config_path, prefix, config, params = _write_fit_inputs(tmp_path)
    _write_stamped_trace(prefix, config, params, rng)

    result = CliRunner().invoke(cli_modes.main, [str(config_path)])

    assert result.exit_code == 0, result.output + "\n" + repr(result.exception)


@pytest.mark.slow
def test_modes_cli_refuses_a_stale_trace(tmp_path):
    """
    Given a trace stamped under a config whose bounds differ from the one on
      disk,
    When `exozippy-modes config.yaml` runs,
    Then it fails with StaleTraceError instead of rewriting the LaTeX/CSV
      tables from the foreign draws.
    """
    from click.testing import CliRunner

    from exozippy import cli_modes

    rng = np.random.default_rng(7)
    config_path, prefix, config, params = _write_fit_inputs(tmp_path)
    other_params = copy.deepcopy(params)
    other_params["orbit.test_orbit.logP"]["upper"] = 4.0
    _write_stamped_trace(prefix, config, other_params, rng)

    result = CliRunner().invoke(cli_modes.main, [str(config_path)])

    assert result.exit_code != 0
    assert isinstance(result.exception, StaleTraceError)
    assert "orbit.test_orbit.logP" in str(result.exception)

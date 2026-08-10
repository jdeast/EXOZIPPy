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
    params the System was built from;
  * the exozippy version and the source tree's git commit are stamped too,
    and a mismatch quotes them plus the git lines that get back to that
    code -- while a version/commit difference on its own NEVER raises, since
    only the structural hash decides staleness;
  * mkprior refuses a structurally stale trace (its output seeds the next
    fit) but proceeds on an unstamped one.

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
    COMMIT_ATTR,
    DESCRIBE_ATTR,
    DIRTY_ATTR,
    HASH_ATTR,
    PAYLOAD_ATTR,
    VERSION_ATTR,
    StaleTraceError,
    check_trace_freshness,
    code_provenance,
    describe_trace_provenance,
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
# Code provenance: diagnostic context in the error, never a staleness test
# ---------------------------------------------------------------------------


def test_stamp_records_the_exozippy_version_and_git_commit():
    """
    Given a trace being stamped,
    When stamp_structural_metadata runs,
    Then the attrs carry the package version, and -- when the code is
    running from a git checkout -- the commit, describe string and dirty
    flag of that tree.
    """
    idata = _idata()

    stamp_structural_metadata(idata, _FakeSystem(_CONFIG, _PARAMS))

    prov = code_provenance()
    assert idata.attrs[VERSION_ATTR] == str(prov["version"])
    if prov["commit"]:
        assert idata.attrs[COMMIT_ATTR] == prov["commit"]
        assert len(idata.attrs[COMMIT_ATTR]) == 40
        assert idata.attrs[DESCRIBE_ATTR] == (prov["describe"] or "")
        assert idata.attrs[DIRTY_ATTR] in ("true", "false")
    else:  # installed wheel / no git available
        assert COMMIT_ATTR not in idata.attrs


def test_stale_message_quotes_the_recorded_code_and_the_way_back():
    """
    Given a trace stamped from a git checkout,
    When a structural mismatch raises,
    Then the message names the version that produced it and prints the git
    worktree lines that recreate that code.
    """
    prov = code_provenance()
    if not prov["commit"]:
        pytest.skip("not running from a git checkout")
    idata = _idata()
    stamp_structural_metadata(idata, _FakeSystem(_CONFIG, _PARAMS))
    other = copy.deepcopy(_CONFIG)
    other["star"] = [{"name": "A"}, {"name": "B"}]

    with pytest.raises(StaleTraceError) as excinfo:
        check_trace_freshness(
            idata, _FakeSystem(other, _PARAMS), "fit_trace.nc"
        )

    message = str(excinfo.value)
    assert f"exozippy {prov['version']}" in message
    assert "worktree add" in message
    assert prov["commit"] in message


def test_a_version_difference_alone_never_raises():
    """
    Given a trace whose structural hash matches but whose recorded version
      and commit are from entirely different code,
    When it is checked,
    Then it reports "match" -- the version is diagnostic context, not a
    second staleness criterion, so newer code with an unchanged model must
    keep reusing its trace.
    """
    system = _FakeSystem(_CONFIG, _PARAMS)
    idata = _idata()
    stamp_structural_metadata(idata, system)
    idata.attrs[VERSION_ATTR] = "0.0.0-ancient"
    idata.attrs[COMMIT_ATTR] = "0" * 40
    idata.attrs[DESCRIBE_ATTR] = "v0.0.1-1-g0000000"
    idata.attrs[DIRTY_ATTR] = "true"

    result = check_trace_freshness(idata, system, "fit_trace.nc")

    assert result == "match"


def test_provenance_report_without_git_does_not_print_git_lines():
    """
    Given attrs from a trace stamped by an installed package (version, no
      commit),
    When the provenance is described,
    Then it says the source cannot be checked out instead of printing a git
    command with a missing commit.
    """
    lines = describe_trace_provenance({VERSION_ATTR: "1.2.3"})

    text = " ".join(lines)
    assert "exozippy 1.2.3" in text
    assert "installed package" in text
    assert "worktree add" not in text


def test_provenance_report_without_any_metadata_says_so_plainly():
    """
    Given attrs from a trace written before this metadata existed,
    When the provenance is described,
    Then it states that plainly and prints no instructions.
    """
    lines = describe_trace_provenance({})

    text = " ".join(lines)
    assert "predates" in text
    assert "git" not in text.replace("git commit", "")


def test_provenance_report_flags_a_dirty_source_tree():
    """
    Given a trace stamped from a tree with uncommitted changes,
    When the provenance is described,
    Then the recovery lines carry the caveat that the commit alone does not
    reproduce the code that ran.
    """
    lines = describe_trace_provenance(
        {
            VERSION_ATTR: "1.2.3",
            COMMIT_ATTR: "a" * 40,
            DESCRIBE_ATTR: "v1.2.3-4-gaaaaaaa-dirty",
            DIRTY_ATTR: "true",
        }
    )

    text = " ".join(lines)
    assert "worktree add" in text
    assert "uncommitted changes" in text


# ---------------------------------------------------------------------------
# mkprior: its output seeds the NEXT fit, so a stale trace there corrupts a
# run that has not happened yet.
# ---------------------------------------------------------------------------

_MKPRIOR_CONFIG = {
    "prefix": "fitresults/model",
    "parameter_file": None,
    "star": [{"name": "Host"}],
}


def _mkprior_trace(tmp_path, stamp_source=None):
    """A one-draw trace mkprior can consume, optionally stamped."""
    import xarray as xr

    posterior = xr.Dataset(
        {
            "star.mass": xr.DataArray(
                np.array([[0.95]]), dims=["chain", "draw"]
            ),
            "star.mass_raw": xr.DataArray(
                np.array([[0.1]]), dims=["chain", "draw"]
            ),
        }
    )
    stats = xr.Dataset(
        {"lp": xr.DataArray(np.array([[-10.0]]), dims=["chain", "draw"])}
    )
    idata = az.from_dict({"posterior": posterior, "sample_stats": stats})
    if stamp_source is not None:
        stamp_structural_metadata(idata, stamp_source)
    path = tmp_path / "trace.nc"
    idata.to_netcdf(str(path))
    return path


def test_mkprior_refuses_a_structurally_stale_trace(tmp_path):
    """
    Given a trace stamped under a different config,
    When mkprior is asked to seed a restart file from it,
    Then it raises StaleTraceError instead of writing start values drawn
    from a foreign posterior.
    """
    from exozippy.mkparam import mkprior

    other = dict(_MKPRIOR_CONFIG)
    other["star"] = [{"name": "Host"}, {"name": "Companion"}]
    trace = _mkprior_trace(tmp_path, stamp_source=_FakeSystem(other, {}))

    with pytest.raises(StaleTraceError):
        mkprior(
            dict(_MKPRIOR_CONFIG),
            base_dir=tmp_path,
            trace_path=trace,
            output_path=tmp_path / "out.yaml",
        )

    assert not (tmp_path / "out.yaml").exists()


def test_mkprior_accepts_a_matching_fingerprint(tmp_path):
    """
    Given a trace stamped with the fingerprint mkprior computes for itself,
    When mkprior runs,
    Then it writes the restart file as before.
    """
    from exozippy.mkparam import mkprior

    trace = _mkprior_trace(
        tmp_path, stamp_source=_FakeSystem(_MKPRIOR_CONFIG, {})
    )

    out = mkprior(
        dict(_MKPRIOR_CONFIG),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    assert out.exists()


def test_mkprior_proceeds_on_an_unstamped_trace(tmp_path, caplog):
    """
    Given a trace written before this metadata existed,
    When mkprior runs,
    Then it warns that the trace is unverifiable and still writes the file.
    """
    from exozippy.mkparam import mkprior

    trace = _mkprior_trace(tmp_path)

    with caplog.at_level(logging.WARNING, logger="exozippy.trace_meta"):
        out = mkprior(
            dict(_MKPRIOR_CONFIG),
            base_dir=tmp_path,
            trace_path=trace,
            output_path=tmp_path / "out.yaml",
        )

    assert out.exists()
    assert "UNVERIFIABLE TRACE" in caplog.text


# ---------------------------------------------------------------------------
# The snapshot must survive components normalizing their own config blocks.
# ---------------------------------------------------------------------------

_RELATION_CONFIG = {
    "star": [{"name": "A"}, {"name": "B"}],
    "torres": [{"star": "A", "constrain": ["mass"]}],
    "mann": [{"star": "B", "constrain": ["mass", "radius"]}],
}


def test_fingerprint_survives_component_config_normalization():
    """
    Given a config whose components rewrite their own blocks while the
      System is being constructed (Mann/Torres derive `name:` from `star:`),
    When the fingerprint is recomputed from that config dict afterwards, the
      way mkprior does,
    Then it reproduces the System's snapshot exactly.

    Measured regression, not a hypothetical: the snapshot was originally
    taken before the component-instantiation loop, so a mann/torres config
    fingerprinted its instances as '0'/'1' at snapshot time and 'B'/'C'
    afterwards. Every kelt4-style fit would then have refused to write its
    own restart file. Verified against
    examples/kelt4/kelt4_rv+transit+sed.yaml, whose config is mutated in
    exactly three places, all inside System.__init__ -- stages 1-6 mutate it
    zero times.
    """
    config = copy.deepcopy(_RELATION_CONFIG)
    params = {"star.A.teff": {"initval": 5800.0}}

    system = System(config, user_params=copy.deepcopy(params))
    recomputed = structural_hash(config, params)

    # The normalization really did happen (else this proves nothing).
    assert config["mann"][0]["name"] == "B"
    assert config["torres"][0]["name"] == "A"
    assert system.structural_fingerprint()[0] == recomputed


def test_fingerprint_tracks_which_star_a_relation_constrains():
    """
    Given two configs whose Mann relation constrains different stars,
    When each is fingerprinted through a System,
    Then the hashes differ.

    Falls out of snapshotting after the component loop: the relation
    components key on `star:`, which _component_skeleton cannot see, but the
    `name:` they derive from it is exactly the missing information. Before
    the fix both spelled their instance '0' and swapping the constrained
    star was invisible.
    """
    config_b = copy.deepcopy(_RELATION_CONFIG)
    config_a = copy.deepcopy(_RELATION_CONFIG)
    config_a["mann"][0]["star"] = "A"
    config_a["torres"][0]["star"] = "B"

    hash_b = System(config_b, user_params={}).structural_fingerprint()[0]
    hash_a = System(config_a, user_params={}).structural_fingerprint()[0]

    assert hash_a != hash_b


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

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
  * mkparam refuses a structurally stale trace (its output seeds the next
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
    POSTERIOR_UNITS,
    UNITS_ATTR,
    VERSION_ATTR,
    StaleTraceError,
    check_posterior_units,
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
# Posterior units (review 2.3.4): the save path converts to user units, so a
# trace predating that conversion holds internal-unit draws that nothing
# downstream could detect.  Stamp it; warn on absence; never convert.
# ---------------------------------------------------------------------------


def test_stamp_records_the_posterior_unit_system():
    """
    Given a trace about to be written,
    When the structural metadata is stamped,
    Then the posterior's unit system is recorded alongside the fingerprint.
    """
    idata = _idata()

    stamp_structural_metadata(idata, _FakeSystem(_CONFIG, _PARAMS))

    assert idata.attrs[UNITS_ATTR] == POSTERIOR_UNITS


def test_a_stamped_trace_reloads_without_a_units_warning(caplog):
    """
    Given a trace this code stamped,
    When it is reloaded,
    Then nothing is said about units -- the check is inert on a good trace.
    """
    system = _FakeSystem(_CONFIG, _PARAMS)
    idata = _idata()
    stamp_structural_metadata(idata, system)

    with caplog.at_level(logging.WARNING, logger="exozippy.trace_meta"):
        assert check_trace_freshness(idata, system, "t.nc") == "match"

    assert "POSTERIOR UNITS" not in caplog.text


def test_a_trace_without_a_units_stamp_warns_and_still_loads(caplog):
    """
    Given a trace written before the stamp existed,
    When its units are checked,
    Then it warns that the draws may be in INTERNAL units and says how to
      fix it -- and does not raise, and does not convert.

    Warn rather than raise, because this is the trace_meta unverifiable
    case rather than a detected mismatch: the posterior conversion is older
    than the stamp, so only a genuinely pre-2026 trace is affected and
    refusing them all would invalidate working traces to catch a shrinking
    population.
    """
    with caplog.at_level(logging.WARNING, logger="exozippy.trace_meta"):
        result = check_posterior_units({}, "old_trace.nc")

    assert result == "unverifiable"
    assert "UNVERIFIABLE POSTERIOR UNITS" in caplog.text
    assert "old_trace.nc" in caplog.text
    assert "recompute_trace: true" in caplog.text


def test_an_unrecognized_units_stamp_warns_by_name(caplog):
    """
    Given a trace declaring a unit system this code does not know,
    When its units are checked,
    Then the warning names it and says every value may be off by a
      conversion factor.
    """
    with caplog.at_level(logging.WARNING, logger="exozippy.trace_meta"):
        result = check_posterior_units({UNITS_ATTR: "internal"}, "t.nc")

    assert result == "unknown"
    assert "UNKNOWN POSTERIOR UNITS" in caplog.text
    assert "internal" in caplog.text


def test_the_units_check_never_raises_and_never_touches_the_draws():
    """
    Given a trace with an unknown units stamp,
    When it is reloaded against a matching model,
    Then the reload still succeeds and the posterior is untouched.

    Converting on a guess would corrupt every trace already in user units --
    nearly all of them -- and the numbers give no way to tell the two apart
    (radians against degrees, solar against jupiter masses are all
    plausible), which is exactly why the stamp had to exist.
    """
    system = _FakeSystem(_CONFIG, _PARAMS)
    idata = _idata()
    stamp_structural_metadata(idata, system)
    idata.attrs[UNITS_ATTR] = "something-else"
    before = idata.posterior["star.mass_raw"].values.copy()

    assert check_trace_freshness(idata, system, "t.nc") == "match"

    np.testing.assert_array_equal(
        idata.posterior["star.mass_raw"].values, before
    )


def test_the_units_stamp_survives_a_netcdf_round_trip(tmp_path):
    """
    Given a stamped trace written to disk,
    When it is read back,
    Then the units stamp is still there -- a root attr netCDF preserves.
    """
    idata = _idata()
    stamp_structural_metadata(idata, _FakeSystem(_CONFIG, _PARAMS))
    path = tmp_path / "trace.nc"
    idata.to_netcdf(str(path))

    reloaded = az.from_netcdf(str(path))

    assert reloaded.attrs[UNITS_ATTR] == POSTERIOR_UNITS


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
# mkparam: its output seeds the NEXT fit, so a stale trace there corrupts a
# run that has not happened yet.
# ---------------------------------------------------------------------------

_MKPARAM_CONFIG = {
    "prefix": "fitresults/model",
    "parameter_file": None,
    "star": [{"name": "Host"}],
}


def _mkparam_trace(tmp_path, stamp_source=None):
    """A one-draw trace mkparam can consume, optionally stamped."""
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


def test_mkparam_refuses_a_structurally_stale_trace(tmp_path):
    """
    Given a trace stamped under a different config,
    When mkparam is asked to seed a restart file from it,
    Then it raises StaleTraceError instead of writing start values drawn
    from a foreign posterior.
    """
    from exozippy.mkparam import write_param_file

    other = dict(_MKPARAM_CONFIG)
    other["star"] = [{"name": "Host"}, {"name": "Companion"}]
    trace = _mkparam_trace(tmp_path, stamp_source=_FakeSystem(other, {}))

    with pytest.raises(StaleTraceError):
        write_param_file(
            dict(_MKPARAM_CONFIG),
            base_dir=tmp_path,
            trace_path=trace,
            output_path=tmp_path / "out.yaml",
        )

    assert not (tmp_path / "out.yaml").exists()


def test_mkparam_accepts_a_matching_fingerprint(tmp_path):
    """
    Given a trace stamped with the fingerprint mkparam computes for itself,
    When mkparam runs,
    Then it writes the restart file as before.
    """
    from exozippy.mkparam import write_param_file

    trace = _mkparam_trace(
        tmp_path, stamp_source=_FakeSystem(_MKPARAM_CONFIG, {})
    )

    out = write_param_file(
        dict(_MKPARAM_CONFIG),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    assert out.exists()


def test_mkparam_proceeds_on_an_unstamped_trace(tmp_path, caplog):
    """
    Given a trace written before this metadata existed,
    When mkparam runs,
    Then it warns that the trace is unverifiable and still writes the file.
    """
    from exozippy.mkparam import write_param_file

    trace = _mkparam_trace(tmp_path)

    with caplog.at_level(logging.WARNING, logger="exozippy.trace_meta"):
        out = write_param_file(
            dict(_MKPARAM_CONFIG),
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
      way mkparam does,
    Then it reproduces the System's snapshot exactly.

    Measured regression, not a hypothetical: the snapshot was originally
    taken before the component-instantiation loop, so a mann/torres config
    fingerprinted its instances as '0'/'1' at snapshot time and 'B'/'C'
    afterwards. Every kelt4-style fit would then have refused to write its
    own restart file. Verified against
    examples/kelt4/kelt4_rv+transit+sed.yaml, whose config is mutated in
    exactly three places, all inside System.__init__ -- stages 1-7 mutate it
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


# ---------------------------------------------------------------------------
# The whitening file beside the trace: run.py's reuse branch must restore it,
# never re-measure it, and never write to it.
# ---------------------------------------------------------------------------


def _write_whitening_file(prefix, config, params):
    """Measure and persist a whitening state for (config, params)."""
    from exozippy import whitening

    system = System(copy.deepcopy(config), user_params=copy.deepcopy(params))
    system.prepare()
    model = system.build_model()
    report = whitening.measure_and_whiten(
        system, model, system.get_raw_start(model)
    )
    path = str(prefix) + "_whitening.json"
    whitening.save_whitening(system, path, map_lp=report["map_lp"])
    return path


@pytest.mark.slow
def test_run_fit_reuse_path_never_overwrites_the_whitening_file(tmp_path):
    """
    Given a saved trace being reused (`recompute_trace: false`) and a
      whitening file beside it that no longer applies to the build,
    When run_fit reaches its whitening step,
    Then it raises StaleWhiteningError and the whitening file on disk is
      byte-for-byte unchanged.

    Pre-fix, run.py's whitening step made no distinction between "about to
    sample" and "about to decode existing draws": it re-probed and re-saved
    on both.  On the reuse path that silently re-coordinated the very trace
    being reused, and overwrote the only record of the coordinates those
    draws were actually taken in.  The file-bytes assertion below is the
    behavioural pin (it fails on pre-fix code whatever run_fit goes on to
    do); the exception check pins the diagnosis on top of it.
    """
    import json
    import os

    import yaml

    from exozippy.run import run_fit

    # Arrange -- a trace + a whitening file that describe this model...
    rng = np.random.default_rng(11)
    config_path, prefix, config, params = _write_fit_inputs(tmp_path)
    _write_stamped_trace(prefix, config, params, rng)
    whitening_path = _write_whitening_file(prefix, config, params)
    # ...then break the whitening file the way a truncated write or an edited
    # model does: drop one entry.  The trace's structural fingerprint is
    # untouched, so this is a whitening mismatch and nothing else.
    data = json.loads(open(whitening_path).read())
    dropped = sorted(data["params"])[0]
    del data["params"][dropped]
    with open(whitening_path, "w") as f:
        json.dump(data, f)
    before_bytes = open(whitening_path, "rb").read()

    run_config = copy.deepcopy(config)
    run_config["sampler"] = {"recompute_trace": False}
    with open(config_path, "w") as f:
        yaml.safe_dump(run_config, f)

    # Act -- deliberately NOT pytest.raises: the point is what the file on
    # disk looks like afterwards, which must be asserted whether run_fit
    # raised, returned, or blew up somewhere else entirely.
    cwd = os.getcwd()
    error = None
    os.chdir(tmp_path)
    try:
        run_fit(run_config, user_params=copy.deepcopy(params))
    except BaseException as exc:  # noqa: BLE001 - re-asserted below
        error = exc
    finally:
        os.chdir(cwd)

    # Assert
    assert open(whitening_path, "rb").read() == before_bytes, (
        "the reuse path rewrote the whitening state a saved trace was "
        "sampled under; its raw draws no longer decode to the values the "
        "sampler visited"
    )
    assert type(error).__name__ == "StaleWhiteningError", error
    assert dropped in str(error)
    assert "recompute_trace: true" in str(error)


# --------------------------------------------------------------------------
# Model-selecting per-instance keys must reach the hash. These flip WHICH
# likelihood is built while leaving the component set, the file list and
# every parameter's structure untouched, so before 2026-08-15 they were
# invisible: a trace sampled with one setting reloaded silently under the
# other. `light_travel_time` is the sharpest case because it defaults to ON.
# --------------------------------------------------------------------------

_LTT_BASE = {
    "transit": [{"name": "T1", "file": "a.dat", "band": "B"}],
    "star": [{"name": "A"}],
}


def _with_transit_key(**kw):
    cfg = copy.deepcopy(_LTT_BASE)
    cfg["transit"][0].update(kw)
    return cfg


@pytest.mark.parametrize(
    "key,value",
    [
        ("light_travel_time", False),
        ("gp", "rotation"),
        ("likelihood", "hogg"),
        ("mask", [3, 4]),
    ],
    ids=["light_travel_time", "gp", "likelihood", "mask"],
)
def test_model_selecting_file_keys_change_the_structural_hash(key, value):
    """
    Given a config that sets a per-file key selecting a different model,
    When its structural hash is compared with the config that omits it,
    Then the two differ, so a trace sampled under one setting cannot be
    silently reloaded under the other.
    """
    assert structural_hash(
        _with_transit_key(**{key: value})
    ) != structural_hash(_LTT_BASE)


def test_a_cosmetic_file_key_does_not_change_the_hash():
    """
    Given a purely cosmetic per-file key (plot styling),
    When the hash is compared against the config without it,
    Then it is unchanged -- the denylist in
    _NON_STRUCTURAL_INSTANCE_KEYS is the ONLY thing dropped.
    """
    assert structural_hash(
        _with_transit_key(plot={"color": "#123456"})
    ) == structural_hash(_LTT_BASE)


def test_a_model_affecting_numeric_key_now_changes_the_hash():
    """
    Given exptime, which sets the exposure-smearing window and so changes
    the model,
    When the hash is compared against the config without it,
    Then it differs.

    Under the old allowlist this was invisible, along with every other key
    nobody had thought to enumerate. The denylist inverts the failure
    mode: a forgotten cosmetic key costs an honest re-run, where a
    forgotten model key cost silently-reused foreign draws.
    """
    assert structural_hash(_with_transit_key(exptime=30.0)) != structural_hash(
        _LTT_BASE
    )


def test_instance_config_reaches_the_payload_so_a_mismatch_can_name_it():
    """
    Given two configs differing only in light_travel_time,
    When their structural payloads are compared,
    Then the difference is visible in the payload (not just in the hash),
    so trace_meta can tell the user WHAT changed.
    """
    on = structural_payload(_LTT_BASE)["components"]["transit"]
    off = structural_payload(_with_transit_key(light_travel_time=False))[
        "components"
    ]["transit"]

    assert on[0]["cfg"].get("light_travel_time") is None
    assert off[0]["cfg"]["light_travel_time"] is False
    assert on[0]["name"] == off[0]["name"] == "T1"


def test_file_paths_are_not_double_hashed_in_the_instance_config():
    """
    Given an instance whose data file is named by `file`,
    When its skeleton entry is inspected,
    Then `file` is absent from the per-instance config -- it is already
    hashed under the payload's own "files" key, and hashing it twice would
    be redundant rather than wrong.
    """
    entry = structural_payload(_LTT_BASE)["components"]["transit"][0]

    assert "file" not in entry["cfg"]
    assert "a.dat" in structural_payload(_LTT_BASE)["files"]

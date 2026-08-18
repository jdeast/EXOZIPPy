"""Tests for mkparam MAP-seeding logic."""

import copy

import numpy as np
import pytest
import xarray as xr

from exozippy.mkparam import write_param_file


def _make_idata(
    var_values: dict, lp: float = -10.0, tmpdir=None, derived_vars=None
):
    """Build a minimal ArviZ InferenceData with one chain, one draw.

    For every variable that does not end in ``_raw`` and is not listed in
    ``derived_vars``, a ``<name>_raw`` companion is automatically added.
    This mirrors real PyMC traces where every sampled variable has an
    unconstrained raw counterpart.

    Pass ``derived_vars`` (a set of names) for variables that represent
    Deterministic nodes and should NOT get a ``_raw`` companion — their
    absence of a raw counterpart is what signals "derived" to mkparam.
    """
    import arviz as az

    derived = set(derived_vars or [])

    # Auto-add _raw companions for sampled vars if not already provided
    full_values = dict(var_values)
    for name in list(var_values):
        raw_name = name + "_raw"
        if (
            not name.endswith("_raw")
            and name not in derived
            and raw_name not in var_values
        ):
            full_values[raw_name] = var_values[name] * 0.1

    data_vars = {}
    for name, val in full_values.items():
        arr = np.array([[val]], dtype=float)  # shape (chain=1, draw=1)
        data_vars[name] = xr.DataArray(arr, dims=["chain", "draw"])

    lp_arr = xr.DataArray(np.array([[lp]]), dims=["chain", "draw"])
    posterior_ds = xr.Dataset(data_vars)
    sample_stats_ds = xr.Dataset({"lp": lp_arr})

    idata = az.from_dict(
        {"posterior": posterior_ds, "sample_stats": sample_stats_ds}
    )

    trace_path = tmpdir / "trace.nc"
    idata.to_netcdf(str(trace_path))
    return trace_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_prior_writes_initval(tmp_path):
    """
    Given a sampled parameter with no existing mu/sigma,
    When mkparam runs,
    Then the output entry has initval=MAP and init_scale=std, not mu.
    """
    trace = _make_idata({"star.mass": 0.95}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": None,
        "star": [{"name": "Host"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    import yaml

    result = yaml.safe_load(open(out))
    entry = result["star.Host.mass"]

    assert "initval" in entry, "Expected initval for parameter without prior"
    assert "mu" not in entry, "Should not have mu when no prior exists"
    assert entry["initval"] == pytest.approx(0.95, abs=1e-6)


def test_with_explicit_mu_preserved(tmp_path):
    """
    Given a sampled parameter whose existing entry has an explicit mu+sigma,
    When mkparam runs,
    Then initval is set to the MAP, mu is preserved unchanged, sigma is preserved.
    The prior center (mu) must never drift toward the MAP.
    """
    import yaml

    existing_params = {"star.Host.teff": {"mu": 5800.0, "sigma": 100.0}}
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.teff": 5750.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    result = yaml.safe_load(open(out))
    entry = result["star.Host.teff"]

    assert entry["initval"] == pytest.approx(5750.0, abs=1e-6), (
        "initval must be MAP"
    )
    assert entry["mu"] == pytest.approx(5800.0, abs=1e-6), (
        "mu must stay at original prior center"
    )
    assert entry["sigma"] == pytest.approx(100.0, abs=1e-6)


def test_initval_sigma_promotes_mu(tmp_path):
    """
    Given an existing entry with initval+sigma but no explicit mu,
    When mkparam runs,
    Then the original initval is promoted to mu (preserving the prior center)
    and initval is updated to the MAP so the chain starts there next run.
    """
    import yaml

    existing_params = {"star.Host.teff": {"initval": 6207.0, "sigma": 100.0}}
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.teff": 6193.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    result = yaml.safe_load(open(out))
    entry = result["star.Host.teff"]

    assert entry["initval"] == pytest.approx(6193.0, abs=1e-6), (
        "initval must be MAP"
    )
    assert entry["mu"] == pytest.approx(6207.0, abs=1e-6), (
        "original initval promoted to mu"
    )
    assert entry["sigma"] == pytest.approx(100.0, abs=1e-6)


def test_fixed_sigma_zero_no_mu_promotion(tmp_path):
    """
    Given an existing entry with sigma=0 (fixed parameter),
    When mkparam runs,
    Then initval is updated to the MAP and mu is NOT added — sigma=0 means
    fixed, not a Gaussian prior, so the original initval is not a prior center.
    """
    import yaml

    existing_params = {"star.Host.radius": {"initval": 1.0, "sigma": 0.0}}
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.radius": 1.05}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    result = yaml.safe_load(open(out))
    entry = result["star.Host.radius"]

    assert entry["initval"] == pytest.approx(1.05, abs=1e-6), (
        "initval must be MAP"
    )
    assert "mu" not in entry, "sigma=0 is fixed, not a prior — must not add mu"
    assert entry["sigma"] == pytest.approx(0.0)


def test_non_sampled_initval_only_is_discarded(tmp_path):
    """
    Given an existing entry with only initval (no mu/sigma/upper/lower)
    for a parameter that was NOT sampled in the trace,
    When mkparam runs,
    Then that entry is absent from the output (stale guess, not a prior).
    """
    import yaml

    existing_params = {
        "lens.Lens.t_0": {"initval": 2456836.22},
        "lens.Lens.u_0": {"mu": 0.5},  # mu without sigma — not a prior
        "star.Lens.ra": {"initval": 266.8, "sigma": 0.0},
    }
    param_file = tmp_path / "ob.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    # Trace only contains star.mass — the lens and star.ra params are not sampled
    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/ob",
        "parameter_file": "ob.params.yaml",
        "star": [{"name": "Lens"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    result = yaml.safe_load(open(out))
    assert "lens.Lens.t_0" not in result, (
        "initval-only non-sampled entry should be dropped"
    )
    assert "lens.Lens.u_0" not in result, (
        "mu-only entry (no sigma) should be dropped"
    )
    assert "star.Lens.ra" in result, (
        "entry with sigma constraint should be kept"
    )


def test_non_sampled_with_upper_limit_is_kept(tmp_path):
    """
    Given an existing entry with only an upper limit and no other prior fields,
    When mkparam runs,
    Then the entry is preserved (the bound is a meaningful constraint).
    """
    import yaml

    existing_params = {
        "mulensinstrument.Spitzer.err_scale": {"upper": 1.1},
        "mulensinstrument.OGLE.err_scale": {"lower": 0.5},
    }
    param_file = tmp_path / "ob.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/ob",
        "parameter_file": "ob.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    result = yaml.safe_load(open(out))
    assert "mulensinstrument.Spitzer.err_scale" in result
    assert result["mulensinstrument.Spitzer.err_scale"][
        "upper"
    ] == pytest.approx(1.1)
    assert "mulensinstrument.OGLE.err_scale" in result
    assert result["mulensinstrument.OGLE.err_scale"]["lower"] == pytest.approx(
        0.5
    )


def test_output_filename_uses_dots(tmp_path):
    """
    Given parameter_file = "kelt4.params.yaml",
    When mkparam runs without an explicit output_path,
    Then the output file is named "kelt4.params.2.yaml" (dots, not underscores).
    """
    import yaml

    param_file = tmp_path / "kelt4.params.yaml"
    param_file.write_text("{}\n")

    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/kelt4",
        "parameter_file": "kelt4.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(config, base_dir=tmp_path, trace_path=trace)

    assert out.name == "kelt4.params.2.yaml"


def test_derived_parameter_excluded_from_output(tmp_path):
    """
    Given a trace containing orbit.logP (sampled, has orbit.logP_raw) and
      orbit.period (derived Deterministic, no _raw counterpart),
    When mkparam runs,
    Then orbit.logP IS written to the output but orbit.period is NOT.

    This is a regression test: the old code filtered on "_raw" not in v, so
    orbit.period (no _raw) was included, creating a redundant user-rank
    constraint that conflicted with orbit.logP and slowed down sampling.
    """
    import yaml

    # Trace has the sampled variable (orbit.logP, gets a _raw companion
    # automatically) and a derived Deterministic (orbit.period, listed in
    # derived_vars so it does NOT get a _raw companion).
    trace = _make_idata(
        {
            "orbit.logP": 0.47,
            "orbit.period": 2.989,  # derived Deterministic — no _raw companion
        },
        tmpdir=tmp_path,
        derived_vars={"orbit.period"},
    )
    config = {
        "prefix": "fitresults/model",
        "parameter_file": None,
        "orbit": [{"name": "b"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    result = yaml.safe_load(open(out))

    assert "orbit.b.logP" in result, "Sampled parameter must be written"
    assert "orbit.b.period" not in result, (
        "Derived Deterministic (no _raw) must be excluded — "
        "writing it creates a conflicting redundant constraint"
    )
    # Internal _raw variables must never appear in the output
    for key in result:
        assert not key.endswith("_raw"), (
            f"Raw variable leaked into output: {key}"
        )


def test_output_filename_increments(tmp_path):
    """
    Given parameter_file = "kelt4.params.2.yaml",
    When mkparam runs,
    Then the output is kelt4.params.3.yaml.
    """
    import yaml

    param_file = tmp_path / "kelt4.params.2.yaml"
    param_file.write_text("{}\n")

    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/kelt4",
        "parameter_file": "kelt4.params.2.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(config, base_dir=tmp_path, trace_path=trace)

    assert out.name == "kelt4.params.3.yaml"


def test_flat_dict_component_writes_two_part_key(tmp_path):
    """
    Regression: components with a flat-dict YAML config (e.g. sed: {path: ...})
    have no named instance list, so mkparam used to generate 'sed.0.errscale'
    (3-part key with a numeric index) instead of 'sed.errscale' (2-part key).
    When that generated file was fed back as parameter_file, standardize_param_names
    tried to enumerate the dict's string keys and crashed with
    AttributeError: 'str' object has no attribute 'get'.
    """
    import yaml

    # Config where 'sed' is a flat dict (not a list of named instances)
    config = {
        "prefix": "fitresults/hat3",
        "parameter_file": None,
        "sed": {"path": "*.sed.*", "file": "hat3_sed.yaml"},
    }

    trace = _make_idata({"sed.errscale": 0.75}, tmpdir=tmp_path)
    out = write_param_file(config, base_dir=tmp_path, trace_path=trace)

    with open(out) as f:
        written = yaml.safe_load(f)

    # mkparam must write 'sed.errscale' (2-part) NOT 'sed.0.errscale' (3-part)
    assert "sed.errscale" in written, (
        f"Expected 'sed.errscale' in output; got keys: {list(written)}"
    )
    assert not any(k.startswith("sed.0.") for k in written), (
        f"3-part indexed key found in output: {[k for k in written if k.startswith('sed.0.')]}"
    )


def test_non_sampled_constraint_gets_mu_promotion(tmp_path):
    """
    Regression: non-sampled constraint parameters (e.g. a Gaia parallax prior
    applied as a potential on the derived parallax / sampled distance) went
    through the pass-through path without mu-promotion.  On successive mkparam
    runs the prior center was carried only implicitly via initval; editing
    initval would silently shift the prior.

    After the fix, a pass-through entry with sigma != 0 and no existing mu
    gets mu = initval written explicitly so the prior center is pinned.
    """
    import yaml

    existing = tmp_path / "hat3.params.yaml"
    existing.write_text(
        "star.parallax:\n  initval: 7.45278\n  sigma: 0.01745\n"
    )

    config = {
        "prefix": "fitresults/hat3",
        "parameter_file": "hat3.params.yaml",
        "star": [{"name": "A"}],
    }
    # distance is sampled; parallax is derived → not in trace
    trace = _make_idata({"star.distance": 133.3}, tmpdir=tmp_path)

    out = write_param_file(config, base_dir=tmp_path, trace_path=trace)

    with open(out) as f:
        written = yaml.safe_load(f)

    # Find the parallax entry (may appear under star.parallax or star.A.parallax)
    parallax_entry = written.get("star.parallax") or written.get(
        "star.A.parallax"
    )
    assert parallax_entry is not None, (
        f"parallax key missing from output: {list(written)}"
    )
    assert "mu" in parallax_entry, (
        f"parallax entry has no 'mu' — prior center would drift on successive runs. "
        f"Got: {parallax_entry}"
    )
    assert np.isclose(parallax_entry["mu"], 7.45278, rtol=1e-6)
    assert np.isclose(parallax_entry["sigma"], 0.01745, rtol=1e-6)


def test_angle_entry_takes_map_initval_and_keeps_prior(tmp_path):
    """
    Given an existing lens.L.alpha entry carrying a Gaussian prior
      (initval + sigma, no explicit mu) and a trace whose xalpha/yalpha MAP
      direction is a different angle,
    When mkparam runs,
    Then the written alpha entry has the NEW MAP angle as initval, keeps
      sigma, and promotes the stale initval to mu (the prior center must not
      follow the MAP).

    Regression (review 1.17): alpha/bigomega are synthesized from the sampled
    x/y pair AFTER the main loop, so they were never in consumed_existing.
    The pass-through loop then overwrote the fresh MAP angle with the stale
    entry, so restart files violated the "initval at the trace MAP" contract.
    """
    import yaml

    existing_params = {"lens.L.alpha": {"initval": 12.0, "sigma": 5.0}}
    param_file = tmp_path / "ob.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    # arctan2(1, 0) = +90 deg
    trace = _make_idata(
        {"lens.xalpha": 0.0, "lens.yalpha": 1.0}, tmpdir=tmp_path
    )
    config = {
        "prefix": "fitresults/ob",
        "parameter_file": "ob.params.yaml",
        "lens": [{"name": "L"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    result = yaml.safe_load(open(out))
    entry = result["lens.L.alpha"]

    assert entry["initval"] == pytest.approx(90.0, abs=1e-6), (
        "alpha initval must be the trace MAP angle, not the stale entry"
    )
    assert entry["sigma"] == pytest.approx(5.0), "user prior must survive"
    assert entry["mu"] == pytest.approx(12.0, abs=1e-6), (
        "stale initval promoted to mu -- the prior center must not move"
    )
    assert not any(k.endswith(("xalpha", "yalpha")) for k in result)


def test_angle_entry_index_notation_merges_without_duplicate(tmp_path):
    """
    Given an existing orbit.0.bigomega entry written in index notation with an
      explicit mu + sigma, and a trace whose xbigomega/ybigomega MAP is a
      different angle,
    When mkparam runs,
    Then exactly one bigomega entry is written, under the name notation, with
      the MAP initval and the explicit mu/sigma untouched.
    """
    import yaml

    existing_params = {"orbit.0.bigomega": {"mu": 30.0, "sigma": 10.0}}
    param_file = tmp_path / "kelt4.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    # arctan2(0, -1) = 180 deg
    trace = _make_idata(
        {"orbit.xbigomega": -1.0, "orbit.ybigomega": 0.0}, tmpdir=tmp_path
    )
    config = {
        "prefix": "fitresults/kelt4",
        "parameter_file": "kelt4.params.yaml",
        "orbit": [{"name": "b"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    result = yaml.safe_load(open(out))

    assert "orbit.0.bigomega" not in result, (
        "stale index-notation entry must be consumed, not passed through"
    )
    entry = result["orbit.b.bigomega"]
    assert entry["initval"] == pytest.approx(180.0, abs=1e-6)
    assert entry["mu"] == pytest.approx(30.0, abs=1e-6), (
        "explicit mu must be preserved exactly"
    )
    assert entry["sigma"] == pytest.approx(10.0)


def test_angle_entry_keeps_bounds_and_adds_no_mu(tmp_path):
    """
    Given an existing lens.L.alpha entry carrying only bounds (a constraint,
      so the pass-through loop used to keep it and clobber the angle),
    When mkparam runs,
    Then the bounds survive, initval is the MAP angle, and no mu is invented
      (bounds are not a Gaussian prior center).
    """
    import yaml

    existing_params = {"lens.L.alpha": {"lower": -180.0, "upper": 180.0}}
    param_file = tmp_path / "ob.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    # arctan2(-1, 0) = -90 deg
    trace = _make_idata(
        {"lens.xalpha": 0.0, "lens.yalpha": -1.0}, tmpdir=tmp_path
    )
    config = {
        "prefix": "fitresults/ob",
        "parameter_file": "ob.params.yaml",
        "lens": [{"name": "L"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    result = yaml.safe_load(open(out))
    entry = result["lens.L.alpha"]

    assert entry["initval"] == pytest.approx(-90.0, abs=1e-6)
    assert entry["lower"] == pytest.approx(-180.0)
    assert entry["upper"] == pytest.approx(180.0)
    assert "mu" not in entry, "bounds are not a prior center"


def test_uncentered_sigma_in_params_file_is_fatal(tmp_path):
    """
    Given an existing params file with sigma > 0 and no mu/initval,
    When mkparam runs,
    Then it raises and names the offending file.

    mkparam reads the params file directly (bypassing ConfigManager), and its
    pass-through loop copies constraint-bearing entries verbatim -- so without
    this check it would launder an uncentered Gaussian prior into the restart
    file, where the prior ends up centered on a data-derived start value.
    """
    import yaml

    existing_params = {"star.Host.teff": {"sigma": 100.0}}
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    with pytest.raises(ValueError) as exc:
        write_param_file(
            config,
            base_dir=tmp_path,
            trace_path=trace,
            output_path=tmp_path / "out.yaml",
        )

    msg = str(exc.value)
    assert "star.Host.teff" in msg
    assert "star.params.yaml" in msg, "the input file must be named"


def test_standardize_param_names_flat_dict_component_no_crash(tmp_path):
    """
    Regression: standardize_param_names crashed with AttributeError when a
    3-part key (e.g. 'sed.0.errscale') referenced a component whose config
    block is a flat dict rather than a list of named instances.
    The dict was enumerated, yielding string keys, and .get() was called on
    a string instead of a dict.
    """
    from exozippy.config import ConfigManager

    user_params = {"sed.0.errscale": {"initval": 0.75}}
    system_config = {
        "sed": {"path": "*.sed.*", "file": "hat3_sed.yaml"},
        "star": [{"name": "A"}],
    }

    # Must not raise AttributeError
    result = ConfigManager.standardize_param_names(user_params, system_config)

    # The key should pass through unchanged (no list to look up names in)
    assert "sed.0.errscale" in result


# ---------------------------------------------------------------------------
# In-memory user_params (run_fit(config, user_params=<dict>))
# ---------------------------------------------------------------------------


def test_in_memory_user_params_beat_the_file_on_disk(tmp_path):
    """
    Given a fit driven by an in-memory user_params dict while a DIFFERENT
    params file happens to sit at config['parameter_file'],
    When mkparam is given those in-memory params,
    Then the restart file carries the priors that were actually fitted and
    none of the on-disk file's.

    The two differ only in the magnitudes of mu/sigma, which
    evaluator.structural_hash deliberately does not cover -- so nothing else
    in the pipeline would have caught the substitution.
    """
    import yaml

    stale_on_disk = {"star.Host.teff": {"mu": 4000.0, "sigma": 500.0}}
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(stale_on_disk, f)

    fitted = {"star.Host.teff": {"mu": 5800.0, "sigma": 100.0}}

    trace = _make_idata({"star.teff": 5750.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
        user_params=fitted,
    )

    entry = yaml.safe_load(open(out))["star.Host.teff"]
    assert entry["mu"] == pytest.approx(5800.0), (
        "prior center must be the one fitted"
    )
    assert entry["sigma"] == pytest.approx(100.0)
    assert entry["initval"] == pytest.approx(5750.0, abs=1e-6)


def test_in_memory_user_params_are_not_mutated(tmp_path):
    """
    Given an in-memory user_params dict,
    When mkparam consumes it,
    Then the caller's dict is untouched -- run_fit hands over the same object
    the live System was built from.
    """
    fitted = {"star.Host.teff": {"mu": 5800.0, "sigma": 100.0}}
    before = copy.deepcopy(fitted)

    trace = _make_idata({"star.teff": 5750.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": None,
        "star": [{"name": "Host"}],
    }

    write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
        user_params=fitted,
    )

    assert fitted == before


def test_in_memory_user_params_still_validated(tmp_path):
    """
    Given in-memory params with a centerless Gaussian prior (sigma, no
    mu/initval),
    When mkparam consumes them,
    Then the same fatal check the on-disk path runs fires, naming the source.
    """
    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": None,
        "star": [{"name": "Host"}],
    }

    with pytest.raises(ValueError) as exc:
        write_param_file(
            config,
            base_dir=tmp_path,
            trace_path=trace,
            output_path=tmp_path / "out.yaml",
            user_params={"star.Host.teff": {"sigma": 100.0}},
        )

    assert "star.Host.teff" in str(exc.value)
    assert "in-memory" in str(exc.value)


# --- 1.3.2: a linked sigma is legal and must not crash the writer -----------


def test_a_linked_sigma_is_copied_across_without_crashing(tmp_path):
    """
    Given a params entry whose sigma is a LINK expression (legal --
      linking.LINKABLE_FIELDS includes sigma),
    When mkparam runs,
    Then the entry is written with its sigma verbatim and no mu promotion.

    float() on the link string used to raise ValueError, which under run_fit
    silently skipped the restart file at the end of a multi-day fit and under
    the standalone CLI died raw.  There is nothing here that could evaluate
    the link to decide whether it is zero, so the promotion is skipped -- the
    stated constraint itself still survives the round trip.
    """
    import yaml

    existing_params = {
        "star.Host.teff": {"initval": 6207.0, "sigma": "star.Host.teff_err"}
    }
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.teff": 6193.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    entry = yaml.safe_load(open(out))["star.Host.teff"]
    assert entry["sigma"] == "star.Host.teff_err"
    assert entry["initval"] == pytest.approx(6193.0, abs=1e-6)
    assert "mu" not in entry


def test_a_linked_sigma_on_a_non_sampled_entry_is_passed_through(tmp_path):
    """
    Given the same linked sigma on a parameter the trace never sampled,
    When mkparam runs,
    Then the pass-through loop copies it instead of raising.

    The second, independent float(sigma) call -- the one in the
    passthrough loop -- had the same crash.
    """
    import yaml

    existing_params = {
        "star.Host.av": {"initval": 0.1, "sigma": "star.Other.av_err"}
    }
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.teff": 6193.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    entry = yaml.safe_load(open(out))["star.Host.av"]
    assert entry["sigma"] == "star.Other.av_err"
    assert "mu" not in entry


# --- 7.3.1: the config-as-path branch (only the CLI exercises it) ----------


def test_config_given_as_a_path_anchors_base_dir_at_its_parent(tmp_path):
    """
    Given a config passed as a PATH rather than a dict,
    When mkparam runs with no base_dir,
    Then base_dir is the config's own directory -- so the params file, the
      trace and the output all resolve next to the config.

    Every other test in this file passes a dict; only `scripts/mkparam.py`
    takes this branch, and base_dir is what anchors the params read and the
    output path, so getting it wrong writes the restart file into the CWD
    and silently drops the previous file's priors.
    """
    import yaml

    workdir = tmp_path / "fit"
    workdir.mkdir()
    (workdir / "kelt4.params.yaml").write_text(
        yaml.safe_dump({"star.Host.teff": {"mu": 5800.0, "sigma": 100.0}})
    )
    trace = _make_idata({"star.teff": 5750.0}, tmpdir=workdir)
    config_path = workdir / "kelt4.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "prefix": "fitresults/kelt4",
                "parameter_file": "kelt4.params.yaml",
                "star": [{"name": "Host"}],
            }
        )
    )

    # The CWD is deliberately NOT the config's directory: that is the whole
    # thing base_dir has to get right.
    out = write_param_file(str(config_path), trace_path=trace)

    assert out.parent == workdir
    assert out.name == "kelt4.params.2.yaml"
    entry = yaml.safe_load(out.read_text())["star.Host.teff"]
    # The previous file's prior was found and carried across verbatim...
    assert entry["mu"] == pytest.approx(5800.0, abs=1e-6)
    assert entry["sigma"] == pytest.approx(100.0, abs=1e-6)
    # ...while the start value moved to the MAP.
    assert entry["initval"] == pytest.approx(5750.0, abs=1e-6)


def test_config_given_as_a_Path_object_works_too(tmp_path):
    """
    Given the same config passed as a pathlib.Path,
    When mkparam runs,
    Then it behaves identically -- the branch tests (str, Path), and the CLI
      may hand over either.
    """
    import yaml

    (tmp_path / "kelt4.params.yaml").write_text("{}\n")
    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config_path = tmp_path / "kelt4.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "prefix": "fitresults/kelt4",
                "parameter_file": "kelt4.params.yaml",
                "star": [{"name": "Host"}],
            }
        )
    )

    out = write_param_file(config_path, trace_path=trace)

    assert out == tmp_path / "kelt4.params.2.yaml"
    assert "star.Host.mass" in yaml.safe_load(out.read_text())


def test_a_config_path_beats_an_explicit_base_dir(tmp_path):
    """
    Given a config path AND a base_dir argument,
    When mkparam runs,
    Then the config's own directory wins.

    Pins the precedence rather than assuming it: the branch overwrites
    base_dir unconditionally, and a reader could reasonably expect the
    explicit argument to win.  Passing both is a caller error either way;
    what matters is that the behavior is stated somewhere.
    """
    import yaml

    workdir = tmp_path / "fit"
    workdir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (workdir / "kelt4.params.yaml").write_text("{}\n")
    trace = _make_idata({"star.mass": 1.0}, tmpdir=workdir)
    config_path = workdir / "kelt4.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "prefix": "fitresults/kelt4",
                "parameter_file": "kelt4.params.yaml",
                "star": [{"name": "Host"}],
            }
        )
    )

    out = write_param_file(
        str(config_path), base_dir=elsewhere, trace_path=trace
    )

    assert out.parent == workdir


# --- 1.3.3: auto-versioning must never destroy an existing restart file -----


def test_output_version_skips_a_name_already_on_disk(tmp_path):
    """
    Given kelt4.params.yaml AND an existing kelt4.params.2.yaml,
    When mkparam runs a second time from the same input,
    Then it writes kelt4.params.3.yaml and leaves the 2 file untouched.

    Bumping the version once protects only the INPUT; two fits started from
    the same params file both resolved to .2.yaml and the second destroyed
    the first's restart file.  Bit us in practice (2026-08).
    """
    (tmp_path / "kelt4.params.yaml").write_text("{}\n")
    already_there = tmp_path / "kelt4.params.2.yaml"
    already_there.write_text("# an earlier run's restart file\n")

    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/kelt4",
        "parameter_file": "kelt4.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(config, base_dir=tmp_path, trace_path=trace)

    assert out.name == "kelt4.params.3.yaml"
    assert already_there.read_text() == "# an earlier run's restart file\n"


def test_bare_scalar_entries_are_discarded_like_initval_only_dicts(tmp_path):
    """
    Given a non-sampled entry written bare (`star.teff: 5800`) and its dict
      equivalent (`{initval: 5800}`),
    When mkparam runs,
    Then BOTH are discarded -- the "initval-only entries are stale guesses"
      policy is about content, not spelling.

    The bare form is not a dict, so it fell through both branches of the
    passthrough guard and was re-emitted into every successive restart file
    forever.
    """
    import yaml

    existing_params = {
        "star.Host.teff": 5800.0,
        "star.Host.av": [0.1, 0.2],  # bare list = per-seed starts
        "star.Host.feh": {"initval": 0.0},
        "star.Host.distance": {"initval": 100.0, "sigma": 5.0},
    }
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    result = yaml.safe_load(open(out))
    assert "star.Host.teff" not in result
    assert "star.Host.av" not in result
    assert "star.Host.feh" not in result
    # ...and the one that really does state a constraint still survives.
    assert result["star.Host.distance"]["sigma"] == pytest.approx(5.0)


def test_mu_promotion_takes_seed_0_of_a_multi_seed_initval(tmp_path):
    """
    Given a previous restart file's length-K initval list with a hand-added
      sigma and no mu,
    When mkparam promotes the implicit prior center,
    Then mu is seed 0, not the whole list.

    A prior center is one number.  The list is a set of PER-SEED STARTS, so
    promoting it wholesale hands the next fit a Gaussian potential with a
    vector center nobody wrote.  Seed 0 is canonical everywhere else --
    run._user_initval reads a list initval as v[0], and this writer's own
    bounds already come from seed 0.
    """
    import yaml

    existing_params = {
        "star.Host.teff": {"initval": [6207.0, 6180.0, 6230.0], "sigma": 100.0}
    }
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.teff": 6193.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    entry = yaml.safe_load(open(out))["star.Host.teff"]
    assert entry["mu"] == pytest.approx(6207.0, abs=1e-6)
    assert not isinstance(entry["mu"], list)


def test_mu_promotion_of_a_multi_seed_initval_on_a_non_sampled_entry(tmp_path):
    """
    Given the same multi-seed entry on a parameter the trace never sampled,
    When mkparam runs,
    Then the passthrough loop promotes seed 0 too -- the second, independent
      copy of the promotion had the same bug.
    """
    import yaml

    existing_params = {
        "star.Host.distance": {"initval": [100.0, 102.0], "sigma": 5.0}
    }
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(
        config,
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )

    entry = yaml.safe_load(open(out))["star.Host.distance"]
    assert entry["mu"] == pytest.approx(100.0, abs=1e-6)


def test_output_version_skips_a_run_of_existing_versions(tmp_path):
    """
    Given versions 2 through 5 already on disk,
    When mkparam runs,
    Then it lands on 6 -- the loop is bounded by the filesystem, not a cap.
    """
    (tmp_path / "kelt4.params.yaml").write_text("{}\n")
    for n in range(2, 6):
        (tmp_path / f"kelt4.params.{n}.yaml").write_text("{}\n")

    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/kelt4",
        "parameter_file": "kelt4.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = write_param_file(config, base_dir=tmp_path, trace_path=trace)

    assert out.name == "kelt4.params.6.yaml"


def test_output_version_loops_when_the_config_names_no_parameter_file(
    tmp_path,
):
    """
    Given a config with no parameter_file (legal: a blind fit seeds itself)
      and an existing <runname>.params.2.yaml,
    When mkparam runs,
    Then it writes .3.yaml instead of overwriting.

    That branch used to hardcode ".params.2.yaml", which is exactly the case
    where several runs share one directory.
    """
    already_there = tmp_path / "kelt4.params.2.yaml"
    already_there.write_text("# an earlier run's restart file\n")

    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {"prefix": "fitresults/kelt4", "star": [{"name": "Host"}]}

    out = write_param_file(config, base_dir=tmp_path, trace_path=trace)

    assert out.name == "kelt4.params.3.yaml"
    assert already_there.read_text() == "# an earlier run's restart file\n"

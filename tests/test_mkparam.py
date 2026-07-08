"""Tests for mkparam MAP-seeding logic."""

import numpy as np
import pytest
import xarray as xr

from exozippy.mkparam import mkprior


def _make_idata(var_values: dict, lp: float = -10.0, tmpdir=None,
                derived_vars=None):
    """Build a minimal ArviZ InferenceData with one chain, one draw.

    For every variable that does not end in ``_raw`` and is not listed in
    ``derived_vars``, a ``<name>_raw`` companion is automatically added.
    This mirrors real PyMC traces where every sampled variable has an
    unconstrained raw counterpart.

    Pass ``derived_vars`` (a set of names) for variables that represent
    Deterministic nodes and should NOT get a ``_raw`` companion — their
    absence of a raw counterpart is what signals "derived" to mkprior.
    """
    import arviz as az

    derived = set(derived_vars or [])

    # Auto-add _raw companions for sampled vars if not already provided
    full_values = dict(var_values)
    for name in list(var_values):
        raw_name = name + "_raw"
        if (not name.endswith("_raw")
                and name not in derived
                and raw_name not in var_values):
            full_values[raw_name] = var_values[name] * 0.1

    data_vars = {}
    for name, val in full_values.items():
        arr = np.array([[val]], dtype=float)  # shape (chain=1, draw=1)
        data_vars[name] = xr.DataArray(arr, dims=["chain", "draw"])

    lp_arr = xr.DataArray(np.array([[lp]]), dims=["chain", "draw"])
    posterior_ds = xr.Dataset(data_vars)
    sample_stats_ds = xr.Dataset({"lp": lp_arr})

    idata = az.from_dict({"posterior": posterior_ds, "sample_stats": sample_stats_ds})

    trace_path = tmpdir / "trace.nc"
    idata.to_netcdf(str(trace_path))
    return trace_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_prior_writes_initval(tmp_path):
    """
    Given a sampled parameter with no existing mu/sigma,
    When mkprior runs,
    Then the output entry has initval=MAP and init_scale=std, not mu.
    """
    trace = _make_idata({"star.mass": 0.95}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": None,
        "star": [{"name": "Host"}],
    }

    out = mkprior(config, base_dir=tmp_path, trace_path=trace,
                  output_path=tmp_path / "out.yaml")

    import yaml
    result = yaml.safe_load(open(out))
    entry = result["star.Host.mass"]

    assert "initval" in entry, "Expected initval for parameter without prior"
    assert "mu" not in entry, "Should not have mu when no prior exists"
    assert entry["initval"] == pytest.approx(0.95, abs=1e-6)


def test_with_explicit_mu_preserved(tmp_path):
    """
    Given a sampled parameter whose existing entry has an explicit mu+sigma,
    When mkprior runs,
    Then initval is set to the MAP, mu is preserved unchanged, sigma is preserved.
    The prior center (mu) must never drift toward the MAP.
    """
    import yaml

    existing_params = {
        "star.Host.teff": {"mu": 5800.0, "sigma": 100.0}
    }
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.teff": 5750.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = mkprior(config, base_dir=tmp_path, trace_path=trace,
                  output_path=tmp_path / "out.yaml")

    result = yaml.safe_load(open(out))
    entry = result["star.Host.teff"]

    assert entry["initval"] == pytest.approx(5750.0, abs=1e-6), "initval must be MAP"
    assert entry["mu"] == pytest.approx(5800.0, abs=1e-6), "mu must stay at original prior center"
    assert entry["sigma"] == pytest.approx(100.0, abs=1e-6)


def test_initval_sigma_promotes_mu(tmp_path):
    """
    Given an existing entry with initval+sigma but no explicit mu,
    When mkprior runs,
    Then the original initval is promoted to mu (preserving the prior center)
    and initval is updated to the MAP so the chain starts there next run.
    """
    import yaml

    existing_params = {
        "star.Host.teff": {"initval": 6207.0, "sigma": 100.0}
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

    out = mkprior(config, base_dir=tmp_path, trace_path=trace,
                  output_path=tmp_path / "out.yaml")

    result = yaml.safe_load(open(out))
    entry = result["star.Host.teff"]

    assert entry["initval"] == pytest.approx(6193.0, abs=1e-6), "initval must be MAP"
    assert entry["mu"] == pytest.approx(6207.0, abs=1e-6), "original initval promoted to mu"
    assert entry["sigma"] == pytest.approx(100.0, abs=1e-6)


def test_fixed_sigma_zero_no_mu_promotion(tmp_path):
    """
    Given an existing entry with sigma=0 (fixed parameter),
    When mkprior runs,
    Then initval is updated to the MAP and mu is NOT added — sigma=0 means
    fixed, not a Gaussian prior, so the original initval is not a prior center.
    """
    import yaml

    existing_params = {
        "star.Host.radius": {"initval": 1.0, "sigma": 0.0}
    }
    param_file = tmp_path / "star.params.yaml"
    with open(param_file, "w") as f:
        yaml.dump(existing_params, f)

    trace = _make_idata({"star.radius": 1.05}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/model",
        "parameter_file": "star.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = mkprior(config, base_dir=tmp_path, trace_path=trace,
                  output_path=tmp_path / "out.yaml")

    result = yaml.safe_load(open(out))
    entry = result["star.Host.radius"]

    assert entry["initval"] == pytest.approx(1.05, abs=1e-6), "initval must be MAP"
    assert "mu" not in entry, "sigma=0 is fixed, not a prior — must not add mu"
    assert entry["sigma"] == pytest.approx(0.0)


def test_non_sampled_initval_only_is_discarded(tmp_path):
    """
    Given an existing entry with only initval (no mu/sigma/upper/lower)
    for a parameter that was NOT sampled in the trace,
    When mkprior runs,
    Then that entry is absent from the output (stale guess, not a prior).
    """
    import yaml

    existing_params = {
        "lens.Lens.t_0": {"initval": 2456836.22},
        "lens.Lens.u_0": {"mu": 0.5},        # mu without sigma — not a prior
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

    out = mkprior(config, base_dir=tmp_path, trace_path=trace,
                  output_path=tmp_path / "out.yaml")

    result = yaml.safe_load(open(out))
    assert "lens.Lens.t_0" not in result, "initval-only non-sampled entry should be dropped"
    assert "lens.Lens.u_0" not in result, "mu-only entry (no sigma) should be dropped"
    assert "star.Lens.ra" in result, "entry with sigma constraint should be kept"


def test_non_sampled_with_upper_limit_is_kept(tmp_path):
    """
    Given an existing entry with only an upper limit and no other prior fields,
    When mkprior runs,
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

    out = mkprior(config, base_dir=tmp_path, trace_path=trace,
                  output_path=tmp_path / "out.yaml")

    result = yaml.safe_load(open(out))
    assert "mulensinstrument.Spitzer.err_scale" in result
    assert result["mulensinstrument.Spitzer.err_scale"]["upper"] == pytest.approx(1.1)
    assert "mulensinstrument.OGLE.err_scale" in result
    assert result["mulensinstrument.OGLE.err_scale"]["lower"] == pytest.approx(0.5)


def test_output_filename_uses_dots(tmp_path):
    """
    Given parameter_file = "kelt4.params.yaml",
    When mkprior runs without an explicit output_path,
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

    out = mkprior(config, base_dir=tmp_path, trace_path=trace)

    assert out.name == "kelt4.params.2.yaml"


def test_derived_parameter_excluded_from_output(tmp_path):
    """
    Given a trace containing orbit.logP (sampled, has orbit.logP_raw) and
      orbit.period (derived Deterministic, no _raw counterpart),
    When mkprior runs,
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
            "orbit.period": 2.989,    # derived Deterministic — no _raw companion
        },
        tmpdir=tmp_path,
        derived_vars={"orbit.period"},
    )
    config = {
        "prefix": "fitresults/model",
        "parameter_file": None,
        "orbit": [{"name": "b"}],
    }

    out = mkprior(config, base_dir=tmp_path, trace_path=trace,
                  output_path=tmp_path / "out.yaml")

    result = yaml.safe_load(open(out))

    assert "orbit.b.logP" in result, "Sampled parameter must be written"
    assert "orbit.b.period" not in result, (
        "Derived Deterministic (no _raw) must be excluded — "
        "writing it creates a conflicting redundant constraint"
    )
    # Internal _raw variables must never appear in the output
    for key in result:
        assert not key.endswith("_raw"), f"Raw variable leaked into output: {key}"


def test_output_filename_increments(tmp_path):
    """
    Given parameter_file = "kelt4.params.2.yaml",
    When mkprior runs,
    Then the output is kelt4.params.3.yaml (existing 3.yaml would be overwritten).
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

    out = mkprior(config, base_dir=tmp_path, trace_path=trace)

    assert out.name == "kelt4.params.3.yaml"


def test_flat_dict_component_writes_two_part_key(tmp_path):
    """
    Regression: components with a flat-dict YAML config (e.g. sed: {path: ...})
    have no named instance list, so mkprior used to generate 'sed.0.errscale'
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
    out = mkprior(config, base_dir=tmp_path, trace_path=trace)

    with open(out) as f:
        written = yaml.safe_load(f)

    # mkprior must write 'sed.errscale' (2-part) NOT 'sed.0.errscale' (3-part)
    assert "sed.errscale" in written, f"Expected 'sed.errscale' in output; got keys: {list(written)}"
    assert not any(k.startswith("sed.0.") for k in written), (
        f"3-part indexed key found in output: {[k for k in written if k.startswith('sed.0.')]}"
    )


def test_non_sampled_constraint_gets_mu_promotion(tmp_path):
    """
    Regression: non-sampled constraint parameters (e.g. a Gaia parallax prior
    applied as a potential on the derived parallax / sampled distance) went
    through the pass-through path without mu-promotion.  On successive mkprior
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

    out = mkprior(config, base_dir=tmp_path, trace_path=trace)

    with open(out) as f:
        written = yaml.safe_load(f)

    # Find the parallax entry (may appear under star.parallax or star.A.parallax)
    parallax_entry = written.get("star.parallax") or written.get("star.A.parallax")
    assert parallax_entry is not None, f"parallax key missing from output: {list(written)}"
    assert "mu" in parallax_entry, (
        f"parallax entry has no 'mu' — prior center would drift on successive runs. "
        f"Got: {parallax_entry}"
    )
    assert np.isclose(parallax_entry["mu"], 7.45278, rtol=1e-6)
    assert np.isclose(parallax_entry["sigma"], 0.01745, rtol=1e-6)


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

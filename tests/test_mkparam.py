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

    idata = az.InferenceData(posterior=posterior_ds, sample_stats=sample_stats_ds)

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


def test_with_prior_writes_mu(tmp_path):
    """
    Given a sampled parameter whose existing entry has mu+sigma,
    When mkprior runs,
    Then the output entry promotes MAP to mu and preserves sigma.
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

    assert "mu" in entry, "Expected mu for parameter with existing prior"
    assert "initval" not in entry, "Should not write initval when prior exists"
    assert entry["mu"] == pytest.approx(5750.0, abs=1e-6)
    assert entry["sigma"] == pytest.approx(100.0, abs=1e-6)


def test_with_sigma_only_writes_mu(tmp_path):
    """
    Given an existing entry with only sigma (fixed param, sigma=0),
    When mkprior runs,
    Then the entry is treated as having a prior and MAP goes to mu.
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

    assert "mu" in entry
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
    Given kelt4.params.2.yaml already exists,
    When mkprior runs again,
    Then the output is kelt4.params.3.yaml.
    """
    import yaml

    param_file = tmp_path / "kelt4.params.yaml"
    param_file.write_text("{}\n")
    (tmp_path / "kelt4.params.2.yaml").write_text("{}\n")

    trace = _make_idata({"star.mass": 1.0}, tmpdir=tmp_path)
    config = {
        "prefix": "fitresults/kelt4",
        "parameter_file": "kelt4.params.yaml",
        "star": [{"name": "Host"}],
    }

    out = mkprior(config, base_dir=tmp_path, trace_path=trace)

    assert out.name == "kelt4.params.3.yaml"

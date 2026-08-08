"""Tests for mkprior's multi-seed emission (notes/todo.txt #3).

With n_seeds > 1, mkprior writes list-valued initvals -- K mutually-consistent
JOINT posterior draws -- that the next run consumes as P4 multi-seed starts, so
walkers begin spread across the posterior covariance instead of clustered at a
single point. seed 0 stays the MAP; bounds stay scalar (seed 0).
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

az = pytest.importorskip("arviz")

from exozippy.mkparam import mkprior


def _make_trace(tmp_path, nchain=4, ndraw=400, seed=0):
    """Synthetic trace: a scalar lens param, a 2-star vector param, an
    xalpha/yalpha direction pair, each with a _raw counterpart, plus lp."""
    rng = np.random.default_rng(seed)

    def pair(shape):
        raw = rng.standard_normal(shape)
        return raw

    post = {
        "lens.t_0": 2000.0 + pair((nchain, ndraw)),
        "lens.t_0_raw": pair((nchain, ndraw)),
        "star.mass": 1.0 + 0.1 * pair((nchain, ndraw, 2)),
        "star.mass_raw": pair((nchain, ndraw, 2)),
        "lens.xalpha": pair((nchain, ndraw)),
        "lens.xalpha_raw": pair((nchain, ndraw)),
        "lens.yalpha": pair((nchain, ndraw)),
        "lens.yalpha_raw": pair((nchain, ndraw)),
    }
    lp = -0.5 * (post["lens.t_0_raw"] ** 2)
    idata = az.from_dict({"posterior": post, "sample_stats": {"lp": lp}})
    trace_path = tmp_path / "run_trace.nc"
    idata.to_netcdf(str(trace_path))
    return trace_path


def _config():
    return {
        "prefix": "run",
        "star": [{"name": "A"}, {"name": "B"}],
        "lens": [{"name": "L"}],
    }


def test_single_seed_emits_scalars(tmp_path):
    # Given a trace and n_seeds=1 (the default)
    trace = _make_trace(tmp_path)
    out = tmp_path / "out.params.yaml"
    # When mkprior runs
    mkprior(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=out,
        n_seeds=1,
    )
    params = yaml.safe_load(out.read_text())
    # Then every sampled initval is a plain scalar (legacy behavior)
    assert isinstance(params["lens.L.t_0"]["initval"], float)
    assert isinstance(params["star.A.mass"]["initval"], float)


def test_multi_seed_emits_length_k_lists(tmp_path):
    # Given a trace and n_seeds=3
    trace = _make_trace(tmp_path)
    out = tmp_path / "out.params.yaml"
    # When mkprior runs
    mkprior(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=out,
        n_seeds=3,
    )
    params = yaml.safe_load(out.read_text())
    # Then each sampled initval is a length-3 list (and no obsolete
    # init_scale is written -- whitening scales are measured at startup)
    for key in ("lens.L.t_0", "star.A.mass", "star.B.mass"):
        iv = params[key]["initval"]
        assert isinstance(iv, list) and len(iv) == 3, key
        assert "init_scale" not in params[key], key


def test_multi_seed_lists_share_one_length(tmp_path):
    # Given a multi-seed emission (config._build_seed_overrides requires all
    # initval lists in a file to share one length K, or be length 1)
    trace = _make_trace(tmp_path)
    out = tmp_path / "out.params.yaml"
    # When mkprior runs with n_seeds=3
    mkprior(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=out,
        n_seeds=3,
    )
    params = yaml.safe_load(out.read_text())
    # Then every list-valued initval has length exactly 3
    lengths = {
        len(v["initval"])
        for v in params.values()
        if isinstance(v.get("initval"), list)
    }
    assert lengths == {3}


def test_multi_seed_converts_direction_pair_to_angle_list(tmp_path):
    # Given xalpha/yalpha in the trace
    trace = _make_trace(tmp_path)
    out = tmp_path / "out.params.yaml"
    # When mkprior runs with n_seeds=3
    mkprior(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=out,
        n_seeds=3,
    )
    params = yaml.safe_load(out.read_text())
    # Then the pair collapses to a length-3 alpha angle list (no x/y written)
    assert "lens.L.alpha" in params
    assert isinstance(params["lens.L.alpha"]["initval"], list)
    assert len(params["lens.L.alpha"]["initval"]) == 3
    assert not any(k.endswith(("xalpha", "yalpha")) for k in params)


def test_multi_seed_seed0_is_map(tmp_path):
    # Given a trace whose MAP (max lp) is a specific joint draw
    trace = _make_trace(tmp_path)
    idata = az.from_netcdf(str(trace))
    lp = idata.sample_stats["lp"].values
    mc, md = np.unravel_index(np.argmax(lp), lp.shape)
    map_t0 = float(idata.posterior["lens.t_0"].values[mc, md])
    out = tmp_path / "out.params.yaml"
    # When mkprior emits multiple seeds
    mkprior(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=out,
        n_seeds=4,
    )
    params = yaml.safe_load(out.read_text())
    # Then seed 0 of the list is exactly the MAP value
    assert params["lens.L.t_0"]["initval"][0] == pytest.approx(
        map_t0, abs=1e-6
    )


def _make_bimodal_trace(tmp_path, nchain=6, ndraw=400, seed=0):
    """Trace where chains 0-4 sit in one tight mode and chain 5 in another,
    displaced far beyond the raw-z threshold, with HIGHER lp (the DC2018
    event 128 topology). Without mode stratification, good_chain_mask
    drops the five majority chains (their lp never reaches the best
    chain's median) or the fallback pools by occupancy."""
    rng = np.random.default_rng(seed)
    t0_raw = 0.001 * rng.standard_normal((nchain, ndraw))
    t0_raw[5] += 8.0  # minority mode, ~8000 robust-z away
    lp = 1000.0 + 3.0 * rng.standard_normal((nchain, ndraw))
    lp[5] += 500.0  # minority mode fits better
    post = {
        "lens.t_0": 2000.0 + t0_raw,
        "lens.t_0_raw": t0_raw,
    }
    idata = az.from_dict({"posterior": post, "sample_stats": {"lp": lp}})
    trace_path = tmp_path / "run_trace.nc"
    idata.to_netcdf(str(trace_path))
    return trace_path


def test_multi_seed_stratifies_across_modes(tmp_path):
    """
    Given a bimodal trace (5 chains in one basin, 1 chain in a displaced,
      better-lp basin),
    When mkprior emits 8 seeds,
    Then the initval list contains draws from BOTH basins -- a restart can
      never launder a multimodal posterior into a single-basin seed set.
    """
    trace_path = _make_bimodal_trace(tmp_path)
    out = mkprior(
        {"prefix": "run", "lens": [{"name": "L"}]},
        base_dir=tmp_path,
        trace_path=trace_path,
        n_seeds=8,
    )
    params = yaml.safe_load(Path(out).read_text())

    t0_seeds = np.asarray(params["lens.L.t_0"]["initval"], dtype=float)
    assert len(t0_seeds) == 8
    in_minority = t0_seeds > 2004.0  # displaced basin sits at ~2008
    in_majority = t0_seeds < 2004.0
    assert in_minority.any(), f"no minority-mode seeds: {t0_seeds}"
    assert in_majority.any(), f"no majority-mode seeds: {t0_seeds}"
    # seed 0 is the global MAP, which lives in the better-lp minority basin
    assert t0_seeds[0] > 2004.0

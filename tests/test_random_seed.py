"""A fit must be reproducible from its own output (review 2.14.4b).

`run.py` passed no `random_seed` to anything -- not `pm.sample`, not
`sample_jax_nuts`, not nutpie, and not the `seed=` argument all four in-house
samplers already accepted -- so a user could not reproduce their own fit.

The design, and what these tests pin:

* `sampler: {seed: N}` reaches every sampler run.py can dispatch to;
* when the key is ABSENT a seed is DRAWN, not left unset and not hardcoded --
  a fixed default would be strictly worse than none, correlating every user's
  chains while looking responsible;
* the drawn seed is stamped on the trace and echoed into the restart file's
  header, so the run that actually happened stays reproducible after the fact;
* the seed is a COMMENT in the params file, never a key -- that file is
  parameter overrides and `sampler:` belongs in the config.

The forwarding test is static, like tests/test_sampler_kwarg_plumbing.py: the
failure being guarded is "a branch nobody exercised in the suite quietly stops
passing the seed", and a static scan of run.py's dispatch covers every branch
including the ones that need jax, nutpie or a cluster.
"""

import ast
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from exozippy.mkparam import write_param_file
from exozippy.run import KNOWN_SAMPLER_KEYS

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "src" / "exozippy" / "run.py"

# Every sampler entry point run.py dispatches to, and the keyword each one
# spells the seed with.  pm.sample and sample_jax_nuts are pymc's, so their
# spelling is not ours to choose; the four in-house samplers all use `seed`.
_SAMPLER_CALLS = {
    "ptde_sample": "seed",
    "ptde_async_sample": "seed",
    "nested_sample": "seed",
    "de_metropolis_sample": "seed",
    "sample_jax_nuts": "random_seed",
}


def _call_sources(src, func_name):
    """Every `func_name(...)` call's source text in ``src``."""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name == func_name:
            out.append(ast.get_source_segment(src, node))
    return out


@pytest.mark.parametrize("func,kw", sorted(_SAMPLER_CALLS.items()))
def test_every_sampler_call_site_is_seeded(func, kw):
    """
    Given each sampler run.py can dispatch to,
    When its call site is read out of run.py's source,
    Then it passes the resolved seed.

    Regression: none of them did.  ptde/ptde_async/de_metropolis/nested all
    ALREADY accepted `seed=`; the argument had simply never been passed.
    """
    src = RUN_PY.read_text(encoding="utf-8")

    # ACT
    calls = _call_sources(src, func)

    # ASSERT
    assert calls, f"no {func}(...) call found in run.py"
    for call in calls:
        assert re.search(rf"\b{kw}\s*=\s*seed\b", call), (
            f"{func}(...) does not pass {kw}=seed:\n{call}"
        )


def test_every_pm_sample_call_site_is_seeded():
    """
    Given run.py's two pm.sample branches (plain NUTS and nutpie),
    When their call sites are read,
    Then both pass random_seed=seed.

    Handled apart from the parametrized cases because `pm.sample` is an
    attribute call and appears more than once.
    """
    src = RUN_PY.read_text(encoding="utf-8")

    # ACT
    calls = _call_sources(src, "sample")
    pm_calls = [c for c in calls if c.startswith("pm.sample(")]

    # ASSERT
    assert len(pm_calls) >= 2, f"expected 2+ pm.sample calls, got {pm_calls}"
    for call in pm_calls:
        assert re.search(r"\brandom_seed\s*=\s*seed\b", call), (
            f"pm.sample(...) does not pass random_seed=seed:\n{call}"
        )


def test_seed_is_a_recognized_sampler_key():
    """
    Given the sampler-block vocabulary,
    When `seed` is looked up,
    Then it is there -- otherwise warn_unknown_sampler_keys would tell a user
      who set it that the key is not recognized, which would be true.
    """
    assert "seed" in KNOWN_SAMPLER_KEYS


def test_an_absent_seed_is_drawn_and_not_a_constant():
    """
    Given no `sampler: seed:` key,
    When run.py resolves the seed,
    Then it DRAWS one rather than leaving it None or using a fixed default.

    A hardcoded default would be strictly worse than no seed at all: every
    user's chains would be correlated while the code looked responsible.  The
    expression is asserted by reading it out of run.py -- resolution happens
    inline in run_fit, which cannot be called without a model.
    """
    src = RUN_PY.read_text(encoding="utf-8")

    # ASSERT: the drawing expression exists, and there is no literal default.
    assert "np.random.SeedSequence().entropy" in src
    assert 'sampler_cfg.get("seed", None)' in src
    assert not re.search(r'sampler_cfg\.get\("seed",\s*\d', src)


def test_the_drawn_seed_expression_varies_and_stays_in_range():
    """
    Given the drawing expression run.py uses,
    When it is evaluated repeatedly,
    Then the values differ and every one is a non-negative int32-safe int.

    The bound matters: numpy's legacy seeding and several downstream
    libraries reject a seed outside [0, 2**32).
    """
    # ACT
    seeds = [
        int(np.random.SeedSequence().entropy % (2**31 - 1)) for _ in range(8)
    ]

    # ASSERT
    assert len(set(seeds)) > 1
    assert all(0 <= s < 2**31 - 1 for s in seeds)


def test_run_py_stamps_the_seed_on_the_trace():
    """
    Given a finished sampling run,
    When the trace is written,
    Then the seed the run actually used rides on posterior.attrs, beside
      nthin -- so a trace is self-describing and mkparam has something to
      read.
    """
    src = RUN_PY.read_text(encoding="utf-8")
    assert 'idata.posterior.attrs["random_seed"] = int(seed)' in src


def _trace_with_seed(tmp_path, seed):
    """Minimal one-draw trace carrying (or not) a random_seed attr."""
    import arviz as az

    post = xr.Dataset(
        {
            "star.mass": xr.DataArray([[0.95]], dims=["chain", "draw"]),
            "star.mass_raw": xr.DataArray([[0.1]], dims=["chain", "draw"]),
        }
    )
    if seed is not None:
        post.attrs["random_seed"] = int(seed)
    stats = xr.Dataset({"lp": xr.DataArray([[-10.0]], dims=["chain", "draw"])})
    idata = az.from_dict({"posterior": post, "sample_stats": stats})
    if seed is not None:
        idata.posterior.attrs["random_seed"] = int(seed)
    path = tmp_path / "trace.nc"
    idata.to_netcdf(str(path))
    return path


def _config():
    return {
        "prefix": "fitresults/model",
        "parameter_file": None,
        "star": [{"name": "Host"}],
    }


def test_the_restart_file_carries_the_seed_as_a_comment(tmp_path):
    """
    Given a trace stamped with the seed its fit ran under,
    When mkparam writes the restart file,
    Then the seed appears in the HEADER COMMENT, spelled the way it goes into
      the config, and is NOT a YAML key.

    A key would be wrong twice over: a params file holds parameter overrides,
    so `sampler:` there would be read as a two-part broadcast path naming a
    component that does not exist, and the user would be told so.
    """
    # ARRANGE
    trace = _trace_with_seed(tmp_path, 12345)

    # ACT
    out = write_param_file(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )
    text = Path(out).read_text()

    # ASSERT
    header = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert any("seed: 12345" in ln for ln in header), header

    import yaml

    parsed = yaml.safe_load(text)
    assert "sampler" not in parsed


def test_a_trace_without_the_attr_writes_no_seed_line(tmp_path):
    """
    Given an OLDER trace, written before the seed was stamped,
    When mkparam writes the restart file,
    Then no seed line is emitted -- an invented one would claim a
      reproducibility the trace cannot support.
    """
    # ARRANGE
    trace = _trace_with_seed(tmp_path, None)

    # ACT
    out = write_param_file(
        _config(),
        base_dir=tmp_path,
        trace_path=trace,
        output_path=tmp_path / "out.yaml",
    )
    text = Path(out).read_text()

    # ASSERT
    assert "seed:" not in text


def test_the_async_caveat_is_documented_where_a_user_meets_it():
    """
    Given ptde_async's arrival-order nondeterminism, which no seed can fix,
    When the docs are searched,
    Then samplers.md names it AND names the remedy (`method: ptde`) -- the
      point of review 2.14.4c is that it is a trade the user bought, not a
      bare disclaimer.
    """
    doc = (
        REPO_ROOT / "src" / "exozippy" / "samplers" / "samplers.md"
    ).read_text()
    assert "arrival order" in doc
    assert "method: ptde" in doc
    assert "seed" in doc


def test_get_draws_says_its_unseeded_draw_is_deliberate():
    """
    Given run.get_draws' unseeded np.random.choice,
    When its docstring is read,
    Then it says the spaghetti draws differ run to run BY DESIGN -- so no
      future reader chases it as a bug or helpfully pins a seed (2.14.4a).
    """
    from exozippy.run import get_draws

    doc = get_draws.__doc__
    assert "UNSEEDED" in doc
    assert "BY DESIGN" in doc

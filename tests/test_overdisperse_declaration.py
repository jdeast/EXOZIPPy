"""The params-file writer declares whether its seeds want dispersing (8.3.3).

Multiple seeds mean one of exactly two things and the sampler cannot tell them
apart at run time: either the user is seeding at several MODES, each of which
still wants scattering, or they are iterating from a params file whose seeds
are ALREADY properly dispersed (joint posterior draws).  So the WRITER declares
it, through a reserved non-parameter key `overdisperse:` in the params file.

One test per clause of the ruling:

  1. mmexofast_to_params writes `overdisperse: true`  (single optima).
  2. mkparam writes `overdisperse: false` for a multi-seed file, and records
     the min ESS / max Rhat it observed.
  3. An ABSENT key means True.
  4. `overdisperse: false` with only ONE seed RAISES.
  5. Few unique seeds WARN, in two tiers, with the affine-hull consequence in
     the escalated text.
  6. `overdisperse: false` actually suppresses the jitter.

Plus the trap the key had to be designed around: a bare top-level params key
is otherwise registered as a LEAF SYMBOL by `finalize_user_params`, with its
own ledger row and its own inject-back initval (see CLAUDE.md).
"""

import logging

import numpy as np
import pytest
import yaml

from exozippy.config import ConfigManager, split_reserved_param_keys
from exozippy.samplers._common import (
    de_span_floor,
    warn_if_seed_population_degenerate,
)
from exozippy.samplers.ptde import _make_starts

az = pytest.importorskip("arviz")

from exozippy.mkparam import write_param_file  # noqa: E402


def _quad_logp(point):
    """A finite, well-behaved logp so _make_starts never retries."""
    return -0.5 * sum(
        float(np.sum(np.asarray(v) ** 2)) for v in point.values()
    )


def _seeds(k, n_elements=1):
    return [
        {"a": np.full(n_elements, float(i), dtype=float)} for i in range(k)
    ]


# ---------------------------------------------------------------------------
# The trap: a reserved key must never become an engine symbol.
# ---------------------------------------------------------------------------


def test_overdisperse_never_becomes_an_engine_symbol():
    """
    Given a params mapping carrying the reserved key `overdisperse` alongside
      a real parameter,
    When a ConfigManager is built and the relaxation engine is finalized,
    Then `overdisperse` is absent from the symbol map, from the resolved
      solution and from export_solution, and nothing is injected back into
      user_params under that name -- while the real parameter is untouched.

    Regression: `finalize_user_params` registers every UNMAPPED user_params
    key as a leaf symbol (CLAUDE.md).  Measured before the fix, a bare
    `overdisperse: false` became the symbol `overdisperse`, entered
    `_last_resolved`, and the engine wrote `{'initval': 0.0, 'derived': True}`
    back over the user's own boolean.
    """
    # Given
    cm = ConfigManager(
        {"overdisperse": False, "star.A.teff": {"initval": 5800.0}},
        system_config={"star": [{"name": "A"}]},
    )
    # When
    cm.finalize_user_params()
    # Then
    assert "overdisperse" not in cm.user_params
    assert "overdisperse" not in cm.master_symbol_map
    assert not [k for k in cm._last_resolved if "overdisperse" in str(k)]
    assert not [k for k in cm.export_solution() if "overdisperse" in str(k)]
    # The real parameter still went through untouched.
    assert "star.0.teff" in cm.user_params


def test_unknown_top_level_scalar_warns_by_name(caplog):
    """
    Given a params mapping with a dot-less scalar key that is not reserved,
    When it is split,
    Then the key is dropped with a warning NAMING it, rather than silently
      becoming a leaf symbol for a parameter that does not exist.
    """
    with caplog.at_level(logging.WARNING):
        params, reserved = split_reserved_param_keys({"overdispers": False})
    assert params == {} and reserved == {}
    assert "overdispers" in caplog.text


def test_non_boolean_overdisperse_raises():
    """
    Given `overdisperse` written as something other than a boolean,
    When the params mapping is split,
    Then it RAISES naming the key -- a truthy string would otherwise read as
      True and silently mean the opposite of `overdisperse: "no"`.
    """
    with pytest.raises(ValueError, match="overdisperse"):
        split_reserved_param_keys({"overdisperse": "no"})


def test_reserved_keys_are_not_reported_as_unmatched():
    """
    Given the model auditor's unmatched-key check, which reads
      `system.user_params` -- the params FILE exactly as written, reserved keys
      and all, not the ConfigManager's stripped copy,
    When it decides what matched no parameter,
    Then reserved keys are exempt, the way "run" already is.

    Regression, caught by `test_mkparam_roundtrip.py`: without the exemption
    every file mkparam writes reported `overdisperse` as a key the model could
    not read back -- on every restart, forever, which is precisely the silent-
    typo warning this check exists to make credible.
    """
    from exozippy.diagnostics import ModelAuditor

    class _Auditor:
        all_params = ()
        user_params = {
            "overdisperse": False,
            "star.typo": {"initval": 1.0},
        }

        def _engine_consumed(self, key):
            return False

    unused = ModelAuditor.check_unused_yaml(_Auditor())
    assert "overdisperse" not in unused
    # The check still has teeth: a real unmatched key is still reported.
    assert "star.typo" in unused


# ---------------------------------------------------------------------------
# Clause 3: an ABSENT key means True.
# ---------------------------------------------------------------------------


def test_absent_key_means_true():
    """
    Given a params file that says nothing about dispersion,
    When ConfigManager is asked,
    Then it answers True -- the safe direction for a hand-written file, since
      over-dispersing a good seed set costs burn-in while under-dispersing a
      bad one makes Rhat read ~1.00 on chains that never mixed.
    """
    cm = ConfigManager({"star.A.teff": {"initval": 5800.0}})
    assert cm.overdisperse is True
    assert ConfigManager({"overdisperse": False}).overdisperse is False
    assert ConfigManager({"overdisperse": True}).overdisperse is True


# ---------------------------------------------------------------------------
# Clause 4: overdisperse false + ONE seed RAISES.
# ---------------------------------------------------------------------------


def test_single_seed_with_overdisperse_false_raises():
    """
    Given one seed and `overdisperse: false`,
    When _make_starts builds the chain starts,
    Then it RAISES: every chain would start at the identical point, so every
      DE difference vector would be exactly zero and the population could
      never move apart.
    """
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="overdisperse"):
        _make_starts(
            6,
            _seeds(1),
            _quad_logp,
            rng,
            raw_scales={"a": np.ones(1)},
            overdisperse=False,
        )


def test_single_seed_with_overdisperse_true_is_fine():
    """
    Given the SAME single seed but no declaration (so True),
    When _make_starts runs,
    Then it builds the population as it always has -- the raise above is about
      the declaration, not about seed counts as such.
    """
    rng = np.random.default_rng(0)
    starts, _ = _make_starts(
        6, _seeds(1), _quad_logp, rng, raw_scales={"a": np.ones(1)}
    )
    assert len(starts) == 6


# ---------------------------------------------------------------------------
# Clause 6: overdisperse false suppresses the jitter.
# ---------------------------------------------------------------------------


def test_overdisperse_false_starts_every_chain_exactly_at_its_seed():
    """
    Given 6 seeds, 12 chains and `overdisperse: false`,
    When _make_starts runs,
    Then EVERY chain start is bit-identical to its round-robin seed -- not
      just the first of each group, which is all the exact-start budget
      (max_exact = n_chains // 2) allows when dispersion is on.
    """
    rng = np.random.default_rng(3)
    seeds = _seeds(6)
    starts, chain_seed_index = _make_starts(
        12,
        seeds,
        _quad_logp,
        rng,
        raw_scales={"a": np.ones(1)},
        overdisperse=False,
    )
    assert chain_seed_index == [j % 6 for j in range(12)]
    for j, st in enumerate(starts):
        assert np.array_equal(st["a"], seeds[j % 6]["a"])


def test_overdisperse_true_still_caps_exact_starts():
    """
    Given the same 6 seeds and 12 chains with dispersion ON (the default),
    When _make_starts runs,
    Then at most half the chains start exactly at a seed -- the pre-existing
      contract is unchanged by the new flag.
    """
    rng = np.random.default_rng(3)
    seeds = _seeds(6)
    starts, _ = _make_starts(
        12, seeds, _quad_logp, rng, raw_scales={"a": np.ones(1)}
    )
    n_exact = sum(
        any(np.array_equal(st["a"], sd["a"]) for sd in seeds) for st in starts
    )
    assert n_exact <= 12 // 2


def test_make_starts_reads_the_declaration_off_the_system():
    """
    Given a system object carrying `overdisperse = False` (as System forwards
      it from its ConfigManager) and no explicit argument,
    When _make_starts runs,
    Then it honours the declaration -- the sampler call sites pass only the
      system, so this is the path production actually takes.
    """

    class _Sys:
        overdisperse = False

    rng = np.random.default_rng(1)
    seeds = _seeds(4)
    starts, _ = _make_starts(
        4,
        seeds,
        _quad_logp,
        rng,
        raw_scales={"a": np.ones(1)},
        system=_Sys(),
    )
    for j, st in enumerate(starts):
        assert np.array_equal(st["a"], seeds[j]["a"])


# ---------------------------------------------------------------------------
# Clause 5: the two-tier warning, at sampler start.
# ---------------------------------------------------------------------------


def test_seed_degeneracy_warning_escalates_below_the_span_floor(caplog):
    """
    Given fewer unique seeds than n_params + 2 -- the number of population
      members it takes to span parameter space, since a DE proposal for
      member i draws its difference vector from the OTHER members,
    When the seed-population check runs,
    Then the warning says the population cannot span parameter space AT ALL
      and states the consequence: the only escape is the epsilon jitter, whose
      off-hull diffusion goes as jitter*sqrt(steps), so covering one whitened
      sigma takes ~1e8 accepted steps.
    """
    log = logging.getLogger("t.escalated")
    with caplog.at_level(logging.WARNING, logger="t.escalated"):
        warn_if_seed_population_degenerate(4, 10, "PTDE init", log)
    text = caplog.text
    assert "cannot span" in text or "span parameter space" in text
    assert "n_params + 2 = 12" in text
    assert "1e8" in text
    assert "sqrt(steps)" in text


def test_seed_degeneracy_warning_is_a_mixing_complaint_above_the_floor(caplog):
    """
    Given a unique-seed count at or above n_params + 2 but below the default
      chain count 2 x n_params (ter Braak's mixing recommendation),
    When the seed-population check runs,
    Then it warns about MIXING and does NOT claim the population fails to
      span -- the two tiers say different things and must not be conflated.
    """
    log = logging.getLogger("t.mixing")
    with caplog.at_level(logging.WARNING, logger="t.mixing"):
        warn_if_seed_population_degenerate(12, 10, "PTDE init", log)
    text = caplog.text
    assert "mix slowly" in text
    assert "1e8" not in text


def test_seed_degeneracy_silent_at_the_default_chain_count(caplog):
    """
    Given as many unique seeds as the default chain count (2 x n_params),
    When the seed-population check runs,
    Then nothing is warned: this is the configuration the defaults produce.
    """
    log = logging.getLogger("t.quiet")
    with caplog.at_level(logging.WARNING, logger="t.quiet"):
        warn_if_seed_population_degenerate(20, 10, "PTDE init", log)
    assert caplog.text == ""


def test_span_floor_is_one_number_shared_with_the_chain_check():
    """
    Given the two places the span floor is used -- the CHAIN count and the
      UNIQUE-SEED count,
    When either asks for it,
    Then it comes from the one helper, so a second constant meaning the same
      thing cannot drift away from it.
    """
    assert de_span_floor(10) == 12
    assert de_span_floor(1) == 3


def test_two_tier_warning_fires_from_make_starts(caplog):
    """
    Given `overdisperse: false` with too few unique seeds for the model's
      parameter count,
    When _make_starts builds the population,
    Then the warning is emitted THERE -- at sampler start, where n_params
      belongs to the model about to be fitted.  mkparam cannot evaluate this
      threshold: its seeds may be fed to a different model (added data, a
      changed parameterization).
    """
    rng = np.random.default_rng(0)
    with caplog.at_level(logging.WARNING):
        _make_starts(
            8,
            _seeds(3, n_elements=8),
            _quad_logp,
            rng,
            raw_scales={"a": np.ones(8)},
            overdisperse=False,
        )
    assert "overdisperse" in caplog.text
    assert "1e8" in caplog.text


# ---------------------------------------------------------------------------
# Clause 1: the MMEXOFAST converter declares True.
# ---------------------------------------------------------------------------


def test_mmexofast_converter_declares_overdisperse_true(tmp_path):
    """
    Given an MMEXOFAST JSON with two solutions,
    When it is converted to a params file,
    Then the file declares `overdisperse: true`: those seeds are single
      optima, one per solution, not draws from any posterior, so the chains
      still have to be scattered around them.
    """
    import json

    from exozippy.utilities.mmexofast_to_params import mmexofast_to_params

    src = tmp_path / "mm.json"
    fit = {
        "parameters": {
            "t_0": 2455000.1,
            "u_0": 0.2,
            "t_E": 30.0,
            "s": 1.1,
            "alpha": 45.0,
            "rho": 0.0,
            "q": 1e-3,
        }
    }
    src.write_text(json.dumps({"fits": [fit, fit]}))
    out = tmp_path / "mm.params.yaml"
    mmexofast_to_params(str(src), out_path=str(out))

    text = out.read_text()
    assert "overdisperse: true" in text
    assert yaml.safe_load(text)["overdisperse"] is True


# ---------------------------------------------------------------------------
# Clause 2: mkparam declares False and records the mixing it observed.
# ---------------------------------------------------------------------------


def _trace(tmp_path, nchain=4, ndraw=400, seed=0):
    rng = np.random.default_rng(seed)
    post = {
        "source.t_0": 2000.0 + rng.standard_normal((nchain, ndraw)),
        "source.t_0_raw": rng.standard_normal((nchain, ndraw)),
    }
    lp = -0.5 * (post["source.t_0_raw"] ** 2)
    idata = az.from_dict({"posterior": post, "sample_stats": {"lp": lp}})
    path = tmp_path / "run_trace.nc"
    idata.to_netcdf(str(path))
    return path


def _cfg():
    return {"prefix": "run", "source": [{"name": "SourceA"}]}


def test_mkparam_multiseed_declares_overdisperse_false(tmp_path):
    """
    Given a finished fit's trace and n_seeds > 1,
    When mkparam writes the restart file,
    Then it declares `overdisperse: false` -- a true statement about where
      those seeds came from: they are joint draws from a sampled posterior and
      so are already spread across its covariance.
    """
    out = tmp_path / "out.params.yaml"
    write_param_file(
        _cfg(),
        base_dir=tmp_path,
        trace_path=_trace(tmp_path),
        output_path=out,
        n_seeds=4,
    )
    text = out.read_text()
    assert "overdisperse: false" in text
    assert yaml.safe_load(text)["overdisperse"] is False


def test_mkparam_single_seed_declares_overdisperse_true(tmp_path):
    """
    Given the default n_seeds=1, so the file carries the MAP draw and nothing
      else,
    When mkparam writes the restart file,
    Then it declares `overdisperse: true`.  A lone MAP is not a dispersed
      population, and the clause-4 raise means a `false` here would make every
      default restart file unusable on arrival.
    """
    out = tmp_path / "out.params.yaml"
    write_param_file(
        _cfg(),
        base_dir=tmp_path,
        trace_path=_trace(tmp_path),
        output_path=out,
        n_seeds=1,
    )
    assert yaml.safe_load(out.read_text())["overdisperse"] is True


def test_mkparam_records_the_ess_and_rhat_it_observed(tmp_path):
    """
    Given a trace whose mixing mkparam measures on its way to picking seeds,
    When it writes the restart file,
    Then the min bulk-ESS and max split-Rhat are RECORDED in the header
      comment -- for the human, not as a gate.  Nobody reruns a well-mixed
      fit, so a "did it converge?" boolean would read False almost always and
      `overdisperse: false` would be dead code; and the two numbers gate
      different failures (Rhat: are these seeds from the right distribution at
      all; ESS: how many EFFECTIVELY INDEPENDENT seeds there are).
    """
    out = tmp_path / "out.params.yaml"
    write_param_file(
        _cfg(),
        base_dir=tmp_path,
        trace_path=_trace(tmp_path),
        output_path=out,
        n_seeds=4,
    )
    text = out.read_text()
    assert "min bulk-ESS" in text
    assert "max split-Rhat" in text
    # A comment, not a key: nothing machine-reads these numbers.
    parsed = yaml.safe_load(text)
    assert not [k for k in parsed if "ess" in str(k).lower()]
    assert not [k for k in parsed if "rhat" in str(k).lower()]


def test_mkparam_output_survives_a_configmanager_round_trip(tmp_path):
    """
    Given the file mkparam just wrote, read back as user params,
    When a ConfigManager consumes it,
    Then the declaration is available on the manager and `overdisperse` is
      NOT one of the parameters -- the round trip is the whole point of a
      reserved key, and this is the path a restart actually takes.
    """
    out = tmp_path / "out.params.yaml"
    write_param_file(
        _cfg(),
        base_dir=tmp_path,
        trace_path=_trace(tmp_path),
        output_path=out,
        n_seeds=4,
    )
    cm = ConfigManager(yaml.safe_load(out.read_text()), system_config=_cfg())
    assert cm.overdisperse is False
    assert "overdisperse" not in cm.user_params
